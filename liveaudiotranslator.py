import queue
import threading
import sys
import time
from collections import deque
import numpy as np
import torch
from faster_whisper import WhisperModel
import pyaudio # Added dependency

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font # Added font for default setting

# --- Configuration ---

# Whisper settings
WHISPER_MODEL_SIZE = "medium" # tiny, base, small, medium, large-v2, large-v3
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "float32" # Explicitly use float32 on CPU

# Audio settings
AUDIO_SAMPLERATE = 16000 # Whisper requires 16kHz
AUDIO_CHANNELS = 1 # Mono
AUDIO_FORMAT = pyaudio.paInt16 # PyAudio format constant
AUDIO_BUFFER_SECONDS = 3 # Process audio in chunks of this duration (Adjust if needed)
PYAUDIO_FRAMES_PER_BUFFER = 1024 # How many frames PyAudio reads at a time

#Phrases the ai translates too often in error
PHRASES_TO_IGNORE = [
    "thanks for watching",
    "thank you for watching",
    "thank you for watching!",
    "thank you for your viewing",
    "thank you very much for watching until the end",
    # Add other similar variations if you notice them
]

# --- Appearance & History Settings ---
HISTORY_MINUTES = 3       # How many minutes of translation history to keep
WINDOW_ALPHA = 0.85       # Window transparency (0.0=invisible to 1.0=opaque)

# --- Color Constants for Dark Theme ---
DARK_BG = '#2b2b2b'        # Dark gray background
LIGHT_FG = '#ffffff'       # White text
ENTRY_BG = '#3c3f41'       # Slightly lighter gray for entry background
BUTTON_BG = '#555555'       # Medium gray for button
BUTTON_ACTIVE_BG = '#666666' # Slightly lighter gray for active/hover button
TEXT_CURSOR = '#ffffff'    # White cursor color
INFO_FG = '#6cace4'        # Light blue for info messages
ERROR_FG = '#e46c6c'       # Light red for error messages

# --- Global Queues & Events ---
audio_queue = queue.Queue()
gui_queue = queue.Queue()      # Single queue for all GUI updates
stop_event = threading.Event() # Event to signal threads to stop

# --- Helper Function ---
def get_audio_devices():
    """Gets a list of available audio input devices."""
    devices = []
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            devices.append({
                'index': i,
                'name': device_info.get('name'),
                'channels': device_info.get('maxInputChannels'),
                'rate': device_info.get('defaultSampleRate')
            })
    p.terminate()
    return devices

# --- Thread Functions (Adapted for PyAudio, Whisper Translate, GUI Queue, Stop Event) ---

def capture_audio_thread_gui(device_index, audio_q, gui_q, stop_event_flag):
    """ Captures audio using PyAudio, puts chunks in audio_q, reports status/errors to gui_q, checks stop_event. """
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Starting audio capture from device index {device_index}...")
    audio_buffer = bytearray()
    target_buffer_size = AUDIO_SAMPLERATE * AUDIO_CHANNELS * 2 * AUDIO_BUFFER_SECONDS # 2 bytes/sample
    p = None
    stream = None

    try:
        p = pyaudio.PyAudio()
        # Basic device check
        device_info = p.get_device_info_by_index(device_index)
        gui_q.put(f"[{thread_name}] Selected device: {device_info.get('name')}")
        if device_info.get('maxInputChannels') < AUDIO_CHANNELS:
             raise ValueError(f"Device does not support required channels ({AUDIO_CHANNELS})")
        # Add rate check maybe? p.is_format_supported(...)

        stream = p.open(format=AUDIO_FORMAT,
                        channels=AUDIO_CHANNELS,
                        rate=AUDIO_SAMPLERATE,
                        input=True,
                        frames_per_buffer=PYAUDIO_FRAMES_PER_BUFFER,
                        input_device_index=device_index)
        gui_q.put(f"[{thread_name}] Audio stream opened. Capturing...")

        while not stop_event_flag.is_set():
            try:
                data = stream.read(PYAUDIO_FRAMES_PER_BUFFER, exception_on_overflow=False)
                audio_buffer.extend(data)
                while len(audio_buffer) >= target_buffer_size:
                    if stop_event_flag.is_set(): break
                    chunk_to_process = audio_buffer[:target_buffer_size]
                    audio_q.put(bytes(chunk_to_process))
                    audio_buffer = audio_buffer[target_buffer_size:]

            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed: # -9981
                    gui_q.put(f"[{thread_name}] Warning: PyAudio Input Overflowed.") # Report non-critically
                else:
                    gui_q.put(f"[{thread_name}] ERROR: PyAudio read error: {e}")
                    time.sleep(0.1) # Avoid busy-loop on persistent error

        if stop_event_flag.is_set():
             gui_q.put(f"[{thread_name}] Stop event received.")

    except ValueError as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Device config error: {e}")
    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Initializing/running PyAudio: {e}")
    finally:
        audio_q.put(None) # Signal downstream thread
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if p is not None:
            p.terminate()
        gui_q.put(f"[{thread_name}] Audio capture thread finished.")
        # gui_q.put(None) # Don't signal GUI end here, let transcribe thread do it


def transcribe_translate_thread_gui(audio_q, gui_q, stop_event_flag):
    """ Transcribes audio using Whisper model and translates directly to English, filtering unwanted phrases. """
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Initializing Whisper model ({WHISPER_MODEL_SIZE})...")
    model = None
    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        gui_q.put(f"[{thread_name}] Whisper model initialized on {WHISPER_DEVICE}. Ready for translation.")

        while not stop_event_flag.is_set():
            try:
                audio_data_bytes = audio_q.get(timeout=0.5) # Use timeout to check stop_event
                if audio_data_bytes is None:
                    break # Upstream closed, finish up
            except queue.Empty:
                continue # No data, check stop_event and loop

            if stop_event_flag.is_set(): break

            try:
                audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                if audio_np.size > 0:
                    start_time = time.time()
                    segments, info = model.transcribe(
                        audio_np, beam_size=5, vad_filter=False,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        task='translate' # Directly translate to English
                    )
                    # Detected source language info is available in info.language
                    detected_lang = info.language
                    detected_prob = info.language_probability

                    full_translation = " ".join(segment.text for segment in segments).strip()
                    latency = time.time() - start_time

                    # --- >>> MODIFICATION START <<< ---
                    # Normalize the translation for comparison
                    normalized_translation = full_translation.lower().strip().rstrip('.?!')

                    # Check if the normalized translation is in the ignore list
                    should_ignore = normalized_translation in PHRASES_TO_IGNORE

                    if full_translation and not should_ignore: # Only queue if not empty AND not ignored
                         # Send translation for history deque
                        gui_q.put(f"[EN ({latency:.2f}s, src={detected_lang}:{detected_prob:.2f})] {full_translation}")
                    elif should_ignore:
                        # Optional: Log that a phrase was ignored (can be noisy)
                        # print(f"Ignoring phrase: '{full_translation}'")
                        pass # Just don't queue it
                    # else: VAD might have filtered, or no speech detected
                    # --- >>> MODIFICATION END <<< ---

            except Exception as e:
                gui_q.put(f"[{thread_name}] ERROR during translation: {e}")
            finally:
                audio_q.task_done()

        if stop_event_flag.is_set():
             gui_q.put(f"[{thread_name}] Stop event received.")

    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Failed to load/run Whisper model: {e}")
    finally:
        if model is not None:
            # Try to release GPU memory
            try:
                del model
                if WHISPER_DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                gui_q.put(f"[{thread_name}] Whisper model resources released.")
            except Exception as e:
                gui_q.put(f"[{thread_name}] Warning: Error releasing model resources: {e}")
        gui_q.put(f"[{thread_name}] Transcription/Translation thread finished.")
        gui_q.put(None) # Signal GUI that this processing pipeline is done


# --- Tkinter GUI Application (Adapted from Old Code) ---

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Audio Translator (Whisper Direct)")
        self.root.geometry("700x350") # Keep original size

        # --- Dark Theme Configuration ---
        self.root.config(bg=DARK_BG)
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam') # Keep clam theme attempt
        except tk.TclError:
            print("Warning: 'clam' theme not available, using default.")

        style.configure('.', background=DARK_BG, foreground=LIGHT_FG, font=('Segoe UI', 9)) # Set default font slightly smaller
        style.configure('TFrame', background=DARK_BG)
        style.configure('TLabel', background=DARK_BG, foreground=LIGHT_FG)
        style.configure('TButton', background=BUTTON_BG, foreground=LIGHT_FG, padding=5)
        style.map('TButton', background=[('active', BUTTON_ACTIVE_BG), ('disabled', DARK_BG)])
        # Entry/Combobox styling from old code
        style.configure('TCombobox', fieldbackground=ENTRY_BG, background=BUTTON_BG, foreground=LIGHT_FG,
                        arrowcolor=LIGHT_FG, selectbackground=ENTRY_BG, selectforeground=LIGHT_FG,
                        insertcolor=TEXT_CURSOR) # Use ENTRY_BG for select background for consistency
        style.map('TCombobox', fieldbackground=[('readonly', ENTRY_BG)],
                           selectbackground=[('!focus', ENTRY_BG)], # Keep selection color when not focused
                           background=[('readonly', BUTTON_BG)])
        # Combobox list styling
        root.option_add('*TCombobox*Listbox.background', ENTRY_BG)
        root.option_add('*TCombobox*Listbox.foreground', LIGHT_FG)
        root.option_add('*TCombobox*Listbox.selectBackground', BUTTON_ACTIVE_BG)
        root.option_add('*TCombobox*Listbox.selectForeground', LIGHT_FG)

        # --- Window Attributes ---
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', WINDOW_ALPHA)

        self.is_running = False
        self.active_threads = []
        self.message_deque = deque() # Keep history deque

        # --- Input Frame ---
        input_frame = ttk.Frame(root, padding="10")
        input_frame.pack(fill=tk.X)

        ttk.Label(input_frame, text="Input Device:").pack(side=tk.LEFT, padx=(0, 5))

        # --- Device Selection ---
        self.devices = get_audio_devices()
        self.device_map = {f"{d['index']}: {d['name']} (Rate: {int(d['rate'])})": d['index'] for d in self.devices}
        self.device_combobox = ttk.Combobox(input_frame,
                                             values=list(self.device_map.keys()),
                                             state="readonly", width=55) # Adjust width
        if self.device_map:
            self.device_combobox.current(0)
        self.device_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)


        self.start_button = ttk.Button(input_frame, text="Start", command=self.start_translation, width=7)
        self.start_button.pack(side=tk.LEFT, padx=(5, 2))

        self.stop_button = ttk.Button(input_frame, text="Stop", command=self.stop_translation, state=tk.DISABLED, width=7)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 0))


        # --- Output Area ---
        output_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state=tk.DISABLED, height=10)
        # Use specific config for scrolledtext background/foreground
        self.output_text.config(bg=DARK_BG, fg=LIGHT_FG, insertbackground=TEXT_CURSOR, font=('Segoe UI', 10)) # Slightly larger font for output
        self.output_text.pack(fill=tk.BOTH, expand=True)
        # Define tags for coloring output text
        self.output_text.tag_config("error", foreground=ERROR_FG)
        self.output_text.tag_config("info", foreground=INFO_FG)
        self.output_text.tag_config("translation", foreground=LIGHT_FG) # Default translation color

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Select Input Device and press Start.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.FLAT, anchor=tk.W, padding=5) # Flat relief might look better
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def add_output_direct(self, text, tag=None):
        """ Adds text (like status/errors) directly without history limit, applying optional tag """
        if not self.root.winfo_exists(): return # Avoid errors if window closed
        try:
            self.output_text.config(state=tk.NORMAL)
            if tag:
                 self.output_text.insert(tk.END, text + "\n", tag)
            else:
                 self.output_text.insert(tk.END, text + "\n")
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        except tk.TclError as e:
            print(f"Error adding text to GUI (window closed?): {e}")

    def update_output_display(self):
        """ Clears and rewrites the text area from the message deque """
        if not self.root.winfo_exists(): return
        try:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete('1.0', tk.END)
            for _, msg in self.message_deque:
                 # Check message type for coloring (simple prefix check)
                if msg.startswith("[EN"):
                    self.output_text.insert(tk.END, msg + "\n", "translation")
                elif msg.startswith("[ERROR"):
                     self.output_text.insert(tk.END, msg + "\n", "error")
                elif msg.startswith("[INFO") or msg.startswith("["): # Catch other status/thread messages
                     self.output_text.insert(tk.END, msg + "\n", "info")
                else:
                     self.output_text.insert(tk.END, msg + "\n") # Default if needed
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        except tk.TclError as e:
            print(f"Error updating GUI display (window closed?): {e}")


    def check_gui_queue(self):
        """ Checks the GUI queue, manages history deque, and updates the GUI """
        if not self.is_running: return # Stop checking if not running

        try:
            while True:
                message = gui_queue.get_nowait()
                current_time = time.time()

                if message is None:
                     # This signals the end from the transcribe thread
                     self.status_var.set("Processing stopped or finished.")
                     self.add_output_direct("--- Processing thread finished ---", "info")
                     self.stop_translation(graceful_stop=True) # Trigger GUI state reset
                     return # Stop checking queue for now

                elif isinstance(message, str):
                    # Check if it's a translation message for history deque
                    if message.startswith("[EN"):
                        self.message_deque.append((current_time, message))
                        # Prune deque based on time
                        cutoff_time = current_time - (HISTORY_MINUTES * 60)
                        while self.message_deque and self.message_deque[0][0] < cutoff_time:
                            self.message_deque.popleft()
                        self.update_output_display() # Update view with new history
                        # Optionally update status bar minimally for translations
                        # self.status_var.set("Processing...")
                    # Check for critical errors to stop everything
                    elif message.startswith("CRITICAL ERROR"):
                         self.status_var.set(message.split("] ", 1)[-1])
                         self.add_output_direct(f"*** {message} ***", "error")
                         self.stop_translation(error=True) # Stop due to critical error
                         return # Stop queue processing
                    # Handle other errors/warnings/info
                    elif message.startswith("[ERROR") or message.startswith("[Warning"):
                        self.status_var.set("Error/Warning occurred.")
                        self.add_output_direct(message, "error")
                    elif message.startswith("["): # Assume other bracketed messages are status/info
                        status_part = message.split("] ", 1)[-1]
                        # Only update status bar for significant status, maybe?
                        if "initialized" in status_part or "Capturing" in status_part or "finished" in status_part:
                             self.status_var.set(status_part)
                        # Log all info messages to the text area for debugging/visibility
                        self.add_output_direct(message, "info")
                    else:
                         # Unformatted messages? Log them.
                         self.add_output_direct(message, "info")

                gui_queue.task_done()
        except queue.Empty:
            pass # No messages currently
        except Exception as e:
            print(f"Error processing GUI queue: {e}")
            self.add_output_direct(f"[ERROR] GUI Error: {e}", "error")

        # Reschedule check if still running
        if self.is_running and self.root.winfo_exists():
             self.root.after(100, self.check_gui_queue)


    def start_translation(self):
        """ Starts the translation process in separate threads """
        if self.is_running:
            messagebox.showwarning("Already Running", "Translation is already in progress.")
            return

        selected_device_str = self.device_combobox.get()
        if not selected_device_str:
            messagebox.showerror("Input Error", "Please select an Input Device.")
            return
        device_index = self.device_map.get(selected_device_str)
        if device_index is None:
            messagebox.showerror("Internal Error", "Invalid device selection.")
            return

        # Clear previous output and deque
        try:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete('1.0', tk.END)
            self.output_text.config(state=tk.DISABLED)
            self.message_deque.clear()

            self.status_var.set(f"Starting... Device: {device_index}. Loading model...")
            self.add_output_direct("--- Starting translation ---", "info") # Log start
            self.is_running = True
            stop_event.clear() # Ensure stop event is not set initially
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.device_combobox.config(state=tk.DISABLED)
            self.active_threads = []
        except tk.TclError:
            print("Error starting: GUI already closed?")
            return

        # Clear queues before starting threads
        while not audio_queue.empty(): audio_queue.get_nowait()
        while not gui_queue.empty(): gui_queue.get_nowait()

        # --- Create and Start Threads ---
        capture = threading.Thread(
            target=capture_audio_thread_gui,
            args=(device_index, audio_queue, gui_queue, stop_event),
            name="AudioCapture", daemon=True)

        transcriber = threading.Thread(
            target=transcribe_translate_thread_gui,
            args=(audio_queue, gui_queue, stop_event),
            name="TranscribeTranslate", daemon=True)

        self.active_threads.extend([capture, transcriber])
        capture.start()
        transcriber.start()

        self.root.after(100, self.check_gui_queue) # Start polling GUI queue

    def stop_translation(self, error=False, graceful_stop=False):
        """ Stops the translation process """
        if not self.is_running and not error and not graceful_stop: return # Don't do anything if already stopped normally

        status = "Processing stopped by user."
        if error: status = "Processing stopped due to CRITICAL error."
        if graceful_stop: status = "Processing finished." # If called because None received on queue

        if self.is_running: # Only set event if we think we are running
            stop_event.set() # Signal threads to stop

        self.is_running = False # Stop queue checking loop

        # Update GUI state
        try:
            if self.root.winfo_exists():
                self.status_var.set(status)
                self.add_output_direct(f"--- {status} ---", "info" if not error else "error")
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.device_combobox.config(state="readonly")
        except tk.TclError:
            print("Info: GUI window closed during stop sequence.")

        # Clear queues (best effort, threads might still put None after this)
        while not audio_queue.empty():
            try: audio_queue.get_nowait()
            except queue.Empty: break
        # Don't clear gui_queue here, allow final messages to potentially be processed briefly

        # Clear active threads list (they are daemons, should exit eventually)
        self.active_threads = []
        print("Stop signal sent.")


    def on_closing(self):
        """ Handle window closing """
        if self.is_running:
             if messagebox.askokcancel("Quit", "Translation is running. Stop and quit?"):
                 self.stop_translation()
                 # Give threads a moment to react to stop_event before destroying window
                 self.root.after(200, self.root.destroy)
             else:
                 return # Don't close yet
        else:
             self.root.destroy()


# --- Main Execution ---
if __name__ == "__main__":
    # Check PyAudio early? Optional.
    try:
        p = pyaudio.PyAudio()
        p.terminate()
    except Exception as e:
        print(f"CRITICAL ERROR: PyAudio initialization failed: {e}", file=sys.stderr)
        print("Please ensure PyAudio is installed correctly for your system.", file=sys.stderr)
        # Show GUI error and exit cleanly
        root = tk.Tk()
        root.withdraw() # Hide root window
        messagebox.showerror("PyAudio Error", f"Could not initialize PyAudio. Please ensure it is installed correctly.\n\nError: {e}")
        sys.exit(1)

    # --- Create and Run GUI ---
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
