import queue
import threading
import sys
import time
from collections import deque
import numpy as np
import torch
from faster_whisper import WhisperModel
import ollama # Using Ollama again
import pyaudio # Using PyAudio for capture

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font

# --- Configuration ---

# Define supported target languages (Display Name: ISO 639-1 Code)
SUPPORTED_TARGET_LANGUAGES = {
    "English": "en",
    "Indonesian": "id",
    "Russian": "ru",
    "Korean": "ko",
    # "Japanese": "ja", # Maybe less useful if source is often JP
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh",
}
DEFAULT_TARGET_LANG = "English" # Default selection in the dropdown

# Whisper settings
WHISPER_MODEL_SIZE = "medium" # tiny, base, small, medium, large-v2, large-v3
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "default" # or: int8_float16, int8, float32

# Ollama settings
OLLAMA_MODEL = 'mistral' # Ensure this model is pulled in Ollama

# Audio settings
AUDIO_SAMPLERATE = 16000 # Whisper requires 16kHz
AUDIO_CHANNELS = 1 # Mono
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_BUFFER_SECONDS = 3 # Process audio in chunks (Adjust if needed)
PYAUDIO_FRAMES_PER_BUFFER = 1024 # PyAudio read size

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
transcribed_queue = queue.Queue() # Holds (lang_code, text) tuples again
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

# --- Thread Functions (Capture=PyAudio, Transcribe=Whisper, Translate=Ollama, check stop_event) ---

def capture_audio_thread_gui(device_index, audio_q, gui_q, stop_event_flag):
    """ Captures audio using PyAudio, puts chunks in audio_q, reports status/errors to gui_q, checks stop_event. """
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Starting audio capture from device index {device_index}...")
    audio_buffer = bytearray()
    target_buffer_size = AUDIO_SAMPLERATE * AUDIO_CHANNELS * 2 * AUDIO_BUFFER_SECONDS
    p = None
    stream = None
    try:
        p = pyaudio.PyAudio()
        device_info = p.get_device_info_by_index(device_index) # Check device early
        gui_q.put(f"[{thread_name}] Selected device: {device_info.get('name')}")
        if device_info.get('maxInputChannels') < AUDIO_CHANNELS:
            raise ValueError(f"Device does not support required channels ({AUDIO_CHANNELS})")

        stream = p.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_SAMPLERATE,
                        input=True, frames_per_buffer=PYAUDIO_FRAMES_PER_BUFFER,
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
                if e.errno == pyaudio.paInputOverflowed:
                    gui_q.put(f"[{thread_name}] Warning: PyAudio Input Overflowed.")
                else:
                    gui_q.put(f"[{thread_name}] ERROR: PyAudio read error: {e}")
                    time.sleep(0.1)
        if stop_event_flag.is_set():
            gui_q.put(f"[{thread_name}] Stop event received.")
    except ValueError as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Device config error: {e}")
    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Initializing/running PyAudio: {e}")
    finally:
        audio_q.put(None) # Signal downstream thread
        if stream: stream.stop_stream(); stream.close()
        if p: p.terminate()
        gui_q.put(f"[{thread_name}] Audio capture thread finished.")
        # Don't signal GUI end here

def transcribe_thread_gui(audio_q, transcribed_q, gui_q, stop_event_flag):
    """ Transcribes audio using Whisper model, detects language, puts (lang, text) on transcribed_q, checks stop_event. """
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Initializing Whisper model ({WHISPER_MODEL_SIZE})...")
    model = None
    detection_phase = True
    locked_language = None
    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        gui_q.put(f"[{thread_name}] Whisper model initialized. Detecting initial language...")

        while not stop_event_flag.is_set():
            try:
                audio_data_bytes = audio_q.get(timeout=0.5)
                if audio_data_bytes is None: break
            except queue.Empty: continue

            if stop_event_flag.is_set(): break

            try:
                audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                if audio_np.size > 0:
                    current_language_setting = locked_language if not detection_phase else None
                    segments, info = model.transcribe(
                        audio_np, language=current_language_setting, beam_size=5,
                        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    full_text = " ".join(segment.text for segment in segments).strip()

                    if detection_phase:
                        if full_text:
                            locked_language = info.language
                            detection_phase = False
                            gui_q.put(f"[{thread_name}] Initial language detected and locked: {locked_language} (Prob: {info.language_probability:.2f})")
                            if locked_language:
                                transcribed_q.put((locked_language, full_text))
                    else:
                        if full_text and locked_language:
                            transcribed_q.put((locked_language, full_text))
            except Exception as e:
                gui_q.put(f"[{thread_name}] ERROR during transcription: {e}")
            finally:
                audio_q.task_done()

        if stop_event_flag.is_set():
            gui_q.put(f"[{thread_name}] Stop event received.")

    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Failed to load/run Whisper model: {e}")
    finally:
        transcribed_q.put(None) # Signal downstream thread
        if model is not None:
             try:
                 del model
                 if WHISPER_DEVICE == 'cuda': torch.cuda.empty_cache()
                 gui_q.put(f"[{thread_name}] Whisper model resources released.")
             except Exception as e: gui_q.put(f"[{thread_name}] Warning: Error releasing model resources: {e}")
        gui_q.put(f"[{thread_name}] Transcription thread finished.")
        # Don't signal GUI end here

def translate_thread_gui(transcribed_q, gui_q, target_lang_code, stop_event_flag):
    """ Translates text using Ollama, puts results on gui_q, checks stop_event. """
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Initializing Ollama client (Model: {OLLAMA_MODEL}, Target: {target_lang_code})...")
    try:
        ollama.list() # Ping Ollama
        gui_q.put(f"[{thread_name}] Ollama connection successful. Waiting for text...")
    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Ollama connection failed: {e}")
        gui_q.put(None) # Signal GUI about the failure
        return

    while not stop_event_flag.is_set():
        try:
            data = transcribed_q.get(timeout=0.5)
            if data is None: break
        except queue.Empty: continue

        if stop_event_flag.is_set(): break

        source_lang_code, text_to_translate = data
        if not target_lang_code:
             gui_q.put(f"[{thread_name}] Error: No target language selected for translation.")
             transcribed_q.task_done(); continue

        try:
            if text_to_translate:
                start_time = time.time()
                # Using the strict prompt from the original script
                context_info = "Context: The text is dialogue from a Japanese Hololive VTuber live stream."
                instruction = f"Translate the following Source Text into {target_lang_code}. IMPORTANT: Your entire response must consist ONLY of the raw translated text in {target_lang_code} and absolutely nothing else. Do not add explanations, commentary, notes, labels, source language text, or translations into other languages."
                prompt = f"{context_info}\n\n{instruction}\n\nSource Text ({source_lang_code}): {text_to_translate}\n\n{target_lang_code} Translation:"

                response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, stream=False,
                                           options={"temperature": 0.5}) # Keep low temp
                translation = response['response'].strip()
                latency = time.time() - start_time

                if translation:
                    # Basic filtering (can be enhanced)
                    if "(translation from" in translation.lower():
                        gui_q.put(f"[{thread_name}] Warning: Filtered out potential explanation.")
                        translation = translation.split("\n")[0].strip()
                    import re
                    translation = re.sub(r'\s*\([^)]*\)$', '', translation).strip()

                    if translation:
                        # Send translation for history deque, using target lang code prefix
                        gui_q.put(f"[{target_lang_code.upper()} ({latency:.2f}s)] {translation}")
                    elif len(text_to_translate.strip()) > 2: # Check length to avoid logging noise
                         gui_q.put(f"[{thread_name}] Warning: Translation empty after filtering.")
                elif len(text_to_translate.strip()) > 2:
                    gui_q.put(f"[{thread_name}] Warning: Empty translation response from LLM.")
        except Exception as e:
            gui_q.put(f"[{thread_name}] ERROR during translation: {e}")
        finally:
            transcribed_q.task_done()

    if stop_event_flag.is_set():
        gui_q.put(f"[{thread_name}] Stop event received.")

    gui_q.put(f"[{thread_name}] Translation thread finished.")
    gui_q.put(None) # Signal GUI that this processing pipeline is done


# --- Tkinter GUI Application (Adapted from Old Code, using PyAudio input) ---

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multilingual Live Translator (PyAudio + Ollama)")
        self.root.geometry("700x350") # Keep original size

        # --- Dark Theme Configuration (Copied from original) ---
        self.root.config(bg=DARK_BG)
        style = ttk.Style(self.root)
        try: style.theme_use('clam')
        except tk.TclError: print("Warning: 'clam' theme not available, using default.")
        style.configure('.', background=DARK_BG, foreground=LIGHT_FG, font=('Segoe UI', 9))
        style.configure('TFrame', background=DARK_BG)
        style.configure('TLabel', background=DARK_BG, foreground=LIGHT_FG)
        style.configure('TButton', background=BUTTON_BG, foreground=LIGHT_FG, padding=5)
        style.map('TButton', background=[('active', BUTTON_ACTIVE_BG), ('disabled', DARK_BG)])
        style.configure('TCombobox', fieldbackground=ENTRY_BG, background=BUTTON_BG, foreground=LIGHT_FG,
                        arrowcolor=LIGHT_FG, selectbackground=ENTRY_BG, selectforeground=LIGHT_FG,
                        insertcolor=TEXT_CURSOR)
        style.map('TCombobox', fieldbackground=[('readonly', ENTRY_BG)],
                           selectbackground=[('!focus', ENTRY_BG)],
                           background=[('readonly', BUTTON_BG)])
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

        # --- Device Selection ---
        ttk.Label(input_frame, text="Input Device:").pack(side=tk.LEFT, padx=(0, 5))
        self.devices = get_audio_devices()
        self.device_map = {f"{d['index']}: {d['name']} (Rate: {int(d['rate'])})": d['index'] for d in self.devices}
        self.device_combobox = ttk.Combobox(input_frame, values=list(self.device_map.keys()),
                                             state="readonly", width=40) # Adjust width if needed
        if self.device_map: self.device_combobox.current(0)
        self.device_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # --- Target Language Selection ---
        ttk.Label(input_frame, text="To:").pack(side=tk.LEFT, padx=(5, 2))
        self.target_lang_var = tk.StringVar()
        lang_options = list(SUPPORTED_TARGET_LANGUAGES.keys())
        self.lang_combobox = ttk.Combobox(input_frame, textvariable=self.target_lang_var,
                                             values=lang_options, state="readonly", width=15)
        if DEFAULT_TARGET_LANG in lang_options: self.target_lang_var.set(DEFAULT_TARGET_LANG)
        elif lang_options: self.target_lang_var.set(lang_options[0])
        self.lang_combobox.pack(side=tk.LEFT, padx=(0, 5))

        # --- Start/Stop Buttons ---
        self.start_button = ttk.Button(input_frame, text="Start", command=self.start_translation, width=7)
        self.start_button.pack(side=tk.LEFT, padx=(5, 2))
        self.stop_button = ttk.Button(input_frame, text="Stop", command=self.stop_translation, state=tk.DISABLED, width=7)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 0))

        # --- Output Area ---
        output_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state=tk.DISABLED, height=10)
        self.output_text.config(bg=DARK_BG, fg=LIGHT_FG, insertbackground=TEXT_CURSOR, font=('Segoe UI', 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        # Define tags for coloring
        self.output_text.tag_config("error", foreground=ERROR_FG)
        self.output_text.tag_config("info", foreground=INFO_FG)
        # Add tag for each language prefix? Or keep default? Keep default for now.
        # for code in SUPPORTED_TARGET_LANGUAGES.values():
        #     self.output_text.tag_config(code.upper(), foreground=LIGHT_FG) # Example

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Select Input Device & Target Language, then press Start.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.FLAT, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # add_output_direct and update_output_display methods remain largely the same as the previous version
    # Minor tweaks might be needed in update_output_display for tag handling if desired
    def add_output_direct(self, text, tag=None):
        """ Adds text (like status/errors) directly without history limit, applying optional tag """
        if not self.root.winfo_exists(): return
        try:
            self.output_text.config(state=tk.NORMAL)
            if tag: self.output_text.insert(tk.END, text + "\n", tag)
            else: self.output_text.insert(tk.END, text + "\n")
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        except tk.TclError as e: print(f"Error adding text to GUI: {e}")

    def update_output_display(self):
        """ Clears and rewrites the text area from the message deque """
        if not self.root.winfo_exists(): return
        try:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete('1.0', tk.END)
            for _, msg in self.message_deque:
                 # Basic tag assignment based on prefix (can be expanded)
                tag_to_use = None
                if msg.startswith("[ERROR"): tag_to_use = "error"
                elif msg.startswith("[INFO") or msg.startswith("[Audio") or msg.startswith("[Transcribe") or msg.startswith("[Translate"): tag_to_use = "info"
                # Add other checks if needed e.g., for language codes like [EN ...]
                # elif msg.startswith("[EN"): tag_to_use = "EN" # If tags defined above

                if tag_to_use: self.output_text.insert(tk.END, msg + "\n", tag_to_use)
                else: self.output_text.insert(tk.END, msg + "\n")

            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        except tk.TclError as e: print(f"Error updating GUI display: {e}")


    def check_gui_queue(self):
        """ Checks the GUI queue, manages history deque, and updates the GUI """
        if not self.is_running: return

        try:
            while True:
                message = gui_queue.get_nowait()
                current_time = time.time()

                if message is None:
                    # Translate thread finished
                    self.status_var.set("Processing stopped or finished.")
                    self.add_output_direct("--- Processing thread finished ---", "info")
                    self.stop_translation(graceful_stop=True)
                    return

                elif isinstance(message, str):
                    # Check if it's a translation message using defined lang codes
                    is_translation = False
                    for code in SUPPORTED_TARGET_LANGUAGES.values():
                        if message.startswith(f"[{code.upper()}"):
                            is_translation = True
                            break

                    if is_translation:
                        self.message_deque.append((current_time, message))
                        cutoff_time = current_time - (HISTORY_MINUTES * 60)
                        while self.message_deque and self.message_deque[0][0] < cutoff_time:
                            self.message_deque.popleft()
                        self.update_output_display() # Update view with new history
                        # self.status_var.set("Processing...") # Maybe too noisy
                    elif message.startswith("CRITICAL ERROR"):
                        self.status_var.set(message.split("] ", 1)[-1])
                        self.add_output_direct(f"*** {message} ***", "error")
                        self.stop_translation(error=True)
                        return
                    elif message.startswith("[ERROR") or message.startswith("[Warning"):
                        self.status_var.set("Error/Warning occurred.")
                        self.add_output_direct(message, "error")
                    elif message.startswith("["): # Status/info
                        status_part = message.split("] ", 1)[-1]
                        if "initialized" in status_part or "Capturing" in status_part or "finished" in status_part:
                             self.status_var.set(status_part)
                        # Log all info to the text area
                        self.add_output_direct(message, "info")
                    else: # Unformatted
                        self.add_output_direct(message, "info")

                gui_queue.task_done()
        except queue.Empty: pass
        except Exception as e:
            print(f"Error processing GUI queue: {e}")
            self.add_output_direct(f"[ERROR] GUI Error: {e}", "error")

        if self.is_running and self.root.winfo_exists():
            self.root.after(100, self.check_gui_queue)


    def start_translation(self):
        """ Starts the translation process (PyAudio -> Whisper -> Ollama) """
        if self.is_running:
            messagebox.showwarning("Already Running", "Translation is already in progress.")
            return

        selected_device_str = self.device_combobox.get()
        if not selected_device_str: messagebox.showerror("Input Error", "Please select an Input Device."); return
        device_index = self.device_map.get(selected_device_str)
        if device_index is None: messagebox.showerror("Internal Error", "Invalid device selection."); return

        selected_lang_name = self.target_lang_var.get()
        if not selected_lang_name: messagebox.showerror("Input Error", "Please select a target language."); return
        selected_target_lang_code = SUPPORTED_TARGET_LANGUAGES.get(selected_lang_name, None)
        if not selected_target_lang_code: messagebox.showerror("Internal Error", "Invalid target language selected."); return

        try:
            self.output_text.config(state=tk.NORMAL); self.output_text.delete('1.0', tk.END); self.output_text.config(state=tk.DISABLED)
            self.message_deque.clear()
            self.status_var.set(f"Starting... Device: {device_index}, Target: {selected_lang_name}")
            self.add_output_direct("--- Starting translation ---", "info")
            self.is_running = True
            stop_event.clear()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.device_combobox.config(state=tk.DISABLED)
            self.lang_combobox.config(state=tk.DISABLED)
            self.active_threads = []
        except tk.TclError: print("Error starting: GUI already closed?"); return

        while not audio_queue.empty(): audio_queue.get_nowait()
        while not transcribed_queue.empty(): transcribed_queue.get_nowait()
        while not gui_queue.empty(): gui_queue.get_nowait()

        # --- Create and Start Threads (Capture -> Transcribe -> Translate) ---
        capture = threading.Thread(target=capture_audio_thread_gui, args=(device_index, audio_queue, gui_queue, stop_event), name="AudioCapture", daemon=True)
        transcriber = threading.Thread(target=transcribe_thread_gui, args=(audio_queue, transcribed_queue, gui_queue, stop_event), name="Transcribe", daemon=True)
        translator = threading.Thread(target=translate_thread_gui, args=(transcribed_queue, gui_queue, selected_target_lang_code, stop_event), name="Translate", daemon=True)

        self.active_threads.extend([capture, transcriber, translator])
        capture.start()
        transcriber.start()
        translator.start()

        self.root.after(100, self.check_gui_queue) # Start polling GUI queue


    def stop_translation(self, error=False, graceful_stop=False):
        """ Stops the translation process using the stop_event """
        if not self.is_running and not error and not graceful_stop: return

        status = "Processing stopped by user."
        if error: status = "Processing stopped due to CRITICAL error."
        if graceful_stop: status = "Processing finished."

        if self.is_running: stop_event.set() # Signal threads via event
        self.is_running = False # Stop queue checking loop

        try:
            if self.root.winfo_exists():
                self.status_var.set(status)
                self.add_output_direct(f"--- {status} ---", "info" if not error else "error")
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.device_combobox.config(state="readonly")
                self.lang_combobox.config(state="readonly")
        except tk.TclError: print("Info: GUI window closed during stop sequence.")

        # Clear queues (best effort)
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break # Exit the while loop if queue is empty

        while not transcribed_queue.empty():
            try:
                transcribed_queue.get_nowait()
            except queue.Empty:
                break # Exit the while loop if queue is empty

        self.active_threads = []
        print("Stop signal sent.")


    def on_closing(self):
        """ Handle window closing """
        if self.is_running:
             if messagebox.askokcancel("Quit", "Translation is running. Stop and quit?"):
                 self.stop_translation()
                 self.root.after(200, self.root.destroy) # Delay destroy slightly
             else: return
        else: self.root.destroy()


# --- Main Execution ---
if __name__ == "__main__":
    # Optional: Check PyAudio and Ollama early
    try:
        p = pyaudio.PyAudio(); p.terminate()
    except Exception as e:
        root = tk.Tk(); root.withdraw(); messagebox.showerror("PyAudio Error", f"PyAudio init failed: {e}"); sys.exit(1)
    try:
        ollama.list()
        print("Ollama connection check successful.")
    except Exception as e:
        print(f"WARNING: Ollama connection check failed: {e}. Ensure Ollama is running.", file=sys.stderr)
        root = tk.Tk(); root.withdraw(); messagebox.showwarning("Ollama Warning", f"Could not connect to Ollama. Translation may fail.\nError: {e}")
        # Allow GUI to start anyway, but translation thread will likely fail.

    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()