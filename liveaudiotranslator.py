import queue
import threading
import sys
import time
from collections import deque
import numpy as np
import torch
from faster_whisper import WhisperModel
import pyaudio # Added dependency
import re # For ignore patterns

import tkinter as tk
from tkinter import ttk, messagebox, font # Removed scrolledtext as it's replaced

# --- Configuration ---

# --- MODIFICATION: Added Source Language and Model Size Config ---
SUPPORTED_SOURCE_LANGUAGES = {
    # Display Name: Whisper language code (or None for auto)
    "Auto Detect": None,
    "English": "en",
    "Japanese": "ja",
    "Chinese": "zh",
    "Spanish": "es",
    "Russian": "ru",
    "Korean": "ko",
    "French": "fr",
    "German": "de",
    "Indonesian": "id",
    # Add more languages as needed from Whisper's supported list
}
DEFAULT_SOURCE_LANG = "Auto Detect"

SUPPORTED_MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"] # large-v2 removed for simplicity, add if needed
DEFAULT_MODEL_SIZE = "medium" # Default to small for broader compatibility
# --- END MODIFICATION ---


# Whisper settings
# WHISPER_MODEL_SIZE = "medium" # <-- REMOVED global constant
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "float32" # Explicitly use float32 on CPU

# Audio settings
AUDIO_SAMPLERATE = 16000 # Whisper requires 16kHz
AUDIO_CHANNELS = 1 # Mono
AUDIO_FORMAT = pyaudio.paInt16 # PyAudio format constant
AUDIO_BUFFER_SECONDS = 3 # Process audio in chunks of this duration (Adjust if needed)
PYAUDIO_FRAMES_PER_BUFFER = 1024 # How many frames PyAudio reads at a time

# Ignore Patterns (using Regex)
IGNORE_PATTERNS = [
    re.compile(r"(?:thank\s*(?:you|s)\s*(?:so|very|a\s*lot)?\s*(?:much)?\s*)?(?:for\s+(?:watching|viewing|your\s+viewing)).*?(?:in\s+this\s+video)?", re.IGNORECASE),
    re.compile(r"see\s+you(?:\s+all|\s+again)?\s+(?:next\s+(?:time|video)|later|in\s+the\s+next\s+video)", re.IGNORECASE),
    re.compile(r"subscribe\s+to\s+(?:my|the)\s+channel", re.IGNORECASE), # Example addition
]

# --- Appearance & History Settings ---
MAX_HISTORY_MESSAGES = 10 # Change to what you want
WINDOW_ALPHA = 0.60      # Window transparency (0.0=invisible to 1.0=opaque)

# --- Color Constants for Dark Theme ---
DEFAULT_FONT_SIZE = 14    # Default font size for output
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 24
DARK_BG = '#2b2b2b'          # Dark gray background
LIGHT_FG = '#ffffff'         # White text
ENTRY_BG = '#3c3f41'         # Slightly lighter gray for entry background
BUTTON_BG = '#555555'         # Medium gray for button
BUTTON_ACTIVE_BG = '#666666' # Slightly lighter gray for active/hover button
TEXT_CURSOR = '#ffffff'      # White cursor color
INFO_FG = '#6cace4'          # Light blue for info messages
ERROR_FG = '#e46c6c'         # Light red for error messages
SCROLLBAR_BG = '#454545'      # Color for the scrollbar thumb/arrows
SCROLLBAR_TROUGH = '#333333' # Color for the scrollbar channel/track

# --- Global Queues & Events ---
audio_queue = queue.Queue()
gui_queue = queue.Queue()      # Single queue for all GUI updates
stop_event = threading.Event() # Event to signal threads to stop

# --- Helper Function ---
def get_audio_devices():
    """Gets a list of available audio input devices."""
    devices = []
    p = pyaudio.PyAudio()
    try:
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount', 0)
        for i in range(0, numdevices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels', 0) > 0:
                devices.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxInputChannels'),
                    'rate': device_info.get('defaultSampleRate')
                })
    except Exception as e:
        print(f"Error getting audio devices: {e}")
    finally:
        p.terminate()
    return devices

# --- Thread Functions (Adapted for PyAudio, Whisper Translate, GUI Queue, Stop Event) ---

def capture_audio_thread_gui(device_index, audio_q, gui_q, stop_event_flag):
    # (Function remains the same as previous version)
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Starting audio capture from device index {device_index}...")
    audio_buffer = bytearray()
    target_buffer_size = int(AUDIO_SAMPLERATE * AUDIO_CHANNELS * 2 * AUDIO_BUFFER_SECONDS)
    p = None
    stream = None
    try:
        p = pyaudio.PyAudio()
        device_info = p.get_device_info_by_index(device_index)
        gui_q.put(f"[{thread_name}] Selected device: {device_info.get('name')}")
        if device_info.get('maxInputChannels', 0) < AUDIO_CHANNELS:
              raise ValueError(f"Device does not support required channels ({AUDIO_CHANNELS})")

        stream = p.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_SAMPLERATE,
                        input=True, frames_per_buffer=PYAUDIO_FRAMES_PER_BUFFER,
                        input_device_index=device_index)
        gui_q.put(f"[{thread_name}] Audio stream opened. Capturing...")
        while not stop_event_flag.is_set():
            try:
                if stop_event_flag.is_set(): break
                data = stream.read(PYAUDIO_FRAMES_PER_BUFFER, exception_on_overflow=False)
                audio_buffer.extend(data)
                while len(audio_buffer) >= target_buffer_size:
                    if stop_event_flag.is_set(): break
                    chunk_to_process = audio_buffer[:target_buffer_size]
                    audio_q.put(bytes(chunk_to_process))
                    audio_buffer = audio_buffer[target_buffer_size:]
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed or str(e) == "Input overflowed":
                    gui_q.put(f"[{thread_name}] Warning: PyAudio Input Overflowed.")
                else:
                    gui_q.put(f"[{thread_name}] ERROR: PyAudio read error: {e}")
                    time.sleep(0.1) # Brief pause on error
            except Exception as e:
                gui_q.put(f"[{thread_name}] ERROR: Unexpected error during stream read: {e}")
                time.sleep(0.1) # Brief pause on error

        if stop_event_flag.is_set(): gui_q.put(f"[{thread_name}] Stop event received, exiting capture loop.")
    except ValueError as e: gui_q.put(f"[{thread_name}] CRITICAL ERROR: Device config error: {e}")
    except Exception as e: gui_q.put(f"[{thread_name}] CRITICAL ERROR: Initializing/running PyAudio: {e}")
    finally:
        gui_q.put(f"[{thread_name}] Cleaning up audio resources...")
        if stream is not None:
            try:
                if stream.is_active(): stream.stop_stream()
                stream.close(); gui_q.put(f"[{thread_name}] Audio stream closed.")
            except Exception as e: gui_q.put(f"[{thread_name}] Warning: Error closing stream: {e}")
        if p is not None: p.terminate(); gui_q.put(f"[{thread_name}] PyAudio terminated.")
        # Signal end by putting None in the *next* queue (audio_queue)
        audio_q.put(None)
        gui_q.put(f"[{thread_name}] Audio capture thread finished.")


# --- MODIFICATION: Added model_size and source_language_code arguments ---
def transcribe_translate_thread_gui(audio_q, gui_q, stop_event_flag, model_size, source_language_code):
    """ Transcribes audio using chosen Whisper model/language and translates directly to English, filtering unwanted phrases. """
    thread_name = threading.current_thread().name
    # --- MODIFICATION: Use selected model_size in message ---
    gui_q.put(f"[{thread_name}] Initializing Whisper model ({model_size})...")
    model = None
    try:
        # --- MODIFICATION: Load selected model_size ---
        model = WhisperModel(model_size, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        gui_q.put(f"__INIT_STATUS__Whisper model '{model_size}' initialized on {WHISPER_DEVICE}. Ready for translation.")

        while not stop_event_flag.is_set():
            try:
                # Wait for audio data from the capture thread
                audio_data_bytes = audio_q.get(timeout=0.5) # Use timeout to allow checking stop_event
                if audio_data_bytes is None:
                    # This is the signal from the capture thread that it's done
                    gui_q.put(f"[{thread_name}] Received None from audio queue, stopping.")
                    break # Exit the loop
            except queue.Empty:
                # Timeout occurred, loop again to check stop_event or wait longer
                continue

            # Check stop event *after* getting data (or timeout) but before processing
            if stop_event_flag.is_set():
                gui_q.put(f"[{thread_name}] Stop event detected after getting audio data/timeout.")
                # If we received actual data before stop, mark it as done if queue requires it
                # if audio_data_bytes is not None: audio_q.task_done() # Not needed for standard queue unless joining
                break

            # --- Process the audio data ---
            try:
                audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                if audio_np.size > 0:
                    start_time = time.time()
                    # --- MODIFICATION: Pass selected source_language_code ---
                    # language=None means auto-detect, which is handled by the GUI selection mapping
                    segments, info = model.transcribe(
                        audio_np,
                        language=source_language_code, # Pass selected language code (or None)
                        beam_size=5, # Adjust beam size as needed
                        vad_filter=False, # VAD filter can sometimes cut off speech; disable if unsure
                        task='translate' # Force translation to English
                    )
                    # --- END MODIFICATION ---

                    # Use detected language even if source was specified, for info display
                    detected_lang = info.language
                    detected_prob = info.language_probability
                    latency = time.time() - start_time
                    full_translation = " ".join(segment.text for segment in segments).strip()

                    # Use Regex Ignore Patterns
                    should_ignore = False
                    if full_translation:
                        normalized_translation_for_regex = full_translation.strip()
                        for pattern in IGNORE_PATTERNS:
                            if pattern.search(normalized_translation_for_regex): # Use search()
                                should_ignore = True
                                print(f"Ignoring phrase matching pattern '{pattern.pattern}': '{full_translation}'")
                                break

                    # Send valid, non-ignored translation to GUI
                    if full_translation and not should_ignore:
                        # Display detected language for info, even if source was set
                        gui_q.put(f"[EN ({latency:.2f}s, src={detected_lang}:{detected_prob:.2f})] {full_translation}")
                    # else: print("Translation empty or ignored.") # Optional debug

                # else: Audio chunk was empty or too small

            except Exception as e:
                # Log errors during transcription/translation
                gui_q.put(f"[{thread_name}] ERROR during translation: {e}")
            # finally:
                # Mark task done if using JoinableQueue, not needed for standard Queue generally
                # audio_q.task_done() # Removed, not needed for standard queue clear/get pattern here

        # End of while loop (either stopped or audio queue signaled done)
        if stop_event_flag.is_set(): gui_q.put(f"[{thread_name}] Stop event was set, exiting transcription loop.")

    except Exception as e:
        # --- MODIFICATION: Include model_size in critical error message ---
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Failed to load/run Whisper model '{model_size}': {e}")
    finally:
        # --- Cleanup ---
        if model is not None:
            try:
                gui_q.put(f"[{thread_name}] Releasing Whisper model resources...")
                # For faster-whisper, explicit deletion helps trigger GC
                del model
                if WHISPER_DEVICE == 'cuda':
                    torch.cuda.empty_cache() # Clear CUDA cache if applicable
                gui_q.put(f"[{thread_name}] Whisper model resources released.")
            except Exception as e:
                gui_q.put(f"[{thread_name}] Warning: Error releasing model resources: {e}")

        # Signal that this thread is finished by putting None in the GUI queue
        gui_q.put(f"[{thread_name}] Transcription/Translation thread finished.")
        gui_q.put(None) # Signal GUI that this processing pipeline is done


# --- Tkinter GUI Application (Adding Language and Model Selection) ---

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Audio Translator (Whisper)")
        # Adjusted geometry slightly for the new slider
        self.root.geometry("920x400")

        # --- Dark Theme Configuration ---
        self.root.config(bg=DARK_BG)
        style = ttk.Style(self.root)
        try:
            if 'clam' in style.theme_names(): style.theme_use('clam')
            else: print("Warning: 'clam' theme not available, using default.")
        except tk.TclError: print("Warning: Error setting 'clam' theme, using default.")

        # --- Style Definitions (Added Scale) ---
        style.configure('.', background=DARK_BG, foreground=LIGHT_FG, font=('Segoe UI', 9))
        style.configure('TFrame', background=DARK_BG)
        style.configure('TLabel', background=DARK_BG, foreground=LIGHT_FG)
        style.configure('TButton', background=BUTTON_BG, foreground=LIGHT_FG, padding=5, borderwidth=0, relief=tk.FLAT)
        style.map('TButton', background=[('active', BUTTON_ACTIVE_BG), ('disabled', '#444444')], foreground=[('disabled', '#999999')])
        style.configure('TCombobox', fieldbackground=ENTRY_BG, background=BUTTON_BG, foreground=LIGHT_FG, arrowcolor=LIGHT_FG, selectbackground=ENTRY_BG, selectforeground=LIGHT_FG, insertcolor=TEXT_CURSOR, borderwidth=0, padding=(5, 3))
        style.map('TCombobox', fieldbackground=[('readonly', ENTRY_BG)], selectbackground=[('!focus', ENTRY_BG)], background=[('readonly', BUTTON_BG)], foreground=[('disabled', '#999999')])
        root.option_add('*TCombobox*Listbox.background', ENTRY_BG); root.option_add('*TCombobox*Listbox.foreground', LIGHT_FG)
        root.option_add('*TCombobox*Listbox.selectBackground', BUTTON_ACTIVE_BG); root.option_add('*TCombobox*Listbox.selectForeground', LIGHT_FG)
        root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 9))
        style.configure('Vertical.TScrollbar', gripcount=0, background=SCROLLBAR_BG, troughcolor=SCROLLBAR_TROUGH, bordercolor=DARK_BG, arrowcolor=LIGHT_FG, relief=tk.FLAT)
        style.map('Vertical.TScrollbar', background=[('active', BUTTON_ACTIVE_BG)], arrowcolor=[('pressed', '#cccccc'), ('disabled', '#666666')])
        # Basic Scale styling <<<< ADDED
        style.configure('Horizontal.TScale', background=DARK_BG, troughcolor=ENTRY_BG)
        style.map('Horizontal.TScale', background=[('active', BUTTON_ACTIVE_BG)], troughcolor=[('active', ENTRY_BG)])

        # --- Window Attributes ---
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', WINDOW_ALPHA) # Set initial alpha <<<< ADDED

        # --- App State Variables ---
        self.is_running = False
        self.active_threads = []
        self.message_deque = deque()
        self.initial_status_shown = False
        # This version has 2 processing threads (Capture + TranscribeTranslate)
        self.expected_threads = 2

        # =============================================
        # --- GUI Layout ---
        # =============================================
        control_frame = ttk.Frame(root, padding="10"); control_frame.pack(fill=tk.X)
        # Device Selection
        ttk.Label(control_frame, text="Input Device:").pack(side=tk.LEFT, padx=(0, 5))
        self.devices = get_audio_devices()
        if not self.devices:
            self.device_map = {"No Input Devices Found": None}; messagebox.showerror("Audio Device Error", "No audio input devices found.")
        else:
            self.device_map = {f"{d['index']}: {d['name']} (Rate: {int(d.get('rate', 0))})": d['index'] for d in self.devices}
        self.device_combobox = ttk.Combobox(control_frame, values=list(self.device_map.keys()), state="readonly", width=30)
        if self.device_map and list(self.device_map.values())[0] is not None: self.device_combobox.current(0)
        self.device_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))
        self.start_button_state = tk.NORMAL if any(idx is not None for idx in self.device_map.values()) else tk.DISABLED
        # Source Language Selection
        ttk.Label(control_frame, text="Src Lang:").pack(side=tk.LEFT, padx=(0, 2))
        self.source_lang_var = tk.StringVar()
        lang_options = list(SUPPORTED_SOURCE_LANGUAGES.keys())
        self.source_lang_combobox = ttk.Combobox(control_frame, textvariable=self.source_lang_var, values=lang_options, state="readonly", width=10)
        if DEFAULT_SOURCE_LANG in lang_options: self.source_lang_var.set(DEFAULT_SOURCE_LANG)
        elif lang_options: self.source_lang_var.set(lang_options[0])
        self.source_lang_combobox.pack(side=tk.LEFT, padx=(0, 10))
        # Model Size Selection
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 2))
        self.model_size_var = tk.StringVar()
        self.model_size_combobox = ttk.Combobox(control_frame, textvariable=self.model_size_var, values=SUPPORTED_MODEL_SIZES, state="readonly", width=8)
        if DEFAULT_MODEL_SIZE in SUPPORTED_MODEL_SIZES: self.model_size_var.set(DEFAULT_MODEL_SIZE)
        elif SUPPORTED_MODEL_SIZES: self.model_size_var.set(SUPPORTED_MODEL_SIZES[0])
        self.model_size_combobox.pack(side=tk.LEFT, padx=(0, 10))
        # Font Size Control
        ttk.Label(control_frame, text="Size:").pack(side=tk.LEFT, padx=(0, 2))
        self.font_size_var = tk.IntVar(value=DEFAULT_FONT_SIZE)
        self.font_size_spinbox = tk.Spinbox(
            control_frame, from_=MIN_FONT_SIZE, to=MAX_FONT_SIZE, textvariable=self.font_size_var, width=3, command=self.update_font_size,
            state="readonly", bg=ENTRY_BG, fg=LIGHT_FG, buttonbackground=BUTTON_BG, relief=tk.FLAT, readonlybackground=ENTRY_BG,
            highlightthickness=1, highlightbackground=DARK_BG, highlightcolor=BUTTON_ACTIVE_BG, insertbackground=TEXT_CURSOR, buttoncursor="arrow",
            disabledbackground=ENTRY_BG, disabledforeground="#999999"
        )
        self.font_size_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        # --- Alpha/Transparency Slider --- <<<< ADDED
        ttk.Label(control_frame, text="Alpha:").pack(side=tk.LEFT, padx=(10, 2))
        self.alpha_var = tk.DoubleVar(value=WINDOW_ALPHA)
        self.alpha_slider = ttk.Scale(
            control_frame, from_=0.2, to=1.0, orient=tk.HORIZONTAL,
            variable=self.alpha_var, command=self.update_alpha,
            length=70, style='Horizontal.TScale'
        )
        self.alpha_slider.pack(side=tk.LEFT, padx=(0, 10))
        # --- END Alpha Slider ---

        # Start/Stop Buttons
        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_translation, width=6, style='TButton', state=self.start_button_state)
        self.start_button.pack(side=tk.LEFT, padx=(0, 2))
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_translation, state=tk.DISABLED, width=6, style='TButton')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 0))

        # Output Area
        output_frame = ttk.Frame(root, padding=(10, 0, 10, 10)); output_frame.pack(fill=tk.BOTH, expand=True)
        self.scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, style="Vertical.TScrollbar")
        self.output_text = tk.Text(
            output_frame, wrap=tk.WORD, state=tk.DISABLED, height=10, bg=DARK_BG, fg=LIGHT_FG, insertbackground=TEXT_CURSOR,
            selectbackground=BUTTON_BG, selectforeground=LIGHT_FG, relief=tk.FLAT, borderwidth=0, highlightthickness=0,
            yscrollcommand=self.scrollbar.set, font=('Segoe UI', self.font_size_var.get())
        )
        self.scrollbar.config(command=self.output_text.yview); self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.output_text.tag_config("error", foreground=ERROR_FG); self.output_text.tag_config("info", foreground=INFO_FG)
        self.output_text.tag_config("translation", foreground=LIGHT_FG)

        # Status Bar
        self.status_var = tk.StringVar()
        initial_status = "Ready." if self.start_button_state == tk.NORMAL else "No input devices found."
        self.status_var.set(initial_status)
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.FLAT, anchor=tk.W, padding=5, style='TLabel')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Initialize font object cache
        self._output_font = font.Font(font=self.output_text['font'])

    # --- App Methods ---

    # --- update_alpha method --- <<<< ADDED
    def update_alpha(self, value=None): # value is passed by the Scale command
        """Updates the window transparency based on the slider value."""
        if not self.root.winfo_exists(): return
        try:
            new_alpha = self.alpha_var.get()
            self.root.attributes('-alpha', new_alpha)
        except tk.TclError as e: print(f"Info: TclError updating alpha (window might be closing): {e}")
        except Exception as e: print(f"Error updating alpha: {e}")
    # --- END update_alpha method ---

    def update_font_size(self):
        if not self.root.winfo_exists(): return
        new_size = self.font_size_var.get()
        try:
            self.output_text.config(font=(self._output_font.actual()['family'], new_size))
        except tk.TclError as e: print(f"Error updating font size (Tcl): {e}")
        except Exception as e: print(f"Error updating font size: {e}")

    def add_output_direct(self, text, tag=None):
        if not self.root.winfo_exists(): return
        try:
            is_scrolled_to_bottom = self.output_text.yview()[1] >= 0.99
            self.output_text.config(state=tk.NORMAL)
            if tag: self.output_text.insert(tk.END, text + "\n", tag)
            else: self.output_text.insert(tk.END, text + "\n")
            self.output_text.config(state=tk.DISABLED)
            if is_scrolled_to_bottom: self.output_text.see(tk.END)
        except tk.TclError as e: print(f"Error adding text to GUI (Tcl): {e}")
        except Exception as e: print(f"Error adding text to GUI: {e}")

    def update_output_display(self):
        if not self.root.winfo_exists(): return
        try:
            current_scroll = self.output_text.yview()
            scroll_pos_before_update = current_scroll[0] if current_scroll[1] < 0.99 else None
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete('1.0', tk.END)
            for _, msg in self.message_deque:
                tag_to_use = "translation"
                if msg.startswith("[EN"): tag_to_use = "translation"
                elif msg.startswith("[ERROR") or "CRITICAL ERROR" in msg: tag_to_use = "error"
                elif msg.startswith("[INFO") or msg.startswith("[Audio") or msg.startswith("[Transcribe") or msg.startswith("[Warning") or msg.startswith("---"): tag_to_use = "info"
                self.output_text.insert(tk.END, msg + "\n", tag_to_use)
            if scroll_pos_before_update is not None: self.output_text.yview_moveto(scroll_pos_before_update)
            else: self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        except tk.TclError as e: print(f"Error updating GUI display (Tcl): {e}")
        except Exception as e: print(f"Error updating GUI display: {e}")

    def check_gui_queue(self): # Includes fixes for thread completion tracking
        """ Checks the GUI queue for messages from threads and updates the UI. """
        if not self.is_running and self.threads_completed < self.expected_threads: pass
        elif not self.is_running: return

        process_queue = True
        initial_call = True
        while process_queue or initial_call:
            initial_call = False
            try:
                message = gui_queue.get_nowait()
                process_queue = True
                current_time = time.time()
                if message is None:
                    print("GUI received None signal (thread finished).")
                    self.threads_completed += 1 # Increment counter
                    print(f"Threads completed count: {self.threads_completed}/{self.expected_threads}")
                    # Only stop gracefully when ALL threads are done
                    if self.threads_completed >= self.expected_threads and self.is_running:
                        print(f"All {self.expected_threads} threads signaled completion.")
                        self.stop_translation(graceful_stop=True)
                        if self.root.winfo_exists(): self.status_var.set("Processing finished. Ready.")
                    continue # <<<< Use continue instead of return
                elif isinstance(message, str):
                    # (Keep the message handling logic from your file)
                    if message.startswith("[EN"):
                        if self.initial_status_shown and "CRITICAL" not in self.status_var.get():
                            self.status_var.set("Translating...")
                            self.initial_status_shown = False
                        self.message_deque.append((current_time, message))
                        while len(self.message_deque) > MAX_HISTORY_MESSAGES: self.message_deque.popleft()
                        self.update_output_display()
                    elif "CRITICAL ERROR" in message:
                        status_part = message.split("CRITICAL ERROR:", 1)[-1].strip() if "CRITICAL ERROR:" in message else "Critical Error Occurred"
                        self.status_var.set(f"CRITICAL ERROR: {status_part}")
                        self.add_output_direct(f"*** {message} ***", "error")
                        self.initial_status_shown = False
                        if self.is_running: self.stop_translation(error=True)
                        # gui_queue.task_done() # Not needed
                        return
                    elif message.startswith("__INIT_STATUS__"):
                        status_part = message.split("__INIT_STATUS__", 1)[-1]
                        self.status_var.set(status_part)
                        self.initial_status_shown = True
                    elif message.startswith("[ERROR") or message.startswith("[Warning"):
                        if "CRITICAL" not in self.status_var.get() and self.is_running:
                            self.status_var.set("Error/Warning occurred.")
                        self.add_output_direct(message, "error")
                        self.initial_status_shown = False
                    elif message.startswith("["):
                        status_part = message.split("] ", 1)[-1]
                        significant_init = "initialized" in status_part or "Capturing" in status_part
                        if significant_init and "CRITICAL" not in self.status_var.get():
                            self.status_var.set(status_part)
                            self.initial_status_shown = True
                        self.add_output_direct(message, "info")
                    else:
                        self.add_output_direct(message, "info")
                # gui_queue.task_done() # Not needed

            except queue.Empty: process_queue = False
            except tk.TclError as e:
                print(f"Error: GUI TclError in check_gui_queue: {e}")
                process_queue = False
                if self.is_running: self.stop_translation(error=True)
                return
            except Exception as e:
                print(f"Error processing GUI queue: {e}")
                self.add_output_direct(f"[ERROR] GUI Error: {e}", "error")
                self.initial_status_shown = False
                process_queue = False
                if self.is_running: self.stop_translation(error=True)
                return

        # End of while loop
        if self.root.winfo_exists() and (self.is_running or self.threads_completed < self.expected_threads):
             self.root.after(100, self.check_gui_queue)

    def start_translation(self):
        """ Starts the translation process in separate threads """
        self.threads_completed = 0 # <<<< Reset counter
        if self.is_running: messagebox.showwarning("Already Running", "Translation is already in progress."); return

        selected_device_str = self.device_combobox.get()
        selected_source_lang_name = self.source_lang_var.get()
        selected_model_size = self.model_size_var.get()
        if not selected_device_str or self.device_map.get(selected_device_str) is None: messagebox.showerror("Input Error", "Please select a valid Input Device."); return
        device_index = self.device_map.get(selected_device_str)
        selected_source_lang_code = SUPPORTED_SOURCE_LANGUAGES.get(selected_source_lang_name)
        if selected_source_lang_name not in SUPPORTED_SOURCE_LANGUAGES: messagebox.showerror("Input Error", f"Invalid source language name: {selected_source_lang_name}"); return
        if not selected_model_size or selected_model_size not in SUPPORTED_MODEL_SIZES: messagebox.showerror("Input Error", "Please select a valid model size."); return

        try:
            self.output_text.config(state=tk.NORMAL); self.output_text.delete('1.0', tk.END); self.output_text.config(state=tk.DISABLED)
            self.message_deque.clear()
            self.status_var.set(f"Starting... Model: {selected_model_size}, Src Lang: {selected_source_lang_name}...")
            self.add_output_direct("--- Starting translation ---", "info")
            self.is_running = True; stop_event.clear(); self.expected_threads = 2 # Capture + TranscribeTranslate
            self.start_button.config(state=tk.DISABLED); self.stop_button.config(state=tk.NORMAL)
            self.device_combobox.config(state=tk.DISABLED); self.source_lang_combobox.config(state=tk.DISABLED)
            self.model_size_combobox.config(state=tk.DISABLED); self.font_size_spinbox.config(state=tk.DISABLED)
            self.active_threads = []; self.initial_status_shown = False
        except tk.TclError as e: print(f"Error starting (TclError): {e}"); self.is_running = False; return
        except Exception as e: print(f"Error configuring UI for start: {e}"); self.is_running = False; return

        print("Clearing queues before starting threads..."); self._clear_queues(clear_gui=True); print("Queues cleared.")
        print("Starting background threads...")
        try:
            capture = threading.Thread(target=capture_audio_thread_gui, args=(device_index, audio_queue, gui_queue, stop_event), name="AudioCapture", daemon=True)
            transcriber = threading.Thread(
                target=transcribe_translate_thread_gui, args=(audio_queue, gui_queue, stop_event, selected_model_size, selected_source_lang_code), name="TranscribeTranslate", daemon=True)
            self.active_threads.extend([capture, transcriber])
            capture.start(); transcriber.start()
            self.root.after(100, self.check_gui_queue); print("Threads started, polling GUI queue.")
        except Exception as e:
             print(f"CRITICAL ERROR: Failed to start threads: {e}"); self.status_var.set(f"ERROR: Failed to start threads: {e}")
             self.add_output_direct(f"*** CRITICAL ERROR starting threads: {e} ***", "error"); self.stop_translation(error=True)

    def stop_translation(self, error=False, graceful_stop=False):
        """ Stops the translation process and resets the GUI state. """
        if not self.is_running:
            print("Stop called but already not running. Ensuring UI state.")
            try:
                if self.root.winfo_exists():
                    self.start_button.config(state=self.start_button_state); self.stop_button.config(state=tk.DISABLED)
                    self.device_combobox.config(state="readonly"); self.source_lang_combobox.config(state="readonly")
                    self.model_size_combobox.config(state="readonly"); self.font_size_spinbox.config(state="readonly")
                    current_status = self.status_var.get()
                    if "error" not in current_status.lower() and "stopped" not in current_status.lower() and "finished" not in current_status.lower(): self.status_var.set("Ready.")
            except tk.TclError: pass
            except Exception as e: print(f"Error resetting UI state in stop_translation: {e}")
            return

        call_source = "graceful stop" if graceful_stop else ("error stop" if error else "manual stop")
        print(f"Stop signal initiated ({call_source}).")
        status = "Processing stopped by user."; log_status = status
        if error:
             current_status = self.status_var.get(); status = current_status if "CRITICAL ERROR" in current_status else "Stopped due to CRITICAL ERROR."
             log_status = "Processing stopped due to CRITICAL ERROR."
        elif graceful_stop: status = "Processing finished."; log_status = status
        stop_event.set(); print("Stop event set.")
        self.is_running = False; print("is_running set to False.")
        try:
            if self.root.winfo_exists():
                print("Updating GUI elements to stopped state..."); self.status_var.set(status)
                if not graceful_stop: self.add_output_direct(f"--- {log_status} ---", "error" if error else "info")
                self.start_button.config(state=self.start_button_state); self.stop_button.config(state=tk.DISABLED)
                self.device_combobox.config(state="readonly"); self.source_lang_combobox.config(state="readonly")
                self.model_size_combobox.config(state="readonly"); self.font_size_spinbox.config(state="readonly")
                self.alpha_slider.config(state=tk.NORMAL) # <<<< Re-enable alpha slider
                print("GUI elements updated.")
            else: print("GUI window closed during stop sequence.")
        except tk.TclError as e: print(f"Info: TclError updating GUI during stop: {e}")
        except Exception as e: print(f"Error updating GUI elements in stop_translation: {e}")
        self._clear_queues(clear_gui=True)
        self.active_threads = []; self.initial_status_shown = False
        self.threads_completed = 0 # Reset completion counter here too
        print("Stop translation method finished.")

    def _clear_queues(self, clear_gui=False):
        """Clears audio queue and optionally the GUI queue."""
        print("Clearing queues (best effort)...")
        queues_to_clear = [(audio_queue, "audio")]
        if clear_gui: queues_to_clear.append((gui_queue, "gui"))
        cleared_counts = {"audio": 0, "gui": 0}
        for q, name in queues_to_clear:
            cleared_count = 0
            while not q.empty():
                try: item = q.get_nowait(); cleared_count += 1
                except queue.Empty: break
                except Exception as e: print(f"Error clearing {name}_q item: {e}"); break
            cleared_counts[name] = cleared_count
            print(f"Cleared approx {cleared_counts[name]} items from {name}_queue.")

    def on_closing(self):
        print("Close window requested ('X' button).")
        if self.is_running:
            if messagebox.askokcancel("Quit", "Translation is running. Stop and quit?"):
                print("User confirmed quit while running. Stopping translation...")
                self.stop_translation(error=False, graceful_stop=False)
                print("Scheduling root destroy."); self.root.after(300, self._destroy_root)
            else: print("Quit cancelled by user."); return
        else: print("Not running, destroying root window immediately."); self._destroy_root()

    def _destroy_root(self):
       print("Attempting to destroy root window..."); stop_event.set()
       if self.root and self.root.winfo_exists():
           try: self.root.destroy(); print("Root window destroyed successfully.")
           except tk.TclError as e: print(f"TclError during root destroy (window might already be gone): {e}")
           except Exception as e: print(f"Unexpected error during root destroy: {e}")
       else: print("Root window does not exist or already destroyed.")

# --- End of TranslatorApp Class ---

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting application...")
    # --- Pre-checks (Optional but recommended) ---
    try:
        p_test = pyaudio.PyAudio()
        p_test.terminate()
        print("PyAudio initialized and terminated successfully.")
    except Exception as e:
        # Use Tkinter for error message if possible
        try:
             root_err = tk.Tk(); root_err.withdraw()
             messagebox.showerror("PyAudio Error", f"PyAudio initialization failed: {e}\n\nPlease ensure PortAudio is installed and audio drivers are working.\nThe application cannot function without audio input.")
             root_err.destroy()
        except Exception: pass # Fallback to print
        print(f"CRITICAL PYAUDIO ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Start the main application ---
    root = None # Initialize root to None
    try:
        root = tk.Tk()
        root.withdraw() # Hide until ready
        print("Tk root created.")
        app = TranslatorApp(root)
        print("TranslatorApp initialized.")
        root.deiconify() # Show the window
        print("Entering Tk mainloop...")
        root.mainloop()
        # --- Mainloop finished ---
        print("Exited Tk mainloop normally.")
    except tk.TclError as e:
        print(f"CRITICAL ERROR: Could not initialize Tkinter GUI: {e}", file=sys.stderr)
        # Attempt to show a final message box if possible
        try:
             root_final_err = tk.Tk(); root_final_err.withdraw()
             messagebox.showerror("GUI Error", f"A critical error occurred during GUI initialization:\n{e}")
             root_final_err.destroy()
        except Exception: pass # Fallback
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred during GUI setup or mainloop: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Attempt to show a final message box
        try:
             root_final_err = tk.Tk(); root_final_err.withdraw()
             messagebox.showerror("Application Error", f"An unexpected critical error occurred:\n{e}")
             root_final_err.destroy()
        except Exception: pass # Fallback
        sys.exit(1)
    finally:
        # Ensure stop event is set on any exit path
        stop_event.set()
        print("Application finished.")
