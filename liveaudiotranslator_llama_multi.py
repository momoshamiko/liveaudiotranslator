# -*- coding: utf-8 -*-
import queue
import threading
import sys
import time
from collections import deque
import numpy as np
import torch
from faster_whisper import WhisperModel
import ollama # Using Ollama
import pyaudio # Using PyAudio for capture
import re # For ignore patterns

import tkinter as tk
from tkinter import ttk, messagebox, font # Removed scrolledtext

# --- Configuration ---

EMPTY_TRANSLATIONS = {"nothing", "none", "null", "silence", ""} # Add variations if needed (lowercase)

# --- MODIFICATION: Added Source Language Config ---
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
    # Add more languages as needed
}
DEFAULT_SOURCE_LANG = "Auto Detect"
# --- END MODIFICATION ---

# Define supported target languages (Display Name: ISO 639-1 Code)
SUPPORTED_TARGET_LANGUAGES = {
    "English": "en",
    "Indonesian": "id",
    "Russian": "ru",
    "Korean": "ko",
    "Japanese": "ja",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh",
}
DEFAULT_TARGET_LANG = "English" # Default selection in the dropdown

# Whisper settings (Model size is fixed here)
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "float32"

# Ollama settings
OLLAMA_MODEL = 'llama3' # Ensure this model is pulled in Ollama

# Ignore Patterns (using Regex)
IGNORE_PATTERNS = [
    re.compile(r"(?:thank\s*(?:you|s)\s*(?:so|very|a\s*lot)?\s*(?:much)?\s*)?(?:for\s+(?:watching|viewing|your\s+viewing)).*?(?:in\s+this\s+video)?", re.IGNORECASE),
    re.compile(r"see\s+you(?:\s+all|\s+again)?\s+(?:next\s+(?:time|video)|later|in\s+the\s+next\s+video)", re.IGNORECASE),
    re.compile(r"subscribe\s+to\s+(?:my|the)\s+channel", re.IGNORECASE),
    re.compile(r"thank(?:s| you).*?(?:watch|view)", re.IGNORECASE),
    re.compile(r"see you.*?(?:next|later)", re.IGNORECASE),
]

# Audio settings
AUDIO_SAMPLERATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_BUFFER_SECONDS = 3
PYAUDIO_FRAMES_PER_BUFFER = 1024

# --- Appearance & History Settings ---
MAX_HISTORY_MESSAGES = 10 #(Adjust as needed)
WINDOW_ALPHA = 0.60

# --- Color Constants & Font Size ---
DEFAULT_FONT_SIZE = 14
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 24
DARK_BG = '#2b2b2b'
LIGHT_FG = '#ffffff'
ENTRY_BG = '#3c3f41'
BUTTON_BG = '#555555'
BUTTON_ACTIVE_BG = '#666666'
TEXT_CURSOR = '#ffffff'
INFO_FG = '#6cace4'
ERROR_FG = '#e46c6c'
SCROLLBAR_BG = '#454545' # Color for the scrollbar itself
SCROLLBAR_TROUGH = '#333333'# Color for the scrollbar channel

# --- Global Queues & Events ---
audio_queue = queue.Queue()
transcribed_queue = queue.Queue()
gui_queue = queue.Queue()
stop_event = threading.Event()

# --- Helper Function ---
def get_audio_devices():
    # (Function remains the same)
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
    except Exception as e: print(f"Error getting audio devices: {e}")
    finally: p.terminate()
    return devices

# --- Thread Functions ---

def capture_audio_thread_gui(device_index, audio_q, gui_q, stop_event_flag):
    # (Function remains the same)
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
                # Check stop event before reading
                if stop_event_flag.is_set(): break
                data = stream.read(PYAUDIO_FRAMES_PER_BUFFER, exception_on_overflow=False)
                audio_buffer.extend(data)
                while len(audio_buffer) >= target_buffer_size:
                    # Check stop event before putting data
                    if stop_event_flag.is_set(): break
                    chunk_to_process = audio_buffer[:target_buffer_size]
                    audio_q.put(bytes(chunk_to_process))
                    audio_buffer = audio_buffer[target_buffer_size:]
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed or str(e) == "Input overflowed": gui_q.put(f"[{thread_name}] Warning: PyAudio Input Overflowed.")
                else: gui_q.put(f"[{thread_name}] ERROR: PyAudio read error: {e}"); time.sleep(0.1)
            except Exception as e: gui_q.put(f"[{thread_name}] ERROR: Unexpected error during stream read: {e}"); time.sleep(0.1)
        # End of while loop
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
        # Signal next thread (transcriber)
        audio_q.put(None)
        gui_q.put(f"[{thread_name}] Audio capture thread finished.")


def transcribe_thread_gui(audio_q, transcribed_q, gui_q, stop_event_flag, source_language_code):
    # (Function remains the same)
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Initializing Whisper model ({WHISPER_MODEL_SIZE})...")
    model = None
    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        lang_msg = f"(Lang: {source_language_code})" if source_language_code else "(Lang: Auto Detect)"
        gui_q.put(f"__INIT_STATUS__Whisper model '{WHISPER_MODEL_SIZE}' initialized {lang_msg}. Ready.")

        while not stop_event_flag.is_set():
            try:
                audio_data_bytes = audio_q.get(timeout=0.5)
                if audio_data_bytes is None:
                    gui_q.put(f"[{thread_name}] Received None from audio queue, stopping.")
                    break # Exit loop
            except queue.Empty: continue # Timeout, check stop_event

            if stop_event_flag.is_set():
                gui_q.put(f"[{thread_name}] Stop event detected after getting audio data/timeout.")
                break # Exit loop

            try:
                audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                if audio_np.size > 0:
                    segments, info = model.transcribe(
                        audio_np, language=source_language_code, beam_size=5, vad_filter=False
                    )
                    full_text = " ".join(segment.text for segment in segments).strip()
                    detected_language = info.language
                    if full_text and detected_language:
                        transcribed_q.put((detected_language, full_text))
                        if source_language_code is None:
                             gui_q.put(f"[{thread_name}] Auto-detected language: {detected_language} (Prob: {info.language_probability:.2f})")
                    elif full_text and not detected_language:
                         gui_q.put(f"[{thread_name}] Warning: Transcription succeeded but no language detected?")
            except Exception as e:
                gui_q.put(f"[{thread_name}] ERROR during transcription: {e}")

        if stop_event_flag.is_set(): gui_q.put(f"[{thread_name}] Stop event was set, exiting transcription loop.")
    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Failed to load/run Whisper model '{WHISPER_MODEL_SIZE}': {e}")
    finally:
        # Signal next thread (translator)
        transcribed_q.put(None)
        if model is not None:
            try:
                gui_q.put(f"[{thread_name}] Releasing Whisper model resources...")
                del model;
                if WHISPER_DEVICE == 'cuda': torch.cuda.empty_cache()
                gui_q.put(f"[{thread_name}] Whisper model resources released.")
            except Exception as e: gui_q.put(f"[{thread_name}] Warning: Error releasing model resources: {e}")
        gui_q.put(f"[{thread_name}] Transcription thread finished.")


def translate_thread_gui(transcribed_q, gui_q, target_lang_code, stop_event_flag):
    # (Function remains the same)
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Initializing Ollama client (Model: {OLLAMA_MODEL}, Target: {target_lang_code})...")
    try:
        ollama.list(); gui_q.put(f"[{thread_name}] Ollama connection successful. Waiting for text...")
    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Ollama connection failed: {e}")
        gui_q.put(None); return # Signal GUI and exit

    while not stop_event_flag.is_set():
        try:
            data = transcribed_q.get(timeout=0.5)
            if data is None:
                gui_q.put(f"[{thread_name}] Received None from transcribed queue, stopping.")
                break # Exit loop
        except queue.Empty: continue # Timeout, check stop_event

        if stop_event_flag.is_set():
            gui_q.put(f"[{thread_name}] Stop event detected after getting transcribed data/timeout.")
            break # Exit loop

        source_lang_code, text_to_translate = data
        if not target_lang_code: gui_q.put(f"[{thread_name}] Error: No target language selected."); continue

        import string
        if not text_to_translate or len(text_to_translate.strip(string.punctuation + string.whitespace)) < 1:
           continue

        try:
            if text_to_translate:
                start_time = time.time()
                context_info = "Context: The text is dialogue potentially from a live stream or conversation."
                instruction = f"Translate the following Source Text ({source_lang_code}) strictly into {target_lang_code}. IMPORTANT: Your entire response must contain ONLY the translated text in {target_lang_code} and nothing else. No extra words, no explanations, no labels, no source text. Do not include conversational closings like 'thank you for watching' or 'see you next time'. If the Source Text appears to be silence, background noise, punctuation-only, or contains no translatable content, respond with an empty string ONLY."
                prompt = f"{context_info}\n\n{instruction}\n\nSource Text ({source_lang_code}): {text_to_translate}\n\n{target_lang_code} Translation:"
                response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, stream=False, options={"temperature": 0.3})
                translation = response['response'].strip()
                latency = time.time() - start_time
                should_ignore = False
                if translation:
                    lines = translation.split('\n')
                    filtered_lines = [line for line in lines if not (
                        ':' in line and ('source text' in line.lower() or 'translation' in line.lower() or source_lang_code in line.lower())
                        or line.lower().startswith(f"{target_lang_code} translation:")
                        or line.lower().startswith(f"translation ({target_lang_code}):")
                        or line.lower().startswith(f"{source_lang_code}:")
                        or line.lower().startswith(f"source text ({source_lang_code}):")
                        )]
                    translation = " ".join(filtered_lines).strip().strip('"`')
                    if translation.strip().lower() in EMPTY_TRANSLATIONS or not translation:
                        should_ignore = True
                    else:
                        normalized_translation = translation.strip()
                        for pattern in IGNORE_PATTERNS:
                            if pattern.search(normalized_translation):
                                should_ignore = True; print(f"Ignoring phrase matching pattern '{pattern.pattern}': '{translation}'"); break
                if translation and not should_ignore:
                    gui_q.put(f"[{target_lang_code.upper()} ({latency:.2f}s)] {translation}")
                elif not translation and len(text_to_translate.strip()) > 2 and not should_ignore:
                     gui_q.put(f"[{thread_name}] Warning: Translation empty after filtering. Original Response: '{response['response']}'")
                elif len(text_to_translate.strip()) > 2 and not translation:
                    gui_q.put(f"[{thread_name}] Warning: Empty translation response from LLM for input: '{text_to_translate}'")
        except Exception as e:
            gui_q.put(f"[{thread_name}] ERROR during translation API call: {e}")

    # --- End of while loop ---
    if stop_event_flag.is_set(): gui_q.put(f"[{thread_name}] Stop event was set, exiting translation loop.")
    gui_q.put(f"[{thread_name}] Translation thread finished.")
    # Signal GUI that this *final* processing thread is done
    gui_q.put(None)


# --- Tkinter GUI Application ---

# --- Tkinter GUI Application ---

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multilingual Live Translator (Ollama)")
        self.root.geometry("920x400") # Adjusted geometry slightly for new slider
        self.root.config(bg=DARK_BG)
        style = ttk.Style(self.root)
        try:
            if 'clam' in style.theme_names(): style.theme_use('clam')
            else: print("Warning: 'clam' theme not available, using default.")
        except tk.TclError: print("Warning: Error setting 'clam' theme, using default.")

        # --- Style Definitions (including Scrollbar & Scale) ---
        style.configure('.', background=DARK_BG, foreground=LIGHT_FG, font=('Segoe UI', 9))
        style.configure('TFrame', background=DARK_BG)
        style.configure('TLabel', background=DARK_BG, foreground=LIGHT_FG)
        style.configure('TButton', background=BUTTON_BG, foreground=LIGHT_FG, padding=5, borderwidth=0, relief=tk.FLAT)
        style.map('TButton', background=[('active', BUTTON_ACTIVE_BG), ('disabled', '#444444')], foreground=[('disabled', '#999999')])
        style.configure('TCombobox', fieldbackground=ENTRY_BG, background=BUTTON_BG, foreground=LIGHT_FG, arrowcolor=LIGHT_FG, selectbackground=ENTRY_BG, selectforeground=LIGHT_FG, insertcolor=TEXT_CURSOR, borderwidth=0)
        style.map('TCombobox', fieldbackground=[('readonly', ENTRY_BG)], selectbackground=[('!focus', ENTRY_BG)], background=[('readonly', BUTTON_BG)], foreground=[('disabled', '#999999')])
        root.option_add('*TCombobox*Listbox.background', ENTRY_BG); root.option_add('*TCombobox*Listbox.foreground', LIGHT_FG)
        root.option_add('*TCombobox*Listbox.selectBackground', BUTTON_ACTIVE_BG); root.option_add('*TCombobox*Listbox.selectForeground', LIGHT_FG)
        root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 9))
        style.configure('Vertical.TScrollbar', gripcount=0, background=SCROLLBAR_BG, troughcolor=SCROLLBAR_TROUGH, bordercolor=DARK_BG, arrowcolor=LIGHT_FG, relief=tk.FLAT)
        style.map('Vertical.TScrollbar', background=[('active', BUTTON_ACTIVE_BG)], arrowcolor=[('pressed', '#cccccc'), ('disabled', '#666666')])
        # Basic Scale styling
        style.configure('Horizontal.TScale', background=DARK_BG, troughcolor=ENTRY_BG)
        style.map('Horizontal.TScale', background=[('active', BUTTON_ACTIVE_BG)], troughcolor=[('active', ENTRY_BG)])


        # --- Window Attributes ---
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', WINDOW_ALPHA) # Set initial alpha

        # --- App State Variables ---
        self.is_running = False
        self.active_threads = []
        self.message_deque = deque()
        self.initial_status_shown = False
        self.expected_threads = 3 # Capture, Transcribe, Translate
        self.threads_completed = 0 # <<<< ADDED for thread tracking

        # =============================================
        # --- GUI Layout ---
        # =============================================
        control_frame = ttk.Frame(root, padding="10"); control_frame.pack(fill=tk.X)
        # Device
        ttk.Label(control_frame, text="Input Device:").pack(side=tk.LEFT, padx=(0, 5))
        self.devices = get_audio_devices()
        if not self.devices: self.device_map = {"No Input Devices Found": None}; messagebox.showerror("Audio Device Error", "No audio input devices found.")
        else: self.device_map = {f"{d['index']}: {d['name']} (Rate: {int(d.get('rate', 0))})": d['index'] for d in self.devices}
        self.device_combobox = ttk.Combobox(control_frame, values=list(self.device_map.keys()), state="readonly", width=30)
        if self.device_map and list(self.device_map.values())[0] is not None: self.device_combobox.current(0)
        self.device_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))
        self.start_button_state = tk.NORMAL if any(idx is not None for idx in self.device_map.values()) else tk.DISABLED
        # Source Lang
        ttk.Label(control_frame, text="From:").pack(side=tk.LEFT, padx=(0, 2))
        self.source_lang_var = tk.StringVar()
        source_lang_options = list(SUPPORTED_SOURCE_LANGUAGES.keys())
        self.source_lang_combobox = ttk.Combobox(control_frame, textvariable=self.source_lang_var, values=source_lang_options, state="readonly", width=10)
        if DEFAULT_SOURCE_LANG in source_lang_options: self.source_lang_var.set(DEFAULT_SOURCE_LANG)
        elif source_lang_options: self.source_lang_var.set(source_lang_options[0])
        self.source_lang_combobox.pack(side=tk.LEFT, padx=(0, 10))
        # Target Lang
        ttk.Label(control_frame, text="To:").pack(side=tk.LEFT, padx=(0, 2))
        self.target_lang_var = tk.StringVar()
        target_lang_options = list(SUPPORTED_TARGET_LANGUAGES.keys())
        self.target_lang_combobox = ttk.Combobox(control_frame, textvariable=self.target_lang_var, values=target_lang_options, state="readonly", width=15)
        if DEFAULT_TARGET_LANG in target_lang_options: self.target_lang_var.set(DEFAULT_TARGET_LANG)
        elif target_lang_options: self.target_lang_var.set(target_lang_options[0])
        self.target_lang_combobox.pack(side=tk.LEFT, padx=(0, 10))
        # Font Size
        ttk.Label(control_frame, text="Size:").pack(side=tk.LEFT, padx=(0, 2))
        self.font_size_var = tk.IntVar(value=DEFAULT_FONT_SIZE)
        self.font_size_spinbox = tk.Spinbox(
            control_frame, from_=MIN_FONT_SIZE, to=MAX_FONT_SIZE, textvariable=self.font_size_var, width=3, command=self.update_font_size,
            state="readonly", bg=ENTRY_BG, fg=LIGHT_FG, buttonbackground=BUTTON_BG, relief=tk.FLAT, readonlybackground=ENTRY_BG,
            highlightthickness=1, highlightbackground=DARK_BG, highlightcolor=BUTTON_ACTIVE_BG, insertbackground=TEXT_CURSOR, buttoncursor="arrow",
            disabledbackground=ENTRY_BG, disabledforeground="#999999",
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

        # Buttons
        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_translation, width=6, style='TButton', state=self.start_button_state)
        self.start_button.pack(side=tk.LEFT, padx=(0, 2))
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_translation, state=tk.DISABLED, width=6, style='TButton')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 0))

        # Output Area
        output_frame = ttk.Frame(root, padding=(10, 0, 10, 10)); output_frame.pack(fill=tk.BOTH, expand=True)
        self.scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, style='Vertical.TScrollbar')
        self.output_text = tk.Text(
            output_frame, wrap=tk.WORD, state=tk.DISABLED, height=10, font=('Segoe UI', self.font_size_var.get()),
            bg=DARK_BG, fg=LIGHT_FG, insertbackground=TEXT_CURSOR, selectbackground=BUTTON_BG, selectforeground=LIGHT_FG,
            relief=tk.FLAT, highlightthickness=0, yscrollcommand=self.scrollbar.set
        )
        self.scrollbar.config(command=self.output_text.yview); self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.output_text.tag_config("error", foreground=ERROR_FG); self.output_text.tag_config("info", foreground=INFO_FG)

        # Status Bar
        self.status_var = tk.StringVar()
        initial_status = "Ready." if self.start_button_state == tk.NORMAL else "No input devices found."
        self.status_var.set(initial_status)
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.FLAT, anchor=tk.W, padding=5, style='TLabel')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Initialize font object cache only once
        self._output_font = font.Font(font=self.output_text['font'])


    # --- App Methods ---

    # --- ADDED update_alpha method (Correct Indentation) ---
    def update_alpha(self, value=None): # value is passed by the Scale command
        """Updates the window transparency based on the slider value."""
        if not self.root.winfo_exists():
             return
        try:
            # Read value from the DoubleVar linked to the slider
            new_alpha = self.alpha_var.get()
            # Apply the new alpha value to the root window
            self.root.attributes('-alpha', new_alpha)
        except tk.TclError as e:
            # Handle potential errors if the window is closing etc.
            print(f"Info: TclError updating alpha (window might be closing): {e}")
        except Exception as e:
            print(f"Error updating alpha: {e}")
    # --- END update_alpha method ---

    def update_font_size(self):
        if not self.root.winfo_exists(): return
        new_size = self.font_size_var.get()
        try:
            # Use the stored font object
            self.output_text.config(font=(self._output_font.actual()['family'], new_size))
            # Update the cached font object if needed (optional, depends if family changes)
            # self._output_font = font.Font(font=self.output_text['font'])
        except Exception as e: print(f"Error setting font: {e}")

    def add_output_direct(self, text, tag=None):
        if not self.root.winfo_exists(): return
        try:
            is_scrolled_to_bottom = self.output_text.yview()[1] >= 0.99
            self.output_text.config(state=tk.NORMAL)
            if tag: self.output_text.insert(tk.END, text + "\n", tag)
            else: self.output_text.insert(tk.END, text + "\n")
            if is_scrolled_to_bottom: self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        except Exception as e: print(f"Error in add_output_direct: {e}")


    def update_output_display(self):
        if not self.root.winfo_exists(): return
        try:
            current_scroll = self.output_text.yview()
            scroll_pos_before_update = current_scroll[0] if current_scroll[1] < 0.99 else None
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete('1.0', tk.END)
            for _, msg in self.message_deque:
                tag_to_use = None; is_translation = False
                if msg.startswith("[ERROR") or "CRITICAL ERROR" in msg: tag_to_use = "error"
                elif msg.startswith("[INFO") or msg.startswith("[Audio") or msg.startswith("[Transcribe") or msg.startswith("[Translate") or msg.startswith("[Warning") or msg.startswith("---"): tag_to_use = "info"
                else:
                    # Check if message starts with any supported target language code tag like '[EN (...)]'
                    for code in SUPPORTED_TARGET_LANGUAGES.values():
                        if msg.startswith(f"[{code.upper()}"):
                            is_translation = True; break
                # Only apply special tags for non-translation messages
                if not is_translation and tag_to_use:
                    self.output_text.insert(tk.END, msg + "\n", tag_to_use)
                else: # Use default styling for translations or untagged messages
                    self.output_text.insert(tk.END, msg + "\n")

            if scroll_pos_before_update is not None: self.output_text.yview_moveto(scroll_pos_before_update)
            else: self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        except Exception as e: print(f"Error in update_output_display: {e}")


    def check_gui_queue(self):
        """ Checks the GUI queue for messages from threads and updates the UI. """
        # --- UPDATED logic based on previous fix ---
        if not self.is_running and self.threads_completed < self.expected_threads:
             pass
        elif not self.is_running:
             return

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
                    # (Keep the existing message handling logic from your file here)
                    is_translation = False
                    for code in SUPPORTED_TARGET_LANGUAGES.values():
                        if message.startswith(f"[{code.upper()}"): is_translation = True; break

                    if is_translation:
                        if self.initial_status_shown and "CRITICAL" not in self.status_var.get():
                            self.status_var.set("Translating...")
                            self.initial_status_shown = False
                        self.message_deque.append((current_time, message))
                        while len(self.message_deque) > MAX_HISTORY_MESSAGES: self.message_deque.popleft()
                        self.update_output_display()
                    elif "CRITICAL ERROR" in message:
                        status_part = message.split("CRITICAL ERROR:", 1)[-1].strip() if "CRITICAL ERROR:" in message else "Critical Error"
                        self.status_var.set(f"CRITICAL ERROR: {status_part}")
                        self.add_output_direct(f"*** {message} ***", "error")
                        self.initial_status_shown = False;
                        if self.is_running: self.stop_translation(error=True)
                        return # Stop processing queue on critical error
                    elif message.startswith("__INIT_STATUS__"):
                        status_part = message.split("__INIT_STATUS__", 1)[-1]
                        self.status_var.set(status_part); self.initial_status_shown = True
                    elif message.startswith("[ERROR") or message.startswith("[Warning"):
                        if "CRITICAL" not in self.status_var.get() and self.is_running: self.status_var.set("Error/Warning occurred.")
                        self.add_output_direct(message, "error"); self.initial_status_shown = False
                    elif message.startswith("["): # Info messages
                        status_part = message.split("] ", 1)[-1]
                        significant_init = "initialized" in status_part or "Capturing" in status_part or "connection successful" in status_part
                        if significant_init and "CRITICAL" not in self.status_var.get():
                            self.status_var.set(status_part); self.initial_status_shown = True
                        self.add_output_direct(message, "info")
                    else: # Other messages
                        self.add_output_direct(message, "info")

            except queue.Empty:
                process_queue = False # No more messages

            except tk.TclError as e:
                print(f"Error: GUI TclError in check_gui_queue: {e}")
                process_queue = False
                if self.is_running: self.stop_translation(error=True)
                return # Stop checking on TclError

            except Exception as e:
                print(f"Error processing GUI queue: {e}")
                self.add_output_direct(f"[ERROR] GUI Error: {e}", "error")
                self.initial_status_shown = False
                process_queue = False
                if self.is_running: self.stop_translation(error=True)
                return # Stop checking on other errors

        # End of while loop
        if self.root.winfo_exists() and (self.is_running or self.threads_completed < self.expected_threads):
             self.root.after(100, self.check_gui_queue)


    def start_translation(self):
        """ Starts the translation process (PyAudio -> Whisper -> Ollama) """
        if self.is_running: messagebox.showwarning("Already Running", "Translation is already in progress."); return

        # --- Reset thread counter --- <<<< ADDED
        self.threads_completed = 0

        selected_device_str = self.device_combobox.get()
        selected_source_lang_name = self.source_lang_var.get()
        selected_target_lang_name = self.target_lang_var.get()
        if not selected_device_str or self.device_map.get(selected_device_str) is None: messagebox.showerror("Input Error", "Please select a valid Input Device."); return
        device_index = self.device_map.get(selected_device_str)
        selected_source_lang_code = SUPPORTED_SOURCE_LANGUAGES.get(selected_source_lang_name)
        if selected_source_lang_name not in SUPPORTED_SOURCE_LANGUAGES: messagebox.showerror("Input Error", "Invalid source language selected."); return
        selected_target_lang_code = SUPPORTED_TARGET_LANGUAGES.get(selected_target_lang_name)
        if not selected_target_lang_code: messagebox.showerror("Input Error", "Please select a target language."); return

        try:
            self.output_text.config(state=tk.NORMAL); self.output_text.delete('1.0', tk.END); self.output_text.config(state=tk.DISABLED)
            self.message_deque.clear()
            self.status_var.set(f"Starting... Src: {selected_source_lang_name}, Target: {selected_target_lang_name}")
            self.add_output_direct("--- Starting translation ---", "info")
            self.is_running = True; stop_event.clear(); self.expected_threads = 3 # Set expected threads
            self.start_button.config(state=tk.DISABLED); self.stop_button.config(state=tk.NORMAL)
            self.device_combobox.config(state=tk.DISABLED); self.source_lang_combobox.config(state=tk.DISABLED)
            self.target_lang_combobox.config(state=tk.DISABLED); self.font_size_spinbox.config(state=tk.DISABLED)
            self.active_threads = []; self.initial_status_shown = False
        except Exception as e: print(f"Error configuring UI for start: {e}"); self.is_running = False; return

        print("Clearing queues before starting threads...")
        self._clear_queues(clear_gui=True) # Clear audio, transcribed, and GUI queues
        print("Queues cleared.")

        print("Starting background threads...")
        try:
            capture = threading.Thread(target=capture_audio_thread_gui, args=(device_index, audio_queue, gui_queue, stop_event), name="AudioCapture", daemon=True)
            transcriber = threading.Thread(target=transcribe_thread_gui, args=(audio_queue, transcribed_queue, gui_queue, stop_event, selected_source_lang_code), name="Transcribe", daemon=True)
            translator = threading.Thread(target=translate_thread_gui, args=(transcribed_queue, gui_queue, selected_target_lang_code, stop_event), name="Translate", daemon=True)
            self.active_threads.extend([capture, transcriber, translator])
            capture.start(); transcriber.start(); translator.start()
            self.root.after(100, self.check_gui_queue) # Start polling GUI queue
            print("Threads started, polling GUI queue.")
        except Exception as e:
             print(f"CRITICAL ERROR: Failed to start threads: {e}")
             self.status_var.set(f"ERROR: Failed to start threads: {e}")
             self.add_output_direct(f"*** CRITICAL ERROR starting threads: {e} ***", "error")
             self.stop_translation(error=True)


    def stop_translation(self, error=False, graceful_stop=False):
        """ Stops the translation process and resets the GUI state. """
        call_source = "graceful stop" if graceful_stop else ("error stop" if error else "manual stop by BUTTON")
        print(f"\n--- ENTERING stop_translation (source: {call_source}) ---")

        # --- Added check to prevent multiple stop calls ---
        if not self.is_running and not stop_event.is_set():
             print("Stop called but already not running and stop event not set. Ensuring UI state.")
             try: # Reset UI elements to their ready state
                 if self.root.winfo_exists():
                     self.start_button.config(state=self.start_button_state); self.stop_button.config(state=tk.DISABLED)
                     self.device_combobox.config(state="readonly"); self.source_lang_combobox.config(state="readonly")
                     self.target_lang_combobox.config(state="readonly"); self.font_size_spinbox.config(state="readonly")
                     current_status = self.status_var.get()
                     if "error" not in current_status.lower() and "stopped" not in current_status.lower() and "finished" not in current_status.lower(): self.status_var.set("Ready.")
             except Exception as e: print(f"Error resetting UI state in stop_translation: {e}")
             print(f"--- EXITING stop_translation (already stopped) ---")
             return
        elif stop_event.is_set() and not self.is_running:
             print("Stop called but already stopping/stopped (stop_event is set). Ignoring duplicate call.")
             print(f"--- EXITING stop_translation (duplicate call) ---")
             return
        # --- End check ---

        status = "Processing stopped by user."; log_status = status
        if error:
             current_status = self.status_var.get(); status = current_status if "CRITICAL ERROR" in current_status else "Stopped due to CRITICAL ERROR."
             log_status = "Processing stopped due to CRITICAL ERROR."
        elif graceful_stop: status = "Processing finished."; log_status = status

        print(f"stop_translation: Setting stop_event...")
        stop_event.set()
        print(f"stop_translation: Setting self.is_running = False")
        self.is_running = False

        print(f"stop_translation: Attempting to update GUI elements...")
        try:
            if self.root.winfo_exists():
                self.status_var.set(status)
                if not graceful_stop: self.add_output_direct(f"--- {log_status} ---", "error" if error else "info")
                self.start_button.config(state=self.start_button_state); self.stop_button.config(state=tk.DISABLED)
                self.device_combobox.config(state="readonly"); self.source_lang_combobox.config(state="readonly")
                self.target_lang_combobox.config(state="readonly"); self.font_size_spinbox.config(state="readonly")
                self.alpha_slider.config(state=tk.NORMAL) # <<<< Re-enable alpha slider
                print(f"stop_translation: GUI elements updated.")
            else: print("stop_translation: GUI window closed during UI update.")
        except Exception as e: print(f"Error updating GUI elements in stop_translation: {e}")

        print(f"stop_translation: Clearing queues...")
        self._clear_queues(clear_gui=True) # Clear all queues
        print(f"stop_translation: Queues cleared.")

        print(f"stop_translation: Resetting active threads list and status shown flag.")
        self.active_threads = []
        self.initial_status_shown = False

        print(f"--- EXITING stop_translation (source: {call_source}) ---\n")


    def _clear_queues(self, clear_gui=False):
        """Clears audio, transcribed, and optionally the GUI queue."""
        print("Clearing queues (best effort)...")
        queues_to_clear = [(audio_queue, "audio"), (transcribed_queue, "transcribed")] # Clear intermediate queue too
        if clear_gui: queues_to_clear.append((gui_queue, "gui"))
        cleared_counts = {"audio": 0, "transcribed": 0, "gui": 0}
        for q, name in queues_to_clear:
            cleared_count = 0
            while not q.empty():
                try: item = q.get_nowait(); cleared_count += 1
                except queue.Empty: break
                except Exception as e: print(f"Error clearing {name}_q item: {e}"); break
            cleared_counts[name] = cleared_count
            print(f"Cleared approx {cleared_counts[name]} items from {name}_queue.")


    def on_closing(self):
        print("\n--- ENTERING on_closing ('X' button) ---")
        if self.is_running:
            print("on_closing: Running, asking user...")
            if messagebox.askokcancel("Quit", "Translation is running. Stop and quit?"):
                print("on_closing: User confirmed quit while running. Stopping translation...")
                self.stop_translation(error=False, graceful_stop=False)
                print("on_closing: Scheduling root destroy via _destroy_root.")
                self.root.after(300, self._destroy_root)
            else:
                print("on_closing: Quit cancelled by user."); print("--- EXITING on_closing (cancelled) ---")
                return
        else:
            print("on_closing: Not running, calling _destroy_root immediately.")
            self._destroy_root()


    def _destroy_root(self):
       print("\n--- ENTERING _destroy_root ---")
       print("_destroy_root: Setting stop_event (final check).")
       stop_event.set()
       print("_destroy_root: Checking if root exists...")
       if self.root and self.root.winfo_exists():
           print("_destroy_root: Root exists, calling root.destroy().")
           try: self.root.destroy(); print("_destroy_root: Root window destroyed successfully.")
           except Exception as e: print(f"_destroy_root: Error during root destroy: {e}")
       else: print("_destroy_root: Root window does not exist or already destroyed.")
       print("--- EXITING _destroy_root ---")

# --- End of TranslatorApp Class ---

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting application...")
    # (Pre-checks remain the same)
    try: p_test = pyaudio.PyAudio(); p_test.terminate(); print("PyAudio check successful.")
    except Exception as e: print(f"CRITICAL PYAUDIO ERROR: {e}", file=sys.stderr); sys.exit(1)
    try: ollama.list(); print("Ollama connection check successful.")
    except Exception as e: print(f"WARNING: Ollama connection check failed: {e}. Ensure Ollama server is running.", file=sys.stderr)

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        print("Tk root created.")
        app = TranslatorApp(root)
        print("TranslatorApp initialized.")
        root.deiconify()
        # --- ADDED Logging ---
        print("\n>>> Starting mainloop...")
        root.mainloop()
        # --- ADDED Logging ---
        print("\n>>> Mainloop finished.\n")
    except Exception as e:
        print(f"CRITICAL ERROR during GUI setup or mainloop: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        # --- ADDED Logging ---
        print(">>> Application finishing, ensuring stop_event is set.")
        stop_event.set() # Ensure stop event is set on any exit
        print(">>> Application finished.")
