import argparse
import subprocess
import queue
import threading
import sys
import time
from collections import deque
import numpy as np
import torch
from faster_whisper import WhisperModel
import ollama

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox

# --- Configuration ---

# Define supported target languages (Display Name: ISO 639-1 Code)
# Add more languages here if needed and if your LLM supports them
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

# Whisper settings
WHISPER_MODEL_SIZE = "medium" # tiny, base, small, medium, large-v2, large-v3
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "default" # or: int8_float16, int8, float32

# Ollama settings
OLLAMA_MODEL = 'mistral' # Ensure this model is pulled in Ollama (ollama pull mistral)

# Audio settings
AUDIO_SAMPLERATE = 16000 # Whisper requires 16kHz
AUDIO_CHANNELS = 1 # Mono
AUDIO_BUFFER_SECONDS = 5 # Process audio in chunks of this duration

# --- Appearance & History Settings ---
HISTORY_MINUTES = 3        # How many minutes of translation history to keep
WINDOW_ALPHA = 0.85      # Window transparency (0.0=invisible to 1.0=opaque)

# --- Color Constants for Dark Theme ---
DARK_BG = '#2b2b2b'        # Dark gray background
LIGHT_FG = '#ffffff'       # White text
ENTRY_BG = '#3c3f41'       # Slightly lighter gray for entry background
BUTTON_BG = '#555555'      # Medium gray for button
BUTTON_ACTIVE_BG = '#666666' # Slightly lighter gray for active/hover button
TEXT_CURSOR = '#ffffff'    # White cursor color

# --- Global Queues ---
audio_queue = queue.Queue()
transcribed_queue = queue.Queue() # Will hold tuples: (detected_lang_code, text)
gui_queue = queue.Queue()         # For messages intended for the GUI display

# --- Thread Functions ---

# Using YouTube version as requested for now
def capture_audio_thread(url, audio_q, gui_q):
    """ Captures audio using yt-dlp/ffmpeg, puts chunks in audio_q, reports status/errors to gui_q. """
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Starting audio capture (YouTube via yt-dlp)...")
    ffmpeg_process = None
    try:
        yt_dlp_cmd = ['yt-dlp', '-f', 'bestaudio/best', '--get-url', url]
        stream_url = subprocess.check_output(yt_dlp_cmd, stderr=subprocess.STDOUT).decode('utf-8', errors='ignore').strip()
        gui_q.put(f"[{thread_name}] Got stream URL...")

        ffmpeg_cmd = [
            'ffmpeg', '-i', stream_url, '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(AUDIO_SAMPLERATE), '-ac', str(AUDIO_CHANNELS),
            '-f', 's16le', '-bufsize', '8192k', '-loglevel', 'error', '-'
        ]
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chunk_size = AUDIO_SAMPLERATE * AUDIO_CHANNELS * 2 * AUDIO_BUFFER_SECONDS

        gui_q.put(f"[{thread_name}] FFmpeg process started. Capturing audio...")

        while True:
            in_bytes = ffmpeg_process.stdout.read(chunk_size)
            if not in_bytes:
                if ffmpeg_process.poll() is not None:
                     stderr_data = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                     if stderr_data:
                           gui_q.put(f"[{thread_name}] FFmpeg Error: {stderr_data.strip()}")
                     gui_q.put(f"[{thread_name}] Audio stream ended.")
                     break
                else:
                     time.sleep(0.1)
                     continue
            audio_q.put(in_bytes)

    except subprocess.CalledProcessError as e:
        error_output = e.output.decode('utf-8', errors='ignore').strip()
        gui_q.put(f"[{thread_name}] ERROR: yt-dlp failed.\nURL: {url}\nOutput:\n{error_output}")
    except FileNotFoundError:
         gui_q.put(f"[{thread_name}] ERROR: 'yt-dlp' or 'ffmpeg' command not found. Make sure they are installed and in your system's PATH.")
    except Exception as e:
        gui_q.put(f"[{thread_name}] ERROR during audio capture: {e}")
    finally:
        audio_q.put(None)
        if ffmpeg_process and ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()
            try: ffmpeg_process.wait(timeout=2)
            except subprocess.TimeoutExpired: ffmpeg_process.kill()
        gui_q.put(f"[{thread_name}] Audio capture thread finished.")
        gui_q.put(None)


def transcribe_thread(audio_q, transcribed_q, gui_q):
    """ Transcribes audio: detects language initially, then locks it for the session. """
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Initializing Whisper model ({WHISPER_MODEL_SIZE})...")
    model = None
    detection_phase = True  # Start in detection phase
    locked_language = None  # Language to use after detection

    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        gui_q.put(f"[{thread_name}] Whisper model initialized. Detecting initial language...")

        while True:
            audio_data_bytes = audio_q.get()
            if audio_data_bytes is None:
                transcribed_q.put(None) # Pass signal downstream
                break

            try:
                audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                if audio_np.size > 0:

                    current_language_setting = None # Default to auto-detect in detection phase
                    if not detection_phase:
                        current_language_setting = locked_language # Use locked language after detection

                    segments, info = model.transcribe(
                        audio_np,
                        language=current_language_setting, # Use None during detection, locked lang after
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                    )

                    full_text = " ".join(segment.text for segment in segments).strip()

                    if detection_phase:
                        # Still in detection phase, check if we got usable text
                        if full_text:
                            # First chunk with actual text detected, lock the language
                            locked_language = info.language
                            detection_phase = False # Exit detection phase
                            detected_prob = info.language_probability
                            gui_q.put(f"[{thread_name}] Initial language detected and locked: {locked_language} (Prob: {detected_prob:.2f})")
                            # Send this first transcription result
                            if locked_language: # Ensure language was actually detected
                                transcribed_q.put((locked_language, full_text))
                        # else: still detecting, wait for a chunk with text
                    else:
                        # Detection phase is over, use the locked language
                        # Only put non-empty results
                        if full_text and locked_language:
                             transcribed_q.put((locked_language, full_text))

            except Exception as e:
                gui_q.put(f"[{thread_name}] Transcription error: {e}")
            finally:
                audio_q.task_done()

    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Failed to load Whisper model: {e}")
        transcribed_q.put(None)
    finally:
        gui_q.put(f"[{thread_name}] Transcription thread finished.")
        gui_q.put(None) # Signal GUI

def translate_thread(transcribed_q, gui_q, target_lang_code):
    """ Translates text using Ollama with a strict main prompt """
    thread_name = threading.current_thread().name
    initial_target_display = target_lang_code if target_lang_code else "N/A"
    # Note: System prompt context removed from this status message
    gui_q.put(f"[{thread_name}] Initializing Ollama client (Model: {OLLAMA_MODEL}, Target Lang: {initial_target_display})...")
    try:
        ollama.list()
        gui_q.put(f"[{thread_name}] Ollama connection successful. Waiting for text...")
    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Ollama connection failed: {e}")
        gui_q.put(None)
        return

    while True:
        data = transcribed_q.get()
        if data is None:
            break

        source_lang_code, text_to_translate = data
        if not target_lang_code:
             gui_q.put(f"[{thread_name}] Error: No target language selected for translation.")
             transcribed_q.task_done()
             continue

        try:
            if text_to_translate:
                start_time = time.time()

                # *** MODIFICATION HERE: Strict All-in-One Prompt ***
                # Combine context (optional) and strict output instructions directly
                context_info = "Context: The text is dialogue from a Japanese Hololive VTuber live stream." # Optional, can remove if it causes issues
                instruction = f"Translate the following Source Text into {target_lang_code}. IMPORTANT: Your entire response must consist ONLY of the raw translated text in {target_lang_code} and absolutely nothing else. Do not add explanations, commentary, notes, labels, source language text, or translations into other languages."

                # Construct the single prompt
                prompt = f"{context_info}\n\n{instruction}\n\nSource Text ({source_lang_code}): {text_to_translate}\n\n{target_lang_code} Translation:"
                # ******************************************************

                response = ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,           # Pass the combined prompt
                    # system=...          # REMOVED system parameter
                    stream=False,
                    options={"temperature": 0.5} # Optional: Lower temperature might help consistency
                )

                translation = response['response'].strip()
                latency = time.time() - start_time

                if translation:
                    # Basic filtering for common unwanted additions (can be expanded)
                    if "(translation from" in translation.lower():
                        gui_q.put(f"[{thread_name}] Warning: Filtered out potential explanation from translation.")
                        translation = translation.split("\n")[0].strip() # Try taking only the first line

                    # Remove common parenthetical notes (adjust regex if needed)
                    import re
                    translation = re.sub(r'\s*\([^)]*\)$', '', translation).strip()

                    # Check again if anything remains after filtering
                    if translation:
                         gui_q.put(f"[{target_lang_code.upper()} ({latency:.2f}s)] {translation}")
                    elif len(text_to_translate.strip()) > 2:
                         gui_q.put(f"[{thread_name}] Warning: Translation became empty after filtering: '{response['response'].strip()[:50]}...'")

                else:
                     if len(text_to_translate.strip()) > 2:
                          gui_q.put(f"[{thread_name}] Warning: Empty translation response from LLM for input: '{text_to_translate[:50]}...'")

        except Exception as e:
            gui_q.put(f"[{thread_name}] Translation error: {e}")
        finally:
            transcribed_q.task_done()

    gui_q.put(f"[{thread_name}] Translation thread finished.")
    gui_q.put(None) # Signal GUI


# --- Tkinter GUI Application ---

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multilingual Live Translator")
        self.root.geometry("700x350") # Adjust size as needed

        # --- Dark Theme Configuration ---
        self.root.config(bg=DARK_BG)
        style = ttk.Style(self.root)
        try:
             style.theme_use('clam')
        except tk.TclError:
             print("Warning: 'clam' theme not available, using default.")

        style.configure('.', background=DARK_BG, foreground=LIGHT_FG)
        style.configure('TFrame', background=DARK_BG)
        style.configure('TLabel', background=DARK_BG, foreground=LIGHT_FG)
        style.configure('TButton', background=BUTTON_BG, foreground=LIGHT_FG, padding=5)
        style.map('TButton', background=[('active', BUTTON_ACTIVE_BG)])
        style.configure('TEntry', fieldbackground=ENTRY_BG, foreground=LIGHT_FG, insertcolor=TEXT_CURSOR)
        style.configure('TCombobox',
                        fieldbackground=ENTRY_BG, background=BUTTON_BG, foreground=LIGHT_FG,
                        arrowcolor=LIGHT_FG, selectbackground=DARK_BG, selectforeground=LIGHT_FG)
        # Ensure dropdown list uses theme (might need more specific TComboboxPopdown styling if available/needed)
        root.option_add('*TCombobox*Listbox.background', ENTRY_BG)
        root.option_add('*TCombobox*Listbox.foreground', LIGHT_FG)
        root.option_add('*TCombobox*Listbox.selectBackground', BUTTON_ACTIVE_BG)
        root.option_add('*TCombobox*Listbox.selectForeground', LIGHT_FG)


        # --- Window Attributes for Overlay ---
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', WINDOW_ALPHA)

        self.is_running = False
        self.active_threads = []
        self.message_deque = deque()

        # --- Input Frame ---
        input_frame = ttk.Frame(root, padding="10")
        input_frame.pack(fill=tk.X)

        ttk.Label(input_frame, text="Stream URL:").pack(side=tk.LEFT, padx=(0, 5))
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(input_frame, textvariable=self.url_var, width=50)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # --- Target Language Selection ---
        ttk.Label(input_frame, text="To:").pack(side=tk.LEFT, padx=(5, 2))
        self.target_lang_var = tk.StringVar()
        lang_options = list(SUPPORTED_TARGET_LANGUAGES.keys())
        self.lang_combobox = ttk.Combobox(input_frame,
                                          textvariable=self.target_lang_var,
                                          values=lang_options,
                                          state="readonly",
                                          width=15) # Increased width slightly
        # Set default value
        if DEFAULT_TARGET_LANG in lang_options:
            self.target_lang_var.set(DEFAULT_TARGET_LANG)
        elif lang_options:
             self.target_lang_var.set(lang_options[0])
        self.lang_combobox.pack(side=tk.LEFT, padx=(0, 5))

        self.start_button = ttk.Button(input_frame, text="Start", command=self.start_translation)
        self.start_button.pack(side=tk.LEFT, padx=(5, 0))

        # --- Output Area ---
        output_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state=tk.DISABLED, height=10)
        self.output_text.config(bg=DARK_BG, fg=LIGHT_FG, insertbackground=TEXT_CURSOR)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Enter URL, select target language, and press Start.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def add_output_direct(self, text):
        """ Adds text (like status/errors) directly without history limit """
        try:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, text + "\n")
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        except tk.TclError as e:
            print(f"Error adding text to GUI (window closed?): {e}")


    def update_output_display(self):
        """ Clears and rewrites the text area from the message deque """
        try:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete('1.0', tk.END)
            for _, msg in self.message_deque:
                 self.output_text.insert(tk.END, msg + "\n")
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        except tk.TclError as e:
            print(f"Error updating GUI display (window closed?): {e}")


    def check_gui_queue(self):
        """ Checks the GUI queue, manages history deque, and updates the GUI """
        try:
            while True:
                message = gui_queue.get_nowait()
                current_time = time.time()

                if message is None:
                    pass
                elif isinstance(message, str):
                    if message.startswith("[EN") or message.startswith("[ID") or \
                       message.startswith("[RU") or message.startswith("[KO") or \
                       message.startswith("[JA") or message.startswith("[ES") or \
                       message.startswith("[FR") or message.startswith("[DE") or \
                       message.startswith("[ZH"): # Check if it starts like a translation
                        self.message_deque.append((current_time, message))
                        cutoff_time = current_time - (HISTORY_MINUTES * 60)
                        while self.message_deque and self.message_deque[0][0] < cutoff_time:
                            self.message_deque.popleft()
                        self.update_output_display()
                        self.status_var.set("Processing...")
                    elif message.startswith("CRITICAL ERROR"):
                         self.status_var.set(message)
                         self.add_output_direct(f"*** {message} ***")
                         self.stop_translation(error=True)
                    elif message.startswith("ERROR") or message.startswith("Warning"):
                         self.status_var.set("Error/Warning occurred.")
                         self.add_output_direct(message)
                    else: # Status message
                         self.status_var.set(message.split("] ", 1)[-1])
                         self.add_output_direct(f"STATUS: {message}")

                gui_queue.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing GUI queue: {e}")


        if self.is_running:
            # Use try-except in case window is destroyed during check
            try:
                self.root.after(100, self.check_gui_queue)
            except tk.TclError:
                 print("Info: GUI window closed, stopping queue check.")
                 self.is_running = False # Stop trying to schedule


    def monitor_threads(self):
        """ Checks if processing threads are still alive """
        if not self.is_running:
            return

        all_done = True
        for t in self.active_threads:
            if t.is_alive():
                all_done = False
                break

        if all_done:
            try:
                self.status_var.set("Processing finished.")
                self.add_output_direct("--- All processing threads finished ---")
                self.is_running = False
                self.start_button.config(state=tk.NORMAL)
                self.url_entry.config(state=tk.NORMAL)
                self.lang_combobox.config(state=tk.NORMAL)
                self.active_threads = []
            except tk.TclError:
                 print("Info: GUI window closed before processing fully finished.")
        else:
            try:
                self.root.after(500, self.monitor_threads)
            except tk.TclError:
                print("Info: GUI window closed, stopping thread monitor.")
                self.is_running = False


    def start_translation(self):
        """ Starts the translation process in separate threads """
        if self.is_running:
            messagebox.showwarning("Already Running", "Translation is already in progress.")
            return

        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Input Error", "Please enter a Stream URL.")
            return

        selected_lang_name = self.target_lang_var.get()
        if not selected_lang_name:
             messagebox.showerror("Input Error", "Please select a target language.")
             return
        selected_target_lang_code = SUPPORTED_TARGET_LANGUAGES.get(selected_lang_name, None)
        if not selected_target_lang_code:
             messagebox.showerror("Internal Error", "Invalid target language selected.")
             return

        # Clear previous output and deque
        try:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete('1.0', tk.END)
            self.output_text.config(state=tk.DISABLED)
            self.message_deque.clear()

            self.status_var.set(f"Starting threads... Target: {selected_lang_name}")
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.url_entry.config(state=tk.DISABLED)
            self.lang_combobox.config(state=tk.DISABLED)
            self.active_threads = []
        except tk.TclError:
            print("Error starting: GUI already closed?")
            return # Don't start if GUI isn't usable

        # --- Create and Start Threads ---
        while not audio_queue.empty(): audio_queue.get()
        while not transcribed_queue.empty(): transcribed_queue.get()
        while not gui_queue.empty(): gui_queue.get()

        capture_func = capture_audio_thread # Using YT version

        capture = threading.Thread(
            target=capture_func,
            args=(url, audio_queue, gui_queue),
            name="AudioCapture", daemon=True)

        transcriber = threading.Thread(
            target=transcribe_thread,
            args=(audio_queue, transcribed_queue, gui_queue),
            name="Transcribe", daemon=True)

        translator = threading.Thread(
            target=translate_thread,
            args=(transcribed_queue, gui_queue, selected_target_lang_code),
            name="Translate", daemon=True)

        self.active_threads.extend([capture, transcriber, translator])

        capture.start()
        transcriber.start()
        translator.start()

        self.root.after(100, self.check_gui_queue) # Start polling GUI queue
        self.root.after(500, self.monitor_threads) # Start monitoring threads


    def stop_translation(self, error=False):
         """ (Basic attempt) Resets GUI state """
         if not self.is_running and not error: return

         self.is_running = False # Important to stop check_gui_queue loop
         status = "Processing stopped by user." if not error else "Processing stopped due to error."

         try:
             self.status_var.set(status)
             self.add_output_direct(f"--- {status} ---")
             self.start_button.config(state=tk.NORMAL)
             self.url_entry.config(state=tk.NORMAL)
             self.lang_combobox.config(state=tk.NORMAL)
         except tk.TclError:
             print("Info: GUI window closed during stop sequence.")

         # Clear queues (best effort)
         while not audio_queue.empty():
             try: audio_queue.get_nowait()
             except queue.Empty: break
         while not transcribed_queue.empty():
             try: transcribed_queue.get_nowait()
             except queue.Empty: break
         while not gui_queue.empty():
             try: gui_queue.get_nowait()
             except queue.Empty: break

         self.active_threads = [] # Threads are daemons, just clear the list


    def on_closing(self):
        """ Handle window closing """
        if self.is_running:
             if messagebox.askokcancel("Quit", "Translation is running. Quit anyway?"):
                 self.stop_translation()
                 # Give threads a moment to potentially exit cleanly after flags are set
                 self.root.after(200, self.root.destroy)
             else:
                 return # Don't close yet
        else:
            self.root.destroy()


# --- Main Execution ---
if __name__ == "__main__":
    # Check Ollama availability early (optional but helpful)
    try:
        ollama.list()
        print("Ollama connection check successful on startup.")
    except Exception as e:
         # Show warning in console, still try to start GUI
         print(f"WARNING: Could not connect to Ollama on startup: {e}", file=sys.stderr)
         print("Ensure the Ollama application is running.", file=sys.stderr)
         # Alternative: Show GUI error and exit
         # root = tk.Tk()
         # root.withdraw() # Hide root window
         # messagebox.showerror("Ollama Error", f"Could not connect to Ollama. Please ensure it is running.\n\nError: {e}")
         # sys.exit(1)

    # --- Create and Run GUI ---
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
