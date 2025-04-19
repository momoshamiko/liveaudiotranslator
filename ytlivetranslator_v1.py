import argparse
import subprocess
import queue
import threading
import sys
import time  # <--- Import time
from collections import deque # <--- Import deque
import numpy as np
import torch
from faster_whisper import WhisperModel
import ollama

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox

# --- Configuration ---
TARGET_LANGUAGE = "en"
SOURCE_LANGUAGE = "ja"
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "default"
OLLAMA_MODEL = 'mistral'
OLLAMA_TRANSLATION_PROMPT = f"Translate the following Japanese text accurately and concisely to English. Only output the English translation:\n\nJapanese: {{text}}\n\nEnglish:"
AUDIO_SAMPLERATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BUFFER_SECONDS = 5
HISTORY_MINUTES = 3 # <--- Added: How many minutes of history to keep
WINDOW_ALPHA = 0.85 # <--- Added: Window transparency (0.0 to 1.0)

# --- Color Constants for Dark Theme ---
DARK_BG = '#2b2b2b'        # Dark gray background
LIGHT_FG = '#ffffff'       # White text
ENTRY_BG = '#3c3f41'       # Slightly lighter gray for entry background
BUTTON_BG = '#555555'      # Medium gray for button
BUTTON_ACTIVE_BG = '#666666' # Slightly lighter gray for active/hover button
TEXT_CURSOR = '#ffffff'    # White cursor color

# --- Window Attributes for Overlay ---
# Moved HISTORY_MINUTES and WINDOW_ALPHA here too for clarity
HISTORY_MINUTES = 3 # How many minutes of history to keep
WINDOW_ALPHA = 0.85 # Window transparency (0.0 to 1.0)

# --- Global Queues ---
audio_queue = queue.Queue()
transcribed_queue = queue.Queue()
gui_queue = queue.Queue()

# --- Thread Functions (Keep as before) ---
def capture_audio_thread(url, audio_q, gui_q):
    """ Captures audio, puts chunks in audio_q, reports status/errors to gui_q. """
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Starting audio capture...")
    ffmpeg_process = None # Define ffmpeg_process to ensure it's available in finally block
    try:
        yt_dlp_cmd = ['yt-dlp', '-f', 'bestaudio/best', '--get-url', url]
        stream_url = subprocess.check_output(yt_dlp_cmd, stderr=subprocess.STDOUT).decode('utf-8', errors='ignore').strip() # Capture stderr too
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
                if ffmpeg_process.poll() is not None: # Check if process ended
                     stderr_data = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                     if stderr_data:
                           gui_q.put(f"[{thread_name}] FFmpeg Error: {stderr_data.strip()}")
                     gui_q.put(f"[{thread_name}] Audio stream ended.")
                     break
                else:
                     # Stream might just be temporarily silent, wait a bit
                     time.sleep(0.1)
                     continue # Continue reading

            audio_q.put(in_bytes)

    except subprocess.CalledProcessError as e:
        error_output = e.output.decode('utf-8', errors='ignore').strip()
        gui_q.put(f"[{thread_name}] ERROR: yt-dlp failed.\nURL: {url}\nOutput:\n{error_output}")
    except FileNotFoundError:
         gui_q.put(f"[{thread_name}] ERROR: 'yt-dlp' or 'ffmpeg' command not found. Make sure they are installed and in your system's PATH.")
    except Exception as e:
        gui_q.put(f"[{thread_name}] ERROR during audio capture: {e}")
    finally:
        audio_q.put(None) # Signal end of stream to transcriber
        if ffmpeg_process and ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                ffmpeg_process.kill()
        gui_q.put(f"[{thread_name}] Audio capture thread finished.")
        # Signal GUI from here too, in case other threads fail early
        gui_q.put(None)


def transcribe_thread(audio_q, transcribed_q, gui_q):
    """ Transcribes audio from audio_q, puts text in transcribed_q, reports status/errors to gui_q. """
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Initializing Whisper model ({WHISPER_MODEL_SIZE})...")
    model = None
    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        gui_q.put(f"[{thread_name}] Whisper model initialized on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE}). Waiting for audio...")

        while True:
            audio_data_bytes = audio_q.get()
            if audio_data_bytes is None:
                transcribed_q.put(None) # Pass signal downstream
                break

            try:
                audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                if audio_np.size > 0:
                    segments, info = model.transcribe(
                        audio_np, language=SOURCE_LANGUAGE, beam_size=5,
                        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
                    )
                    full_text = " ".join(segment.text for segment in segments).strip()
                    if full_text:
                        transcribed_q.put(full_text)

            except Exception as e:
                gui_q.put(f"[{thread_name}] Transcription error: {e}")
            finally:
                audio_q.task_done()

    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Failed to load Whisper model: {e}")
        transcribed_q.put(None) # Ensure downstream knows failure occurred
    finally:
        gui_q.put(f"[{thread_name}] Transcription thread finished.")
        gui_q.put(None) # Signal GUI

def translate_thread(transcribed_q, gui_q):
    """ Translates text from transcribed_q using Ollama, puts results/errors into gui_q. """
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Initializing Ollama client for model '{OLLAMA_MODEL}'...")
    try:
        ollama.list() # Ping Ollama server
        gui_q.put(f"[{thread_name}] Ollama connection successful. Waiting for text...")
    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Ollama connection failed. Is Ollama running? Error: {e}")
        gui_q.put(None) # Signal GUI about critical failure
        return # Exit thread

    while True:
        text_to_translate = transcribed_q.get()
        if text_to_translate is None:
            break # End signal from upstream

        try:
            if text_to_translate:
                start_time = time.time()
                prompt = OLLAMA_TRANSLATION_PROMPT.format(text=text_to_translate)
                response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, stream=False)
                translation = response['response'].strip()
                latency = time.time() - start_time

                if translation:
                    # Send final translation to GUI
                    gui_q.put(f"[EN ({latency:.2f}s)] {translation}")
                else:
                     gui_q.put(f"[{thread_name}] Warning: Empty translation response from LLM for input: '{text_to_translate[:50]}...'")

        except Exception as e:
            gui_q.put(f"[{thread_name}] Translation error: {e}")
        finally:
            transcribed_q.task_done()

    gui_q.put(f"[{thread_name}] Translation thread finished.")
    gui_q.put(None) # Signal GUI

    


# --- Tkinter GUI Application (Modified) ---

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hololive Live Translator")
        self.root.geometry("700x350") # Adjusted size slightly

        # --- Dark Theme Configuration ---
        self.root.config(bg=DARK_BG) # Set root background

        style = ttk.Style(self.root)
        # Set theme - 'clam' is often good for styling
        # On some systems you might try 'alt' or others if 'clam' looks odd
        try:
             style.theme_use('clam')
        except tk.TclError:
             print("Warning: 'clam' theme not available, using default.")
             # Default theme might not respond as well to styling below

        # Configure styles for ttk widgets
        style.configure('.', background=DARK_BG, foreground=LIGHT_FG) # Basic default for ttk
        style.configure('TFrame', background=DARK_BG)
        style.configure('TLabel', background=DARK_BG, foreground=LIGHT_FG)
        style.configure('TButton', background=BUTTON_BG, foreground=LIGHT_FG, padding=5)
        # Change button color slightly when hovered/pressed
        style.map('TButton', background=[('active', BUTTON_ACTIVE_BG)])
        style.configure('TEntry',
                        fieldbackground=ENTRY_BG, # Background of the text area
                        foreground=LIGHT_FG,      # Text color
                        insertcolor=TEXT_CURSOR)  # Cursor color

        # --- Window Attributes for Overlay ---
        self.root.attributes('-topmost', True) # Keep window always on top
        self.root.attributes('-alpha', WINDOW_ALPHA) # Set semi-transparency

        self.is_running = False
        self.active_threads = []
        self.message_deque = deque()

        # --- Input Frame (using ttk) ---
        input_frame = ttk.Frame(root, padding="10") # Uses configured TFrame style
        input_frame.pack(fill=tk.X)

        # Label uses configured TLabel style
        ttk.Label(input_frame, text="YouTube Live URL:").pack(side=tk.LEFT, padx=(0, 5))

        self.url_var = tk.StringVar()
        # Entry uses configured TEntry style
        self.url_entry = ttk.Entry(input_frame, textvariable=self.url_var, width=60)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Button uses configured TButton style
        self.start_button = ttk.Button(input_frame, text="Start", command=self.start_translation)
        self.start_button.pack(side=tk.LEFT, padx=(5, 0))

        # --- Output Area (using standard tk ScrolledText) ---
        output_frame = ttk.Frame(root, padding=(10, 0, 10, 10)) # Uses configured TFrame style
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state=tk.DISABLED, height=10)
        # Configure ScrolledText background/foreground manually as it's not a ttk widget
        self.output_text.config(bg=DARK_BG,             # Background color
                                fg=LIGHT_FG,             # Text color
                                insertbackground=TEXT_CURSOR) # Cursor color
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # --- Status Bar (using ttk) ---
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Enter URL and press Start.")
        # Status bar uses configured TLabel style, relief adds border
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- Other methods (add_output_direct, update_output_display, check_gui_queue, etc.) remain the same ---
    # (Make sure they are still present below this __init__ method in your class)
    def add_output_direct(self, text):
        """ Adds text (like status/errors) directly without history limit """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

    def update_output_display(self):
        """ Clears and rewrites the text area from the message deque """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete('1.0', tk.END)
        for _, msg in self.message_deque: # Iterate through deque messages
             self.output_text.insert(tk.END, msg + "\n")
        self.output_text.see(tk.END) # Scroll to the end
        self.output_text.config(state=tk.DISABLED)


    def check_gui_queue(self):
        """ Checks the GUI queue, manages history deque, and updates the GUI """
        try:
            while True:
                message = gui_queue.get_nowait()
                current_time = time.time() # Get time for timestamping

                if message is None:
                    pass # Handle thread completion in monitor_threads
                elif isinstance(message, str):
                    if message.startswith("[EN"): # It's a translation
                        # Add to deque with timestamp
                        self.message_deque.append((current_time, message))
                        # Prune old messages from deque
                        cutoff_time = current_time - (HISTORY_MINUTES * 60)
                        while self.message_deque and self.message_deque[0][0] < cutoff_time:
                            self.message_deque.popleft()
                        # Update the display with current deque content
                        self.update_output_display()
                        self.status_var.set("Processing...")
                    elif message.startswith("CRITICAL ERROR"):
                         self.status_var.set(message)
                         self.add_output_direct(f"*** {message} ***") # Add directly
                         self.stop_translation(error=True)
                    elif message.startswith("ERROR") or message.startswith("Warning"):
                         self.status_var.set("Error/Warning occurred.")
                         self.add_output_direct(message) # Add directly
                    else: # Status message
                        self.status_var.set(message.split("] ", 1)[-1])
                        self.add_output_direct(f"STATUS: {message}") # Add directly

                gui_queue.task_done()
        except queue.Empty:
            pass

        if self.is_running:
            self.root.after(100, self.check_gui_queue)

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
            self.status_var.set("Processing finished.")
            self.add_output_direct("--- All processing threads finished ---") # Use direct add
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.url_entry.config(state=tk.NORMAL)
            self.active_threads = []
        else:
            # Reschedule monitoring
            self.root.after(500, self.monitor_threads) # Check every 500ms


    def start_translation(self):
        """ Starts the translation process in separate threads """
        if self.is_running:
            messagebox.showwarning("Already Running", "Translation is already in progress.")
            return

        url = self.url_var.get().strip()
        if not url:
            # NOTE: Ensure messagebox is imported: from tkinter import messagebox
            messagebox.showerror("Input Error", "Please enter a YouTube Live Stream URL.")
            return

        # Clear previous output and deque
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete('1.0', tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.message_deque.clear() # <--- Clear history deque

        self.status_var.set("Starting threads...")
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.url_entry.config(state=tk.DISABLED)
        self.active_threads = [] # Reset thread list

        # --- Create and Start Threads ---
        while not audio_queue.empty(): audio_queue.get()
        while not transcribed_queue.empty(): transcribed_queue.get()
        while not gui_queue.empty(): gui_queue.get()

        capture = threading.Thread(
            target=capture_audio_thread,
            args=(url, audio_queue, gui_queue),
            name="AudioCapture", daemon=True)

        transcriber = threading.Thread(
            target=transcribe_thread,
            args=(audio_queue, transcribed_queue, gui_queue),
            name="Transcribe", daemon=True)

        translator = threading.Thread(
            target=translate_thread,
            args=(transcribed_queue, gui_queue),
            name="Translate", daemon=True)

        self.active_threads.extend([capture, transcriber, translator])

        capture.start()
        transcriber.start()
        translator.start()

        self.root.after(100, self.check_gui_queue)
        self.root.after(500, self.monitor_threads)


    def stop_translation(self, error=False):
         """ (Basic attempt) Signals threads should stop """
         if not self.is_running and not error: return

         self.is_running = False
         status = "Processing stopped by user." if not error else "Processing stopped due to error."
         self.status_var.set(status)
         self.add_output_direct(f"--- {status} ---") # Use direct add

         self.start_button.config(state=tk.NORMAL)
         self.url_entry.config(state=tk.NORMAL)

         while not audio_queue.empty(): audio_queue.get()
         while not transcribed_queue.empty(): transcribed_queue.get()
         while not gui_queue.empty(): gui_queue.get()

         self.active_threads = []


    def on_closing(self):
        """ Handle window closing """
        if self.is_running:
             # NOTE: Ensure messagebox is imported: from tkinter import messagebox
             if messagebox.askokcancel("Quit", "Translation is running. Quit anyway?"):
                 self.stop_translation()
                 self.root.destroy()
             else:
                 return
        else:
            self.root.destroy()


# --- Main Execution ---
if __name__ == "__main__":
    try:
        ollama.list()
    except Exception as e:
         print("WARNING: Could not connect to Ollama on startup. Ensure Ollama is running.", file=sys.stderr)
         # Optionally show a Tkinter messagebox error here and exit
         # root = tk.Tk()
         # root.withdraw() # Hide root window
         # messagebox.showerror("Ollama Error", "Could not connect to Ollama. Please ensure it is running.")
         # sys.exit(1)


    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
