import queue
import os
import threading
import sys
import time
import datetime
import warnings
from collections import deque
import numpy as np
import tkinter as tk
import torch
from tkinter import ttk, messagebox, font
from tkinter import font as tkFont
from PIL import Image, ImageTk
import re
import pyaudio

# Filter PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda") 
warnings.filterwarnings("ignore", message=".*GPU available but not used.*")
warnings.filterwarnings("ignore", message=".*Could not infer.*")

# Import from our modules
from config import *
from audio_utils import get_audio_devices, capture_audio_thread_gui, AUDIO_FORMAT
from transcribe_utils import transcribe_translate_thread_gui
from gui_utils import create_queue_processor

# --- Global Queues & Events ---
audio_queue = queue.Queue()
gui_queue = queue.Queue()      # Single queue for all GUI updates
stop_event = threading.Event() # Event to signal threads to stop

# --- Ignore Patterns ---
IGNORE_PATTERNS = [
    re.compile(r"(?:thank\s*(?:you|s)\s*(?:so|very|a\s*lot)?\s*(?:much)?\s*)?(?:for\s+(?:watching|viewing|your\s+viewing)).*?(?:in\s+this\s+video)?", re.IGNORECASE),
    re.compile(r"see\s+you(?:\s+all|\s+again)?\s+(?:next\s+(?:time|video)|later|in\s+the\s+next\s+video)", re.IGNORECASE),
    re.compile(r"subscribe\s+to\s+(?:my|the)\s+channel", re.IGNORECASE),
    # Add more patterns as needed
]

#--------- Enhanced Overlay Bar ----------#
class SubtitleOverlay:
    """Manages the floating subtitle overlay window."""
    
    def __init__(self, initial_settings):
        self.root = tk.Toplevel()
        self.root.title("Subtitles")
        self.root.overrideredirect(True) # Remove window decorations

        # --- Make window background fully transparent (if supported) ---
        self.transparent_color = '#abcdef' # Use an unlikely color
        self.root.config(bg=self.transparent_color)
        self.supports_transparent_color = False
        try:
            # This attribute allows clicks to pass through the transparent parts of the window on some systems (Windows)
            self.root.attributes("-transparentcolor", self.transparent_color)
            self.supports_transparent_color = True
            print("INFO: Overlay using -transparentcolor for background.", flush=True)
        except tk.TclError:
            print("WARNING: Overlay -transparentcolor not supported. Falling back to window alpha.", flush=True)
            # If not supported, the canvas background itself will need to be drawn or be the window bg

        # --- Apply initial settings ---
        self.font_family = initial_settings.get('font_family', DEFAULT_OVERLAY_FONT_FAMILY)
        self.font_size = initial_settings.get('font_size', DEFAULT_OVERLAY_FONT_SIZE)
        self.font_weight = initial_settings.get('font_weight', DEFAULT_OVERLAY_FONT_WEIGHT)
        self.text_color = initial_settings.get('text_color', DEFAULT_OVERLAY_TEXT_COLOR).lower() # Store color name
        self.background_color_hex = initial_settings.get('bg_color', DEFAULT_OVERLAY_BG_COLOR)
        self.overlay_alpha = initial_settings.get('alpha', DEFAULT_OVERLAY_ALPHA)
        self.width_factor = initial_settings.get('width_factor', DEFAULT_OVERLAY_WIDTH_FACTOR)
        self.enable_background = initial_settings.get('enable_background', DEFAULT_ENABLE_OVERLAY_BACKGROUND)

        self.padding_x = 15
        self.padding_y = 10

        # Apply overall window alpha (controls transparency of everything in the window)
        self.root.attributes('-alpha', self.overlay_alpha)

        # --- Always On Top Handling ---
        self.root.attributes('-topmost', True)
        if sys.platform == "darwin": # macOS specific
            try:
                self.root.tk.call('::tk::unsupported::MacWindowStyle', 'style', self.root._w, 'floating', 'nonactivating')
                print("INFO: Applied macOS specific floating window style.", flush=True)
            except tk.TclError as e:
                print(f"WARNING: Failed to apply macOS floating style: {e}", flush=True)

        # --- Canvas Setup ---
        # Set canvas background based on whether true transparency is supported AND background is disabled
        canvas_bg_color = self.transparent_color if self.supports_transparent_color else self.background_color_hex
        if not self.enable_background and not self.supports_transparent_color:
             canvas_bg_color = self.transparent_color # Try anyway

        self.canvas = tk.Canvas(
            self.root,
            bg=canvas_bg_color,
            highlightthickness=0 # Remove canvas border
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Store IDs of canvas items
        self._text_item = None
        self._rect_item = None # For the optional background bar
        self._text_font = tkFont.Font(family=self.font_family, size=self.font_size, weight=self.font_weight)
        self._current_text = ""

        # --- Drag Functionality ---
        self._drag_data = {"x": 0, "y": 0}
        # Bind dragging to the canvas AND the background rectangle (if it exists)
        self.canvas.bind("<ButtonPress-1>", self.start_move)
        self.canvas.bind("<ButtonRelease-1>", self.stop_move)
        self.canvas.bind("<B1-Motion>", self.do_move)

        # --- Initial Positioning and Sizing ---
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self._calculate_initial_geometry()

        # Initial draw
        self.redraw_canvas()

    def _calculate_initial_geometry(self):
        """Calculates and sets the initial window size and position."""
        target_width = int(self.screen_width * self.width_factor)
        # Estimate height based on font size and padding
        try:
            # Use font metrics for a more accurate initial height estimate
            lines = self._current_text.count('\n') + 1 # Start with at least one line
            font_height = self._text_font.metrics('linespace') # Height of one line
            target_height = (font_height * lines) + (self.padding_y * 2)
        except tk.TclError: # Fallback if font metrics fail early
             target_height = self.font_size + (self.padding_y * 4) # Rough estimate

        # Ensure minimum dimensions
        target_width = max(target_width, 50)
        target_height = max(target_height, self.padding_y * 2 + 10) # Min height based on padding

        # Position near bottom-center
        initial_x = (self.screen_width - target_width) // 2
        initial_y = self.screen_height - (int(target_height) + 60) # 60 pixels from bottom

        self.root.geometry(f"{target_width}x{int(target_height)}+{initial_x}+{initial_y}")
        self.root.update_idletasks() # Ensure geometry is applied before first draw

    def redraw_canvas(self, event=None):
        """Redraws canvas items based on current text and settings."""
        if not hasattr(self, 'canvas') or not self.canvas.winfo_exists():
            return

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1: # Avoid drawing if canvas isn't ready
             # Use idle tasks to prevent potential recursion depth issues if called rapidly
             self.root.after_idle(self.redraw_canvas)
             return

        # --- Clear previous items ---
        if self._rect_item: self.canvas.delete(self._rect_item); self._rect_item = None
        if self._text_item: self.canvas.delete(self._text_item); self._text_item = None

        # --- Draw Optional Background Rectangle ---
        if self.enable_background:
            self._rect_item = self.canvas.create_rectangle(
                0, 0, width, height,
                fill=self.background_color_hex, # Use the defined background color
                outline="" # No border on the rectangle itself
            )
            if self._rect_item:
                 self.canvas.tag_bind(self._rect_item, "<ButtonPress-1>", self.start_move)
                 self.canvas.tag_bind(self._rect_item, "<ButtonRelease-1>", self.stop_move)
                 self.canvas.tag_bind(self._rect_item, "<B1-Motion>", self.do_move)

        # --- Text Color (Directly use the stored color) ---
        effective_text_color = self.text_color # Use the stored color ("white" or "black")

        # --- Draw Text ---
        if self._current_text:
            wrap_width = max(width - (self.padding_x * 2), 1)
            try:
                self._text_item = self.canvas.create_text(
                    width / 2, height / 2, # Center anchor
                    text=self._current_text,
                    font=self._text_font,
                    fill=effective_text_color, # Use the direct color
                    justify="center",
                    width=wrap_width
                )
                if self._text_item:
                    self.canvas.tag_bind(self._text_item, "<ButtonPress-1>", self.start_move)
                    self.canvas.tag_bind(self._text_item, "<ButtonRelease-1>", self.stop_move)
                    self.canvas.tag_bind(self._text_item, "<B1-Motion>", self.do_move)
            except tk.TclError as e:
                 print(f"ERROR creating canvas text item: {e}", flush=True)
                 print(f"  Attempted color: {effective_text_color}", flush=True)
                 # Fallback to default text color if creation fails
                 try:
                      self._text_item = self.canvas.create_text(
                           width / 2, height / 2, text=self._current_text, font=self._text_font,
                           fill=DEFAULT_OVERLAY_TEXT_COLOR, justify="center", width=wrap_width
                      )
                      if self._text_item:
                           self.canvas.tag_bind(self._text_item, "<ButtonPress-1>", self.start_move)
                           self.canvas.tag_bind(self._text_item, "<ButtonRelease-1>", self.stop_move)
                           self.canvas.tag_bind(self._text_item, "<B1-Motion>", self.do_move)
                 except Exception as fallback_e:
                      print(f"ERROR: Fallback text creation also failed: {fallback_e}", flush=True)

        # --- Adjust Height (Scheduled using after_idle) ---
        self.root.after_idle(self._adjust_height_to_text)


    def _adjust_height_to_text(self):
         """Adjusts the window height to fit the text content."""
         if not hasattr(self, 'root') or not self.root or not self.root.winfo_exists(): 
              return # Added hasattr check
         
         new_height = self.padding_y * 2 + 10 # Minimum height
         if self._text_item:
             try:
                  # It might be safer to update idle tasks *before* bbox calculation
                  self.canvas.update_idletasks()
                  bbox = self.canvas.bbox(self._text_item)
                  if bbox:
                       text_height = bbox[3] - bbox[1]
                       required_height = text_height + (self.padding_y * 2)
                       new_height = max(required_height, new_height)
             except tk.TclError as e:
                  if "invalid command name" not in str(e):
                       print(f"Warning: TclError getting text bounding box: {e}", flush=True)
             except Exception as e:
                  print(f"Error calculating text height: {e}", flush=True)

         try: # Wrap geometry update in try/except
             current_height = self.root.winfo_height()
             if abs(int(new_height) - current_height) > 1:
                  current_width = self.root.winfo_width()
                  current_x = self.root.winfo_x()
                  current_y = self.root.winfo_y()
                  self.root.geometry(f"{current_width}x{int(new_height)}+{current_x}+{current_y}")
         except tk.TclError as e:
              if "invalid command name" not in str(e):
                   print(f"Warning: TclError setting geometry during height adjustment: {e}", flush=True)
         except Exception as e:
              print(f"Error setting geometry during height adjustment: {e}", flush=True)


    def update_text(self, text):
        """Updates the text displayed on the canvas."""
        new_text = text.strip()
        if new_text != self._current_text:
            self._current_text = new_text
            self.redraw_canvas() # Let redraw handle scheduling if needed

    def clear_text(self):
        """Clears the text."""
        if self._current_text != "":
            self._current_text = ""
            self.redraw_canvas() # Let redraw handle scheduling if needed

    # --- Settings Update Methods ---
    def update_font_size(self, size):
        """Updates the font size and redraws."""
        self.font_size = size
        self._text_font.config(size=size)
        self.redraw_canvas() # Let redraw handle scheduling if needed

    def update_alpha(self, alpha):
        """Updates the overall window transparency."""
        self.overlay_alpha = alpha
        if self.root and self.root.winfo_exists():
            try:
                self.root.attributes('-alpha', alpha)
            except tk.TclError: pass # Ignore if window closing

    def update_text_color(self, color_name):
        """Updates the text color and redraws."""
        self.text_color = color_name.lower() # Store as "white" or "black"
        print(f"DEBUG: Overlay text color set to: {self.text_color}", flush=True) # Debug
        self.redraw_canvas() # Let redraw handle scheduling if needed

    def update_width_factor(self, factor):
         """Updates the overlay width factor, resizes window, and redraws."""
         self.width_factor = factor
         if self.root and self.root.winfo_exists():
              new_width = int(self.screen_width * self.width_factor)
              new_width = max(new_width, 50) # Min width
              try: # Wrap geometry update
                  current_height = self.root.winfo_height()
                  # Recalculate X based on new width to keep centered
                  new_x = (self.screen_width - new_width) // 2
                  current_y = self.root.winfo_y()
                  self.root.geometry(f"{new_width}x{current_height}+{new_x}+{current_y}")
                  self.redraw_canvas() # Let redraw handle scheduling if needed
              except tk.TclError as e:
                   if "invalid command name" not in str(e):
                        print(f"Warning: TclError setting geometry during width adjustment: {e}", flush=True)
              except Exception as e:
                  print(f"Error setting geometry during width adjustment: {e}", flush=True)


    def update_background_enable(self, enabled):
         """Updates whether the background rectangle is drawn."""
         self.enable_background = enabled
         if not self.supports_transparent_color and hasattr(self, 'canvas') and self.canvas:
              canvas_bg = self.background_color_hex if enabled else self.transparent_color
              try:
                   self.canvas.config(bg=canvas_bg)
              except tk.TclError: pass # Ignore if closing
         self.redraw_canvas() # Let redraw handle scheduling if needed


    # --- Drag and Drop Methods ---
    def start_move(self, event):
        """Records starting position for window dragging."""
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def stop_move(self, event):
        """Resets drag data when mouse button is released."""
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0

    def do_move(self, event):
        """Moves the window based on mouse drag."""
        if not self.root or not self.root.winfo_exists(): return
        new_x = event.x_root - self._drag_data["x"]
        new_y = event.y_root - self._drag_data["y"]
        try:
             self.root.geometry(f"+{new_x}+{new_y}")
        except tk.TclError: pass # Ignore if closing rapidly

    def close(self):
        """Destroys the overlay window."""
        if hasattr(self, 'root') and self.root and self.root.winfo_exists():
             print(f"Attempting to close overlay window {self.root}", flush=True) # Debug
             try:
                 self.root.destroy()
                 print(f"Overlay window destroyed.", flush=True) # Debug
             except tk.TclError as e:
                 if "invalid command name" not in str(e):
                      print(f"Info: TclError closing subtitle overlay: {e}", flush=True)
             except Exception as e:
                  print(f"Error during overlay close: {e}", flush=True)
        else:
             print("Overlay close called but window doesn't exist or already closed.", flush=True) # Debug
        # Ensure attributes are cleared even if destroy failed
        self.root = None
        self.canvas = None
        self._text_item = None # Clear item refs
        self._rect_item = None

# --- Tkinter GUI Application ---
class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Audio Translator v2.0")
        self.root.geometry("1100x500")

        # --- Dark Theme Configuration ---
        self.root.config(bg=DARK_BG)
        style = ttk.Style(self.root)
        try:
            available_themes = style.theme_names()
            if 'clam' in available_themes: style.theme_use('clam')
            elif sys.platform == "win32" and 'vista' in available_themes: style.theme_use('vista')
            elif sys.platform == "darwin" and 'aqua' in available_themes: style.theme_use('aqua')
            elif available_themes: style.theme_use(available_themes[0])
            else: print("WARNING: No ttk themes found.", flush=True)
            print(f"INFO: Using '{style.theme_use()}' ttk theme.", flush=True)
        except tk.TclError as e:
            print(f"Warning: Error setting ttk theme: {e}. Using default Tk widgets.", flush=True)

        # --- Style Definitions ---
        style.configure('.', background=DARK_BG, foreground=LIGHT_FG, font=('Segoe UI', 9), borderwidth=0, relief=tk.FLAT)
        style.configure('TFrame', background=DARK_BG)
        style.configure('TLabel', background=DARK_BG, foreground=LIGHT_FG)
        style.configure('TButton', background=BUTTON_BG, foreground=LIGHT_FG, padding=5, borderwidth=1, relief=tk.FLAT)
        style.map('TButton', background=[('active', BUTTON_ACTIVE_BG), ('disabled', '#444444')], foreground=[('disabled', '#999999')], relief=[('pressed', tk.SUNKEN), ('!pressed', tk.FLAT)])
        style.configure('TCombobox', fieldbackground=ENTRY_BG, background=BUTTON_BG, foreground=LIGHT_FG, arrowcolor=LIGHT_FG, selectbackground=ENTRY_BG, selectforeground=LIGHT_FG, insertcolor=TEXT_CURSOR, borderwidth=1, padding=(5, 3))
        style.map('TCombobox', fieldbackground=[('readonly', ENTRY_BG)], selectbackground=[('!focus', ENTRY_BG)], background=[('readonly', BUTTON_BG)], foreground=[('disabled', '#999999')])
        try:
            root.option_add('*TCombobox*Listbox.background', ENTRY_BG)
            root.option_add('*TCombobox*Listbox.foreground', LIGHT_FG)
            root.option_add('*TCombobox*Listbox.selectBackground', BUTTON_ACTIVE_BG)
            root.option_add('*TCombobox*Listbox.selectForeground', LIGHT_FG)
            root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 9))
            print("INFO: Applied dark theme options to Combobox Listbox.", flush=True)
        except Exception as e: print(f"Warning: Could not apply Combobox Listbox options: {e}", flush=True)
        style.configure('Vertical.TScrollbar', gripcount=0, background=SCROLLBAR_BG, troughcolor=SCROLLBAR_TROUGH, bordercolor=DARK_BG, arrowcolor=LIGHT_FG, relief=tk.FLAT, arrowsize=14)
        style.map('Vertical.TScrollbar', background=[('active', BUTTON_ACTIVE_BG)], arrowcolor=[('pressed', '#cccccc'), ('disabled', '#666666')])
        style.configure('Horizontal.TScale', background=DARK_BG, troughcolor=ENTRY_BG, sliderrelief=tk.FLAT, borderwidth=0)
        style.map('Horizontal.TScale', background=[('active', BUTTON_ACTIVE_BG)], troughcolor=[('active', ENTRY_BG)])
        style.configure('TCheckbutton', background=DARK_BG, foreground=LIGHT_FG, indicatorcolor=ENTRY_BG, padding=2)
        style.map('TCheckbutton', indicatorcolor=[('selected', LIGHT_FG), ('active', BUTTON_ACTIVE_BG)], background=[('active', DARK_BG)])

        # --- Window Attributes ---
        self.root.attributes('-topmost', True)
        self.main_alpha_var = tk.DoubleVar(value=MAIN_WINDOW_DEFAULT_ALPHA)
        self.root.attributes('-alpha', self.main_alpha_var.get())

        # --- App State Variables ---
        self.is_running = False
        self.active_threads = []
        self.message_deque = deque(maxlen=MAX_HISTORY_MESSAGES)
        self.enable_overlay = tk.BooleanVar(value=True)
        self.enable_logging = tk.BooleanVar(value=False)
        self.subtitle_overlay = None
        self.log_file_path = self._create_log_file()
        self._pending_close = False

        # --- Speech Engine Selection ---
        self.engine_type_var = tk.StringVar(value="whisper") # Hardcoded to whisper

        # --- Overlay Setting Variables ---
        self.overlay_font_size_var = tk.IntVar(value=DEFAULT_OVERLAY_FONT_SIZE)
        self.overlay_alpha_var = tk.DoubleVar(value=DEFAULT_OVERLAY_ALPHA)
        self.overlay_width_var = tk.DoubleVar(value=DEFAULT_OVERLAY_WIDTH_FACTOR)
        self.enable_overlay_background_var = tk.BooleanVar(value=DEFAULT_ENABLE_OVERLAY_BACKGROUND)
        self.overlay_font_color_var = tk.StringVar(value=DEFAULT_OVERLAY_TEXT_COLOR.capitalize())

        # --- Model and Language Variables ---
        self.source_lang_var = tk.StringVar()
        self.model_size_var = tk.StringVar(value=DEFAULT_MODEL_SIZE)

        # --- Create GUI Queue Processor ---
        self.queue_processor = create_queue_processor(
            self.root, 
            gui_queue, 
            self.add_output_direct, 
            self.update_status,
            self.update_overlay_text
        )

        # Build the GUI elements
        self._build_gui()

        # --- Window Close Handler ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) 

    def _build_gui(self):
        """Builds all GUI elements for the application."""
        # --- Top Control Frame ---
        top_control_frame = ttk.Frame(self.root, padding="10 5 10 5")
        top_control_frame.pack(fill=tk.X)

        # Audio device selection
        ttk.Label(top_control_frame, text="Input Device:").pack(side=tk.LEFT, padx=(0, 5))
        self.devices = get_audio_devices()
        if not self.devices:
            self.device_map = {"No Input Devices Found": None}
            messagebox.showerror("Audio Device Error", "No audio input devices found. Please check connections and drivers.")
        else:
            self.device_map = {f"{d['index']}: {d['name']} (Rate: {int(d.get('rate', 0))})": d['index'] for d in self.devices}
        
        self.device_combobox = ttk.Combobox(top_control_frame, values=list(self.device_map.keys()), state="readonly", width=35)
        if self.device_map and list(self.device_map.values())[0] is not None:
            self.device_combobox.current(0)
        self.device_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))

        self.start_button_state = tk.NORMAL if any(idx is not None for idx in self.device_map.values()) else tk.DISABLED

        # Source language selection
        ttk.Label(top_control_frame, text="Src Lang:").pack(side=tk.LEFT, padx=(0, 2))
        lang_options = list(SUPPORTED_SOURCE_LANGUAGES.keys())
        self.source_lang_combobox = ttk.Combobox(
            top_control_frame, textvariable=self.source_lang_var, 
            values=lang_options, state="readonly", width=10
        )
        if DEFAULT_SOURCE_LANG in lang_options: 
            self.source_lang_var.set(DEFAULT_SOURCE_LANG)
        elif lang_options: 
            self.source_lang_var.set(lang_options[0])
        self.source_lang_combobox.pack(side=tk.LEFT, padx=(0, 10))

        # Model size selection (for Whisper)
        ttk.Label(top_control_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 2))
        self.model_size_combobox = ttk.Combobox(
            top_control_frame, textvariable=self.model_size_var, 
            values=SUPPORTED_MODEL_SIZES, state="readonly", width=9
        )
        if DEFAULT_MODEL_SIZE in SUPPORTED_MODEL_SIZES: 
            self.model_size_var.set(DEFAULT_MODEL_SIZE)
        elif SUPPORTED_MODEL_SIZES: 
            self.model_size_var.set(SUPPORTED_MODEL_SIZES[0])
        self.model_size_combobox.pack(side=tk.LEFT, padx=(0, 10))

        # Start/Stop buttons
        self.start_button = ttk.Button(
            top_control_frame, text="Start", command=self.start_translation, 
            width=6, style='TButton', state=self.start_button_state
        )
        self.start_button.pack(side=tk.LEFT, padx=(5, 2))
        
        self.stop_button = ttk.Button(
            top_control_frame, text="Stop", command=self.stop_translation, 
            state=tk.DISABLED, width=6, style='TButton'
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))

        # --- Middle Control Frame ---
        mid_control_frame = ttk.Frame(self.root, padding="10 5 10 5")
        mid_control_frame.pack(fill=tk.X)

        # Font size control
        ttk.Label(mid_control_frame, text="Font Size:").pack(side=tk.LEFT, padx=(0, 2))
        self.font_size_spinbox = tk.Spinbox(
            mid_control_frame, from_=MIN_FONT_SIZE, to=MAX_FONT_SIZE,
            textvariable=self.overlay_font_size_var, width=4, command=self.update_font_size,
            state="readonly", bg=ENTRY_BG, fg=LIGHT_FG, buttonbackground=BUTTON_BG, relief=tk.FLAT,
            readonlybackground=ENTRY_BG, highlightthickness=1, highlightbackground=DARK_BG, highlightcolor=BUTTON_ACTIVE_BG,
            insertbackground=TEXT_CURSOR, buttoncursor="arrow", disabledbackground=ENTRY_BG, disabledforeground="#999999"
        )
        self.font_size_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        # Overlay width control
        ttk.Label(mid_control_frame, text="Overlay Width:").pack(side=tk.LEFT, padx=(0, 2))
        self.overlay_width_slider = ttk.Scale(
            mid_control_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
            variable=self.overlay_width_var, command=self.update_overlay_width,
            length=80, style='Horizontal.TScale'
        )
        self.overlay_width_slider.pack(side=tk.LEFT, padx=(0, 10))

        # Overlay alpha control
        ttk.Label(mid_control_frame, text="Overlay Alpha:").pack(side=tk.LEFT, padx=(0, 2))
        self.overlay_alpha_slider = ttk.Scale(
            mid_control_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
            variable=self.overlay_alpha_var, command=self.update_overlay_alpha,
            length=80, style='Horizontal.TScale'
        )
        self.overlay_alpha_slider.pack(side=tk.LEFT, padx=(0, 10))

        # Font color selection
        ttk.Label(mid_control_frame, text="Font Color:").pack(side=tk.LEFT, padx=(0, 2))
        self.overlay_font_color_combobox = ttk.Combobox(
            mid_control_frame, textvariable=self.overlay_font_color_var,
            values=["White", "Black"], state="readonly", width=6
        )
        self.overlay_font_color_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.overlay_font_color_combobox.bind("<<ComboboxSelected>>", self.update_overlay_font_color)

        # Main window alpha control
        ttk.Label(mid_control_frame, text="Main Alpha:").pack(side=tk.LEFT, padx=(5, 2))
        self.main_alpha_slider = ttk.Scale(
            mid_control_frame, from_=0.2, to=1.0, orient=tk.HORIZONTAL,
            variable=self.main_alpha_var, command=self.update_main_alpha,
            length=80, style='Horizontal.TScale'
        )
        self.main_alpha_slider.pack(side=tk.LEFT, padx=(0, 10))

        # Checkboxes
        checkbox_frame = ttk.Frame(mid_control_frame)
        checkbox_frame.pack(side=tk.LEFT, padx=(5, 0))

        self.overlay_checkbox = ttk.Checkbutton(
            checkbox_frame, text="Show Subtitles", variable=self.enable_overlay,
            command=self.toggle_overlay_visibility, style='TCheckbutton'
        )
        self.overlay_checkbox.pack(side=tk.TOP, anchor=tk.W, pady=(0, 2))

        self.overlay_bg_checkbox = ttk.Checkbutton(
            checkbox_frame, text="Show Background", variable=self.enable_overlay_background_var,
            command=self.update_overlay_background_enable, style='TCheckbutton'
        )
        self.overlay_bg_checkbox.pack(side=tk.TOP, anchor=tk.W, pady=(0, 2))

        self.logging_checkbox = ttk.Checkbutton(
            checkbox_frame, text="Save Log", variable=self.enable_logging, style='TCheckbutton'
        )
        self.logging_checkbox.pack(side=tk.TOP, anchor=tk.W)

        # --- Output Area ---
        output_frame = ttk.Frame(self.root, padding=(10, 5, 10, 10))
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, style="Vertical.TScrollbar")
        self.output_text = tk.Text(
            output_frame, wrap=tk.WORD, state=tk.DISABLED, height=10,
            bg=DARK_BG, fg=LIGHT_FG, insertbackground=TEXT_CURSOR,
            selectbackground=BUTTON_BG, selectforeground=LIGHT_FG,
            relief=tk.FLAT, borderwidth=0, highlightthickness=0,
            yscrollcommand=self.scrollbar.set,
            font=('Segoe UI', 10)
        )
        self.scrollbar.config(command=self.output_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.output_text.tag_config("error", foreground=ERROR_FG)
        self.output_text.tag_config("info", foreground=INFO_FG)
        self.output_text.tag_config("translation", foreground=LIGHT_FG)

        self._output_font = font.Font(font=self.output_text['font'])

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        initial_status = "Ready." if self.start_button_state == tk.NORMAL else "No input devices found."
        self.status_var.set(initial_status)
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.FLAT, 
            anchor=tk.W, padding=5, style='TLabel'
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Call enable_disable_overlay_controls once after initialization
        self.enable_disable_overlay_controls()
        
        # Set model visibility based on engine (now always Whisper)
        self.on_engine_change()

    def _create_log_file(self):
        """Creates a timestamped log file in a 'logs' subdirectory."""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"translation_log_{timestamp}.txt"
            return os.path.join(LOG_DIR, filename)
        except OSError as e:
            print(f"Error creating log directory/file: {e}", flush=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            return f"translation_log_{timestamp}_fallback.txt"

    # --- GUI Update Callbacks ---
    def update_main_alpha(self, value=None):
        """Updates the main window transparency."""
        if not self.root.winfo_exists(): return
        try:
            self.root.attributes('-alpha', self.main_alpha_var.get())
        except tk.TclError as e: print(f"Info: TclError updating main alpha (window might be closing): {e}", flush=True)
        except Exception as e: print(f"Error updating main alpha: {e}", flush=True)

    def update_font_size(self, value=None):
        """Updates font size for overlay."""
        if not self.root.winfo_exists(): return
        new_size = self.overlay_font_size_var.get()
        try:
            if self.subtitle_overlay:
                self.subtitle_overlay.update_font_size(new_size)
        except tk.TclError as e: print(f"Error updating font size (Tcl): {e}", flush=True)
        except Exception as e: print(f"Error updating font size: {e}", flush=True)

    def update_overlay_alpha(self, value=None):
        """Updates the subtitle overlay transparency."""
        if self.subtitle_overlay:
            self.subtitle_overlay.update_alpha(self.overlay_alpha_var.get())

    def update_overlay_font_color(self, event=None):
        """Updates the subtitle overlay font color."""
        selected_color = self.overlay_font_color_var.get()
        print(f"DEBUG: Combobox selected: {selected_color}", flush=True)
        if self.subtitle_overlay:
            self.subtitle_overlay.update_text_color(selected_color)

    def update_overlay_width(self, value=None):
        """Updates the subtitle overlay width."""
        if self.subtitle_overlay:
            self.subtitle_overlay.update_width_factor(self.overlay_width_var.get())

    def update_overlay_background_enable(self, value=None):
        """Updates the subtitle overlay background visibility."""
        if self.subtitle_overlay:
            self.subtitle_overlay.update_background_enable(self.enable_overlay_background_var.get())

    def on_engine_change(self, event=None):
        """Handle engine selection changes (simplified for Whisper only)."""
        # Model size combobox is always relevant for Whisper and should be editable
        self.model_size_combobox.config(state="readonly")

    def toggle_overlay_visibility(self):
        """Hides or shows the overlay window based on the 'Show Subtitles' checkbox."""
        enable = self.enable_overlay.get()
        if enable:
            if not self.subtitle_overlay:
                self.create_subtitle_overlay()
            elif self.subtitle_overlay and self.subtitle_overlay.root and not self.subtitle_overlay.root.winfo_viewable():
                try:
                    self.subtitle_overlay.root.deiconify()
                    if self.root.winfo_exists():
                        self.root.attributes('-topmost', False)
                except tk.TclError:
                    self.subtitle_overlay = None
                    self.create_subtitle_overlay()
        else:
            if self.subtitle_overlay and self.subtitle_overlay.root and self.subtitle_overlay.root.winfo_viewable():
                try:
                    self.subtitle_overlay.root.withdraw()
                    if self.root.winfo_exists():
                        self.root.attributes('-topmost', True)
                except tk.TclError:
                    print("Warning: TclError during overlay withdraw (already closed?).", flush=True)
                    self.subtitle_overlay = None

        self.enable_disable_overlay_controls()

    def enable_disable_overlay_controls(self):
        """Enables or disables overlay-specific controls based on visibility checkbox."""
        try:
            state = tk.NORMAL if self.enable_overlay.get() else tk.DISABLED
        except tk.TclError:
            state = tk.DISABLED
            
        widgets_to_toggle = [
            getattr(self, 'overlay_width_slider', None),
            getattr(self, 'overlay_alpha_slider', None),
            getattr(self, 'overlay_font_color_combobox', None),
            getattr(self, 'overlay_bg_checkbox', None)
        ]
        
        for widget in widgets_to_toggle:
            if widget:
                try: widget.config(state=state)
                except tk.TclError: pass

    # --- Output Handling ---
    def add_output_direct(self, text, tag=None):
        """Adds a line directly to the main output area."""
        if not self.root.winfo_exists(): return
        try:
            is_scrolled_to_bottom = self.output_text.yview()[1] >= 0.95

            self.output_text.config(state=tk.NORMAL)
            if tag:
                self.output_text.insert(tk.END, text + "\n", tag)
            else:
                self.output_text.insert(tk.END, text + "\n")
            self.output_text.config(state=tk.DISABLED)

            # Store in message deque for history
            self.message_deque.append((tag, text))

            if is_scrolled_to_bottom:
                self.output_text.see(tk.END)

            # Log translations if enabled
            if tag == "translation" and self.enable_logging.get():
                try:
                    clean_text = text
                    if text.startswith("[EN"):
                        clean_text = text.split("] ", 1)[-1].strip()
                        
                    with open(self.log_file_path, 'a', encoding='utf-8') as f:
                        f.write(clean_text + "\n")
                except Exception as e:
                    self.add_output_direct(f"[ERROR] Failed to write to log file: {e}", "error")
                    self.enable_logging.set(False)
                    if hasattr(self, 'logging_checkbox'):
                        try: self.logging_checkbox.config(state=tk.DISABLED)
                        except tk.TclError: pass

        except tk.TclError as e:
            if "invalid command name" not in str(e):
                print(f"Error adding text to GUI (Tcl): {e}", flush=True)
        except Exception as e:
            print(f"Error adding text to GUI: {e}", flush=True)

    def update_status(self, text):
        """Updates the status bar text."""
        try:
            self.status_var.set(text)
        except tk.TclError:
            pass  # Ignore if window is closing

    def update_overlay_text(self, text):
        """Updates the text in the subtitle overlay."""
        if self.enable_overlay.get() and self.subtitle_overlay:
            try:
                self.subtitle_overlay.update_text(text)
            except Exception as e:
                print(f"Error updating overlay text: {e}", flush=True)

    def create_subtitle_overlay(self):
        """Creates or recreates the subtitle overlay window instance."""
        if self.subtitle_overlay:
            print("Closing existing overlay before creating new one.", flush=True)
            try:
                if self.subtitle_overlay.root and self.subtitle_overlay.root.winfo_exists():
                    self.subtitle_overlay.close()
            except Exception as e: print(f"Info: Error closing previous overlay: {e}", flush=True)
            self.subtitle_overlay = None

        if not self.enable_overlay.get():
            print("Overlay creation skipped: 'Show Subtitles' is disabled.", flush=True)
            return

        try:
            print("Creating subtitle overlay...", flush=True)
            initial_settings = {
                'font_size': self.overlay_font_size_var.get(),
                'alpha': self.overlay_alpha_var.get(),
                'text_color': self.overlay_font_color_var.get(),
                'width_factor': self.overlay_width_var.get(),
                'enable_background': self.enable_overlay_background_var.get(),
                'bg_color': DEFAULT_OVERLAY_BG_COLOR,
                'font_family': DEFAULT_OVERLAY_FONT_FAMILY,
                'font_weight': DEFAULT_OVERLAY_FONT_WEIGHT,
            }
            self.subtitle_overlay = SubtitleOverlay(initial_settings)

            if self.subtitle_overlay and self.root.winfo_exists():
                if self.subtitle_overlay.root and self.subtitle_overlay.root.winfo_exists():
                    self.root.attributes('-topmost', False)
            print("Subtitle overlay created.", flush=True)

        except Exception as e:
            print(f"Error creating subtitle overlay: {e}", flush=True)
            messagebox.showerror("Overlay Error", f"Failed to create subtitle overlay:\n{e}")
            self.subtitle_overlay = None
            try:
                self.enable_overlay.set(False)
                self.enable_disable_overlay_controls()
            except tk.TclError: pass

    def _clear_queues(self, clear_gui=False):
        """Clears audio queue and optionally the GUI queue."""
        queues_to_clear = [(audio_queue, "audio")]
        if clear_gui:
            queues_to_clear.append((gui_queue, "gui"))

        cleared_counts = {"audio": 0, "gui": 0}
        for q, name in queues_to_clear:
            cleared_count = 0
            while True:
                try:
                    item = q.get_nowait()
                    cleared_count += 1
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Error getting item from {name}_queue during clear: {e}", flush=True)
                    break
            cleared_counts[name] = cleared_count
            print(f"Cleared approx {cleared_count} items from {name}_queue.", flush=True)

    def start_translation(self):
        """Starts the translation process in separate threads."""
        print("Attempting to start translation...", flush=True)
        if self.is_running:
            messagebox.showwarning("Already Running", "Translation is already in progress.")
            print("Start aborted: Already running.", flush=True)
            return

        # --- Get Settings ---
        selected_device_str = self.device_combobox.get()
        selected_source_lang_name = self.source_lang_var.get()
        selected_model_size = self.model_size_var.get()
        selected_engine = self.engine_type_var.get() # This will always be "whisper"

        # --- Validate Settings ---
        if not selected_device_str or self.device_map.get(selected_device_str) is None:
            messagebox.showerror("Input Error", "Please select a valid Input Device.")
            print("Start aborted: Invalid device selected.", flush=True)
            return
        device_index = self.device_map.get(selected_device_str)

        if selected_source_lang_name not in SUPPORTED_SOURCE_LANGUAGES:
            messagebox.showerror("Input Error", f"Invalid source language name: {selected_source_lang_name}")
            print("Start aborted: Invalid source language.", flush=True)
            return
        selected_source_lang_code = SUPPORTED_SOURCE_LANGUAGES.get(selected_source_lang_name)

        if selected_engine == "whisper" and (not selected_model_size or selected_model_size not in SUPPORTED_MODEL_SIZES):
            messagebox.showerror("Input Error", "Please select a valid model size.")
            print("Start aborted: Invalid model size.", flush=True)
            return

        # --- Prepare GUI and State ---
        try:
            self.is_running = True
            self._pending_close = False
            stop_event.clear()
            
            # Tell queue processor how many threads to expect
            self.queue_processor.set_expected_threads(2)

            # Clear previous output and messages
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete('1.0', tk.END)
            self.output_text.config(state=tk.DISABLED)
            self.message_deque.clear()

            # Update status bar and log start message
            engine_info = f"Engine: {selected_engine.capitalize()}"
            if selected_engine == "whisper":
                engine_info += f", Model: {selected_model_size}"
            self.status_var.set(f"Starting... {engine_info}")
            self.add_output_direct(f"--- Starting translation with {engine_info} ---", "info")

            # Disable controls that shouldn't be changed during run
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.device_combobox.config(state=tk.DISABLED)
            self.source_lang_combobox.config(state=tk.DISABLED)
            self.model_size_combobox.config(state=tk.DISABLED) # Model size not changeable during run
            self.overlay_checkbox.config(state=tk.DISABLED)
            self.overlay_bg_checkbox.config(state=tk.DISABLED)
            self.logging_checkbox.config(state=tk.DISABLED)

            self.active_threads = []

            # --- Create Overlay if Enabled ---
            self.create_subtitle_overlay()
            if self.subtitle_overlay:
                self.subtitle_overlay.clear_text()

        except tk.TclError as e:
            print(f"Error configuring GUI for start (TclError): {e}", flush=True)
            self.status_var.set("Error starting GUI.")
            self.is_running = False
            # Re-enable controls
            try:
                self.start_button.config(state=self.start_button_state)
                self.stop_button.config(state=tk.DISABLED)
                self.device_combobox.config(state="readonly")
                self.source_lang_combobox.config(state="readonly")
                self.model_size_combobox.config(state="readonly") # Model size is always readonly for Whisper when not running
                self.overlay_checkbox.config(state=tk.NORMAL)
                self.overlay_bg_checkbox.config(state=tk.NORMAL)
                self.logging_checkbox.config(state=tk.NORMAL)
            except tk.TclError: pass
            return
        except Exception as e:
            print(f"Error configuring UI for start: {e}", flush=True)
            self.status_var.set(f"Error starting: {e}")
            self.is_running = False
            return

        # --- Clear Queues Before Starting Threads ---
        print("Clearing queues before starting threads...", flush=True)
        self._clear_queues(clear_gui=True)
        print("Queues cleared.", flush=True)

        # --- Start Background Threads ---
        print("Starting background threads...", flush=True)
        try:
            capture_thread = threading.Thread(
                target=capture_audio_thread_gui,
                args=(device_index, audio_queue, gui_queue, stop_event),
                name="AudioCapture",
                daemon=True
            )
            
            transcribe_thread = threading.Thread(
                target=transcribe_translate_thread_gui,
                args=(audio_queue, gui_queue, stop_event, selected_engine, selected_source_lang_code, selected_model_size),
                name="TranscribeTranslate",
                daemon=True
            )

            self.active_threads.extend([capture_thread, transcribe_thread])
            capture_thread.start()
            transcribe_thread.start()

            # Start polling the GUI queue
            self.queue_processor.schedule_check(self._finalize_stop_state)
            print("Threads started, polling GUI queue.", flush=True)

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to start threads: {e}", flush=True)
            self.status_var.set(f"ERROR: Failed to start threads: {e}")
            self.add_output_direct(f"*** CRITICAL ERROR starting threads: {e} ***", "error")
            self.stop_translation(error=True)

    def stop_translation(self, error=False, graceful_stop=False):
        """Signals threads to stop and updates GUI to 'stopping' state."""
        if not self.root.winfo_exists():
            print("Root window is gone, skipping stop_translation.", flush=True)
            return

        # Check if already stopping or stopped
        try:
            stop_button_disabled = self.stop_button['state'] == tk.DISABLED
        except tk.TclError:
            print("Stop button TclError (likely destroyed), assuming stopped.", flush=True)
            stop_button_disabled = True

        if not self.is_running and stop_button_disabled:
            print("Stop called but already stopped/stopping.", flush=True)
            return

        call_source = "graceful stop" if graceful_stop else ("error stop" if error else "manual stop")
        print(f"Stop signal initiated ({call_source}). Setting stop event...", flush=True)

        # --- Signal Threads to Stop ---
        stop_event.set()
        self.is_running = False
        print("Stop event set. is_running set to False.", flush=True)

        # --- Update GUI (Minimal updates here) ---
        try:
            if self.root.winfo_exists():
                print("Updating GUI elements to stopping state...", flush=True)
                self.stop_button.config(state=tk.DISABLED)
                status = "Stopping..."
                log_status = status
                
                if error:
                    current_status = self.status_var.get()
                    status = current_status if "CRITICAL ERROR" in current_status else "Stopping due to CRITICAL ERROR."
                    log_status = "Processing stopped due to CRITICAL ERROR."
                elif graceful_stop:
                    status = "Finishing..."
                    log_status = "Processing finished."
                else:
                    log_status = "Processing stopped by user."

                self.status_var.set(status)
                self.add_output_direct(f"--- {log_status} ---", "error" if error else "info")
                print("GUI status set to stopping.", flush=True)
            else:
                print("GUI window closed during stop sequence.", flush=True)
                return

            print("Stop signal sent. Waiting for threads to finish.", flush=True)

        except tk.TclError as e:
            if "invalid command name" not in str(e):
                print(f"CRITICAL TCL ERROR during initial stop_translation phase: {e}", flush=True)
        except Exception as e:
            print(f"CRITICAL UNEXPECTED ERROR during initial stop_translation phase: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _finalize_stop_state(self):
        """Called after all threads signal completion."""
        print(f"Finalizing stop state. All threads reported finished.", flush=True)
        if not self.root.winfo_exists():
            print("Root window gone before finalizing stop state.", flush=True)
            return

        try:
            # Set final status only if it wasn't a critical error already shown
            current_status = self.status_var.get()
            status_message = "Processing stopped."
            
            if "CRITICAL ERROR" not in current_status:
                self.status_var.set(status_message)
            else:
                status_message = current_status

            # Re-enable controls fully
            self.start_button.config(state=self.start_button_state)
            self.device_combobox.config(state="readonly")
            self.source_lang_combobox.config(state="readonly")
            self.model_size_combobox.config(state="readonly") # Always readonly for Whisper when not running
            self.overlay_checkbox.config(state=tk.NORMAL)
            self.overlay_bg_checkbox.config(state=tk.NORMAL)
            
            log_state = tk.NORMAL
            try:
                if self.logging_checkbox['state'] == tk.DISABLED and "Failed to write to log file" in status_message:
                    log_state = tk.DISABLED
            except Exception: pass
            self.logging_checkbox.config(state=log_state)

            self.enable_disable_overlay_controls()

            # Close Overlay now that threads are done
            if self.subtitle_overlay:
                print("Closing subtitle overlay during finalization...", flush=True)
                try:
                    self.subtitle_overlay.close()
                except Exception as e: print(f"Error closing overlay in finalize: {e}", flush=True)
                self.subtitle_overlay = None
                print("Subtitle overlay closed.", flush=True)

            # Restore Main Window Topmost
            if self.root.winfo_exists():
                self.root.attributes('-topmost', True)
                print("Main window topmost attribute reset.", flush=True)

            # Clear active threads list
            self.active_threads = []
            print("Final stop state finalized.", flush=True)

            # Check if a close was requested
            if getattr(self, '_pending_close', False):
                print("Pending close detected after finalization. Destroying root.", flush=True)
                self.root.after_idle(self._destroy_root)
                self._pending_close = False

            return False  # Tell queue processor to stop checking

        except tk.TclError as e:
            if "invalid command name" not in str(e):
                print(f"TCL ERROR during _finalize_stop_state: {e}", flush=True)
            return False
        except Exception as e:
            print(f"UNEXPECTED ERROR during _finalize_stop_state: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def on_closing(self):
        """Handles the event when the user clicks the window's close button."""
        print("Close window requested ('X' button).", flush=True)
        if self.is_running:
            if messagebox.askokcancel("Quit", "Translation is running. Stop and quit?"):
                print("User confirmed quit while running. Initiating stop sequence...", flush=True)
                self._pending_close = True
                self.stop_translation(error=False, graceful_stop=False)
                print("Stop initiated. Window will close after threads finish.", flush=True)
            else:
                print("Quit cancelled by user.", flush=True)
                self._pending_close = False
                return
        else:
            print("Not running, destroying root window immediately.", flush=True)
            self._pending_close = False
            self._destroy_root()

    def _destroy_root(self):
        """Safely destroys the main Tkinter window and ensures stop event is set."""
        print("Attempting to destroy root window...", flush=True)
        stop_event.set()

        # Cancel pending queue check job
        if hasattr(self, 'queue_processor'):
            try:
                self.queue_processor.cancel_check()
                print("Cancelled queue processor check.", flush=True)
            except Exception as e:
                print(f"Error cancelling queue processor: {e}", flush=True)

        # Close overlay first if it exists
        if self.subtitle_overlay:
            print("Closing overlay during destroy sequence...", flush=True)
            try:
                if self.subtitle_overlay.root and self.subtitle_overlay.root.winfo_exists():
                    self.subtitle_overlay.close()
                self.subtitle_overlay = None
            except Exception as e:
                print(f"Error closing overlay during destroy: {e}", flush=True)

        # Destroy main window
        if self.root and self.root.winfo_exists():
            try:
                self.root.destroy()
                print("Root window destroyed successfully.", flush=True)
            except tk.TclError as e:
                if "invalid command name" not in str(e):
                    print(f"TclError during root destroy: {e}", flush=True)
            except Exception as e:
                print(f"Unexpected error during root destroy: {e}", flush=True)
        else:
            print("Root window does not exist or already destroyed.", flush=True) 

# --- Main Execution Block ---
if __name__ == "__main__":
    # Add flush=True to all print statements in this block for better debugging
    print("Starting application...", flush=True)
    
    # --- Pre-checks ---
    try:
        print("Checking PyAudio...", flush=True)
        p_test = pyaudio.PyAudio()
        devices = get_audio_devices()
        p_test.terminate()
        print(f"PyAudio check OK. Found {len(devices)} input devices.", flush=True)
        if not devices:
            print("WARNING: No audio input devices found. Application might not function.", flush=True)
            try:
                root_warn = tk.Tk(); root_warn.withdraw()
                messagebox.showwarning("Audio Warning", "No audio input devices found.\nPlease check your microphone connections and drivers.")
                root_warn.destroy()
            except Exception: pass

    except Exception as e:
        print(f"CRITICAL PYAUDIO ERROR: {e}", file=sys.stderr, flush=True)
        print("Please ensure PortAudio is installed and audio drivers are working.", file=sys.stderr, flush=True)
        try:
            root_err = tk.Tk(); root_err.withdraw()
            messagebox.showerror("PyAudio Error", f"PyAudio initialization failed: {e}\n\nPlease ensure PortAudio is installed and audio drivers are working.\nThe application cannot function without audio input.")
            root_err.destroy()
        except Exception: pass
        sys.exit(1)

    # --- Check Torch/CUDA ---
    try:
        print(f"Checking PyTorch... Version: {torch.__version__}", flush=True)
        if torch.cuda.is_available():
            print(f"CUDA available. Device: {torch.cuda.get_device_name(0)}", flush=True)
            try:
                tensor = torch.rand(3, 3).to('cuda')
                print("CUDA check OK (created tensor on GPU).", flush=True)
                del tensor
                torch.cuda.empty_cache()
            except Exception as cuda_err:
                print(f"WARNING: CUDA detected but test failed: {cuda_err}", flush=True)
                print("Falling back to CPU for Whisper.", flush=True)
        else:
            print("CUDA not available. Using CPU.", flush=True)
    except Exception as torch_err:
        print(f"ERROR checking PyTorch/CUDA: {torch_err}", flush=True)
        print("Proceeding with CPU fallback if possible.", flush=True)

    # --- Create directories if needed ---
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Start the main application ---
    root = None
    app = None
    try:
        print("Initializing Tkinter root...", flush=True)
        root = tk.Tk()
        root.withdraw()  # Hide the main window initially
        print("Tk root created.", flush=True)

        # --- START OF LOADING SCREEN CODE ---
        loading_window = tk.Toplevel(root)
        loading_window.overrideredirect(True)  # Remove window decorations
        
        try:
            loading_window.config(bg=DARK_BG)
        except NameError:  # Fallback if DARK_BG isn't defined yet
            loading_window.config(bg="black")

        try:
            # Get the directory where the script itself is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Create the full path to the image file
            icon_path = os.path.join(script_dir, "loading_icon.png")
            print(f"Attempting to load icon from: {icon_path}", flush=True)

            # Load the image using the full path
            img_pil = Image.open(icon_path)
            img_tk = ImageTk.PhotoImage(img_pil)
            loading_label = tk.Label(loading_window, image=img_tk, bd=0)
            loading_label.image = img_tk  # Keep a reference! Important!
            loading_label.pack(pady=20, padx=20)

            # Add a "Loading..." text
            try:
                loading_text = tk.Label(loading_window, text="Loading...", fg=LIGHT_FG, bg=DARK_BG, font=('Segoe UI', 10))
                loading_text.pack(pady=(0, 20))
            except NameError:  # Fallback colors/font
                loading_text = tk.Label(loading_window, text="Loading...", fg="white", bg="black", font=('Arial', 10))
                loading_text.pack(pady=(0, 20))

        except FileNotFoundError:
            print("ERROR: loading_icon.png not found. Using text fallback.", flush=True)
            # Fallback text if image fails to load
            try:
                fallback_label = tk.Label(loading_window, text="Loading...", fg=LIGHT_FG, bg=DARK_BG, font=('Segoe UI', 14))
                fallback_label.pack(pady=40, padx=60)
            except NameError:
                fallback_label = tk.Label(loading_window, text="Loading...", fg="white", bg="black", font=('Arial', 14))
                fallback_label.pack(pady=40, padx=60)
        except Exception as e:
            print(f"ERROR loading image or creating loading screen: {e}", flush=True)
            # Fallback text on other errors
            try:
                fallback_label = tk.Label(loading_window, text="Loading...", fg=LIGHT_FG, bg=DARK_BG, font=('Segoe UI', 14))
                fallback_label.pack(pady=40, padx=60)
            except NameError:
                fallback_label = tk.Label(loading_window, text="Loading...", fg="white", bg="black", font=('Arial', 14))
                fallback_label.pack(pady=40, padx=60)

        # Center the loading window
        loading_window.update_idletasks()  # Ensure window size is calculated
        width = loading_window.winfo_width()
        height = loading_window.winfo_height()
        screen_width = loading_window.winfo_screenwidth()
        screen_height = loading_window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        # Ensure minimum dimensions just in case image loading failed badly
        width = max(width, 100)
        height = max(height, 50)
        loading_window.geometry(f'{width}x{height}+{x}+{y}')
        loading_window.lift()  # Bring loading window to the front
        # --- END OF LOADING SCREEN CODE ---

        print("Initializing TranslatorApp...", flush=True)
        app = TranslatorApp(root)
        print("TranslatorApp initialized.", flush=True)

        # --- Destroy Loading Screen & Show Main Window ---
        loading_window.destroy()  # Close the loading screen
        root.deiconify()  # Show the main window now
        root.lift()  # Ensure main window is frontmost
        root.focus_force()  # Give focus to main window

        print("Entering Tk mainloop...", flush=True)
        root.mainloop()
        # --- Mainloop finished ---
        print("Exited Tk mainloop normally.", flush=True)

    except tk.TclError as e:
        print(f"CRITICAL TKINTER ERROR during initialization or mainloop: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        try:
            # Try closing loading window if it still exists on error
            if 'loading_window' in locals() and loading_window.winfo_exists(): loading_window.destroy()
            root_final_err = tk.Tk(); root_final_err.withdraw()
            messagebox.showerror("GUI Error", f"A critical Tkinter error occurred:\n{e}\n\nSee console for details.")
            root_final_err.destroy()
        except Exception: pass
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL UNEXPECTED ERROR during GUI setup or mainloop: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        try:
            # Try closing loading window if it still exists on error
            if 'loading_window' in locals() and loading_window.winfo_exists(): loading_window.destroy()
            root_final_err = tk.Tk(); root_final_err.withdraw()
            messagebox.showerror("Application Error", f"An unexpected critical error occurred:\n{e}\n\nSee console for details.")
            root_final_err.destroy()
        except Exception: pass
        sys.exit(1)
    finally:
        # --- Final Cleanup ---
        print("Application exiting. Setting stop event...", flush=True)
        stop_event.set()  # Ensure stop event is set on any exit path

        # Explicitly attempt to close overlay if app object exists and it wasn't closed already
        if app and hasattr(app, 'subtitle_overlay') and app.subtitle_overlay:
            print("Performing final overlay close check...", flush=True)
            try:
                if app.subtitle_overlay.root and app.subtitle_overlay.root.winfo_exists():
                    app.subtitle_overlay.close()
            except Exception as final_close_err: print(f"Error during final overlay close: {final_close_err}", flush=True)

        print("Application finished.", flush=True) 