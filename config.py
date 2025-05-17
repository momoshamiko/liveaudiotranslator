import torch
import os

# --- Engine Selection ---
# Options: "whisper"
DEFAULT_ENGINE = "whisper"

# --- Audio Settings ---
AUDIO_SAMPLERATE = 16000  # Whisper requires 16kHz
AUDIO_CHANNELS = 1  # Mono
AUDIO_BUFFER_SECONDS = 1.5  # Reduced from 3 seconds for more responsive transcription
AUDIO_OVERLAP_SECONDS = 0.5  # Overlap between audio chunks to avoid missing words
PYAUDIO_FRAMES_PER_BUFFER = 1024

# --- Whisper Model Settings ---
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "float32"
SUPPORTED_MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEFAULT_MODEL_SIZE = "large-v3"

# --- VAD (Voice Activity Detection) Settings ---
VAD_MIN_SILENCE_DURATION_MS = 300  # Reduced from 500ms for better sentence detection
VAD_SPEECH_PAD_MS = 400  # Padding around speech segments

# --- Source Language Configuration ---
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

# --- GUI Update Settings ---
GUI_UPDATE_INTERVAL = 0.2  # seconds between GUI updates (to prevent lag)

# --- Appearance & History Settings ---
MAX_HISTORY_MESSAGES = 15  # Number of messages to keep in history
MAIN_WINDOW_DEFAULT_ALPHA = 1.00  # Main window transparency

# --- Overlay Defaults ---
DEFAULT_OVERLAY_ALPHA = 0.75
DEFAULT_OVERLAY_FONT_SIZE = 40
DEFAULT_OVERLAY_WIDTH_FACTOR = 1.0
DEFAULT_OVERLAY_BG_COLOR = "black"
DEFAULT_OVERLAY_TEXT_COLOR = "white"
DEFAULT_OVERLAY_FONT_FAMILY = "Segoe UI"
DEFAULT_OVERLAY_FONT_WEIGHT = "bold"
DEFAULT_ENABLE_OVERLAY_BACKGROUND = True

# --- Font Size Limits ---
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 72

# --- Color Constants for Dark Theme (Main Window) ---
DARK_BG = '#2b2b2b'
LIGHT_FG = '#ffffff'
ENTRY_BG = '#3c3f41'
BUTTON_BG = '#555555'
BUTTON_ACTIVE_BG = '#666666'
TEXT_CURSOR = '#ffffff'
INFO_FG = '#6cace4'
ERROR_FG = '#e46c6c'
SCROLLBAR_BG = '#454545'
SCROLLBAR_TROUGH = '#333333'

# --- API Keys (for cloud services) ---
# The following keys are no longer used as only Whisper engine is supported.
# If you were using a .env file, it's no longer actively loaded by this application.

# --- Default Paths ---
LOG_DIR = "logs"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models") 