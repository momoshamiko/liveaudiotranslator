import os
import sys # Added for sys.frozen and sys._MEIPASS
import time
import torch
import numpy as np
import re
import warnings

# Configure warning filters for PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", message=".*GPU available but not used.*")
warnings.filterwarnings("ignore", message=".*Triggered internally.*")
warnings.filterwarnings("ignore", message=".*Could not infer.*")

# Make noisereduce optional
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    print("Warning: noisereduce module not found. Noise reduction will be disabled.")
    HAS_NOISEREDUCE = False

# Make faster_whisper optional
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    print("Warning: faster_whisper module not found. Whisper engine will be unavailable.")
    HAS_WHISPER = False

from abc import ABC, abstractmethod
# import json # No longer used
# import queue # No longer used

# Configure dotenv for environment variables (REMOVED - dotenv is no longer used)
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
# print("python-dotenv not installed. API keys must be set in environment variables.")

# Import config
from config import (
    WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, VAD_MIN_SILENCE_DURATION_MS,
    VAD_SPEECH_PAD_MS, AUDIO_SAMPLERATE
)

# Sentence completion/boundary detection utilities
class SentenceProcessor:
    """Handles sentence boundary detection and completion"""
    
    def __init__(self):
        self.incomplete_sentence = ""
        self.punctuation_endings = '.!?'
        self.conjunction_words = ['and', 'but', 'or', 'so', 'because', 'if', 'when', 'while', 'though', 'although', 'however', 'therefore', 'thus']
        self.question_starters = ['what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'will', 'would', 'can', 'could', 'should', 'did', 'do', 'does']
        self.question_indicators = ['?', ' right', ' ok', ' okay', ' yeah', ' correct', ' isn\'t it', ' weren\'t they', ' don\'t you', ' doesn\'t it']
        self.filler_words = ['um', 'uh', 'ah', 'er', 'like', 'you know']
        
    def _clean_text(self, text):
        """Clean up text by removing filler words and extra whitespace"""
        # Replace multiple spaces with a single space
        cleaned = ' '.join(text.split())
        
        # Remove filler words at the beginning
        for filler in self.filler_words:
            pattern = f'^{filler} '
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove filler words in the middle surrounded by spaces
        for filler in self.filler_words:
            pattern = f' {filler} '
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
            
        return cleaned.strip()
    
    def _detect_question(self, text):
        """Checks if text is likely a question"""
        # If it already ends with a question mark, it's a question
        if text.rstrip().endswith('?'):
            return True
            
        # Check for question starters at the beginning of the sentence
        words = text.lower().split()
        if words and words[0] in self.question_starters:
            return True
            
        # Check for question indicators at the end
        for indicator in self.question_indicators:
            if text.lower().endswith(indicator):
                return True
        
        return False
    
    def process_text(self, text):
        """Process text to improve sentence boundaries"""
        # Clean up the text first
        text = self._clean_text(text)
        
        # Combine with any incomplete sentence from before
        combined_text = self.incomplete_sentence + " " + text if self.incomplete_sentence else text
        combined_text = combined_text.strip()
        
        # If empty, nothing to do
        if not combined_text:
            return ""
            
        # Check if the text ends with punctuation
        if combined_text[-1] in self.punctuation_endings:
            self.incomplete_sentence = ""
            return combined_text
            
        # Split into words to check ending
        words = combined_text.split()
        if not words:
            return ""
            
        # Check if the last word is a conjunction (suggesting incomplete sentence)
        if words[-1].lower() in self.conjunction_words:
            self.incomplete_sentence = combined_text
            return ""
        
        # If it's an obvious incomplete phrase under 4 words, save it for the next chunk
        if len(words) < 4 and not any(combined_text[-1] == p for p in self.punctuation_endings):
            self.incomplete_sentence = combined_text
            return ""
            
        # If it seems like a complete sentence but doesn't end with punctuation,
        # add a period or question mark to make it look better
        self.incomplete_sentence = ""
        if not any(combined_text[-1] == p for p in self.punctuation_endings):
            if self._detect_question(combined_text):
                combined_text += "?"
            else:
                combined_text += "."
            
        return combined_text

# Base class for all speech recognition engines
class SpeechEngine(ABC):
    def __init__(self, source_language):
        self.source_language = source_language
        self.sentence_processor = SentenceProcessor()
        
    @abstractmethod
    def initialize(self):
        """Initialize the speech recognition engine"""
        pass
        
    @abstractmethod
    def process_audio(self, audio_data):
        """Process audio data and return transcription/translation"""
        pass
        
    def preprocess_audio(self, audio_data):
        """Apply common preprocessing to audio data"""
        # Convert bytes to float32 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Apply noise reduction if the chunk has sufficient data
        if len(audio_np) > 0:
            # For basic audio processing that doesn't require noisereduce
            
            # 1. Simple normalization - ensure consistent volume
            if np.abs(audio_np).max() > 0:
                normalized = audio_np / np.abs(audio_np).max() * 0.9
                # Only apply if it doesn't reduce the volume
                if np.abs(normalized).mean() >= np.abs(audio_np).mean() * 0.7:
                    audio_np = normalized
            
            # 2. DC offset removal (center the waveform around zero)
            audio_np = audio_np - np.mean(audio_np)
            
            # 3. Apply advanced noise reduction if the library is available
            if HAS_NOISEREDUCE:
                try:
                    energy = np.sqrt(np.mean(audio_np**2))
                    # Only apply noise reduction if there's enough signal
                    if energy > 0.005:  # Threshold for "non-silent" audio
                        # Get a noise profile from the quietest part of the audio
                        # (We assume the first 0.1 seconds might contain background noise)
                        samples_per_100ms = int(AUDIO_SAMPLERATE * 0.1)
                        if len(audio_np) > samples_per_100ms:
                            # Find the quietest segment to use as noise profile
                            chunk_size = samples_per_100ms
                            lowest_energy = float('inf')
                            noise_sample = audio_np[:chunk_size]
                            
                            for i in range(0, len(audio_np) - chunk_size, chunk_size // 2):
                                chunk = audio_np[i:i+chunk_size]
                                chunk_energy = np.sqrt(np.mean(chunk**2))
                                if chunk_energy < lowest_energy and chunk_energy > 0:
                                    lowest_energy = chunk_energy
                                    noise_sample = chunk
                            
                            # Apply noise reduction with adaptive parameters
                            if energy > 0.02:  # For louder audio, use more aggressive settings
                                audio_np = nr.reduce_noise(
                                    y=audio_np,
                                    sr=AUDIO_SAMPLERATE,
                                    stationary=False,
                                    prop_decrease=0.75,
                                    n_fft=1024
                                )
                            else:  # For quieter audio, use gentler settings
                                audio_np = nr.reduce_noise(
                                    y=audio_np,
                                    sr=AUDIO_SAMPLERATE,
                                    stationary=True,
                                    prop_decrease=0.5,
                                    n_fft=1024
                                )
                        else:
                            # For very short audio, use basic noise reduction
                            audio_np = nr.reduce_noise(
                                y=audio_np,
                                sr=AUDIO_SAMPLERATE,
                                stationary=False
                            )
                except Exception as e:
                    print(f"Warning: Advanced noise reduction failed: {e}")
                    # Fall back to basic noise reduction
                    try:
                        audio_np = nr.reduce_noise(
                            y=audio_np,
                            sr=AUDIO_SAMPLERATE,
                            stationary=False
                        )
                    except Exception as e2:
                        print(f"Warning: Basic noise reduction also failed: {e2}")
                
        return audio_np
        
    def cleanup(self):
        """Clean up resources (if needed)"""
        pass


class WhisperEngine(SpeechEngine):
    def __init__(self, source_language, model_size):
        super().__init__(source_language)
        self.model_size = model_size
        self.model = None
        
    def initialize(self):
        """Initialize Whisper model"""
        if not HAS_WHISPER:
            return False, "Whisper engine is not available. Please install faster-whisper package."
        
        try:
            model_path_or_name = self.model_size
            
            # Check if running in a PyInstaller bundle and if a bundled model exists
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                # Define the expected base directory for bundled models within the EXE's temporary space
                # This target path in 'datas' should match: ('path/to/local/model_dir', 'bundled_models/model_name')
                bundled_model_dir = os.path.join(sys._MEIPASS, 'bundled_models', self.model_size)
                if os.path.exists(bundled_model_dir):
                    model_path_or_name = bundled_model_dir
                    print(f"INFO: Using bundled Whisper model from: {model_path_or_name}", flush=True)
                else:
                    print(f"WARNING: Bundled model directory for '{self.model_size}' not found at '{bundled_model_dir}'. Attempting to use default cache/download.", flush=True)
            
            self.model = WhisperModel(
                model_path_or_name, 
                device=WHISPER_DEVICE, 
                compute_type=WHISPER_COMPUTE_TYPE
            )
            # You might need to explicitly download the model here if not bundled and not in cache for the first run
            # This is usually handled by faster-whisper itself, but something to keep in mind.
            return True, f"Whisper model '{model_path_or_name if isinstance(model_path_or_name, str) else self.model_size}' initialized on {WHISPER_DEVICE}"
        except Exception as e:
            # Print more detailed error for debugging PyInstaller issues
            import traceback
            print(f"ERROR: Failed to initialize Whisper model. Path/Name used: {model_path_or_name}", flush=True)
            print(traceback.format_exc(), flush=True)
            return False, f"Failed to initialize Whisper model: {e}"
    
    def process_audio(self, audio_data):
        """Process audio data through Whisper"""
        if not self.model:
            return None, 0, "Error: Model not initialized", None
            
        try:
            start_time = time.time()
            
            # Preprocess audio
            audio_np = self.preprocess_audio(audio_data)
            
            if audio_np.size == 0:
                return None, 0, "Empty audio chunk", None
                
            # Improved VAD parameters
            vad_params = dict(
                min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
                speech_pad_ms=VAD_SPEECH_PAD_MS
            )
                
            # Perform transcription and translation
            segments, info = self.model.transcribe(
                audio_np,
                language=self.source_language,  # Can be None for auto-detect
                beam_size=5,
                vad_filter=False,  # Changed from True to False to disable VAD
                vad_parameters=vad_params, # These params will be ignored if vad_filter is False
                task='translate'  # Force translation to English
            )
            
            # Combine segments into a single translation string
            full_translation = " ".join(segment.text for segment in segments).strip()
            
            # Apply sentence processing
            processed_translation = self.sentence_processor.process_text(full_translation)
            
            latency = time.time() - start_time
            
            # Get language detection info
            detected_lang = info.language
            detected_prob = info.language_probability
            
            # Clear CUDA cache to free memory
            if WHISPER_DEVICE == "cuda":
                torch.cuda.empty_cache()
                
            return processed_translation, latency, detected_lang, detected_prob
            
        except Exception as e:
            return None, 0, f"Error processing with Whisper: {e}", None
    
    def cleanup(self):
        """Free Whisper model resources"""
        if self.model:
            # Let Python's garbage collector handle it
            self.model = None
            # Explicitly clear CUDA cache
            if WHISPER_DEVICE == "cuda":
                torch.cuda.empty_cache()


def create_engine(engine_type, source_language, model_size=None):
    """Creates the specified speech recognition engine."""
    # Simplified: only Whisper engine is supported
    if engine_type == "whisper":
        if not HAS_WHISPER:
            print("ERROR: Whisper (faster_whisper) is not installed but is the selected engine.")
            # We might want to raise an error or handle this more gracefully
            # For now, returning None will likely cause a crash later, which signals a problem.
            return None 
        return WhisperEngine(source_language, model_size)
    else:
        # This case should ideally not be reached if GUI restricts engine choice
        print(f"Error: Unsupported engine type '{engine_type}'. Defaulting to Whisper if available.")
        if HAS_WHISPER:
            return WhisperEngine(source_language, model_size)
    return None # Fallback if Whisper isn't available either 