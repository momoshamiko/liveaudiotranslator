import queue
import re
import threading
import time
import warnings
import torch

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*Triggered internally.*")

from speech_engines import create_engine

# --- Regex Ignore Patterns ---
# These are patterns of text that should be ignored in the output
IGNORE_PATTERNS = [
    re.compile(r"(?:thank\s*(?:you|s)\s*(?:so|very|a\s*lot)?\s*(?:much)?\s*)?(?:for\s+(?:watching|viewing|your\s+viewing)).*?(?:in\s+this\s+video)?", re.IGNORECASE),
    re.compile(r"see\s+you(?:\s+all|\s+again)?\s+(?:next\s+(?:time|video)|later|in\s+the\s+next\s+video)", re.IGNORECASE),
    re.compile(r"subscribe\s+to\s+(?:my|the)\s+channel", re.IGNORECASE),
    # Add more patterns here as needed
]

def transcribe_translate_thread_gui(audio_q, gui_q, stop_event_flag, engine_type, source_language_code, model_size=None):
    """Transcribes audio using selected engine and translates to English."""
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Initializing {engine_type} engine...")
    
    # Create the appropriate engine
    engine = None
    try:
        engine = create_engine(engine_type, source_language_code, model_size)
        success, message = engine.initialize()
        
        if not success:
            gui_q.put(f"[{thread_name}] CRITICAL ERROR: {message}")
            return
            
        gui_q.put(f"__INIT_STATUS__{message}. Ready for translation.")
        
        # Main processing loop
        while not stop_event_flag.is_set():
            try:
                # Check stop event *before* getting from queue
                if stop_event_flag.is_set(): break
                
                # Get audio data from queue with timeout
                audio_data_bytes = audio_q.get(timeout=0.5)
                
                # Check for None signal (end of audio capture)
                if audio_data_bytes is None:
                    gui_q.put(f"[{thread_name}] Received None from audio queue, stopping.")
                    break  # Exit loop cleanly
                    
            except queue.Empty:
                # Timeout occurred, loop again to check stop_event
                continue
                
            # Check stop event *after* getting data/timeout
            if stop_event_flag.is_set(): break
            
            # Process the audio data
            try:
                # Use the engine to process the audio
                result, latency, detected_info, confidence = engine.process_audio(audio_data_bytes)
                
                # Check if we got a valid result
                if result:
                    # Apply ignore patterns
                    should_ignore = False
                    for pattern in IGNORE_PATTERNS:
                        if pattern.search(result):
                            should_ignore = True
                            print(f"Ignoring phrase matching pattern '{pattern.pattern}': '{result}'", flush=True)
                            break
                            
                    # Send valid, non-ignored translation to GUI
                    if result and not should_ignore:
                        if isinstance(detected_info, str):
                            detected_lang = detected_info
                            detected_prob = confidence if confidence is not None else 0.0
                        else:
                            detected_lang = "unknown"
                            detected_prob = 0.0
                            
                        gui_message = f"[EN ({latency:.2f}s, src={detected_lang}:{detected_prob:.2f})] {result}"
                        gui_q.put(gui_message)
                
                # If the result is None but we have an error message
                elif detected_info and isinstance(detected_info, str) and \
                     not detected_info.startswith("Empty") and \
                     not detected_info.startswith("Partial") and \
                     detected_info.lower() != "en" and \
                     detected_info.lower() != "ja":
                    gui_q.put(f"[{thread_name}] WARNING: {detected_info}")
                    
            except Exception as e:
                # Check stop event even on error
                if stop_event_flag.is_set(): break
                
                gui_q.put(f"[{thread_name}] ERROR during translation: {e}")
                
        # End of while loop (either stopped or audio queue signaled done)
        if stop_event_flag.is_set():
            gui_q.put(f"[{thread_name}] Stop event was set, exiting transcription loop.")
            
    except Exception as e:
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Failed to initialize/run {engine_type} engine: {e}")
    finally:
        print(f"[{thread_name}] Entering finally block...", flush=True)
        
        # Clean up resources
        if engine is not None:
            try:
                gui_q.put(f"[{thread_name}] Releasing engine resources...")
                engine.cleanup()
                gui_q.put(f"[{thread_name}] Engine resources released.")
            except Exception as e:
                gui_q.put(f"[{thread_name}] Warning: Error during engine cleanup: {e}")
                
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                print(f"[{thread_name}] CUDA memory cache cleared.", flush=True)
            except Exception as e:
                print(f"[{thread_name}] Warning: Error clearing CUDA cache: {e}", flush=True)
                
        gui_q.put(f"[{thread_name}] Transcription/Translation thread finished.")
        gui_q.put(None)  # Signal GUI that this processing pipeline is done
        print(f"[{thread_name}] Exiting finally block.", flush=True) 