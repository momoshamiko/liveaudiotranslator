import pyaudio
import numpy as np
import queue
import threading
import time
from config import (
    AUDIO_SAMPLERATE, AUDIO_CHANNELS, AUDIO_BUFFER_SECONDS,
    AUDIO_OVERLAP_SECONDS, PYAUDIO_FRAMES_PER_BUFFER
)

# Constants
AUDIO_FORMAT = pyaudio.paInt16  # PyAudio format constant

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
        print(f"Error getting audio devices: {e}", flush=True)
    finally:
        p.terminate()
    return devices

def calculate_audio_buffer_size():
    """Calculate the size of the audio buffer in bytes."""
    # 2 bytes per sample for paInt16
    return int(AUDIO_SAMPLERATE * AUDIO_CHANNELS * 2 * AUDIO_BUFFER_SECONDS)

def calculate_overlap_size():
    """Calculate the size of the overlap buffer in bytes."""
    # 2 bytes per sample for paInt16
    return int(AUDIO_SAMPLERATE * AUDIO_CHANNELS * 2 * AUDIO_OVERLAP_SECONDS)

def capture_audio_thread_gui(device_index, audio_q, gui_q, stop_event_flag):
    """Improved audio capture thread with overlapping buffers."""
    thread_name = threading.current_thread().name
    gui_q.put(f"[{thread_name}] Starting audio capture from device index {device_index}...")
    
    # Main audio buffer
    audio_buffer = bytearray()
    # Buffer for keeping overlap between chunks
    overlap_buffer = bytearray()
    
    # Calculate target buffer sizes
    target_buffer_size = calculate_audio_buffer_size()
    overlap_size = calculate_overlap_size()
    
    p = None
    stream = None
    
    try:
        p = pyaudio.PyAudio()
        device_info = p.get_device_info_by_index(device_index)
        gui_q.put(f"[{thread_name}] Selected device: {device_info.get('name')}")
        
        if device_info.get('maxInputChannels', 0) < AUDIO_CHANNELS:
            raise ValueError(f"Device does not support required channels ({AUDIO_CHANNELS})")

        stream = p.open(
            format=AUDIO_FORMAT, 
            channels=AUDIO_CHANNELS, 
            rate=AUDIO_SAMPLERATE,
            input=True, 
            frames_per_buffer=PYAUDIO_FRAMES_PER_BUFFER,
            input_device_index=device_index
        )
        
        gui_q.put(f"[{thread_name}] Audio stream opened. Capturing with {AUDIO_BUFFER_SECONDS}s buffers and {AUDIO_OVERLAP_SECONDS}s overlap...")
        
        # Track audio energy levels for automatic gain control
        recent_audio_levels = []
        
        while not stop_event_flag.is_set():
            try:
                # Check stop event *before* reading to exit faster
                if stop_event_flag.is_set(): break
                
                # Read audio data
                data = stream.read(PYAUDIO_FRAMES_PER_BUFFER, exception_on_overflow=False)
                
                # Calculate energy level for this chunk (for future automatic gain control)
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                energy = np.sqrt(np.mean(audio_np**2))
                recent_audio_levels.append(energy)
                if len(recent_audio_levels) > 30:  # Keep ~1 second of history
                    recent_audio_levels.pop(0)
                
                # Add to main buffer
                audio_buffer.extend(data)
                
                # Process chunks as they become available
                while len(audio_buffer) >= target_buffer_size:
                    if stop_event_flag.is_set(): break  # Check again before processing
                    
                    # Extract chunk to process (including overlap from previous if available)
                    if overlap_buffer:
                        combined_chunk = overlap_buffer + audio_buffer[:target_buffer_size]
                        chunk_to_process = bytes(combined_chunk)
                    else:
                        chunk_to_process = bytes(audio_buffer[:target_buffer_size])
                    
                    # Keep overlap for next chunk
                    overlap_buffer = audio_buffer[target_buffer_size - overlap_size:target_buffer_size]
                    
                    # Put the chunk in the queue
                    audio_q.put(chunk_to_process)
                    
                    # Keep the rest of the buffer
                    audio_buffer = audio_buffer[target_buffer_size:]
                    
            except IOError as e:
                # Check stop event even on error
                if stop_event_flag.is_set(): break
                
                if e.errno == pyaudio.paInputOverflowed or "Input overflowed" in str(e):
                    gui_q.put(f"[{thread_name}] Warning: PyAudio Input Overflowed. Buffer size might be too small.")
                else:
                    gui_q.put(f"[{thread_name}] ERROR: PyAudio read error: {e}")
                    time.sleep(0.1)  # Brief pause on other IOErrors
                    
            except Exception as e:
                # Check stop event even on error
                if stop_event_flag.is_set(): break
                
                gui_q.put(f"[{thread_name}] ERROR: Unexpected error during stream read: {e}")
                time.sleep(0.1)  # Brief pause on error

        if stop_event_flag.is_set(): 
            gui_q.put(f"[{thread_name}] Stop event received, exiting capture loop.")

    except ValueError as e: 
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Device config error: {e}")
    except Exception as e: 
        gui_q.put(f"[{thread_name}] CRITICAL ERROR: Initializing/running PyAudio: {e}")
    finally:
        print(f"[{thread_name}] Entering finally block...", flush=True)
        gui_q.put(f"[{thread_name}] Cleaning up audio resources...")
        
        if stream is not None:
            try:
                # Make sure stream is stopped first
                if stream.is_active():
                    print(f"[{thread_name}] Stopping PyAudio stream...", flush=True)
                    stream.stop_stream()
                    print(f"[{thread_name}] Stream stopped.", flush=True)
                
                # Now close
                print(f"[{thread_name}] Closing PyAudio stream...", flush=True)
                stream.close()
                print(f"[{thread_name}] Stream closed.", flush=True)
                gui_q.put(f"[{thread_name}] Audio stream stopped and closed.")
            except Exception as e:
                print(f"[{thread_name}] Warning: Error during explicit stream stop/close: {e}", flush=True)
                gui_q.put(f"[{thread_name}] Warning: Error closing stream: {e}")
                
        if p is not None:
            print(f"[{thread_name}] Terminating PyAudio...", flush=True)
            p.terminate()
            gui_q.put(f"[{thread_name}] PyAudio terminated.")
            print(f"[{thread_name}] PyAudio terminated.", flush=True)

        # Signal end by putting None in the *next* queue (audio_queue)
        print(f"[{thread_name}] Putting None in audio_queue...", flush=True)
        audio_q.put(None)
        print(f"[{thread_name}] Putting final status msg in gui_queue...", flush=True)
        gui_q.put(f"[{thread_name}] Audio capture thread finished.")
        print(f"[{thread_name}] Putting final None signal in gui_queue...", flush=True)
        gui_q.put(None)  # Signal for GUI queue
        print(f"[{thread_name}] Exiting finally block.", flush=True) 