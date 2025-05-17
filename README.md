# Live Audio Translator v2.0

A real-time audio translation application that captures audio input, transcribes it, and translates it to English. The application supports multiple speech recognition engines and features a floating subtitle overlay.

## Features

- **Multiple Speech Recognition Engines**:
  - **Whisper**: High-quality local speech recognition with various model sizes
  - **Vosk**: Lightweight offline speech recognition
  - **Google Cloud Speech**: Cloud-based high-accuracy recognition
  - **Azure Speech**: Microsoft's cloud speech services with neural voices

- **Advanced Audio Processing**:
  - Overlapping audio buffers to prevent missing speech at segment boundaries
  - Noise reduction (when noisereduce library is available)
  - Automatic volume normalization
  - DC offset removal

- **Enhanced Sentence Processing**:
  - Intelligent sentence boundary detection
  - Filler word removal (um, uh, etc.)
  - Question detection for proper punctuation
  - Conjunction detection to maintain sentence flow

- **Customizable UI**:
  - Floating subtitle overlay with adjustable position, size, color, and transparency
  - Dark theme interface with batch GUI updates to prevent lag
  - History view with scrollable transcript

- **Performance Optimizations**:
  - Efficient memory management with CUDA cache clearing
  - Batched GUI updates to prevent UI lag
  - Overlapping buffers to capture all speech
  - Modular architecture for easy maintenance and customization

## Requirements

- Python 3.9+ (tested with Python 3.10)
- PyAudio for audio capture
- CUDA-compatible GPU recommended for Whisper engine
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository or download the source code.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. (Optional) For cloud services, set up API keys:
   - For Google Cloud Speech: Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
   - For Azure Speech: Set `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` environment variables

4. (Optional) For Vosk engine: Download appropriate models from https://alphacephei.com/vosk/models and place in the `models` directory.

## Usage

Run the application with:

```bash
python liveaudiotranslator_v2.0.py
```

### UI Controls

- **Engine**: Select the speech recognition engine (Whisper, Vosk, Google, Azure)
- **Input Device**: Choose your microphone or audio input device
- **Src Lang**: Source language for recognition (or Auto Detect)
- **Model**: For Whisper, select model size (tiny, base, small, medium, large-v2, large-v3)
- **Start/Stop**: Control translation process
- **Font Size**: Adjust subtitle text size
- **Overlay Width**: Change the width of subtitles overlay
- **Overlay Alpha**: Adjust transparency of subtitle overlay
- **Font Color**: Select text color (White/Black)
- **Main Alpha**: Adjust transparency of main window
- **Show Subtitles**: Toggle visibility of floating subtitle overlay
- **Show Background**: Toggle background bar behind subtitles
- **Save Log**: Enable/disable saving translations to log file

## Configuration

You can adjust various settings in the `config.py` file:

- Audio buffer size and overlap
- VAD (Voice Activity Detection) settings
- Default model sizes and languages
- UI appearance and behavior

## Troubleshooting

- **No audio input devices found**: Check your microphone connections and drivers
- **"No module named..." errors**: Install missing Python packages with pip
- **CUDA out of memory**: Try a smaller Whisper model size
- **Speech recognition issues**: Try adjusting VAD settings or using a different engine

## Credits

This application uses:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for Whisper speech recognition
- [Vosk](https://alphacephei.com/vosk/) for offline speech recognition
- Google and Azure Speech APIs for cloud-based recognition
- [noisereduce](https://github.com/timsainb/noisereduce) for audio noise reduction

## License

MIT License 