# livetranslator
python code that enables live translations on any video in the form of a gui.

There are three versions:

1. liveaudiotranslator - this one is basic and uses your desktop audio and whisper to tanslate, can only be used to translate to english.
2. liveaudiotranslator Potato - this is the same as above but uses float32 for cpu only mode. should work on potato pcs
3. liveaudiotranslator_llama_multi - this one uses your desktop audio plus Ollama to translate, can be used to translate to any language.

The ollama version needs you to install Ollama from their website first and have it running in the background before launching the program.
You then need to "pull" the model "llama3". you can do so by opening terminal or power script and typing "ollama pull llama3".

----- To install -----

This uses your desktop audio channel to live translate using whisper. Only translates to english.

To unpackage the files:

1. Download ALL parts (.001, .002, ...).
2. Have a tool like 7-Zip installed.
3. Put all the downloaded parts in the same folder.
4. Right-click on the first file (.001) and choose 7-Zip -> Extract Here or Extract files.... 7-Zip will automatically find the other parts and reassemble the original file.

If you cannot find the 7zip in the right click menu select shop more options at the bottom.
