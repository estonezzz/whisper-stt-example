Whisper ASR Speech-to-Text Python Script

This Python script uses the Whisper ASR (Automatic Speech Recognition) system, developed by OpenAI, to convert spoken language into written text. It continuously listens to audio input from a microphone, processes the audio data in real-time, and outputs the transcribed text to the console. This script is useful for a variety of applications, such as transcription services, voice assistants, and more.
Settings and Options

This script provides several command-line arguments to customize its behavior:

    --model: Specifies the model to use for the Whisper ASR system. Options include tiny, base, small, medium, and large. The default value is medium.

    --non_english: If this flag is set, the script won't use the English model.

    --energy_threshold: Sets the energy level for the microphone to detect. The default value is 1000.

    --record_timeout: Defines how real-time the recording is, in seconds. The default value is 2.

    --phrase_timeout: Specifies how much empty space between recordings before we consider it a new line in the transcription, in seconds. The default value is 3.

    --default_microphone: Sets the default microphone name for SpeechRecognition. The default value is pulse. If list is provided as an argument, the script will list available microphones.

Usage

First, make sure all the necessary Python packages are installed by running:

bash

pip install -r requirements.txt

Then, you can run the script using the following command:

bash

python script_name.py

Replace script_name.py with the actual name of the script.

You can also specify command-line arguments as follows:

bash

python script_name.py --model large --energy_threshold 1500 --record_timeout 3 --phrase_timeout 4

Note

For more information about the Whisper ASR system, please visit the official OpenAI Whisper project.