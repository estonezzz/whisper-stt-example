# Real-Time Speech Recognition with OpenAI's Whisper

This script uses OpenAI's Whisper Automatic Speech Recognition (ASR) system to transcribe
 speech in real-time. t continuously listens to audio input from a microphone, processes 
 the audio data in real-time, and outputs the transcribed text to the console. This script
 is useful for a variety of applications, such as transcription services, voice assistants,
   and more.

## Prerequisites

- Python 3.7+
- whisper
- speech_recognition
- pyaudio
- torch

You can install the required Python packages with pip:

```sh
pip install -r requirements.txt
```

## Usage

```sh
python whisper_stt.py --model medium --energy_threshold 1000 --record_timeout 2 --phrase_timeout 3
```

### Options

- `--model`: The model to use for Whisper. Options are "tiny", "base", "small", "medium", "large". Default is "medium".
- `--non_english`: If set, the script will use the non-English version of the specified Whisper model.
- `--energy_threshold`: The energy level for the microphone to detect. Default is 1000.
- `--record_timeout`: The real-time recording duration in seconds. Default is 2 seconds.
- `--phrase_timeout`: The time in seconds of silence between recordings before the script considers it a new line in the transcription. Default is 3 seconds.

## Notes

For more information on Whisper, visit [OpenAI's Whisper ASR System](https://github.com/openai/whisper).


## Credits

This implementation is inspired by the real-time Whisper demo provided by [davabase/whisper_real_time](https://github.com/davabase/whisper_real_time).
