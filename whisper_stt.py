import pyaudio
import argparse
import io
import speech_recognition as sr
import whisper
import torch
from datetime import datetime, timedelta
from enum import Enum
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

class ModelSize(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class RealTimeTranscriber:
    """
    A class that encapsulates the functionality for real-time speech transcription using the Whisper ASR API.
    """
    def __init__(self, args):
        self.recorder = self.initialize_recorder(args.energy_threshold)
        self.source = self.initialize_microphone(16000, args.default_microphone)
        self.model = self.initialize_model(args)
        self.data_queue = Queue()
        self.temp_file = NamedTemporaryFile().name
        self.transcription = ['']

    @staticmethod
    def initialize_recorder(energy_threshold):
        """Initializes the recorder with a specific energy threshold."""
        recorder = sr.Recognizer()
        recorder.energy_threshold = energy_threshold
        recorder.dynamic_energy_threshold = False
        return recorder

    @staticmethod
    def initialize_microphone(sample_rate, mic_name):
        """Initializes the microphone with a specific sample rate and microphone name."""
        if 'linux' in platform and mic_name:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    mic = sr.Microphone(sample_rate=sample_rate, device_index=index)
        else:
            mic = sr.Microphone(sample_rate=sample_rate)
        
        mic.audio = pyaudio.PyAudio()
        return mic

    @staticmethod
    def initialize_model(args):
        """Initializes the Whisper model."""
        model = getattr(ModelSize, args.model.upper()).value
        if model in ['tiny', 'base', 'small', 'medium'] and not args.non_english:
            model = model + ".en"
        return whisper.load_model(model)

    def handle_recording_callback(self, _, audio):
        """Callback function to handle the recording. It puts the raw audio data into the queue."""
        self.data_queue.put(audio.get_raw_data())

    def create_background_listener(self, record_callback, record_timeout):
        """Creates a background listener that listens to the microphone and triggers the callback function when it detects speech."""
        return self.recorder.listen_in_background(self.source, record_callback, phrase_time_limit=record_timeout)

    def process_audio(self, phrase_timeout):
        """
        Processes the audio data from the queue.
        It transcribes the speech and prints the transcription in real time.
        """
        phrase_time = None
        last_sample = bytes()

        while True:
            now = datetime.utcnow()
            if not self.data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                phrase_time = now

                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                with open(self.temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                result = self.model.transcribe(self.temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if phrase_complete:
                    self.transcription.append(text)
                    print(text)
                else:
                    self.transcription[-1] = text
                    print('\r' + text, end='', flush=True)


                # for line in self.transcription:
                #     print(line)
                # print('', end='', flush=True)

                sleep(0.25)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=ModelSize.MEDIUM.value, help="Model to use",
                        choices=[e.value for e in ModelSize])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the English model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    transcriber = RealTimeTranscriber(args)
    transcriber.create_background_listener(transcriber.handle_recording_callback, args.record_timeout)
    
    print("Model loaded. You can start speaking now.\n")
    try:
        transcriber.process_audio(args.phrase_timeout)
    except KeyboardInterrupt:     
        print("\n\nTranscription:")
        for line in transcriber.transcription:
            print(line)

if __name__ == "__main__":
    main()
