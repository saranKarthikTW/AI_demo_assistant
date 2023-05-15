import time
from enum import Enum
from utils.whisper_adapter import text_from_audio
import re
from utils.gpt_adapter import seq2seq_response
from utils.gtts_adapter import text_to_speech
import gradio
import warnings
import threading
import os

warnings.filterwarnings("ignore")


class Status(Enum):
    Sleeping = "Call 'buddy' to wake me"
    Processing = "Processing..."
    Inferring = "Computing..."
    Responding = "Responding..."

    def __str__(self):
        return self.value


class SpeechAssistant:
    __text_from_speech = ""
    __inferred_response = ""

    def __init__(self, name, language, audio_output_path):
        self.name = name
        self.language = language
        self.audio_output_path = audio_output_path
        self.status = Status.Sleeping

    def assist_handler(self, audio, state=""):
        if self.status != Status.Responding:
            if audio is None:
                return ["No audio input was received", "", "", None, state]

            self.__wait_for_audio()

            transcribed_text, language = text_from_audio(audio, self.language)
            print("transcription: ", transcribed_text)
            self.__update_status_on_audio(transcribed_text)

            if self.status == Status.Processing:
                state = self.__update_state_with_incoming_text(state, transcribed_text)
                # preparing for return
                self.__text_from_speech, self.__inferred_response = state, ""

            elif self.status == Status.Inferring:
                self.__text_from_speech = self.__get_query_only(state)
                self.__inferred_response = self.__infer_query(self.__text_from_speech)
                text_to_speech(self.__inferred_response, language, self.audio_output_path)
                self.__callback_to_play_audio(self.audio_output_path)
                # Preparing for next conversation
                self.status, state = Status.Responding, ""

        return [self.status, self.__text_from_speech, self.__inferred_response, self.audio_output_path, state]

    def __wait_for_audio(self):
        if self.status == Status.Sleeping:
            self.__listen_for_seconds(2)
        elif self.status == Status.Processing:
            self.__listen_for_seconds(3)

    def __update_status_on_audio(self, current_text):
        if self.status == Status.Sleeping:
            if self.name in current_text.lower():
                self.status = Status.Processing
        elif self.status == Status.Processing:
            if current_text == "":
                self.status = Status.Inferring

    def __update_state_with_incoming_text(self, state, text):
        state = state + " " + self.__remove_punctuations(text)
        return state

    def __get_query_only(self, state):
        wake_word_pos = state.lower().find(self.name)
        query_without_wake_word = state[wake_word_pos + len(self.name):]
        return self.__remove_punctuations(query_without_wake_word)

    def __infer_query(self, state):
        query_text = self.__trim_query_from_transcription(state, self.name)
        inferred_response = seq2seq_response(query_text)
        return inferred_response

    def __callback_to_play_audio(self, audio_file_path):
        t = threading.Thread(target=self.__play_audio, args=[audio_file_path])
        t.setDaemon(False)
        t.start()

    def __play_audio(self, file_path):
        time.sleep(0.5)
        os.system(f"afplay {file_path}")
        self.status = Status.Sleeping

    @staticmethod
    def __listen_for_seconds(seconds):
        time.sleep(seconds)

    @staticmethod
    def __remove_punctuations(text):
        return re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', text)

    @staticmethod
    def __trim_query_from_transcription(state, name):
        wake_word_index = state.lower().find(name)
        wake_word_ending_index = wake_word_index + len(name)
        trimmed_query = state[wake_word_ending_index + 1:]
        return trimmed_query


if __name__ == "__main__":
    input_language = "en"  # or auto
    final_audio_path = "Output.mp3"

    speech_assistant = SpeechAssistant("buddy", input_language, final_audio_path)

    gradio.Interface(
        title='Speech Assistant',
        fn=speech_assistant.assist_handler,
        inputs=[
            gradio.Audio(source="microphone", type="filepath", streaming=True),
            "state"
        ],
        outputs=[
            gradio.Textbox(label="Status"),
            gradio.Textbox(label="Speech to Text"),
            gradio.Textbox(label="Model Output"),
            gradio.Audio(speech_assistant.audio_output_path),
            "state"
        ],
        live=True,
    ).launch(debug=True)
