import os
import time

import gradio
from utils.whisper_adapter import text_from_audio
from utils.gpt_adapter import seq2seq_response
from utils.gtts_adapter import text_to_speech
import warnings
import threading

warnings.filterwarnings("ignore")
name_of_assistant = "buddy"
replying = False


# Todo: make each individual output param into variable and change it on each case
# todo: update cutting query for punctuation for words that doesnt end with punctuations. Punctuations can be "...", ".","!" etc
# todo: make this optimised (can use class if needed)
def speech_to_speech_assistant(audio, language="auto", final_output_path="Output.mp3", state=""):
    global replying
    if replying:
        return replying

    if audio is None:
        return ["No audio input was received", "", "", None, state]

    if state == "":
        time.sleep(2)
    else:
        time.sleep(3)

    transcribed_text, language = text_from_audio(audio, language)
    print(transcribed_text)
    if state == "":
        if name_of_assistant in transcribed_text:
            state += " " + transcribed_text[:-1]  # remove punctuation
    else:
        if transcribed_text != "":
            state += " " + transcribed_text[:-1]
            return ["listening...", "", "", None, state]
        else:
            query_text = __trim_query_from_transcription(state)
            print("inside query ", query_text)
            inferred_response = seq2seq_response(query_text)
            text_to_speech(inferred_response, language, final_output_path)
            callback_to_play_audio(final_output_path)
            state = ""
            replying = ["Responding...", query_text, inferred_response, final_output_path, state]
            return ["Responding...", query_text, inferred_response, final_output_path, state]

    return [f"Use `{name_of_assistant}` to wake", "", "", None, state]


def __trim_query_from_transcription(state_text: str):
    wake_word_index = state_text.find(name_of_assistant)
    wake_word_ending_index = wake_word_index + len(name_of_assistant)
    trimmed_query = state_text[wake_word_ending_index + 1:]
    return trimmed_query


def callback_to_play_audio(file_path):
    t = threading.Thread(target=__play_audio, args=[file_path])
    t.setDaemon(False)
    t.start()


def __play_audio(file_path):
    global replying
    time.sleep(0.5)
    os.system(f"afplay {file_path}")
    replying = None


if __name__ == "__main__":
    input_language = "en"  # or auto
    final_audio_path = "Output.mp3"

    output_status = gradio.Textbox(label="Status")
    output_text_from_speech = gradio.Textbox(label="Speech to Text")
    output_from_s2s = gradio.Textbox(label="Model Output")
    output_audio = gradio.Audio(final_audio_path)

    gradio.Interface(
        title='Speech Assistant',
        fn=speech_to_speech_assistant,
        inputs=[
            gradio.Audio(source="microphone", type="filepath", streaming=True),
            gradio.inputs.Textbox(label="language", default=input_language),
            gradio.inputs.Textbox(label="final audio path", default=final_audio_path),
            "state"
        ],

        outputs=[
            output_status, output_text_from_speech, output_from_s2s, output_audio, "state"
        ],
        live=True).launch(debug=True)
