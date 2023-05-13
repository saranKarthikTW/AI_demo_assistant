import os
import gradio
from utils.whisper_adapter import text_from_audio
from utils.gpt_adapter import seq2seq_response
from utils.gtts_adapter import text_to_speech
import warnings

warnings.filterwarnings("ignore")


def speech_to_speech_assistant(audio, language="auto", final_output_path="Output.mp3"):
    transcribed_text, language = text_from_audio(audio, language)
    inference_from_model = seq2seq_response(transcribed_text)
    text_to_speech(inference_from_model, language, final_output_path)
    callback_to_play_audio(final_output_path)
    return [transcribed_text, inference_from_model, final_output_path]


def callback_to_play_audio(file_path):
    os.system(f"afplay {file_path}")


if __name__ == "__main__":
    input_language = "en"  # or auto
    final_audio_path = "Output.mp3"

    output_text_from_speech = gradio.Textbox(label="Speech to Text")
    output_from_s2s = gradio.Textbox(label="Model Output")
    output_audio = gradio.Audio(final_audio_path)

    gradio.Interface(
        title='Speech Assistant',
        fn=speech_to_speech_assistant,
        inputs=[
            gradio.inputs.Audio(source="microphone", type="filepath"),
            gradio.inputs.Textbox(label="language", default=input_language),
            gradio.inputs.Textbox(label="final audio path", default=final_audio_path),
        ],

        outputs=[
            output_text_from_speech, output_from_s2s, output_audio
        ],
        live=True).launch(debug=True)
