import os

from gtts import gTTS


def text_to_speech(text, language, output_path="Output.mp3"):
    audio_object = gTTS(text=text,
                        lang=language,
                        slow=False)
    audio_object.save(output_path)

# os.system("afplay Output.mp3")
