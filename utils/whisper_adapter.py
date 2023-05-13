import whisper

model = whisper.load_model("base")


def text_from_audio(audio, language="auto"):
    audio_as_lms = __get_lms(audio)
    if language == "auto":
        language = __detect_language(audio_as_lms)
        print(f"The language detected with max probability is {language}")
    inferred_text = __decode_lms_to_text(audio_as_lms, language)
    return inferred_text, language


def __get_lms(audio):
    audio = whisper.load_audio(file=audio)  # undergoes resampling
    audio = whisper.pad_or_trim(audio)  # defaults to 30 seconds
    audio_as_lms = whisper.log_mel_spectrogram(audio).to(model.device)  # passed to the encoder block
    return audio_as_lms


def __detect_language(audio_as_lms):
    _, probability_distribution_of_languages = model.detect_language(audio_as_lms)
    language_detected = max(probability_distribution_of_languages, key=probability_distribution_of_languages.get)
    return language_detected


def __decode_lms_to_text(audio_as_lms, language):
    options = whisper.DecodingOptions(fp16=False, language=language)
    result = whisper.decode(model, audio_as_lms, options)
    return result.text