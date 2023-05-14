import whisper

model = whisper.load_model("base")


def text_from_audio(audio, language=None):
    audio_as_lms = __get_lms(audio)
    decoded_lms = __decode_lms(audio_as_lms, language)
    inferred_text = decoded_lms.text if not __is_silence(decoded_lms) else ""
    if language is None:
        language = decoded_lms.language
    return inferred_text, language


def text_from_audio_simplified(audio, language=None):
    transcribe_results = model.transcribe(audio, language=language)
    if language is None:
        language = transcribe_results["language"]
    return transcribe_results["text"], language


def __is_silence(decoded_lms):
    return decoded_lms.no_speech_prob > 0.5


def __get_lms(audio):
    audio = whisper.load_audio(file=audio)  # undergoes resampling
    audio = whisper.pad_or_trim(audio)  # defaults to 30 seconds
    audio_as_lms = whisper.log_mel_spectrogram(audio).to(model.device)  # passed to the encoder block
    return audio_as_lms


def __decode_lms(audio_as_lms, language):
    options = whisper.DecodingOptions(fp16=False, language=language)
    result = whisper.decode(model, audio_as_lms, options)
    print('result.no_speech_prob ', result.no_speech_prob)
    return result


# def __detect_language(audio_as_lms):
#     _, probability_distribution_of_languages = model.detect_language(audio_as_lms)
#     language_detected = max(probability_distribution_of_languages, key=probability_distribution_of_languages.get)
#     return language_detected

# print(text_from_audio("../sample_audio_base.flac"))
