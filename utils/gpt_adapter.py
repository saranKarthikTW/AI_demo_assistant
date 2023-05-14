import os.path

import openai
import json


def seq2seq_response(input_text):
    messages = __get_prompt_messages(input_text)
    __set_openai_api_key()
    reply = __get_response_from_gpt(messages)
    return reply


def __get_prompt_messages(input_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    if input_text:
        messages.append({"role": "user", "content": input_text})
    return messages


def __set_openai_api_key():
    with open('/Users/sarankarthik/Documents/ml/asr-demo/GPT_SECRET_KEY.json') as f:
        data = json.load(f)
    openai.api_key = data["API_KEY"]


def __get_response_from_gpt(messages):
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    reply = chat_completion.choices[0].message.content
    return reply