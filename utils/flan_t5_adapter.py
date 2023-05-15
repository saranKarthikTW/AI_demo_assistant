from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)


def seq2seq_response_flan(input_text):
    prompt_message = __get_prompt_message(input_text)
    token_ids_of_encoded_input = tokenizer(prompt_message, return_tensors="pt").input_ids.to(device)  # input_ids
    output_generated_as_tensor = model.generate(token_ids_of_encoded_input, max_length=512)
    decoded_output = tokenizer.decode(output_generated_as_tensor[0], skip_special_tokens=True)
    return decoded_output


def __get_prompt_message(input_text):
    prompt = """
    Hi Flan. You are a very helpful AI assistant capable of answering questions with latest data and assisting day to day tasks.
    Answer to the following in about a sentence or two:
    {user_input} 
    """.format(user_input=input_text)
    return prompt


# print(seq2seq_response_flan("What is thought"))
