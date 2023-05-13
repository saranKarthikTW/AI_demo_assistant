from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model_path = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

input_text = """
Hi Flan. You are a very helpful AI assistant capable of answering questions with latest data and assisting day to day tasks.
{user_input} 
""".format(user_input="What is facebook")

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
output = model.generate(input_ids, max_length=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
