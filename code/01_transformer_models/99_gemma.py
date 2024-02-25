import os

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

token = os.getenv("HF_TOKEN")
login(token)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
generation_config = GenerationConfig(max_new_tokens=200)

inputs = tokenizer("Colombia is", return_tensors="pt", )
print("Inputs tokenized")
outputs = model.generate(**inputs, generation_config=generation_config)
print("Outputs generated")

print(tokenizer.decode(outputs[0]))
