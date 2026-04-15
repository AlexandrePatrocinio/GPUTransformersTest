import torch
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
model_dir = "Models/Coders/Qwen"
device = ""

try:    
    gpu_count = torch.cuda.device_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    gpu_count = 0
    device = "cpu"

print(f"Device mode: {device}")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print(f"Detected CUDA GPUs: {gpu_count}")
    print("Total memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("CUDA not available")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=model_dir,
    device_map="auto"
).eval()

messages = [
    {"role": "user", "content": "Create a minimal api in C# dotnet 10 with an endpoint that returns a JSON response. It should have a single endpoint at /api/hello that returns { \"message\": \"Hello, World!\" } but without using any external libraries or controllers."},
]

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",    
).to(device)

start_time = time()

outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

end_time = time()
elapsed_time_ms = (end_time - start_time) * 1000

print(f"Prompt: {messages[0]['content']}\n\nGenerated text: {output_text}")
print(f"Elapsed time: {elapsed_time_ms:.2f} ms")