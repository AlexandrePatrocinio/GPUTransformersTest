import torch
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
model_dir = "Models/Coders/Qwen"
device_map = ""
device = ""

try:    
    gpu_count = torch.cuda.device_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "cuda" if gpu_count > 0 else "cpu"
except Exception:
    gpu_count = 0
    device_map = "cpu"
    device = "cpu"

print(f"Device mode: {device}")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print(f"Detected CUDA GPUs: {gpu_count}")
    print("Total memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("CUDA not available")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=model_dir,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map=device_map,
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

outputs = model.generate(
    **inputs, 
    max_new_tokens=512, 
    do_sample=True, 
    temperature=0.7, 
    min_p=0.5,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)
output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

end_time = time()
elapsed_time_ms = (end_time - start_time) * 1000

print(f"Prompt: {messages[0]['content']}\n\nGenerated text: {output_text}")
print(f"Elapsed time: {elapsed_time_ms:.2f} ms")