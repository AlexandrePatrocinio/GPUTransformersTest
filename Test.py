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

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=model_dir,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map=device_map,
).eval()

context = []

system_prompt = "You are a helpful assistant for coding and a dotnet expert. " \
"Answer questions accurately and provide code examples when necessary. " \
"Be brief in your answers. Try to adapt them to my context of a maximum of 512 tokens. " \
"If you don't know the answer, say you don't know instead of making up an answer."

context.append({"role": "system", "content": system_prompt})

while True:
    prompt = input("Enter your prompt (or 'quit' to exit):\n")
    if prompt.lower() == "quit":
        break

    context.append({"role": "user", "content": prompt})

    start_time = time()

    inputs = tokenizer.apply_chat_template(
        context,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",    
    ).to(device)

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

    m = elapsed_time_ms // 60000
    s = elapsed_time_ms % 60000 / 1000

    context.append({"role": "assistant", "content": output_text})

    print(f"\n\nGenerated text: {output_text}")

    print(f"Elapsed time: {m:.0f} m {s:.0f} s")