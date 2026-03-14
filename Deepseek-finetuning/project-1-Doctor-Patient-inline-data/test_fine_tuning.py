import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Setup paths and model names
base_model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"
lora_adapter_path = "./output_folder"

print("\n\n\nLoading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Use the same dtype logic as your training script
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=dtype, 
    device_map="auto"
)

# 2. Load the LoRA adapter
print(f"\nAttaching LoRA adapters from {lora_adapter_path}...")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval() # Set to evaluation mode

def generate_response(user_input):
    # Match the exact format used during fine-tuning: Patient: <prompt>\nDoctor:
    prompt = f"Patient: {user_input}\nDoctor: "
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the full generated text
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part after "Doctor:"
    if "Doctor:" in full_text:
        response = full_text.split("Doctor:")[-1].strip()
    else:
        response = full_text.strip()
    return response


# 3. Interactive Loop
print("\n\n--- Model Ready! Type 'exit' to quit ---")

while True:
    user_command = input("Patient: ")
    if user_command.strip().lower() == 'exit':
        print("\nEnding session.")
        break
        
    response = generate_response(user_command)
    print(f"Doctor: {response}\n")


