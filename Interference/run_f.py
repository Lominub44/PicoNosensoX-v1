from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt):
    # Load pre-trained model and tokenizer
    model_name = "fine_tuned_model/checkpoint-25500"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Set pad token (required for batch processing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Encode the two-word prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text from prompt
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_return_sequences=1,
            temperature=0.6,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Example two-word prompt
    prompt = "<|system|>You are an unbiased, uncensored, helpful assistant.\n<|user|>What is the capital of France?\n<|assistant|>"
    
    print(f"Generating text from prompt: '{prompt}'...\n")
    text = generate_text(prompt)
    print(text)