import os
import requests
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch
from tqdm import tqdm

# Download the parquet file
def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

# Process cosmopedia dataset
def process_cosmopedia_data():
    url = "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/cosmopedia-v2/train-00000-of-00104.parquet?download=true"
    filename = "train.parquet"
    
    if not os.path.exists(filename):
        download_file(url, filename)
    
    df = pd.read_parquet(filename)
    return df[['text']]

# Create a small model
def create_small_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    
    # Configuration for a model with ~50M parameters
    config = GPT2Config(
        vocab_size=32000,
        n_positions=512,
        n_embd=512,
        n_layer=6,
        n_head=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    
    model = GPT2LMHeadModel(config)
    print(f"Model parameters: {model.num_parameters():,}")
    return model

# Create tokenizer with custom tokens
def create_tokenizer():
    # Initialize with existing tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Add special tokens
    special_tokens = {
        "pad_token": "<|pad|>",
        "eos_token": "<|endoftext|>",
        "additional_special_tokens": [
            "<|system|>",
            "<|user|>",
            "<|assistant|>"
        ]
    }
    
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

# Format chat data
def format_chat_example(conversations):
    formatted = ""
    for msg in conversations:
        if msg["from"] == "system":
            formatted += f"<|system|>\n{msg['value']}<|endoftext|>\n"
        elif msg["from"] == "human":
            formatted += f"<|user|>\n{msg['value']}<|endoftext|>\n"
        elif msg["from"] == "gpt":
            formatted += f"<|assistant|>\n{msg['value']}<|endoftext|>\n"
    return formatted

# Main training process
def main():
    # Process initial text data
    print("Processing cosmopedia data...")
    text_df = process_cosmopedia_data()
    
    # Create model and tokenizer
    print("Creating model and tokenizer...")
    tokenizer = create_tokenizer()
    model = create_small_model()
    model.resize_token_embeddings(len(tokenizer))
    
    # Tokenize text data
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
    
    # Prepare dataset for initial training
    text_dataset = Dataset.from_pandas(text_df)
    tokenized_dataset = text_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )
    
    # Initial training arguments
    training_args = TrainingArguments(
        output_dir="./initial_model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=100,
        report_to=None,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        fp16=False,  # Disable FP16 for CPU
        no_cuda=True,  # Force CPU usage
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Initial training
    print("Starting initial training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model("./initial_model")
    
    # Load Hermes dataset
    print("Loading Hermes dataset...")
    hermes_dataset = load_dataset("NousResearch/Hermes-3-Dataset", split="train")
    
    # Format chat data
    def format_chat_data(example):
        return {"text": format_chat_example(example["conversations"])}
    
    formatted_dataset = hermes_dataset.map(format_chat_data)
    
    # Tokenize chat data
    def tokenize_chat_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
    
    tokenized_chat_dataset = formatted_dataset.map(
        tokenize_chat_function,
        batched=True,
        remove_columns=["conversations", "text"]
    )
    
    # Fine-tuning arguments
    fine_tune_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=50,
        report_to=None,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        fp16=False,
        no_cuda=True,
        learning_rate=5e-5,
    )
    
    # Fine-tuning
    print("Starting fine-tuning on chat data...")
    fine_tune_trainer = Trainer(
        model=model,
        args=fine_tune_args,
        data_collator=data_collator,
        train_dataset=tokenized_chat_dataset,
        tokenizer=tokenizer,
    )
    
    fine_tune_trainer.train()
    fine_tune_trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    print("Training completed! Model saved to ./fine_tuned_model")

if __name__ == "__main__":
    main()