import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure the pad_token_id is set correctly
model.config.pad_token_id = model.config.eos_token_id

# Add pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Resize the token embeddings in the model to account for the new tokens 
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

# Load and preprocess dataset using datasets library 
def preprocess_data(examples):
    tokenized_examples = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    return tokenized_examples

# Load dataset (assuming dataset.txt is a text file with one sentence per line) 
dataset = load_dataset('text', data_files={'train': 'AI/dataset.txt'})

# Preprocess dataset 
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Prepare data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Set training arguments
training_args = TrainingArguments(
    output_dir='E:AI_Stuff/results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2, # Adjust as needed
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
    tokenizer=tokenizer
)

# Train model
trainer.train()

# Save model
trainer.save_model('E:AI_Stuff/fine_tuned_gpt2')

# Function to generate text
def generate_text(prompt, max_length=150, temperature=0.9, top_k=50, top_p=0.9):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Create attention mask (True for actual tokens, False for padding tokens)
    attention_mask = input_ids.ne(model.config.pad_token_id).long()

    # Generate text
    output = model.generate(input_ids,
                            attention_mask=attention_mask,
                            max_length=max_length,
                            num_return_sequences=1,
                            pad_token_id=model.config.eos_token_id,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=True # Enable sampling mode
                            )

    # Decode the output text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# Example usage
prompt = ""
generated_text = generate_text(prompt, temperature=0.5, top_k=30, top_p=0.85)
print(generated_text)