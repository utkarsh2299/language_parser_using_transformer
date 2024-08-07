import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# Load the tokenizer and model (use T5-base)
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained('./extended-tokenizer')
model = T5ForConditionalGeneration.from_pretrained(model_name)


model.resize_token_embeddings(len(tokenizer))


train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)
# Convert pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenize the input and target texts
def preprocess_data(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')

    # Tokenize the targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = train_dataset.map(preprocess_data, batched=True)
val_dataset = val_dataset.map(preprocess_data, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    weight_decay=0.01,
    warmup_steps=500,  # Add warmup steps
    gradient_accumulation_steps=2,  # Accumulate gradients if memory is an issue
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=200,  # Log every 200 steps
    save_steps=1000,  # Save the model every 1000 steps
    save_total_limit=5, 
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-t5-base')
tokenizer.save_pretrained('./fine-tuned-t5-base')

