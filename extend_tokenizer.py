
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# Load the tokenizer and model (use T5-base)
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained('t5-small')
#  = T5ForConditionalGeneration.from_pretrained(model_name)
# Load the prepared data
data = pd.read_csv('hindi.csv')
data.dropna(inplace=True)

all_text = ' '.join(data['words'].tolist() + data['parsed_output'].tolist())

unique_chars = set(all_text)
# print(unique_chars) 

additional_tokens = list(unique_chars)
tokenizer.add_tokens(additional_tokens)

# Save the updated tokenizer
tokenizer.save_pretrained('./extended-tokenizer')