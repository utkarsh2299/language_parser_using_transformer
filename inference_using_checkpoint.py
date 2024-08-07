from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# List of checkpoint directories
checkpoints = ['./results/checkpoint-4079010']# './results/checkpoint-1627000', './results/checkpoint-1628000']  # Add more checkpoints as needed

def generate_parsed_output(input_text, checkpoint_path):
    # Verify if the checkpoint directory exists
    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"Checkpoint directory '{checkpoint_path}' does not exist.")
    
    # Load the model and tokenizer from the local checkpoint
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)

    input_text = "translate Hindi to Parsed: " + input_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# Test the inference function with each checkpoint
test_word = "कुलाधिपति"
for checkpoint in checkpoints:
    try:
        parsed_output = generate_parsed_output(test_word, checkpoint)
        print(f"Parsed output for '{test_word}' from '{checkpoint}': {parsed_output}")
    except ValueError as e:
        print(e)