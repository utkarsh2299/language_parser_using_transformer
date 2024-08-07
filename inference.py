from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./fine-tuned-t5-base')
tokenizer = T5Tokenizer.from_pretrained('./fine-tuned-t5-base')

def generate_parsed_output(input_text):
    input_text = "translate Hindi to Parsed: " + input_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(**inputs)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# Test the inference function
test_word = "example"
parsed_output = generate_parsed_output(test_word)
print(f"Parsed output for '{test_word}': {parsed_output}")
