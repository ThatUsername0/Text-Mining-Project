from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize the tokenizer and model for T5-long
tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large")
model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large")

def summarise_script(tokenized_scenes, climax_position_percentage):
    # Flatten the tokenized scenes into a single list of tokens up to the climax position
    flat_tokens = [token for scene in tokenized_scenes for token in scene]
    climax_token_count = int((climax_position_percentage / 100) * len(flat_tokens))

    tokens_up_to_climax = flat_tokens[:climax_token_count]

    # Convert the tokens back to text
    text_up_to_climax = " ".join(tokens_up_to_climax)

    # Tokenize the input and generate the summary
    inputs = tokenizer.encode("summarize: " + text_up_to_climax, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary