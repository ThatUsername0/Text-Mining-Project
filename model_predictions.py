import torch
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from preprocessing import preprocess_script

# Load the large T5 model and tokenizer
model_checkpoint = 't5-large'
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, model_max_length=512)

# Path to the text file containing the Friends script
script_file = 'data/testing/friends_test_script.txt'

# Read the script from the text file
with open(script_file, 'r') as file:
    script = file.read()

# Preprocess the script
cleaned_script = preprocess_script(script)
print("Cleaned Script:")
print(cleaned_script)

# Tokenize the cleaned script
tokenized_script = tokenizer(cleaned_script, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
print("Tokenized Script:")
print(tokenized_script)

# Prepare input tensors
input_ids = tokenized_script['input_ids']
attention_mask = tokenized_script['attention_mask']

# Move model and tensors to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Generate description
model.eval()
predicted_description = ""

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        max_length=512, 
        num_beams=5, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95, 
        early_stopping=True
    )
    print("Model Outputs:")
    print(outputs)
    predicted_description = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

print("Predicted Description:")
print(predicted_description)

# Create a DataFrame with the script and predicted description
results_df = pd.DataFrame({
    "Script": [cleaned_script],  # Single script
    "Predicted Description": [predicted_description]
})

# Save the results to a CSV file
results_df.to_csv("generated_script_with_description.csv", index=False)

print("Results saved to generated_script_with_description.csv")
