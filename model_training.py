import torch
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_metric, Dataset
import pandas as pd
import multiprocessing

# Set the start method for multiprocessing
multiprocessing.set_start_method("spawn", force=True)

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(logs)

def train_model(data, verbose=False):
    # Convert columns to string type
    data['Script'] = data['Script'].astype(str)
    data['Description'] = data['Description'].astype(str)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

    # Verify data sizes
    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")

    # Load the T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Tokenize the data
    def tokenize_function(examples):
        model_inputs = tokenizer(examples['Script'], padding='max_length', truncation=True, max_length=512)
        labels = tokenizer(examples['Description'], padding='max_length', truncation=True, max_length=512)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # Convert pandas DataFrame to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set the format of the dataset
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Verify datasets sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Load the T5 model
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,  # Adjust as needed
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=1,  # Log every step
    )

    # Set up logging callback if verbose is True
    callbacks = [LoggingCallback()] if verbose else []

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate the model using ROUGE metric
    metric = load_metric("rouge")

    # Generate predictions
    predictions = trainer.predict(test_dataset)
    pred_ids = predictions.predictions  # Access predictions directly
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]  # Access the logits if it's a tuple
    pred_ids = pred_ids.argmax(axis=-1)
    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # Get references
    references = test_data['Description'].tolist()

    # Align predictions and references
    aligned_preds = [pred for pred, ref in zip(pred_texts, references)]
    aligned_refs = [ref for pred, ref in zip(pred_texts, references)]

    results = metric.compute(predictions=aligned_preds, references=aligned_refs)

    # Print evaluation results
    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    return model
