import torch
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback, T5Tokenizer
from datasets import Dataset, load_metric
import pandas as pd
import preprocessing

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(logs)

def train_model(data, model_size='small', verbose=False):
    # Flatten the scenes and repeat the descriptions to match the number of scenes
    all_scenes = [scene for scenes in data['Scenes'] for scene in scenes]
    all_descriptions = [desc for desc, scenes in zip(data['Description'], data['Scenes']) for _ in scenes]

    # Split the data into training and testing sets
    train_scenes, test_scenes, train_descs, test_descs = train_test_split(all_scenes, all_descriptions, test_size=0.2, random_state=0)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(f't5-{model_size}')

    # Tokenize the scenes and descriptions
    train_encodings = tokenizer(train_scenes, padding='max_length', truncation=True, max_length=512)
    train_labels = tokenizer(train_descs, padding='max_length', truncation=True, max_length=512)
    test_encodings = tokenizer(test_scenes, padding='max_length', truncation=True, max_length=512)
    test_labels = tokenizer(test_descs, padding='max_length', truncation=True, max_length=512)

    # Create dataset objects
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels['input_ids'],
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': test_labels['input_ids'],
    })

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Load the T5 model
    model = T5ForConditionalGeneration.from_pretrained(f't5-{model_size}')

    # Check if a CUDA or MPS device is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        fp16 = True
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        fp16 = False
    else:
        device = torch.device('cpu')
        fp16 = False
    print(f"Using device: {device}")

    # Move the model to the appropriate device
    model.to(device)
    print("Model moved to device")

    # Define training arguments with mixed precision if CUDA is available
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        fp16=fp16,  # Enable mixed precision training only if on CUDA
        dataloader_num_workers=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
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
    if verbose:
        print("Starting training...")
    trainer.train()

    # Evaluate the model using ROUGE metric
    metric = load_metric("rouge")

    if verbose:
        print("Starting prediction...")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        shuffle=False
    )

    model.eval()
    if fp16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

    predictions = []
    references = []

    with torch.no_grad():  # Disable gradient calculation
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            if fp16 and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=512)
            else:
                generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=512)

            pred_texts = [preprocessing.tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated_ids]
            predictions.extend(pred_texts)

            labels = batch['labels'].cpu().numpy()
            ref_texts = [preprocessing.tokenizer.decode(l, skip_special_tokens=True).strip() for l in labels]
            references.extend(ref_texts)

            # Explicitly free up memory
            del input_ids, attention_mask, generated_ids, labels
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print predictions for inspection
    if verbose:
        print("Predictions:")
        for i, pred_text in enumerate(pred_texts):
            print(f"{i + 1}: {pred_text}")

    results = metric.compute(predictions=predictions, references=references)

    # Print evaluation results
    if verbose:
        print("\nEvaluation Results:")
        for key, value in results.items():
            print(f"{key}: {value}")

    return model

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)