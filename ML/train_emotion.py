import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def simplify_labels(example):
    # Convert multi-label list to single label (take the first one)
    if isinstance(example["labels"], list) and len(example["labels"]) > 0:
        example["labels"] = example["labels"][0]
    else:
        example["labels"] = 0
    return example

# Load the GoEmotions dataset (using the "simplified" configuration if available)
dataset = load_dataset("go_emotions", "simplified")

# Use a very small subset for quick training
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(500)).map(simplify_labels)
small_val_dataset = dataset["validation"].shuffle(seed=42).select(range(100)).map(simplify_labels)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_val = small_val_dataset.map(tokenize_function, batched=True)

# Try to get number of labels; if not, use max(label)+1
try:
    num_labels = tokenized_train.features["labels"].feature.num_classes
except AttributeError:
    num_labels = max(tokenized_train["labels"]) + 1

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./results_emotion",
    evaluation_strategy="epoch",       # Evaluate at end of each epoch.
    save_strategy="epoch",             # Save at end of each epoch.
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1,                # One epoch for fast training.
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

trainer.train()

model.save_pretrained("./fine_tuned_emotion")
tokenizer.save_pretrained("./fine_tuned_emotion")
