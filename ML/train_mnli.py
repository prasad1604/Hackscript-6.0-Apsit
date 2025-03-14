import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the MultiNLI dataset
dataset = load_dataset("multi_nli")

# For quick testing, select a small subset of the training data:
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
small_val_dataset = dataset["validation_matched"].shuffle(seed=42).select(range(200))

# Use DistilBERT for a lighter model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding="max_length", max_length=128)

tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_val = small_val_dataset.map(tokenize_function, batched=True)

num_labels = 3

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./results_mnli_quick",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1,  # Only one epoch for a quick test
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

model.save_pretrained("./fine_tuned_mnli_quick")
tokenizer.save_pretrained("./fine_tuned_mnli_quick")
