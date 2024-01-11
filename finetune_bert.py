import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL = "bert-base-cased"
SEED = 42

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Loading dataset
raw_datasets = load_dataset("imdb")

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Creating smaller train and test datasets
small_train_dataset = tokenized_datasets["train"].shuffle(seed=SEED).select(range(1000))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=SEED).select(range(1000))

# full_train_dataset = tokenized_datasets["train"]
# full_test_dataset = tokenized_datasets["test"]

# Instantiating model and metric
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
metric = load_metric("accuracy")

# Training arguments setup
training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    compute_metrics=compute_metrics,
)

# Training and evaluation
trainer.train()
trainer.save_model("bert_imdb")
trainer.evaluate()
