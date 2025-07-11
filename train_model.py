# train_model.py

import os
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "google/flan-t5-small"
DATA_FILE  = "final_medchat_data.jsonl"  # from prepare_data.py
OUTPUT_DIR = "./medchat_model"

# ─── Load Tokenizer & Model ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ─── Load Dataset ───────────────────────────────────────────────────────────
# Manually read JSONL and wrap as a Hugging Face Dataset
with open(DATA_FILE, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f]
raw_ds = Dataset.from_list(examples)

# ─── Preprocessing ──────────────────────────────────────────────────────────
def preprocess_batch(batch):
    # tokenize inputs & targets, pad/truncate to fixed lengths
    inputs = tokenizer(
        batch["input"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    targets = tokenizer(
        batch["target"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

train_ds = raw_ds.map(
    preprocess_batch,
    batched=True,
    remove_columns=["input", "target"],
)

# ─── Training Arguments ──────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    fp16=True,           # if you have a GPU
    weight_decay=0.01,
)

# ─── Trainer Setup ─────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer,
)

# ─── Train & Save ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting training.")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
