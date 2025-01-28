#!/usr/bin/env python3

import argparse
import logging
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Deepseek R1 7B on a public dataset")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="path/to/deepseek-r1-7b",
        help="Path or identifier of the base model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/deepseek-r1-7b-finetuned-wikitext2",
        help="Where to save the final model and checkpoints",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Token sequence length for each training example",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate for AdamW",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Loading model and tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # 1. Load a public dataset: wikitext-2
    logger.info("Loading the wikitext-2 dataset...")
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    # This gives us splits: train, validation, test

    # 2. Preprocessing the data
    #    We'll group the text into blocks of block_size for a causal language modeling task
    block_size = args.block_size

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    logger.info("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,  # Adjust for multi-processing if desired
        remove_columns=["text"]
    )

    def group_texts(examples):
        """Group texts into blocks of block_size."""
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {}
        for k in concatenated.keys():
            result[k] = [
                concatenated[k][i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
        return result

    logger.info("Grouping tokens into blocks...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
    )

    # 3. Define train and eval datasets
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # 4. Prepare training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=200,  # Evaluate every 200 steps
        save_steps=200,  # Save a checkpoint every 200 steps
        logging_steps=50,
        learning_rate=args.learning_rate,
        fp16=True if torch.cuda.is_available() else False,
        seed=args.seed,
        report_to="none",  # or "wandb"/"tensorboard" if integrated
    )

    # 5. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 6. Train
    logger.info("Starting training...")
    trainer.train()

    # 7. Save the final model
    logger.info("Saving final model...")
    trainer.save_model(args.output_dir)

    logger.info("Done!")

if __name__ == "__main__":
    main()
