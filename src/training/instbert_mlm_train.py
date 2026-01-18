"""InstBERT MLM (Masked Language Model) training script."""
import os
import argparse
from datetime import datetime

from transformers import (
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk

from src.config import load_config
from src.utils.utils import get_logger
from src.model import Inst2VecTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config file"
    )
    return parser.parse_args()


def get_model_and_tokenizer(cfg, logger):
    """Load model and tokenizer, resize embeddings and sync special token ids."""
    model_id = cfg["model"]["model_id"]
    tokenizer_id = cfg["model"]["tokenizer_id"]
    
    # Load tokenizer
    logger.info(f"Loading Inst2Vec tokenizer from {tokenizer_id}")
    tokenizer = Inst2VecTokenizer.from_pretrained(tokenizer_id)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Load model
    logger.info(f"Loading model from {model_id}")
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    
    # Resize token embeddings
    old_vocab_size = model.get_input_embeddings().num_embeddings
    logger.info(f"Original model vocab size: {old_vocab_size}")
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model vocab to {len(tokenizer)}")
    
    # 同步 tokenizer 的特殊 token id 到模型配置
    model.config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.cls_token_id = tokenizer.cls_token_id
    model.config.sep_token_id = tokenizer.sep_token_id
    logger.info(f"Synced special token ids: pad={model.config.pad_token_id}, bos={model.config.bos_token_id}, eos={model.config.eos_token_id}")
    
    return model, tokenizer


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Config values
    data_dir = cfg["data"]["data_dir"]
    max_length = cfg["data"]["max_length"]
    base_work_dir = cfg["output"]["base_work_dir"]
    mlm_probability = cfg["mlm"]["mlm_probability"]
    args_cfg = cfg["training_args"]
    
    # Create work directory
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(base_work_dir, time_str)
    os.makedirs(work_dir, exist_ok=True)
    
    # Setup logging
    logger = get_logger(work_dir)
    logger.info(f"Work directory created at {work_dir}")
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(cfg, logger)
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_from_disk(data_dir)
    
    def tokenize_fn(example):
        return tokenizer(
            example['llvm'],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
    
    # Tokenize dataset
    logger.info(f"Tokenizing dataset with max_length={max_length}")
    tokenized_data = dataset.map(
        tokenize_fn,
        batched=False,
        num_proc=32,
        remove_columns=['llvm', 'label']
    )
    logger.info("Tokenization finished")
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=work_dir,
        logging_dir=work_dir,
        **args_cfg
    )
    
    # Trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data.get('test')
    )
    
    # Train
    logger.info("Starting MLM training")
    trainer.train()
    
    # Save model
    final_model_dir = os.path.join(work_dir, "final_model")
    logger.info(f"Saving model and tokenizer to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
