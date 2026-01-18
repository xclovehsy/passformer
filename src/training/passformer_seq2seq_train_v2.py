"""LLVM Optimization Sequence Generation training script."""
import os
import argparse
from datetime import datetime

import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    EncoderDecoderModel,
    AutoModel,
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Config
)
from datasets import load_from_disk, DatasetDict

from src.config import load_config  
from src.utils.utils import get_logger
from src.model import Inst2VecTokenizer, OptiSeqTokenizer
from src.model import PassformerModel, PassformerConfig
from src.data import create_tokenize_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config file"
    )
    return parser.parse_args()


def get_model(cfg, logger):
    """Build the encoder-decoder model."""
    instbert_id = cfg["model"]["instbert_id"]
    inst2vec_tokenizer_id = cfg["model"]["inst2vec_tokenizer_id"]
    opti_seq_tokenizer_id = cfg["model"]["opti_seq_tokenizer_id"]
    decoder_cfg = cfg['decoder_cfg']
    
    # 加载 tokenizer
    logger.info(f"Loading Inst2Vec tokenizer from {inst2vec_tokenizer_id}")
    inst2vec_tokenizer = Inst2VecTokenizer.from_pretrained(inst2vec_tokenizer_id)
    
    # Encoder
    logger.info(f"Loading encoder from {instbert_id}")
    encoder = AutoModel.from_pretrained(instbert_id)
    
    # Decoder
    logger.info(f"Loading OptiSeq tokenizer from {opti_seq_tokenizer_id}")
    opti_seq_tokenizer = OptiSeqTokenizer.from_pretrained(opti_seq_tokenizer_id)
    decoder_cfg = GPT2Config(**decoder_cfg)
    decoder = GPT2LMHeadModel(decoder_cfg)
    
    # Encoder-decoder model
    logger.info("Building Passformer model")
    model = PassformerModel(encoder=encoder, decoder=decoder)
    
    return model, inst2vec_tokenizer, opti_seq_tokenizer

def tokenize_dataset(cfg, logger, dataset, inst2vec_tokenizer, opti_seq_tokenizer):
    tokenize_fn = create_tokenize_fn(
        inst2vec_tokenizer,
        opti_seq_tokenizer,
        cfg["data"]["encoder_maxlen"],
        cfg["data"]["decoder_maxlen"],
        include_autophase=True
    )

    remove_columns = [
        'Benchmark', 'CpuInfo', 'IrInstructionCountO0',
        'IrInstructionCountO3', 'IrInstructionCountOz',
        'InstCount', 'Reward', 'LLVM_IR', 'Commandline', 'Autophase'
    ]
    existing_columns = set(dataset.column_names)
    remove_columns = [col for col in remove_columns if col in existing_columns]

    # dataset = dataset.select(range(100))

    tokenized_data = dataset.map(
        tokenize_fn,
        batched=False,
        remove_columns=remove_columns,
        num_proc=32, 
        desc="Tokenizing"
    )

    tokenized_data = tokenized_data.train_test_split(test_size=cfg["data"]["test_size"], seed=cfg["data"]["split_seed"])
    
    return tokenized_data

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Config values
    data_dir = cfg["data"]["data_dir"]
    base_work_dir = cfg["output"]["base_work_dir"]
    args_cfg = cfg["training_args"]
    
    # Create work directory
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(base_work_dir, time_str)
    os.makedirs(work_dir, exist_ok=True)
    
    # Setup logging
    logger = get_logger(work_dir)
    logger.info(f"Work directory created at {work_dir}")
    
    # Load model and tokenizers
    model, inst2vec_tokenizer, opti_seq_tokenizer = get_model(cfg, logger)
    
    # Load datasets
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_from_disk(data_dir)

    # tokenize dataset
    logger.info("Tokenizing dataset")
    tokenized_data = tokenize_dataset(cfg, logger, dataset, inst2vec_tokenizer, opti_seq_tokenizer)
    
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
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test']
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    final_model_dir = os.path.join(work_dir, "final_model")
    logger.info(f"Saving model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    inst2vec_tokenizer.save_pretrained(os.path.join(final_model_dir, 'encoder_tokenizer'))
    opti_seq_tokenizer.save_pretrained(os.path.join(final_model_dir, 'decoder_tokenizer'))
    logger.info("Training complete")


if __name__ == "__main__":
    main()

