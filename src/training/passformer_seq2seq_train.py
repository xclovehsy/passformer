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
    gpt2_cfg = cfg['gpt2_config']
    
    # 加载 tokenizer
    logger.info(f"Loading Inst2Vec tokenizer from {inst2vec_tokenizer_id}")
    inst2vec_tokenizer = Inst2VecTokenizer.from_pretrained(inst2vec_tokenizer_id)
    
    # Encoder
    logger.info(f"Loading encoder from {instbert_id}")
    encoder = AutoModel.from_pretrained(instbert_id)
    
    # Decoder
    logger.info(f"Loading OptiSeq tokenizer from {opti_seq_tokenizer_id}")
    opti_seq_tokenizer = OptiSeqTokenizer.from_pretrained(opti_seq_tokenizer_id)
    gpt2_config = GPT2Config(**gpt2_cfg)
    gpt2 = GPT2LMHeadModel(gpt2_config)
    
    # Encoder-decoder model
    logger.info("Building encoder-decoder model")
    model = EncoderDecoderModel(encoder=encoder, decoder=gpt2)
    model.config.decoder_start_token_id = opti_seq_tokenizer.eos_token_id
    model.config.pad_token_id = opti_seq_tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    return model, inst2vec_tokenizer, opti_seq_tokenizer


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Config values
    data_dir = cfg["data"]["data_dir"]
    encoder_maxlen = cfg["data"]["encoder_maxlen"]
    decoder_maxlen = cfg["data"]["decoder_maxlen"]
    tokenized_data_dir = cfg["data"].get("tokenized_data_dir", None)
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
    
    # 尝试加载已 tokenize 的数据集
    logger.info(f"Loading tokenized dataset from {tokenized_data_dir}")
    tokenized_data = load_from_disk(tokenized_data_dir)

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

