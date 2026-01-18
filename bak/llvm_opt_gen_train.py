"""LLVM Optimization Sequence Generation training script."""
import os
import argparse
from datetime import datetime

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EncoderDecoderModel,
    ModernBertModel,
    GPT2LMHeadModel,
    GPT2Config
)
from datasets import load_from_disk

from src.config import load_config
from src.utils.utils import get_logger
from src.model import OptiSeqTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config file"
    )
    return parser.parse_args()


def get_model(cfg, logger):
    """Build the encoder-decoder model."""
    modern_bert_id = cfg["model"]["modern_bert_id"]
    opti_seq_tokenizer_id = cfg["model"]["opti_seq_tokenizer_id"]
    gpt2_cfg = cfg['gpt2_config']
    
    # Encoder
    logger.info(f"Loading encoder from {modern_bert_id}")
    modern_bert = ModernBertModel.from_pretrained(modern_bert_id)
    modern_bert_tokenizer = AutoTokenizer.from_pretrained(modern_bert_id)
    
    # Decoder
    logger.info(f"Loading OptiSeq tokenizer from {opti_seq_tokenizer_id}")
    opti_seq_tokenizer = OptiSeqTokenizer.from_pretrained(opti_seq_tokenizer_id)
    gpt2_config = GPT2Config(**gpt2_cfg)
    gpt2 = GPT2LMHeadModel(gpt2_config)
    
    # Encoder-decoder model
    logger.info("Building encoder-decoder model")
    model = EncoderDecoderModel(encoder=modern_bert, decoder=gpt2)
    model.config.decoder_start_token_id = opti_seq_tokenizer.eos_token_id
    model.config.pad_token_id = opti_seq_tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    return model, modern_bert_tokenizer, opti_seq_tokenizer


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Config values
    data_dir = cfg["data"]["data_dir"]
    encoder_maxlen = cfg["data"]["encoder_maxlen"]
    decoder_maxlen = cfg["data"]["decoder_maxlen"]
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
    model, modern_bert_tokenizer, opti_seq_tokenizer = get_model(cfg, logger)
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_from_disk(data_dir)
    
    def tokenize_fn(examples):
        llvm_ir = examples['LLVM_IR']
        input_ids = modern_bert_tokenizer(
            llvm_ir, 
            truncation=True,
            padding="max_length",
            max_length=encoder_maxlen,
            return_tensors='pt'
        ).input_ids

        commandline = examples['Commandline']
        labels = opti_seq_tokenizer(
            commandline,
            truncation=True,
            padding=True,
            max_length=decoder_maxlen,
        )['input_ids']

        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    # Tokenize dataset
    logger.info("Tokenizing dataset")
    remove_columns = [
        'Benchmark', 'CpuInfo', 'IrInstructionCountO0', 
        'IrInstructionCountO3', 'IrInstructionCountOz', 
        'InstCount', 'Autophase', 'Reward', 'LLVM_IR', 'Commandline'
    ]
    tokenized_data = dataset.map(
        tokenize_fn, 
        batched=True,
        remove_columns=remove_columns
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
    modern_bert_tokenizer.save_pretrained(os.path.join(final_model_dir, 'modern_bert_tokenizer'))
    opti_seq_tokenizer.save_pretrained(os.path.join(final_model_dir, 'opti_seq_tokenizer'))
    logger.info("Training complete")


if __name__ == "__main__":
    main()

