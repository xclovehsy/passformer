"""ModernBERT MLM training script using base trainer."""
import argparse
from typing import Dict, Any

from transformers import AutoTokenizer

from src.training.base_trainer import BaseMLMTrainer, parse_args


class ModernBertMLMTrainer(BaseMLMTrainer):
    """Trainer for ModernBERT with standard HuggingFace tokenizer."""
    
    def load_tokenizer(self):
        """Load the HuggingFace tokenizer."""
        tokenizer_id = self.cfg.tokenizer_id or self.cfg.model_id
        self.logger.info(f"Loading tokenizer from {tokenizer_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def tokenize_function(self, examples) -> Dict[str, Any]:
        """Tokenize using HuggingFace tokenizer."""
        return self.tokenizer(
            examples['llvm'],
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt"
        )


def main():
    args = parse_args()
    trainer = ModernBertMLMTrainer(args.config)
    trainer.run(remove_columns=['llvm', 'label'])


if __name__ == "__main__":
    main()

