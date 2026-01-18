"""Tokenization function for Passformer dataset."""

import torch
from typing import Optional, Dict, Any


def create_tokenize_fn(
    encoder_tokenizer,
    decoder_tokenizer,
    encoder_max_length: int,
    decoder_max_length: int,
    include_autophase: bool = True
):

    def tokenize_fn(example: Dict[str, Any]) -> Dict[str, Any]:

        llvm_ir = example['LLVM_IR']
        encoder_output = encoder_tokenizer(
            llvm_ir,
            max_length=encoder_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # __call__ 方法会添加 batch 维度，需要先移除
        input_ids = encoder_output['input_ids'].squeeze(0).tolist()
        attention_mask = encoder_output['attention_mask'].squeeze(0).tolist()
        
        commandline = example['Commandline']
        decoder_output = decoder_tokenizer(
            commandline,
            max_length=decoder_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # __call__ 方法会添加 batch 维度，需要先移除
        labels = decoder_output['input_ids'].squeeze(0).tolist()
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        
        if include_autophase:
            autophase = example.get('Autophase', [])
            if isinstance(autophase, list):
                result['autophase'] = autophase
            else:
                result['autophase'] = list(autophase) if hasattr(autophase, '__iter__') else []
        
        return result
    
    return tokenize_fn

