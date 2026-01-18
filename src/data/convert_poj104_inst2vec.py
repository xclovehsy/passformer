import sys
import os
import csv
import json
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
import glob
from src.model.tokenizer import Inst2VecTokenizer
from pathlib import Path

# def load_single_file(file_path, label):
#     with open(file_path, 'r') as file:
#         return {'llvm': file.read(), 'label': label}

# def load_data(split_dir):
#     data = {'llvm': [], 'label': []}
#     split_dir = Path(split_dir)
#     labels = os.listdir(split_dir)
    
#     with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust max_workers as needed
#         futures = []
#         for label in tqdm(labels):
#             data_files = os.listdir(split_dir / label)
#             for data_file in data_files:
#                 file_path = split_dir / label / data_file
#                 futures.append(executor.submit(load_single_file, file_path, label))
        
#         for future in tqdm(as_completed(futures), total=len(futures), desc='Loading data'):
#             result = future.result()
#             data['llvm'].append(result['llvm'])
#             data['label'].append(result['label'])
    
#     return data

def load_data(split_dir):
    data = {'llvm': [], 'label': []}
    split_dir = Path(split_dir)
    labels = os.listdir(split_dir)
    
    samples = []
    for label in tqdm(labels):
        data_files = os.listdir(split_dir / label)
        for data_file in data_files:
            file_path = split_dir / label / data_file
            samples.append((file_path, label))
    
    for sample in tqdm(samples, desc='Loading data'):
        with open(sample[0], 'r') as file:
            data['llvm'].append(file.read())
            data['label'].append(sample[1])
            
    return data


def tokenize_func(example, tokenizer):
    return tokenizer(
        example['llvm'], 
        max_length=512, 
        truncation=True,
        padding=True
    )

if __name__ == "__main__":
    data_dir = "/home/xucong24/Compiler/datasets/poj104"
    save_dir = "/home/xucong24/Compiler/datasets"
    tokenizer_path = '/home/xucong24/Compiler/checkpoints/Inst2VecTokenizer'
    
    
    train_data = load_data(os.path.join(data_dir, 'ir_train'))
    test_data = load_data(os.path.join(data_dir, 'ir_test'))
    val_data = load_data(os.path.join(data_dir, 'ir_val'))

    dataset_dict = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'test': Dataset.from_dict(test_data),
        'val': Dataset.from_dict(val_data)
    })
    
    dataset_dict.save_to_disk(os.path.join(save_dir, 'POJ104Dataset'))
    
    tokenizer = Inst2VecTokenizer.from_pretrained(tokenizer_path)
    tokenized_dataset_dict = dataset_dict.map(tokenize_func, 
                                              batched=False, 
                                              fn_kwargs={'tokenizer': tokenizer},
                                              num_proc=16
                                              ).remove_columns(['llvm', 'label'])

    tokenized_dataset_dict.save_to_disk(os.path.join(save_dir, 'POJ104TokenizedDataset'))
    
    
    