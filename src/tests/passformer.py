from transformers import (
    EncoderDecoderConfig,
    EncoderDecoderModel,
    AutoConfig,
    GPT2Config,
    GPT2LMHeadModel
)
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import os

from src.model import (
    Inst2VecTokenizer,
    OptiSeqTokenizer,
    PassformerModel,
    PassformerConfig
)
from src.data import create_tokenize_fn

from datasets import load_from_disk

from transformers import (
    GPT2Config,
    AutoModel
)


if __name__ == "__main__":
    # load tokenizer
    encoder_tokenizer = Inst2VecTokenizer.from_pretrained("D:/dev/passformer/checkpoints/Inst2VecTokenizer")
    decoder_tokenizer = OptiSeqTokenizer.from_pretrained("D:/dev/passformer/checkpoints/OptiSeqTokenizer")

    config = {
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 6,
        "bos_token_id": 126,
        "eos_token_id": 127,
        "vocab_size": 128,
        "add_cross_attention": True,
        "architectures": [
            "GPT2LMHeadModel"
        ],
        "task_specific_params": {
            "text-generation": {
            "do_sample": True,
            "max_length": 50
            }
        },
    }

    decoder_config = GPT2Config(**config)
    decoder = GPT2LMHeadModel(decoder_config)
    encoder = AutoModel.from_pretrained('D:/dev/passformer/checkpoints/final_model')

    model = PassformerModel(encoder=encoder, decoder=decoder)
    print(model.config)
    print(model)

    dataset = load_from_disk("D:/dev/passformer/datasets/ga_llvm_100")
    print(dataset)
    
    # Tokenize 数据集
    tokenize_fn = create_tokenize_fn(
        encoder_tokenizer,
        decoder_tokenizer,
        128,
        32,
        include_autophase=True
    )

    remove_columns = [
        'Benchmark', 'CpuInfo', 'IrInstructionCountO0',
        'IrInstructionCountO3', 'IrInstructionCountOz',
        'InstCount', 'Reward', 'LLVM_IR', 'Commandline', 'Autophase'
    ]
    existing_columns = set(dataset.column_names)
    remove_columns = [col for col in remove_columns if col in existing_columns]

    # 在 Windows 上，多进程会导致每个子进程重新加载模型，非常耗时
    # 建议使用 num_proc=1 或者使用 Linux 系统
    tokenized_data = dataset.map(
        tokenize_fn,
        batched=False,
        remove_columns=remove_columns,
        num_proc=1,  # Windows 上建议使用单进程，Linux 可以使用多进程
        desc="Tokenizing"
    )

    print(tokenized_data)
    print(tokenized_data[0])
    tokenized_data = tokenized_data.train_test_split(test_size=0.2, seed=42)

    # tokenized_data = tokenized_data.train_test_split(test_size=0.2, seed=42)
    # print(tokenized_data)

    from transformers import Trainer, TrainingArguments


    training_args = TrainingArguments(
        # output_dir="./checkpoints/passformer_test_output",  # 添加输出目录
        num_train_epochs=5,              # 30 epochs 通常足够，避免过拟合
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        # gradient_accumulation_steps=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,               # 多保留一个检查点
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # 用 eval_loss 选最佳模型
        greater_is_better=False,          # loss 越小越好
        logging_strategy="steps",
        logging_steps=10,                 # 增加logging_steps，避免loss显示异常
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        report_to="tensorboard",
        overwrite_output_dir=False,
        # 添加以下参数来确保训练和评估的一致性
        dataloader_drop_last=False,      # 不要丢弃最后一个不完整的batch
        prediction_loss_only=True,        # 只计算loss，不计算其他指标
        remove_unused_columns=True,       # 移除未使用的列
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test']
    )

    trainer.train()

    with open("D:/dev/passformer/src/utils/qsort.ll", "r") as f:
        llvm = f.read()
    input_ids = encoder_tokenizer(llvm, max_length=50, return_tensors="pt")
    labels = decoder_tokenizer("mem2reg instcombine gvn", max_length=25, return_tensors="pt")
    autophase = torch.randn(1, 56)
    print(input_ids)
    print(labels)

    outputs = model(input_ids=input_ids["input_ids"], labels=labels["input_ids"], autophase=autophase)
    loss, logits = outputs.loss, outputs.logits
    print(loss)
    print(logits)

    print("before generate:")
    print(model.generate(input_ids["input_ids"], autophase=autophase))

    model.save_pretrained("D:/dev/passformer/checkpoints/passformer_test")
    
    model_test = PassformerModel.from_pretrained("D:/dev/passformer/checkpoints/passformer_test")
    print(model_test.generate(input_ids["input_ids"], autophase=autophase))
