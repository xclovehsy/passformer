from transformers import AutoTokenizer
import logging
from transformers.utils import logging as hf_logging
import os
import torch
import math


def get_logger(logging_path, logging_name='train.log'):
    hf_logging.enable_default_handler()
    hf_logging.set_verbosity_info()

    file_handler = logging.FileHandler(os.path.join(logging_path, logging_name), mode="w", encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger = hf_logging.get_logger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger

def compute_metrics(eval_pred):
    logits, labels = eval_pred  # numpy arrays: logits shape (batch_size, seq_len, vocab_size), labels shape (batch_size, seq_len)
    
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    
    # 交叉熵计算 perplexity
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    perplexity = math.exp(loss.item())
    
    # 计算 Masked 位置的准确率
    # 标签为 -100 的位置不计入准确率
    mask = labels != -100
    
    # 预测词ID
    preds = torch.argmax(logits, dim=-1)
    
    # 只选mask位置的预测和标签
    masked_preds = preds[mask]
    masked_labels = labels[mask]
    
    correct = (masked_preds == masked_labels).sum().item()
    total = mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "perplexity": perplexity,
        "mask_accuracy": accuracy
    }


def convert_to_float(config_dict):
    """
    递归地遍历配置字典，将其中所有字符串形式的数字转换为 float 或 int 类型
    """
    for section, value in config_dict.items():
        if isinstance(value, dict):
            # 如果值是字典，递归调用
            config_dict[section] = convert_to_float(value)
        elif isinstance(value, str):
            try:
                # 尝试将字符串转换为 float
                config_dict[section] = float(value)
            except ValueError:
                try:
                    # 如果不能转换为 float，尝试转换为 int
                    config_dict[section] = int(value)
                except ValueError:
                    pass  # 如果两者都不行，则保留原始字符串
            
    return config_dict