"""
测试 Passformer 优化序列生成数据集加载
测试 tokenized_data_dir 中的预处理数据集
"""
import os
import torch
from datasets import load_from_disk, DatasetDict

from src.config import load_config

# 配置路径
CONFIG_PATH = "configs/passformer_gallvm_seq2seq.yaml"


def test_dataset_loading():
    """测试数据集加载"""
    cfg = load_config(CONFIG_PATH)
    tokenized_data_dir = cfg["data"]["tokenized_data_dir"]
    
    print(f"[INFO] 尝试加载数据集: {tokenized_data_dir}")
    
    if not os.path.exists(tokenized_data_dir):
        print(f"[ERROR] 数据集路径不存在: {tokenized_data_dir}")
        print("[INFO] 请先运行训练脚本生成 tokenized 数据集")
        return False
    
    # 加载数据集
    dataset = load_from_disk(tokenized_data_dir)
    print(f"[OK] 数据集加载成功!")
    print(f"[INFO] 数据集类型: {type(dataset)}")
    
    return dataset


def test_dataset_structure(dataset):
    """测试数据集结构"""
    print("\n" + "=" * 50)
    print("数据集结构检查")
    print("=" * 50)
    
    # 检查是否是 DatasetDict
    if isinstance(dataset, DatasetDict):
        print(f"[OK] 数据集为 DatasetDict 类型")
        print(f"[INFO] 包含的 splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"\n--- {split_name} split ---")
            print(f"  样本数量: {len(split_data)}")
            print(f"  特征列: {split_data.column_names}")
            print(f"  特征类型: {split_data.features}")
    else:
        print(f"[INFO] 数据集为普通 Dataset 类型")
        print(f"  样本数量: {len(dataset)}")
        print(f"  特征列: {dataset.column_names}")
        print(f"  特征类型: {dataset.features}")
    
    return True


def test_dataset_fields(dataset):
    """测试数据集字段是否符合预期"""
    print("\n" + "=" * 50)
    print("数据集字段检查")
    print("=" * 50)
    
    required_fields = ['input_ids', 'attention_mask', 'labels']
    
    # 获取第一个 split 的数据
    if isinstance(dataset, DatasetDict):
        sample_split = list(dataset.keys())[0]
        sample_data = dataset[sample_split]
    else:
        sample_data = dataset
    
    columns = sample_data.column_names
    
    for field in required_fields:
        if field in columns:
            print(f"[OK] 字段 '{field}' 存在")
        else:
            print(f"[ERROR] 缺少必需字段 '{field}'")
            return False
    
    return True


def test_sample_data(dataset, num_samples=3):
    """查看样本数据"""
    print("\n" + "=" * 50)
    print(f"样本数据预览 (前 {num_samples} 条)")
    print("=" * 50)
    
    # 获取训练集数据
    if isinstance(dataset, DatasetDict):
        if 'train' in dataset:
            sample_data = dataset['train']
        else:
            sample_data = dataset[list(dataset.keys())[0]]
    else:
        sample_data = dataset
    
    for i in range(min(num_samples, len(sample_data))):
        sample = sample_data[i]
        print(f"\n--- 样本 {i + 1} ---")
        
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        labels = sample['labels']
        
        # 转换为 tensor 以便查看形状
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  attention_mask shape: {attention_mask.shape}")
        print(f"  labels shape: {labels.shape}")
        
        # 显示序列长度（非 padding 部分）
        non_pad_len = attention_mask.sum().item()
        print(f"  有效 token 数量: {non_pad_len}")
        
        # 显示前几个 token
        print(f"  input_ids (前10个): {input_ids[:10].tolist()}")
        print(f"  labels (前10个): {labels[:10].tolist()}")


def test_data_shape_consistency(dataset):
    """测试数据形状一致性"""
    print("\n" + "=" * 50)
    print("数据形状一致性检查")
    print("=" * 50)
    
    cfg = load_config(CONFIG_PATH)
    encoder_maxlen = cfg["data"]["encoder_maxlen"]
    decoder_maxlen = cfg["data"]["decoder_maxlen"]
    
    print(f"[INFO] 配置中的 encoder_maxlen: {encoder_maxlen}")
    print(f"[INFO] 配置中的 decoder_maxlen: {decoder_maxlen}")
    
    # 获取数据
    if isinstance(dataset, DatasetDict):
        sample_data = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
    else:
        sample_data = dataset
    
    # 检查前几个样本的形状
    num_check = min(10, len(sample_data))
    shape_ok = True
    
    for i in range(num_check):
        sample = sample_data[i]
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        input_len = len(input_ids) if hasattr(input_ids, '__len__') else input_ids.shape[0]
        labels_len = len(labels) if hasattr(labels, '__len__') else labels.shape[0]
        
        if input_len != encoder_maxlen:
            print(f"[WARN] 样本 {i}: input_ids 长度 {input_len} != encoder_maxlen {encoder_maxlen}")
            shape_ok = False
        
        # decoder 可能使用动态 padding，所以这里只检查是否超过最大长度
        if labels_len > decoder_maxlen:
            print(f"[WARN] 样本 {i}: labels 长度 {labels_len} > decoder_maxlen {decoder_maxlen}")
            shape_ok = False
    
    if shape_ok:
        print(f"[OK] 前 {num_check} 个样本形状检查通过")
    
    return shape_ok


def test_batch_loading(dataset, batch_size=4):
    """测试批量加载"""
    print("\n" + "=" * 50)
    print(f"批量加载测试 (batch_size={batch_size})")
    print("=" * 50)
    
    from torch.utils.data import DataLoader
    
    if isinstance(dataset, DatasetDict):
        sample_data = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
    else:
        sample_data = dataset
    
    # 设置格式为 torch
    sample_data = sample_data.with_format("torch")
    
    # 创建 DataLoader
    dataloader = DataLoader(sample_data, batch_size=batch_size, shuffle=False)
    
    # 获取一个 batch
    batch = next(iter(dataloader))
    
    print(f"[OK] 批量加载成功!")
    print(f"  Batch keys: {batch.keys()}")
    print(f"  input_ids batch shape: {batch['input_ids'].shape}")
    print(f"  attention_mask batch shape: {batch['attention_mask'].shape}")
    print(f"  labels batch shape: {batch['labels'].shape}")
    
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Passformer 数据集测试")
    print("=" * 60)
    
    # 1. 测试数据集加载
    dataset = test_dataset_loading()
    if dataset is False:
        return
    
    # 2. 测试数据集结构
    test_dataset_structure(dataset)
    
    # 3. 测试数据集字段
    test_dataset_fields(dataset)
    
    # 4. 查看样本数据
    test_sample_data(dataset)
    
    # 5. 测试形状一致性
    test_data_shape_consistency(dataset)
    
    # 6. 测试批量加载
    test_batch_loading(dataset)
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

