"""
测试 Passformer tokenized 数据集的脚本

验证 tokenize 后的数据集格式、形状和统计信息是否正确。

Usage:
    python -m src.tests.test_tokenized_passformer_dataset \
        --data_dir /path/to/tokenized/dataset \
        --inst2vec_tokenizer_id /path/to/inst2vec/tokenizer \
        --opti_seq_tokenizer_id /path/to/optiseq/tokenizer \
        --num_samples 5
"""
import os
import argparse
import torch
import numpy as np
from collections import Counter
from datasets import load_from_disk, DatasetDict

from src.model import Inst2VecTokenizer, OptiSeqTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test tokenized Passformer dataset"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the tokenized dataset directory"
    )
    parser.add_argument(
        "--inst2vec_tokenizer_id", type=str, default=None,
        help="Path to Inst2Vec tokenizer (optional, for decoding samples)"
    )
    parser.add_argument(
        "--opti_seq_tokenizer_id", type=str, default=None,
        help="Path to OptiSeq tokenizer (optional, for decoding samples)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of samples to display (default: 5)"
    )
    return parser.parse_args()


def check_dataset_structure(dataset):
    """检查数据集结构"""
    print("\n" + "="*60)
    print("1. 数据集结构检查")
    print("="*60)
    
    if isinstance(dataset, DatasetDict):
        print(f"  数据集类型: DatasetDict")
        print(f"  分割: {list(dataset.keys())}")
        for split, ds in dataset.items():
            print(f"\n  [{split}]")
            print(f"    样本数: {len(ds)}")
            print(f"    列名: {ds.column_names}")
            print(f"    特征: {ds.features}")
    else:
        print(f"  数据集类型: Dataset")
        print(f"  样本数: {len(dataset)}")
        print(f"  列名: {dataset.column_names}")
        print(f"  特征: {dataset.features}")
    
    return True


def check_sample_shapes(dataset, num_samples=5):
    """检查样本形状"""
    print("\n" + "="*60)
    print("2. 样本形状检查")
    print("="*60)
    
    # 获取第一个 split 的数据
    if isinstance(dataset, DatasetDict):
        ds = dataset[list(dataset.keys())[0]]
    else:
        ds = dataset
    
    required_fields = ['input_ids', 'attention_mask', 'labels']
    optional_fields = ['autophase']
    
    missing_fields = [f for f in required_fields if f not in ds.column_names]
    present_optional = [f for f in optional_fields if f in ds.column_names]
    
    if missing_fields:
        print(f"  ❌ 缺少必要字段: {missing_fields}")
        return False
    
    print(f"  ✅ 所有必要字段存在: {required_fields}")
    if present_optional:
        print(f"  ✅ 可选字段存在: {present_optional}")
    
    # 检查前几个样本的形状
    print(f"\n  前 {min(num_samples, len(ds))} 个样本的形状:")
    print(f"  {'Index':<8} {'input_ids':<15} {'attention_mask':<18} {'labels':<15}")
    print(f"  {'-'*56}")
    
    shapes_consistent = True
    first_shape = None
    
    for i in range(min(num_samples, len(ds))):
        sample = ds[i]
        
        # 转换为 tensor 以获取形状
        input_ids = torch.tensor(sample['input_ids']) if not isinstance(sample['input_ids'], torch.Tensor) else sample['input_ids']
        attention_mask = torch.tensor(sample['attention_mask']) if not isinstance(sample['attention_mask'], torch.Tensor) else sample['attention_mask']
        labels = torch.tensor(sample['labels']) if not isinstance(sample['labels'], torch.Tensor) else sample['labels']
        
        shape_info = (input_ids.shape, attention_mask.shape, labels.shape)
        
        if first_shape is None:
            first_shape = shape_info
        elif shape_info != first_shape:
            shapes_consistent = False
        
        print(f"  {i:<8} {str(list(input_ids.shape)):<15} {str(list(attention_mask.shape)):<18} {str(list(labels.shape)):<15}")
    
    if shapes_consistent:
        print(f"\n  ✅ 所有样本形状一致")
    else:
        print(f"\n  ⚠️ 样本形状不一致（可能是动态长度）")
    
    return True


def check_data_statistics(dataset):
    """检查数据统计信息"""
    print("\n" + "="*60)
    print("3. 数据统计信息")
    print("="*60)
    
    if isinstance(dataset, DatasetDict):
        for split, ds in dataset.items():
            print(f"\n  [{split}] ({len(ds)} 样本)")
            _compute_split_stats(ds)
    else:
        _compute_split_stats(dataset)


def _compute_split_stats(ds, max_samples=1000):
    """计算单个 split 的统计信息"""
    # 采样（避免处理过大数据集）
    sample_size = min(len(ds), max_samples)
    indices = np.random.choice(len(ds), sample_size, replace=False)
    
    input_lens = []
    label_lens = []
    pad_ratios = []
    ignored_ratios = []  # labels 中 -100 的比例
    
    for idx in indices:
        sample = ds[int(idx)]
        
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        labels = sample['labels']
        
        # 转换为 list 以便处理
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        if hasattr(attention_mask, 'tolist'):
            attention_mask = attention_mask.tolist()
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        
        input_lens.append(len(input_ids))
        
        # 计算实际长度（非 padding）
        actual_len = sum(attention_mask)
        pad_ratio = 1 - (actual_len / len(input_ids)) if len(input_ids) > 0 else 0
        pad_ratios.append(pad_ratio)
        
        # 计算 labels 中有效 token 数（非 -100）
        valid_labels = [l for l in labels if l != -100]
        label_lens.append(len(valid_labels))
        ignored_ratio = 1 - (len(valid_labels) / len(labels)) if len(labels) > 0 else 0
        ignored_ratios.append(ignored_ratio)
    
    print(f"    (基于 {sample_size} 个样本的统计)")
    print(f"\n    [Encoder Input (input_ids)]")
    print(f"      序列长度: min={min(input_lens)}, max={max(input_lens)}, mean={np.mean(input_lens):.1f}")
    print(f"      Padding 比例: min={min(pad_ratios)*100:.1f}%, max={max(pad_ratios)*100:.1f}%, mean={np.mean(pad_ratios)*100:.1f}%")
    
    print(f"\n    [Decoder Labels]")
    print(f"      有效 token 数: min={min(label_lens)}, max={max(label_lens)}, mean={np.mean(label_lens):.1f}")
    print(f"      忽略比例 (-100): min={min(ignored_ratios)*100:.1f}%, max={max(ignored_ratios)*100:.1f}%, mean={np.mean(ignored_ratios)*100:.1f}%")


def check_autophase(dataset):
    """检查 Autophase 特征"""
    print("\n" + "="*60)
    print("3.5 Autophase 特征检查")
    print("="*60)
    
    if isinstance(dataset, DatasetDict):
        ds = dataset[list(dataset.keys())[0]]
    else:
        ds = dataset
    
    if 'autophase' not in ds.column_names:
        print("  ⚠️ 数据集中不包含 autophase 特征")
        return False
    
    print("  ✅ 数据集包含 autophase 特征")
    
    # 检查 autophase 形状和统计信息
    sample = ds[0]
    autophase = sample['autophase']
    
    if hasattr(autophase, 'tolist'):
        autophase = autophase.tolist()
    elif isinstance(autophase, np.ndarray):
        autophase = autophase.tolist()
    
    print(f"\n  [Autophase 特征]")
    print(f"    维度: {len(autophase)}")
    print(f"    数据类型: {type(autophase[0]).__name__}")
    
    # 采样统计
    sample_size = min(100, len(ds))
    autophase_stats = []
    
    for i in range(sample_size):
        ap = ds[i]['autophase']
        if hasattr(ap, 'tolist'):
            ap = ap.tolist()
        elif isinstance(ap, np.ndarray):
            ap = ap.tolist()
        autophase_stats.append(ap)
    
    autophase_array = np.array(autophase_stats)
    
    print(f"\n    (基于 {sample_size} 个样本的统计)")
    print(f"    形状: {autophase_array.shape}")
    print(f"    均值范围: [{autophase_array.mean(axis=0).min():.2f}, {autophase_array.mean(axis=0).max():.2f}]")
    print(f"    标准差范围: [{autophase_array.std(axis=0).min():.2f}, {autophase_array.std(axis=0).max():.2f}]")
    print(f"    最小值: {autophase_array.min():.2f}")
    print(f"    最大值: {autophase_array.max():.2f}")
    
    # 检查是否有全零或异常值
    zero_count = np.sum(np.all(autophase_array == 0, axis=1))
    if zero_count > 0:
        print(f"    ⚠️ 发现 {zero_count} 个全零样本")
    else:
        print(f"    ✅ 无全零样本")
    
    return True


def check_special_tokens(dataset, inst2vec_tokenizer=None, opti_seq_tokenizer=None):
    """检查特殊 token"""
    print("\n" + "="*60)
    print("4. 特殊 Token 检查")
    print("="*60)
    
    if isinstance(dataset, DatasetDict):
        ds = dataset[list(dataset.keys())[0]]
    else:
        ds = dataset
    
    sample = ds[0]
    input_ids = sample['input_ids']
    labels = sample['labels']
    
    if hasattr(input_ids, 'tolist'):
        input_ids = input_ids.tolist()
    if hasattr(labels, 'tolist'):
        labels = labels.tolist()
    
    # Encoder 特殊 token
    print("\n  [Encoder Input]")
    if inst2vec_tokenizer:
        print(f"    BOS token ID: {inst2vec_tokenizer.bos_token_id}")
        print(f"    EOS token ID: {inst2vec_tokenizer.eos_token_id}")
        print(f"    PAD token ID: {inst2vec_tokenizer.pad_token_id}")
        print(f"    首位 token: {input_ids[0]} ({'✅ BOS' if input_ids[0] == inst2vec_tokenizer.bos_token_id else '❓'})")
        
        # 查找 EOS 位置
        eos_positions = [i for i, t in enumerate(input_ids) if t == inst2vec_tokenizer.eos_token_id]
        if eos_positions:
            print(f"    EOS 位置: {eos_positions}")
    else:
        print(f"    首位 token ID: {input_ids[0]}")
        print(f"    末位非 padding token 需要 tokenizer 才能分析")
    
    # Decoder 特殊 token
    print("\n  [Decoder Labels]")
    valid_labels = [l for l in labels if l != -100]
    print(f"    有效 labels 数量: {len(valid_labels)}")
    print(f"    忽略位置 (-100) 数量: {len(labels) - len(valid_labels)}")
    
    if opti_seq_tokenizer and valid_labels:
        print(f"    BOS token ID: {opti_seq_tokenizer.bos_token_id}")
        print(f"    EOS token ID: {opti_seq_tokenizer.eos_token_id}")
        print(f"    PAD token ID: {opti_seq_tokenizer.pad_token_id}")
        print(f"    首个有效 label: {valid_labels[0]} ({'✅ BOS' if valid_labels[0] == opti_seq_tokenizer.bos_token_id else '❓'})")
        print(f"    末个有效 label: {valid_labels[-1]} ({'✅ EOS' if valid_labels[-1] == opti_seq_tokenizer.eos_token_id else '❓'})")


def display_samples(dataset, inst2vec_tokenizer=None, opti_seq_tokenizer=None, num_samples=20):
    """显示样本内容"""
    print("\n" + "="*60)
    print("5. 样本内容预览")
    print("="*60)
    
    if isinstance(dataset, DatasetDict):
        ds = dataset[list(dataset.keys())[0]]
    else:
        ds = dataset
    
    for i in range(min(num_samples, len(ds))):
        sample = ds[i]
        print(f"\n  --- 样本 {i} ---")
        
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        labels = sample['labels']
        
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        if hasattr(attention_mask, 'tolist'):
            attention_mask = attention_mask.tolist()
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        
        # 显示 input_ids 的前 20 个
        print(f"  input_ids (前20): {input_ids[:20]}...")
        print(f"  attention_mask (前20): {attention_mask[:20]}...")
        
        # 显示有效 labels
        valid_labels = [l for l in labels if l != -100]
        print(f"  labels (有效部分): {valid_labels}")
        
        # 显示 autophase（如果存在）
        if 'autophase' in sample:
            autophase = sample['autophase']
            if hasattr(autophase, 'tolist'):
                autophase = autophase.tolist()
            elif isinstance(autophase, np.ndarray):
                autophase = autophase.tolist()
            # 格式化显示，保留2位小数
            autophase_str = [f"{v:.1f}" for v in autophase[:10]]
            print(f"  autophase (前10维): [{', '.join(autophase_str)}...]")
            print(f"  autophase 维度: {len(autophase)}, 范围: [{min(autophase):.1f}, {max(autophase):.1f}]")
        
        # 如果有 tokenizer，解码显示
        if inst2vec_tokenizer:
            try:
                decoded_input = inst2vec_tokenizer.decode(
                    torch.tensor(input_ids), 
                    skip_special_tokens=True
                )
                # 只显示前 200 字符
                if len(decoded_input) > 200:
                    decoded_input = decoded_input[:200] + "..."
                print(f"\n  Decoded Encoder Input (前200字符):\n    {decoded_input}")
            except Exception as e:
                print(f"  解码 encoder input 失败: {e}")
        
        if opti_seq_tokenizer and valid_labels:
            try:
                decoded_labels = opti_seq_tokenizer.decode(
                    torch.tensor(valid_labels),
                    skip_special_tokens=True
                )
                print(f"\n  Decoded Labels (优化序列):\n    {decoded_labels}")
            except Exception as e:
                print(f"  解码 labels 失败: {e}")


def run_training_compatibility_check(dataset):
    """检查与 HuggingFace Trainer 的兼容性"""
    print("\n" + "="*60)
    print("6. Trainer 兼容性检查")
    print("="*60)
    
    if isinstance(dataset, DatasetDict):
        ds = dataset[list(dataset.keys())[0]]
    else:
        ds = dataset
    
    sample = ds[0]
    
    checks = []
    warnings = []
    
    # 检查 input_ids
    input_ids = sample['input_ids']
    if hasattr(input_ids, '__len__'):
        checks.append(("input_ids 可迭代", True))
    else:
        checks.append(("input_ids 可迭代", False))
    
    # 检查 attention_mask
    attention_mask = sample['attention_mask']
    if hasattr(attention_mask, '__len__'):
        checks.append(("attention_mask 可迭代", True))
    else:
        checks.append(("attention_mask 可迭代", False))
    
    # 检查 labels
    labels = sample['labels']
    if hasattr(labels, '__len__'):
        checks.append(("labels 可迭代", True))
    else:
        checks.append(("labels 可迭代", False))
    
    # 检查形状一致性
    if len(input_ids) == len(attention_mask):
        checks.append(("input_ids 与 attention_mask 长度一致", True))
    else:
        checks.append(("input_ids 与 attention_mask 长度一致", False))
    
    # 检查 labels 中是否有 -100（这是可选的，取决于数据）
    labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
    has_ignore_index = -100 in labels_list
    
    # 采样多个样本检查是否有 -100
    sample_size = min(100, len(ds))
    samples_with_ignore = 0
    for i in range(sample_size):
        sample_labels = ds[i]['labels']
        sample_labels_list = sample_labels.tolist() if hasattr(sample_labels, 'tolist') else list(sample_labels)
        if -100 in sample_labels_list:
            samples_with_ignore += 1
    
    # 打印必要检查结果
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    # 打印 -100 检查结果（信息性，非强制）
    print(f"\n  [信息] Labels 中 -100 统计:")
    print(f"    - 采样 {sample_size} 个样本中，{samples_with_ignore} 个包含 -100")
    
    if samples_with_ignore == 0:
        print(f"    - ⚠️ 没有样本包含 -100")
        print(f"    - 可能原因: 所有 decoder 序列都 >= max_length，被 truncate 无 padding")
        print(f"    - 如果这是预期行为，则无需担心")
    elif samples_with_ignore < sample_size:
        print(f"    - ℹ️ 部分样本有 padding ({samples_with_ignore}/{sample_size})")
    else:
        print(f"    - ✅ 所有样本都有正确的 -100 padding mask")
    
    if all_passed:
        print(f"\n  ✅ 数据集核心结构与 HuggingFace Trainer 兼容!")
    else:
        print(f"\n  ❌ 存在兼容性问题，请检查上述失败项")
    
    return all_passed


def main():
    args = parse_args()
    
    print("="*60)
    print("Passformer Tokenized Dataset 测试")
    print("="*60)
    print(f"数据集路径: {args.data_dir}")
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = load_from_disk(args.data_dir)
    
    # 加载 tokenizer（可选）
    inst2vec_tokenizer = None
    opti_seq_tokenizer = None
    
    if args.inst2vec_tokenizer_id:
        print(f"加载 Inst2Vec tokenizer: {args.inst2vec_tokenizer_id}")
        inst2vec_tokenizer = Inst2VecTokenizer.from_pretrained(args.inst2vec_tokenizer_id)
    
    if args.opti_seq_tokenizer_id:
        print(f"加载 OptiSeq tokenizer: {args.opti_seq_tokenizer_id}")
        opti_seq_tokenizer = OptiSeqTokenizer.from_pretrained(args.opti_seq_tokenizer_id)
    
    # 运行检查
    check_dataset_structure(dataset)
    check_sample_shapes(dataset, args.num_samples)
    check_data_statistics(dataset)
    check_autophase(dataset)
    check_special_tokens(dataset, inst2vec_tokenizer, opti_seq_tokenizer)
    display_samples(dataset, inst2vec_tokenizer, opti_seq_tokenizer, min(args.num_samples, 20))
    run_training_compatibility_check(dataset)
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()

