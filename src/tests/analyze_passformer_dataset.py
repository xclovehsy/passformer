"""
分析 Passformer tokenized 数据集的分布

检查内容：
1. 重复样本分析（相同 input_ids）
2. 每个程序的优化序列数量分布
3. 序列长度分布
4. Autophase 特征分布
5. Labels 分布统计

Usage:
    python -m src.tests.analyze_passformer_dataset \
        --data_dir /path/to/tokenized/dataset \
        --output_dir /path/to/save/analysis
"""
import os
import argparse
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Passformer tokenized dataset distribution"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the tokenized dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Path to save analysis plots (optional)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum samples to analyze (default: all)"
    )
    return parser.parse_args()


def compute_input_hash(input_ids):
    """计算 input_ids 的哈希值用于快速比较"""
    if hasattr(input_ids, 'tolist'):
        input_ids = input_ids.tolist()
    return hashlib.md5(str(input_ids).encode()).hexdigest()


def analyze_duplicates(dataset, max_samples=None):
    """分析重复样本"""
    print("\n" + "="*70)
    print("1. 重复样本分析（相同 input_ids / 相同程序）")
    print("="*70)
    
    if isinstance(dataset, DatasetDict):
        ds = dataset[list(dataset.keys())[0]]
        split_name = list(dataset.keys())[0]
    else:
        ds = dataset
        split_name = "dataset"
    
    num_samples = len(ds) if max_samples is None else min(max_samples, len(ds))
    
    # 计算每个样本的 input_ids 哈希
    input_hashes = {}  # hash -> list of indices
    
    print(f"\n  分析 {num_samples} 个样本...")
    for i in tqdm(range(num_samples), desc="  Computing hashes"):
        sample = ds[i]
        h = compute_input_hash(sample['input_ids'])
        if h not in input_hashes:
            input_hashes[h] = []
        input_hashes[h].append(i)
    
    # 统计
    unique_programs = len(input_hashes)
    total_samples = num_samples
    
    # 每个程序的样本数量分布
    samples_per_program = [len(indices) for indices in input_hashes.values()]
    
    print(f"\n  [统计结果]")
    print(f"    总样本数: {total_samples}")
    print(f"    唯一程序数: {unique_programs}")
    print(f"    重复率: {(1 - unique_programs/total_samples)*100:.2f}%")
    print(f"\n  [每个程序的样本数分布]")
    print(f"    最小: {min(samples_per_program)}")
    print(f"    最大: {max(samples_per_program)}")
    print(f"    平均: {np.mean(samples_per_program):.2f}")
    print(f"    中位数: {np.median(samples_per_program):.1f}")
    
    # 分布统计
    count_distribution = Counter(samples_per_program)
    print(f"\n  [样本数分布详情]")
    for count, num_programs in sorted(count_distribution.items())[:20]:
        bar = "█" * min(num_programs, 50)
        print(f"    {count:3d} 个序列: {num_programs:5d} 个程序 {bar}")
    if len(count_distribution) > 20:
        print(f"    ... (共 {len(count_distribution)} 种不同的样本数)")
    
    return input_hashes, samples_per_program


def analyze_sequence_lengths(dataset, max_samples=None):
    """分析序列长度分布"""
    print("\n" + "="*70)
    print("2. 序列长度分布")
    print("="*70)
    
    if isinstance(dataset, DatasetDict):
        ds = dataset[list(dataset.keys())[0]]
    else:
        ds = dataset
    
    num_samples = len(ds) if max_samples is None else min(max_samples, len(ds))
    
    encoder_lengths = []  # 实际长度（非 padding）
    decoder_lengths = []  # 有效 labels 长度
    encoder_total_lengths = []  # 总长度（包括 padding）
    
    for i in tqdm(range(num_samples), desc="  Analyzing lengths"):
        sample = ds[i]
        
        attention_mask = sample['attention_mask']
        labels = sample['labels']
        input_ids = sample['input_ids']
        
        if hasattr(attention_mask, 'tolist'):
            attention_mask = attention_mask.tolist()
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        
        # Encoder 实际长度
        encoder_lengths.append(sum(attention_mask))
        encoder_total_lengths.append(len(input_ids))
        
        # Decoder 有效长度
        valid_labels = [l for l in labels if l != -100]
        decoder_lengths.append(len(valid_labels))
    
    print(f"\n  [Encoder 序列长度（实际内容，非 padding）]")
    print(f"    最小: {min(encoder_lengths)}")
    print(f"    最大: {max(encoder_lengths)}")
    print(f"    平均: {np.mean(encoder_lengths):.1f}")
    print(f"    中位数: {np.median(encoder_lengths):.1f}")
    print(f"    标准差: {np.std(encoder_lengths):.1f}")
    
    # 检查是否所有样本都达到 max_length
    max_len = max(encoder_total_lengths)
    at_max_count = sum(1 for l in encoder_lengths if l == max_len)
    print(f"    达到最大长度 ({max_len}) 的样本: {at_max_count} ({at_max_count/num_samples*100:.1f}%)")
    
    print(f"\n  [Decoder 序列长度（有效 labels）]")
    print(f"    最小: {min(decoder_lengths)}")
    print(f"    最大: {max(decoder_lengths)}")
    print(f"    平均: {np.mean(decoder_lengths):.1f}")
    print(f"    中位数: {np.median(decoder_lengths):.1f}")
    print(f"    标准差: {np.std(decoder_lengths):.1f}")
    
    return encoder_lengths, decoder_lengths


def analyze_labels_distribution(dataset, max_samples=None):
    """分析 labels（优化 pass）分布"""
    print("\n" + "="*70)
    print("3. Labels（优化 Pass）分布")
    print("="*70)
    
    if isinstance(dataset, DatasetDict):
        ds = dataset[list(dataset.keys())[0]]
    else:
        ds = dataset
    
    num_samples = len(ds) if max_samples is None else min(max_samples, len(ds))
    
    all_labels = []
    label_sequences = []
    
    for i in tqdm(range(num_samples), desc="  Analyzing labels"):
        sample = ds[i]
        labels = sample['labels']
        
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        
        valid_labels = [l for l in labels if l != -100]
        all_labels.extend(valid_labels)
        label_sequences.append(tuple(valid_labels))
    
    # Token 频率统计
    label_counter = Counter(all_labels)
    unique_labels = len(label_counter)
    
    print(f"\n  [Token 统计]")
    print(f"    总 token 数: {len(all_labels)}")
    print(f"    唯一 token 数: {unique_labels}")
    
    print(f"\n  [Top 20 最常见的 token ID]")
    for token_id, count in label_counter.most_common(20):
        percentage = count / len(all_labels) * 100
        bar = "█" * int(percentage * 2)
        print(f"    Token {token_id:3d}: {count:7d} ({percentage:5.2f}%) {bar}")
    
    # 序列唯一性
    unique_sequences = len(set(label_sequences))
    print(f"\n  [序列唯一性]")
    print(f"    总序列数: {num_samples}")
    print(f"    唯一序列数: {unique_sequences}")
    print(f"    序列重复率: {(1 - unique_sequences/num_samples)*100:.2f}%")
    
    return label_counter, label_sequences


def analyze_autophase(dataset, max_samples=None):
    """分析 Autophase 特征分布"""
    print("\n" + "="*70)
    print("4. Autophase 特征分布")
    print("="*70)
    
    if isinstance(dataset, DatasetDict):
        ds = dataset[list(dataset.keys())[0]]
    else:
        ds = dataset
    
    if 'autophase' not in ds.column_names:
        print("  ⚠️ 数据集中不包含 autophase 特征")
        return None
    
    num_samples = len(ds) if max_samples is None else min(max_samples, len(ds))
    
    autophase_list = []
    
    for i in tqdm(range(num_samples), desc="  Analyzing autophase"):
        sample = ds[i]
        ap = sample['autophase']
        
        if hasattr(ap, 'tolist'):
            ap = ap.tolist()
        elif isinstance(ap, np.ndarray):
            ap = ap.tolist()
        
        autophase_list.append(ap)
    
    autophase_array = np.array(autophase_list)
    
    print(f"\n  [基本统计]")
    print(f"    样本数: {len(autophase_list)}")
    print(f"    特征维度: {autophase_array.shape[1]}")
    print(f"    全局最小值: {autophase_array.min():.2f}")
    print(f"    全局最大值: {autophase_array.max():.2f}")
    print(f"    全局均值: {autophase_array.mean():.2f}")
    
    # 每个特征维度的统计
    print(f"\n  [各维度统计（前10维）]")
    print(f"    {'维度':<6} {'均值':>10} {'标准差':>10} {'最小':>10} {'最大':>10}")
    print(f"    {'-'*46}")
    for dim in range(min(10, autophase_array.shape[1])):
        col = autophase_array[:, dim]
        print(f"    {dim:<6} {col.mean():>10.2f} {col.std():>10.2f} {col.min():>10.2f} {col.max():>10.2f}")
    
    # 检查唯一的 autophase（唯一程序）
    unique_autophase = len(set(tuple(ap) for ap in autophase_list))
    print(f"\n  [唯一性]")
    print(f"    唯一 autophase 向量数: {unique_autophase}")
    print(f"    与样本数比值: {unique_autophase/num_samples*100:.2f}%")
    
    return autophase_array


def analyze_input_output_correlation(dataset, input_hashes, max_samples=None):
    """分析相同程序的不同优化序列"""
    print("\n" + "="*70)
    print("5. 相同程序的优化序列多样性分析")
    print("="*70)
    
    if isinstance(dataset, DatasetDict):
        ds = dataset[list(dataset.keys())[0]]
    else:
        ds = dataset
    
    # 找出有多个优化序列的程序
    multi_seq_programs = {h: indices for h, indices in input_hashes.items() if len(indices) > 1}
    
    if not multi_seq_programs:
        print("  没有发现相同程序有多个优化序列的情况")
        return
    
    print(f"\n  有多个优化序列的程序数: {len(multi_seq_programs)}")
    
    # 分析前 5 个有多个序列的程序
    print(f"\n  [示例分析（前5个多序列程序）]")
    
    for i, (h, indices) in enumerate(list(multi_seq_programs.items())[:5]):
        print(f"\n  程序 {i+1} (hash: {h[:8]}..., 共 {len(indices)} 个序列):")
        
        sequences = []
        for idx in indices[:5]:  # 最多显示 5 个序列
            sample = ds[idx]
            labels = sample['labels']
            if hasattr(labels, 'tolist'):
                labels = labels.tolist()
            valid_labels = [l for l in labels if l != -100]
            sequences.append(valid_labels)
            print(f"    序列 {idx}: 长度={len(valid_labels)}, 前5个token={valid_labels[:5]}")
        
        # 计算序列之间的相似度
        if len(sequences) >= 2:
            # 简单的 Jaccard 相似度
            set1 = set(sequences[0])
            set2 = set(sequences[1])
            jaccard = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
            print(f"    序列 0 和 1 的 Jaccard 相似度: {jaccard:.2f}")


def save_plots(encoder_lengths, decoder_lengths, samples_per_program, output_dir):
    """保存分析图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Encoder 长度分布
    plt.figure(figsize=(10, 6))
    plt.hist(encoder_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Encoder Sequence Length')
    plt.ylabel('Count')
    plt.title('Encoder Sequence Length Distribution')
    plt.savefig(os.path.join(output_dir, 'encoder_length_dist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Decoder 长度分布
    plt.figure(figsize=(10, 6))
    plt.hist(decoder_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Decoder Sequence Length (Valid Labels)')
    plt.ylabel('Count')
    plt.title('Decoder Sequence Length Distribution')
    plt.savefig(os.path.join(output_dir, 'decoder_length_dist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 每个程序的样本数分布
    plt.figure(figsize=(10, 6))
    plt.hist(samples_per_program, bins=range(1, max(samples_per_program)+2), edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Sequences per Program')
    plt.ylabel('Number of Programs')
    plt.title('Sequences per Program Distribution')
    plt.savefig(os.path.join(output_dir, 'sequences_per_program.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  图表已保存到: {output_dir}")


def main():
    args = parse_args()
    
    print("="*70)
    print("Passformer Tokenized Dataset 分布分析")
    print("="*70)
    print(f"数据集路径: {args.data_dir}")
    if args.max_samples:
        print(f"最大分析样本数: {args.max_samples}")
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = load_from_disk(args.data_dir)
    
    if isinstance(dataset, DatasetDict):
        total_samples = sum(len(ds) for ds in dataset.values())
        print(f"数据集分割: {list(dataset.keys())}")
    else:
        total_samples = len(dataset)
    print(f"总样本数: {total_samples}")
    
    # 运行分析
    input_hashes, samples_per_program = analyze_duplicates(dataset, args.max_samples)
    encoder_lengths, decoder_lengths = analyze_sequence_lengths(dataset, args.max_samples)
    analyze_labels_distribution(dataset, args.max_samples)
    analyze_autophase(dataset, args.max_samples)
    analyze_input_output_correlation(dataset, input_hashes, args.max_samples)
    
    # 保存图表
    if args.output_dir:
        print("\n" + "="*70)
        print("保存分析图表")
        print("="*70)
        save_plots(encoder_lengths, decoder_lengths, samples_per_program, args.output_dir)
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()

