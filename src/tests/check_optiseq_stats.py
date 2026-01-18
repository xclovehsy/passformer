"""检查优化序列数据集的统计信息"""
import os
from collections import Counter
from datasets import load_from_disk

from src.config import load_config
from src.model import OptiSeqTokenizer

CONFIG_PATH = "configs/passformer_gallvm_seq2seq.yaml"


def main():
    cfg = load_config(CONFIG_PATH)
    data_dir = cfg["data"]["data_dir"]
    decoder_maxlen = cfg["data"]["decoder_maxlen"]
    opti_seq_tokenizer_id = cfg["model"]["opti_seq_tokenizer_id"]
    
    print(f"[INFO] decoder_maxlen: {decoder_maxlen}")
    print(f"[INFO] 加载数据集: {data_dir}")
    dataset = load_from_disk(data_dir)
    
    print(f"[INFO] 加载 tokenizer: {opti_seq_tokenizer_id}")
    tokenizer = OptiSeqTokenizer.from_pretrained(opti_seq_tokenizer_id)
    print(f"[INFO] Tokenizer vocab size: {len(tokenizer)}")
    
    # 获取训练集
    if hasattr(dataset, 'keys'):
        train_data = dataset['train']
    else:
        train_data = dataset
    
    print(f"\n[INFO] 训练集样本数: {len(train_data)}")
    
    # 统计序列长度
    lengths = []
    first_tokens = []  # 第一个 pass
    all_passes = Counter()
    truncated_count = 0
    
    print("\n[INFO] 分析中...")
    for i, sample in enumerate(train_data):
        commandline = sample['Commandline']
        tokens = commandline.strip().split()
        seq_len = len(tokens) + 2  # +2 for bos and eos
        lengths.append(seq_len)
        
        if seq_len > decoder_maxlen:
            truncated_count += 1
        
        if tokens:
            first_tokens.append(tokens[0])
            all_passes.update(tokens)
        
        if i < 5:
            print(f"\n  样本 {i}: 长度={seq_len}, benchmark={sample['Benchmark']} 序列={commandline[:100]}...")
    
    # 统计结果
    print("\n" + "=" * 60)
    print("序列长度统计")
    print("=" * 60)
    print(f"  最小长度: {min(lengths)}")
    print(f"  最大长度: {max(lengths)}")
    print(f"  平均长度: {sum(lengths) / len(lengths):.1f}")
    print(f"  被截断的样本数: {truncated_count} / {len(lengths)} ({100*truncated_count/len(lengths):.1f}%)")
    
    # 长度分布
    print("\n长度分布:")
    bins = [0, 32, 64, 128, 256, 512, float('inf')]
    for i in range(len(bins) - 1):
        count = sum(1 for l in lengths if bins[i] < l <= bins[i+1])
        pct = 100 * count / len(lengths)
        label = f"{bins[i]+1}-{bins[i+1]}" if bins[i+1] != float('inf') else f">{bins[i]}"
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Pass 统计")
    print("=" * 60)
    print(f"  不同 pass 数量: {len(all_passes)}")
    print(f"\n  最常见的 10 个 pass:")
    for pass_name, count in all_passes.most_common(10):
        print(f"    {pass_name}: {count}")
    
    print(f"\n  最常见的 10 个开头 pass:")
    first_counter = Counter(first_tokens)
    for pass_name, count in first_counter.most_common(10):
        pct = 100 * count / len(first_tokens)
        print(f"    {pass_name}: {count} ({pct:.1f}%)")
    
    # 检查是否有重复序列
    print("\n" + "=" * 60)
    print("重复序列检查")
    print("=" * 60)
    all_commandlines = [sample['Commandline'] for sample in train_data]
    unique_count = len(set(all_commandlines))
    print(f"  总序列数: {len(all_commandlines)}")
    print(f"  唯一序列数: {unique_count}")
    print(f"  重复率: {100 * (1 - unique_count / len(all_commandlines)):.1f}%")


if __name__ == "__main__":
    main()

