"""
测试 ga_llvm_37k 数据集
测试 collect_ga_llvm_dataset.py 生成的数据集
"""
import os
from pathlib import Path
from datasets import load_from_disk

# 数据集路径
DATASET_PATH = Path("/home/xucong24/Compiler/datasets/ga_llvm_37k")

# 必需的字段
REQUIRED_FIELDS = [
    'Benchmark',
    'CpuInfo',
    'IrInstructionCountO0',
    'IrInstructionCountO3',
    'IrInstructionCountOz',
    'InstCount',
    'Autophase',
    'Reward',
    'Commandline',
    'LLVM_IR'
]


def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "=" * 60)
    print("数据集加载测试")
    print("=" * 60)
    
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] 数据集路径不存在: {DATASET_PATH}")
        print("[INFO] 请先运行 collect_ga_llvm_dataset.py 生成数据集")
        return None
    
    try:
        dataset = load_from_disk(str(DATASET_PATH))
        print(f"[OK] 数据集加载成功!")
        print(f"[INFO] 数据集类型: {type(dataset)}")
        print(f"[INFO] 数据集路径: {DATASET_PATH}")
        return dataset
    except Exception as e:
        print(f"[ERROR] 加载数据集失败: {e}")
        return None


def test_dataset_structure(dataset):
    """测试数据集结构"""
    print("\n" + "=" * 60)
    print("数据集结构检查")
    print("=" * 60)
    
    if dataset is None:
        print("[ERROR] 数据集未加载，跳过结构检查")
        return False
    
    print(f"[INFO] 样本数量: {len(dataset)}")
    print(f"[INFO] 特征列: {dataset.column_names}")
    print(f"[INFO] 特征类型: {dataset.features}")
    
    return True


def test_dataset_fields(dataset):
    """测试数据集字段是否符合预期"""
    print("\n" + "=" * 60)
    print("数据集字段检查")
    print("=" * 60)
    
    if dataset is None:
        print("[ERROR] 数据集未加载，跳过字段检查")
        return False
    
    columns = dataset.column_names
    missing_fields = []
    
    for field in REQUIRED_FIELDS:
        if field in columns:
            print(f"[OK] 字段 '{field}' 存在")
        else:
            print(f"[ERROR] 缺少必需字段 '{field}'")
            missing_fields.append(field)
    
    if missing_fields:
        print(f"\n[ERROR] 缺少以下必需字段: {missing_fields}")
        return False
    
    print(f"\n[OK] 所有必需字段都存在!")
    return True


def test_sample_data(dataset, num_samples=3):
    """查看样本数据"""
    print("\n" + "=" * 60)
    print(f"样本数据预览 (前 {num_samples} 条)")
    print("=" * 60)
    
    if dataset is None:
        print("[ERROR] 数据集未加载，跳过样本预览")
        return False
    
    num_samples = min(num_samples, len(dataset))
    
    for i in range(num_samples):
        sample = dataset[i]
        print(f"\n--- 样本 {i + 1} ---")
        print(f"  Benchmark: {sample.get('Benchmark', 'N/A')}")
        print(f"  Reward: {sample.get('Reward', 'N/A')}")
        print(f"  Commandline: {sample.get('Commandline', 'N/A')[:100]}..." if sample.get('Commandline') else "  Commandline: N/A")
        
        # 检查数值字段
        if 'IrInstructionCountO0' in sample:
            print(f"  IrInstructionCountO0: {sample['IrInstructionCountO0']}")
        if 'IrInstructionCountO3' in sample:
            print(f"  IrInstructionCountO3: {sample['IrInstructionCountO3']}")
        if 'IrInstructionCountOz' in sample:
            print(f"  IrInstructionCountOz: {sample['IrInstructionCountOz']}")
        if 'InstCount' in sample:
            print(f"  InstCount: {sample['InstCount']}")
        
        # 检查 LLVM_IR 长度
        if 'LLVM_IR' in sample and sample['LLVM_IR']:
            llvm_ir = sample['LLVM_IR']
            if isinstance(llvm_ir, str):
                print(f"  LLVM_IR 长度: {len(llvm_ir)} 字符")
            else:
                print(f"  LLVM_IR 类型: {type(llvm_ir)}")
        
        # 检查 Autophase 维度
        if 'Autophase' in sample and sample['Autophase'] is not None:
            autophase = sample['Autophase']
            if hasattr(autophase, '__len__'):
                print(f"  Autophase 维度: {len(autophase)}")
            else:
                print(f"  Autophase 值: {autophase}")
    
    return True


def test_data_statistics(dataset):
    """测试数据统计信息"""
    print("\n" + "=" * 60)
    print("数据统计信息")
    print("=" * 60)
    
    if dataset is None:
        print("[ERROR] 数据集未加载，跳过统计信息")
        return False
    
    # Reward 统计
    if 'Reward' in dataset.column_names:
        rewards = [sample['Reward'] for sample in dataset]
        rewards = [r for r in rewards if r is not None]
        if rewards:
            print(f"\n[INFO] Reward 统计:")
            print(f"  样本数量: {len(rewards)}")
            print(f"  最小值: {min(rewards)}")
            print(f"  最大值: {max(rewards)}")
            print(f"  平均值: {sum(rewards) / len(rewards):.4f}")
    
    # Benchmark 唯一值统计
    if 'Benchmark' in dataset.column_names:
        benchmarks = [sample['Benchmark'] for sample in dataset]
        unique_benchmarks = set(benchmarks)
        print(f"\n[INFO] Benchmark 统计:")
        print(f"  总样本数: {len(benchmarks)}")
        print(f"  唯一 Benchmark 数量: {len(unique_benchmarks)}")
        if len(unique_benchmarks) <= 20:
            print(f"  Benchmark 列表: {sorted(unique_benchmarks)}")
    
    # LLVM_IR 长度统计
    if 'LLVM_IR' in dataset.column_names:
        llvm_lengths = []
        for sample in dataset:
            llvm_ir = sample.get('LLVM_IR')
            if llvm_ir and isinstance(llvm_ir, str):
                llvm_lengths.append(len(llvm_ir))
        
        if llvm_lengths:
            print(f"\n[INFO] LLVM_IR 长度统计:")
            print(f"  有效样本数: {len(llvm_lengths)}")
            print(f"  最小长度: {min(llvm_lengths)}")
            print(f"  最大长度: {max(llvm_lengths)}")
            print(f"  平均长度: {sum(llvm_lengths) / len(llvm_lengths):.2f}")
    
    # Autophase 维度统计
    if 'Autophase' in dataset.column_names:
        autophase_dims = []
        for sample in dataset:
            autophase = sample.get('Autophase')
            if autophase is not None:
                if hasattr(autophase, '__len__'):
                    autophase_dims.append(len(autophase))
        
        if autophase_dims:
            unique_dims = set(autophase_dims)
            print(f"\n[INFO] Autophase 维度统计:")
            print(f"  有效样本数: {len(autophase_dims)}")
            print(f"  唯一维度值: {sorted(unique_dims)}")
            if len(unique_dims) == 1:
                print(f"  [OK] 所有样本的 Autophase 维度一致: {unique_dims.pop()}")
            else:
                print(f"  [WARN] Autophase 维度不一致!")
    
    return True


def test_data_types(dataset):
    """测试数据类型"""
    print("\n" + "=" * 60)
    print("数据类型检查")
    print("=" * 60)
    
    if dataset is None or len(dataset) == 0:
        print("[ERROR] 数据集未加载或为空，跳过类型检查")
        return False
    
    sample = dataset[0]
    type_errors = []
    
    # 检查各字段类型
    type_checks = {
        'Benchmark': str,
        'Reward': (int, float),
        'Commandline': str,
        'LLVM_IR': str,
        'IrInstructionCountO0': (int, float),
        'IrInstructionCountO3': (int, float),
        'IrInstructionCountOz': (int, float),
        'InstCount': (int, float),
    }
    
    for field, expected_type in type_checks.items():
        if field in sample:
            value = sample[field]
            if value is not None:
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        print(f"[WARN] 字段 '{field}' 类型为 {type(value).__name__}, 期望为 {[t.__name__ for t in expected_type]}")
                        type_errors.append(field)
                    else:
                        print(f"[OK] 字段 '{field}' 类型正确: {type(value).__name__}")
                else:
                    if not isinstance(value, expected_type):
                        print(f"[WARN] 字段 '{field}' 类型为 {type(value).__name__}, 期望为 {expected_type.__name__}")
                        type_errors.append(field)
                    else:
                        print(f"[OK] 字段 '{field}' 类型正确: {type(value).__name__}")
        else:
            print(f"[WARN] 字段 '{field}' 不存在")
    
    if type_errors:
        print(f"\n[WARN] 以下字段类型不符合预期: {type_errors}")
        return False
    else:
        print(f"\n[OK] 所有字段类型检查通过!")
        return True


def test_data_completeness(dataset):
    """测试数据完整性（检查是否有 None 值）"""
    print("\n" + "=" * 60)
    print("数据完整性检查")
    print("=" * 60)
    
    if dataset is None:
        print("[ERROR] 数据集未加载，跳过完整性检查")
        return False
    
    total_samples = len(dataset)
    completeness_issues = {}
    
    for field in REQUIRED_FIELDS:
        if field in dataset.column_names:
            null_count = sum(1 for sample in dataset if sample.get(field) is None)
            if null_count > 0:
                completeness_issues[field] = null_count
                print(f"[WARN] 字段 '{field}' 有 {null_count}/{total_samples} 个 None 值")
            else:
                print(f"[OK] 字段 '{field}' 无缺失值")
    
    if completeness_issues:
        print(f"\n[WARN] 以下字段存在缺失值: {completeness_issues}")
        return False
    else:
        print(f"\n[OK] 所有字段数据完整!")
        return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("ga_llvm_37k 数据集测试")
    print("=" * 60)
    
    # 1. 测试数据集加载
    dataset = test_dataset_loading()
    if dataset is None:
        return
    
    # 2. 测试数据集结构
    test_dataset_structure(dataset)
    
    # 3. 测试数据集字段
    test_dataset_fields(dataset)
    
    # 4. 查看样本数据
    test_sample_data(dataset, num_samples=3)
    
    # 5. 测试数据统计信息
    test_data_statistics(dataset)
    
    # 6. 测试数据类型
    test_data_types(dataset)
    
    # 7. 测试数据完整性
    test_data_completeness(dataset)
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

