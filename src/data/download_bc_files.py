"""
下载 CompilerGym benchmark 的 bc (bitcode) 文件

支持两种模式：
1. 从 CSV 文件读取 benchmark 列表下载
2. 直接下载 CompilerGym 内置数据集（如 cbench-v1, csmith-v0 等）

支持的数据集参考: https://compilergym.com/llvm/index.html#datasets
"""
import compiler_gym
import pandas as pd
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse

# 默认路径配置
DEFAULT_INPUT_CSV = "/home/xucong24/Compiler/datasets/results.csv"
DEFAULT_SAVE_DIR = "/home/xucong24/Compiler/datasets/bc_files"

# CompilerGym 支持的数据集列表
AVAILABLE_DATASETS = [
    "anghabench-v1",    # 1,041,333 benchmarks
    "blas-v0",          # 300 benchmarks
    "cbench-v1",        # 23 benchmarks (可运行)
    "chstone-v0",       # 12 benchmarks
    "clgen-v0",         # 996 benchmarks
    "github-v0",        # 49,738 benchmarks
    "jotaibench-v0",    # 18,761 benchmarks
    "linux-v0",         # 13,894 benchmarks
    "mibench-v1",       # 40 benchmarks
    "npb-v0",           # 122 benchmarks
    "opencv-v0",        # 442 benchmarks
    "poj104-v1",        # 49,816 benchmarks
    "tensorflow-v0",    # 1,985 benchmarks
]

# 进程内全局环境变量
_env = None


def init_worker():
    """每个进程启动时调用，初始化 CompilerGym 环境"""
    global _env
    _env = compiler_gym.make("llvm-v0")


def download_bc(benchmark: str, save_dir: Path) -> dict:
    """
    下载单个 benchmark 的 bc 文件
    
    Args:
        benchmark: benchmark URI，如 "generator://csmith-v0/0"
        save_dir: 保存目录
    
    Returns:
        dict: 包含下载结果的字典
    """
    global _env
    
    try:
        _env.reset(benchmark=benchmark)
        
        # 获取 bc 文件的临时路径
        bc_temp_path = _env.observation["BitcodeFile"]
        
        # 生成保存文件名：将 benchmark URI 转换为合法文件名
        # 例如: "generator://csmith-v0/0" -> "generator_csmith-v0_0.bc"
        safe_name = benchmark.replace("://", "_").replace("/", "_")
        dest_path = save_dir / f"{safe_name}.bc"
        
        # 复制文件
        shutil.copy(bc_temp_path, dest_path)
        
        return {
            "benchmark": benchmark,
            "status": "success",
            "path": str(dest_path)
        }
        
    except Exception as e:
        return {
            "benchmark": benchmark,
            "status": "failed",
            "error": str(e)
        }


def process_benchmark(args: tuple) -> dict:
    """包装函数，用于多进程调用"""
    benchmark, save_dir = args
    return download_bc(benchmark, save_dir)


def get_dataset_benchmarks(dataset_name: str) -> list:
    """
    从 CompilerGym 获取指定数据集的所有 benchmark URI
    
    Args:
        dataset_name: 数据集名称，如 "cbench-v1"
    
    Returns:
        list: benchmark URI 列表
    """
    env = compiler_gym.make("llvm-v0")
    try:
        # 获取数据集
        dataset = env.datasets[dataset_name]
        # 获取所有 benchmark 的 URI
        benchmarks = [str(bm.uri) for bm in dataset.benchmarks()]
        return benchmarks
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="下载 CompilerGym benchmark 的 bc 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:
  # 从 CSV 文件下载
  python download_bc_files.py --input results.csv --output ./bc_files
  
  # 下载 cbench 数据集
  python download_bc_files.py --dataset cbench-v1 --output ./cbench_bc
  
  # 下载多个数据集
  python download_bc_files.py --dataset cbench-v1 mibench-v1 --output ./bc_files

支持的数据集: {', '.join(AVAILABLE_DATASETS)}
""")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="输入的 CSV 文件路径（与 --dataset 互斥）")
    parser.add_argument("--dataset", "-d", type=str, nargs="+", default=None,
                        help=f"要下载的 CompilerGym 数据集名称（与 --input 互斥）")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_SAVE_DIR,
                        help="bc 文件保存目录")
    parser.add_argument("--workers", "-w", type=int, default=16,
                        help="并行进程数")
    parser.add_argument("--list-datasets", action="store_true",
                        help="列出所有可用的数据集")
    args = parser.parse_args()
    
    # 列出数据集
    if args.list_datasets:
        print("可用的 CompilerGym 数据集:")
        for ds in AVAILABLE_DATASETS:
            print(f"  - {ds}")
        return
    
    # 检查参数互斥
    if args.input and args.dataset:
        parser.error("--input 和 --dataset 不能同时使用")
    
    if not args.input and not args.dataset:
        # 默认使用 CSV 模式
        args.input = DEFAULT_INPUT_CSV
    
    save_dir = Path(args.output)
    
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取 benchmark 列表
    if args.dataset:
        # 从 CompilerGym 数据集获取
        benchmarks = []
        for ds_name in args.dataset:
            if ds_name not in AVAILABLE_DATASETS:
                print(f"警告: 未知数据集 '{ds_name}'，尝试加载...")
            print(f"正在获取数据集 {ds_name} 的 benchmark 列表...")
            try:
                ds_benchmarks = get_dataset_benchmarks(ds_name)
                print(f"  数据集 {ds_name}: {len(ds_benchmarks)} 个 benchmark")
                benchmarks.extend(ds_benchmarks)
            except Exception as e:
                print(f"  错误: 无法加载数据集 {ds_name}: {e}")
        print(f"共发现 {len(benchmarks)} 个 benchmark")
    else:
        # 从 CSV 文件读取
        input_csv = Path(args.input)
        print(f"正在读取 {input_csv}...")
        df = pd.read_csv(input_csv, index_col=0)
        benchmarks = df['benchmark'].unique().tolist()
        print(f"共发现 {len(benchmarks)} 个唯一的 benchmark")
    
    # 检查已下载的文件，跳过已存在的
    existing_files = set(f.stem for f in save_dir.glob("*.bc"))
    benchmarks_to_download = []
    for bm in benchmarks:
        safe_name = bm.replace("://", "_").replace("/", "_")
        if safe_name not in existing_files:
            benchmarks_to_download.append(bm)
    
    skipped = len(benchmarks) - len(benchmarks_to_download)
    if skipped > 0:
        print(f"跳过 {skipped} 个已下载的 benchmark")
    
    if not benchmarks_to_download:
        print("所有 benchmark 已下载完成！")
        return
    
    print(f"开始下载 {len(benchmarks_to_download)} 个 benchmark 的 bc 文件...")
    
    # 准备参数
    task_args = [(bm, save_dir) for bm in benchmarks_to_download]
    
    success_count = 0
    failed_count = 0
    failed_benchmarks = []
    
    # 使用多进程下载
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
        futures = {executor.submit(process_benchmark, arg): arg[0] for arg in task_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            result = future.result()
            if result["status"] == "success":
                success_count += 1
            else:
                failed_count += 1
                failed_benchmarks.append({
                    "benchmark": result["benchmark"],
                    "error": result.get("error", "Unknown error")
                })
    
    # 打印统计信息
    print(f"\n下载完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  保存目录: {save_dir}")
    
    # 如果有失败的，保存失败列表
    if failed_benchmarks:
        failed_log = save_dir / "failed_downloads.txt"
        with open(failed_log, "w") as f:
            for item in failed_benchmarks:
                f.write(f"{item['benchmark']}: {item['error']}\n")
        print(f"  失败列表已保存至: {failed_log}")


if __name__ == "__main__":
    main()

