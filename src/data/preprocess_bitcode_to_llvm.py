import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset
from itertools import islice
import subprocess
import tempfile
import os

from src.utils.llvm import convert_bit_ir_inst

data_path = '/home/xucong24/ComPile'
save_path = '/home/xucong24/Compiler/datasets/ComPile_llvm18_100k'
streaming_data = load_dataset(data_path, streaming=True)
streaming_iter = iter(streaming_data['train'])


llvm_path = '/home/xucong24/llvm11-18/install-llvm18.1.8/bin/llvm-dis'

def convert_wrapper(example):
    try:
        size_in_bytes = len(example['content'])
        if size_in_bytes > 50000:
            return None 

        llvm = convert_bit_ir_inst(example['content'], llvm_path)
        example['content'] = llvm
        return example  # 返回修改后的样本

    except Exception as e:
        return None


if __name__ == '__main__':
    # mp.set_start_method('fork')  # Linux 上用 fork，Windows 要改为 'spawn'

    first_100k = []
    batch_size = 1000  # 每批处理的流数据大小
    num_proc = 32       # 并行 worker 数量

    with mp.Pool(num_proc) as pool, tqdm(total=100_000, desc="Collected valid samples") as pbar:
        while len(first_100k) < 100_000:
        # while len(first_100k) < 100:
            # 尝试拉取一批数据
            batch = list(islice(streaming_iter, batch_size))
            if not batch:  # 数据耗尽，退出
                break

            results = pool.map(convert_wrapper, batch)

            valid_samples = [r for r in results if r is not None]
            first_100k.extend(valid_samples)
            pbar.update(len(valid_samples))

    from datasets import Dataset
    full_dataset = Dataset.from_list(first_100k)
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    split_dataset.save_to_disk(save_path)
