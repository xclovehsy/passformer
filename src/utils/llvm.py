import os
import re
import tempfile
import subprocess
import llvmlite.binding as llvm
from typing import Union, Optional
import shutil

import os
import re
import tempfile
import subprocess
import llvmlite.binding as llvm
from pathlib import Path


def bitcode_to_llvm_ir(input_data: Union[str, bytes], llvm_path: str) -> str:
    """将字节码解析llvm"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        bc_path = tmpdir_path / "input.bc"
        ll_path = tmpdir_path / "output.ll"

        if isinstance(input_data, bytes):
            with open(bc_path, "wb") as f:
                f.write(input_data)
        elif isinstance(input_data, str) and os.path.isfile(input_data):
            with open(input_data, "rb") as src, open(bc_path, "wb") as dst:
                dst.write(src.read())
        else:
            raise ValueError("input_data 必须是有效路径字符串或字节串")

        try:
            subprocess.run([llvm_path, str(bc_path), "-o", str(ll_path)], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"llvm-dis 运行失败: {e}")

        with open(ll_path, "r", encoding="utf-8") as f:
            llvm_ir = f.read()
        return llvm_ir

def count_llvm_ir_instructions(llvm_ir: str) -> int:
    """
    计算 LLVM-IR 代码中的指令数目。
    """

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    # 解析字符串中的LLVM IR
    module = llvm.parse_assembly(llvm_ir)

    # 计算指令数量
    instruction_count = 0
    for function in module.functions:
        for block in function.blocks:
            for instruction in block.instructions:  # 使用 .instructions 获取块中的指令
                instruction_count += 1
                
    return instruction_count

def get_llvm_utils(llvm_path):
    if llvm_path is None:
        return 'llvm-dis', 'llvm-as', 'opt', 'clang'
    else:
        return os.path.join(llvm_path, 'llvm-dis'), os.path.join(llvm_path, 'llvm-as'), \
                os.path.join(llvm_path, 'opt'), os.path.join(llvm_path, 'clang')
        

def compile_c_to_llvm_ir(source, opt_flags=None, llvm_path=None):
    """
    编译 C 代码到 LLVM IR，并应用优化选项，使用 .bc 进行中间存储。

    参数:
    - source: C 源代码字符串或 C 文件路径。
    - opt_flags: 需要应用的 LLVM 优化选项列表，如 ["-adce", "-instcombine"]。
    - llvm_path: llvm 编译器路径。

    返回:
    - 优化后的 LLVM IR 字符串
    """
    opt_flags = opt_flags or []
    llvm_dis_path, llvm_as_path, opt_path, clang_path = get_llvm_utils(llvm_path)

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    if os.path.isfile(source):
        c_file = source
    else:
        c_file = os.path.join(temp_dir, "input.c")
        with open(c_file, "w") as f:
            f.write(source)

    bc_file = os.path.join(temp_dir, "output.bc")
    opt_bc_file = os.path.join(temp_dir, "optimized.bc")
    opt_ll_file = os.path.join(temp_dir, "optimized.ll")

    try:
        # 生成 LLVM Bitcode (.bc)
        subprocess.run([
            clang_path, 
            "-S", 
            "-emit-llvm", 
            "-o", bc_file, 
            c_file, 
            "-O1", 
            "-Xclang", "-disable-llvm-passes", 
            "-Xclang", "-disable-llvm-optzns",
            "-isystem", "/usr/include/c++/11",
            "-isystem", "/usr/include/x86_64-linux-gnu/c++/11",
            "-isystem", "/usr/include/c++/11/backward",
            "-isystem", "/usr/lib/gcc/x86_64-linux-gnu/11/include",
            "-isystem", "/usr/local/include",
            "-isystem", "/usr/include/x86_64-linux-gnu",
            "-isystem", "/usr/include"
        ], check=True)
        # subprocess.run([
        #     clang_path,
        #     "-emit-llvm", "-c",  # 编译为 LLVM IR
        #     "-O0",  # 无优化
        #     "-Xclang", "-disable-O0-optnone",      # 移除 optnone 限制
        #     "-Xclang", "-disable-llvm-passes",     # 禁用 LLVM 优化 Pass
        #     "-isystem", "/usr/include/c++/11",
        #     "-isystem", "/usr/include/x86_64-linux-gnu/c++/11",
        #     "-isystem", "/usr/include/c++/11/backward",
        #     "-isystem", "/usr/lib/gcc/x86_64-linux-gnu/11/include",
        #     "-isystem", "/usr/local/include",
        #     "-isystem", "/usr/include/x86_64-linux-gnu",
        #     "-isystem", "/usr/include",
        #     c_file,
        #     "-o", bc_file
        # ], check=True)
        # 应用优化
        subprocess.run([opt_path, *opt_flags, bc_file, "-o", opt_bc_file], check=True)

        # 转换优化后的 Bitcode 为 LLVM IR (.ll) 反编译
        subprocess.run([llvm_dis_path, opt_bc_file, "-o", opt_ll_file], check=True)

        with open(opt_ll_file, "r") as f:
            llvm_ir = f.read()

    except Exception as e:
        print(e)
        llvm_ir = ""

    finally:
        shutil.rmtree(temp_dir)

    return llvm_ir


def optimize_llvm_ir(llvm_ir: str, opt_flags: Optional[Union[str, list]] = None, llvm_path: Optional[str] = None) -> str:
    """优化 LLVM IR 代码，并返回优化后的 LLVM IR 代码"""
    opt_flags = opt_flags or []
    llvm_dis_path, llvm_as_path, opt_path, clang_path = get_llvm_utils(llvm_path)
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    if os.path.isfile(llvm_ir):
        ll_file = llvm_ir
    else:
        ll_file = os.path.join(temp_dir, "input.ll")
        with open(ll_file, "w") as f:
            f.write(llvm_ir)

    bc_file = os.path.join(temp_dir, "input.bc")
    opt_bc_file = os.path.join(temp_dir, "optimized.bc")
    opt_ll_file = os.path.join(temp_dir, "optimized.ll")

    try:
        # 生成 LLVM Bitcode (.bc)
        subprocess.run([llvm_as_path, ll_file, "-o", bc_file], check=True)

        # 应用优化
        subprocess.run([opt_path, *opt_flags, bc_file, "-o", opt_bc_file], check=True)

        # 转换优化后的 Bitcode 为 LLVM IR (.ll) 反编译
        subprocess.run([llvm_dis_path, opt_bc_file, "-o", opt_ll_file], check=True)

        with open(opt_ll_file, "r") as f:
            llvm_ir = f.read()

    except Exception as e:
        print(e)
        llvm_ir = ""

    finally:
        shutil.rmtree(temp_dir)

    return llvm_ir


def calc_reward(llvm_ir: str, opt_flags: Union[Union[str, list]],  llvm_path: Optional[str] = None) -> int:
    """计算编译优化序列Reward"""
    if isinstance(opt_flags, str):
        opt_flags = opt_flags.split()

    ir_count = count_llvm_ir_instructions(llvm_ir)

    opt_ir = optimize_llvm_ir(llvm_ir, opt_flags=opt_flags, llvm_path=llvm_path)
    opt_count = count_llvm_ir_instructions(opt_ir)    

    opt_ir_by_o3 = optimize_llvm_ir(llvm_ir, opt_flags=["-O3"], llvm_path=llvm_path)    
    opt_count_by_o3 = count_llvm_ir_instructions(opt_ir_by_o3)

    reward = (ir_count - opt_count) / (ir_count - opt_count_by_o3)

    print(f"IR指令数量: {ir_count}, 优化后指令数量: {opt_count}, O3优化后指令数量: {opt_count_by_o3}, 优化奖励: {reward}")
    return reward

    


if __name__ == '__main__':
    opt_flags = ["-adce", "-instcombine", "-simplifycfg", "-mem2reg"]
    c_file_path = r'/home/xucong24/Compiler/datasets/qsort.c'
    llvm_ir = compile_c_to_llvm_ir(c_file_path, llvm_path='/home/xucong24/.local/share/compiler_gym/llvm-v0/bin')

    # ll_path = '/home/xucong24/Compiler/tmp/37902.ll'
    # with open(ll_path, 'r') as f:
    #     llvm_ir = f.read()
    print(f"优化前IR指令数量: {count_llvm_ir_instructions(llvm_ir)}")
    
    opt_llvm_ir = optimize_llvm_ir(llvm_ir, opt_flags=opt_flags, llvm_path='/home/xucong24/.local/share/compiler_gym/llvm-v0/bin')
    print(f"优化后IR指令数量: {count_llvm_ir_instructions(opt_llvm_ir)}")

    print(calc_reward(llvm_ir, opt_flags, llvm_path='/home/xucong24/.local/share/compiler_gym/llvm-v0/bin'))

    # ir_file = '/home/xucong24/Compiler/tmp/37902.ll'
    # with open(ir_file, 'r') as f:
    #     llvm_ir = f.read()
    # print(f"IR指令数量: {count_llvm_ir_instructions(llvm_ir)}")
    # llvm_ir = optimize_llvm_ir(llvm_ir, opt_flags=opt_flags, llvm_path='/usr/lib/llvm-14/bin')
    # print(f"优化后IR指令数量: {count_llvm_ir_instructions(llvm_ir)}")

    # print(calc_reward(llvm_ir, opt_flags, llvm_path='/usr/lib/llvm-14/bin'))

    