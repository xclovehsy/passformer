import subprocess

def run_executable(executable_path, args=None):
    """
    运行可执行文件并读取其输出。
    
    :param executable_path: 可执行文件路径
    :param args: 可选的参数列表
    :return: 可执行文件的标准输出
    """
    if args is None:
        args = []
    
    try:
        # 使用 subprocess.run 执行命令
        result = subprocess.run([executable_path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result.check_returncode()  # 确保命令成功执行，否则抛出 CalledProcessError
        
        # 解析标准输出
        output = result.stdout.strip()
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error executing the command: {e}")
    except FileNotFoundError:
        print("Executable not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def read_ir_from_file(ir_path):
    """从文件中读取ir"""
    
    with open(ir_path, 'r') as f:
        ir = f.read()

    return ir