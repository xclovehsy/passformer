CompilerGym编译优化结果

benchmark://user-v0/20250219T194127-dc9f
优化前IR指令数量: 146
-adce 1
None None False {'action_had_no_effect': True, 'new_action_space': False}
-instcombine 53
None None False {'action_had_no_effect': False, 'new_action_space': False}
-simplifycfg 10
None None False {'action_had_no_effect': False, 'new_action_space': False}
-mem2reg 103
None None False {'action_had_no_effect': False, 'new_action_space': False}
优化后IR指令数量: 56
opt -adce -instcombine -simplifycfg -mem2reg input.bc -o output.bc

安装clang以及工具链
sudo apt update
sudo apt install llvm clang

macos
brew install llvm


ProGraML依赖库
安装 nlohmann/json.hpp
sudo apt update
sudo apt install nlohmann-json3-dev




g++ -o compute_autophase compute_autophase.cc InstCount.cc \
    $(llvm-config --cxxflags --ldflags --system-libs --libs core irreader support analysis transformutils) \
    -fno-rtti -std=c++14

g++ -o compute_ir_instruction_count_mac compute_ir_instruction_count.cc InstCount.cc \
    $(llvm-config --cxxflags --ldflags --system-libs --libs core irreader support analysis transformutils) \
    -fno-rtti -std=c++17

sudo apt update
sudo apt install aria2
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh bert-base-uncased

modernbert
https://huggingface.co/answerdotai/ModernBERT-base


llvm18 路径
/home/xucong24/llvm11-18/install-llvm18.1.8/bin



# 一定要使用llvm14！！！！！！！！ llvm15的透明指针机制会导致无法统计指针条数
安装llvm14 
apt-cache show llvm-14  # 查看llvm15版本
sudo apt update
sudo apt install llvm-14 clang-14
llvm15路径：  /usr/lib/llvm-14/bin/
（添加到环境变量）
export PATH=/usr/lib/llvm-14/bin:$PATH
source ~/.zshrc


CompilerGym使用的是llvm10
/home/xucong24/.local/share/compiler_gym/llvm-v0/bin


/home/xucong24/.local/share/compiler_gym/llvm-v0/bin/opt -analyze -passes=instcount /home/xucong24/Compiler/tmp/optimized.bc


ComPile数据集使用LLVM18.0.0git编译， 使用inst2vec会造成大量的unk-token

/home/xucong24/Compiler/datasets/poj104/ir_test/1/24.ll 
原始IR指令行数： 232
使用ModernBert的fastbpetokenizer 编码token长度3725
使用Inst2vec编码token长度173其中有42unk-token 大约在25%左右

使用模块化运行 python -m src.data.convert_poj104_inst2vec

1. Inst2VecTokenizer适配huggingface接口
2. 使用DataCollatorForLanguageModeling随机掩码
3. 调整训练代码，ModernBert模型调整Embedding层和输出层vocab_size

cursor连接服务器 ssh xucong24@192.168.5.101 -p 30001
tensorboard --logdir /home/.. --bind_all# passformer
