import torch
from trl import PPOTrainer, PPOConfig, create_reference_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.model import PassformerModel, PassformerConfig, Inst2VecTokenizer, OptiSeqTokenizer
import gym
import compiler_gym

if __name__ == "__main__":
    model_id = "/home/xucong24/Compiler/work_dirs/passformer_gallvm_seq2seq_concat/20260120_195517/checkpoint-16470"
    encoder_tokenizer_id = "/home/xucong24/Compiler/checkpoints/Inst2VecTokenizer"
    decoder_tokenizer_id = "/home/xucong24/Compiler/checkpoints/OptiSeqTokenizer"
    
    # 1. 配置 PPO
    config = PPOConfig(
        model_name="passformer",
        learning_rate=1e-5,
        batch_size=8,
        mini_batch_size=2,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
    )

    # 2. 加载模型 
    # 假设你已经预训练了一个能根据代码生成 Pass 的 T5 模型
    model = PassformerModel.from_pretrained(model_id)
    ref_model = create_reference_model(model) # 复制一个参考模型用于计算 KL 散度
    encoder_tokenizer = Inst2VecTokenizer.from_pretrained(encoder_tokenizer_id)
    decoder_tokenizer = OptiSeqTokenizer.from_pretrained(decoder_tokenizer_id)

    # 3. 初始化 PPOTrainer
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=decoder_tokenizer,
    )

    # 1. 初始化
    # 务必传入负责生成 Pass 的 decoder_tokenizer
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=decoder_tokenizer, 
    )

    for epoch in range(epochs):
        for batch in dataloader:
            # batch["llvm_code"] -> 原始 LLVM IR
            # batch["autophase"] -> [batch_size, 56] 的特征向量
            
            # A. 使用 Encoder Tokenizer 编码 IR
            query_tensors = [
                encoder_tokenizer(code, return_tensors="pt")["input_ids"].squeeze(0).to(model.device)
                for code in batch["llvm_code"]
            ]
            
            # 准备 autophase tensor
            autophase_tensor = batch["autophase"].to(model.device) # [batch, 56]

            # B. 生成阶段
            # 必须把 autophase 传进去，否则你的 model.generate 会报错
            generation_kwargs = {
                "max_new_tokens": 25,
                "do_sample": True,
                "pad_token_id": decoder_tokenizer.pad_token_id,
                "autophase": autophase_tensor, # 透传给 PassformerModel.generate
            }
            
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            
            # C. 环境交互
            batch_responses = [decoder_tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            rewards = []
            for i, pass_seq in enumerate(batch_responses):
                # 这里的 env 需要是你 CompilerGym 的实例
                # 执行模型生成的 pass 序列，获取性能提升作为 reward
                reward = get_compiler_reward(batch["llvm_code"][i], pass_seq) 
                rewards.append(torch.tensor(reward))

                rewards = []
for pass_str in clean_pass_sequences:
    # 比如转换后是 "-mem2reg -gvn -simplifycfg"
    pass_list = pass_str.strip().split() 
    
    if len(pass_list) == 0:
        # 如果模型只生成了 [bos, eos]，说明它不想做任何优化
        reward = 0.0 
    else:
        try:
            # 执行并获得 reward (如代码压缩率)
            _, reward, _, _ = env.multistep(pass_list)
        except:
            reward = -1.0 # 编译崩溃或无效 Pass 序列的惩罚
            
    rewards.append(torch.tensor(reward))

            # D. PPO 优化步
            # 关键：ppo_trainer.step 内部会调用 model.forward
            # 它需要接收 autophase 才能计算 logits
            stats = ppo_trainer.step(
                query_tensors, 
                response_tensors, 
                rewards,
                autophase=autophase_tensor # 再次透传给 forward
            )
            
            ppo_trainer.log_stats(stats, batch, rewards)

    # for epoch in range(10):
    #     for batch in dataset_dataloader:
    #         query_tensors = batch["input_ids"] # 原始代码的编码
            
    #         # --- A. 模型生成阶段 ---
    #         # 生成参数需要注意：我们希望保留随机性以供探索
    #         generation_kwargs = {
    #             "min_length": -1,
    #             "top_k": 0.0,
    #             "top_p": 1.0,
    #             "do_sample": True,
    #             "pad_token_id": tokenizer.pad_token_id,
    #             "max_new_tokens": 20, # 每次预测 20 个 Pass 组成的序列
    #         }
            
    #         # 模型生成 Response (Pass 序列)
    #         response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            
    #         # 将 Tensor 转换为可读的 Pass 名称
    #         batch_responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            
    #         # --- B. 环境交互阶段 ---
    #         rewards = []
    #         env = compiler_gym.make("llvm-v0")
            
    #         for i, pass_string in enumerate(batch_responses):
    #             env.reset()
    #             # 这里假设 pass_string 是以空格分隔的 Pass 列表，如 "mem2reg gvn simplifycfg"
    #             # 你需要将其映射回 CompilerGym 的 Action 索引
    #             try:
    #                 # 执行序列，获取 reward (例如：代码体积压缩率)
    #                 _, reward, done, info = env.multistep(pass_string.split())
    #                 # 如果执行失败或编译崩溃，给一个极大的负惩罚
    #                 rewards.append(torch.tensor(float(reward)))
    #             except Exception:
    #                 rewards.append(torch.tensor(-1.0)) 
            
    #         env.close()

    #         # --- C. 优化阶段 ---
    #         # 核心：将 (queries, responses, rewards) 喂给 PPO
    #         # PPO 会计算：当前的概率比、价值函数损失、以及与 ref_model 的 KL 惩罚
    #         stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
    #         # 打印日志（如平均奖励）
    #         ppo_trainer.log_stats(stats, batch, rewards)