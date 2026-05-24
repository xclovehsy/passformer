# test inst2vec tokenizer
# python -m src.tests.test_inst2vec_tokenizer

# python -m src.llvm_opt_gen_train --config yaml/llvm_opt_gen_train.yaml
# python -m src.inst2vec_bert_mlm_train --config /home/xucong24/Compiler/yaml/inst2vec_poj104_modernbert_train.yaml

# inst2vec_modernbert poj104 classify 
# python -m src.experiments.modernbert_classifyapp_inst2vec

# inst2vec_modernbert poj104 mlm train
# python -m src.training.instbert_mlm_trainer --config /home/xucong24/Compiler/configs/instbert_poj104_mlm.yaml

# passformer optseq seq2seq train
# python -m src.training.passformer_seq2seq_train --config /home/xucong24/Compiler/configs/passformer_gallvm_seq2seq.yaml

# passformer autophase train
# python -m src.training.passformer_autophase_train --config /home/xucong24/Compiler/configs/passformer_gallvm_autophase.yaml
# python -m src.training.passformer_seq2seq_train_v2 --config /home/xucong24/Compiler/configs/passformer_gallvm_seq2seq_add.yaml


# tokenize llvm_opti_seq dataset
# python -m src.data.tokenize_passformer_dataset \
#     --data_dir /home/xucong24/Compiler/datasets/ga_llvm_37k \
#     --output_dir /home/xucong24/Compiler/datasets/ga_llvm_37k_passformer_1024_tokenized \
#     --inst2vec_tokenizer_id /home/xucong24/Compiler/checkpoints/Inst2VecTokenizer \
#     --opti_seq_tokenizer_id /home/xucong24/Compiler/checkpoints/OptiSeqTokenizer \
#     --encoder_maxlen 1024 \
#     --decoder_maxlen 256 \
#     --num_proc 32 \
#     --split_train_test \
#     --test_size 0.1 \
#     --split_seed 42

# test tokenized passformer dataset
# python -m src.tests.test_tokenized_passformer_dataset \
#     --data_dir /home/xucong24/Compiler/datasets/ga_llvm_37k_passformer_tokenized \
#     --inst2vec_tokenizer_id /home/xucong24/Compiler/checkpoints/Inst2VecTokenizer \
#     --opti_seq_tokenizer_id /home/xucong24/Compiler/checkpoints/OptiSeqTokenizer \
#     --num_samples 20

# passformer optseq seq2seq inference
# python -m src.inference.optseq_gen_inference \
#     --model_path /home/xucong24/Compiler/work_dirs/passformer_gallvm_seq2seq/20260110_082146/final_model \
#     --input /home/xucong24/Compiler/tmp/37902.lll \
#     --max_input_length 1024 \
#     --max_output_length 32 \
#     --num_beams 1 \
#     --encoder_tokenizer_type inst2vec \
#     --decoder_tokenizer_type optiseq \
#     --device cpu

# passformer optseq seq2seq evaluate
# python -m src.evaluation.passformer_evaluate \
#     --model_path /home/xucong24/Compiler/work_dirs/passformer_gallvm_seq2seq/20260110_082146/final_model \
#     --benchmark_dir /home/xucong24/Compiler/datasets/cbench-v1 \
#     --llvm_path /home/xucong24/.local/share/compiler_gym/llvm-v0/bin \
#     --output_dir /home/xucong24/Compiler/work_dirs/passformer_gallvm_seq2seq/20260110_082146/evaluation_results \
#     --max_input_length 1024 \
#     --max_output_length 256 \
#     --num_beams 1 \
#     --encoder_tokenizer_type inst2vec \
#     --device cpu

# verify evaluation results with CompilerGym

# test passformer autophase
# python -m src.model.passformer \
#     --encoder_path /home/xucong24/Compiler/checkpoints/modernbert_poj104_mlm \
#     --decoder_path /home/xucong24/Compiler/checkpoints/gpt2 \
#     --test_fusion_method decoder_prefix
#     --device cpu

# rl
# python -m src.rl.train --config /home/xucong24/Compiler/configs/grpo.yaml
# python -m src.reinforce.train --config /home/xucong24/Compiler/configs/reinforce.yaml
# python -m src.reinforce.train_sequential --config /home/xucong24/Compiler/configs/reinforce_sequential.yaml


# python -m src.reinforce.train_largescale --config /home/xucong24/Compiler/configs/reinforce_codecontest_v2.yaml --max_train_samples 100
# python -m src.reinforce.train_largescale --config /home/xucong24/Compiler/configs/reinforce_codecontest_v2.yaml
# python -m src.reinforce.train_largescale --config /home/xucong24/Compiler/configs/reinforce_codecontest_v3.yaml
# python -m src.reinforce.train_largescale --config /home/xucong24/Compiler/configs/reinforce_codecontest_v4.yaml
# python -m src.reinforce.train_largescale --config /home/xucong24/Compiler/configs/reinforce_codecontest_add_v1.yaml
# python -m src.reinforce.train_largescale --config /home/xucong24/Compiler/configs/reinforce_codecontest_v7.yaml
# python -m src.reinforce.train_largescale --config /home/xucong24/Compiler/configs/reinforce_codecontest_add_v2.yaml
# python -m src.reinforce.train_largescale --config /home/xucong24/Compiler/configs/reinforce_codecontest_only_v1.yaml
# python -m src.reinforce.train_largescale --config /home/xucong24/Compiler/configs/reinforce_cbench_v7.yaml

# python -m src.reinforce.test_cbench --config configs/reinforce_test_cbench.yaml
# python -m src.reinforce.test_cbench --config configs/reinforce_test_cbench.yaml \
#     --model_path /home/xucong24/Compiler/work_dirs/reinforce_sequential/best/final_model \
#     --strategies sampling


# # 只运行某些策略
# python -m src.reinforce.test_cbench --config configs/reinforce_test_cbench.yaml \
#     --strategies greedy sampling



# python -m src.reinforce.test \
#     --model_path /home/xucong24/Compiler/work_dirs/reinforce_cbench_v2/20260317_160719/best_model \
#     --encoder_tokenizer_path /home/xucong24/Compiler/checkpoints/Inst2VecTokenizer \
#     --decoder_tokenizer_path /home/xucong24/Compiler/checkpoints/OptiSeqTokenizer \
#     --num_rollouts 16 \
#     --temperatures 0.3,0.7 \
#     --max_gen_length 32 \
#     --leaderboard_results passformer_results.csv \
#     --n 1


# test_decode
# 温度 6 格点（0.2→1.2、步进 0.2）：自低到高覆盖。每个温度各 --num_rollouts 条；条数 x 6 为 sampling 量。
# 只输入目录（递归子目录内所有 .bc）
python -m src.reinforce.test_decode \
    --model_path /home/xucong24/Compiler/work_dirs/reinforce_codecontest_v7/20260425_154502/best_model \
    --bc_dir /home/xucong24/Compiler/datasets/compilerdream_data/codecontest_test   \
    --modes greedy,beam,sampling,sampling_topp \
    --num_rollouts 32 \
    --temperatures=0.2,0.4,0.6,0.8,1.0,1.2 \
    --encoder_tokenizer_path /home/xucong24/Compiler/checkpoints/Inst2VecTokenizer \
    --decoder_tokenizer_path /home/xucong24/Compiler/checkpoints/OptiSeqTokenizer \
    --seed 42

# only 
# /home/xucong24/Compiler/work_dirs/reinforce_codecontest_only_v1/20260427_145345/best_model
# add
# /home/xucong24/Compiler/work_dirs/reinforce_codecontest_add_v2/20260426_153759/best_model
# concat 
# 
# /home/xucong24/Compiler/work_dirs/reinforce_codecontest_v7/20260425_154502/best_model