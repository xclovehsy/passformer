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
python -m src.training.passformer_seq2seq_train_v2 --config /home/xucong24/Compiler/configs/passformer_gallvm_seq2seq_v2.yaml


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