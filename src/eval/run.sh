#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${SCRIPT_DIR}"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

python3 ./eval.py \
  --model_path="/home/xucong24/Compiler/work_dirs/reinforce_codecontest_v7/20260425_154502/best_model" \
  --encoder_tokenizer_path="/home/xucong24/Compiler/checkpoints/Inst2VecTokenizer" \
  --decoder_tokenizer_path="/home/xucong24/Compiler/checkpoints/OptiSeqTokenizer" \
  --decode_method="sampling_topp" \
  --num_samples=16 \
  --temperature=0.7 \
  --top_p=0.95 \
  --num_beams=16 \
  --max_input_length=512 \
  --max_ir_lines=1024 \
  --soft_eval_time_limit_s=5 \
  --warmup_generate_rounds=1 \
  --max_gen_length=32 \
  --num_eval_workers=16 \
  --bc_dir="/home/xucong24/Compiler/datasets/cbench-v1" \
  --bc_recursive=true \
  --leaderboard_results="./passformer_codecontest_test.csv" \
  --leaderboard_logfile="./passformer_codecontest_test.log" \
  --eval_output_dir="." \
  --n=1 \
  --max_benchmarks=0
