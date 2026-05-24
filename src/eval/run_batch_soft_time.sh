#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${SCRIPT_DIR}"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

WORK_DIR_BASE="./work_dirs"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
TIME_LIMITS="5 10 15 20 25 30"

mkdir -p "${WORK_DIR_BASE}"

for t in ${TIME_LIMITS}; do
  OUT_DIR="${WORK_DIR_BASE}/passformer_concat_${t}s_${RUN_TAG}"
  mkdir -p "${OUT_DIR}"

  echo "=== Running soft_eval_time_limit_s=${t} (output: ${OUT_DIR}) ==="
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
    --soft_eval_time_limit_s="${t}" \
    --warmup_generate_rounds=1 \
    --max_gen_length=32 \
    --num_eval_workers=16 \
    --bc_dir="/home/xucong24/Compiler/datasets/cbench-v1" \
    --bc_recursive=true \
    --leaderboard_results="${OUT_DIR}/passformer_codecontest_test.csv" \
    --leaderboard_logfile="${OUT_DIR}/passformer_codecontest_test.log" \
    --eval_output_dir="${OUT_DIR}" \
    --n=10 \
    --max_benchmarks=0
done

echo "All runs completed under ${WORK_DIR_BASE} (run tag: ${RUN_TAG})."
