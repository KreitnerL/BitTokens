#!/usr/bin/env bash

REASONING_CONTROL_MODELS=(
    "openai/gpt-5"
    "openai/gpt-oss-120b"
    "deepseek/deepseek-chat-v3.1"
    "google/gemini-2.5-pro"
    "google/gemini-2.5-flash"
)


OTHER_MODELS=(
    "meta-llama/llama-4-maverick"
    "qwen/qwen3-235b-a22b-thinking-2507"
    "qwen/qwen3-235b-a22b-2507"
    "moonshotai/kimi-k2-0905"
)

TASKS=(
    "Division"
    "Addition"
    "Exponentiation"
    "Interval"
    "Mean"
    "MinMax"
    "Multiplication"
    "Sorting"
    "Std"
)

LIMIT=500

DATA_BASE_PATH="./frontier_model_analysis/results"
FILE_SUFFIX="_decimal_uniform_test_500_shuffle.csv"


for model in ${OTHER_MODELS[@]}; do
    for task in ${TASKS[@]}; do
        uv run ./frontier_model_analysis/openrouter_eval.py --model_name $model --input_file $DATA_BASE_PATH/$task$FILE_SUFFIX &
    done
done

for model in ${REASONING_CONTROL_MODELS[@]}; do
    for task in ${TASKS[@]}; do
        for reasoning_effort in "high" "none"; do
            uv run ./frontier_model_analysis/openrouter_eval.py --model_name $model --input_file $DATA_BASE_PATH/$task$FILE_SUFFIX --reasoning_effort $reasoning_effort --use_batch --batch_size 1 --batch_concurrency 10 &
        done
    done
done
