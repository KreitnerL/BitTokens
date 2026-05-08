#!/usr/bin/env bash
source .env

REASONING_CONTROL_MODELS=(
    # "openai/gpt-5"
    # "openai/gpt-oss-120b"
    # "deepseek/deepseek-chat-v3.1"
    # "google/gemini-2.5-pro"
    "google/gemini-2.5-flash"
    # "x-ai/grok-4-fast:free"
)


OTHER_MODELS=(
    # "meta-llama/llama-4-maverick"
    # "qwen/qwen3-235b-a22b-thinking-2507"
    # "qwen/qwen3-235b-a22b-2507"
    # "moonshotai/kimi-k2-0905"
)
    # "anthropic/claude-opus-4.1"
    # "x-ai/grok-4"

TASKS=(
    "Division"
    "Addition"
    # "Exponentiation"
    # "Interval"
    # "Mean"
    # "MinMax"
    # "Multiplication"
    # "Sorting"
    # "Std"
)

LIMIT=500

DATA_BASE_PATH="$PROJECT_PATH/test_data_2025-09-04_shuffled_correct"
FILE_SUFFIX="_decimal_uniform_test_10k_shuffle.csv"

cd $PROJECT_PATH

for model in ${OTHER_MODELS[@]}; do
    for task in ${TASKS[@]}; do
        uv run $PROJECT_PATH/eval_scripts/openrouter_eval.py --model_name $model --input_file $DATA_BASE_PATH/$task$FILE_SUFFIX --limit $LIMIT &
    done
done

for model in ${REASONING_CONTROL_MODELS[@]}; do
    for task in ${TASKS[@]}; do
        # for reasoning_effort in "high" "none"; do
        # for reasoning_effort in "none"; do
        for reasoning_effort in "high"; do
            uv run $PROJECT_PATH/eval_scripts/openrouter_eval.py --model_name $model --input_file $DATA_BASE_PATH/$task$FILE_SUFFIX --reasoning_effort $reasoning_effort --limit $LIMIT --use_batch --batch_size 1 --batch_concurrency 10 &
        done
    done
done
