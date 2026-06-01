#!/bin/bash
set -e

# Parse named parameters
tested_model=""
client_model="deepseek-v3-2"
judge_model="deepseek-v3-2"
n_turns=3
test_oracle=false
persuader_local=false
inference_parallel=16

while [[ $# -gt 0 ]]; do
    case $1 in
        --tested_model) tested_model="$2"; shift 2 ;;
        --client_model) client_model="$2"; shift 2 ;;
        --judge_model) judge_model="$2"; shift 2 ;;
        --n_turns) n_turns="$2"; shift 2 ;;
        --test_oracle) test_oracle=true; shift ;;
        --persuader_local) persuader_local=true; shift ;;
        --inference_parallel) inference_parallel="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$tested_model" ]; then
    echo "Error: --tested_model is required"
    exit 1
fi

model_name="${tested_model##*/}"
persuader_local_flag=""
if [ "$persuader_local" = true ]; then
    persuader_local_flag="--persuader_local"
fi

if [ "$test_oracle" = true ]; then
    python src/inference.py \
        --client_model "$client_model" \
        --persona_file data/request/persona_profile.json \
        --conv_file data/request/queries.json \
        --persuader_model "$tested_model" --task request \
        --output eval/request/convs/${model_name}_oracle.json \
        --n_turns "$n_turns" --profile_mode oracle \
        --inference_parallel "$inference_parallel" $persuader_local_flag

    python src/llm_judge_eval.py \
        --judge_model "$judge_model" \
        --conv_file eval/request/convs/${model_name}_oracle.json \
        --persona_file data/request/persona_profile.json \
        --output eval/request/${model_name}_oracle_judge.json \
        --inference_parallel "$inference_parallel" --task request
else
    python src/inference.py \
        --client_model "$client_model" \
        --persona_file data/request/persona_profile.json \
        --conv_file data/request/queries.json \
        --persuader_model "$tested_model" --task request \
        --output eval/request/convs/${model_name}.json \
        --n_turns "$n_turns" \
        --inference_parallel "$inference_parallel" $persuader_local_flag

    python src/llm_judge_eval.py \
        --judge_model "$judge_model" \
        --conv_file eval/request/convs/${model_name}.json \
        --persona_file data/request/persona_profile.json \
        --output eval/request/${model_name}_judge.json \
        --inference_parallel "$inference_parallel" --task request
fi