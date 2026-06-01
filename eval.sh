#!/bin/bash

show_help() {
    cat << EOF
Psi-Bench Evaluation Script

USAGE:
    eval.sh [COMMAND] [OPTIONS]

COMMANDS:
    cmv       - Evaluate on CMV (Change My View) dataset
    counsel   - Evaluate on Counsel dataset
    request   - Evaluate on Request dataset
    all       - Run all three evaluations
    help      - Show this help message

COMMON OPTIONS:
    --tested_model MODEL          (required) Model to test (e.g., "gpt-4", "Qwen/Qwen3-8B")
    --client_model MODEL          Client model (default: deepseek-v3-2)
    --judge_model MODEL           Judge model for evaluation (default: deepseek-v3-2)
    --n_turns N                   Number of conversation turns (default: 3)
    --test_oracle                 Test oracle mode (uses reference responses)
    --persuader_local            Use local persuader model
    --inference_parallel N        Number of parallel inference workers (default: 16)

EXAMPLES:
    # Evaluate Qwen model on CMV dataset
    ./eval.sh cmv --tested_model Qwen/Qwen3-8B

    # Test oracle mode on all datasets with custom judge model
    ./eval.sh all --tested_model gpt-4 --judge_model gpt-4-turbo --test_oracle

    # Run counsel evaluation with 5 turns
    ./eval.sh counsel --tested_model mistral-7b --n_turns 5 --persuader_local

EOF
}

# Show help if no arguments or help command
if [ $# -eq 0 ] || [ "$1" = "help" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Extract command and remove it from arguments
COMMAND="$1"
shift

# Run the appropriate evaluation script
case "$COMMAND" in
    cmv)
        echo "Running CMV evaluation..."
        bash scripts/eval_model_cmv.sh "$@"
        ;;
    counsel)
        echo "Running Counsel evaluation..."
        bash scripts/eval_model_counsel.sh "$@"
        ;;
    request)
        echo "Running Request evaluation..."
        bash scripts/eval_model_request.sh "$@"
        ;;
    all)
        echo "Running all evaluations..."
        bash scripts/eval_model_cmv.sh "$@"
        bash scripts/eval_model_counsel.sh "$@"
        bash scripts/eval_model_request.sh "$@"
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        show_help
        exit 1
        ;;
esac