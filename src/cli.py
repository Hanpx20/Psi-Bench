#!/usr/bin/env python
"""
Psi-Bench CLI - Command line interface for running evaluations
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path

def print_help():
    """Print help message"""
    help_text = """
Psi-Bench Evaluation CLI

USAGE:
    psi-bench [COMMAND] [OPTIONS]

COMMANDS:
    eval          - Run evaluations on benchmark datasets
    download      - Download benchmark datasets
    help          - Show this help message

EVAL USAGE:
    psi-bench eval [TASK] [OPTIONS]

    TASKS:
        cmv       - Evaluate on CMV (Change My View) dataset
        counsel   - Evaluate on Counsel dataset
        request   - Evaluate on Request dataset
        all       - Run all three evaluations

    OPTIONS:
        --tested_model MODEL          (required) Model to test
        --client_model MODEL          Client model (default: deepseek-v3-2)
        --judge_model MODEL           Judge model (default: deepseek-v3-2)
        --n_turns N                   Number of turns (default: 3)
        --test_oracle                 Test oracle mode
        --persuader_local             Use local persuader model
        --inference_parallel N        Parallel workers (default: 16)

DOWNLOAD USAGE:
    psi-bench download [DATASET]

    DATASETS:
        cmv       - Change My View dataset
        counsel   - Counsel dataset
        request   - Request dataset
        all       - All datasets (default)

    OPTIONS:
        --output PATH                 Output directory (default: ./data)

EXAMPLES:
    # Evaluate model on all datasets
    psi-bench eval all --tested_model gpt-4

    # Evaluate on specific dataset with local model
    psi-bench eval cmv --tested_model Qwen/Qwen3-8B --persuader_local

    # Download all datasets
    psi-bench download all

For more details, see:
- API Key Setup: https://github.com/yourusername/Psi-Bench/blob/main/API_KEY_CONFIG.md
- User Guide: https://github.com/yourusername/Psi-Bench/blob/main/USER_GUIDE.md
"""
    print(help_text)


def run_eval_inference(task, tested_model, client_model, judge_model, n_turns, test_oracle,
                       persuader_local, inference_parallel):
    """Run inference for a specific task"""
    project_root = Path(__file__).parent.parent

    # Data and output file configurations
    data_config = {
        "cmv": {
            "persona_file": "data/cmv/persona_profile.json",
            "conv_file": "data/cmv/queries.json",
        },
        "counsel": {
            "persona_file": "data/counsel/persona_profile.json",
            "conv_file": "data/counsel/queries.json",
        },
        "request": {
            "persona_file": "data/request/persona_profile.json",
            "conv_file": "data/request/queries.json",
        },
    }

    if task not in data_config:
        print(f"Error: Unknown task '{task}'")
        return 1

    config = data_config[task]
    model_name = tested_model.split("/")[-1]

    # Build inference.py command
    inference_cmd = [
        sys.executable,
        str(project_root / "src" / "inference.py"),
        "--client_model", client_model,
        "--persona_file", str(project_root / config["persona_file"]),
        "--conv_file", str(project_root / config["conv_file"]),
        "--persuader_model", tested_model,
        "--output", str(project_root / f"eval/{task}/convs/{model_name}{'_oracle' if test_oracle else ''}.json"),
        "--size", "500",
        "--n_turns", str(n_turns),
        "--inference_parallel", str(inference_parallel),
    ]

    if test_oracle:
        inference_cmd.extend(["--profile_mode", "oracle"])

    if persuader_local:
        inference_cmd.append("--persuader_local")

    print(f"\n{'='*60}")
    print(f"Running {task.upper()} Inference")
    print(f"{'='*60}")
    print(f"Command: {' '.join(inference_cmd)}\n")

    result = subprocess.run(inference_cmd, cwd=project_root)
    if result.returncode != 0:
        print(f"Error: Inference failed for {task}")
        return result.returncode

    # Build llm_judge_eval.py command
    judge_cmd = [
        sys.executable,
        str(project_root / "src" / "llm_judge_eval.py"),
        "--judge_model", judge_model,
        "--conv_file", str(project_root / f"eval/{task}/convs/{model_name}{'_oracle' if test_oracle else ''}.json"),
        "--persona_file", str(project_root / config["persona_file"]),
        "--output", str(project_root / f"eval/{task}/{model_name}{'_oracle' if test_oracle else ''}_judge.json"),
        "--inference_parallel", str(inference_parallel),
    ]

    print(f"\n{'='*60}")
    print(f"Running {task.upper()} Judge Evaluation")
    print(f"{'='*60}")
    print(f"Command: {' '.join(judge_cmd)}\n")

    result = subprocess.run(judge_cmd, cwd=project_root)
    if result.returncode != 0:
        print(f"Error: Judge evaluation failed for {task}")
        return result.returncode

    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Psi-Bench: Evaluating Persona-Sensitive Influencing in Persuasive Dialogues",
        add_help=False
    )

    parser.add_argument('command', nargs='?', help='Command to run (eval, download, help)')
    parser.add_argument('task', nargs='?', help='Task for eval command (cmv, counsel, request, all)')

    # Eval options
    parser.add_argument('--tested_model', type=str, help='Model to test')
    parser.add_argument('--client_model', type=str, default='deepseek-v3-2', help='Client model')
    parser.add_argument('--judge_model', type=str, default='deepseek-v3-2', help='Judge model')
    parser.add_argument('--n_turns', type=int, default=3, help='Number of turns')
    parser.add_argument('--test_oracle', action='store_true', help='Test oracle mode')
    parser.add_argument('--persuader_local', action='store_true', help='Use local persuader model')
    parser.add_argument('--inference_parallel', type=int, default=16, help='Number of parallel workers')

    # Download options
    parser.add_argument('--output', type=str, default='./data', help='Output directory')

    parser.add_argument('--help', '-h', action='store_true', help='Show help message')

    args, remaining = parser.parse_known_args()

    # Handle help
    if args.help or not args.command or args.command in ['help', '-h', '--help']:
        print_help()
        return 0

    if args.command == 'eval':
        # Validate eval command
        if not args.task:
            print("Error: eval command requires a task (cmv, counsel, request, or all)")
            print("\nUsage: psi-bench eval [cmv|counsel|request|all] [OPTIONS]")
            return 1

        if not args.tested_model:
            print("Error: --tested_model is required")
            return 1

        if args.task not in ['cmv', 'counsel', 'request', 'all']:
            print(f"Error: Unknown task '{args.task}'")
            return 1

        # Ensure output directories exist
        project_root = Path(__file__).parent.parent
        for task_dir in ['cmv', 'counsel', 'request']:
            (project_root / f"eval/{task_dir}/convs").mkdir(parents=True, exist_ok=True)

        # Run evaluation(s)
        tasks = ['cmv', 'counsel', 'request'] if args.task == 'all' else [args.task]

        for task in tasks:
            result = run_eval_inference(
                task=task,
                tested_model=args.tested_model,
                client_model=args.client_model,
                judge_model=args.judge_model,
                n_turns=args.n_turns,
                test_oracle=args.test_oracle,
                persuader_local=args.persuader_local,
                inference_parallel=args.inference_parallel
            )
            if result != 0:
                return result

        print(f"\n{'='*60}")
        print("✓ All evaluations completed successfully!")
        print(f"{'='*60}\n")
        return 0

    elif args.command == 'download':
        from . import download_data
        sys.argv = ['psi-bench-download'] + ([args.task] if args.task else []) + remaining
        return download_data.main()

    else:
        print(f"Error: Unknown command '{args.command}'")
        print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
