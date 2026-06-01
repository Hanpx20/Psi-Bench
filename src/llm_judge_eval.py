import json
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import *

task_to_keys = {
    "cmv": ["general_conversation_quality", "personality_perception", "personalized_argumentation", "persuasion_effect"],
    "counsel": ["general_conversation_quality", "personality_perception", "personalized_response", "treatment_effect"],
    "request": ["general_conversation_quality", "personality_perception", "personalized_response", "request_effect"],
}

def init_stats_dict(metric_keys, turn_count=None):
    """Initialize statistics dictionary with metric keys."""
    stats = {}
    if turn_count is not None:
        stats["turn_count"] = turn_count
    for metric_key in metric_keys:
        stats[metric_key] = {"scores": [], "mean": 0}
    return stats

def collect_scores_from_response(response, metric_key):
    """Extract scores for a specific metric from a response."""
    if isinstance(response, dict) and metric_key in response:
        score = response[metric_key].get("score")
        if score is not None:
            return score
    return None

def collect_all_scores(threads, metric_keys, response_getter):
    """Collect scores for all metrics from threads.
    
    Args:
        threads: List of thread objects
        metric_keys: List of metric key strings
        response_getter: Function that takes a thread and returns the response (or dict of responses)
    
    Returns:
        Dictionary mapping metric_key to list of scores
    """
    scores_dict = {key: [] for key in metric_keys}
    
    for thread in threads:
        response = response_getter(thread)
        for metric_key in metric_keys:
            score = collect_scores_from_response(response, metric_key)
            if score is not None:
                scores_dict[metric_key].append(score)
    
    return scores_dict

def calculate_metric_stats(scores):
    """Calculate mean and distribution for a set of scores."""
    if not scores:
        return {"mean": 0, "distribution": []}
    
    mean = round(sum(scores) / len(scores), 3)
    max_score = max(scores)
    distribution = [0] * int(max_score)
    for score in scores:
        if 1 <= score <= max_score:
            distribution[int(score) - 1] += 1
    
    return {"mean": mean, "distribution": json.dumps(distribution)}

def finalize_stats(stats, metric_keys):
    """Convert stats from scores lists to metrics with mean and distribution."""
    finalized = {}
    for metric_key in metric_keys:
        scores = stats[metric_key].pop("scores")
        metric_stats = calculate_metric_stats(scores)
        finalized[metric_key] = metric_stats
    
    # Copy other keys (e.g., turn_count, total_conversations)
    for key, value in stats.items():
        if key not in metric_keys:
            finalized[key] = value
    
    return finalized

def calculate_avg_similarity(threads):
    """Calculate average similarity and coverage for even-indexed conversation elements.
    
    Args:
        threads: List of thread objects, each with a "conv" field
    
    Returns:
        Dictionary with "avg_similarity" and "avg_coverage" keys, each mapping index (as string) to average value.
        Returns empty dict if no relevant fields found.
    """
    if not threads or "conv" not in threads[0] or len(threads[0]["conv"]) == 0:
        return {}
    
    first_conv_item = threads[0]["conv"][0]
    if not isinstance(first_conv_item, dict):
        return {}
    
    # Map source field names to output keys
    metric_mapping = {
        "persona_similarity": "avg_similarity",
        "persona_coverage": "avg_coverage"
    }
    
    # Initialize metrics dictionary with fields that exist in data
    metrics = {src: {} for src in metric_mapping if src in first_conv_item}
    if not metrics:
        return {}
    
    # Collect values from all threads at even indices
    for thread in threads:
        if "conv" not in thread:
            continue
        
        for idx in range(0, len(thread["conv"]), 2):
            conv_item = thread["conv"][idx]
            if not isinstance(conv_item, dict):
                continue
            
            for metric_name in metrics:
                if metric_name in conv_item:
                    value = conv_item[metric_name]
                    if isinstance(value, (int, float)):
                        if idx not in metrics[metric_name]:
                            metrics[metric_name][idx] = []
                        metrics[metric_name][idx].append(value)
    
    # Calculate and format averages
    result = {}
    for metric_name, output_key in metric_mapping.items():
        if metric_name in metrics and metrics[metric_name]:
            avg_dict = {}
            for idx in sorted(metrics[metric_name].keys()):
                values = metrics[metric_name][idx]
                if values:
                    # Exclude zero values as they indicate erroneous data
                    non_zero_values = [v for v in values if v != 0]
                    if non_zero_values:
                        avg = round(sum(non_zero_values) / len(non_zero_values), 3)
                        avg_dict[str(idx)] = avg
            if avg_dict:
                result[output_key] = avg_dict
    
    return result

def calculate_growth_rate(stats_by_turns, metric_keys):
    turn_counts = sorted(stats_by_turns.keys())
    
    growth_rates = {}
    for metric_key in metric_keys:
        points = [(tc, stats_by_turns[tc][metric_key].get("mean", 0)) 
                  for tc in turn_counts if metric_key in stats_by_turns[tc]]
        
        if len(points) >= 2:
            x_vals, y_vals = zip(*points)
            growth_rates[metric_key] = round(float(np.polyfit(x_vals, y_vals, 1)[0]), 3)
        else:
            growth_rates[metric_key] = 0
    
    return growth_rates
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--judge_model", type=str, default="deepseek-ai/deepseek-v3.2")
    parser.add_argument("--conv_file", type=str, default="data/cmv/queries.json", help="Input Conv path")
    parser.add_argument("--persona_file", type=str, default="data/cmv/cmv_persona_profile.json", help="Input Persona path")
    parser.add_argument("--output", type=str, default="eval/cmv/cmv_llmjudge_results.json", help="Output file path")
    parser.add_argument("--size", type=int, default=-1, help="Number of threads to process")
    parser.add_argument("--enumerate_turns", action="store_true", help="Whether to evaluate turn 1, turn 1-2, ... turn 1-n separately.")
    parser.add_argument("--from_turn", type=int, default=1, help="Start turn count (inclusive) for enumerate_turns mode")
    parser.add_argument("--to_turn", type=int, default=2, help="End turn count (inclusive) for enumerate_turns mode")
    parser.add_argument("--inference_parallel", type=int, default=16, help="Number of threads for API inference")
    parser.add_argument("--task", type=str, default="cmv", choices = ["cmv", "counsel", "request"], help="Task type for evaluation")
    
    args = parser.parse_args()
    
    # Get metric keys for the current task
    current_metrics = task_to_keys[args.task]
    if args.task == "cmv":
        sys_prompt = JUDGE_SYS_PROMPT
    elif args.task == "counsel":
        sys_prompt = JUDGE_SYS_PROMPT_COUNSEL
    elif args.task == "request":
        sys_prompt = JUDGE_SYS_PROMPT_REQUEST
    
    requests = []
    request_metadata = []  # Track which thread and turn count each request corresponds to
    
    persona_profiles = json.load(open(args.persona_file, "r"))
    threads = json.load(open(args.conv_file, "r"))
    
    if isinstance(threads, dict):
        threads = threads["content"]
    else:
        assert False, "Unexpected format for conv_file, expected a dict with 'content' key"
    if args.size != -1:
        threads = threads[:args.size]
            
    for thread_idx, thread in enumerate(threads):
        persona = persona_profiles[thread["persona"]]
        question = thread["question"]
        conv = thread["conv"]
        
        if args.enumerate_turns:
            max_turns = len(conv) // 2
            to_turn = min(args.to_turn, max_turns)
            turn_counts = list(range(args.from_turn, to_turn + 1, 1))
            
            for num_turns in turn_counts:
                conv_prefix = conv[: (num_turns * 2)]
                request = [{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": format_judge_prompt(persona, question, conv_prefix, task=args.task)}]
                requests.append(request)
                request_metadata.append({"thread_idx": thread_idx, "num_turns": num_turns})
        else:
            # Original behavior: single request with all turns
            request = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": format_judge_prompt(persona, question, conv, task=args.task)}]
            requests.append(request)
            request_metadata.append({"thread_idx": thread_idx, "num_turns": len(conv)})
    
    
    print(f"Running evaluation on {len(requests)} conversations...")
    results = api_batch_inference(requests, sampling_params = {"temperature": 0.0}, model=args.judge_model, n_threads=args.inference_parallel, progress=True, role='judge')
    
    # Process results and match them back to threads with turn information
    if args.enumerate_turns:
        # Initialize turn evaluations for each thread
        for thread in threads:
            thread["llm_judge_responses"] = []  # List of {num_turns, response} dicts
        
        for result_idx, (metadata, result) in enumerate(zip(request_metadata, results)):
            thread_idx = metadata["thread_idx"]
            num_turns = metadata["num_turns"]
            threads[thread_idx]["llm_judge_responses"].append({
                "num_turns": num_turns,
                "response": process_json(result)
            })
    else:
        idx = 0
        for thread in threads:
            thread["llm_judge_response"] = process_json(results[idx])
            idx += 1
    
    output_path = args.output
    with open(output_path, "w") as f:
        if args.enumerate_turns:
            # Collect responses grouped by turn count
            from collections import defaultdict
            responses_by_turns = defaultdict(list)
            
            for thread in threads:
                for item in thread["llm_judge_responses"]:
                    num_turns = item["num_turns"]
                    responses_by_turns[num_turns].append(item["response"])
            
            # Calculate statistics for each turn count separately
            stats_by_turns = {}
            for turn_count in sorted(responses_by_turns.keys()):
                stats = init_stats_dict(current_metrics, turn_count=turn_count)
                
                # Create a temporary thread list with only responses for this turn
                temp_threads = []
                for response in responses_by_turns[turn_count]:
                    temp_threads.append({"response": response})
                
                # Collect scores for this specific turn count
                def get_response(thread):
                    return thread.get("response", {})
                
                scores_dict = collect_all_scores(temp_threads, current_metrics, get_response)
                
                # Update stats with collected scores
                for metric_key in current_metrics:
                    stats[metric_key]["scores"] = scores_dict[metric_key]
                
                # Finalize statistics
                stats_by_turns[turn_count] = finalize_stats(stats, current_metrics)
            
            # Calculate growth rates for each metric
            growth_rates = calculate_growth_rate(stats_by_turns, current_metrics)
            
            # Add growth_rate to each metric in stats_by_turns
            for turn_count in stats_by_turns:
                for metric_key in current_metrics:
                    if metric_key in stats_by_turns[turn_count]:
                        stats_by_turns[turn_count][metric_key]["growth_rate"] = growth_rates[metric_key]
            
            # Calculate and add average similarity and coverage for even-indexed conversation elements
            avg_metrics = calculate_avg_similarity(threads)
            
            # Merge avg_metrics into stats_by_turns
            stats_by_turns["growth_rates"] = growth_rates
            for key in ["avg_similarity", "avg_coverage"]:
                if key in avg_metrics:
                    stats_by_turns[key] = avg_metrics[key]
            
            json.dump({"hparams": vars(args), "stats_by_turns": stats_by_turns, "content": threads}, f, indent=2, ensure_ascii=False)
        else:
            # Original behavior: single statistics calculation
            stats = init_stats_dict(current_metrics)
            stats["total_conversations"] = len(requests)
            
            # Collect scores from all threads
            def get_response(thread):
                return thread.get("llm_judge_response", {})
            
            scores_dict = collect_all_scores(threads, current_metrics, get_response)
            
            # Update stats with collected scores
            for metric_key in current_metrics:
                stats[metric_key]["scores"] = scores_dict[metric_key]
            
            # Finalize statistics
            stats = finalize_stats(stats, current_metrics)
            
            # Calculate and add average similarity and coverage for even-indexed conversation elements
            avg_metrics = calculate_avg_similarity(threads)
            for key in ["avg_similarity", "avg_coverage"]:
                if key in avg_metrics:
                    stats[key] = avg_metrics[key]

            json.dump({"hparams": vars(args), "stats": stats, "content": threads}, f, indent=2, ensure_ascii=False)