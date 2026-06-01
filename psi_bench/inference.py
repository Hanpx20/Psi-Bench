import json
import argparse
from transformers import AutoTokenizer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import *


def process_persuader_response(response, keys={"think", "argument"}, tokenizer=None, cutoff=200):
    results = {k: extract_answer(response, k, strict=False) for k in keys}
    if "argument" in results and results["argument"] and tokenizer:
        t = tokenizer.encode(results["argument"], add_special_tokens=False)[:cutoff]
        s = tokenizer.decode(t, skip_special_tokens=True)
        i = max(s.rfind('.'), s.rfind('!'), s.rfind('?'))
        results["argument"] = s[:i+1].strip() if i != -1 else s.strip()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--client_model", type=str, required=True)
    parser.add_argument("--persuader_model", type=str)
    parser.add_argument("--persuader_local", action="store_true", help="Use local persuader model")
    parser.add_argument("--persona_infer_model", type=str)
    parser.add_argument("--inference_parallel", type=int, default=16, help="Number of threads for API inference")
    parser.add_argument("--infer_model_local", action="store_true")
    
    parser.add_argument("--persona_file", type=str, required=True, help="Input Persona path")
    parser.add_argument("--conv_file", type=str, required=True, help="Input Conv path")
    parser.add_argument("--output", type=str, default="eval/cmv/cmv_predictiveness_results.json", help="Output file path")
    parser.add_argument("--size", type=int, default=0, help="Number of conversations to process")
    parser.add_argument("--no_persona", action="store_true", help="Whether to include persona information in the prompt")
    parser.add_argument("--cont", action="store_true", help="Continue generation from existing conversations in conv_file (requires --generation_mode=generate)")
    
    
    parser.add_argument("--task", type=str, default="cmv", choices = ["cmv", "counsel", "request"], help="Task type for evaluation")
    parser.add_argument("--generation_mode", type=str, default="generate", choices=["append", "generate"], help="Generate is the normal method; append is continuing on human dialogues.")
    parser.add_argument("--profile_mode", type=str, default="none", choices=["none", "oracle", "random", "infer", "neighbour"], help="Persona profiles presented to the persuader.")
    parser.add_argument("--n_turns", type=int, default=1, help="Number of turns to generate. Note that append mode may have several conversations per thread.")
    
    args = parser.parse_args()

    persona_profiles = json.load(open(args.persona_file, "r"))
    threads = json.load(open(args.conv_file, "r"))
    if isinstance(threads, dict):
        threads = threads["content"]

    if args.size != 0:
        if args.size > 0:
            threads = threads[:args.size] # keep first N
        else:
            threads = threads[-args.size:] # remove first N

    if args.generation_mode == "append":
        # APPEND MODE: Use existing conversation and replace the last message
        assert args.n_turns == 1, "n_turns must be 1 in append mode"
        
        requests = []
        for thread_idx, thread in enumerate(threads):
            persona = persona_profiles[thread["persona"]]
            question = thread["question"]
            for conv_idx, conv in enumerate(thread["convs"]):
                request = [{"role": "system", "content": gen_client_prompt(persona, question, no_persona=args.no_persona, task=args.task)}]

                for idx in range(len(conv["conv"]) - 1, -1, -1): # Filter out the last comment by the author (ask LLM to append)
                    if conv["conv"][idx].get("is_author", False):
                        conv["conv"].pop(idx)
                        break
                
                for comment in conv["conv"]:
                    request.append({"role": "assistant" if comment["is_author"] else "user", "content": comment["body"]})
                requests.append(request)
                
        # Run inference on all requests
        print(f"Running inference on {len(requests)} conversations (APPEND mode)...")
        responses = api_batch_inference(requests, model=args.client_model, n_threads=args.inference_parallel, progress=True, role='client')

    else:
        # GENERATE MODE: Generate conversations from scratch with n_turns
        assert args.no_persona == False, "Persona information must be included in generate mode"
        assert args.persuader_model is not None, "Consultant model must be specified in generate mode"
        data = []
        
        for thread_idx, thread in enumerate(threads):
            persona = persona_profiles[thread["persona"]]
            question = thread["question"]
            
            # Initialize or load existing conversation based on --cont flag
            if args.cont and "conv" in thread:
                generated_conv = thread["conv"].copy()
            else:
                generated_conv = []
            
            data.append({
                "persona": persona,
                "persona_id": thread["persona"],
                "question": question,
                "generated_conv": generated_conv,
            })
        
        dummy_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B") # only for cutoff
        response_keys = {"think", "argument"}
        start_turn = len(data[0]["generated_conv"]) // 2 if args.cont else 0
        
        if args.profile_mode == "infer":
            encoder = SentenceTransformer("BAAI/bge-m3")
            assert args.persona_infer_model, "Persona inference model must be specified for infer profile"
            # counsel and request have same templates
            persona_template_text = PERSONA_TEMPLATE if args.task == "cmv" else PERSONA_TEMPLATE_COUNSEL
            persona_template_json = persona_template if args.task == "cmv" else persona_template_counsel
        
        
        # Generate conversations turn by turn
        # Consultant speaks first (responding to the question), then client responds, repeat n_turns
        
        for turn in range(start_turn, args.n_turns):
            # === INFER PERSONA ===
            if args.profile_mode == "infer":
                requests = []
                for data_idx, item in enumerate(data):
                    beginner = REQUEST_BEGINNER if args.task == "request" else item["question"]
                    user_prompt = format_conv(beginner, item["generated_conv"])
                    requests.append([{"role": "system", "content": PREDICT_PERSONA_PROMPT.format(template=persona_template_json)},
                                     {"role": "user", "content": user_prompt}])
                
                # Write second sample requests to file
                if len(requests) > 1:
                    with open("infer_requests_sample2.txt", "w") as f:
                        for idx, msg in enumerate(requests[1]):
                            f.write(f"--- Element {idx} ---\n")
                            f.write(f"Role: {msg.get('role', 'N/A')}\n")
                            f.write(f"Content: {msg.get('content', 'N/A')}\n\n")
                        
                print(f"Running {len(requests)} requests for persona inference... (GENERATE mode, infer profile)")
                persona_responses = api_batch_inference(requests, model=args.persona_infer_model, n_threads=args.inference_parallel, progress=True, local=args.infer_model_local, role='client' if not args.infer_model_local else None)

                for data_idx, response in enumerate(persona_responses):
                    predicted_persona = process_json(response)
                    data[data_idx]["predicted_persona"] = predicted_persona

                                    
            
            # === PERSUADER TURN ===
            persuader_requests = []
            persuader_data_indices = []
            
            '''From the view of persuader, the author of forum thread is "user"; the persuader is "assistant".'''
            for data_idx, item in enumerate(data):
                if args.profile_mode == "oracle":
                    persuader_request = [{"role": "system", "content": gen_persuader_prompt(item["question"], persona=item["persona"], task=args.task)}]
                elif args.profile_mode == "random":
                    rand_persona = random.choice(list(persona_profiles.values()))
                    persuader_request = [{"role": "system", "content": gen_persuader_prompt(item["question"], persona=rand_persona, task=args.task)}]
                elif args.profile_mode == "neighbour":
                    neighbours = persona_profiles[item["persona_id"]].get("neighbours", [])
                    if neighbours:
                        neighbour_persona = persona_profiles[neighbours[0]]
                    else:
                        assert False, "neighbour not provided"
                        neighbour_persona = item["persona"]
                        
                    # print(json.dumps(neighbour_persona, indent=2, ensure_ascii=False))
                    # print('-'*50)
                    # print(json.dumps(item["persona"], indent=2, ensure_ascii=False))
                    # assert False
                    persuader_request = [{"role": "system", "content": gen_persuader_prompt(item["question"], persona=neighbour_persona, task=args.task)}]
                elif args.profile_mode == "infer":
                    persuader_request = [{"role": "system", "content": gen_persuader_prompt(item["question"], persona=item["predicted_persona"], task=args.task)}]
                else:
                    persuader_request = [{"role": "system", "content": gen_persuader_prompt(item["question"], task=args.task)}]

                persuader_request.append({"role": "user", "content": REQUEST_BEGINNER if args.task == "request" else item["question"]})
                for msg in item["generated_conv"]:
                    is_author = msg["is_author"]
                    persuader_request.append({"role": "user" if is_author else "assistant", "content": msg["body"]})
                persuader_requests.append(persuader_request)
                persuader_data_indices.append(data_idx)
            
            # Write second sample requests to file
            if len(persuader_requests) > 1:
                with open("persuader_requests_sample2.txt", "w") as f:
                    for idx, msg in enumerate(persuader_requests[1]):
                        f.write(f"--- Element {idx} ---\n")
                        f.write(f"Role: {msg.get('role', 'N/A')}\n")
                        f.write(f"Content: {msg.get('content', 'N/A')}\n\n")
            
            print(f"Running {len(persuader_requests)} requests on persuader turn {turn + 1}... (GENERATE mode)")
            persuader_responses = api_batch_inference(persuader_requests, model=args.persuader_model, n_threads=args.inference_parallel, progress=True, local=args.persuader_local, role='persuader' if not args.persuader_local else None)
            
            for data_idx, response in zip(persuader_data_indices, persuader_responses):
                processed_result = process_persuader_response(response, keys = response_keys, tokenizer=dummy_tokenizer)
                message_dict = {
                    "think": processed_result.get("think", ""),
                    "body": processed_result.get("argument", ""),
                    "is_author": False,
                }
                
                # Calculate persona similarity in infer mode
                if args.profile_mode == "infer":
                    predicted_persona = data[data_idx].get("predicted_persona", {})
                    true_persona = data[data_idx]["persona"]
                    
                    similarity = calc_persona_similarity(
                        template=persona_template_json,
                        ground_truth=true_persona,
                        model_ans=predicted_persona,
                        model=encoder
                    )
                    message_dict["persona_similarity"] = round(similarity, 5)
                    message_dict["predicted_persona"] = json.dumps(predicted_persona)

                data[data_idx]["generated_conv"].append(message_dict)
            
            # === CLIENT TURN ===
            client_requests = []
            client_data_indices = []
            
            '''From the view of client (author), the author of forum thread is "assistant"; the persuader is "user".'''
            for data_idx, item in enumerate(data):
                client_request = [{"role": "system", "content": gen_client_prompt(item["persona"], item["question"], task=args.task)}]
                # Add conversation history
                for msg in item["generated_conv"]:
                    is_author = msg["is_author"]
                    client_request.append({"role": "assistant" if is_author else "user", "content": msg["body"]})
                client_requests.append(client_request)
                client_data_indices.append(data_idx)
            
            # Write second sample requests to file
            if len(client_requests) > 1:
                with open("client_requests_sample2.txt", "w") as f:
                    for idx, msg in enumerate(client_requests[1]):
                        f.write(f"--- Element {idx} ---\n")
                        f.write(f"Role: {msg.get('role', 'N/A')}\n")
                        f.write(f"Content: {msg.get('content', 'N/A')}\n\n")
            
            print(f"Running {len(client_requests)} requests on client turn {turn + 1}... (GENERATE mode)")

            client_responses = api_batch_inference(client_requests, model=args.client_model, n_threads=args.inference_parallel, progress=True, local=False, role='client')
            
            # Add client responses to conversations
            for data_idx, response in zip(client_data_indices, client_responses):
                data[data_idx]["generated_conv"].append({
                    "body": response,
                    "is_author": True
                })
        
        responses = [item["generated_conv"] for item in data]
    
    # Reconstruct threads with simulated conversations (shared logic for both modes)
    result_threads = threads.copy()
    response_idx = 0
    for thread_idx, thread in enumerate(result_threads):
        if args.generation_mode == "generate":
            # GENERATE mode: Replace entire conversation
            if "convs" in thread:
                del thread["convs"]
            thread["conv"] = responses[response_idx]
            response_idx += 1
        else:
            # APPEND mode: Replace only the last message
            for conv_idx, conv in enumerate(thread["convs"]):
                conv["conv"].append({
                    "body": responses[response_idx],
                    "is_author": True
                })
                response_idx += 1
    
    # Save to file
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"hparams": vars(args), "content": result_threads}, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")


    