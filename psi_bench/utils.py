from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
import os
import json
from json_repair import repair_json
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from prompts import *



def fill_persona_template(persona_dict, template):
    if not isinstance(persona_dict, dict):
        persona_dict = {}
        print(f"Warning: Expected persona_dict to be a dict, but got {type(persona_dict)}.")
    p = persona_dict.get("personality", {})
    l = persona_dict.get("lifestyle", {})
    f = persona_dict.get("family", {})
    lc = persona_dict.get("language_and_culture", {})
    s = persona_dict.get("speaking_style", {})
    
    return template.format(
        age=persona_dict.get("age", "Not specified"),
        gender=persona_dict.get("gender", "Not specified"),
        place_of_birth=persona_dict.get("place_of_birth", "Not specified"),
        education=persona_dict.get("education", "Not specified"),
        occupation=persona_dict.get("occupation", "Not specified"),
        hobbies_and_interests=persona_dict.get("hobbies_and_interests", "Not specified"),
        
        marital_status=f.get("marital_status", "Not specified"),
        children=f.get("children", "Not specified"),
        parents=f.get("parents", "Not specified"),
        
        housing=l.get("housing", "Not specified"),
        transportation=l.get("transportation", "Not specified"),
        
        traits=p.get("traits", "Not specified"),
        political_views=p.get("political_views", "Not specified"),
        religion=p.get("religion", "Not specified"),
        
        first_language=lc.get("first_language", "Not specified"),
        accent=lc.get("accent", "Not specified"),
        cultural_identity=lc.get("cultural_identity", "Not specified"),
        
        tone=s.get("tone", "Not specified"),
        formality=s.get("formality", "Not specified"),
        pacing=s.get("pacing", "Not specified"),
        vocabulary=s.get("vocabulary", "Not specified"),
        interaction_pattern=s.get("interaction_pattern", "Not specified"),
        clarity=s.get("clarity", "Not specified"),
        
        short_persona=persona_dict.get("short_persona", "Unknown"),
    )

def gen_client_prompt(persona_dict: dict, question: str, task = "cmv", no_persona = False) -> str:
    if task == "cmv":
        persona_dict["personality"]["traits"] += " You often hold your stand firmly and do not easily accept other people's viewpoints."
        prompt_base = CLIENT_PROMPT_TEMPLATE_NO_PERSONA if no_persona else CLIENT_PROMPT_TEMPLATE
        prompt = fill_persona_template(persona_dict, prompt_base)
        add_on = CMV_ADDON.format(question=question)
    elif task == "counsel":
        persona_dict["personality"]["traits"] += " You are deeply mired in your psychological issue and you're resisting change out of fear of sustaining further harm. Your stubbornness, obsession, and conflicted mindset prevent you from accepting advice from others."
        prompt_base = CLIENT_PROMPT_TEMPLATE_COUNSEL
        prompt = fill_persona_template(persona_dict, prompt_base)
        add_on = COUNSEL_ADDON.format(question=question)
    elif task == "request":
        persona_dict["personality"]["traits"]  += " You live a very busy life and prefer to go your own way. Therefore, you prefer not to interfere in other people's lives."
        prompt_base = CLIENT_PROMPT_TEMPLATE_REQUEST
        prompt = fill_persona_template(persona_dict, prompt_base)
        add_on = REQUEST_ADDON
    else:
        raise ValueError(f"Unsupported task: {task}")
        
    prompt += '\n' + add_on
    return prompt

def gen_persuader_prompt(question: str, persona = None, guess = False, task = "cmv") -> str:
    if task == "cmv":
        if persona:
            prompt = PERSUADER_BASE.format(question=question) + PERSUADER_USER_INFO + fill_persona_template(persona, PERSONA_TEMPLATE) + PERSUADER_GUIDELINES
        else:
            prompt = PERSUADER_BASE.format(question=question) + PERSUADER_GUIDELINES
    elif task == "counsel":
        if persona:
            prompt = PERSUADER_BASE_COUNSEL.format(question=question) + PERSUADER_USER_INFO + fill_persona_template(persona, PERSONA_TEMPLATE_COUNSEL) + PERSUADER_GUIDELINES
        else:
            prompt = PERSUADER_BASE_COUNSEL.format(question=question) + PERSUADER_GUIDELINES
    elif task == "request":
        if persona:
            prompt = PERSUADER_BASE_REQUEST.format(question=question) + PERSUADER_USER_INFO + fill_persona_template(persona, PERSONA_TEMPLATE_REQUEST) + PERSUADER_GUIDELINES
        else:
            prompt = PERSUADER_BASE_REQUEST.format(question=question) + PERSUADER_GUIDELINES
    else:
        raise ValueError(f"Unsupported task: {task}")
        
    return prompt

def format_conv(beginner, conv):
    prompt = "User A: " + beginner
    for turn in conv:
        role = "User A" if turn["is_author"] else "User B"
        prompt += f"\n\n{role}: {turn['body'].strip()}"
    return prompt

import math
import copy
def format_judge_prompt(persona_dict, question, conv, task = "cmv") -> str:
    prompt = "## Persona profile of User A\n"
    if task == "cmv":
        prompt += fill_persona_template(persona_dict, PERSONA_TEMPLATE)
    elif task == "counsel":
        prompt += fill_persona_template(persona_dict, PERSONA_TEMPLATE_COUNSEL)
    elif task == "request":
        prompt += fill_persona_template(persona_dict, PERSONA_TEMPLATE_REQUEST)
    else:
        raise ValueError(f"Unsupported task: {task}")

    beginner = REQUEST_BEGINNER if task == "request" else question
    prompt += '\n## Conversation\n'
    prompt += format_conv(beginner, conv)
    
    return prompt


def _extract_relevant_fields(template: dict) -> set:
    """Extract all level-1 and level-2 field names from template."""
    fields = set()
    for key, value in template.items():
        # Level-1 fields
        fields.add(key)
        # Level-2 fields
        if isinstance(value, dict):
            for subkey in value.keys():
                fields.add(f"{key}.{subkey}")
    return fields


def _flatten_dict(d: dict, parent_key: str = "") -> dict:
    """Flatten a nested dictionary using dot notation."""
    if not isinstance(d, dict):
        return {}
    
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def calc_persona_similarity(
    template: dict,
    ground_truth: dict,
    model_ans: dict,
    model: SentenceTransformer,
) -> float:
    """
    Calculate similarity between two personas using field-level embeddings.
    
    Args:
        template: Dictionary defining the persona structure (level-1 and level-2 fields)
        ground_truth: Ground truth persona dictionary
        model_ans: Model output persona dictionary
        model: SentenceTransformer model for encoding
    
    Returns:
        float: Average cosine similarity across all relevant fields (0-1)
    """
    # Extract relevant fields from template
    relevant_fields = _extract_relevant_fields(template)
    
    # Flatten both personas
    flat_ground_truth = _flatten_dict(ground_truth)
    flat_model_ans = _flatten_dict(model_ans)
    
    # Filter to only relevant fields
    flat_ground_truth = {k: v for k, v in flat_ground_truth.items() if k in relevant_fields}
    flat_model_ans = {k: v for k, v in flat_model_ans.items() if k in relevant_fields}
    
    # Prepare texts to encode - only include fields that exist in both personas
    texts1 = []
    texts2 = []
    field_names = []
    missing_fields = []
    
    for field_name in sorted(relevant_fields):
        if field_name in flat_ground_truth and field_name in flat_model_ans:
            value1 = flat_ground_truth[field_name]
            value2 = flat_model_ans[field_name]
            texts1.append(str(value1))
            texts2.append(str(value2))
            field_names.append(field_name)
        else:
            # Field doesn't exist in both personas, mark as missing
            missing_fields.append(field_name)
    
    # Encode all texts at once
    all_texts = texts1 + texts2
    if not all_texts:
        return 0.0  # No valid fields to compare, return 0 score
    
    embeddings = model.encode(
        all_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    
    # Calculate similarity for each field
    num_fields = len(texts1)
    embeddings1 = embeddings[:num_fields]
    embeddings2 = embeddings[num_fields:]
    
    similarities = []
    for emb1, emb2 in zip(embeddings1, embeddings2):
        # Cosine similarity (dot product since embeddings are normalized)
        sim = np.dot(emb1, emb2)
        similarities.append(max(0, sim))  # Ensure non-negative
    
    # Add 0 similarity for missing fields
    for _ in missing_fields:
        similarities.append(0)
    
    return float(np.mean(similarities)) if similarities else 0.0



def process_json(text: str):
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text, re.DOTALL)
    if not match:
        return {}
    json_str = match.group()
    json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

    json_str_fixed = repair_json(json_str)
    try:
        return json.loads(json_str_fixed)
    except Exception as e:
        with open("json_parsing_error.txt", "a") as f:
            f.write(f"Failed to parse JSON:\n{text}\n\n{json_str_fixed}\n\n\n")
        print(f"Failed: {e}")
        return {}



import time
import random
import atexit
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer



# Global cache for vllm instance to avoid reloading the same model
_vllm_cache = {
    "model": None,
    "vllm_instance": None,
    "tokenizer": None,
}

def cleanup_vllm_cache():
    """Manually cleanup vllm cache. Call this when you're done with vllm inference."""
    global _vllm_cache
    if _vllm_cache["vllm_instance"] is not None:
        print(f"Cleaning up vllm cache for model: {_vllm_cache['model']}")
        _vllm_cache["vllm_instance"] = None
        _vllm_cache["tokenizer"] = None
        _vllm_cache["model"] = None

atexit.register(cleanup_vllm_cache) # Register cleanup function to run at program exit

def api_batch_inference(requests, sampling_params = {"temperature": 1.0, "max_tokens": 8192}, model="gpt-4o", n_threads=8, progress=False, local=False, role=None):
    if local:
        global _vllm_cache

        if _vllm_cache["model"] != model:
            print(f"Loading model: {model}")
            n_gpus = torch.cuda.device_count()

            if _vllm_cache["vllm_instance"] is not None:
                print(f"Releasing cached model: {_vllm_cache['model']}")
                cleanup_vllm_cache()

            _vllm_cache["vllm_instance"] = LLM(
                model=model,
                tokenizer=model,
                dtype='bfloat16',
                tensor_parallel_size=n_gpus,
                disable_custom_all_reduce=True,
                enforce_eager=True,
                gpu_memory_utilization=0.65,
                max_model_len=8192
            )
            _vllm_cache["tokenizer"] = AutoTokenizer.from_pretrained(model)
            _vllm_cache["model"] = model
        else:
            print(f"Using cached model: {model}")

        vllm_instance = _vllm_cache["vllm_instance"]
        tokenizer = _vllm_cache["tokenizer"]
        sampling_params = SamplingParams(**sampling_params)
        requests = [tokenizer.apply_chat_template(request, tokenize=False, add_generation_prompt=True) for request in requests]
        vllm_outputs = vllm_instance.generate(requests, sampling_params)
        vllm_outputs = [decoded_answer.outputs[0].text.strip() for decoded_answer in vllm_outputs]
        return vllm_outputs

    else:
        if role is None:
            raise ValueError("role parameter is required when local=False")

        if role not in ["client", "judge", "persuader"]:
            raise ValueError(f"Invalid role: {role}. Must be one of 'client', 'judge', 'persuader'")

        base_url = os.getenv(f"{role.upper()}_BASE_URL")
        api_key = os.getenv(f"{role.upper()}_API_KEY")

        if not base_url or not api_key:
            raise ValueError(f"Missing environment variables {role.upper()}_BASE_URL or {role.upper()}_API_KEY")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=5,
            timeout=120
        )

        class EmptyResult:
            def __init__(self):
                self.choices = [{"message": {"content": "This is an empty response due to an API error."}}]

        def get_completion(request):
            assert isinstance(request, list) and all(isinstance(turn, dict) for turn in request), \
                "Request format error: expected list of dicts"
            time.sleep(random.uniform(2, 10))
            try:
                result = client.chat.completions.create(
                    model=model,
                    messages=request,
                    **sampling_params
                )
                return result
            except Exception as e:
                print(f"API error: {e}")
                return EmptyResult()

        with ThreadPoolExecutor(max_workers=min(len(requests), n_threads)) as executor:
            if progress:
                results = list(tqdm(
                    executor.map(get_completion, requests),
                    total=len(requests),
                    desc=f"Inference (Parallel, Model: {model}, Role: {role})"
                ))
            else:
                results = list(executor.map(get_completion, requests))

        processed_results = []
        for r in results:
            try:
                content = r.choices[0].message.content
                if not content:
                    processed_results.append("This is an empty response due to an API error.")
                else:
                    processed_results.append(content)
            except:
                processed_results.append("This is an empty response due to an API error.")

        return processed_results
    
    
def extract_answer(original_str, key_word, n_answers=1, strict=True):
    answer_pattern = fr'<{key_word}>(.*?)</{key_word}>'
    matches = [match.group(1).strip() for match in re.finditer(answer_pattern, original_str, re.DOTALL) if match.group(1).strip() != ""]
    if matches == []:
        answer_pattern = fr'<{key_word}>(.*?)$'
        matches = [match.group(1).strip() for match in re.finditer(answer_pattern, original_str, re.DOTALL) if match.group(1).strip() != ""]
    
    if matches == []:
        if strict:
            list_to_return =  [""]
        else:
            list_to_return = [original_str]
    else:
        list_to_return = matches[-n_answers:]

    if n_answers == 1:
        return list_to_return[0]
    else:
        return list_to_return
    
