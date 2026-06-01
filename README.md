<div style="width: 100%;">
  <img src="figs/title.png" style="width: 100%;" alt="sotopia"></img>
</div>

<h1 align="center">Psi-Bench: Evaluating Persona-Sensitive Influencing in Persuasive Dialogues</h1>


[![PyPI](https://img.shields.io/pypi/v/your-package?logo=pypi&logoColor=white)](https://pypi.org/project/your-package/)·
[![arXiv](https://img.shields.io/badge/arXiv-2401.12345-b31b1b?logo=arxiv)](https://arxiv.org/abs/2401.12345)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/your-org/your-dataset)


## Introduction



## Get Started

Before using Psi-Bench, you need to configure your API through environment variables. Psi-Bench uses DeepSeek-v3.2 to serve as the client and judge. DeepSeek official API no longer supports this model, so it's recommended to access it via [NVIDIA](build.nvidia.com) or [vocanic platform](https://console.volcengine.com/ark) (identifier is deepseek-v3-2-251201 in Vocanic).

For example:
```bash
export CLIENT_BASE_URL=https://api.deepseek.com/v1
export CLIENT_API_KEY=sk-...
export JUDGE_BASE_URL=https://api.deepseek.com/v1
export JUDGE_API_KEY=sk-...
export PERSUADER_BASE_URL=... # If you want to evaluate API-based persuader models
export PERSUADER_API_KEY=sk-...
```

### Download the Package
This is the easiest way of using Psi-Bench.
```bash
pip install psi-bench
psi-bench download all # data will be saved in ./data

# Run evaluation with local persuader model
CUDA_VISIBLE_DEVICES=0 psi-bench eval all \
  --tested_model Qwen/Qwen3-8B 
  --persuader_local \
  --client_model deepseek-v3-2 \
  --judge_model deepseek-v3-2 \
```


### Clone the Repository
If you wish to develop based on Psi-Bench, or evaluate on more settings (oracle, profile analyzer, ...), you can clone the Git repo. Below are some examples:

```bash
git clone https://github.com/Hanpx20/Psi-Bench
cd Psi-Bench

# Basic evaluation with local persuader
CUDA_VISIBLE_DEVICES=0 bash eval.sh all \
  --tested_model Qwen/Qwen3-8B \
  --client_model deepseek-v3-2 \
  --judge_model deepseek-v3-2 \
  --persuader_local

# Inference with oracle setting (client profile provided)
CUDA_VISIBLE_DEVICES=0 bash eval.sh all \
  --tested_model Qwen/Qwen3-8B \
  --client_model deepseek-v3-2 \
  --judge_model deepseek-v3-2 \
  --persuader_local \
  --test_oracle

# Inference with profile analyzer (client profile predocted by a LLM)
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
  --client_model deepseek-v3-2 \
  --task request \
  --conv_file data/request/queries.json \
  --persona_file data/request/persona_profile.json \
  --persuader_model Qwen/Qwen3-8B \
  --persuader_local \
  --profile_mode infer \
  --persona_infer_model deepseek-v3.2 \
  --output eval/test.json
```



## Cite this paper
If you find this repo or the paper useful, please cite:
```
xxxxxx
```

Reach out to [Peixuan Han](mailto:ph16@illinois.edu) for any questions.
