import os
import re



os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix tokenizer warning
# Fix vLLM multiprocessing issues with CUDA
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

import json
import argparse
import random
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import AutoConfig


def get_max_seq_length(model_dir, max_seq_length=None, max_new_tokens=None) -> (int, int):
    """Get max sequence length from model config or tokenizer."""

    if max_seq_length is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            if "deepseek" in model_dir.lower():
                max_seq_length = 32768
            elif hasattr(tokenizer, "model_max_length"):
                max_seq_length =  tokenizer.model_max_length
            # Fallback to config if available
            elif hasattr(AutoConfig.from_pretrained(model_dir), "max_position_embeddings"):
                max_seq_length = AutoConfig.from_pretrained(model_dir).max_position_embeddings
            else:
                raise ValueError("Could not determine max sequence length from tokenizer or model config")
        except Exception as e:
            print(f"Could not determine max sequence length from model config: {e}")
    max_seq_length = min(max_seq_length, 200000)
    if max_new_tokens is None:
        # if "deepseek" in model_dir.lower():
        #     max_new_tokens = 4000
        # else:
        max_new_tokens = max_seq_length
    # Default fallback
    return max_seq_length, max_new_tokens

def format_prompt_with_tokenizer(prompt: str, tokenizer, chat_template_kwargs: dict | None = None):
    # Use tokenizer's chat template if available and defined
    if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None):
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(chat_template_kwargs or {})
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    else:
        raise ValueError(f"No chat template found for tokenizer: {tokenizer}")
        # # Manual templates by model family when chat template is missing
        # name = (getattr(tokenizer, 'name_or_path', '') or '').lower()
        # if 'llama' in name or 'meta-llama' in name:
        #     # Llama 3.x header tokens
        #     formatted_text = (
        #         f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}"
        #         f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        #     )
        # else:
        #     # Generic instruction style
        #     formatted_text = f"User:\n{prompt}\nAssistant:"
    return formatted_text


def merge_lora_weights(base_model_path, lora_path, output_path):
    """Merge LoRA weights with base model and save the merged model."""

    
    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Load on CPU for merging
    )
    
    print(f"Loading LoRA weights from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    print("Converting merged model to bfloat16 before saving...")
    merged_model = merged_model.to(torch.bfloat16)
    
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    
    # Also save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("Model merging completed!")
    return output_path


def load_vllm_model(model_path, max_seq_length=8096,
                    tensor_parallel_size=None,
                    distributed_executor_backend=None,
                    seed=None):
    """Load model using vLLM for optimized inference."""

    print(f"Loading model with vLLM...")

    p_size = tensor_parallel_size if tensor_parallel_size else torch.cuda.device_count()

    args = {
        "model": model_path,
        "max_model_len": max_seq_length,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9,  # Conservative memory usage
        "dtype": "bfloat16" if torch.cuda.is_available() else "float16",
        "tensor_parallel_size": p_size,
    }
    if distributed_executor_backend:
        args["distributed_executor_backend"] = distributed_executor_backend
    if 'DeepSeek-R1' in model_path and 'Qwen3' in model_path:
        # Newer R1 distills on Qwen3 base — use qwen3 parser (tokenizer has <think> tokens)
        args['reasoning_parser'] = 'qwen3'
    elif 'DeepSeek-R1' in model_path and 'qwen' in model_path.lower():
        args['enforce_eager'] = True  # Disable CUDA graphs which can cause issues
        args['data_parallel_size'] = 1  #  DeepSeek models can be unstable with DP > 1
        args['tensor_parallel_size'] = 1  #  DeepSeek models can be unstable with DP > 1
        args['reasoning_parser'] = 'deepseek_r1'
    elif 'DeepSeek-R1' in model_path and 'llama' not in model_path.lower() and 'qwen' not in model_path.lower():
        # Full DeepSeek-R1 / R1-0528 — uses native <think> tokens
        args['reasoning_parser'] = 'deepseek_r1'
        # del args['dtype']  # Let vLLM decide TP size automatically        
    elif 'llama-4' in model_path.lower():
        args['enforce_eager'] = True
        args['override_generation_config'] = {
            "attn_temperature_tuning": True,
        }
    elif 'qwen3-vl' in model_path.lower() or 'qwen3-next' in model_path.lower() or 'qwen3-omni' in model_path.lower() or '2507' in model_path.lower() or '235b' in model_path.lower():
        args['reasoning_parser'] = 'qwen3'
    elif 'qwen3' in model_path.lower() and any(x in model_path.lower() for x in ['122b', '397b']):
        # Large Qwen3.5 MoE: use full TP across all available GPUs (passed via --parallel-size)
        args['reasoning_parser'] = 'qwen3'
    elif 'qwen3' in model_path.lower():
        # args['enforce_eager'] = True
        args['data_parallel_size'] = 1  #  DeepSeek models can be unstable with DP > 1
        args['tensor_parallel_size'] = 4  #  DeepSeek models can be unstable with DP > 1
        args['reasoning_parser'] = 'qwen3'
    elif 'Kimi-K2-Thinking' in model_path:
        # No reasoning_parser: Kimi tokenizer lacks special <think> tokens as vocab entries;
        # strip <think> blocks in post-processing instead
        pass
    elif 'kimi-k2.5' in model_path.lower() or 'kimi_k2.5' in model_path.lower() or 'kimi-k25' in model_path.lower():
        args['enforce_eager'] = True  # Large model, save memory for KV cache
        args['gpu_memory_utilization'] = 0.95  # 595 GB model, needs more than default 90%
        # No reasoning_parser: rely on <think> stripping in post-processing
    elif 'kimi-vl' in model_path.lower() or 'kimi_vl' in model_path.lower():
        args['enforce_eager'] = True  # CUDA graph capture crashes with TritonMLA backend; no reasoning_parser (tokenizer lacks special think tokens)
    elif 'glm-5' in model_path.lower() or 'glm5' in model_path.lower():
        args['reasoning_parser'] = 'glm45'
        # Note: disable_custom_all_reduce NOT set here — multi-node Ray needs vLLM's
        # custom all-reduce; gloo fallback breaks with GLOO_SOCKET_IFNAME=ib0 on container nets
    elif 'glm-4.7' in model_path.lower() or 'glm4.7' in model_path.lower():
        args['reasoning_parser'] = 'glm45'
        args['disable_custom_all_reduce'] = True
    elif 'glm-4.5' in model_path.lower() or 'glm4.5' in model_path.lower() or 'glm4_5' in model_path.lower():
        args['reasoning_parser'] = 'glm45'
        args['disable_custom_all_reduce'] = True  # Avoid CUDA illegal memory access during distributed init
    elif 'minimax-m2.5' in model_path.lower() or 'minimax_m2.5' in model_path.lower() or 'minimax-m25' in model_path.lower() or 'minimax_m25' in model_path.lower() or 'MiniMax-M2.5' in model_path:
        args['reasoning_parser'] = 'minimax_m2'
        # FP8 block-quant bug: MoE expert gate+up output_size=96 is not divisible by
        # block_n=128. Exclude FusedMoE layers from block quantization to avoid the error.
        args['override_quantization_config'] = {
            "modules_to_not_convert": ["gate", "e_score_correction_bias", "lm_head", "block_sparse_moe"]
        }
    elif 'minimax-m2' in model_path.lower() or 'minimax_m2' in model_path.lower() or 'MiniMax-M2' in model_path:
        args['reasoning_parser'] = 'minimax_m2'
        # FP8 block-quant bug: MoE expert gate+up output_size=96 is not divisible by
        # block_n=128. Exclude FusedMoE layers from block quantization to avoid the error.
        args['override_quantization_config'] = {
            "modules_to_not_convert": ["gate", "e_score_correction_bias", "lm_head", "block_sparse_moe"]
        }
    elif 'nemotron' in model_path.lower():
        pass  # No reasoning_parser: tokenizer lacks special think tokens; strip <think> blocks in post-processing
    elif 'gemma-4' in model_path.lower():
        args['reasoning_parser'] = 'gemma4'
  
          
    if seed is not None:
        args["seed"] = seed
    print(f"Model path: {model_path} with args: {args}")
    # Initialize vLLM model with spawn-safe settings
    llm = LLM(**args)
    print("vLLM model loaded successfully!")
    return llm

def get_sampling_params_for_model(model_name, tokenizer, llm, reasoning_effort=None, **kargs):
    """Get optimized sampling parameters based on model type."""
    # Determine stop tokens based on model type
    if "deepseek" in model_name.lower():
        model_config = {
            "temperature": 0.6,
            "top_p": 0.95,
        }
    elif "qwen3" in model_name.lower():
        model_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            # "n": 8,
            # "repetition_penalty": 1,
            "presence_penalty": 1,
            }
    elif 'kimi' in model_name.lower():
        model_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            'min_p': 0,
            'presence_penalty': 1
            }
    elif 'glm-4.5' in model_name.lower() or 'glm4.5' in model_name.lower() or 'glm4_5' in model_name.lower():
        model_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1,
        }
    elif 'minimax' in model_name.lower():
        model_config = {
            "temperature": 0.6,
            "top_p": 0.95,
        }
    elif 'nemotron' in model_name.lower():
        model_config = {
            "temperature": 0.6,
            "top_p": 0.95,
        }
    elif 'gemma-4' in model_name.lower():
        model_config = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        }
    elif 'glm-5' in model_name.lower() or 'glm5' in model_name.lower():
        model_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1,
        }
    elif 'glm-4.7' in model_name.lower() or 'glm4.7' in model_name.lower():
        model_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1,
        }
    elif 'olmo' in model_name.lower():
        model_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            }
    else:
        model_config = {
            "temperature": 0.6,
            "top_p": 0.9
            }
    model_config.update(kargs)
    stops = ["<｜end▁of▁sentence｜>", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]
    # if tokenizer hast end of text token, add it to stops
    if tokenizer.eos_token:
        stops.append(tokenizer.eos_token)
    # get stop tokens from model config if available
    if hasattr(llm.llm_engine.model_config, 'stop') and llm.llm_engine.model_config.stop:
        stops = llm.llm_engine.model_config.stop

    
    return SamplingParams(
        **model_config,
        # n=8,  # TO later compute pass@X we need 8 samples per prompt
        stop=stops,
    )


def get_chat_template_kwargs_for_model(model_name: str, enable_thinking: bool, reasoning_effort: str | None):
    """Return chat_template kwargs supported by model templates."""
    model_id = (model_name or "").lower()
    kwargs = {}

    # GPT-OSS effort control is exposed by the tokenizer chat template.
    if "gpt-oss" in model_id and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    # Keep existing thinking toggle behavior for hybrid thinking families.
    is_hybrid_thinking = (
        "qwen3" in model_id
        or "glm-4.5" in model_id
        or "glm4.5" in model_id
        or "glm4_5" in model_id
        or "glm-4.7" in model_id
        or "glm4.7" in model_id
        or "glm-5" in model_id
        or "glm5" in model_id
        or "gemma-4" in model_id
    )
    if is_hybrid_thinking:
        kwargs["enable_thinking"] = bool(enable_thinking)

    return kwargs

DEFAULT_SEED = 42


def evaluate_model_vllm(llm, tokenizer, test_dataset, max_new_tokens=512, enable_thinking=False, cot_phrase: str | None = None, reasoning_effort: str | None = None):
    """Evaluate the model using vLLM for fast inference."""
    
    model_name = llm.llm_engine.model_config.model
    print(f"Evaluating on test set with model: {model_name}")
    print(f"Test set size: {len(test_dataset)}")
    chat_template_kwargs = get_chat_template_kwargs_for_model(
        model_name,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    print(f"Using chat_template_kwargs: {chat_template_kwargs or '{}'}")
    
    # Prepare prompts for generation
    formatted_prompts = []
    problem_ids = []
    ground_truths = []
    validation_programs = []
    
    # Build list of raw prompts (optionally with CoT phrase) and formatted prompts
    input_prompts = []
    for example in tqdm(test_dataset, desc="Preparing prompts"):
        if 'Qwen3-VL' in model_name and 'Instruct' in model_name:
            raw_prompt = example["prompt"]
        else:
            raw_prompt = example["prompt"]
        if cot_phrase:
            raw_prompt = raw_prompt.rstrip() + "\n\n" + cot_phrase
        # raw_prompt += "\n\n Please enclose the final Prolog rule between [RULE] and [/RULE] tags."

        input_prompts.append(raw_prompt)
        formatted_prompts.append(format_prompt_with_tokenizer(
            raw_prompt, tokenizer, chat_template_kwargs=chat_template_kwargs
        ))
        problem_ids.append(example["id"])
        ground_truths.append(example["ground-truth rule"])
        validation_programs.append(example["validation program"])
    
    # Set up optimized sampling parameters for better logical reasoning
    # Determine stop tokens based on model type
    sampling_params = get_sampling_params_for_model(
        model_name,
        tokenizer,
        llm,
        reasoning_effort=reasoning_effort,
        max_tokens=max_new_tokens
    )
    
    # Alternative: Greedy decoding for deterministic, focused outputs
    greedy_sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding - always pick most likely token
        max_tokens=max_new_tokens,
        top_p=1.0,  # No nucleus sampling in greedy mode
        top_k=-1,   # No top-k filtering in greedy mode
        repetition_penalty=1.0,  # No repetition penalty for deterministic output
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    model_id = llm.llm_engine.model_config.model.lower()
    has_chat_tmpl = bool(getattr(tokenizer, 'chat_template', None))

    kargs = {'sampling_params': sampling_params}
    if chat_template_kwargs:
        kargs["chat_template_kwargs"] = chat_template_kwargs


    print(f"Using sampling: {sampling_params}")
    print(f"Generating predictions with vLLM using {'greedy decoding' if sampling_params.temperature == 0.0 else 'sampling'} strategy...")
    if has_chat_tmpl and not 'deepseek' in model_id and 'olmo' not in model_id.lower():
        messages = [[{"role": "user", "content": p}] for p in input_prompts]
        outputs = llm.chat(messages, **kargs)
    elif 'olmo' in model_id.lower():
        outputs = llm.generate(input_prompts, sampling_params)
    else:
        print("Using standard generation...")
        outputs = llm.generate(formatted_prompts, sampling_params)
    # Process outputs
    model_outputs = []
    exceeded_count = 0

    for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
        for o_id, o in enumerate(output.outputs):
            example = test_dataset[i]
            generated_text = o.text.strip()
            # Strip thinking blocks that may remain when no reasoning_parser is used
            generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()

            # Calculate token counts on the actual formatted prompt
            prompt_tokens = len(tokenizer.encode(formatted_prompts[i]))
            completion_tokens = len(tokenizer.encode(generated_text))
            
            if completion_tokens > max_new_tokens - 20:
                exceeded_count += 1

            model_outputs.append({
                "problem_id": problem_ids[i],
                "pass": o_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model_completion": generated_text,
                "ground_truth": ground_truths[i],
                "reference": {
                    "validation_program": validation_programs[i],
                    "evaluation_config": {
                        "positive_predicate": "eastbound",
                        "negative_predicate": "westbound"
                    }
                }
        })
    print(f"Number of outputs exceeding max_new_tokens ({max_new_tokens}): {exceeded_count}/{len(test_dataset)}")
    return model_outputs

def main():
    """Main evaluation function using vLLM."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate a model on SLR-Bench using vLLM")
    # New unified interface
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model identifier: HF repo id or local path. If this is a LoRA adapter folder, also provide --base-model.")
    parser.add_argument("--base-model", default=None,
                       help="Base HF model id required when --model points to a LoRA adapter directory.")
    # Backward-compat flags (deprecated)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                       help="Maximum new tokens to generate")
    parser.add_argument("--max-seq-length", type=int, default=None,
                       help="Maximum sequence length")
    parser.add_argument("--parallel-size", type=int, default=None,
                       help="Number of parallel processes for vLLM")
    parser.add_argument("--test-subset", type=int, default=None,
                       help="Evaluate on a subset of test examples (for quick testing)")
    parser.add_argument("--enable-thinking", action="store_true",
                    help="Enable thinking for the models that support it (e.g., Qwen3)")
    parser.add_argument("--cot", action="store_true",
                    help="Append a Chain-of-Thought trigger phrase to the prompt.")
    parser.add_argument("--cot-phrase", default="Let's think step by step.",
                    help="Custom CoT trigger phrase appended when --cot is set.")
    parser.add_argument("--out-path", default=None,
                        help="Output path for saving results (overrides default path).")
    parser.add_argument("--distributed-backend", default=None, choices=['ray', 'mp'],
                        help="vLLM distributed executor backend for multi-node inference.")
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
                        help="Reasoning effort for supported models (e.g., GPT-OSS).")
    parser.add_argument("--rerun-truncated", action="store_true",
                        help="Re-run only samples that hit the token limit in an existing run, then merge results.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides default 42). Appended to output tag to avoid collisions.")
    args = parser.parse_args()

    # Resolve model path/id from unified and legacy args
    model_arg = args.model
    if not model_arg:
        raise ValueError("Please provide --model (HF id or local path). For LoRA adapters, also provide --base-model.")

    # Determine output directory tag
    if os.path.exists(model_arg):
        tag = os.path.basename(model_arg.rstrip('/'))
    else:
        tag = model_arg.split('/')[-1]
    if args.enable_thinking:
        tag += "-Thinking"
    if args.cot:
        tag += "-CoT"
    if args.reasoning_effort:
        tag += f"-effort-{args.reasoning_effort}"
    if args.seed is not None:
        tag += f"-seed{args.seed}"
    if args.out_path:
        out_dir = os.path.join(args.out_path, tag)
    else:
        out_dir = os.path.join('output', 'evalv2', tag)

    # if output results already exist, skip evaluation (unless --rerun-truncated)
    outputs_file_path = os.path.join(out_dir, "model_outputs.json")
    existing_outputs = None
    truncated_ids = None
    if os.path.exists(outputs_file_path):
        if args.rerun_truncated:
            with open(outputs_file_path) as f:
                existing_outputs = json.load(f)
            # Determine old token limit from meta.json
            meta_path = os.path.join(out_dir, "meta.json")
            old_max = None
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    old_meta = json.load(f)
                old_max = old_meta.get("max_new_tokens")
            if old_max is None:
                old_max = max(r["completion_tokens"] for r in existing_outputs)
            threshold = old_max - 10
            truncated_ids = {r["problem_id"] for r in existing_outputs if r["completion_tokens"] >= threshold}
            print(f"--rerun-truncated: found {len(truncated_ids)} truncated samples (old limit={old_max}). Re-running only those.")
        else:
            print(f"Output results already exist at {out_dir}. Skipping evaluation. To re-evaluate, please remove the existing directory.")
            return

    # Handle LoRA merge if the provided model is an adapter directory
    model_path = model_arg
    if os.path.isdir(model_arg) and os.path.exists(os.path.join(model_arg, 'adapter_model.safetensors')):
        base_model = args.base_model or args.model_name  # backward compat
        if not base_model:
            raise ValueError("--base-model is required when --model points to a LoRA adapter directory")
        merged_model_dir = os.path.join(model_arg, "merged")
        if not os.path.exists(merged_model_dir):
            print(f"Found LoRA adapter at {model_arg}. Merging with base model {base_model}...")
            merge_lora_weights(base_model, model_arg, merged_model_dir)
        else:
            print(f"Using existing merged model from {merged_model_dir}")
        model_path = merged_model_dir
    
    # Gemma 4 tokenizer_config.json stores extra_special_tokens as a list,
    # but transformers expects a dict and calls .keys() on it — patch the cached config.
    if 'gemma-4' in model_path.lower():
        import glob as _glob
        hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        hf_model_id = model_path.replace('/', '--')
        for cfg_path in _glob.glob(f"{hf_home}/hub/models--{hf_model_id}/**/tokenizer_config.json", recursive=True):
            with open(cfg_path) as _f:
                _cfg = json.load(_f)
            if isinstance(_cfg.get('extra_special_tokens'), list):
                _cfg['extra_special_tokens'] = {}  # video/multimodal tokens not needed for text eval
                with open(cfg_path, 'w') as _f:
                    json.dump(_cfg, _f, indent=2)
                print(f"Patched extra_special_tokens list→dict in {cfg_path}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load test dataset
    print(f"Loading SLR-Bench test dataset...")
    test_set = load_dataset("AIML-TUDA/SLR-Bench", "v1-All", split="test")
    
    if args.test_subset:
        print(f"Using subset of {args.test_subset} examples")
        test_set = test_set.select(range(min(args.test_subset, len(test_set))))

    if truncated_ids is not None:
        test_set = test_set.filter(lambda x: x["id"] in truncated_ids)
        print(f"Filtered dataset to {len(test_set)} truncated samples.")
    
    max_seq_length, max_new_tokens = get_max_seq_length(model_path, args.max_seq_length, args.max_new_tokens)
    # update tokenizer max length if specified
    if args.max_seq_length is not None:
        tokenizer.model_max_length = max_seq_length

    # Set global seeds for reproducibility
    _seed = args.seed if args.seed is not None else DEFAULT_SEED
    try:
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        os.environ["PYTHONHASHSEED"] = str(_seed)
    except Exception:
        pass

    # CoT phrase selection
    cot_phrase = args.cot_phrase if args.cot else None
    if cot_phrase:
        print(f"Using CoT phrase: {cot_phrase}")

    # Load vLLM model
    llm = load_vllm_model(
        model_path,
        max_seq_length=max_seq_length,
        tensor_parallel_size=args.parallel_size,
        distributed_executor_backend=args.distributed_backend,
        seed=args.seed,
    )
    # Evaluate the model
    print('--' * 20)
    print(f"Starting evaluating {model_path} (max_new_tokens={max_new_tokens}, max_seq_length={max_seq_length})")
    model_outputs = evaluate_model_vllm(
        llm, tokenizer, test_set,
        max_new_tokens=max_new_tokens,
        enable_thinking=args.enable_thinking,
        cot_phrase=cot_phrase,
        reasoning_effort=args.reasoning_effort,
    )
    
    # add a single top-level metadata file to avoid per-item duplication
    meta = {
        "model": model_arg,
        "model_path": model_path,
        "tag": tag,
        "max_seq_length": max_seq_length,
        "max_new_tokens": max_new_tokens,
        "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
    }

    # Merge with existing outputs if doing a partial rerun
    if existing_outputs is not None and truncated_ids is not None:
        new_by_id = {r["problem_id"]: r for r in model_outputs}
        merged = [new_by_id.get(r["problem_id"], r) for r in existing_outputs]
        model_outputs = merged
        print(f"Merged {len(new_by_id)} re-run samples into {len(model_outputs)} total results.")

    # Save model outputs
    os.makedirs(out_dir, exist_ok=True)
    outputs_file = f"{out_dir}/model_outputs.json"
    with open(outputs_file, "w") as f:
        json.dump(model_outputs, f, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model outputs saved to {outputs_file}")
    
    # compute_metrics()

if __name__ == "__main__":
    main()
