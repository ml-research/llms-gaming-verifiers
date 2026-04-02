
def get_pricing(model_name: str) -> dict:
    """ Get pricing information for a specific model.
    Args:
        model_name (str): The name of the model to get pricing for.
    Returns:
        dict: A dictionary containing pricing information for the specified model.
    Raises:
        ValueError: If the model name is not found in the pricing data.
    """
    pricing = {
        "gpt-5": {
            "model_tag": "gpt-5",
            "release_date": "2025-10-01",
            "input": 1.25,
            "input_cached": 1.25,
            "output": 10.00
        },
        "gpt-5-mini": {
            "model_tag": "gpt-5-mini",
            "release_date": "2025-04-14",
            "input": 0.25,
            "input_cached": 0.25,
            "output": 2.00
        },
        "gpt-5-nano": {
            "model_tag": "gpt-5-nano",
            "release_date": "2025-04-14",
            "input": 0.05,
            "input_cached": 0.005,
            "output": 0.40
        },
        "gpt-4.1": {
            "model_tag": "gpt-4.1-2025-04-14",
            "release_date": "2025-04-14",
            "input": 2.00,
            "input_cached": 0.50,
            "output": 8.00
        },
        "gpt-4.1-mini": {
            "model_tag": "gpt-4.1-mini-2025-04-14",
            "release_date": "2025-04-14",
            "input": 0.40,
            "input_cached": 0.10,
            "output": 1.60
        },
        "gpt-4.1-nano": {
            "model_tag": "gpt-4.1-nano-2025-04-14",
            "release_date": "2025-04-14",
            "input": 0.10,
            "input_cached": 0.025,
            "output": 0.40
        },
        "gpt-4.5-preview": {
            "model_tag": "gpt-4.5-preview-2025-02-27",
            "release_date": "2025-02-27",
            "input": 75.00,
            "input_cached": 37.50,
            "output": 150.00
        },
        "gpt-4o": {
            "model_tag": "gpt-4o-2024-08-06",
            "release_date": "2024-08-06",
            "input": 2.50,
            "input_cached": 1.25,
            "output": 10.00
        },
        "gpt-4o-mini": {
            "model_tag": "gpt-4o-mini-2024-07-18",
            "release_date": "2024-07-18",
            "input": 0.15,
            "input_cached": 0.075,
            "output": 0.60
        },
        "o1": {
            "model_tag": "o1-2024-12-17",
            "release_date": "2024-12-17",
            "input": 15.00,
            "input_cached": 7.50,
            "output": 60.00
        },
        "gpt-4-turbo": {
            "model_tag": "gpt-4-turbo-2024-04-09",
            "release_date": "2024-04-09",
            "input": 10,
            "input_cached": 10,
            "output": 30.00
        },
        "gpt-3.5": {
            "model_tag": "gpt-3.5-turbo-0125",
            "release_date": "2024-04-09",
            "input": .5,
            "input_cached": 10,
            "output": 1.5
        },
        "o1-pro": {
            "model_tag": "o1-pro-2025-03-19",
            "release_date": "2025-03-19",
            "input": 150.00,
            "input_cached": None,
            "output": 600.00
        },
        "o3": {
            "model_tag": "o3-2025-04-16",
            "release_date": "2025-04-16",
            "input": 10.00,
            "input_cached": 2.50,
            "output": 40.00
        },
        "o4-mini": {
            "model_tag": "o4-mini-2025-04-16",
            "release_date": "2025-04-16",
            "input": 1.10,
            "input_cached": 0.275,
            "output": 4.40
        },
        "o4-mini-high": {
            "model_tag": "o4-mini-2025-04-16",
            "release_date": "2025-04-16",
            "input": 1.10,
            "input_cached": 0.275,
            "output": 4.40
        },
        "o4-mini-low": {
            "model_tag": "o4-mini-2025-04-16",
            "release_date": "2025-04-16",
            "input": 1.10,
            "input_cached": 0.275,
            "output": 4.40
        },
        "o4-mini-fail": {
            "model_tag": "o4-mini-2025-04-16",
            "release_date": "2025-04-16",
            "input": 1.10,
            "input_cached": 0.275,
            "output": 4.40
        },
        "o3-mini": {
            "model_tag": "o3-mini-2025-01-31",
            "release_date": "2025-01-31",
            "input": 1.10,
            "input_cached": 0.55,
            "output": 4.40
        },
        "o1-mini": {
            "model_tag": "o1-mini-2024-09-12",
            "release_date": "2024-09-12",
            "input": 1.10,
            "input_cached": 0.55,
            "output": 4.40
        },
        # Open Source Models, prices from OpenRouter
        "Llama-3.3-70B": {
            "model_tag": "",
            "release_date": "2024-12-06",
            "input": .10,
            "input_cached": None,
            "output": .25
        },
        "Llama-3.1-8B": {
            "model_tag": "",
            "release_date": "2024-07-23",
            "input": .02,
            "input_cached": None,
            "output": .03
        },
        "llama3-8b": {
            "model_tag": "",
            "release_date": "2024-07-23",
            "input": .02,
            "input_cached": None,
            "output": .03
        },
        "Llama-3.2-3B": {
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .015,
            "input_cached": None,
            "output": .025
        },
        "Llama-3.2-1B": {
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .015,
            "input_cached": None,
            "output": .025
        },
        "Qwen3-0.6B": {
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .015,
            "input_cached": None,
            "output": .025
        },
        "Qwen3-1.7B": {
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .015,
            "input_cached": None,
            "output": .025
        },
        "Qwen3-4B": {
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .015,
            "input_cached": None,
            "output": .025
        },
        "Qwen3-8B": { # check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .035,
            "input_cached": None,
            "output": .138
        },
        "Qwen3-14B": {# check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .05,
            "input_cached": None,
            "output": .22
        },
        "Qwen3-32B": {# check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .05,
            "input_cached": None,
            "output": .2
        },
        "Qwen3-235B-A22B": {# check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .18,
            "input_cached": None,
            "output": .54
        },
        "Qwen3.5-4B": {
            "model_tag": "",
            "release_date": "2026-02-25",
            "input": .015,
            "input_cached": None,
            "output": .025
        },
        "Qwen3.5-9B": {
            "model_tag": "",
            "release_date": "2026-03-10",
            "input": .05,
            "input_cached": None,
            "output": .15
        },
        "Qwen3.5-27B": {
            "model_tag": "",
            "release_date": "2026-02-25",
            "input": .195,
            "input_cached": None,
            "output": 1.56
        },
        "Qwen3.5-35B-A3B": {
            "model_tag": "",
            "release_date": "2026-02-25",
            "input": .1625,
            "input_cached": None,
            "output": 1.30
        },
        "Qwen3.5-122B-A10B": {
            "model_tag": "",
            "release_date": "2026-02-25",
            "input": .26,
            "input_cached": None,
            "output": 2.08
        },
        "Qwen3.5-397B-A17B": {
            "model_tag": "",
            "release_date": "2026-02-25",
            "input": .39,
            "input_cached": None,
            "output": 2.34
        },
        ## Vision Models
        "Qwen3-VL-2B": { # check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": 0,
            "input_cached": None,
            "output": 0
        },
        "Qwen3-VL-4B": { # check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": 0,
            "input_cached": None,
            "output": 0
        },
        "Qwen3-VL-8B": { # check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .008,
            "input_cached": None,
            "output": .5
        },
        "Qwen3-VL-30B-A3B": { # check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .15,
            "input_cached": None,
            "output": .6
        },
        "Qwen3-VL-32B": { # check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .35,
            "input_cached": None,
            "output": 1.1
        },
        "Qwen3-VL-235B-A22B": { # check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .22,
            "input_cached": None,
            "output": .88
        },
        # Specialized Models
        "Qwen3-Coder-30B-A3B": {# check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .06,
            "input_cached": None,
            "output": .25
        },
        "Qwen3-Next-80B-A3B-Instruct": {# check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .1,
            "input_cached": None,
            "output": .8
        },
        "Qwen3-Next-80B-A3B-Thinking": {# check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .15,
            "input_cached": None,
            "output": 1.2
        },
        "Qwen3-Coder-408B-A35B": {# check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .38,
            "input_cached": None,
            "output": 1.53
        },
        "Qwen3-30B-A3B": {# check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .06,
            "input_cached": None,
            "output": .22
        },
        "DeepSeek-R1-0528-Qwen3-8B-Thinking": {
            "model_tag": "",
            "release_date": "2025-05-28",
            "input": .55,
            "input_cached": None,
            "output": .55
        },
        "GLM-4.5-Air-Thinking": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .44,
            "input_cached": None,
            "output": .44
        },
        "GLM-4.7-FP8-Thinking": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .48,
            "input_cached": None,
            "output": .48
        },
        "Kimi-K2": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .57,
            "input_cached": None,
            "output": 2.3
        },
        "Kimi-K2.5-Thinking": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .38,
            "input_cached": None,
            "output": 1.9
        },
        "Llama-3.3-Nemotron-Super-49B-v1-Thinking": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .17,
            "input_cached": None,
            "output": .17
        },
        "Llama-3_1-Nemotron-Ultra-253B-v1-Thinking": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .18,
            "input_cached": None,
            "output": .18
        },
        "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-Thinking": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .48,
            "input_cached": None,
            "output": .48
        },
        "NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Thinking": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .41,
            "input_cached": None,
            "output": .41
        },
        "Nemotron-Cascade-2-30B-A3B-Thinking": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .42,
            "input_cached": None,
            "output": .42
        },
        "gpt-oss-120b": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .29,
            "input_cached": None,
            "output": .29
        },
        "gpt-oss-120b-effort-high": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .039,
            "input_cached": None,
            "output": .19
        },
        "gpt-oss-120b-effort-low": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .039,
            "input_cached": None,
            "output": .19
        },
        "gpt-oss-120b-effort-medium": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .039,
            "input_cached": None,
            "output": .19
        },
        "gpt-oss-20b": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .03,
            "input_cached": None,
            "output": .19
        },
        "gpt-oss-20b-effort-high": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .03,
            "input_cached": None,
            "output": .19
        },
        "gpt-oss-20b-effort-low": {
            "model_tag": "",
            "release_date": "2025-01-01",
            "input": .03,
            "input_cached": None,
            "output": .19
        },
        "DeepSeek-R1-Distill-Llama-8B": {
            "model_tag": "",
            "release_date": "2025-05-01",
            "input": .02,
            "input_cached": None,
            "output": .03
        },
        "DeepSeek-R1-Distill-Llama-70B": {
            "model_tag": "",
            "release_date": "2025-05-01",
            "input": .03,
            "input_cached": None,
            "output": .13
        },
        "DeepSeek-R1-Distill-Qwen-1.5B": {
            "model_tag": "",
            "release_date": "2025-05-01",
            "input": .02,
            "input_cached": None,
            "output": .03
        },
        "DeepSeek-R1-Distill-Qwen-7B": {
            "model_tag": "",
            "release_date": "2025-05-01",
            "input": .02,
            "input_cached": None,
            "output": .03
        },
        "DeepSeek-R1-Distill-Qwen-14B": {
            "model_tag": "",
            "release_date": "2025-05-01",
            "input": .02,
            "input_cached": None,
            "output": .03
        },
        "Llama-4": {
            "model_tag": "",
            "release_date": "2025-05-01",
            "input": .08,
            "input_cached": None,
            "output": .3
        },
        "gemini-2.0-flash-thinking-exp-01-21": {
            "model_tag": "",
            "release_date": "2025-01-21",
            "input": 1.10,
            "input_cached": 0.55,
            "output": 4.40
        },
        "Olmo-3-32B-Think": {
            "model_tag": "",
            "release_date": "2026-01-01",
            "input": .15,
            "input_cached": None,
            "output": .50
        },
        "Olmo-3-7B-Think": {
            "model_tag": "",
            "release_date": "2026-01-01",
            "input": .12,
            "input_cached": None,
            "output": .20
        },
        "Olmo-3.1-32B-Think": {
            "model_tag": "",
            "release_date": "2026-01-01",
            "input": .15,
            "input_cached": None,
            "output": .50
        },
        "mistralai_Ministral-3-14B-Reasoning-2512": {
            "model_tag": "",
            "release_date": "2025-12-02",
            "input": .20,
            "input_cached": None,
            "output": .20
        },
        "mistralai_Ministral-3-3B-Reasoning-2512": {
            "model_tag": "",
            "release_date": "2025-12-02",
            "input": .10,
            "input_cached": None,
            "output": .10
        },
        "mistralai_Ministral-3-8B-Reasoning-2512": {
            "model_tag": "",
            "release_date": "2025-12-02",
            "input": .15,
            "input_cached": None,
            "output": .15
        },
    }
    model_name = model_name.split('-2026')[0].split('-2025')[0].split('-2024')[0]
    model_name = model_name.split('/')[-1]  # Handle model names with slashes
    model_name = model_name.split('-run')[0]  # Handle model names with slashes
    if model_name in pricing:
        return pricing[model_name]
    model_name_lc = model_name.lower()
    models = sorted(pricing.keys(), key=len, reverse=True)
    for m in models:
        m_lc = m.lower()
        if m_lc in model_name_lc or model_name_lc in m_lc:
            return pricing[m]
    print(f"Warning: Pricing for model '{model_name}' not found. Using default pricing.")
    return {
        "model_tag": "",
        "release_date": "2025-05-01",
        "input": .02,
        "input_cached": None,
        "output": .03
    }


import json as _json
import os as _os
import datetime as _datetime

_OPENROUTER_PRICING_CACHE = None

def _load_openrouter_pricing():
    global _OPENROUTER_PRICING_CACHE
    if _OPENROUTER_PRICING_CACHE is not None:
        return _OPENROUTER_PRICING_CACHE
    json_path = _os.path.join(_os.path.dirname(__file__), "openrouter_pricing_02_04_2026.json")
    with open(json_path) as f:
        data = _json.load(f)
    # Build lookup: slug (part after '/', without :free/:extended) -> model entry
    # When there's a collision, prefer the paid (non-zero prompt) entry
    lookup = {}
    for m in data["data"]:
        raw_slug = m["id"].split("/")[-1].lower()
        slug = raw_slug.split(":")[0]
        prompt_price = float(m["pricing"].get("prompt", 0) or 0)
        if slug not in lookup:
            lookup[slug] = m
        else:
            existing_price = float(lookup[slug]["pricing"].get("prompt", 0) or 0)
            if prompt_price > existing_price:
                lookup[slug] = m
    _OPENROUTER_PRICING_CACHE = lookup
    return lookup


def get_pricing_v2(model_name: str) -> dict:
    """Get pricing from the downloaded OpenRouter pricing JSON.
    Prices are in USD per million tokens (same format as get_pricing).
    Prints a warning if the model is not found in the OpenRouter list.
    """
    lookup = _load_openrouter_pricing()

    def _make_result(entry):
        p = entry["pricing"]
        created = entry.get("created")
        release_date = _datetime.datetime.utcfromtimestamp(created).strftime("%Y-%m-%d") if created else ""
        prompt_per_tok = float(p.get("prompt", 0) or 0)
        completion_per_tok = float(p.get("completion", 0) or 0)
        cache_per_tok = p.get("input_cache_read")
        return {
            "model_tag": entry["id"],
            "release_date": release_date,
            "input": prompt_per_tok * 1_000_000,
            "input_cached": float(cache_per_tok) * 1_000_000 if cache_per_tok else None,
            "output": completion_per_tok * 1_000_000,
        }

    # Normalize: strip path prefix and -run suffix, lowercase, unify _ and .
    name = model_name.split('/')[-1].split('-run')[0].lower().replace('_', '.')
    # Also prepare date-stripped variant
    name_no_date = name.split('-2026')[0].split('-2025')[0].split('-2024')[0]
    name_no_thinking = name_no_date.replace('-thinking', '')

    for candidate in [name, name_no_date, name_no_thinking]:
        if candidate in lookup:
            return _make_result(lookup[candidate])

    # Substring match (longest key first to prefer more specific matches)
    for search in [name, name_no_date, name_no_thinking]:
        candidates = sorted(lookup.keys(), key=len, reverse=True)
        for key in candidates:
            if key in search or search in key:
                return _make_result(lookup[key])

    print(f"Warning: Model '{model_name}' not found in OpenRouter pricing list.")
    return {
        "model_tag": "",
        "release_date": "",
        "input": 0.0,
        "input_cached": None,
        "output": 0.0,
    }