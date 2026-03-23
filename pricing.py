
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
        "Qwen3-235B": {# check on 30.10.25 on openrouter
            "model_tag": "",
            "release_date": "2024-09-25",
            "input": .18,
            "input_cached": None,
            "output": .54
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
        "Qwen3-VL-30B": { # check on 30.10.25 on openrouter
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
        "Qwen3-VL-235B": { # check on 30.10.25 on openrouter
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
    }
    model_name = model_name.split('-2025')[0].split('-2024')[0]
    model_name = model_name.split('/')[-1]  # Handle model names with slashes
    model_name = model_name.split('-run')[0]  # Handle model names with slashes
    if model_name in pricing:
        return pricing[model_name]
    models = list(pricing.keys())
    for m in models:
        if m in model_name:
            return pricing[m]
    print(f"Warning: Pricing for model '{model_name}' not found. Using default pricing.")
    return {
        "model_tag": "",
        "release_date": "2025-05-01",
        "input": .02,
        "input_cached": None,
        "output": .03
    }