import numpy as np
import pandas as pd
import glob
import json
import os
import re
from evaluate import load
from openai import OpenAI
import pandas as pd
from plots import create_level_wise_plots
from math import comb
from shortcut_analysis import (
    is_shortcut_used as is_shortcut_used_v2,
    analyze_shortcut,
    run_full_shortcut_analysis,
    ShortcutAnalysis
)
client = OpenAI(api_key="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")


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
        }
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
    
complexity = {
    "v-lol-level1": "basic",
    "v-lol-level2": "basic",
    "v-lol-level3": "basic",
    "v-lol-level4": "basic",
    "v-lol-level5": "basic",
    "v-lol-level6": "easy",
    "v-lol-level7": "easy",
    "v-lol-level8": "easy",
    "v-lol-level9": "easy",
    "v-lol-level10": "easy",
    "v-lol-level11": "medium",
    "v-lol-level12": "medium",
    "v-lol-level13": "medium",
    "v-lol-level14": "medium",
    "v-lol-level15": "medium",
    "v-lol-level16": "hard",
    "v-lol-level17": "hard",
    "v-lol-level18": "hard",
    "v-lol-level19": "hard",
    "v-lol-level20": "hard",
    1: "basic",
    2: "basic",
    3: "basic",
    4: "basic",
    5: "basic",
    6: "easy",
    7: "easy",
    8: "easy",
    9: "easy",
    10: "easy",
    11: "medium",
    12: "medium",
    13: "medium",
    14: "medium",
    15: "medium",
    16: "hard",
    17: "hard",
    18: "hard",
    19: "hard",
    20: "hard",
}

model_rename = {
    'gpt-4o': 'GPT-4o',
    'gpt-4.5-preview': 'GPT-4.5',
    'CodeLlama-34b-Instruct-hf': 'CodeLlama-34B',
    'DeepSeek-R1-Distill-Llama-70B': 'DeepSeek-R1',
    'Internlm2-20b': 'InternLM2-20B',
    'Llama-3.2-3B-Instruct': 'Llama 3.2-3B',
    'Llama-3.3-70B-Instruct': 'Llama 3.3-70B',
    'Llama-3.1-8B-Instruct': 'Llama 3.1-8B',
    'Mixtral-8x7B-Instruct-v0.1': 'Mixtral-8x7B',
    'Qwen2-57B-A14B-Instruct': 'Qwen2-57B',
    'QwQ-32B-Preview': 'QwQ-32B',
    'gemini-2.0-flash-thinking-exp-01-21': 'Gemini Thinking',
    'Llama-3.1-8B-Tuned-FFT': 'MetL Llama-8B',
    'Llama-3.1-8B-Tuned-LoRA': 'MetL Llama-8B LoRA'
}



def _pass_at_k_from_counts(c: int, n: int, k: int) -> float:
    """Unbiased pass@k estimator from n samples with c correct.

    Applies the Codex formula: 1 - C(n-c, k) / C(n, k) when k<=n.
    Handles edge cases including k>n and c==0.
    """
    if n <= 0:
        return 0.0
    k = min(max(int(k), 0), int(n))
    if k == 0:
        return 0.0
    c = int(c)
    n = int(n)
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (comb(n - c, k) / comb(n, k))


def _estimate_pass_k_for_df(df: pd.DataFrame, k: int) -> float:
    """Estimate pass@k over a dataframe of sample-level rows for many problems.

    Group by `Problem ID`, compute (c, n) per problem, then average the
    unbiased estimator across problems.
    """
    if df.empty:
        return float("nan")
    by_problem = df.groupby('Problem ID')['Solved'].agg(correct='sum', n='count').reset_index()
    if by_problem.empty:
        return float("nan")
    pass_vals = by_problem.apply(lambda r: _pass_at_k_from_counts(int(r['correct']), int(r['n']), k), axis=1)
    return float(pass_vals.mean())


def calc_metrics(data: pd.DataFrame) -> pd.DataFrame:
    data['complexity'] = data['Level'].map(complexity)

    reasoning_models = ['o1', 'o1-mini', 'o3-mini', 'o4-mini', 'o4-mini-high', 'o4-mini-low', 'o3', 'o3-pro', 'QwQ-32B',
                        'Gemini Flash Thinking', 'DeepSeek-R1']
    metalearning_models = ['MetL Llama-8B', 'MetL Llama-8B LoRA']

    # Update Completion Tokens for specific models
    for index, row in data.iterrows():
        if row['Model'] in ['Llama-3.1-8B-Tuned-FFT', 'Llama-3.1-8B-Tuned-LoRA']:
            data.at[index, 'Completion Tokens'] = row['Completion Tokens'] - row['Prompt Tokens']

    data['release_date'] = data['Model'].apply(lambda x: get_pricing(x)['release_date'])
    data['input_price'] = data['Model'].apply(lambda x: get_pricing(x)['input'])
    data['output_price'] = data['Model'].apply(lambda x: get_pricing(x)['output'])

    # Rename model
    data['Model'] = data['Model'].map(lambda x: model_rename.get(x, x))
    data['model_type'] = data['Model'].apply(lambda x: 'Reasoning LLM' if x in reasoning_models else (
        'Meta Learning' if x in metalearning_models else 'Default LLM'))

    # Create complexity-level aggregation
    complexity_data = data.groupby(['complexity', 'Model', 'model_type', 'release_date']).agg({
        'Solved': 'mean',
        'Completion Tokens': 'sum',
        'Syntax Score': 'mean'
    }).reset_index()

    # Create overall aggregation
    overall_data = data.groupby(['Model', 'model_type', 'release_date']).agg({
        'Solved': 'mean',
        'Completion Tokens': 'sum',
        'Syntax Score': 'mean'
    }).reset_index()
    overall_data['complexity'] = 'overall'

    # Combine both
    all_data = pd.concat([complexity_data, overall_data], ignore_index=True)

    # Pivot to get separate columns for each complexity level
    acc_data = all_data.pivot_table(
        index=['Model', 'model_type', 'release_date'],
        columns='complexity',
        values='Solved',
        aggfunc='mean'
    ).reset_index()

    # Flatten column names and ensure all complexity levels exist
    acc_data.columns.name = None
    complexity_cols = ['basic', 'easy', 'medium', 'hard', 'overall']
    for col in complexity_cols:
        if col not in acc_data.columns:
            acc_data[col] = np.nan

    # Add Syntax Score from overall_data
    syntax_scores = overall_data.groupby(['Model', 'model_type', 'release_date'])['Syntax Score'].mean().reset_index()
    acc_data = acc_data.merge(syntax_scores, on=['Model', 'model_type', 'release_date'], how='left')

    # Prepare token, cost, and shortcut data
    agg_dict = {
        'Completion Tokens': 'sum',
        'Prompt Tokens': 'sum',
        'input_price': 'first',
        'output_price': 'first',
        'Shortcuts': 'sum'
    }
    # Add shortcut type counts if available
    if 'Shortcut Type' in data.columns:
        # Count shortcuts by type per model
        shortcut_type_counts = data[data['Shortcuts'] == 1].groupby(
            ['Model', 'model_type', 'release_date', 'Shortcut Type']
        ).size().unstack(fill_value=0).reset_index()
    
    token_cost_data = data.groupby(['Model', 'model_type', 'release_date']).agg(agg_dict).reset_index()

    # Estimate how many passes we have per model so usage is reported per pass
    pass_counts = data.groupby(['Model', 'model_type', 'release_date']).agg(
        total_rows=('Problem ID', 'size'),
        unique_problems=('Problem ID', 'nunique')
    ).reset_index()
    pass_counts['unique_problems'] = pass_counts['unique_problems'].replace(0, np.nan)
    pass_counts['num_passes'] = (pass_counts['total_rows'] / pass_counts['unique_problems']).fillna(1.0)
    pass_counts['num_passes'] = pass_counts['num_passes'].clip(lower=1.0)
    
    token_cost_data = token_cost_data.merge(
        pass_counts[['Model', 'model_type', 'release_date', 'num_passes']],
        on=['Model', 'model_type', 'release_date'],
        how='left'
    )

    token_cost_data['num_passes'] = token_cost_data['num_passes'].fillna(1.0)
    token_cost_data['Prompt Tokens'] = token_cost_data['Prompt Tokens'] / token_cost_data['num_passes']
    token_cost_data['Completion Tokens'] = token_cost_data['Completion Tokens'] / token_cost_data['num_passes']
    token_cost_data['Shortcuts'] = token_cost_data['Shortcuts'] / token_cost_data['num_passes']

    token_cost_data['Total Cost'] = (
        token_cost_data['Prompt Tokens'] * token_cost_data['input_price'] +
        token_cost_data['Completion Tokens'] * token_cost_data['output_price']
    ) / 1_000_000  # Convert to dollars

    acc_data = acc_data.merge(
        token_cost_data[['Model', 'model_type', 'release_date', 'Completion Tokens', 'Prompt Tokens', 'Total Cost', 'Shortcuts']],
        on=['Model', 'model_type', 'release_date'],
        how='left'
    )

    # Compute pass@k (overall per model across problems)
    pass_overall = data.groupby(['Model', 'model_type', 'release_date']).apply(
        lambda df: pd.Series({
            'Pass@4': _estimate_pass_k_for_df(df, 4),
            'Pass@8': _estimate_pass_k_for_df(df, 8),
        })
    ).reset_index()

    acc_data = acc_data.merge(pass_overall, on=['Model', 'model_type', 'release_date'], how='left')

    # LRL calculation
    acc_data['LRL'] = (acc_data['overall'] * 20).round(4)
    acc_data.drop(columns=['overall'], inplace=True)

    # LRL@k from pass@k
    if 'Pass@4' in acc_data.columns:
        acc_data['LRL@4'] = (acc_data['Pass@4'] * 20).round(4)
    if 'Pass@8' in acc_data.columns:
        acc_data['LRL@8'] = (acc_data['Pass@8'] * 20).round(4)

    # Scale and rename token columns if present
    if 'Completion Tokens' in acc_data.columns:
        acc_data['Completion Tokens'] = (acc_data['Completion Tokens'] / 1_000_000).round(4)
        acc_data.rename(columns={'Completion Tokens': 'Completion Tokens (M)'}, inplace=True)
    if 'Prompt Tokens' in acc_data.columns:
        acc_data['Prompt Tokens'] = (acc_data['Prompt Tokens'] / 1_000_000).round(4)
        acc_data.rename(columns={'Prompt Tokens': 'Prompt Tokens (M)'}, inplace=True)
    if 'Shortcuts' in acc_data.columns:
        acc_data['Shortcuts'] = acc_data['Shortcuts'].round(2)

    acc_data['LRL per Dollar'] = (acc_data['LRL'] / acc_data['Total Cost']).round(4)
    acc_data['Total Cost'] = acc_data['Total Cost'].round(4)

    # Final column ordering: Model, LRLs, Syntax, tierwise, then the rest
    preferred_order = [
        'Model', 'LRL', 'LRL@4', 'LRL@8', 'Syntax Score',
        'basic', 'easy', 'medium', 'hard'
    ]
    # Include remaining informative columns next
    trailing_priority = [
        'LRL per Dollar', 'Total Cost', 'Completion Tokens (M)', 'Prompt Tokens (M)', 'Shortcuts'
    ]
    final_cols = [c for c in preferred_order if c in acc_data.columns] + \
                 [c for c in trailing_priority if c in acc_data.columns]
    # Append any other leftover columns at the end to avoid dropping
    leftover = [c for c in acc_data.columns if c not in final_cols]
    acc_data = acc_data[final_cols]

    return acc_data




def retrieve_output_items(output_dir, model_name):
    """Retrieve the results of an evaluation run."""
    ids_path = f"{output_dir}/eval_run_info.json"
    # Check if eval run info exists
    if not os.path.exists(ids_path):
        print(f"No eval run info found at {ids_path}. Please run the evaluation first.")
        return []

    # Load eval and run IDs
    try:
        with open(ids_path, "r") as f:
            ids_info = json.load(f)
    except json.JSONDecodeError:
        print(f"Invalid JSON format in {ids_path}. Please check the file.")
        return []

    eval_id, run_id = ids_info["eval_id"], ids_info["run_id"]
    
    
    print(f"Retrieving results for Eval ID: {eval_id}, Run ID: {run_id}")
    result = client.evals.runs.retrieve(run_id=run_id, eval_id=eval_id)
    print(f"Current status: {result.status}")
    if result.status in ["failed", "canceled"]:
        print(f"Evaluation with status '{result.status}': {json.dumps(result, indent=2)}")
        raise ValueError(
            f"Evaluation run failed or was canceled. Please check the OpenAI dashboard for more details."
        )

    # Implement pagination to get all output items
    all_output_items = []
    has_more = True
    after = None

    while has_more:
        page = client.evals.runs.output_items.list(eval_id=eval_id, run_id=run_id,
                                                           status="pass", limit=100, after=after)
        all_output_items.extend(page.data)

        # Check if we need to fetch more pages
        if page.has_more:
            after = page.data[-1].id
        else:
            has_more = False
    print(f"Fetched {len(all_output_items)} output items.")
    p, c , c1= 0, 0, 0
    tot = 0
    if result.status not in ["completed", "failed", "canceled"]:
        raise ValueError(
            f"Evaluation run is still in progress. Current status: {result.status}. "
            "Please wait until the evaluation is completed."
        )
    if len(all_output_items) < 1000:
        raise ValueError(
            f"Expected 1000 output items, but got {len(all_output_items)}. "
            "Please check the evaluation run status and try again later."
        )
    # save results to a file
    model_outputs = []
    for output_item in all_output_items:
        datasource_item = output_item.datasource_item
        id, prompt, validation_program = datasource_item['id'], datasource_item['prompt'], datasource_item[
            'validation program']
        model_completion = output_item.sample.output[0].content

        if output_item.sample.error is not None:
            print(f"Skipping item {id} due to error: {output_item.sample.error.message}")
            if output_item.sample.usage is not None:
                raise ValueError('Sample usage should be None if there is an error.')
            # continue
            prompt_tokens = 0
            completion_tokens = 0
        else:
            prompt_tokens = output_item.sample.usage.prompt_tokens + output_item.sample.usage.cached_tokens
            completion_tokens = output_item.sample.usage.completion_tokens
            p, c, c1, tot = p + output_item.sample.usage.prompt_tokens, c + completion_tokens, c1  + output_item.sample.usage.cached_tokens, tot + output_item.sample.usage.total_tokens
        reference = {
            "validation_program": validation_program,
            "evaluation_config": {
                "positive_predicate": "eastbound",
                "negative_predicate": "westbound"
            }
        }
        model_outputs.append(
            {"model_name": model_name,
             "problem_id": int(id),
             "level": 1 + (int(id) // 50),
             "prompt": prompt,
             "model_completion": model_completion,
             "validation_program": validation_program,
             "prompt_tokens": prompt_tokens,
             "completion_tokens": completion_tokens,
             "reference": reference}
        )
    print(f"Total prompt tokens: {p}, Total completion tokens: {c}, Total cached tokens: {c1}, Total tokens: {tot}")
    # Save the model outputs to a file
    outputs_path = f"{output_dir}/model_outputs.json"
    os.makedirs(os.path.dirname(outputs_path), exist_ok=True)
    with open(outputs_path, "w") as f:
        json.dump(model_outputs, f, indent=2)
    return model_outputs




def load_model_outputs(output_dir):
    outputs_path = f"{output_dir}/model_outputs.json"   
    model_name = output_dir.split('/')[-1]

    # Check if we already have the outputs
    if os.path.exists(outputs_path):
        print(f"Model outputs already exist at {outputs_path}. Skipping retrieval.")
        with open(outputs_path, "r") as f:
            model_outputs = json.load(f)
    elif os.path.exists(f"{output_dir}/eval_run_info.json"):
        print(f"Retrieving model outputs from evaluation run in {output_dir}...")
        # Retrieve output items using the correct parameters
        model_outputs = retrieve_output_items(output_dir, model_name)
    else:
        print(f"No model outputs found in {output_dir}. Please run the evaluation first or check the directory.")
        return []

    # Convert dictionary format to list format if needed
    if isinstance(model_outputs, dict):
        # Check if it has numeric keys (old format)
        items = []
        for key, value in model_outputs.items():
            if isinstance(key, (str, int)) and str(key).isdigit():
                problem_id = int(key)
                items.append({
                    "model_name": model_name,
                    "problem_id": problem_id,
                    "level": 1 + (problem_id // 50),
                    "prompt": value["prompt"],
                    "model_completion": value["model_completion"],
                    "validation_program": value["validation_program"],
                    "prompt_tokens": value["prompt_tokens"],
                    "completion_tokens": value["completion_tokens"],
                    "reference": value["reference"]
                })
        model_outputs = items
    elif isinstance(model_outputs, list):
        # Ensure each item has a model_name; if not, derive from directory
        fixed = []
        for it in model_outputs:
            if isinstance(it, dict) and "model_name" not in it:
                it = {**it, "model_name": model_name}
            fixed.append(it)
        model_outputs = fixed
    return model_outputs


def evaluate(model_outputs: list, symbolic_judge):
    """
    Evaluate model outputs using symbolic judge and detect shortcuts.
    
    Uses improved shortcut detection that identifies:
    - Grounded train references (train0, train1, etc.)
    - Grounded car references (car0_1, car1_2, etc.)
    - Enumeration patterns (explicit listing of cars)
    """
    # Extract all required values from the model outputs
    predictions = []
    references = []
    shortcuts = 0
    shortcut_types = []
    shortcut_analyses = []

    for item in model_outputs:
        if "model_completion" not in item:
            raise ValueError(f"Missing 'model_completion' in item: {item}")
        prediction = item["model_completion"]
        # if prompt is included in the model completion, remove it
        if "prompt" in item and item["prompt"] in prediction:
            prediction = prediction.replace(item["prompt"], "").strip()        
        # remove thinking from the prediction
        if "</think>" in prediction:
            prediction = prediction.split("</think>")[-1].strip()
        # remove weird formating \\
        prediction = prediction.replace("\\", "")
        
        # Use improved shortcut detection
        analysis = analyze_shortcut(prediction)
        if analysis.is_shortcut:
            shortcuts += 1
        shortcut_types.append(analysis.shortcut_type)
        shortcut_analyses.append(analysis)
        
        predictions.append(prediction)
        references.append(item["reference"])
    
    # Compute the symbolic judge score
    score = symbolic_judge.compute(
        predictions=predictions,
        references=references
    )

    # Extract problem IDs for sorting and tracking
    problem_ids = [item["problem_id"] for item in model_outputs]
    levels = [item["problem_id"] // 50 + 1 for item in model_outputs]  # Assuming levels are based on problem IDs

    print(f"Model score: {score['accuracy']:.2f}, Total items: {len(model_outputs)}, "
          f"partial_score: {score['partial_score']:.2f}, syntax_score: {score['syntax_score']:.2f}, shortcuts: {shortcuts}")

    # Model,Level,Prompt Tokens,Completion Tokens,Problem ID,Solved,Syntax Score
    df = pd.DataFrame({
        "Model": [item["model_name"] for item in model_outputs],
        "Level": levels,
        "Prompt Tokens": [item["prompt_tokens"] for item in model_outputs],
        "Completion Tokens": [item["completion_tokens"] for item in model_outputs],
        "Problem ID": problem_ids,
        "Solved": [score['detailed_results'][i]["is_correct"] for i in range(len(problem_ids))],
        "Syntax Score": [score['detailed_results'][i]["syntax_valid"] for i in range(len(problem_ids))],
        "Shortcuts": [1 if shortcut_analyses[i].is_shortcut else 0 for i in range(len(problem_ids))],
        "Shortcut Type": shortcut_types,
        "Shortcut Confidence": [shortcut_analyses[i].confidence for i in range(len(problem_ids))],
    })
    return df

def compute_metrics(output_dir: str, replace_existing: bool = False):
    """Compute metrics aggregating per-model results saved separately.

    This refactored version no longer persists a combined `results.csv` at the
    root of `output_dir`. Instead:
      - Each model directory under `output_dir` should contain a `model_outputs.json`.
      - We evaluate EACH model independently (if not already evaluated) and
        store its per-problem scores in `<model_dir>/results.csv`.
      - Aggregation (concatenation across models) happens on-the-fly for the
        purpose of computing `metrics.csv`, plots, etc., without saving the
        combined per-problem file.
    """

    # symbolic_judge = load("VerifiableRewardsForScalableLogicalReasoning.py")
    symbolic_judge = load("AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning")

    model_dirs = sorted([d for d in glob.glob(f"{output_dir}/*") if os.path.isdir(d)])
    if not model_dirs:
        print(f"No model directories found in {output_dir}.")
        return

    per_model_results = []
    evaluated_models = []

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        per_model_results_path = os.path.join(model_dir, "results.csv")

        if os.path.exists(per_model_results_path) and not replace_existing:
            # Load existing per-model results
            try:
                df_model = pd.read_csv(per_model_results_path)
                per_model_results.append(df_model)
                evaluated_models.append(model_name)
                print(f"Loaded existing per-model results for {model_name}.")
                continue
            except Exception as e:
                # Recover by re-evaluating if the file is corrupted
                print(f"Failed to read existing per-model results for {model_name}: {e}. Recomputing...")

        # Need to (re)evaluate this model
        print(f"Evaluating model: {model_name}")
        model_outputs = load_model_outputs(model_dir)
        if not model_outputs:
            print(f"No outputs for model {model_name}; skipping.")
            continue

        df_model = evaluate(model_outputs, symbolic_judge)
        # Persist only this model's detailed results
        df_model.to_csv(per_model_results_path, index=False)
        print(f"Saved per-model results to {per_model_results_path}")
        per_model_results.append(df_model)
        evaluated_models.append(model_name)

    if not per_model_results:
        print("No per-model results available to aggregate.")
        return

    # Aggregate on the fly (not persisted as combined per-problem CSV)
    all_results = pd.concat(per_model_results, ignore_index=True)
    print(f"Aggregated {len(evaluated_models)} models: {', '.join(evaluated_models)}")

    # Compute high-level metrics
    metrics = calc_metrics(all_results)
    metrics.sort_values(by='LRL', ascending=False, inplace=True)
    metrics.reset_index(drop=True, inplace=True)

    # Persist only aggregated summary artifacts
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    metrics.to_csv(metrics_path)
    print(f"Metrics summary saved to {metrics_path}")

    print("\n## Metrics Summary")
    print(metrics.to_markdown(index=True, tablefmt="grid"))

    # Plots (still rely on aggregated DataFrame in-memory)
    print("\n## Generating Level-wise Performance Plots...")
    level_performance = create_level_wise_plots(all_results, output_dir + '/plots')
    level_perf_path = os.path.join(output_dir, 'metrics_level_wise.csv')
    level_performance.to_csv(level_perf_path, index=False)
    print(f"Level-wise performance data saved to {level_perf_path}")

    print("\n## Plots generated successfully!")
    print(f"Individual model plots: {output_dir}plots/[model_name]_level_performance.png")
    print(f"Comparison plot: {output_dir}plots/all_models_level_performance_comparison.png")
    print(f"Performance heatmap: {output_dir}plots/performance_heatmap.png")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute metrics for LLM evaluations.")
    parser.add_argument("--replace-existing", action="store_true",
                        help="If set, will re-evaluate all models even if results already exist.")
    parser.add_argument("--output-dir", type=str, default="output/eval-openai",
                        help="Directory to save evaluation results.")
    parser.add_argument("--shortcut-analysis", action="store_true",
                        help="Run comprehensive shortcut analysis after evaluation.")
    parser.add_argument("--shortcut-only", action="store_true",
                        help="Only run shortcut analysis (skip main evaluation).")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="Filter to specific models for shortcut analysis.")
    args = parser.parse_args()
    
    if args.shortcut_only:
        # Run only shortcut analysis
        print("Running shortcut analysis only...")
        results = run_full_shortcut_analysis(
            output_dir=args.output_dir,
            models_filter=args.models,
            plots_dir=os.path.join(args.output_dir, 'shortcut_analysis')
        )
    else:
        # Run main evaluation
        replace_existing = args.replace_existing
        compute_metrics(output_dir=args.output_dir, replace_existing=replace_existing)
        
        # Optionally run shortcut analysis after
        if args.shortcut_analysis:
            print("\n" + "=" * 60)
            print("Running comprehensive shortcut analysis...")
            print("=" * 60)
            results = run_full_shortcut_analysis(
                output_dir=args.output_dir,
                models_filter=args.models,
                plots_dir=os.path.join(args.output_dir, 'shortcut_analysis')
            )
