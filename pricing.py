import json
import os
import datetime

_OPENROUTER_PRICING_CACHE = None


def _load_openrouter_pricing():
    global _OPENROUTER_PRICING_CACHE
    if _OPENROUTER_PRICING_CACHE is not None:
        return _OPENROUTER_PRICING_CACHE
    json_path = os.path.join(os.path.dirname(__file__), "openrouter_pricing_02_04_2026.json")
    with open(json_path) as f:
        data = json.load(f)
    # Build lookup: slug (part after '/', without :free/:extended) -> model entry.
    # When there's a collision, prefer the paid (non-zero prompt price) entry.
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
    """Look up pricing from the bundled OpenRouter pricing snapshot.

    Returns a dict with keys: model_tag, release_date, input, input_cached, output.
    Prices are in USD per million tokens.
    Prints a warning and returns zeros if the model is not found.
    """
    lookup = _load_openrouter_pricing()

    def _make_result(entry):
        p = entry["pricing"]
        created = entry.get("created")
        release_date = (
            datetime.datetime.utcfromtimestamp(created).strftime("%Y-%m-%d")
            if created else ""
        )
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
    name = model_name.split("/")[-1].split("-run")[0].lower().replace("_", ".")
    name_no_date = name.split("-2026")[0].split("-2025")[0].split("-2024")[0]
    name_no_thinking = name_no_date.replace("-thinking", "")

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
