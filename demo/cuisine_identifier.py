from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from math import exp
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

CUISINE_PREFIX = "cuisine:"
DEFAULT_MATCH_CUTOFF = 0.78


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_rules_path() -> Path:
    return project_root() / "data" / "mined" / "association_rules.csv"


def default_ingredients_path() -> Path:
    return project_root() / "data" / "processed" / "ingredients.json"


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s-]", " ", value).strip().lower().split())


def load_ingredient_index(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    alias_to_canonical: dict[str, str] = {}
    canonical_to_aliases: dict[str, set[str]] = defaultdict(set)

    for entry in payload:
        raw = normalize_text(str(entry["ingredient"]))
        canonical = normalize_text(str(entry["canonicalized"]))
        alias_to_canonical[raw] = canonical
        alias_to_canonical[canonical] = canonical
        canonical_to_aliases[canonical].add(raw)
        canonical_to_aliases[canonical].add(canonical)

    canonical_terms = sorted(canonical_to_aliases.keys())
    search_terms = sorted(alias_to_canonical.keys())
    return {
        "alias_to_canonical": alias_to_canonical,
        "canonical_to_aliases": canonical_to_aliases,
        "canonical_terms": canonical_terms,
        "search_terms": search_terms,
    }


def similarity_score(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def ranked_matches(query: str, candidates: Iterable[str], limit: int = 5, cutoff: float = 0.60) -> list[tuple[str, float]]:
    cleaned = normalize_text(query)
    scored = [
        (candidate, similarity_score(cleaned, candidate))
        for candidate in candidates
    ]
    scored = [item for item in scored if item[1] >= cutoff]
    scored.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    return scored[:limit]


def resolve_ingredient(token: str, index: dict[str, object], cutoff: float = DEFAULT_MATCH_CUTOFF) -> dict[str, object] | None:
    cleaned = normalize_text(token)
    if not cleaned:
        return None

    alias_to_canonical = index["alias_to_canonical"]
    if cleaned in alias_to_canonical:
        canonical = alias_to_canonical[cleaned]
        return {
            "input": token,
            "matched": cleaned,
            "canonical": canonical,
            "score": 1.0,
        }

    candidates = ranked_matches(cleaned, index["search_terms"], limit=10, cutoff=cutoff)
    if not candidates:
        return None

    best_candidate, score = candidates[0]
    if score < cutoff:
        return None

    canonical = alias_to_canonical[best_candidate]
    return {
        "input": token,
        "matched": best_candidate,
        "canonical": canonical,
        "score": score,
    }


def suggest_ingredients(token: str, index: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
    alias_to_canonical = index["alias_to_canonical"]
    canonical_to_aliases = index["canonical_to_aliases"]
    ranked = ranked_matches(token, index["search_terms"], limit=limit * 4, cutoff=0.60)
    suggestions: dict[str, dict[str, object]] = {}

    for candidate, score in ranked:
        canonical = alias_to_canonical[candidate]
        current = suggestions.get(canonical)
        if current is None or score > current["score"]:
            suggestions[canonical] = {
                "canonical": canonical,
                "matched": candidate,
                "score": score,
                "aliases": sorted(canonical_to_aliases[canonical]),
            }

    ordered = sorted(suggestions.values(), key=lambda item: (item["score"], item["canonical"]), reverse=True)
    return ordered[:limit]


def normalize_user_ingredients(text: str, ingredients_path: Path) -> tuple[list[str], list[dict[str, object]], list[str]]:
    index = load_ingredient_index(ingredients_path)
    selected: list[str] = []
    resolved: list[dict[str, object]] = []
    unmatched: list[str] = []

    seen = set()
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        match = resolve_ingredient(token, index)
        if match is None:
            unmatched.append(token)
            continue
        canonical = match["canonical"]
        if canonical not in seen:
            selected.append(canonical)
            seen.add(canonical)
        resolved.append(match)

    return selected, resolved, unmatched


def load_rules(path: Path) -> pd.DataFrame:
    rules = pd.read_csv(path)
    required = {"antecedent", "consequent", "support", "confidence", "lift", "rule_type"}
    missing = required - set(rules.columns)
    if missing:
        raise ValueError(f"Missing columns in rules CSV: {sorted(missing)}")
    return rules


def parse_itemset(value: str) -> set[str]:
    return {part.strip().lower() for part in str(value).split(",") if part.strip()}


def extract_cuisine(value: str) -> str:
    for item in parse_itemset(value):
        if item.startswith(CUISINE_PREFIX):
            return item.removeprefix(CUISINE_PREFIX)
    return ""


def is_forward_cuisine_rule(consequent: str) -> bool:
    items = parse_itemset(consequent)
    return len(items) == 1 and next(iter(items)).startswith(CUISINE_PREFIX)


def score_cuisines(ingredients: set[str], rules: pd.DataFrame) -> list[dict[str, object]]:
    forward_rules = rules[
        (rules["rule_type"] == "ingredient_to_cuisine")
        & rules["consequent"].map(is_forward_cuisine_rule)
    ].copy()
    if forward_rules.empty:
        return []

    all_cuisines = sorted(forward_rules["consequent"].map(extract_cuisine).dropna().unique())
    cuisine_scores: dict[str, float] = defaultdict(float)
    matches_by_cuisine: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in forward_rules.itertuples(index=False):
        antecedent = parse_itemset(getattr(row, "antecedent"))
        if not antecedent.issubset(ingredients):
            continue

        cuisine = extract_cuisine(getattr(row, "consequent"))
        pmi = float(getattr(row, "pmi"))
        cuisine_scores[cuisine] += pmi
        matches_by_cuisine[cuisine].append(
            {
                "antecedent": sorted(antecedent),
                "consequent": sorted(parse_itemset(getattr(row, "consequent"))),
                "pmi": pmi,
            }
        )

    results: list[dict[str, object]] = []
    for cuisine in all_cuisines:
        result_score = cuisine_scores.get(cuisine, 0.0)
        best_rule = max(matches_by_cuisine[cuisine], key=lambda item: item["pmi"]) if matches_by_cuisine[cuisine] else None
        results.append(
            {
                "cuisine": cuisine,
                "pmi_score": result_score,
                "matched_rules": len(matches_by_cuisine[cuisine]),
                "best_rule": best_rule,
            }
        )

    peak = max((item["pmi_score"] for item in results), default=0.0)
    exp_scores = [exp(item["pmi_score"] - peak) for item in results]
    total = sum(exp_scores)
    for item, exp_score in zip(results, exp_scores, strict=False):
        item["confidence"] = exp_score / total if total else 0.0

    results.sort(key=lambda item: (item["pmi_score"], item["matched_rules"]), reverse=True)
    return results


def predict_cuisine(
    ingredient_text: str,
    rules_path: Path | None = None,
    ingredients_path: Path | None = None,
    top_n: int = 3,
) -> dict[str, object]:
    rules_path = rules_path or default_rules_path()
    ingredients_path = ingredients_path or default_ingredients_path()

    normalized_ingredients, resolved, unmatched = normalize_user_ingredients(ingredient_text, ingredients_path)
    rules = load_rules(rules_path)
    if not normalized_ingredients:
        return {
            "normalized_ingredients": normalized_ingredients,
            "resolved_ingredients": resolved,
            "unmatched_ingredients": unmatched,
            "predictions": [],
            "all_predictions": [],
        }

    predictions = score_cuisines(set(normalized_ingredients), rules)

    return {
        "normalized_ingredients": normalized_ingredients,
        "resolved_ingredients": resolved,
        "unmatched_ingredients": unmatched,
        "predictions": predictions[:top_n],
        "all_predictions": predictions,
    }


def format_bar(score: float, width: int = 20) -> str:
    filled = max(0, min(width, round(score * width)))
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def print_prediction_block(result: dict[str, object], ingredients_path: Path) -> None:
    normalized_ingredients = result["normalized_ingredients"]
    resolved_ingredients = result["resolved_ingredients"]
    unmatched_ingredients = result["unmatched_ingredients"]
    predictions = result["predictions"]
    all_predictions = result["all_predictions"]

    print()
    if normalized_ingredients:
        print("Your ingredients: " + ", ".join(normalized_ingredients))
    else:
        print("(no ingredients yet)")

    if unmatched_ingredients:
        print()
        index = load_ingredient_index(ingredients_path)
        for token in unmatched_ingredients:
            suggestions = suggest_ingredients(token, index)
            print(f"'{token}' not found.")
            if suggestions:
                print("Did you mean:")
                for suggestion in suggestions[:5]:
                    print(f"- {suggestion['canonical']} ({suggestion['score'] * 100:.0f}%, matched '{suggestion['matched']}')")

    print()
    print("Cuisine matches:")
    if not all_predictions:
        print("No matching cuisine rules yet.")
        return

    for prediction in predictions:
        cuisine = prediction["cuisine"]
        score = prediction["confidence"]
        bar = format_bar(score)
        print(f"\n{cuisine.title()} {bar} {score * 100:.1f}%")
        print(f"PMI score: {prediction['pmi_score']:.2f}")

        best_rule = prediction["best_rule"]
        if best_rule is not None:
            print(f"- {', '.join(best_rule['antecedent'])} -> {', '.join(best_rule['consequent'])} (PMI: {best_rule['pmi']:.2f})")

    tail = [item for item in all_predictions if item not in predictions]
    if tail:
        print("\nOther cuisines:")
        for prediction in tail[:5]:
            print(f"- {prediction['cuisine'].title()}: {prediction['confidence'] * 100:.1f}%")


def apply_ingredient_line(line: str, selected: list[str], history: list[str], ingredients_path: Path) -> None:
    index = load_ingredient_index(ingredients_path)
    for raw_token in line.split(","):
        token = raw_token.strip()
        if not token:
            continue
        match = resolve_ingredient(token, index)
        if match is None:
            print(f"'{token}' not found.")
            suggestions = suggest_ingredients(token, index)
            if suggestions:
                print("Did you mean:")
                for suggestion in suggestions:
                    print(f"- {suggestion['canonical']} ({suggestion['score'] * 100:.0f}%, matched '{suggestion['matched']}')")
            continue

        canonical = match["canonical"]
        if canonical in selected:
            print(f"Kept: {canonical} (already selected)")
            continue

        selected.append(canonical)
        history.append(canonical)
        print(f"Added: {canonical} (matched '{match['matched']}' at {match['score'] * 100:.0f}%)")


def interactive_session(
    rules_path: Path | None = None,
    ingredients_path: Path | None = None,
    top_n: int = 3,
) -> None:
    rules_path = rules_path or default_rules_path()
    ingredients_path = ingredients_path or default_ingredients_path()
    selected: list[str] = []
    history: list[str] = []

    print("=" * 70)
    print("CUISINE EXPLORER - INTERACTIVE DEMO")
    print("Enter ingredients to see which cuisines match and what patterns they form.")
    print("Commands:")
    print("Type an ingredient name or a comma-separated list")
    print("Type 'undo' to remove the last ingredient")
    print("Type 'clear' to start over")
    print("Type 'suggest [name]' to see matching ingredients")
    print("Type 'done', 'quit', or 'exit' to finish")
    print("Example: garlic, olive oil, tomato, basil")

    while True:
        result = predict_cuisine(
            ", ".join(selected),
            rules_path=rules_path,
            ingredients_path=ingredients_path,
            top_n=top_n,
        )
        print_prediction_block(result, ingredients_path)

        line = input("\nAdd ingredient(s): ").strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered in {"done", "quit", "exit"}:
            break
        if lowered == "undo":
            if history:
                removed = history.pop()
                if removed in selected:
                    selected.remove(removed)
                print(f"Removed: {removed}")
            else:
                print("Nothing to undo.")
            continue
        if lowered == "clear":
            selected.clear()
            history.clear()
            print("Cleared.")
            continue
        if lowered.startswith("suggest "):
            query = line.split(" ", 1)[1].strip()
            if not query:
                print("Type a name after suggest.")
                continue
            index = load_ingredient_index(ingredients_path)
            suggestions = suggest_ingredients(query, index)
            if not suggestions:
                print("No close matches found.")
                continue
            print("Suggestions:")
            for suggestion in suggestions:
                print(f"- {suggestion['canonical']} ({suggestion['score'] * 100:.0f}%, matched '{suggestion['matched']}')")
            continue

        apply_ingredient_line(line, selected, history, ingredients_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict cuisine from ingredients or open the interactive explorer.")
    parser.add_argument("--ingredients", type=str, help="Comma-separated ingredient list")
    parser.add_argument("--rules", type=Path, default=default_rules_path(), help="Association rules CSV")
    parser.add_argument("--ingredient-map", type=Path, default=default_ingredients_path(), help="ingredients.json path")
    parser.add_argument("--top-n", type=int, default=3, help="Number of cuisine predictions to show")
    parser.add_argument("--interactive", action="store_true", help="Open the interactive explorer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interactive or not args.ingredients:
        interactive_session(rules_path=args.rules, ingredients_path=args.ingredient_map, top_n=args.top_n)
        return

    result = predict_cuisine(
        ingredient_text=args.ingredients,
        rules_path=args.rules,
        ingredients_path=args.ingredient_map,
        top_n=args.top_n,
    )
    print_prediction_block(result, args.ingredient_map)


if __name__ == "__main__":
    main()
