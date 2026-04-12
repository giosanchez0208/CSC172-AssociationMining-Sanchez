from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from math import exp, log1p, sqrt
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


def default_data_path() -> Path:
    return project_root() / "data" / "processed" / "prepared_recipes_cleaned_balanced.csv"


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


def load_cuisine_profile(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, usecols=["cuisine", "ingredients"])
    cuisine_totals: dict[str, int] = defaultdict(int)
    ingredient_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    vocabulary: set[str] = set()

    for row in frame.itertuples(index=False):
        cuisine = normalize_text(getattr(row, "cuisine"))
        cuisine_totals[cuisine] += 1
        items = {normalize_text(part) for part in str(getattr(row, "ingredients")).split(",") if normalize_text(part)}
        vocabulary.update(items)
        for item in items:
            ingredient_counts[cuisine][item] += 1

    return {
        "cuisine_totals": dict(cuisine_totals),
        "ingredient_counts": {cuisine: dict(counts) for cuisine, counts in ingredient_counts.items()},
        "vocabulary_size": len(vocabulary),
    }


def parse_itemset(value: str) -> set[str]:
    return {part.strip().lower() for part in str(value).split(",") if part.strip()}


def extract_cuisine(value: str) -> str:
    for item in parse_itemset(value):
        if item.startswith(CUISINE_PREFIX):
            return item.removeprefix(CUISINE_PREFIX)
    return ""


def cuisine_signature_rules(cuisine: str, rules: pd.DataFrame, limit: int = 2) -> list[dict[str, object]]:
    signature_rules = rules[rules["rule_type"] == "cuisine_to_ingredient"].copy()
    rows: list[dict[str, object]] = []

    for row in signature_rules.itertuples(index=False):
        if extract_cuisine(getattr(row, "antecedent")) != cuisine:
            continue
        rows.append(
            {
                "ingredients": sorted(parse_itemset(getattr(row, "consequent"))),
                "confidence": float(getattr(row, "confidence")),
                "lift": float(getattr(row, "lift")),
                "support": float(getattr(row, "support")),
            }
        )

    rows.sort(key=lambda item: (item["confidence"] * item["lift"], item["confidence"]), reverse=True)
    return rows[:limit]


def signature_evidence(cuisine: str, ingredients: set[str], rules: pd.DataFrame, limit: int = 2) -> list[dict[str, object]]:
    signature_rules = rules[rules["rule_type"] == "cuisine_to_ingredient"].copy()
    rows: list[dict[str, object]] = []

    for row in signature_rules.itertuples(index=False):
        if extract_cuisine(getattr(row, "antecedent")) != cuisine:
            continue
        consequent = parse_itemset(getattr(row, "consequent"))
        overlap = sorted(consequent & ingredients)
        if not overlap:
            continue
        rows.append(
            {
                "ingredients": overlap,
                "confidence": float(getattr(row, "confidence")),
                "lift": float(getattr(row, "lift")),
                "support": float(getattr(row, "support")),
            }
        )

    rows.sort(key=lambda item: (item["confidence"] * item["lift"], item["confidence"]), reverse=True)
    return rows[:limit]


def score_cuisines(
    ingredients: set[str],
    rules: pd.DataFrame,
    profile: dict[str, object],
    top_n: int = 3,
) -> list[dict[str, object]]:
    ingredient_rules = rules[rules["rule_type"] == "ingredient_to_cuisine"].copy()
    cuisine_inventory_weight: dict[str, float] = defaultdict(float)
    matched_weight: dict[str, float] = defaultdict(float)
    matches_by_cuisine: dict[str, list[dict[str, object]]] = defaultdict(list)
    cuisine_totals: dict[str, int] = profile["cuisine_totals"]
    ingredient_counts: dict[str, dict[str, int]] = profile["ingredient_counts"]
    vocabulary_size: int = profile["vocabulary_size"]

    for row in ingredient_rules.itertuples(index=False):
        cuisine = extract_cuisine(getattr(row, "consequent"))
        cuisine_inventory_weight[cuisine] += (
            float(getattr(row, "support"))
            * float(getattr(row, "confidence"))
            * log1p(float(getattr(row, "lift")))
        )

    for row in ingredient_rules.itertuples(index=False):
        antecedent = parse_itemset(getattr(row, "antecedent"))
        if not antecedent.issubset(ingredients):
            continue

        cuisine = extract_cuisine(getattr(row, "consequent"))
        support = float(getattr(row, "support"))
        confidence = float(getattr(row, "confidence"))
        lift = float(getattr(row, "lift"))
        weight = support * confidence * log1p(lift)
        matched_weight[cuisine] += weight
        matches_by_cuisine[cuisine].append(
            {
                "antecedent": sorted(antecedent),
                "consequent": sorted(parse_itemset(getattr(row, "consequent"))),
                "rule_type": "ingredient_to_cuisine",
                "confidence": confidence,
                "lift": lift,
                "support": support,
                "weight": weight,
            }
        )

    all_cuisines = sorted(cuisine_totals.keys())
    profile_scores: dict[str, float] = {}
    for cuisine in all_cuisines:
        total = cuisine_totals.get(cuisine, 0)
        counts = ingredient_counts.get(cuisine, {})
        score = 0.0
        for ingredient in ingredients:
            score += log1p(counts.get(ingredient, 0) + 1) - log1p(total + vocabulary_size)
        profile_scores[cuisine] = score

    raw_scores: dict[str, float] = {}
    for cuisine in all_cuisines:
        weight = matched_weight.get(cuisine, 0.0)
        inventory = cuisine_inventory_weight.get(cuisine, 0.0)
        rule_signal = weight / sqrt(inventory) if inventory > 0 else 0.0
        raw_scores[cuisine] = profile_scores[cuisine] + (1.25 * rule_signal)

    peak = max(raw_scores.values())
    softmax_scores = {cuisine: exp(score - peak) for cuisine, score in raw_scores.items()}
    total = sum(softmax_scores.values())

    results: list[dict[str, object]] = []
    for cuisine in all_cuisines:
        matched = matched_weight.get(cuisine, 0.0)
        best_rule = max(
            matches_by_cuisine[cuisine],
            key=lambda item: (item["weight"], item["confidence"], item["lift"]),
        ) if matches_by_cuisine[cuisine] else None
        results.append(
            {
                "cuisine": cuisine,
                "confidence": softmax_scores[cuisine] / total if total else 0.0,
                "matched_rules": len(matches_by_cuisine[cuisine]),
                "matched_weight": matched,
                "profile_score": profile_scores[cuisine],
                "inventory_weight": cuisine_inventory_weight.get(cuisine, 0.0),
                "best_rule": best_rule,
                "signature_rules": cuisine_signature_rules(cuisine, rules, limit=2),
            }
        )

    results.sort(key=lambda item: (item["confidence"], item["matched_weight"]), reverse=True)
    return results


def predict_cuisine(
    ingredient_text: str,
    rules_path: Path | None = None,
    ingredients_path: Path | None = None,
    data_path: Path | None = None,
    top_n: int = 3,
) -> dict[str, object]:
    rules_path = rules_path or default_rules_path()
    ingredients_path = ingredients_path or default_ingredients_path()
    data_path = data_path or default_data_path()

    normalized_ingredients, resolved, unmatched = normalize_user_ingredients(ingredient_text, ingredients_path)
    rules = load_rules(rules_path)
    profile = load_cuisine_profile(data_path)
    if not normalized_ingredients:
        return {
            "normalized_ingredients": normalized_ingredients,
            "resolved_ingredients": resolved,
            "unmatched_ingredients": unmatched,
            "predictions": [],
            "all_predictions": [],
        }

    predictions = score_cuisines(set(normalized_ingredients), rules, profile, top_n=top_n)

    for prediction in predictions:
        prediction["matched_signature_rules"] = signature_evidence(
            prediction["cuisine"], set(normalized_ingredients), rules
        )

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

        best_rule = prediction["best_rule"]
        if best_rule is not None:
            print(f"Matched rules for {cuisine}:")
            print(
                f"- {', '.join(best_rule['antecedent'])} -> {', '.join(best_rule['consequent'])} "
                f"(conf: {best_rule['confidence']:.2f}, lift: {best_rule['lift']:.2f})"
            )

        matched_signatures = prediction["matched_signature_rules"]
        if matched_signatures:
            print(f"Signature overlap for {cuisine}:")
            for signature in matched_signatures:
                print(f"- {', '.join(signature['ingredients'])} (lift: {signature['lift']:.2f})")
        else:
            print(f"Typical ingredients for {cuisine}:")
            for signature in prediction["signature_rules"]:
                print(f"- {', '.join(signature['ingredients'])} (lift: {signature['lift']:.2f})")

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
    data_path: Path | None = None,
    top_n: int = 3,
) -> None:
    rules_path = rules_path or default_rules_path()
    ingredients_path = ingredients_path or default_ingredients_path()
    data_path = data_path or default_data_path()
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
            data_path=data_path,
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
    parser.add_argument("--data", type=Path, default=default_data_path(), help="Balanced recipe CSV")
    parser.add_argument("--top-n", type=int, default=3, help="Number of cuisine predictions to show")
    parser.add_argument("--interactive", action="store_true", help="Open the interactive explorer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interactive or not args.ingredients:
        interactive_session(
            rules_path=args.rules,
            ingredients_path=args.ingredient_map,
            data_path=args.data,
            top_n=args.top_n,
        )
        return

    result = predict_cuisine(
        ingredient_text=args.ingredients,
        rules_path=args.rules,
        ingredients_path=args.ingredient_map,
        data_path=args.data,
        top_n=args.top_n,
    )
    print_prediction_block(result, args.ingredient_map)


if __name__ == "__main__":
    main()
