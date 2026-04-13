from __future__ import annotations

import argparse
from collections import defaultdict
from math import log2
from pathlib import Path
from typing import Iterable

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

FORBIDDEN_INGREDIENTS = {"twist", "filling"}
CUISINE_PREFIX = "cuisine:"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_input_path() -> Path:
    return project_root() / "data" / "processed" / "prepared_recipes_cleaned_balanced.csv"


def default_output_path() -> Path:
    return project_root() / "data" / "mined" / "association_rules.csv"


def normalize_item(value: str) -> str:
    return " ".join(value.strip().lower().split())


def parse_items(value: str) -> set[str]:
    return {normalize_item(part) for part in str(value).split(",") if normalize_item(part)}


def cuisine_token(cuisine: str) -> str:
    return f"{CUISINE_PREFIX}{normalize_item(cuisine)}"


def load_balanced_dataset(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "cuisine" not in frame.columns or "ingredients" not in frame.columns:
        raise ValueError("Expected columns: cuisine and ingredients")

    cleaned_rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        ingredients = {item for item in parse_items(getattr(row, "ingredients")) if item not in FORBIDDEN_INGREDIENTS}
        if not ingredients:
            continue
        cleaned_rows.append(
            {
                "recipe_id": getattr(row, "recipe_id", None),
                "cuisine": normalize_item(getattr(row, "cuisine")),
                "ingredients": ",".join(sorted(ingredients)),
                "ingredient_count": len(ingredients),
            }
        )

    return pd.DataFrame(cleaned_rows)


def build_transactions(items: Iterable[Iterable[str]]) -> list[list[str]]:
    return [sorted(set(transaction)) for transaction in items]


def encode_transactions(transactions: list[list[str]]) -> pd.DataFrame:
    encoder = TransactionEncoder()
    matrix = encoder.fit(transactions).transform(transactions)
    return pd.DataFrame(matrix, columns=encoder.columns_)


def mine_itemsets(transactions: list[list[str]], min_support: float) -> pd.DataFrame:
    encoded = encode_transactions(transactions)
    itemsets = fpgrowth(encoded, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return itemsets
    return itemsets.sort_values(["support", "itemsets"], ascending=[False, True]).reset_index(drop=True)


def itemset_support(itemset: frozenset[str], index: dict[str, set[int]], total: int) -> tuple[int, float]:
    if not itemset:
        return 0, 0.0
    sets = [index.get(item, set()) for item in itemset]
    if not sets:
        return 0, 0.0
    count = len(set.intersection(*sets))
    return count, count / total if total else 0.0


def build_index(transactions: list[set[str]]) -> dict[str, set[int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for idx, transaction in enumerate(transactions):
        for item in transaction:
            index[item].add(idx)
    return index


def rule_pmi(support: float, antecedent_support: float, consequent_support: float) -> float:
    if support <= 0 or antecedent_support <= 0 or consequent_support <= 0:
        return float("nan")
    return log2(support / (antecedent_support * consequent_support))


def annotate_rule(
    antecedent: frozenset[str],
    consequent: frozenset[str],
    support: float,
    confidence: float,
    lift: float,
    pmi: float,
    rule_type: str,
    cuisine_involved: str,
    support_scope: str,
) -> dict[str, object]:
    return {
        "antecedent": ", ".join(sorted(antecedent)),
        "consequent": ", ".join(sorted(consequent)),
        "support": support,
        "confidence": confidence,
        "lift": lift,
        "pmi": pmi,
        "antecedent_size": len(antecedent),
        "consequent_size": len(consequent),
        "antecedent_set": repr(antecedent),
        "consequent_set": repr(consequent),
        "rule_type": rule_type,
        "cuisine_involved": cuisine_involved,
        "support_scope": support_scope,
    }


def mine_global_ingredient_rules(
    transactions: list[set[str]],
    min_support: float,
    min_confidence: float,
    min_lift: float,
    min_pmi: float,
) -> pd.DataFrame:
    itemsets = mine_itemsets(build_transactions(transactions), min_support)
    if itemsets.empty:
        return itemsets

    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return rules

    rules = rules[
        ~rules["antecedents"].map(lambda items: any(item.startswith(CUISINE_PREFIX) for item in items))
        & ~rules["consequents"].map(lambda items: any(item.startswith(CUISINE_PREFIX) for item in items))
    ].copy()
    if rules.empty:
        return rules

    rules["pmi"] = rules.apply(lambda row: rule_pmi(row["support"], row["antecedent support"], row["consequent support"]), axis=1)
    rules = rules[(rules["confidence"] >= min_confidence) & (rules["lift"] >= min_lift) & (rules["pmi"] >= min_pmi)].copy()
    if rules.empty:
        return rules

    rows = [
        annotate_rule(
            row["antecedents"],
            row["consequents"],
            float(row["support"]),
            float(row["confidence"]),
            float(row["lift"]),
            float(row["pmi"]),
            "ingredient_to_ingredient",
            "",
            "global",
        )
        for _, row in rules.iterrows()
    ]
    return pd.DataFrame(rows)


def mine_local_cuisine_rules(
    frame: pd.DataFrame,
    local_support: float,
    min_confidence: float,
    min_lift: float,
    min_pmi: float,
    max_itemset_size: int = 3,
) -> pd.DataFrame:
    global_transactions = [parse_items(ingredients) for ingredients in frame["ingredients"]]
    global_index = build_index(global_transactions)
    total_recipes = len(frame)
    cuisines = sorted(frame["cuisine"].unique())
    rows: list[dict[str, object]] = []

    for cuisine in cuisines:
        subset = frame[frame["cuisine"] == cuisine]
        cuisine_transactions = [parse_items(ingredients) for ingredients in subset["ingredients"]]
        itemsets = mine_itemsets(build_transactions(cuisine_transactions), local_support)
        if itemsets.empty:
            continue

        cuisine_name = normalize_item(cuisine)
        cuisine_count = len(subset)
        cuisine_support = cuisine_count / total_recipes if total_recipes else 0.0

        for _, itemset_row in itemsets.iterrows():
            itemset = itemset_row["itemsets"]
            if len(itemset) > max_itemset_size:
                continue

            local_support_value = float(itemset_row["support"])
            local_count = round(local_support_value * cuisine_count)
            if local_count <= 0:
                continue

            global_count, global_support = itemset_support(itemset, global_index, total_recipes)
            if global_count <= 0:
                continue

            support_joint = local_count / total_recipes if total_recipes else 0.0

            confidence_to_cuisine = local_count / global_count
            lift_to_cuisine = confidence_to_cuisine / cuisine_support if cuisine_support > 0 else 0.0
            pmi = rule_pmi(support_joint, global_count / total_recipes if total_recipes else 0.0, cuisine_support)
            if confidence_to_cuisine >= min_confidence and lift_to_cuisine >= min_lift and pmi >= min_pmi:
                rows.append(
                    annotate_rule(
                        itemset,
                        frozenset({cuisine_token(cuisine_name)}),
                        support_joint,
                        confidence_to_cuisine,
                        lift_to_cuisine,
                        pmi,
                        "ingredient_to_cuisine",
                        cuisine_name,
                        f"local:{cuisine_name}",
                    )
                )

            confidence_from_cuisine = local_support_value
            lift_from_cuisine = confidence_from_cuisine / (global_support if global_support > 0 else 1.0)
            if confidence_from_cuisine >= min_confidence and lift_from_cuisine >= min_lift and pmi >= min_pmi:
                rows.append(
                    annotate_rule(
                        frozenset({cuisine_token(cuisine_name)}),
                        itemset,
                        support_joint,
                        confidence_from_cuisine,
                        lift_from_cuisine,
                        pmi,
                        "cuisine_to_ingredient",
                        cuisine_name,
                        f"local:{cuisine_name}",
                    )
                )

    if not rows:
        return pd.DataFrame()

    rules = pd.DataFrame(rows)
    return rules.sort_values(
        by=["rule_type", "confidence", "lift", "support", "antecedent", "consequent"],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)


def build_rules(
    input_path: Path,
    output_path: Path,
    global_support: float,
    local_support: float,
    min_confidence: float,
    min_lift: float,
    min_pmi: float,
) -> pd.DataFrame:
    frame = load_balanced_dataset(input_path)

    ingredient_transactions = [parse_items(ingredients) for ingredients in frame["ingredients"]]
    global_rules = mine_global_ingredient_rules(
        transactions=ingredient_transactions,
        min_support=global_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        min_pmi=min_pmi,
    )

    cuisine_rules = mine_local_cuisine_rules(
        frame=frame,
        local_support=local_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        min_pmi=min_pmi,
    )

    combined = pd.concat([global_rules, cuisine_rules], ignore_index=True)
    combined = combined.sort_values(
        by=["rule_type", "confidence", "lift", "support", "antecedent", "consequent"],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    cleaned_path = input_path.with_name("prepared_recipes_cleaned_balanced_filtered.csv")
    frame.to_csv(cleaned_path, index=False)

    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine cuisine association rules with FP-Growth.")
    parser.add_argument("--input", type=Path, default=default_input_path(), help="Balanced recipe CSV")
    parser.add_argument("--output", type=Path, default=default_output_path(), help="Output rules CSV")
    parser.add_argument("--global-support", type=float, default=0.005, help="Global support threshold for ingredient rules")
    parser.add_argument("--local-support", type=float, default=0.015, help="Per-cuisine support threshold for cuisine rules")
    parser.add_argument("--confidence", type=float, default=0.30, help="Minimum confidence threshold")
    parser.add_argument("--lift", type=float, default=1.10, help="Minimum lift threshold")
    parser.add_argument("--pmi", type=float, default=0.25, help="Minimum PMI threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = build_rules(
        input_path=args.input,
        output_path=args.output,
        global_support=args.global_support,
        local_support=args.local_support,
        min_confidence=args.confidence,
        min_lift=args.lift,
        min_pmi=args.pmi,
    )
    print(f"Saved {len(rules)} rules to {args.output}")
    if not rules.empty:
        print(rules["rule_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
