from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import pandas as pd

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


def parse_items(value: str) -> Set[str]:
    return {normalize_item(part) for part in value.split(",") if normalize_item(part)}


def load_transactions(csv_path: Path) -> tuple[list[Set[str]], pd.DataFrame]:
    frame = pd.read_csv(csv_path)
    if "ingredients" not in frame.columns or "cuisine" not in frame.columns:
        raise ValueError("Expected columns: cuisine and ingredients")

    cleaned_rows = []
    transactions: list[Set[str]] = []

    for row in frame.itertuples(index=False):
        items = parse_items(getattr(row, "ingredients"))
        items = {item for item in items if item not in FORBIDDEN_INGREDIENTS}
        if not items:
            continue

        cleaned_rows.append(
            {
                "recipe_id": getattr(row, "recipe_id", None),
                "cuisine": normalize_item(getattr(row, "cuisine")),
                "ingredients": ",".join(sorted(items)),
                "ingredient_count": len(items),
            }
        )
        transactions.append(items | {f"{CUISINE_PREFIX}{normalize_item(getattr(row, 'cuisine'))}"})

    return transactions, pd.DataFrame(cleaned_rows)


def mine_frequent_itemsets(
    transactions: Sequence[Set[str]],
    min_support: float,
    max_itemset_size: int,
) -> Dict[frozenset[str], int]:
    total = len(transactions)
    if total == 0:
        raise ValueError("No transactions available after filtering")
    if max_itemset_size < 1:
        raise ValueError("max_itemset_size must be at least 1")
    if max_itemset_size > 3:
        max_itemset_size = 3

    frequent_itemsets: Dict[frozenset[str], int] = {}

    single_counts = Counter()
    for transaction in transactions:
        single_counts.update(transaction)

    frequent_singles = {
        item for item, count in single_counts.items() if count / total >= min_support
    }
    level_one = {
        frozenset([item]): count
        for item, count in single_counts.items()
        if item in frequent_singles
    }
    frequent_itemsets.update(level_one)

    if max_itemset_size == 1:
        return frequent_itemsets

    pair_counts = Counter()
    for transaction in transactions:
        filtered_items = sorted(item for item in transaction if item in frequent_singles)
        if len(filtered_items) < 2:
            continue
        pair_counts.update(frozenset(pair) for pair in combinations(filtered_items, 2))

    level_two = {
        itemset: count
        for itemset, count in pair_counts.items()
        if count / total >= min_support
    }
    frequent_itemsets.update(level_two)

    if max_itemset_size == 2:
        return frequent_itemsets

    frequent_pairs = set(level_two.keys())
    triple_counts = Counter()
    for transaction in transactions:
        filtered_items = sorted(item for item in transaction if item in frequent_singles)
        if len(filtered_items) < 3:
            continue
        for triple in combinations(filtered_items, 3):
            triple_itemset = frozenset(triple)
            if all(frozenset(pair) in frequent_pairs for pair in combinations(triple, 2)):
                triple_counts[triple_itemset] += 1

    level_three = {
        itemset: count
        for itemset, count in triple_counts.items()
        if count / total >= min_support
    }
    frequent_itemsets.update(level_three)

    return frequent_itemsets


def rule_type_for(antecedent: frozenset[str], consequent: frozenset[str]) -> str | None:
    antecedent_has_cuisine = any(item.startswith(CUISINE_PREFIX) for item in antecedent)
    consequent_has_cuisine = any(item.startswith(CUISINE_PREFIX) for item in consequent)

    if not antecedent_has_cuisine and not consequent_has_cuisine:
        return "ingredient_to_ingredient"
    if not antecedent_has_cuisine and consequent_has_cuisine and len(consequent) == 1:
        return "ingredient_to_cuisine"
    if antecedent_has_cuisine and len(antecedent) == 1 and not consequent_has_cuisine:
        return "cuisine_to_ingredient"
    return None


def extract_cuisine(antecedent: frozenset[str], consequent: frozenset[str]) -> str:
    for item in antecedent | consequent:
        if item.startswith(CUISINE_PREFIX):
            return item.removeprefix(CUISINE_PREFIX)
    return ""


def format_itemset(itemset: frozenset[str]) -> str:
    return ", ".join(sorted(itemset))


def mine_rules(
    frequent_itemsets: Dict[frozenset[str], int],
    total_transactions: int,
    min_confidence: float,
    min_lift: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for itemset, itemset_count in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        for size in range(1, len(itemset)):
            for antecedent_tuple in combinations(sorted(itemset), size):
                antecedent = frozenset(antecedent_tuple)
                consequent = frozenset(itemset - antecedent)
                if not consequent:
                    continue

                rule_type = rule_type_for(antecedent, consequent)
                if rule_type is None:
                    continue

                antecedent_support = frequent_itemsets.get(antecedent)
                consequent_support = frequent_itemsets.get(consequent)
                if antecedent_support is None or consequent_support is None:
                    continue

                support = itemset_count / total_transactions
                confidence = itemset_count / antecedent_support
                lift = confidence / (consequent_support / total_transactions)

                if confidence < min_confidence or lift < min_lift:
                    continue

                rows.append(
                    {
                        "antecedent": format_itemset(antecedent),
                        "consequent": format_itemset(consequent),
                        "support": support,
                        "confidence": confidence,
                        "lift": lift,
                        "antecedent_size": len(antecedent),
                        "consequent_size": len(consequent),
                        "antecedent_set": repr(antecedent),
                        "consequent_set": repr(consequent),
                        "rule_type": rule_type,
                        "cuisine_involved": extract_cuisine(antecedent, consequent),
                    }
                )

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules

    rules = rules.sort_values(
        by=["rule_type", "confidence", "lift", "support", "antecedent", "consequent"],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)
    return rules


def build_rules(
    input_path: Path,
    output_path: Path,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_itemset_size: int,
) -> pd.DataFrame:
    transactions, cleaned_frame = load_transactions(input_path)
    frequent_itemsets = mine_frequent_itemsets(transactions, min_support, max_itemset_size)
    rules = mine_rules(frequent_itemsets, len(transactions), min_confidence, min_lift)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rules.to_csv(output_path, index=False)

    cleaned_path = input_path.with_name("prepared_recipes_cleaned_balanced_filtered.csv")
    cleaned_frame.to_csv(cleaned_path, index=False)

    return rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine cuisine association rules from balanced recipe data.")
    parser.add_argument("--input", type=Path, default=default_input_path(), help="Balanced recipe CSV")
    parser.add_argument("--output", type=Path, default=default_output_path(), help="Output rules CSV")
    parser.add_argument("--support", type=float, default=0.01, help="Minimum support threshold")
    parser.add_argument("--confidence", type=float, default=0.30, help="Minimum confidence threshold")
    parser.add_argument("--lift", type=float, default=1.20, help="Minimum lift threshold")
    parser.add_argument("--max-itemset-size", type=int, default=3, help="Maximum mined itemset size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = build_rules(
        input_path=args.input,
        output_path=args.output,
        min_support=args.support,
        min_confidence=args.confidence,
        min_lift=args.lift,
        max_itemset_size=args.max_itemset_size,
    )

    if rules.empty:
        print("No rules met the thresholds.")
        return

    counts = rules["rule_type"].value_counts().to_dict()
    print(f"Saved {len(rules)} rules to {args.output}")
    print(f"Rule counts: {counts}")


if __name__ == "__main__":
    main()
