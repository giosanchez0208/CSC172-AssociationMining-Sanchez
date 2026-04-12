from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from math import exp, log1p, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CUISINE_PREFIX = "cuisine:"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_rules_path() -> Path:
    return project_root() / "data" / "mined" / "association_rules.csv"


def default_data_path() -> Path:
    return project_root() / "data" / "processed" / "prepared_recipes_cleaned_balanced.csv"


def default_output_dir() -> Path:
    return project_root() / "results" / "analysis"


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def parse_itemset(value: str) -> set[str]:
    return {normalize_text(part) for part in str(value).split(",") if normalize_text(part)}


def extract_cuisine(value: str) -> str:
    for item in parse_itemset(value):
        if item.startswith(CUISINE_PREFIX):
            return item.removeprefix(CUISINE_PREFIX)
    return ""


def load_baskets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["ingredient_set"] = frame["ingredients"].map(parse_itemset)
    frame["basket_size"] = frame["ingredient_set"].map(len)
    return frame


def load_rules(path: Path) -> pd.DataFrame:
    rules = pd.read_csv(path)
    required = {"antecedent", "consequent", "support", "confidence", "lift", "rule_type"}
    missing = required - set(rules.columns)
    if missing:
        raise ValueError(f"Missing columns in rules CSV: {sorted(missing)}")
    return rules


def rule_weight(row: pd.Series) -> float:
    return float(row["support"]) * float(row["confidence"]) * log1p(float(row["lift"]))


@dataclass(frozen=True)
class CompiledRules:
    ingredient_rules: list[dict[str, object]]
    signature_rules: list[dict[str, object]]
    ingredient_inventory: dict[str, float]
    signature_inventory: dict[str, float]


def compile_rules(rules: pd.DataFrame) -> CompiledRules:
    ingredient_rules: list[dict[str, object]] = []
    signature_rules: list[dict[str, object]] = []
    ingredient_inventory: dict[str, float] = defaultdict(float)
    signature_inventory: dict[str, float] = defaultdict(float)

    for _, row in rules[rules["rule_type"] == "ingredient_to_cuisine"].iterrows():
        cuisine = extract_cuisine(row["consequent"])
        weight = rule_weight(row)
        ingredient_inventory[cuisine] += weight
        ingredient_rules.append(
            {
                "antecedent": parse_itemset(row["antecedent"]),
                "cuisine": cuisine,
                "weight": weight,
            }
        )

    for _, row in rules[rules["rule_type"] == "cuisine_to_ingredient"].iterrows():
        cuisine = extract_cuisine(row["antecedent"])
        weight = rule_weight(row)
        signature_inventory[cuisine] += weight
        signature_rules.append(
            {
                "consequent": parse_itemset(row["consequent"]),
                "cuisine": cuisine,
                "weight": weight,
            }
        )

    return CompiledRules(
        ingredient_rules=ingredient_rules,
        signature_rules=signature_rules,
        ingredient_inventory=dict(ingredient_inventory),
        signature_inventory=dict(signature_inventory),
    )


def score_cuisines(ingredients: set[str], compiled: CompiledRules, signature_weight: float) -> dict[str, float]:
    inventory = {
        cuisine: compiled.ingredient_inventory.get(cuisine, 0.0)
        + signature_weight * compiled.signature_inventory.get(cuisine, 0.0)
        for cuisine in set(compiled.ingredient_inventory) | set(compiled.signature_inventory)
    }
    matched: dict[str, float] = defaultdict(float)

    for rule in compiled.ingredient_rules:
        if not rule["antecedent"].issubset(ingredients):
            continue
        matched[rule["cuisine"]] += rule["weight"]

    for rule in compiled.signature_rules:
        overlap = ingredients & rule["consequent"]
        if not overlap:
            continue
        overlap_fraction = len(overlap) / len(rule["consequent"])
        matched[rule["cuisine"]] += signature_weight * rule["weight"] * overlap_fraction

    if not matched:
        return {}

    raw_scores = {
        cuisine: matched_weight / sqrt(max(inventory.get(cuisine, 0.0), 1e-12))
        for cuisine, matched_weight in matched.items()
    }
    peak = max(raw_scores.values())
    exp_scores = {cuisine: exp(score - peak) for cuisine, score in raw_scores.items()}
    total = sum(exp_scores.values())
    return {cuisine: score / total for cuisine, score in exp_scores.items()}


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    basket_size: int
    top1_accuracy: float
    top3_accuracy: float
    coverage: float
    mean_confidence: float


def evaluate_strategy(
    baskets: pd.DataFrame,
    compiled: CompiledRules,
    basket_size: int,
    signature_weight: float,
    repeats: int,
    sample_size: int,
    rng: np.random.Generator,
) -> StrategyResult:
    eligible = baskets[baskets["basket_size"] >= basket_size]
    if eligible.empty:
        return StrategyResult(f"signature_weight={signature_weight:g}", basket_size, 0.0, 0.0, 0.0, 0.0)

    if len(eligible) > sample_size:
        eligible = eligible.sample(n=sample_size, random_state=42 + basket_size)

    total = 0
    top1_hits = 0
    top3_hits = 0
    coverage_hits = 0
    confidences: list[float] = []

    rows = list(eligible.itertuples(index=False))
    for _ in range(repeats):
        for row in rows:
            ingredients = sorted(row.ingredient_set)
            sampled = set(rng.choice(ingredients, size=basket_size, replace=False))
            scores = score_cuisines(sampled, compiled, signature_weight=signature_weight)
            if scores:
                coverage_hits += 1
                ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                confidences.append(ordered[0][1])
                total += 1
                if ordered[0][0] == row.cuisine:
                    top1_hits += 1
                if row.cuisine in {cuisine for cuisine, _ in ordered[:3]}:
                    top3_hits += 1

    return StrategyResult(
        strategy=f"signature_weight={signature_weight:g}",
        basket_size=basket_size,
        top1_accuracy=top1_hits / total if total else 0.0,
        top3_accuracy=top3_hits / total if total else 0.0,
        coverage=coverage_hits / (len(rows) * repeats) if rows else 0.0,
        mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
    )


def build_summary_table(results: list[StrategyResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": result.strategy,
                "basket_size": result.basket_size,
                "top1_accuracy": result.top1_accuracy,
                "top3_accuracy": result.top3_accuracy,
                "coverage": result.coverage,
                "mean_confidence": result.mean_confidence,
            }
            for result in results
        ]
    )


def plot_results(summary: pd.DataFrame, output_path: Path) -> None:
    width = 1000
    height = 620
    margin_left = 80
    margin_right = 40
    margin_top = 50
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_values = sorted(summary["basket_size"].unique())
    strategies = list(summary["strategy"].unique())
    colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c"]

    def x_coord(value: int) -> float:
        if len(x_values) == 1:
            return margin_left + plot_width / 2
        return margin_left + (value - x_values[0]) / (x_values[-1] - x_values[0]) * plot_width

    def y_coord(value: float) -> float:
        return margin_top + (1 - value) * plot_height

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<style>text{font-family:Arial,sans-serif;font-size:14px;fill:#111827} .small{font-size:12px} .axis{stroke:#111827;stroke-width:1.5} .grid{stroke:#d1d5db;stroke-width:1} .legend{font-size:13px} </style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-weight="700">Cuisine prediction accuracy by basket size</text>',
    ]

    for tick in np.linspace(0, 1, 6):
        y = y_coord(float(tick))
        elements.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" />')
        elements.append(f'<text class="small" x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick:.1f}</text>')

    for size in x_values:
        x = x_coord(int(size))
        elements.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" />')
        elements.append(f'<text class="small" x="{x:.1f}" y="{height - margin_bottom + 20}" text-anchor="middle">{int(size)}</text>')

    elements.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" />')
    elements.append(f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" />')

    for index, strategy in enumerate(strategies):
        group = summary[summary["strategy"] == strategy].sort_values("basket_size")
        color = colors[index % len(colors)]
        for metric, dash in [("top1_accuracy", None), ("top3_accuracy", "6,4")]:
            points = " ".join(
                f"{x_coord(int(row.basket_size)):.1f},{y_coord(float(getattr(row, metric))):.1f}"
                for row in group.itertuples(index=False)
            )
            elements.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" '
                f'{"stroke-dasharray=\"%s\" " % dash if dash else ""}points="{points}" />'
            )

        legend_y = margin_top + 10 + index * 22
        elements.append(f'<rect x="{width - margin_right - 170}" y="{legend_y - 12}" width="14" height="3" fill="{color}" />')
        elements.append(f'<text class="legend" x="{width - margin_right - 150}" y="{legend_y}">{strategy}</text>')

    elements.append(f'<text class="small" x="{width/2}" y="{height - 20}" text-anchor="middle">Observed basket size</text>')
    elements.append(f'<text class="small" transform="translate(24,{height/2}) rotate(-90)" text-anchor="middle">Accuracy</text>')
    elements.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cuisine scoring strategies.")
    parser.add_argument("--rules", type=Path, default=default_rules_path(), help="Association rules CSV")
    parser.add_argument("--data", type=Path, default=default_data_path(), help="Balanced recipe CSV")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Directory for analysis artifacts")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Basket sizes to evaluate")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per basket size")
    parser.add_argument("--sample-size", type=int, default=300, help="Max recipes sampled per basket size")
    parser.add_argument(
        "--signature-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.25],
        help="Signature rule weights to compare",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baskets = load_baskets(args.data)
    rules = load_rules(args.rules)
    compiled = compile_rules(rules)
    rng = np.random.default_rng(42)

    results: list[StrategyResult] = []
    for signature_weight in args.signature_weights:
        for basket_size in args.sizes:
            results.append(
                evaluate_strategy(
                    baskets=baskets,
                    compiled=compiled,
                    basket_size=basket_size,
                    signature_weight=signature_weight,
                    repeats=args.repeats,
                    sample_size=args.sample_size,
                    rng=rng,
                )
            )

    summary = build_summary_table(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "strategy_summary.csv"
    figure_path = args.output_dir / "basket_size_accuracy.svg"
    summary.to_csv(summary_path, index=False)
    plot_results(summary, figure_path)

    print(summary.sort_values(["strategy", "basket_size"]).to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
