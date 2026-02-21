from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

NUMERIC = "numeric"
CATEGORICAL = "categorical"
FeatureSpec = Sequence[tuple[str, int]]

DEMOGRAPHIC_SPEC: FeatureSpec = (
    (NUMERIC, 0),
    (CATEGORICAL, 1),
    (CATEGORICAL, 2),
    (CATEGORICAL, 3),
    (CATEGORICAL, 4),
    (NUMERIC, 5),
    (CATEGORICAL, 6),
    (CATEGORICAL, 7),
)

FULL_FEATURE_SPEC: FeatureSpec = (
    (NUMERIC, 0),
    (CATEGORICAL, 1),
    (CATEGORICAL, 2),
    (CATEGORICAL, 3),
    (CATEGORICAL, 4),
    (NUMERIC, 5),
    (CATEGORICAL, 6),
    (CATEGORICAL, 7),
    (CATEGORICAL, 8),
    (NUMERIC, 9),
    (CATEGORICAL, 10),
    (NUMERIC, 11),
    (NUMERIC, 12),
    (NUMERIC, 13),
    (CATEGORICAL, 14),
)

TARGET_COL = 15
EXPECTED_COLS = 16


def parse_k_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k < 2:
            raise ValueError(f"Invalid k={k}; k must be >= 2")
        values.append(k)
    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError("k list cannot be empty")
    return unique_values


def load_dataset(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]

    if not rows:
        raise ValueError("Dataset is empty")

    for idx, row in enumerate(rows, start=1):
        if len(row) != EXPECTED_COLS:
            raise ValueError(
                f"Row {idx} has {len(row)} columns; expected {EXPECTED_COLS}"
            )

    return rows


def minmax_scale(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def build_numeric_stats(rows: Sequence[Sequence[str]], cols: Sequence[int]) -> dict[int, tuple[float, float]]:
    stats: dict[int, tuple[float, float]] = {}
    for col in cols:
        values = [float(row[col]) for row in rows]
        stats[col] = (min(values), max(values))
    return stats


def build_category_maps(rows: Sequence[Sequence[str]], cols: Sequence[int]) -> dict[int, dict[str, int]]:
    maps: dict[int, dict[str, int]] = {}
    for col in cols:
        categories = sorted({row[col] for row in rows})
        maps[col] = {category: idx for idx, category in enumerate(categories)}
    return maps


def encode_matrix(rows: Sequence[Sequence[str]], spec: FeatureSpec) -> np.ndarray:
    numeric_cols = [col for kind, col in spec if kind == NUMERIC]
    categorical_cols = [col for kind, col in spec if kind == CATEGORICAL]

    numeric_stats = build_numeric_stats(rows, numeric_cols)
    category_maps = build_category_maps(rows, categorical_cols)

    matrix: list[list[float]] = []
    for row in rows:
        features: list[float] = []
        for kind, col in spec:
            if kind == NUMERIC:
                min_value, max_value = numeric_stats[col]
                features.append(minmax_scale(float(row[col]), min_value, max_value))
            else:
                one_hot = [0.0] * len(category_maps[col])
                one_hot[category_maps[col][row[col]]] = 1.0
                features.extend(one_hot)
        matrix.append(features)

    return np.array(matrix, dtype=float)


def encode_target(rows: Sequence[Sequence[str]], col: int) -> tuple[np.ndarray, dict[str, int]]:
    classes = sorted({row[col] for row in rows})
    class_map = {label: idx for idx, label in enumerate(classes)}
    encoded = np.array([class_map[row[col]] for row in rows], dtype=int)
    return encoded, class_map


def run_kmeans(X: np.ndarray, k_values: Sequence[int], seed: int) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        model.fit(X)
        inertia = model.inertia_
        n_iter = model.n_iter_
        if inertia is None or n_iter is None:
            raise RuntimeError("KMeans did not expose inertia or iteration count")
        results[str(k)] = {
            "inertia": float(inertia),
            "n_iter": float(n_iter),
        }
    return results


def evaluate_models(X: np.ndarray, y: np.ndarray, cv: int, seed: int) -> dict[str, dict[str, float]]:
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=seed,
        ),
    }

    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=splitter)
        metrics[name] = {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
        }
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bank dataset preprocessing, clustering, and model evaluation"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("dataset.tsv"),
        help="Path to input TSV dataset (default: dataset.tsv in project root)",
    )
    parser.add_argument(
        "--k-list",
        type=str,
        default="3,4,5,6",
        help="Comma-separated k values for KMeans (default: 3,4,5,6)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Cross-validation folds for model evaluation (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save JSON results",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cv < 2:
        raise ValueError("--cv must be >= 2")

    k_values = parse_k_list(args.k_list)
    rows = load_dataset(args.data)

    X_demographic = encode_matrix(rows, DEMOGRAPHIC_SPEC)
    X_full = encode_matrix(rows, FULL_FEATURE_SPEC)
    y, class_map = encode_target(rows, TARGET_COL)

    kmeans_results = run_kmeans(X_demographic, k_values, args.seed)
    model_results = evaluate_models(X_full, y, args.cv, args.seed)

    report = {
        "data_path": str(args.data),
        "num_rows": len(rows),
        "demographic_feature_dim": int(X_demographic.shape[1]),
        "full_feature_dim": int(X_full.shape[1]),
        "target_classes": class_map,
        "kmeans": kmeans_results,
        "models": model_results,
    }

    print("=== Dataset Summary ===")
    print(f"rows: {len(rows)}")
    print(f"demographic feature dim: {int(X_demographic.shape[1])}")
    print(f"full feature dim: {int(X_full.shape[1])}")
    print(f"target classes: {class_map}")

    print("\n=== KMeans (inertia) ===")
    for k in sorted(kmeans_results, key=int):
        item = kmeans_results[k]
        print(f"k={k}: inertia={item['inertia']:.4f}, n_iter={int(item['n_iter'])}")

    print("\n=== Classification (cross-validation accuracy) ===")
    for model_name, metric in model_results.items():
        print(
            f"{model_name}: mean={metric['mean_accuracy']:.4f}, "
            f"std={metric['std_accuracy']:.4f}"
        )

    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {args.output}")


if __name__ == "__main__":
    main()
