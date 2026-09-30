"""Prepare leakage-safe classical and quantum arrays from model_dataset.parquet.

Input is produced by select_biomarkers_leakage_safe.py and already contains
patient-grouped train/validation/test assignments. This script NEVER creates a
new split.

Classical features:
    StandardScaler fitted on the complete training split only.

Quantum features:
    MinMaxScaler fitted on the complete training split only, mapped to [0, pi].
    The quantum training set is balanced by random undersampling of the tumor
    majority class. Validation and test sets remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


LABEL_TO_INT = {"Normal": 0, "Tumor": 1}
RESERVED_COLUMNS = {
    "file_id",
    "file_name",
    "file_path",
    "download_complete",
    "expected_size_bytes",
    "actual_size_bytes",
    "case_id",
    "case_submitter_id",
    "sample_id",
    "sample_submitter_id",
    "sample_type",
    "api_error",
    "label",
    "split",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fixed-split classical and quantum model arrays."
    )
    parser.add_argument(
        "--dataset",
        default="biomarker_outputs/model_dataset.parquet",
        help="Model dataset from leakage-safe biomarker selection",
    )
    parser.add_argument(
        "--selected-biomarkers",
        default="biomarker_outputs/selected_biomarkers.csv",
        help="Selected biomarker summary used to determine feature order",
    )
    parser.add_argument("--output-dir", default="prepared_data")
    parser.add_argument(
        "--quantum-majority-ratio",
        type=float,
        default=1.0,
        help="Maximum Tumor:Normal ratio in quantum training data",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError("Dataset must be .parquet or .csv")


def save_array(output_dir: Path, name: str, value: np.ndarray) -> None:
    np.save(output_dir / f"{name}.npy", value)


def main() -> None:
    args = parse_args()
    if args.quantum_majority_ratio < 1.0:
        raise ValueError("quantum-majority-ratio must be at least 1.0")

    dataset_path = Path(args.dataset).resolve()
    biomarkers_path = Path(args.selected_biomarkers).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_table(dataset_path)
    biomarkers = pd.read_csv(biomarkers_path, encoding="utf-8-sig")
    dataset.columns = dataset.columns.astype(str).str.strip()
    biomarkers.columns = biomarkers.columns.astype(str).str.strip()

    required = {"file_id", "case_id", "label", "split"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Model dataset is missing columns: {sorted(missing)}")

    if "gene_name" not in biomarkers.columns:
        raise ValueError("Selected-biomarkers CSV must contain gene_name.")

    feature_names = biomarkers.sort_values("selection_rank")["gene_name"].astype(str).tolist()
    if len(feature_names) not in (6, 8):
        raise ValueError(f"Expected 6 or 8 biomarkers, found {len(feature_names)}")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Selected biomarker names are not unique.")

    missing_features = set(feature_names).difference(dataset.columns)
    if missing_features:
        raise ValueError(f"Dataset is missing biomarkers: {sorted(missing_features)}")

    dataset["label_int"] = dataset["label"].map(LABEL_TO_INT)
    if dataset["label_int"].isna().any():
        bad = sorted(dataset.loc[dataset["label_int"].isna(), "label"].unique())
        raise ValueError(f"Unexpected labels: {bad}")

    expected_splits = {"train", "validation", "test"}
    found_splits = set(dataset["split"])
    if found_splits != expected_splits:
        raise ValueError(
            f"Expected splits {sorted(expected_splits)}, found {sorted(found_splits)}"
        )

    # Defensive patient-leakage check.
    split_counts_per_case = dataset.groupby("case_id")["split"].nunique()
    if (split_counts_per_case > 1).any():
        raise RuntimeError("At least one case_id occurs in more than one split.")

    X_by_split: dict[str, np.ndarray] = {}
    y_by_split: dict[str, np.ndarray] = {}
    rows_by_split: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        rows = dataset[dataset["split"] == split_name].copy().reset_index(drop=True)
        if set(rows["label_int"]) != {0, 1}:
            raise RuntimeError(f"{split_name} does not contain both classes.")
        rows_by_split[split_name] = rows
        X_by_split[split_name] = (
            rows[feature_names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        )
        y_by_split[split_name] = rows["label_int"].to_numpy(dtype=np.int64)

    classical_scaler = StandardScaler()
    quantum_scaler = MinMaxScaler(feature_range=(0.0, np.pi), clip=True)
    classical_scaler.fit(X_by_split["train"])
    quantum_scaler.fit(X_by_split["train"])

    X_classical = {
        split: classical_scaler.transform(values).astype(np.float32)
        for split, values in X_by_split.items()
    }
    X_quantum = {
        split: np.clip(quantum_scaler.transform(values), 0.0, np.pi).astype(np.float32)
        for split, values in X_by_split.items()
    }

    # Use all training normals and a reproducible subset of training tumors.
    rng = np.random.default_rng(args.seed)
    y_train_full = y_by_split["train"]
    normal_indices = np.flatnonzero(y_train_full == 0)
    tumor_indices = np.flatnonzero(y_train_full == 1)
    max_tumors = min(
        len(tumor_indices),
        int(np.floor(len(normal_indices) * args.quantum_majority_ratio)),
    )
    sampled_tumors = rng.choice(tumor_indices, size=max_tumors, replace=False)
    quantum_train_indices = np.concatenate([normal_indices, sampled_tumors])
    rng.shuffle(quantum_train_indices)

    # Classical arrays retain the full cohort and handle imbalance in the model.
    for split_name in ("train", "validation", "test"):
        save_array(output_dir, f"X_{split_name}_classical", X_classical[split_name])
        save_array(output_dir, f"y_{split_name}", y_by_split[split_name])

    # Quantum validation/test remain representative; only training is balanced.
    save_array(
        output_dir,
        "X_train_quantum",
        X_quantum["train"][quantum_train_indices],
    )
    save_array(
        output_dir,
        "y_train_quantum",
        y_train_full[quantum_train_indices],
    )
    save_array(output_dir, "X_validation_quantum", X_quantum["validation"])
    save_array(output_dir, "X_test_quantum", X_quantum["test"])

    joblib.dump(classical_scaler, output_dir / "classical_scaler.joblib")
    joblib.dump(quantum_scaler, output_dir / "quantum_scaler.joblib")

    sample_records = []
    for split_name, rows in rows_by_split.items():
        records = rows[["file_id", "case_id", "sample_type", "label", "split"]].copy()
        records["used_for_quantum_training"] = False
        if split_name == "train":
            records.loc[quantum_train_indices, "used_for_quantum_training"] = True
        sample_records.append(records)
    pd.concat(sample_records, ignore_index=True).to_csv(
        output_dir / "prepared_samples.csv", index=False
    )

    configuration = {
        "seed": args.seed,
        "feature_names": feature_names,
        "class_mapping": LABEL_TO_INT,
        "classical_scaler": "StandardScaler fitted on complete training split",
        "quantum_scaler": "MinMaxScaler [0, pi] fitted on complete training split",
        "quantum_majority_ratio": args.quantum_majority_ratio,
        "counts": {
            split: {
                "total": int(len(y)),
                "normal": int(np.sum(y == 0)),
                "tumor": int(np.sum(y == 1)),
            }
            for split, y in y_by_split.items()
        },
        "quantum_training_counts": {
            "total": int(len(quantum_train_indices)),
            "normal": int(np.sum(y_train_full[quantum_train_indices] == 0)),
            "tumor": int(np.sum(y_train_full[quantum_train_indices] == 1)),
        },
    }
    with open(output_dir / "preparation_config.json", "w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2)

    print("\n=== Model Data Preparation Complete ===")
    print(f"Features: {feature_names}")
    for split, counts in configuration["counts"].items():
        print(
            f"{split:10s}: total={counts['total']}, "
            f"normal={counts['normal']}, tumor={counts['tumor']}"
        )
    q_counts = configuration["quantum_training_counts"]
    print(
        "Quantum training subset: "
        f"total={q_counts['total']}, normal={q_counts['normal']}, "
        f"tumor={q_counts['tumor']}"
    )
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
