"""Leakage-safe prototype biomarker selection for TCGA-BRCA.

The key difference from the earlier script is ordering:

1. Match expression samples to metadata.
2. Split by patient/case into train, validation, and test sets.
3. Filter, rank, and de-correlate genes using TRAINING samples only.
4. Export the selected features for every sample with its fixed split.

Validation and test expression never influence biomarker selection. This is
still exploratory transcriptomic feature selection, not clinical validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NON_SAMPLE_COLUMNS = {"gene_id", "gene_name", "gene_type"}
LABEL_MAP = {
    "Primary Tumor": "Tumor",
    "Solid Tissue Normal": "Normal",
}


def read_expression(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError("Expression matrix must be .parquet or .csv")


def split_case_ids(
    metadata: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Assign complete GDC cases to splits, stratified by label composition."""
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be greater than zero.")
    if validation_fraction + test_fraction >= 0.5:
        raise ValueError("Validation plus test fraction must be less than 0.5.")

    cases = (
        metadata.groupby("case_id", as_index=False)["label"]
        .agg(lambda labels: "+".join(sorted(set(labels))))
        .rename(columns={"label": "case_stratum"})
    )

    def safe_stratify(frame: pd.DataFrame):
        counts = frame["case_stratum"].value_counts()
        return frame["case_stratum"] if len(counts) > 1 and counts.min() >= 2 else None

    train_validation, test = train_test_split(
        cases,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=safe_stratify(cases),
    )

    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation,
        random_state=seed,
        shuffle=True,
        stratify=safe_stratify(train_validation),
    )

    case_to_split = {
        **{case_id: "train" for case_id in train["case_id"]},
        **{case_id: "validation" for case_id in validation["case_id"]},
        **{case_id: "test" for case_id in test["case_id"]},
    }
    result = metadata.copy()
    result["split"] = result["case_id"].map(case_to_split)

    if result["split"].isna().any():
        raise RuntimeError("At least one case was not assigned to a data split.")

    for split_name, group in result.groupby("split"):
        present = set(group["label"])
        if present != {"Tumor", "Normal"}:
            raise RuntimeError(
                f"{split_name} split does not contain both classes: {sorted(present)}"
            )

    return result


def load_and_split(
    expression_path: Path,
    metadata_path: Path,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = read_expression(expression_path)
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig", dtype=str)
    expression.columns = expression.columns.astype(str).str.strip()
    metadata.columns = metadata.columns.str.replace("\ufeff", "", regex=False).str.strip()

    required_expression = {"gene_id", "gene_name"}
    required_metadata = {"file_id", "case_id", "sample_type"}
    if not required_expression.issubset(expression.columns):
        raise ValueError(f"Expression matrix requires {sorted(required_expression)}")
    if not required_metadata.issubset(metadata.columns):
        raise ValueError(
            f"Metadata requires {sorted(required_metadata)}. Regenerate it with the "
            "updated build_gdc_sample_metadata.py."
        )

    if "download_complete" in metadata.columns:
        metadata = metadata[
            metadata["download_complete"].astype(str).str.lower().eq("true")
        ].copy()

    metadata["label"] = metadata["sample_type"].str.strip().map(LABEL_MAP)
    metadata = metadata[metadata["label"].notna()].copy()
    metadata = metadata.dropna(subset=["file_id", "case_id"])

    duplicated = metadata.duplicated(subset="file_id", keep=False)
    if duplicated.any():
        conflicts = (
            metadata.loc[duplicated]
            .groupby("file_id")[["case_id", "sample_type"]]
            .nunique()
        )
        conflicts = conflicts[(conflicts > 1).any(axis=1)]
        if not conflicts.empty:
            raise ValueError(
                f"{len(conflicts)} file_id values map to conflicting case/sample "
                "metadata. Inspect gdc_sample_metadata.csv before continuing."
            )
        metadata = metadata.drop_duplicates(subset="file_id", keep="first")

    sample_columns = [c for c in expression.columns if c not in NON_SAMPLE_COLUMNS]
    metadata_ids = set(metadata["file_id"])
    matched = [sample_id for sample_id in sample_columns if sample_id in metadata_ids]
    if not matched:
        raise ValueError("No expression sample columns matched metadata file_id values.")

    ignored_expression = len(sample_columns) - len(matched)
    if ignored_expression:
        warnings.warn(f"Ignoring {ignored_expression} expression columns without metadata.")

    expression = expression[
        [c for c in ["gene_id", "gene_name", "gene_type"] if c in expression.columns]
        + matched
    ].copy()
    metadata = metadata.set_index("file_id").loc[matched].reset_index()
    metadata = split_case_ids(
        metadata,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    return expression, metadata


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg correction without an extra statsmodels dependency."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def select_training_biomarkers(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    final_count: int,
    correlation_threshold: float,
    min_log2_expression: float,
    min_sample_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from scipy.stats import ttest_ind

    sample_columns = [c for c in expression.columns if c not in NON_SAMPLE_COLUMNS]
    training = metadata[metadata["split"] == "train"]
    tumor_ids = training.loc[training["label"] == "Tumor", "file_id"].tolist()
    normal_ids = training.loc[training["label"] == "Normal", "file_id"].tolist()

    numeric = expression[sample_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    training_values = numeric[tumor_ids + normal_ids]
    expressed_fraction = (training_values >= min_log2_expression).mean(axis=1)
    keep = expressed_fraction >= min_sample_fraction

    filtered = numeric.loc[keep].reset_index(drop=True)
    gene_columns = [c for c in ["gene_id", "gene_name", "gene_type"] if c in expression]
    genes = expression.loc[keep, gene_columns].reset_index(drop=True)

    tumor = filtered[tumor_ids]
    normal = filtered[normal_ids]
    tumor_mean = tumor.mean(axis=1)
    normal_mean = normal.mean(axis=1)
    difference = tumor_mean - normal_mean

    test_result = ttest_ind(
        tumor.to_numpy(),
        normal.to_numpy(),
        axis=1,
        equal_var=False,
        nan_policy="omit",
    )
    p_values = np.nan_to_num(test_result.pvalue, nan=1.0)

    ranking = genes.copy()
    ranking["train_tumor_mean_log2_tpm1"] = tumor_mean
    ranking["train_normal_mean_log2_tpm1"] = normal_mean
    ranking["log2_mean_difference_tumor_minus_normal"] = difference
    ranking["abs_log2_mean_difference"] = difference.abs()
    ranking["welch_p_value"] = p_values
    ranking["fdr_bh"] = benjamini_hochberg(p_values)
    ranking = ranking.sort_values(
        ["abs_log2_mean_difference", "fdr_bh"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    id_to_row = {
        gene_id: row_index
        for row_index, gene_id in enumerate(genes["gene_id"].astype(str))
    }
    id_to_name = dict(
        zip(genes["gene_id"].astype(str), genes["gene_name"].astype(str))
    )
    selected_ids: list[str] = []
    selected_names: set[str] = set()

    # Correlation is also computed only from training samples.
    for gene_id in ranking["gene_id"].astype(str):
        gene_name = id_to_name[gene_id]
        if gene_name in selected_names:
            continue
        row_index = id_to_row[gene_id]
        if not selected_ids:
            selected_ids.append(gene_id)
            selected_names.add(gene_name)
        else:
            selected_indices = [id_to_row[selected] for selected in selected_ids]
            correlation_block = training_values.loc[
                keep
            ].reset_index(drop=True).iloc[[row_index] + selected_indices].T.corr()
            max_correlation = correlation_block.iloc[0, 1:].abs().max(skipna=True)
            if pd.isna(max_correlation) or max_correlation < correlation_threshold:
                selected_ids.append(gene_id)
                selected_names.add(gene_name)

        if len(selected_ids) == final_count:
            break

    if len(selected_ids) < final_count:
        raise RuntimeError(
            f"Only {len(selected_ids)} genes passed the correlation threshold."
        )

    selected_summary = ranking.set_index("gene_id").loc[selected_ids].reset_index()
    selected_summary.insert(0, "selection_rank", range(1, final_count + 1))

    selected_rows = [id_to_row[gene_id] for gene_id in selected_ids]
    selected_expression = filtered.iloc[selected_rows].copy()
    selected_expression.index = selected_summary["gene_name"].astype(str)

    # Model-ready orientation: one row per sample and one column per selected gene.
    model_features = selected_expression[sample_columns].T
    model_features.index.name = "file_id"
    model_features = model_features.reset_index()
    model_dataset = metadata.merge(model_features, on="file_id", how="inner")

    return selected_summary, ranking, model_dataset


def save_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        table.to_parquet(path, index=False, compression="zstd")
    else:
        table.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select biomarkers without train/validation/test leakage."
    )
    parser.add_argument("--expression", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", default="biomarker_outputs")
    parser.add_argument("--final-count", type=int, default=8, choices=[6, 8])
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument("--min-log2-expression", type=float, default=1.0)
    parser.add_argument("--min-sample-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    expression, metadata = load_and_split(
        Path(args.expression).resolve(),
        Path(args.metadata).resolve(),
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    selected, ranking, model_dataset = select_training_biomarkers(
        expression,
        metadata,
        final_count=args.final_count,
        correlation_threshold=args.correlation_threshold,
        min_log2_expression=args.min_log2_expression,
        min_sample_fraction=args.min_sample_fraction,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_table(selected, output_dir / "selected_biomarkers.csv")
    save_table(ranking, output_dir / "gene_ranking_train_only.csv")
    save_table(metadata, output_dir / "sample_splits.csv")
    save_table(model_dataset, output_dir / "model_dataset.parquet")

    print("\n=== Leakage-Safe Biomarker Selection Complete ===")
    print("\nSamples by fixed split and class:")
    print(metadata.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("\nUnique patients by split:")
    print(metadata.groupby("split")["case_id"].nunique())
    print("\nSelected training-only genes:")
    print(
        selected[
            [
                "selection_rank",
                "gene_name",
                "abs_log2_mean_difference",
                "log2_mean_difference_tumor_minus_normal",
                "fdr_bh",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
