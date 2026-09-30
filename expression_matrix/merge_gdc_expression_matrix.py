"""Merge downloaded GDC STAR-Counts files into one expression matrix.

This script expects ``gdc_sample_metadata.csv`` produced by
``build_gdc_sample_metadata.py``. It reads the absolute ``file_path`` column,
uses the TSV's own header (rather than a hard-coded column order), extracts
``tpm_unstranded``, converts it to log2(TPM + 1), and writes a Parquet matrix.

Output orientation:
    rows    = genes
    columns = gene_id, gene_name, gene_type, then one column per GDC file_id
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RNA_SUFFIX = "rna_seq.augmented_star_gene_counts.tsv"
GENE_COLUMNS = ["gene_id", "gene_name", "gene_type"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge GDC STAR-Counts TSVs into a log2(TPM+1) matrix."
    )
    parser.add_argument(
        "--metadata",
        default=str(BASE_DIR / "gdc_sample_metadata.csv"),
        help="Metadata CSV produced by build_gdc_sample_metadata.py",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "compiled_expression_matrix.parquet"),
        help="Output .parquet (recommended) or .csv file",
    )
    parser.add_argument(
        "--value-column",
        default="tpm_unstranded",
        help="STAR-Counts numeric column to merge (default: tpm_unstranded)",
    )
    return parser.parse_args()


def read_star_counts(path: Path, value_column: str) -> pd.DataFrame:
    """Read one current GDC augmented STAR-Counts file by column name."""
    table = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype={"gene_id": "string", "gene_name": "string", "gene_type": "string"},
        low_memory=False,
    )

    required = set(GENE_COLUMNS + [value_column])
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns {sorted(missing)}. "
            f"Columns found: {list(table.columns)}"
        )

    # Remove STAR summary records such as N_unmapped and keep Ensembl genes.
    table = table.loc[
        table["gene_id"].astype(str).str.startswith("ENSG"),
        GENE_COLUMNS + [value_column],
    ].copy()
    table = table.drop_duplicates(subset="gene_id", keep="first")
    table[value_column] = pd.to_numeric(table[value_column], errors="coerce").fillna(0)

    if table.empty:
        raise ValueError(f"No Ensembl gene rows were found in {path}")
    if (table[value_column] < 0).any():
        raise ValueError(f"Negative values found in {value_column} for {path}")

    return table.reset_index(drop=True)


def save_matrix(matrix: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        try:
            matrix.to_parquet(output_path, index=False, compression="zstd")
        except ImportError as exc:
            raise RuntimeError(
                "Writing Parquet requires pyarrow. Install it with: "
                "python -m pip install pyarrow"
            ) from exc
    elif suffix == ".csv":
        matrix.to_csv(output_path, index=False)
    else:
        raise ValueError("Output must end in .parquet or .csv")


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata).resolve()
    output_path = Path(args.output).resolve()

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig", dtype=str)
    metadata.columns = metadata.columns.str.replace("\ufeff", "", regex=False).str.strip()

    required_metadata = {"file_id", "file_name", "file_path"}
    missing = required_metadata.difference(metadata.columns)
    if missing:
        raise ValueError(
            f"Metadata is missing {sorted(missing)}. Regenerate it with "
            "build_gdc_sample_metadata.py."
        )

    if "download_complete" in metadata.columns:
        complete = metadata["download_complete"].astype(str).str.lower().eq("true")
        incomplete_count = metadata.loc[~complete, "file_id"].nunique()
        if incomplete_count:
            raise RuntimeError(
                f"{incomplete_count} RNA-seq download(s) are incomplete. Wait for "
                "gdc-client to finish, rerun build_gdc_sample_metadata.py, and then "
                "rerun this merger."
            )

    metadata = metadata[
        metadata["file_name"].str.endswith(RNA_SUFFIX, na=False)
    ].drop_duplicates(subset="file_id", keep="first")

    if metadata.empty:
        raise ValueError(f"No {RNA_SUFFIX} files were found in the metadata.")

    missing_paths = [path for path in metadata["file_path"] if not Path(path).is_file()]
    if missing_paths:
        preview = "\n".join(missing_paths[:5])
        raise FileNotFoundError(
            f"{len(missing_paths)} expression file(s) are missing. First examples:\n{preview}"
        )

    sample_count = len(metadata)
    first_row = metadata.iloc[0]
    first_table = read_star_counts(Path(first_row["file_path"]), args.value_column)
    gene_metadata = first_table[GENE_COLUMNS].copy()
    reference_ids = pd.Index(gene_metadata["gene_id"], name="gene_id")

    # float32 is sufficient for log-expression and keeps the full cohort matrix
    # near 300 MB instead of roughly 600 MB for float64.
    values = np.empty((len(reference_ids), sample_count), dtype=np.float32)

    for column_index, (_, row) in enumerate(metadata.iterrows()):
        file_id = row["file_id"]
        path = Path(row["file_path"])
        print(f"[{column_index + 1:04d}/{sample_count:04d}] {file_id}")

        table = first_table if column_index == 0 else read_star_counts(
            path, args.value_column
        )

        if table["gene_id"].equals(gene_metadata["gene_id"]):
            sample_values = table[args.value_column].to_numpy(dtype=np.float32)
        else:
            # Align explicitly if a file has the same genes in a different order.
            aligned = table.set_index("gene_id")[args.value_column].reindex(reference_ids)
            missing_genes = int(aligned.isna().sum())
            if missing_genes:
                raise ValueError(
                    f"{path} is missing {missing_genes} genes present in the first file."
                )
            sample_values = aligned.to_numpy(dtype=np.float32)

        values[:, column_index] = np.log2(sample_values + np.float32(1.0))

    expression = pd.DataFrame(values, columns=metadata["file_id"].tolist())
    final_matrix = pd.concat([gene_metadata.reset_index(drop=True), expression], axis=1)
    save_matrix(final_matrix, output_path)

    print("\n=== Expression Matrix Complete ===")
    print(f"Genes   : {len(gene_metadata):,}")
    print(f"Samples : {sample_count:,}")
    print(f"Shape   : {final_matrix.shape}")
    print(f"Metric  : log2({args.value_column} + 1)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
