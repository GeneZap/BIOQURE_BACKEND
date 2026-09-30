"""Build sample metadata for GDC RNA-seq files downloaded with gdc-client.

Expected layout (all beside this script):

    build_gdc_sample_metadata.py
    gdc_manifest.<date>.txt
    gdc_data/
        <file UUID>/
            <RNA-seq TSV file>

The script can run while gdc-client is downloading.  Each output row records
whether the corresponding file is complete, but downstream matrix generation
should use only rows where ``download_complete`` is True.
"""

from pathlib import Path
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "gdc_data"
OUTPUT = BASE_DIR / "gdc_sample_metadata.csv"

# Leave this as None to automatically use the newest gdc_manifest*.txt file
# beside this script.  Alternatively, provide an exact filename, for example:
# MANIFEST_NAME = "gdc_manifest.2026-09-29.162943.txt"
MANIFEST_NAME = None

RNA_FILENAME_SUFFIX = "rna_seq.augmented_star_gene_counts.tsv"
REQUEST_DELAY_SECONDS = 0.05


def locate_manifest() -> Path:
    """Find the requested manifest or the newest GDC manifest locally."""
    if MANIFEST_NAME:
        manifest_path = BASE_DIR / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return manifest_path

    candidates = list(BASE_DIR.glob("gdc_manifest*.txt"))
    legacy_manifest = BASE_DIR / "MANIFEST.txt"
    if legacy_manifest.is_file():
        candidates.append(legacy_manifest)

    if not candidates:
        raise FileNotFoundError(
            "No gdc_manifest*.txt or MANIFEST.txt file was found beside "
            f"this script: {BASE_DIR}"
        )

    # The manifest downloaded most recently is normally the full cohort.
    return max(candidates, key=lambda path: path.stat().st_mtime)


def make_session() -> requests.Session:
    """Create an HTTP session that retries temporary GDC/API failures."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def main() -> None:
    manifest_path = locate_manifest()

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            "Create it and run gdc-client with: -d .\\gdc_data"
        )

    print(f"Manifest : {manifest_path}")
    print(f"Data dir : {DATA_DIR}")
    print(f"Output   : {OUTPUT}")

    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    required_columns = {"id", "filename"}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_columns)}"
        )

    files = manifest[
        manifest["filename"].str.endswith(RNA_FILENAME_SUFFIX, na=False)
    ].copy()

    if files.empty:
        raise ValueError(
            f"No files ending with {RNA_FILENAME_SUFFIX!r} were found in "
            f"{manifest_path.name}."
        )

    files = files.drop_duplicates(subset="id", keep="first")
    print(f"RNA-seq files in manifest: {len(files):,}")

    session = make_session()
    results = []
    api_failures = []

    for position, (_, row) in enumerate(files.iterrows(), start=1):
        file_id = row["id"]
        filename = row["filename"]
        file_path = DATA_DIR / file_id / filename

        expected_size = pd.to_numeric(row.get("size"), errors="coerce")
        actual_size = file_path.stat().st_size if file_path.is_file() else None
        size_matches = (
            actual_size == int(expected_size)
            if actual_size is not None and pd.notna(expected_size)
            else file_path.is_file()
        )
        download_complete = bool(file_path.is_file() and size_matches)

        print(
            f"[{position:04d}/{len(files):04d}] Fetching metadata: {file_id} "
            f"| downloaded={download_complete}"
        )

        url = f"https://api.gdc.cancer.gov/files/{file_id}"
        params = {
            "expand": "cases.samples",
            "fields": (
                "file_id,file_name,cases.case_id,cases.submitter_id,"
                "cases.samples.sample_id,cases.samples.submitter_id,"
                "cases.samples.sample_type"
            ),
        }

        try:
            response = session.get(url, params=params, timeout=45)
            response.raise_for_status()
            data = response.json().get("data", {})
        except (requests.RequestException, ValueError) as exc:
            api_failures.append((file_id, str(exc)))
            results.append(
                {
                    "file_id": file_id,
                    "file_name": filename,
                    "file_path": str(file_path.resolve()),
                    "download_complete": download_complete,
                    "expected_size_bytes": (
                        int(expected_size) if pd.notna(expected_size) else None
                    ),
                    "actual_size_bytes": actual_size,
                    "case_id": None,
                    "case_submitter_id": None,
                    "sample_id": None,
                    "sample_submitter_id": None,
                    "sample_type": None,
                    "api_error": str(exc),
                }
            )
            continue

        cases = data.get("cases") or []
        if not cases:
            cases = [{}]

        for case in cases:
            samples = case.get("samples") or [{}]
            for sample in samples:
                results.append(
                    {
                        "file_id": file_id,
                        "file_name": filename,
                        "file_path": str(file_path.resolve()),
                        "download_complete": download_complete,
                        "expected_size_bytes": (
                            int(expected_size) if pd.notna(expected_size) else None
                        ),
                        "actual_size_bytes": actual_size,
                        "case_id": case.get("case_id"),
                        "case_submitter_id": case.get("submitter_id"),
                        "sample_id": sample.get("sample_id"),
                        "sample_submitter_id": sample.get("submitter_id"),
                        "sample_type": sample.get("sample_type"),
                        "api_error": None,
                    }
                )

        time.sleep(REQUEST_DELAY_SECONDS)

    metadata = pd.DataFrame(results)
    metadata.to_csv(OUTPUT, index=False)

    completed_file_count = metadata.loc[
        metadata["download_complete"], "file_id"
    ].nunique()
    unique_file_count = metadata["file_id"].nunique()

    print("\nDone!")
    print(f"Saved to: {OUTPUT}")
    print(f"Unique RNA-seq files: {unique_file_count:,}")
    print(f"Completed downloads: {completed_file_count:,}/{unique_file_count:,}")
    print("\nSample types:")
    print(metadata["sample_type"].value_counts(dropna=False))

    duplicate_rows = metadata.duplicated(subset=["file_id"], keep=False)
    if duplicate_rows.any():
        duplicated_files = metadata.loc[duplicate_rows, "file_id"].nunique()
        print(
            f"\nWarning: {duplicated_files} file(s) mapped to multiple metadata "
            "rows. Inspect these before building a one-row-per-file matrix."
        )

    if api_failures:
        print(f"\nWarning: metadata API failed for {len(api_failures)} file(s).")
        print("Rerun the script later; successful downloads will not be altered.")

    if completed_file_count < unique_file_count:
        print(
            "\nThe GDC download is still incomplete. You may inspect this CSV now, "
            "but wait until all files report download_complete=True before "
            "building the expression matrix."
        )


if __name__ == "__main__":
    main()
