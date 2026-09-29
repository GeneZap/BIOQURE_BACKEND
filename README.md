# GDC Download Project

This repository contains code for downloading and processing data from the Genomic Data Commons (GDC).

## Project Structure

- `data/` - Directory for downloaded data files
- `scripts/` - Python scripts for data processing
- `notebooks/` - Jupyter notebooks for analysis
- `EXPRESSSION_MATRIX/` - Directory for compiled expression matrices (excluded from git by .gitignore)

## Important Note About Large Files

The `EXPRESSSION_MATRIX/compiled_expression_matrix.parquet` file is excluded from version control due to its large size (191.41 MB). This file will be generated during the data processing workflow and should not be committed to the repository.

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Management

Large data files are automatically excluded from Git using the `.gitignore` configuration. For more information on managing large files, see [GIT_CLEANUP_GUIDE.md](GIT_CLEANUP_GUIDE.md).

## Usage

Run the main script to download and process data:
```bash
python main.py
```