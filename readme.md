# Parquet Evaluation Analysis

This project contains scripts to analyze and generate reports from parquet files.

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

## Running the Analysis

To run the analysis script on k% of the full parquet, you also need to provide the path to the parquet file as a command-line argument.

```bash
python gpt_4o_analysis.py data/your_file.parquet --k 5
```

Replace `data/your_file.parquet` with the actual path to your parquet file.

## Generating the Report

After running the analysis, you can generate a report using the `gpt_4o_report.py` script.

```bash
python gpt_4o_report.py
```

This will use the output from the analysis script to generate the report.
