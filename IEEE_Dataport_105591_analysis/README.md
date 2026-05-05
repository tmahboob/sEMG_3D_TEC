# Dataset 105591 - Analysis Package

This package contains automated analysis and visualizations for dataset 105591.

## Contents

- `run_analysis.py` - Python script to regenerate visualizations
- `analysis_report.md` - Detailed analysis of the dataset
- `visualizations/` - Pre-generated visualization images
- `output/` - Directory for regenerated visualizations
- `lib/` - Required library dependencies
- `requirements.txt` - Python package dependencies

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure AWS credentials** (for data access):
   The script loads data from S3. Ensure you have AWS credentials configured:
   ```bash
   aws configure
   ```
   Or set environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

3. **Run the analysis:**
   ```bash
   python run_analysis.py
   ```

   Generated visualizations will be saved to the `output/` directory.

## Pre-generated Visualizations

The `visualizations/` folder contains pre-generated images if you want to
view results without running the script.

## Requirements

- Python 3.8+
- AWS credentials with read access to the dataset S3 bucket
- See `requirements.txt` for Python package dependencies

## Generated

This package was generated on 2026-05-01 11:36:02 UTC
