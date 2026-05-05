"""
Minimal S3 utilities for user packages.

This is a simplified version of s3_utils.py designed for the downloadable user
packages. It provides the data loading functions needed to run visualization
scripts locally, with functional parity to the production version.

IMPORTANT: This module must maintain signature and behavior parity with
lib/s3_utils.py to ensure generated scripts work identically in both
AWS Lambda and local user package environments.
"""

import io
import logging
import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# S3 client singleton
_s3_client = None

# WARNING: These values must be kept in sync with lib/config.py.
# They are duplicated here to keep this module self-contained for user packages.
# Any changes in lib/config.py must be manually mirrored here.
HEADERLESS_MAX_COLUMN_NAME_LENGTH = 100  # Characters - longer suggests data as header
HEADERLESS_MAX_COLUMN_WORDS = 10  # Words - more suggests data as header


def _get_s3_client():
    """Get or create S3 client singleton."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def _parse_s3_path(s3_path: str) -> tuple:
    """
    Parse S3 path into bucket and key.

    Args:
        s3_path: S3 path (s3://bucket/key or bucket/key)

    Returns:
        tuple: (bucket, key) or (None, None) on invalid path
    """
    if s3_path.startswith("s3://"):
        path = s3_path[5:]
    else:
        path = s3_path

    parts = path.split("/", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def _detect_prepended_metadata(rows: list) -> dict:
    """
    Detect prepended metadata rows at the top of structured data.

    Mirrors lib.format_detection.prepended_data_detector for full parity.
    Detects metadata rows (key-value pairs) and determines where actual data starts.

    Args:
        rows: List of row values (for Excel: list of lists)

    Returns:
        dict: {
            'skip_count': int,      # Number of rows to skip
            'metadata': dict,       # Extracted key-value pairs
            'pattern_type': str,    # 'metadata' | 'structure' | None
        }
    """
    import re
    from collections import Counter

    if not rows or len(rows) < 3:
        return {"skip_count": 0, "metadata": {}, "pattern_type": None}

    # For Excel rows (list of values), count non-empty cells per row
    def count_non_empty_cells(row):
        if isinstance(row, (list, tuple)):
            return sum(
                1 for x in row if x is not None and str(x).strip() and str(x) != "nan"
            )
        return 1 if str(row).strip() else 0

    # Normalize rows to strings for pattern analysis
    def normalize_row(row):
        if isinstance(row, (list, tuple)):
            cleaned = [str(x) for x in row if x is not None and str(x) != "nan"]
            return "\t".join(cleaned)
        return str(row)

    normalized = [normalize_row(r) for r in rows[:100]]
    cell_counts = [count_non_empty_cells(r) for r in rows[:100]]

    # Method 1: Pattern-based detection (metadata key-value pairs)
    metadata_rows = []
    pattern_data_start = 0

    for idx, row_str in enumerate(normalized):
        if not row_str.strip():
            continue

        # Check for metadata pattern: "Key: Value" or "Key\tValue"
        colon_match = re.match(
            r"^([A-Za-z][A-Za-z0-9\s\(\)#\.:]*?):\s*(.*)$", row_str.strip()
        )
        if colon_match:
            key, value = colon_match.groups()
            key = key.strip()
            if len(key) < 50 and not key.replace(".", "").replace("-", "").isdigit():
                metadata_rows.append({"key": key, "value": value.strip()})
                pattern_data_start = idx + 1
                continue

        # Check for tab-separated key-value (2 columns only)
        if "\t" in row_str:
            parts = row_str.strip().split("\t")
            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                if (
                    len(key) < 50
                    and key[0:1].isupper()
                    and not key.replace(".", "").isdigit()
                ):
                    metadata_rows.append({"key": key, "value": value.strip()})
                    pattern_data_start = idx + 1
                    continue

        # If row has multiple columns (3+), likely data
        delim_count = max(row_str.count("\t"), row_str.count(","))
        if delim_count >= 2:
            break

    # Method 2: Non-empty cell count disparity (for Excel sheets)
    # Metadata/header rows often have fewer filled cells than data rows
    structure_data_start = 0
    if len(cell_counts) >= 5:
        # Find stable cell count in lower portion (actual data)
        lower_start = max(3, len(cell_counts) // 3)
        lower_portion = [c for c in cell_counts[lower_start:] if c > 0]

        if lower_portion:
            stable_count = Counter(lower_portion).most_common(1)[0][0]

            # Find where cell count stabilizes to data pattern
            consecutive_stable = 0
            for idx, count in enumerate(cell_counts):
                # Data row: has >= 80% of stable count filled cells
                if count >= stable_count * 0.8:
                    consecutive_stable += 1
                    if consecutive_stable >= 3:
                        structure_data_start = max(0, idx - 2)
                        break
                else:
                    consecutive_stable = 0

            # Validate: pre-data rows should have significantly fewer cells
            if structure_data_start > 0:
                pre_data_counts = [
                    c for c in cell_counts[:structure_data_start] if c >= 0
                ]
                if pre_data_counts:
                    avg_pre = sum(pre_data_counts) / len(pre_data_counts)
                    # Require significant disparity (pre-data has < 60% of stable count)
                    # Using 0.6 threshold to match production behavior
                    if avg_pre >= stable_count * 0.6:
                        structure_data_start = 0

    # Synthesize results
    if pattern_data_start > 0 and structure_data_start > 0:
        # Both detected - use pattern (more precise)
        final_skip = pattern_data_start
        pattern_type = "metadata"
    elif structure_data_start > 0:
        final_skip = structure_data_start
        pattern_type = "structure"
        logger.info(
            f"Structure-only detection: {structure_data_start} rows (no recognizable patterns)"
        )
    elif pattern_data_start > 0:
        final_skip = pattern_data_start
        pattern_type = "metadata"
    else:
        final_skip = 0
        pattern_type = None

    metadata_dict = {m["key"]: m["value"] for m in metadata_rows}

    return {
        "skip_count": final_skip,
        "metadata": metadata_dict,
        "pattern_type": pattern_type,
    }


def _classify_headerless_data_type(df: pd.DataFrame) -> str:
    """
    Classify the type of headerless numerical data based on value patterns.

    Mirrors lib.data_processing.classify_headerless_data_type for full parity.

    Args:
        df: DataFrame to analyze

    Returns:
        str: Data type classification ('signal_processing', 'sensor_readings',
             'feature_vectors', or 'measurements')
    """
    if df.empty:
        return "measurements"

    # Filter to numeric columns only for numerical analysis
    numeric_df = df.select_dtypes(include=["number"])

    # If no numeric columns, return measurements as fallback
    if numeric_df.empty:
        return "measurements"

    # Sample a subset for analysis to avoid performance issues
    sample_df = (
        numeric_df.sample(n=min(1000, len(numeric_df)), random_state=42)
        if len(numeric_df) > 1000
        else numeric_df
    )

    # Check value ranges and patterns
    try:
        all_negative = (sample_df < 0).all().all()
        mostly_negative = (sample_df < 0).sum().sum() > (sample_df.size * 0.7)
        large_values = (sample_df.abs() > 1000).any().any()
        small_decimals = (sample_df.abs() < 1).all().all()
        many_decimals = sample_df.apply(
            lambda col: col.apply(
                lambda x: len(str(x).split(".")[-1]) if "." in str(x) else 0
            ).mean()
            > 3
        ).any()

        # Signal processing data: mostly negative, small decimal values, many decimal places
        if (all_negative or mostly_negative) and small_decimals and many_decimals:
            return "signal_processing"
        # Sensor readings: large values, mixed signs
        elif large_values and not all_negative:
            return "sensor_readings"
        # Feature vectors: VERY many columns (increased threshold)
        elif df.shape[1] > 200:
            return "feature_vectors"
        # Signal processing fallback: many columns with numeric data (50-200 columns)
        elif df.shape[1] > 50:
            return "signal_processing"
        else:
            return "measurements"

    except Exception as e:
        logger.warning(f"Error classifying headerless data type: {e}")
        return "measurements"


def _detect_and_handle_headerless_data(df: pd.DataFrame, s3_key: str) -> tuple:
    """
    Detect headerless data and generate appropriate column names.

    Mirrors lib.data_processing.detect_and_handle_headerless_data for full parity.
    Handles decimal column names like "-6.9286130287878" and column names with
    leading/trailing whitespace and hyphens.

    Args:
        df: DataFrame to analyze
        s3_key: S3 key for logging purposes

    Returns:
        tuple: (cleaned_df, is_headerless_flag)
    """
    if df.empty:
        return df, False

    # Strip whitespace from column names before pattern matching
    stripped_columns = df.columns.astype(str).str.strip()

    # Check if column names are generic (0, 1, 2... or Unnamed: 0, etc.)
    generic_columns = stripped_columns.str.match(r"^(Unnamed: )?\d+$").all()

    # Normalize columns by removing both whitespace AND hyphens for decimal detection
    # This handles timestamps like "250328-151404.231" which are numeric but have hyphens
    normalized_columns = stripped_columns.str.replace("-", "", regex=False)

    # Check if ALL normalized column names are decimal values (like "250328151404.231", "1531.476878", "0.0.1")
    # Pattern handles multiple decimal points (e.g., "0.0.1")
    decimal_columns = normalized_columns.str.match(r"^\d+(\.\d+)*$").all()

    # Check if first row contains only numbers (likely headerless)
    try:
        first_row_all_numeric = (
            df.iloc[0]
            .apply(
                lambda x: pd.api.types.is_numeric_dtype(type(x))
                or (isinstance(x, (int, float)) and not pd.isna(x))
            )
            .all()
        )
    except (IndexError, TypeError):
        first_row_all_numeric = False

    # Check if all columns are numeric (strong indicator of headerless signal data)
    all_numeric_columns = df.select_dtypes(include=["number"]).shape[1] == df.shape[1]

    # Pattern 3: Very long column names (likely data being treated as header)
    # This catches cases like API call sequences being interpreted as column names
    long_column_names = any(
        len(str(col)) > HEADERLESS_MAX_COLUMN_NAME_LENGTH for col in df.columns
    )

    # Pattern 4: Column names with many space-separated tokens (indicating data, not headers)
    # Normal headers rarely have more than 5-6 words
    multi_token_columns = any(
        len(str(col).split()) > HEADERLESS_MAX_COLUMN_WORDS for col in df.columns
    )

    # Enhanced headerless detection:
    # - Generic columns need numeric data validation
    # - Decimal columns are sufficient evidence alone (regardless of data types)
    # - Long column names or multi-token columns indicate data as headers
    is_headerless = (
        (generic_columns and (first_row_all_numeric or all_numeric_columns))
        or decimal_columns
        or long_column_names
        or multi_token_columns
    )

    if is_headerless:
        if long_column_names:
            logger.info(
                f"Detected headerless data with long column names (>{HEADERLESS_MAX_COLUMN_NAME_LENGTH} chars) in {s3_key}"
            )
        elif multi_token_columns:
            logger.info(
                f"Detected headerless data with multi-token column names (>{HEADERLESS_MAX_COLUMN_WORDS} words) in {s3_key}"
            )
        elif decimal_columns:
            logger.info(
                f"Detected headerless data with decimal column names in {s3_key}"
            )
        else:
            logger.info(
                f"Detected headerless data with generic column names in {s3_key}"
            )

        # Generate meaningful column names based on data characteristics
        n_cols = len(df.columns)
        data_type = _classify_headerless_data_type(df)

        if data_type == "signal_processing":
            df.columns = [f"Signal_{i+1}" for i in range(n_cols)]
        elif data_type == "sensor_readings":
            df.columns = [f"Sensor_{i+1}" for i in range(n_cols)]
        elif data_type == "feature_vectors":
            df.columns = [f"Feature_{i+1}" for i in range(n_cols)]
        else:
            df.columns = [f"Measurement_{i+1}" for i in range(n_cols)]

        logger.info(
            f"Generated {n_cols} column names for headerless data: {list(df.columns[:5])}..."
        )

        return df, True

    return df, False


def _clean_dataframe_columns(df: pd.DataFrame, s3_key: str = "") -> pd.DataFrame:
    """
    Clean problematic columns from DataFrame before analysis.

    Mirrors behavior of lib.s3_client_utils.clean_dataframe_columns for full parity.

    Operations:
    1. Convert all column names to strings (Excel columns can be floats/ints)
    2. Remove unnamed columns (pandas auto-generated 'Unnamed: N' pattern)
    3. Remove completely empty columns
    4. Remove duplicate columns (keep first occurrence)

    Args:
        df: DataFrame to clean
        s3_key: S3 key for logging purposes

    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        return df

    original_shape = df.shape

    # Ensure all column names are strings (Excel columns can be floats/ints)
    df.columns = df.columns.astype(str)

    # Remove unnamed columns (exact pattern match for pandas auto-generated names)
    unnamed_pattern = r"^Unnamed: \d+$"
    try:
        unnamed_columns = df.columns[
            df.columns.astype(str).str.match(unnamed_pattern, na=False)
        ]
    except Exception as e:
        logger.warning(f"Could not check for unnamed columns in {s3_key}: {e}")
        unnamed_columns = []

    if len(unnamed_columns) > 0:
        df = df.drop(columns=unnamed_columns)
        logger.info(f"Removed {len(unnamed_columns)} unnamed columns from {s3_key}")

    # Remove completely empty columns
    empty_columns = df.columns[df.isnull().all()]
    if len(empty_columns) > 0:
        df = df.drop(columns=empty_columns)
        logger.info(f"Removed {len(empty_columns)} empty columns from {s3_key}")

    # Remove duplicate columns (keep first occurrence)
    duplicate_columns = df.columns[df.columns.duplicated()]
    if len(duplicate_columns) > 0:
        df = df.loc[:, ~df.columns.duplicated()]
        logger.info(f"Removed {len(duplicate_columns)} duplicate columns from {s3_key}")

    if df.shape != original_shape:
        logger.info(f"DataFrame cleaning for {s3_key}: {original_shape} -> {df.shape}")

    return df


def load_data_file(s3_url: str, sheet_name: str | None = None) -> pd.DataFrame | None:
    """
    Load a data file from S3 into a pandas DataFrame.

    Supports common data formats: CSV, TSV, JSON, Excel, Parquet.
    For Excel files, supports sheet name via argument or URL syntax: s3://bucket/file.xlsx::SheetName

    Handles dict return types from multi-sheet Excel files by extracting the
    requested sheet or the first available sheet.

    Args:
        s3_url: Full S3 path (s3://bucket/key) or with sheet name (s3://bucket/file.xlsx::Sheet)
        sheet_name: For Excel files, specific sheet to load. If provided and URL doesn't
                    contain ::, will be appended to URL.

    Returns:
        pd.DataFrame: Loaded data, or None on failure (matches production behavior)
    """
    s3 = _get_s3_client()

    try:
        # Append sheet_name to URL if provided separately
        if sheet_name and "::" not in s3_url:
            s3_url = f"{s3_url}::{sheet_name}"

        # Extract sheet name from URL for dict extraction (if present)
        url_sheet_name = s3_url.rsplit("::", 1)[1] if "::" in s3_url else None

        # Parse sheet name from path (for Excel files: path.xlsx::SheetName)
        s3_path_clean = s3_url.split("::")[0]

        logger.info(f"Loading data file: {s3_url}")

        # Parse S3 path
        bucket, key = _parse_s3_path(s3_path_clean)
        if bucket is None:
            logger.error(f"Invalid S3 path: {s3_url}")
            return None

        response = s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()

        # Determine file type from extension
        key_lower = key.lower()
        result = None

        if key_lower.endswith(".csv"):
            result = pd.read_csv(io.BytesIO(body))

        elif key_lower.endswith(".tsv") or key_lower.endswith(".txt"):
            result = pd.read_csv(io.BytesIO(body), sep="\t")

        elif key_lower.endswith(".json"):
            result = pd.read_json(io.BytesIO(body))

        elif key_lower.endswith(".xlsx") or key_lower.endswith(".xls"):
            # Load Excel file with prepended metadata detection
            # Use xlrd engine for .xls files (legacy binary format)
            engine = "xlrd" if key_lower.endswith(".xls") else None

            # Step 1: Load raw to detect prepended rows
            target_sheet = url_sheet_name if url_sheet_name else 0
            raw_df = pd.read_excel(
                io.BytesIO(body), sheet_name=target_sheet, header=None, engine=engine
            )

            # Step 2: Detect prepended metadata rows
            raw_rows = raw_df.values.tolist()
            detection = _detect_prepended_metadata(raw_rows)
            skip_count = detection.get("skip_count", 0)

            if skip_count > 0:
                logger.info(
                    f"Detected metadata rows in sheet '{target_sheet}'. "
                    f"Skipping first {skip_count} rows."
                )
                # Reload with proper header handling
                result = pd.read_excel(
                    io.BytesIO(body),
                    sheet_name=target_sheet,
                    skiprows=range(skip_count),
                    header=0,
                    engine=engine,
                )
            elif url_sheet_name:
                # Specific sheet requested, no metadata
                result = pd.read_excel(io.BytesIO(body), sheet_name=url_sheet_name, engine=engine)
            else:
                # Load all sheets to handle multi-sheet files like production
                result = pd.read_excel(io.BytesIO(body), sheet_name=None, engine=engine)

        elif key_lower.endswith(".parquet"):
            result = pd.read_parquet(io.BytesIO(body))

        elif key_lower.endswith(".dat") or key_lower.endswith(".bin"):
            # Try to load as binary numeric data
            try:
                # Try float32 first (most common for scientific data)
                data = np.frombuffer(body, dtype="<f4")
                # Check if it's a perfect square for matrix data
                n = len(data)
                sqrt_n = int(np.sqrt(n))
                if sqrt_n * sqrt_n == n and sqrt_n > 1:
                    data = data.reshape((sqrt_n, sqrt_n))
                result = pd.DataFrame(data)
            except Exception:
                # Fall back to text parsing
                result = pd.read_csv(io.BytesIO(body), sep=None, engine="python")

        else:
            # Default: try CSV
            result = pd.read_csv(io.BytesIO(body))

        # Handle dict return type (multi-sheet Excel files)
        # This mirrors production behavior in lib/s3_utils.py
        if isinstance(result, dict):
            # Handle excel_workbook dict structure: {"type": "excel_workbook", "sheets": {...}, ...}
            if result.get("type") == "excel_workbook":
                sheets = result.get("sheets", {})
                if url_sheet_name and url_sheet_name in sheets:
                    df = sheets[url_sheet_name]
                    logger.info(f"Extracted sheet '{url_sheet_name}' from workbook")
                elif sheets:
                    sheet_order = result.get("sheet_order", list(sheets.keys()))
                    first_sheet = (
                        sheet_order[0] if sheet_order else next(iter(sheets.keys()))
                    )
                    df = sheets[first_sheet]
                    logger.info(f"Using first sheet '{first_sheet}' from workbook")
                else:
                    logger.warning("Empty sheets in workbook")
                    return None
            # Handle simple dict of sheet_name -> DataFrame (from pd.read_excel with sheet_name=None)
            elif url_sheet_name and url_sheet_name in result:
                df = result[url_sheet_name]
                logger.info(
                    f"Extracted sheet '{url_sheet_name}' from multi-sheet result"
                )
            elif result:
                # Find first DataFrame value (skip metadata keys like "type", "sheet_order")
                df = None
                for sheet_key, value in result.items():
                    if isinstance(value, pd.DataFrame):
                        df = value
                        logger.info(
                            f"Using first sheet '{sheet_key}' from multi-sheet result"
                        )
                        break
                if df is None:
                    logger.warning("No DataFrame found in dict result")
                    return None
            else:
                logger.warning("Empty dict returned from loader")
                return None
        else:
            df = result

        # Apply post-processing (matches production behavior in lib/s3_utils.py)
        if df is not None and isinstance(df, pd.DataFrame):
            # Handle headerless CSVs - assign meaningful column names
            # This ensures visualization code gets the same column names as FILE SUMMARIES
            df, was_headerless = _detect_and_handle_headerless_data(df, key)
            if was_headerless:
                logger.info(
                    f"Applied headerless data handling, columns now: {list(df.columns)}"
                )

            # Clean problematic columns
            df = _clean_dataframe_columns(df, key)
            logger.info(f"Loaded dataframe with shape: {df.shape}")

        return df

    except ClientError as e:
        logger.error(f"S3 error loading {s3_url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading {s3_url}: {e}")
        return None


def load_s3_json(bucket_name: str, s3_key: str) -> dict | None:
    """
    Load a JSON file from S3.

    Matches production signature: load_s3_json(bucket_name, s3_key)

    Args:
        bucket_name: S3 bucket name
        s3_key: S3 object key

    Returns:
        dict: Parsed JSON data, or None on failure (matches production behavior)
    """
    import json

    s3 = _get_s3_client()

    try:
        response = s3.get_object(Bucket=bucket_name, Key=s3_key)
        body = response["Body"].read()
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from s3://{bucket_name}/{s3_key}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load JSON from s3://{bucket_name}/{s3_key}: {e}")
        return None


def create_s3_json_function(bucket_name: str):
    """
    Create a configured JSON load function with bucket pre-set.

    Matches production function in lib/s3_utils.py.

    Args:
        bucket_name: S3 bucket name to bind

    Returns:
        callable: Function(s3_key) that loads JSON from the configured bucket
    """

    def json_function(s3_key: str) -> dict | None:
        return load_s3_json(bucket_name, s3_key)

    return json_function


def save_plot_to_s3(fig, filename: str, bucket_name: str, s3_prefix: str) -> str | None:
    """
    Save matplotlib figure to S3 with lightweight error handling.

    Matches production function with fallback quality save.

    Args:
        fig: matplotlib Figure object
        filename: Filename for the plot (e.g., "01_distribution.png")
        bucket_name: S3 bucket name
        s3_prefix: S3 key prefix

    Returns:
        str: S3 key of saved file, or None on failure
    """
    import matplotlib.pyplot as plt

    s3 = _get_s3_client()

    # Ensure proper path separator between prefix and filename
    s3_key = f"{s3_prefix.rstrip('/')}/{filename}"

    try:
        buffer = io.BytesIO()
        # Try high-quality save first
        fig.savefig(
            buffer, format="png", dpi=300, bbox_inches="tight", facecolor="white"
        )
        buffer.seek(0)

        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType="image/png",
        )
        logger.info(f"Saved visualization: {s3_key}")
        plt.close(fig)
        return s3_key

    except Exception as e:
        logger.warning(
            f"High-quality save failed for {filename}, using basic save: {str(e)}"
        )
        try:
            buffer = io.BytesIO()
            # Fallback: minimal options to avoid PIL encoding issues
            fig.savefig(buffer, format="png", dpi=100)
            buffer.seek(0)

            s3.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType="image/png",
            )
            logger.info(f"Saved visualization (basic quality): {s3_key}")
            plt.close(fig)
            return s3_key

        except Exception as e2:
            logger.error(f"Failed to save plot {filename}: {str(e2)}")
            plt.close(fig)
            return None


def create_s3_save_function(bucket_name: str, s3_prefix: str):
    """
    Create a configured save function with bucket and prefix pre-set.

    Matches production function in lib/s3_utils.py.

    Args:
        bucket_name: S3 bucket name
        s3_prefix: S3 key prefix for uploads

    Returns:
        callable: Function(fig, filename) that saves matplotlib figures to S3
    """

    def save_function(fig, filename: str) -> str | None:
        return save_plot_to_s3(fig, filename, bucket_name, s3_prefix)

    return save_function
