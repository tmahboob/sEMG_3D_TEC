"""
S3 image loading utilities for visualization scripts.

These utilities handle loading images from S3 with automatic path resolution
for datasets where metadata references differ from actual S3 structure.
"""

import logging
import boto3
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# S3 client singleton (lazy loaded)
_s3_client = None

# Cache for directory mappings to avoid repeated S3 list calls
_s3_dir_cache = {}


def _get_s3_client():
    """Get or create S3 client singleton."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client


def resolve_image_path(dataset_base_path, relative_path):
    """
    Resolve metadata image paths to actual S3 paths, handling _zip suffixes.

    Metadata CSVs often reference "Directory/file.jpg" but actual S3 structure
    has "Directory_zip/file.jpg" after unzipping. This function auto-detects
    the correct path.

    Args:
        dataset_base_path (str): Base S3 path like "s3://bucket/dataset_id/"
        relative_path (str): Relative path from metadata like "Papaya Leaves/Papaya_1.jpg"

    Returns:
        str: Full corrected S3 path, or original if no mapping needed
    """
    global _s3_dir_cache
    s3_client = _get_s3_client()

    # Parse base path
    if dataset_base_path.startswith('s3://'):
        dataset_base_path_no_prefix = dataset_base_path[5:]
    else:
        dataset_base_path_no_prefix = dataset_base_path

    parts = dataset_base_path_no_prefix.split('/', 1)
    if len(parts) != 2:
        return dataset_base_path + relative_path

    bucket, prefix = parts
    prefix = prefix.rstrip('/')  # Remove trailing slash to avoid double slashes in S3 keys

    # Extract directory from relative path (e.g., "Papaya Leaves" from "Papaya Leaves/file.jpg")
    if '/' in relative_path:
        dir_name, file_name = relative_path.rsplit('/', 1)

        # Check cache first
        cache_key = f"{bucket}/{prefix}/{dir_name}"
        if cache_key in _s3_dir_cache:
            actual_dir = _s3_dir_cache[cache_key]
            return f"s3://{bucket}/{prefix}/{actual_dir}/{file_name}"

        # Build variations list - handles both simple and nested directory paths
        # For nested paths like "mammograms/BIRAD 1", we need to try _zip on different levels
        variations = [dir_name]  # Original path first

        dir_components = dir_name.split('/')
        if len(dir_components) == 1:
            # Simple case: single directory level
            variations.extend([
                f"{dir_name}_zip",
                f"{dir_name}_0_zip",
                f"_zip-contents/{dir_name}"
            ])
        else:
            # Nested directories: try _zip suffix on first directory level
            # e.g., "mammograms/BIRAD 1" -> "mammograms_zip/BIRAD 1"
            first_dir = dir_components[0]
            rest_dirs = '/'.join(dir_components[1:])

            variations.extend([
                f"{first_dir}_zip/{rest_dirs}",           # mammograms_zip/BIRAD 1
                f"{first_dir}_0_zip/{rest_dirs}",         # mammograms_0_zip/BIRAD 1
                f"{dir_name}_zip",                         # mammograms/BIRAD 1_zip (original behavior)
                f"_zip-contents/{dir_name}",               # _zip-contents/mammograms/BIRAD 1
            ])

            # Also try _zip on last component for completeness
            last_dir = dir_components[-1]
            parent_dirs = '/'.join(dir_components[:-1])
            variations.append(f"{parent_dirs}/{last_dir}_zip")

        for variant in variations:
            try:
                test_key = f"{prefix}/{variant}/{file_name}"
                s3_client.head_object(Bucket=bucket, Key=test_key)
                # Found it! Cache and return
                _s3_dir_cache[cache_key] = variant
                return f"s3://{bucket}/{test_key}"
            except Exception:
                continue

        # If no variation works, try listing directories to find match
        try:
            # For nested paths, list from the first directory level
            list_prefix = prefix
            first_dir_target = dir_components[0] if len(dir_components) > 1 else dir_name

            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=list_prefix + '/',
                Delimiter='/'
            )
            if 'CommonPrefixes' in response:
                for common_prefix in response['CommonPrefixes']:
                    dir_path = common_prefix['Prefix'].rstrip('/')
                    dir_only = dir_path.split('/')[-1]
                    # Check if this directory name matches our first target
                    if first_dir_target.lower() in dir_only.lower() or dir_only.startswith(first_dir_target):
                        # Build the full path with the matched directory
                        if len(dir_components) > 1:
                            rest_dirs = '/'.join(dir_components[1:])
                            actual_dir = f"{dir_only}/{rest_dirs}"
                        else:
                            actual_dir = dir_only
                        _s3_dir_cache[cache_key] = actual_dir
                        return f"s3://{bucket}/{prefix}/{actual_dir}/{file_name}"
        except Exception as e:
            logger.warning(f"Could not list S3 directories: {e}")

    # Fallback: return original path
    return dataset_base_path + relative_path


def load_image_from_s3(s3_path, dataset_base_path=None, resolve_path=True):
    """
    Load an image from S3 using boto3. PIL cannot directly open S3 URLs.

    Automatically handles path resolution for datasets where metadata references
    differ from actual S3 structure (e.g., "Directory/" vs "Directory_zip/").

    Args:
        s3_path (str): Full or partial S3 path
        dataset_base_path (str): Optional base path for path resolution
        resolve_path (bool): Whether to attempt automatic path resolution

    Returns:
        PIL.Image or None: Loaded image, or None if loading failed
    """
    s3_client = _get_s3_client()

    try:
        # Attempt path resolution if base path provided
        if resolve_path and dataset_base_path and not s3_path.startswith('s3://'):
            s3_path = resolve_image_path(dataset_base_path, s3_path)

        # Parse S3 path
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]  # Remove 's3://' prefix

        parts = s3_path.split('/', 1)
        if len(parts) != 2:
            return None

        bucket, key = parts
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        return Image.open(BytesIO(image_data))
    except Exception as e:
        logger.warning(f"Failed to load image from {s3_path}: {e}")
        return None


def clear_path_cache():
    """Clear the S3 directory path cache."""
    global _s3_dir_cache
    _s3_dir_cache = {}
