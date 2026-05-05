"""
Centralized MATLAB file loading utilities.

Handles both MATLAB v7.0 (scipy) and v7.3/HDF5 (h5py) file formats.
"""
import logging
from io import BytesIO
from typing import Optional

import h5py
import numpy as np
import scipy.io

logger = logging.getLogger(__name__)

# HDF5 magic number (first 8 bytes of HDF5 files)
HDF5_SIGNATURE = b'\x89HDF\r\n\x1a\n'


def _load_hdf5_item(item, hdf_file, visited=None):
    """
    Recursively load HDF5 items (handles datasets, groups/structs, and cell arrays).

    MATLAB v7.3 files store data in HDF5 format where:
    - Simple arrays are h5py.Dataset objects
    - MATLAB structs are h5py.Group objects containing nested items
    - MATLAB cell arrays use HDF5 object references that must be dereferenced

    Args:
        item: h5py.Dataset or h5py.Group object
        hdf_file: The parent h5py.File object, needed for dereferencing cell arrays
        visited: Set of visited object IDs to prevent infinite recursion from circular refs

    Returns:
        numpy.ndarray for datasets, dict for groups, list for cell arrays,
        or the item unchanged
    """
    if visited is None:
        visited = set()

    # Track visited objects to prevent infinite recursion from circular references
    if hasattr(item, 'id'):
        item_id = item.id.id if hasattr(item.id, 'id') else id(item)
        if item_id in visited:
            logger.warning(f"Circular reference detected for '{item.name}', it will be skipped.")
            return None
        visited.add(item_id)

    if isinstance(item, h5py.Dataset):
        # Check for HDF5 object references (used in MATLAB cell arrays)
        if h5py.check_dtype(ref=item.dtype):
            return [_load_hdf5_item(hdf_file[ref], hdf_file, visited) for ref in item[()].flatten()]
        return item[()]
    elif isinstance(item, h5py.Group):
        return {key: _load_hdf5_item(item[key], hdf_file, visited) for key in item.keys()}
    else:
        return item


def load_mat_file(file_content: bytes, filename: str = "unknown") -> Optional[dict]:
    """
    Load a MATLAB .mat file from bytes, handling both v7.0 and v7.3 formats.

    MATLAB v7.0 files are loaded using scipy.io.loadmat().
    MATLAB v7.3 files use HDF5 format internally and require h5py.

    h5py can read directly from BytesIO (no temp files needed since h5py 2.9).

    Args:
        file_content: Raw bytes of the .mat file
        filename: Filename for logging purposes

    Returns:
        Dictionary of MATLAB variables, or None if loading fails.
        Keys starting with '__' or '#' (internal/metadata) are excluded.
    """
    file_stream = BytesIO(file_content)
    mat_data = None

    # Check for HDF5 magic number (more reliable than exception message parsing)
    is_hdf5 = file_content.startswith(HDF5_SIGNATURE)

    # Try scipy first (handles v7.0 and earlier)
    try:
        mat_data = scipy.io.loadmat(file_stream)
        logger.info(f"Loaded {filename} using scipy (v7.0 format)")
    except NotImplementedError as e:
        if is_hdf5 or "HDF reader" in str(e) or "v7.3" in str(e):
            # MATLAB v7.3 file - use h5py with BytesIO (no temp file needed)
            logger.info(f"MATLAB v7.3 detected for {filename}, using h5py")
            # h5py can read directly from BytesIO (since h5py 2.9)
            file_stream.seek(0)  # Reset stream position
            with h5py.File(file_stream, 'r') as f:
                mat_data = {}
                for var_name in f.keys():
                    if var_name.startswith('#'):  # Skip HDF5 metadata
                        continue
                    mat_data[var_name] = _load_hdf5_item(f[var_name], f)
                logger.info(f"Loaded {len(mat_data)} variables from HDF5/v7.3")
        else:
            raise

    if mat_data is None:
        return None

    # Filter out internal MATLAB metadata keys
    return {k: v for k, v in mat_data.items() if not k.startswith('__') and not k.startswith('#')}
