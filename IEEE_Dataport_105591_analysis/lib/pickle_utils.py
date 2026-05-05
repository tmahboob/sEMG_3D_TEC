"""
Safe pickle loading utilities with security restrictions.

Provides a RestrictedUnpickler that only allows safe types (numpy arrays, basic Python types)
to prevent arbitrary code execution from malicious pickle files.
"""
import pickle
import logging
from io import BytesIO
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Whitelist of safe classes that can be unpickled
# These are the standard types used in ML datasets like CIFAR-10
SAFE_CLASSES = {
    # NumPy types
    ('numpy', 'ndarray'),
    ('numpy', 'dtype'),
    ('numpy.core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'),
    # Basic Python types
    ('builtins', 'dict'),
    ('builtins', 'list'),
    ('builtins', 'tuple'),
    ('builtins', 'bytes'),
    ('builtins', 'set'),
    ('builtins', 'frozenset'),
    # For Python 2 pickled files (CIFAR uses encoding='bytes')
    ('__builtin__', 'dict'),
    ('__builtin__', 'list'),
}


class RestrictedUnpickler(pickle.Unpickler):
    """
    Unpickler that only allows safe numpy/builtin types.

    This prevents arbitrary code execution by blocking imports of
    dangerous classes like os, subprocess, etc.
    """

    def find_class(self, module: str, name: str) -> Any:
        # Check explicit whitelist first
        if (module, name) in SAFE_CLASSES:
            return super().find_class(module, name)

        # Allow numpy submodules for dtype/array reconstruction
        if module.startswith('numpy'):
            if name in ('dtype', 'ndarray', '_reconstruct', 'scalar'):
                return super().find_class(module, name)

        # Block everything else
        raise pickle.UnpicklingError(
            f"Blocked unsafe class: {module}.{name}. "
            f"Only numpy arrays and basic Python types are allowed."
        )


def load_pickle_safe(file_content: bytes, filename: str = "unknown") -> Optional[Any]:
    """
    Load pickle file with security restrictions.

    Only allows numpy arrays and basic Python types (dict, list, tuple, bytes).
    Blocks arbitrary code execution by preventing import of dangerous modules.

    Args:
        file_content: Raw bytes of the pickle file
        filename: Filename for logging purposes

    Returns:
        Unpickled data (typically dict with 'data' and 'labels' keys for CIFAR),
        or None if loading fails due to security restrictions.

    Example:
        >>> batch = load_pickle_safe(file_bytes, "data_batch_1")
        >>> if batch:
        ...     data = batch.get(b'data') or batch.get('data')
        ...     labels = batch.get(b'labels') or batch.get('labels')
    """
    try:
        stream = BytesIO(file_content)
        # Use encoding='bytes' for Python 2 pickled files (like CIFAR)
        unpickler = RestrictedUnpickler(stream, encoding='bytes')
        data = unpickler.load()
        logger.info(f"Successfully loaded pickle file {filename} with safe unpickler")
        return data
    except pickle.UnpicklingError as e:
        logger.warning(f"Blocked unsafe pickle content in {filename}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load pickle file {filename}: {e}")
        return None
