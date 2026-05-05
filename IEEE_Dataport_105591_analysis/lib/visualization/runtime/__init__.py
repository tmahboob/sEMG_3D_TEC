"""
Runtime support module for user package visualization scripts.

This is the user package version - it does NOT re-export from s3_utils to avoid
dependency chains. Users import from lib.s3_utils directly.
"""

# Configuration
from lib.visualization.runtime.config import (
    configure_matplotlib,
    get_font_paths,
    DEVANAGARI_FONT_PATH,
    CJK_FONT_PATH,
    ARABIC_FONT_PATH,
    BENGALI_FONT_PATH,
    THAI_FONT_PATH,
    TAMIL_FONT_PATH,
    HEBREW_FONT_PATH,
)

# Quality control utilities
from lib.visualization.runtime.quality_utils import (
    limit_to_top_n_categories,
    has_variance,
    calculate_optimal_figure_height,
    get_display_limit_for_classes,
)

# Image loading utilities
from lib.visualization.runtime.image_utils import (
    resolve_image_path,
    load_image_from_s3,
)

# Save utilities - local only (S3 save is in s3_utils)
from lib.visualization.runtime.save_utils import create_local_save_function

# Specialized data loaders
from lib.matlab_utils import load_mat_file
from lib.pickle_utils import load_pickle_safe

__all__ = [
    # Configuration
    "configure_matplotlib",
    "get_font_paths",
    "DEVANAGARI_FONT_PATH",
    "CJK_FONT_PATH",
    "ARABIC_FONT_PATH",
    "BENGALI_FONT_PATH",
    "THAI_FONT_PATH",
    "TAMIL_FONT_PATH",
    "HEBREW_FONT_PATH",
    # Quality utilities
    "limit_to_top_n_categories",
    "has_variance",
    "calculate_optimal_figure_height",
    "get_display_limit_for_classes",
    # Image utilities
    "resolve_image_path",
    "load_image_from_s3",
    # Save functions
    "create_local_save_function",
    # Specialized loaders
    "load_mat_file",
    "load_pickle_safe",
]
