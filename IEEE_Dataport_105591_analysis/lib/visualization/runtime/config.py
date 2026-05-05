"""
Matplotlib configuration and font path management for visualization scripts.

This module handles headless matplotlib setup and non-Latin script font paths.
"""

import logging

logger = logging.getLogger(__name__)

# Font paths imported from lib.config (single source of truth)
from lib.config import (
    DEVANAGARI_FONT_PATH,
    CJK_FONT_PATH,
    ARABIC_FONT_PATH,
    BENGALI_FONT_PATH,
    THAI_FONT_PATH,
    TAMIL_FONT_PATH,
    HEBREW_FONT_PATH,
)


def get_font_paths():
    """
    Get all font paths as a dictionary.

    Returns:
        dict: Mapping of font names to their paths
    """
    return {
        "devanagari": DEVANAGARI_FONT_PATH,
        "cjk": CJK_FONT_PATH,
        "arabic": ARABIC_FONT_PATH,
        "bengali": BENGALI_FONT_PATH,
        "thai": THAI_FONT_PATH,
        "tamil": TAMIL_FONT_PATH,
        "hebrew": HEBREW_FONT_PATH,
    }


def configure_matplotlib():
    """
    Configure matplotlib for headless operation with proper font settings.

    This should be called at the start of visualization scripts.
    Sets up the Agg backend, turns off interactive mode, and configures
    default font and figure settings.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Turn off interactive mode
    plt.ioff()

    # Configure default font and figure settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.max_open_warning'] = 0

    logger.debug("Matplotlib configured for headless operation")
