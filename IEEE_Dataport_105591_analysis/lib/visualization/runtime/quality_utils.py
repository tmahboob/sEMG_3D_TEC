"""
Quality control utilities for visualization scripts.

These functions ensure high-quality visualizations by handling edge cases
like too many categories, uniform values, and dynamic sizing.
"""

import logging
import pandas as pd

from lib.config import (
    DEFAULT_MAX_CATEGORIES,
    CATEGORY_AGGREGATE_LABEL,
    MIN_DISPLAY_CATEGORIES,
    CATEGORY_TIER_1_THRESHOLD,
    CATEGORY_TIER_2_THRESHOLD,
    DISPLAY_TIER_1_LIMIT,
    DISPLAY_TIER_2_LIMIT,
    DISPLAY_TIER_DEFAULT_LIMIT,
)

logger = logging.getLogger(__name__)


def limit_to_top_n_categories(
    df, column, n=None, aggregate_other=True, other_label=None
):
    """
    Limit categorical column to top N values by frequency.

    Args:
        df: DataFrame containing the column
        column: Name of categorical column to limit
        n: Number of top categories to keep (default: DEFAULT_MAX_CATEGORIES from config)
        aggregate_other: If True, combine remaining categories into 'other_label'
        other_label: Label for aggregated categories (default: CATEGORY_AGGREGATE_LABEL from config)

    Returns:
        DataFrame with limited categories

    Example:
        # Before plotting with many classes
        if df['label'].nunique() > DEFAULT_MAX_CATEGORIES:
            display_limit = get_display_limit_for_classes(df['label'].nunique())
            df_plot = limit_to_top_n_categories(df, 'label', n=display_limit)
    """
    # Use config defaults if not specified
    if n is None:
        n = DEFAULT_MAX_CATEGORIES
    if other_label is None:
        other_label = CATEGORY_AGGREGATE_LABEL

    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return df

    unique_count = df[column].nunique()
    if unique_count <= n:
        return df

    top_categories = df[column].value_counts().head(n).index.tolist()

    if aggregate_other:
        df_copy = df.copy()
        df_copy[column] = df_copy[column].apply(
            lambda x: x if x in top_categories else other_label
        )
        logger.info(
            f"Limited '{column}' from {unique_count} to {n} categories (+ '{other_label}')"
        )
        return df_copy
    else:
        df_filtered = df[df[column].isin(top_categories)].copy()
        logger.info(
            f"Limited '{column}' from {unique_count} to {n} categories (filtered)"
        )
        return df_filtered


def has_variance(df, column, min_std=0.01):
    """
    Check if numeric column has meaningful variance.

    Prevents creating uninformative plots for uniform/constant values.

    Args:
        df: DataFrame containing the column
        column: Name of numeric column to check
        min_std: Minimum standard deviation to consider as "has variance"

    Returns:
        True if column has variance, False otherwise

    Example:
        # Skip aspect ratio plot if all images have same aspect ratio
        if not has_variance(df, 'aspect_ratio'):
            logger.info("Skipping aspect ratio plot - no variance")
    """
    if column not in df.columns:
        return False
    if not pd.api.types.is_numeric_dtype(df[column]):
        return False

    col_std = df[column].std()
    if pd.isna(col_std) or col_std <= min_std:
        logger.info(f"Column '{column}' has low/no variance (std={col_std:.4f})")
        return False
    return True


def calculate_optimal_figure_height(
    n_categories, base_height=6, pixels_per_category=30
):
    """
    Calculate figure height that scales with number of categories.

    Prevents cramped visualizations when displaying many categories.

    Args:
        n_categories: Number of categories to display
        base_height: Base height for small number of categories
        pixels_per_category: Additional pixels per category

    Returns:
        Optimal figure height in inches

    Example:
        # Dynamic sizing for horizontal bar chart
        n_classes = df['label'].nunique()
        fig_height = calculate_optimal_figure_height(n_classes)
        fig, ax = plt.subplots(figsize=(12, fig_height))
    """
    min_height, max_height = 6, 20
    calculated_height = base_height + (n_categories * pixels_per_category / 100)
    optimal_height = max(min_height, min(calculated_height, max_height))
    logger.info(
        f"Calculated optimal height: {optimal_height:.1f} inches for {n_categories} categories"
    )
    return optimal_height


def get_display_limit_for_classes(n_classes):
    """
    Determine how many classes to show based on total count.

    Uses tiered approach to balance detail vs readability.
    Thresholds are configured in lib/config.py.

    Args:
        n_classes: Total number of classes in dataset

    Returns:
        Recommended number of classes to display

    Example:
        if df['label'].nunique() > DEFAULT_MAX_CATEGORIES:
            display_limit = get_display_limit_for_classes(df['label'].nunique())
            df_plot = limit_to_top_n_categories(df, 'label', n=display_limit)
    """
    if n_classes <= MIN_DISPLAY_CATEGORIES:
        return n_classes
    elif n_classes <= CATEGORY_TIER_1_THRESHOLD:
        return DISPLAY_TIER_1_LIMIT
    elif n_classes <= CATEGORY_TIER_2_THRESHOLD:
        return DISPLAY_TIER_2_LIMIT
    else:
        return DISPLAY_TIER_DEFAULT_LIMIT


def figure_has_data(fig) -> bool:
    """
    Check whether a matplotlib Figure contains any plotted data.

    Inspects every Axes in the figure for lines, patches (bars, histograms,
    pie wedges), images (imshow, heatmaps), collections (scatter, contour,
    fill_between), and tables. Returns True if ANY axis has ANY plotted
    element — deliberately liberal to avoid false positives.

    Does NOT count titles, axis labels, ticks, or legends as "data."

    Args:
        fig: matplotlib Figure object

    Returns:
        True if the figure contains plotted data, False if all axes are empty.
    """
    for ax in fig.get_axes():
        if ax.get_lines():
            return True
        if ax.patches:
            return True
        if ax.get_images():
            return True
        if ax.collections:
            return True
        if ax.tables:
            return True
        if hasattr(ax, "containers") and ax.containers:
            return True
    return False
