"""
Local save function for user distribution.

The S3 save function (create_s3_save_function) lives in lib/s3_utils.py
to maintain a single source of truth. This module only contains the
local filesystem save function for user-distributed packages.
"""

import os
import logging

logger = logging.getLogger(__name__)


def create_local_save_function(output_dir='./output'):
    """
    Create a local filesystem save function factory.

    This function returns a save function that saves matplotlib figures
    to the local filesystem instead of S3. Used for user-distributed
    packages that run outside of AWS.

    Args:
        output_dir (str): Directory to save visualizations to (default: './output')

    Returns:
        function: A save_plot_function(fig, filename) that saves to local filesystem

    Example:
        save_plot_function = create_local_save_function('./visualizations')

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        save_plot_function(fig, '01_quadratic.png')
    """
    import matplotlib.pyplot as plt

    def save_plot_function(fig, filename):
        """
        Save a matplotlib figure to the local filesystem.

        Args:
            fig: matplotlib Figure object
            filename (str): Filename for the plot (e.g., "01_distribution.png")

        Returns:
            str: Path to saved file, or None on failure
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            logger.info(f"Saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            plt.close(fig)
            return None

    return save_plot_function
