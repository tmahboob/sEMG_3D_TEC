import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from urllib.parse import unquote
from PIL import Image
import boto3
from io import BytesIO
from os.path import basename, dirname, join, splitext
from lib.visualization.runtime import configure_matplotlib, limit_to_top_n_categories, has_variance, calculate_optimal_figure_height, get_display_limit_for_classes, resolve_image_path, load_image_from_s3, DEVANAGARI_FONT_PATH, CJK_FONT_PATH, ARABIC_FONT_PATH, BENGALI_FONT_PATH, THAI_FONT_PATH, TAMIL_FONT_PATH, HEBREW_FONT_PATH
from lib.s3_utils import load_data_file
configure_matplotlib()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Local save configuration (visualizations saved to ./output/)
from lib.visualization.runtime import create_local_save_function
save_plot_function = create_local_save_function('./output')
# S3 client for image loading operations
s3_client = boto3.client('s3')
_s3_dir_cache = {}
# No large text corpus data
# sEMG & 3D Hand Keypoint Dataset Visualization
# Multimodal biosignal analysis: muscle activation patterns and hand movements
from urllib.parse import unquote
import warnings
import numpy as np
# ── Load File 1: sEMG Time-Series ──────────────────────────────────────────
path_emg = 's3://ieee-dataport/data/1319017/105591/Dataset_filterd_50_1(channel_data).csv'
path_combined = 's3://ieee-dataport/data/1319017/105591/sEMG_prediction_dataset(channel_and_3Dkeypoints).csv'
df_emg = load_data_file(unquote(path_emg))
if not isinstance(df_emg, pd.DataFrame) or df_emg.empty:
    logger.error('Failed to load sEMG channel data')
else:
    logger.info(f'Loaded sEMG file: {df_emg.shape}, columns: {list(df_emg.columns)}')
df_combined = load_data_file(unquote(path_combined))
if not isinstance(df_combined, pd.DataFrame) or df_combined.empty:
    logger.error('Failed to load combined feature file')
else:
    logger.info(f'Loaded combined file: {df_combined.shape}, columns: {list(df_combined.columns)}')
# ── Preprocessing ───────────────────────────────────────────────────────────
emg_channels = [f'EXG Channel {i}' for i in range(8)]
keypoint_cols = [f'Measurement_{i}' for i in range(10, 28)]
semg_feature_cols = [f'Measurement_{i}' for i in range(2, 10)]
if isinstance(df_emg, pd.DataFrame):
    # Parse timestamps
    df_emg['Timestamp (Formatted)'] = pd.to_datetime(df_emg['Timestamp (Formatted)'].str.strip(), errors='coerce')
    # Remove initialization artifact (first ~50 samples where values are extreme)
    artifact_threshold = 3000  # µV
    artifact_mask = (df_emg[emg_channels].abs() > artifact_threshold).any(axis=1)
    df_clean = df_emg[~artifact_mask].copy()
    logger.info(f'After artifact removal: {len(df_clean)} rows (removed {artifact_mask.sum()} artifact rows)')
    # Label mapping
    label_map = {0.0: 'Rest', 1.0: 'Gesture 1', 2.0: 'Gesture 2', 3.0: 'Gesture 3', 4.0: 'Gesture 4', 5.0: 'Gesture 5'}
    df_clean['Gesture'] = df_clean['Label'].map(label_map).fillna('Unknown')
# ── Visualization 1: 8-Channel sEMG Temporal Overview ──────────────────────
try:
    if isinstance(df_emg, pd.DataFrame):
        # Sample 5000 points for plotting speed
        n_plot = min(5000, len(df_clean))
        df_plot = df_clean.iloc[:n_plot].copy()
        fig, axes = plt.subplots(8, 1, figsize=(16, 14), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, 8))
        for i, (ch, ax) in enumerate(zip(emg_channels, axes)):
            if ch in df_plot.columns:
                ax.plot(df_plot.index, df_plot[ch], color=colors[i], linewidth=0.6, alpha=0.85)
                ax.set_ylabel(f'Ch {i}\n(µV)', fontsize=8, rotation=0, labelpad=40, va='center')
                ax.grid(alpha=0.25, linewidth=0.5)
                ax.axhline(0, color='gray', linewidth=0.4, linestyle='--')
                # Shade by gesture label
                for label_val, label_name in label_map.items():
                    mask = df_plot['Label'] == label_val
                    if mask.any():
                        ax.fill_between(df_plot.index, ax.get_ylim()[0], ax.get_ylim()[1], where=mask, alpha=0.08, color=plt.cm.Set2(label_val / 5.0))
        axes[-1].set_xlabel('Sample Index', fontsize=11)
        fig.suptitle('8-Channel sEMG Temporal Overview\n(First 5000 clean samples, shaded by gesture label)', fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        save_plot_function(fig, '01_semg_temporal_overview.png')
        plt.close(fig)
        logger.info('Created sEMG temporal overview plot')
except Exception as e:
    logger.error(f'Failed to create sEMG temporal overview: {e}')
# ── Visualization 2: Per-Gesture Channel Activation Heatmap ────────────────
try:
    if isinstance(df_emg, pd.DataFrame):
        # Compute RMS per channel per gesture label
        rms_data = {}
        for label_val, label_name in label_map.items():
            subset = df_clean[df_clean['Label'] == label_val]
            if len(subset) > 10:
                rms_vals = []
                for ch in emg_channels:
                    if ch in subset.columns:
                        rms_vals.append(np.sqrt(np.mean(subset[ch].values ** 2)))
                    else:
                        rms_vals.append(np.nan)
                rms_data[label_name] = rms_vals
        if rms_data:
            rms_df = pd.DataFrame(rms_data, index=[f'Ch {i}' for i in range(8)])
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(rms_df.values, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(rms_df.columns)))
            ax.set_xticklabels(rms_df.columns, fontsize=11)
            ax.set_yticks(range(len(rms_df.index)))
            ax.set_yticklabels(rms_df.index, fontsize=11)
            # Annotate cells
            for r in range(rms_df.shape[0]):
                for c in range(rms_df.shape[1]):
                    val = rms_df.values[r, c]
                    if not np.isnan(val):
                        ax.text(c, r, f'{val:.1f}', ha='center', va='center', fontsize=9, color='black' if val < rms_df.values.max() * 0.7 else 'white')
            plt.colorbar(im, ax=ax, label='RMS Amplitude (µV)')
            ax.set_title('Per-Gesture sEMG Channel Activation Heatmap\n(RMS Amplitude per Channel per Gesture)', fontsize=13, fontweight='bold')
            ax.set_xlabel('Gesture Class', fontsize=11)
            ax.set_ylabel('sEMG Channel', fontsize=11)
            plt.tight_layout()
            save_plot_function(fig, '02_gesture_channel_heatmap.png')
            plt.close(fig)
            logger.info('Created per-gesture channel activation heatmap')
except Exception as e:
    logger.error(f'Failed to create gesture channel heatmap: {e}')
# ── Visualization 3: Gesture Label Distribution ────────────────────────────
try:
    if isinstance(df_emg, pd.DataFrame):
        label_counts = df_clean['Gesture'].value_counts().head(25)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_bar = [plt.cm.Set2(i / len(label_counts)) for i in range(len(label_counts))]
        bars = ax.bar(label_counts.index, label_counts.values, color=colors_bar, edgecolor='white', linewidth=1.2)
        ax.tick_params(axis='x', rotation=45)
        max_val = label_counts.values.max()
        for bar, val in zip(bars, label_counts.values):
            pct = val / len(df_clean) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, val + max_val * 0.02, f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='normal')
        ax.set_ylim(top=max_val * 1.18)
        ax.set_xlabel('Gesture Class', fontsize=12)
        ax.set_ylabel('Sample Count', fontsize=12)
        ax.set_title('Gesture Label Distribution in sEMG Dataset\n(After Artifact Removal)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        save_plot_function(fig, '03_gesture_label_distribution.png')
        plt.close(fig)
        logger.info('Created gesture label distribution plot')
except Exception as e:
    logger.error(f'Failed to create gesture label distribution: {e}')
# ── Visualization 4: Channel Amplitude Distribution by Gesture (Violin) ────
try:
    if isinstance(df_emg, pd.DataFrame):
        # Focus on 4 most informative channels (highest std dev)
        channel_stds = {ch: df_clean[ch].std() for ch in emg_channels if ch in df_clean.columns}
        top_channels = sorted(channel_stds, key=channel_stds.get, reverse=True)[:4]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        palette = sns.color_palette('Set2', n_colors=len(label_map))
        for idx, ch in enumerate(top_channels):
            ax = axes[idx]
            plot_data = df_clean[[ch, 'Gesture']].dropna()
            # Clip extreme values for visualization
            q1, q99 = (plot_data[ch].quantile(0.01), plot_data[ch].quantile(0.99))
            plot_data = plot_data[(plot_data[ch] >= q1) & (plot_data[ch] <= q99)]
            if has_variance(plot_data, ch, min_std=0.01):
                sns.violinplot(data=plot_data, x='Gesture', y=ch, ax=ax, palette=palette, inner='quartile', linewidth=1.2)
                ax.set_title(f'{ch}\n(Std: {channel_stds[ch]:.1f} µV)', fontsize=11, fontweight='bold')
                ax.set_xlabel('Gesture', fontsize=10)
                ax.set_ylabel('Amplitude (µV)', fontsize=10)
                ax.tick_params(axis='x', rotation=20)
                ax.grid(axis='y', alpha=0.3)
                ax.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
        fig.suptitle('sEMG Amplitude Distribution by Gesture\n(Top 4 Channels by Variance, 1st–99th Percentile)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_plot_function(fig, '04_channel_amplitude_by_gesture.png')
        plt.close(fig)
        logger.info('Created channel amplitude by gesture violin plot')
except Exception as e:
    logger.error(f'Failed to create channel amplitude violin plot: {e}')
# ── Visualization 5: Cross-Channel Correlation Matrix ──────────────────────
try:
    if isinstance(df_emg, pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        # Overall correlation
        corr_all = df_clean[emg_channels].corr()
        ch_labels = [f'Ch {i}' for i in range(8)]
        mask_upper = np.triu(np.ones_like(corr_all, dtype=bool), k=1)
        sns.heatmap(corr_all, ax=axes[0], cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=True, fmt='.2f', linewidths=0.5, xticklabels=ch_labels, yticklabels=ch_labels, cbar_kws={'label': 'Pearson r'})
        axes[0].set_title('Overall Channel Correlation\n(All Gestures Combined)', fontsize=12, fontweight='bold')
        # Correlation during active gesture (non-rest)
        df_active = df_clean[df_clean['Label'] != 0.0]
        if len(df_active) > 100:
            corr_active = df_active[emg_channels].corr()
            sns.heatmap(corr_active, ax=axes[1], cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=True, fmt='.2f', linewidths=0.5, xticklabels=ch_labels, yticklabels=ch_labels, cbar_kws={'label': 'Pearson r'})
            axes[1].set_title('Channel Correlation During Active Gestures\n(Rest Excluded)', fontsize=12, fontweight='bold')
        plt.suptitle('sEMG Inter-Channel Correlation Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_plot_function(fig, '05_channel_correlation_matrix.png')
        plt.close(fig)
        logger.info('Created cross-channel correlation matrix')
except Exception as e:
    logger.error(f'Failed to create channel correlation matrix: {e}')
# ── Visualization 6: 3D Keypoint Spatial Distribution ──────────────────────
try:
    if isinstance(df_combined, pd.DataFrame):
        # Measurements 10-27 are keypoint coordinates (groups of 3 = x,y,z)
        # Based on analysis: Measurements 10-15 are negative (one cluster), 16-27 positive (another)
        # Interpret as: Meas 10,11,12 = keypoint1 x,y,z; etc.
        n_keypoints = 6  # 18 columns / 3 axes = 6 keypoints
        kp_names = [f'KP{i + 1}' for i in range(n_keypoints)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        colors_kp = plt.cm.tab10(np.linspace(0, 1, n_keypoints))
        for kp_idx in range(n_keypoints):
            ax = axes[kp_idx]
            x_col = f'Measurement_{10 + kp_idx * 3}'
            y_col = f'Measurement_{11 + kp_idx * 3}'
            z_col = f'Measurement_{12 + kp_idx * 3}'
            if x_col in df_combined.columns and y_col in df_combined.columns:
                sample_size = min(3000, len(df_combined))
                df_sample = df_combined.sample(n=sample_size, random_state=42)
                scatter = ax.scatter(df_sample[x_col], df_sample[y_col], c=df_sample[z_col] if z_col in df_combined.columns else 'steelblue', cmap='viridis', alpha=0.4, s=8)
                ax.set_xlabel(f'X (mm)', fontsize=9)
                ax.set_ylabel(f'Y (mm)', fontsize=9)
                ax.set_title(f'Keypoint {kp_idx + 1}\n(color = Z depth)', fontsize=10, fontweight='bold')
                ax.grid(alpha=0.3)
                if z_col in df_combined.columns:
                    plt.colorbar(scatter, ax=ax, label='Z (mm)', pad=0.02)
        fig.suptitle('3D Hand Keypoint Spatial Distribution\n(X-Y plane, colored by Z depth — 3000 sample points)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_plot_function(fig, '06_keypoint_spatial_distribution.png')
        plt.close(fig)
        logger.info('Created 3D keypoint spatial distribution plot')
except Exception as e:
    logger.error(f'Failed to create keypoint spatial distribution: {e}')
# ── Visualization 7: sEMG Features vs Keypoint Coordinates Correlation ─────
try:
    if isinstance(df_combined, pd.DataFrame):
        # Measurements 2-9 = sEMG features, 10-27 = keypoints
        semg_cols = [f'Measurement_{i}' for i in range(2, 10)]
        kp_cols_subset = [f'Measurement_{i}' for i in range(10, 28)]
        available_semg = [c for c in semg_cols if c in df_combined.columns]
        available_kp = [c for c in kp_cols_subset if c in df_combined.columns]
        if available_semg and available_kp:
            df_corr = df_combined[available_semg + available_kp].dropna()
            cross_corr = df_corr[available_semg].corrwith(df_corr[available_kp[0]]).to_frame()
            # Full cross-correlation matrix: sEMG features vs keypoints
            cross_matrix = pd.DataFrame(index=available_semg, columns=available_kp)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                for sc in available_semg:
                    for kc in available_kp:
                        if df_corr[sc].std() > 0 and df_corr[kc].std() > 0:
                            cross_matrix.loc[sc, kc] = df_corr[sc].corr(df_corr[kc])
                        else:
                            cross_matrix.loc[sc, kc] = 0.0
            cross_matrix = cross_matrix.astype(float)
            semg_labels = [f'sEMG {i + 1}' for i in range(len(available_semg))]
            kp_labels = [f"KP{i // 3 + 1}{'XYZ'[i % 3]}" for i in range(len(available_kp))]
            fig, ax = plt.subplots(figsize=(16, 6))
            im = ax.imshow(cross_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(range(len(kp_labels)))
            ax.set_xticklabels(kp_labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticks(range(len(semg_labels)))
            ax.set_yticklabels(semg_labels, fontsize=10)
            plt.colorbar(im, ax=ax, label='Pearson Correlation', shrink=0.8)
            ax.set_title('Cross-Modal Correlation: sEMG Features vs 3D Hand Keypoints\n(Pearson r — reveals muscle-movement coupling)', fontsize=13, fontweight='bold')
            ax.set_xlabel('3D Keypoint Coordinates (X/Y/Z per keypoint)', fontsize=11)
            ax.set_ylabel('sEMG Channel Features', fontsize=11)
            plt.tight_layout()
            save_plot_function(fig, '07_semg_keypoint_cross_correlation.png')
            plt.close(fig)
            logger.info('Created sEMG-keypoint cross-correlation heatmap')
except Exception as e:
    logger.error(f'Failed to create sEMG-keypoint cross-correlation: {e}')
# ── Visualization 8: RMS Feature Extraction & Gesture Fingerprints ─────────
try:
    if isinstance(df_emg, pd.DataFrame):
        # Compute windowed RMS for each channel (window = 250 samples ~ 1 second)
        window_size = 250
        step_size = 125  # 50% overlap
        rms_windows = []
        labels_windows = []
        for start in range(0, len(df_clean) - window_size, step_size):
            window = df_clean.iloc[start:start + window_size]
            rms_row = {}
            for ch in emg_channels:
                if ch in window.columns:
                    rms_row[ch] = np.sqrt(np.mean(window[ch].values ** 2))
            # Majority label in window
            if 'Label' in window.columns:
                majority_label = window['Label'].mode()[0]
                labels_windows.append(majority_label)
            else:
                labels_windows.append(0.0)
            rms_windows.append(rms_row)
        df_rms = pd.DataFrame(rms_windows)
        df_rms['Label'] = labels_windows
        df_rms['Gesture'] = df_rms['Label'].map(label_map).fillna('Unknown')
        # Radar/spider chart for gesture fingerprints
        gesture_profiles = df_rms.groupby('Gesture')[emg_channels].mean()
        ch_labels_short = [f'Ch{i}' for i in range(8)]
        n_gestures = len(gesture_profiles)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        colors_radar = plt.cm.Set1(np.linspace(0, 1, n_gestures))
        for (gesture_name, row), color in zip(gesture_profiles.iterrows(), colors_radar):
            values = row.values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=gesture_name, color=color)
            ax.fill(angles, values, alpha=0.12, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(ch_labels_short, fontsize=12, fontweight='bold')
        ax.set_title('sEMG Gesture Fingerprints\n(Mean RMS per Channel per Gesture Class)', fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        save_plot_function(fig, '08_gesture_rms_fingerprints.png')
        plt.close(fig)
        logger.info('Created gesture RMS fingerprint radar chart')
except Exception as e:
    logger.error(f'Failed to create gesture RMS fingerprint chart: {e}')
# ── Visualization 9: Keypoint Trajectory Over Time (Combined File) ──────────
try:
    if isinstance(df_combined, pd.DataFrame):
        # Plot first 3 keypoints' X coordinates over sample index to show motion
        fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        kp_groups = {'Keypoints 1-2 (X coords)': ['Measurement_10', 'Measurement_13'], 'Keypoints 3-4 (Y coords)': ['Measurement_11', 'Measurement_14'], 'Keypoints 5-6 (Z coords)': ['Measurement_12', 'Measurement_15']}
        n_plot = min(5000, len(df_combined))
        df_kp_plot = df_combined.iloc[:n_plot]
        colors_traj = ['steelblue', 'coral', 'seagreen', 'purple']
        for ax, (title, cols) in zip(axes, kp_groups.items()):
            for col, color in zip(cols, colors_traj):
                if col in df_kp_plot.columns:
                    ax.plot(df_kp_plot.index, df_kp_plot[col], linewidth=0.8, alpha=0.85, color=color, label=col.replace('Measurement_', 'Meas '))
            ax.set_ylabel('Position (mm)', fontsize=10)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel('Sample Index', fontsize=11)
        fig.suptitle('3D Hand Keypoint Trajectories Over Time\n(First 5000 samples — showing spatial motion dynamics)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_plot_function(fig, '09_keypoint_trajectories.png')
        plt.close(fig)
        logger.info('Created keypoint trajectory plot')
except Exception as e:
    logger.error(f'Failed to create keypoint trajectory plot: {e}')
logger.info('All visualizations complete.')