#!/usr/bin/env python3
"""
CONFOUND REGRESSION ANALYSIS: Does Motion Regression Remove Gastric Brain Signals?

This script analyzes whether standard motion regression in fMRI preprocessing
inadvertently removes brain signals related to gastric processing.

Analyses included:
    1. Frequency content overlap between motion and gastric signals
       (Power spectral analysis of motion at gastric frequencies)
    2. Partial correlation (gastric-brain after controlling for motion)

Usage:
    conda activate brain_gut
    python confound_regression_analysis.py

References:
    - Power JD et al. (2012). NeuroImage - Motion confounds
    - Satterthwaite TD et al. (2013). NeuroImage - Motion correction
    - Birn RM et al. (2006). NeuroImage - Physiological confounds
    - Ciric R et al. (2017). NeuroImage - Confound benchmarking

"""

import os
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample, welch
from scipy.stats import pearsonr, spearmanr
from mne.filter import filter_data

warnings.filterwarnings('ignore')

##############################################################################
# Configuration
##############################################################################

PARENT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = PARENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from config import (main_project_path, clean_level, sample_rate_fmri,
                    intermediate_sample_rate, bandpass_lim, filter_order,
                    transition_width)

# Data paths
META_DATAFRAME_PATH = PROJECT_ROOT / "dataframes" / "egg_brain_meta_data.csv"

MOTION_FILE_TEMPLATE = os.path.join(
    main_project_path, "BIDS_data", "sub_motion_files",
    "sub-{sub}_dfile.r0{run}.1D"
)

EGG_FILE_TEMPLATE = os.path.join(
    main_project_path, "derivatives", "brain_gast", "{sub}", "{sub}{run}",
    "gast_data_{sub}_run{run}{clean_level}.npy"
)

GASTRIC_FREQ_TEMPLATE = os.path.join(
    main_project_path, "derivatives", "brain_gast", "{sub}", "{sub}{run}",
    "max_freq{sub}_run{run}{clean_level}.npy"
)

# Output paths
OUTPUT_DIR = PARENT_DIR / "outputs" / "confound_regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FREQ_OVERLAP_CSV = OUTPUT_DIR / "frequency_overlap_results.csv"
OUTPUT_PARTIAL_CORR_CSV = OUTPUT_DIR / "partial_correlation_results.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "confound_analysis_summary.csv"
OUTPUT_PLOT_PATH = OUTPUT_DIR / "confound_analysis_plots.png"

# Analysis parameters
SAMPLE_RATE_FMRI = sample_rate_fmri
EGG_INTERMEDIATE_SFREQ = intermediate_sample_rate
MOTION_COLS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
GASTRIC_FREQ_BAND = (0.033, 0.067)  # Hz


##############################################################################
# Helper Functions
##############################################################################

def bp_filter_confounds(df, gastric_peak, sample_rate=SAMPLE_RATE_FMRI,
                        bandpass_lim_val=bandpass_lim, filter_order_val=filter_order,
                        transition_width_val=transition_width):
    """Bandpass-filter each column in df around the subject-specific gastric_peak."""
    l_freq = gastric_peak - bandpass_lim_val
    h_freq = gastric_peak + bandpass_lim_val
    filter_length = int(filter_order_val * np.floor(sample_rate / (gastric_peak - bandpass_lim_val)))

    confound_filtered = filter_data(
        data=df.values.T, sfreq=sample_rate,
        l_freq=l_freq, h_freq=h_freq,
        filter_length=filter_length,
        l_trans_bandwidth=transition_width_val * (gastric_peak - bandpass_lim_val),
        h_trans_bandwidth=transition_width_val * (gastric_peak + bandpass_lim_val),
        n_jobs=1, method='fir', phase='zero-double',
        fir_window='hamming', fir_design='firwin2', verbose=False
    )
    return pd.DataFrame(confound_filtered.T, columns=df.columns, index=df.index)


def load_all_subject_data():
    """Load all subject data (gastric signals and motion parameters)."""
    record_meta_pd = pd.read_csv(META_DATAFRAME_PATH)
    if clean_level == 'strict_gs_cardiac':
        record_meta_pd = record_meta_pd.loc[
            (record_meta_pd['ppu_exclude'] == False) &
            (record_meta_pd['ppu_found'] == True)
        ]

    subjects_runs = list(zip(record_meta_pd['subject'], record_meta_pd['run']))
    all_data = {}

    for (subject_name, run) in subjects_runs:
        try:
            motion_path = MOTION_FILE_TEMPLATE.format(sub=subject_name, run=run)
            egg_file = EGG_FILE_TEMPLATE.format(sub=subject_name, run=run, clean_level=clean_level)
            freq_file = GASTRIC_FREQ_TEMPLATE.format(sub=subject_name, run=run, clean_level=clean_level)

            if not all(os.path.isfile(p) for p in [motion_path, egg_file, freq_file]):
                continue

            motion_data = np.loadtxt(motion_path)
            df_confound = pd.DataFrame(motion_data, columns=MOTION_COLS)
            gastric_signal = np.load(egg_file)
            gastric_peak = float(np.load(freq_file).flatten()[0])

            n_points_fmri = int((len(gastric_signal) / EGG_INTERMEDIATE_SFREQ) * SAMPLE_RATE_FMRI)
            if n_points_fmri < 10:
                continue

            gastric_signal_resampled = resample(gastric_signal, n_points_fmri)
            min_length = min(len(gastric_signal_resampled), len(df_confound))
            gastric_signal_resampled = gastric_signal_resampled[:min_length]
            df_confound = df_confound.iloc[:min_length]
            df_confound_filt = bp_filter_confounds(df_confound, gastric_peak)

            all_data[(subject_name, run)] = {
                'subject': subject_name,
                'run': run,
                'gastric': gastric_signal_resampled,
                'gastric_peak': gastric_peak,
                'motion_filtered': df_confound_filt,
                'motion_raw': df_confound
            }
        except Exception as e:
            print(f"  Error loading {subject_name} run {run}: {e}")

    return all_data


##############################################################################
# Analysis 2A: Frequency Content Overlap Analysis
##############################################################################

def analyze_frequency_overlap(motion_params, gastric_signal, motion_names,
                               fs=SAMPLE_RATE_FMRI, gastric_freq_band=GASTRIC_FREQ_BAND):
    """
    Analyze power spectrum overlap between motion and gastric signals.

    If motion has substantial power at gastric frequency, motion regression
    will remove gastric-frequency brain signals.

    Returns:
        DataFrame with percent power in gastric band for each motion parameter
    """
    # Adjust nperseg for short fMRI timeseries
    nperseg = min(64, len(gastric_signal) // 4)
    if nperseg < 16:
        nperseg = len(gastric_signal) // 2

    # Gastric power spectrum
    freqs, psd_gastric = welch(gastric_signal, fs=fs, nperseg=nperseg)

    # Find gastric frequency band
    gastric_band_mask = (freqs >= gastric_freq_band[0]) & (freqs <= gastric_freq_band[1])

    # Gastric peak frequency
    if np.any(gastric_band_mask):
        gastric_peak_idx = np.argmax(psd_gastric[gastric_band_mask])
        gastric_peak_freq = freqs[gastric_band_mask][gastric_peak_idx]
    else:
        gastric_peak_freq = np.mean(gastric_freq_band)

    results = []

    for i, name in enumerate(motion_names):
        # Motion power spectrum
        _, psd_motion = welch(motion_params[:, i], fs=fs, nperseg=nperseg)

        # Power at gastric frequency band
        if np.any(gastric_band_mask):
            power_gastric_band = np.sum(psd_motion[gastric_band_mask])
        else:
            power_gastric_band = 0

        power_total = np.sum(psd_motion)
        if power_total > 0:
            percent_gastric = 100 * power_gastric_band / power_total
        else:
            percent_gastric = 0

        # Power at low frequencies (< gastric band)
        low_freq_mask = freqs < gastric_freq_band[0]
        if np.any(low_freq_mask):
            percent_low_freq = 100 * np.sum(psd_motion[low_freq_mask]) / power_total
        else:
            percent_low_freq = 0

        results.append({
            'motion_param': name,
            'percent_power_gastric_band': percent_gastric,
            'percent_power_low_freq': percent_low_freq,
            'gastric_peak_freq': gastric_peak_freq,
            'total_power': power_total
        })

    return pd.DataFrame(results)


##############################################################################
# Analysis 2B: Partial Correlation Analysis
##############################################################################

def partial_correlation(x, y, covariates):
    """
    Compute partial correlation between x and y after controlling for covariates.

    Uses regression residuals approach:
    1. Regress x on covariates, get residuals
    2. Regress y on covariates, get residuals
    3. Correlate residuals
    """
    import statsmodels.api as sm

    # Add constant to covariates
    covariates_const = sm.add_constant(covariates)

    # Residualize x
    model_x = sm.OLS(x, covariates_const).fit()
    x_residuals = model_x.resid

    # Residualize y
    model_y = sm.OLS(y, covariates_const).fit()
    y_residuals = model_y.resid

    # Correlate residuals
    r_partial, p_partial = pearsonr(x_residuals, y_residuals)

    return r_partial, p_partial


def compute_partial_correlations(gastric_signal, motion_params, motion_names):
    """
    Compute zero-order and partial correlations between gastric and each motion parameter.

    For each motion parameter:
    - Zero-order: correlation(gastric, motion_i)
    - Partial: correlation(gastric, motion_i | other_motion_params)

    This tests whether the gastric-motion relationship is confounded by other motion types.
    """
    results = []

    for i, name in enumerate(motion_names):
        motion_i = motion_params[:, i]

        # Zero-order correlation
        r_zero, p_zero = pearsonr(gastric_signal, motion_i)

        # Partial correlation controlling for other motion parameters
        other_motion_indices = [j for j in range(len(motion_names)) if j != i]
        if len(other_motion_indices) > 0:
            other_motion = motion_params[:, other_motion_indices]
            r_partial, p_partial = partial_correlation(gastric_signal, motion_i, other_motion)
        else:
            r_partial, p_partial = r_zero, p_zero

        # Compute attenuation
        attenuation = np.abs(r_zero) - np.abs(r_partial)
        if np.abs(r_zero) > 1e-10:
            percent_attenuation = 100 * attenuation / np.abs(r_zero)
        else:
            percent_attenuation = 0

        # Spearman correlation (non-parametric)
        rho_zero, p_spearman = spearmanr(gastric_signal, motion_i)

        results.append({
            'motion_param': name,
            'r_zero_order': r_zero,
            'p_zero_order': p_zero,
            'r_partial': r_partial,
            'p_partial': p_partial,
            'attenuation': attenuation,
            'percent_attenuation': percent_attenuation,
            'rho_spearman': rho_zero,
            'p_spearman': p_spearman
        })

    return pd.DataFrame(results)


##############################################################################
# Visualization
##############################################################################

def plot_confound_results(freq_overlap_df, partial_corr_df, summary_df, output_path):
    """Create comprehensive visualization of confound analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    motion_labels = {
        'trans_x': 'Trans X\n(L-R)', 'trans_y': 'Trans Y\n(A-P)',
        'trans_z': 'Trans Z\n(S-I)', 'rot_x': 'Rot X\n(Pitch)',
        'rot_y': 'Rot Y\n(Roll)', 'rot_z': 'Rot Z\n(Yaw)'
    }

    # Aggregate across subjects
    freq_summary = freq_overlap_df.groupby('motion_param').agg({
        'percent_power_gastric_band': ['mean', 'std']
    }).reset_index()
    freq_summary.columns = ['motion_param', 'pct_gastric_mean', 'pct_gastric_std']

    corr_summary = partial_corr_df.groupby('motion_param').agg({
        'r_zero_order': ['mean', 'std'],
        'r_partial': ['mean', 'std'],
        'percent_attenuation': ['mean', 'std']
    }).reset_index()
    corr_summary.columns = ['motion_param', 'r_zero_mean', 'r_zero_std',
                            'r_partial_mean', 'r_partial_std',
                            'pct_atten_mean', 'pct_atten_std']

    x_pos = np.arange(len(MOTION_COLS))

    # Plot 1: Percent Power in Gastric Band
    ax1 = axes[0, 0]
    pct_vals = [freq_summary[freq_summary['motion_param'] == m]['pct_gastric_mean'].values[0]
                for m in MOTION_COLS]
    pct_errs = [freq_summary[freq_summary['motion_param'] == m]['pct_gastric_std'].values[0]
                for m in MOTION_COLS]
    bars = ax1.bar(x_pos, pct_vals, yerr=pct_errs, capsize=5, color='indianred', alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([motion_labels[m] for m in MOTION_COLS], fontsize=10)
    ax1.set_ylabel('% Power in Gastric Band', fontsize=12)
    ax1.set_title('A. Motion Power at Gastric Frequency\n(0.033-0.067 Hz)', fontsize=14)
    ax1.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    ax1.legend(fontsize=10)

    # Plot 2: Zero-order vs Partial Correlations
    ax2 = axes[0, 1]
    width = 0.35
    r_zero = [corr_summary[corr_summary['motion_param'] == m]['r_zero_mean'].values[0]
              for m in MOTION_COLS]
    r_partial = [corr_summary[corr_summary['motion_param'] == m]['r_partial_mean'].values[0]
                 for m in MOTION_COLS]
    ax2.bar(x_pos - width/2, np.abs(r_zero), width, label='Zero-order |r|', color='steelblue', alpha=0.7)
    ax2.bar(x_pos + width/2, np.abs(r_partial), width, label='Partial |r|', color='darkorange', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([motion_labels[m] for m in MOTION_COLS], fontsize=10)
    ax2.set_ylabel('|Correlation|', fontsize=12)
    ax2.set_title('B. Gastric-Motion Correlation\n(Before vs After Controlling for Other Motion)', fontsize=14)
    ax2.legend(fontsize=10)

    # Plot 3: Percent Attenuation
    ax3 = axes[0, 2]
    atten_vals = [corr_summary[corr_summary['motion_param'] == m]['pct_atten_mean'].values[0]
                  for m in MOTION_COLS]
    atten_errs = [corr_summary[corr_summary['motion_param'] == m]['pct_atten_std'].values[0]
                  for m in MOTION_COLS]
    colors = ['green' if a < 0 else 'red' for a in atten_vals]
    ax3.bar(x_pos, atten_vals, yerr=atten_errs, capsize=5, color=colors, alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([motion_labels[m] for m in MOTION_COLS], fontsize=10)
    ax3.set_ylabel('% Attenuation', fontsize=12)
    ax3.set_title('C. Correlation Attenuation After Controlling\nfor Other Motion Parameters', fontsize=14)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot 4: Distribution of Gastric-Motion Correlations
    ax4 = axes[1, 0]
    partial_corr_df.boxplot(column='r_zero_order', by='motion_param', ax=ax4)
    ax4.set_title('D. Distribution of Gastric-Motion\nCorrelations Across Subjects', fontsize=14)
    ax4.set_xlabel('')
    ax4.set_ylabel('Correlation (r)', fontsize=12)
    plt.suptitle('')  # Remove automatic title
    ax4.set_xticklabels([motion_labels[m].replace('\n', ' ') for m in MOTION_COLS],
                        fontsize=9, rotation=45, ha='right')

    # Plot 5: Interpretation Guide
    ax5 = axes[1, 1]
    ax5.axis('off')
    interpretation_text = """
    INTERPRETATION GUIDE (Tentative)
    ====================

    Frequency Overlap (Plot A):
    • >20% power in gastric band = Motion regression
      will significantly affect gastric-band brain signals

    Correlation Analysis (Plot B):
    • If partial ≈ zero-order: Gastric-motion relationship
      is NOT confounded by other motion types
    • If partial < zero-order: Some gastric-motion
      relationship is explained by other motion

    Attenuation (Plot C):
    • Positive attenuation: Other motion explains
      part of gastric-motion relationship
    • Negative attenuation: Gastric-motion relationship
      is STRONGER after controlling for other motion

    IMPLICATIONS FOR fMRI PREPROCESSING:
    • High overlap → Consider NOT regressing motion
      for gut-brain studies
    • Low overlap → Motion regression is safe
    """
    ax5.text(0.05, 0.95, interpretation_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Plot 6: Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')

    table_data = []
    for m in MOTION_COLS:
        pct_g = freq_summary[freq_summary['motion_param'] == m]['pct_gastric_mean'].values[0]
        r_z = corr_summary[corr_summary['motion_param'] == m]['r_zero_mean'].values[0]
        r_p = corr_summary[corr_summary['motion_param'] == m]['r_partial_mean'].values[0]
        att = corr_summary[corr_summary['motion_param'] == m]['pct_atten_mean'].values[0]
        table_data.append([motion_labels[m].replace('\n', ' '),
                          f'{pct_g:.1f}%', f'{r_z:.3f}', f'{r_p:.3f}', f'{att:.1f}%'])

    table = ax6.table(cellText=table_data,
                      colLabels=['Motion', '% Gastric\nPower', 'r (zero)', 'r (partial)', 'Atten.'],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('E. Summary Statistics', fontsize=14, y=0.85)

    plt.suptitle('Confound Analysis: Motion Regression Impact on Gastric Signals',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")


##############################################################################
# Main Execution
##############################################################################

def run_confound_analysis():
    """Run complete confound regression analysis pipeline."""
    print("="*70)
    print("CONFOUND ANALYSIS: Does Motion Regression Remove Gastric Signals?")
    print("="*70)

    # Load data
    print("\n[1/3] Loading subject data...")
    all_data = load_all_subject_data()
    if not all_data:
        print("ERROR: No valid data loaded. Exiting.")
        return
    print(f"  Loaded {len(all_data)} subject-run pairs")

    # Run analyses
    print("\n[2/3] Computing frequency overlap and partial correlations...")

    all_freq_overlap = []
    all_partial_corr = []

    for idx, ((subj, run), data) in enumerate(all_data.items()):
        print(f"  Processing {subj} run {run} ({idx+1}/{len(all_data)})...")

        gastric = data['gastric']
        motion_filt = data['motion_filtered'].values
        gastric_peak = data['gastric_peak']

        # 2A: Frequency overlap
        freq_df = analyze_frequency_overlap(motion_filt, gastric, MOTION_COLS)
        freq_df['subject'] = subj
        freq_df['run'] = run
        freq_df['gastric_peak_freq'] = gastric_peak
        all_freq_overlap.append(freq_df)

        # 2B: Partial correlations
        corr_df = compute_partial_correlations(gastric, motion_filt, MOTION_COLS)
        corr_df['subject'] = subj
        corr_df['run'] = run
        all_partial_corr.append(corr_df)

    # Combine results
    freq_overlap_df = pd.concat(all_freq_overlap, ignore_index=True)
    partial_corr_df = pd.concat(all_partial_corr, ignore_index=True)

    # Create summary statistics
    print("\n[3/3] Computing summary statistics and saving results...")
    summary_data = []
    for motion_param in MOTION_COLS:
        freq_subset = freq_overlap_df[freq_overlap_df['motion_param'] == motion_param]
        corr_subset = partial_corr_df[partial_corr_df['motion_param'] == motion_param]

        summary_data.append({
            'motion_param': motion_param,
            'pct_power_gastric_mean': freq_subset['percent_power_gastric_band'].mean(),
            'pct_power_gastric_std': freq_subset['percent_power_gastric_band'].std(),
            'r_zero_order_mean': corr_subset['r_zero_order'].mean(),
            'r_zero_order_std': corr_subset['r_zero_order'].std(),
            'r_partial_mean': corr_subset['r_partial'].mean(),
            'r_partial_std': corr_subset['r_partial'].std(),
            'pct_attenuation_mean': corr_subset['percent_attenuation'].mean(),
            'pct_attenuation_std': corr_subset['percent_attenuation'].std(),
            'n_subjects': len(freq_subset)
        })

    summary_df = pd.DataFrame(summary_data)

    # Save results
    freq_overlap_df.to_csv(OUTPUT_FREQ_OVERLAP_CSV, index=False)
    partial_corr_df.to_csv(OUTPUT_PARTIAL_CORR_CSV, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"  - Frequency overlap: {OUTPUT_FREQ_OVERLAP_CSV}")
    print(f"  - Partial correlations: {OUTPUT_PARTIAL_CORR_CSV}")
    print(f"  - Summary: {OUTPUT_SUMMARY_CSV}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Confound Analysis Results")
    print("="*70)
    print(f"\n{'Motion':<12} {'%Gastric':>10} {'r(zero)':>10} {'r(partial)':>12} {'%Atten':>10}")
    print("-"*55)
    for _, row in summary_df.iterrows():
        print(f"{row['motion_param']:<12} {row['pct_power_gastric_mean']:>9.1f}% "
              f"{row['r_zero_order_mean']:>10.3f} {row['r_partial_mean']:>12.3f} "
              f"{row['pct_attenuation_mean']:>9.1f}%")

    # Generate plots
    plot_confound_results(freq_overlap_df, partial_corr_df, summary_df, OUTPUT_PLOT_PATH)

    # Print conclusions
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)

    avg_gastric_power = summary_df['pct_power_gastric_mean'].mean()
    max_gastric_power = summary_df['pct_power_gastric_mean'].max()
    max_motion = summary_df.loc[summary_df['pct_power_gastric_mean'].idxmax(), 'motion_param']

    print(f"\n1. FREQUENCY OVERLAP:")
    print(f"   - Average motion power in gastric band: {avg_gastric_power:.1f}%")
    print(f"   - Maximum: {max_gastric_power:.1f}% ({max_motion})")
    if max_gastric_power > 20:
        print(f"   - WARNING: Motion regression may significantly affect gastric signals")
    else:
        print(f"   - Motion regression has limited impact on gastric-band signals")

    avg_attenuation = summary_df['pct_attenuation_mean'].mean()
    print(f"\n2. CORRELATION ATTENUATION:")
    print(f"   - Average attenuation: {avg_attenuation:.1f}%")
    if avg_attenuation > 20:
        print(f"   - Substantial confounding between motion parameters")
    else:
        print(f"   - Limited confounding between motion parameters")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_confound_analysis()

