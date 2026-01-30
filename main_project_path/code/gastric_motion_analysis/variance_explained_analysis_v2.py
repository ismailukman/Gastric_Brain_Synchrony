#!/usr/bin/env python3
"""
VARIANCE EXPLAINED ANALYSIS: Gastric Rhythm Contribution to Head Motion

This script quantifies how much of head motion variance can be attributed to
gastric rhythm using two complementary approaches:
    1. Time-domain regression (R²) - Phase and amplitude predictors
       (Fisher, 1993; Scheeringa et al., 2011)
    2. Frequency-domain coherence - Coupling at gastric frequency band
       (Brillinger, 2001; Rosenberg et al., 1989)


Usage:
    conda activate brain_gut
    python variance_explained_analysis_v2.py

References:
    - Fisher NI (1993). Statistical Analysis of Circular Data - sin/cos regression
    - Scheeringa R et al. (2011). J Neurosci - Phase regression in neuroimaging
    - Brillinger DR (2001). Time Series: Data Analysis and Theory - Coherence
    - Rosenberg JR et al. (1989). Prog Biophys Mol Biol - Coherence in neuroscience
    - Rebollo I et al. (2018). eLife - Gastric-brain synchrony

"""

import os
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample, hilbert, coherence
from scipy.stats import pearsonr
import statsmodels.api as sm
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

# Data paths (same as egg_confounds_synchrony_v5.py)
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
OUTPUT_DIR = PARENT_DIR / "outputs" / "variance_explained"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VARIANCE_CSV = OUTPUT_DIR / "variance_explained_results_v2.csv"
OUTPUT_COHERENCE_CSV = OUTPUT_DIR / "coherence_analysis_results_v2.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "variance_summary_v2.csv"
OUTPUT_PLOT_PATH = OUTPUT_DIR / "variance_explained_plots_v2.png"

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
# Analysis 1A: Time-Domain Regression (R-Squared)
##############################################################################

def compute_variance_explained(gastric_signal, motion_params, motion_names):
    """
    Compute variance in motion explained by gastric rhythm using regression.

    Uses sin/cos decomposition of phase (circular predictor) plus amplitude.
    Returns R², partial R², coefficients, and F-test statistics.

    Model: Motion = β₀ + β₁*sin(phase) + β₂*cos(phase) + β₃*amplitude +
                    β₄*sin(phase)*amplitude + β₅*cos(phase)*amplitude + ε
    """
    # Extract phase and amplitude from gastric signal
    analytic = hilbert(gastric_signal)
    gastric_phase = np.angle(analytic)
    gastric_amplitude = np.abs(analytic)

    # Create design matrix with sin/cos decomposition (circular predictor)
    sin_phase = np.sin(gastric_phase)
    cos_phase = np.cos(gastric_phase)

    # Full model predictors (without constant - added by sm.add_constant)
    X_full = np.column_stack([
        sin_phase,                          # Sine component
        cos_phase,                          # Cosine component
        gastric_amplitude,                  # Amplitude
        sin_phase * gastric_amplitude,      # Phase-amplitude interaction
        cos_phase * gastric_amplitude
    ])
    X_full_const = sm.add_constant(X_full)

    # Phase-only model (sin + cos)
    X_phase = np.column_stack([sin_phase, cos_phase])
    X_phase_const = sm.add_constant(X_phase)

    # Amplitude-only model
    X_amp = gastric_amplitude.reshape(-1, 1)
    X_amp_const = sm.add_constant(X_amp)

    results = []

    for i, motion_name in enumerate(motion_names):
        y = motion_params[:, i]

        # Full model
        model_full = sm.OLS(y, X_full_const).fit()
        r2_full = model_full.rsquared
        r2_adj = model_full.rsquared_adj
        f_stat = model_full.fvalue
        f_pval = model_full.f_pvalue

        # Phase-only model
        model_phase = sm.OLS(y, X_phase_const).fit()
        r2_phase = model_phase.rsquared

        # Amplitude-only model
        model_amp = sm.OLS(y, X_amp_const).fit()
        r2_amp = model_amp.rsquared

        # Correlation between gastric and motion
        r_corr, p_corr = pearsonr(gastric_signal, y)

        results.append({
            'motion_param': motion_name,
            'r2_full': r2_full,
            'r2_adjusted': r2_adj,
            'r2_phase_only': r2_phase,
            'r2_amplitude_only': r2_amp,
            'f_statistic': f_stat,
            'f_pvalue': f_pval,
            'correlation_r': r_corr,
            'correlation_p': p_corr,
            'beta_sin': model_full.params[1],
            'beta_cos': model_full.params[2],
            'beta_amplitude': model_full.params[3],
            'beta_sin_amp_interaction': model_full.params[4],
            'beta_cos_amp_interaction': model_full.params[5]
        })

    return pd.DataFrame(results)


##############################################################################
# Analysis 1B: Frequency-Domain Coherence
##############################################################################

def compute_coherence_analysis(gastric_signal, motion_params, motion_names,
                                fs=SAMPLE_RATE_FMRI, gastric_freq_band=GASTRIC_FREQ_BAND):
    """
    Compute magnitude-squared coherence between gastric and motion signals.

    Coherence measures linear relationship at each frequency.
    High coherence at gastric frequency indicates frequency-specific coupling.
    """
    # Adjust nperseg for short fMRI timeseries
    nperseg = min(64, len(gastric_signal) // 4)
    if nperseg < 16:
        nperseg = len(gastric_signal) // 2

    results = []

    for i, name in enumerate(motion_names):
        try:
            freqs, coh = coherence(gastric_signal, motion_params[:, i],
                                   fs=fs, nperseg=nperseg)

            # Find gastric frequency band
            gastric_band_mask = (freqs >= gastric_freq_band[0]) & (freqs <= gastric_freq_band[1])

            if not np.any(gastric_band_mask):
                # If no frequencies in band, use closest
                center_freq = np.mean(gastric_freq_band)
                closest_idx = np.argmin(np.abs(freqs - center_freq))
                coh_gastric = coh[closest_idx]
                coh_peak = coh[closest_idx]
                peak_freq = freqs[closest_idx]
            else:
                coh_gastric = np.mean(coh[gastric_band_mask])
                coh_peak = np.max(coh[gastric_band_mask])
                peak_freq = freqs[gastric_band_mask][np.argmax(coh[gastric_band_mask])]

            # Broadband coherence (for comparison)
            coh_broadband = np.mean(coh)
            specificity_ratio = coh_gastric / (coh_broadband + 1e-10)

            results.append({
                'motion_param': name,
                'coherence_gastric_band': coh_gastric,
                'coherence_peak': coh_peak,
                'peak_frequency': peak_freq,
                'coherence_broadband': coh_broadband,
                'specificity_ratio': specificity_ratio
            })
        except Exception as e:
            print(f"    Coherence error for {name}: {e}")
            results.append({
                'motion_param': name,
                'coherence_gastric_band': np.nan,
                'coherence_peak': np.nan,
                'peak_frequency': np.nan,
                'coherence_broadband': np.nan,
                'specificity_ratio': np.nan
            })

    return pd.DataFrame(results)


##############################################################################
# Visualization
##############################################################################

def plot_variance_results(variance_df, coherence_df, output_path):
    """Create comprehensive visualization of variance explained analyses."""
    fig = plt.figure(figsize=(18, 12))
    # 2 rows: top row has 3 panels, bottom row has 2 panels (centered)
    axes = {
        (0, 0): fig.add_subplot(2, 3, 1),
        (0, 1): fig.add_subplot(2, 3, 2),
        (0, 2): fig.add_subplot(2, 3, 3),
        (1, 0): fig.add_subplot(2, 2, 3),
        (1, 1): fig.add_subplot(2, 2, 4),
    }

    motion_labels = {
        'trans_x': 'Trans X\n(L-R)', 'trans_y': 'Trans Y\n(A-P)',
        'trans_z': 'Trans Z\n(S-I)', 'rot_x': 'Rot X\n(Pitch)',
        'rot_y': 'Rot Y\n(Roll)', 'rot_z': 'Rot Z\n(Yaw)'
    }

    # Aggregate across subjects
    var_summary = variance_df.groupby('motion_param').agg({
        'r2_full': ['mean', 'std'],
        'r2_phase_only': ['mean', 'std']
    }).reset_index()
    var_summary.columns = ['motion_param', 'r2_full_mean', 'r2_full_std',
                           'r2_phase_mean', 'r2_phase_std']

    coh_summary = coherence_df.groupby('motion_param').agg({
        'coherence_gastric_band': ['mean', 'std']
    }).reset_index()
    coh_summary.columns = ['motion_param', 'coh_mean', 'coh_std']

    # Plot 1: R² Full Model
    ax1 = axes[(0, 0)]
    x_pos = np.arange(len(MOTION_COLS))
    r2_vals = [var_summary[var_summary['motion_param'] == m]['r2_full_mean'].values[0]
               for m in MOTION_COLS]
    r2_errs = [var_summary[var_summary['motion_param'] == m]['r2_full_std'].values[0]
               for m in MOTION_COLS]
    bars = ax1.bar(x_pos, r2_vals, yerr=r2_errs, capsize=5, color='steelblue', alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([motion_labels[m] for m in MOTION_COLS], fontsize=10)
    ax1.set_ylabel('R² (Full Model)', fontsize=12)
    ax1.set_title('A. Variance Explained by Gastric Rhythm\n(Phase + Amplitude)', fontsize=14)
    ax1.set_ylim(0, max(r2_vals) * 1.5 + 0.02)
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    ax1.legend(fontsize=10)

    # Plot 2: R² Phase Only
    ax2 = axes[(0, 1)]
    r2_phase = [var_summary[var_summary['motion_param'] == m]['r2_phase_mean'].values[0]
                for m in MOTION_COLS]
    r2_phase_err = [var_summary[var_summary['motion_param'] == m]['r2_phase_std'].values[0]
                    for m in MOTION_COLS]
    ax2.bar(x_pos, r2_phase, yerr=r2_phase_err, capsize=5, color='darkgreen', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([motion_labels[m] for m in MOTION_COLS], fontsize=10)
    ax2.set_ylabel('R² (Phase Only)', fontsize=12)
    ax2.set_title('B. Variance Explained by Phase Locking Only', fontsize=14)
    ax2.set_ylim(0, max(r2_phase) * 1.5 + 0.02)

    # Plot 3: Coherence
    ax3 = axes[(0, 2)]
    coh_vals = [coh_summary[coh_summary['motion_param'] == m]['coh_mean'].values[0]
                for m in MOTION_COLS]
    coh_errs = [coh_summary[coh_summary['motion_param'] == m]['coh_std'].values[0]
                for m in MOTION_COLS]
    ax3.bar(x_pos, coh_vals, yerr=coh_errs, capsize=5, color='darkorange', alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([motion_labels[m] for m in MOTION_COLS], fontsize=10)
    ax3.set_ylabel('Coherence', fontsize=12)
    ax3.set_title('C. Coherence at Gastric Frequency Band\n(0.033-0.067 Hz)', fontsize=14)
    ax3.set_ylim(0, 1.0)
    ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Substantial coupling')
    ax3.legend(fontsize=10)

    # Plot 4: R² by Subject (boxplot)
    ax4 = axes[(1, 0)]
    variance_pivot = variance_df.pivot_table(values='r2_full', index=['subject', 'run'],
                                              columns='motion_param')
    if not variance_pivot.empty:
        variance_pivot[MOTION_COLS].boxplot(ax=ax4)
        ax4.set_ylabel('R² (Full Model)', fontsize=12)
        ax4.set_title('D. R² Distribution Across Subjects', fontsize=14)
        ax4.set_xticklabels([motion_labels[m].replace('\n', ' ') for m in MOTION_COLS],
                           fontsize=9, rotation=45, ha='right')

    # Plot 5: Summary Table
    ax5 = axes[(1, 1)]
    ax5.axis('off')

    # Create summary table
    table_data = []
    for m in MOTION_COLS:
        r2 = var_summary[var_summary['motion_param'] == m]['r2_full_mean'].values[0]
        coh = coh_summary[coh_summary['motion_param'] == m]['coh_mean'].values[0]
        table_data.append([motion_labels[m].replace('\n', ' '),
                          f'{r2:.3f}', f'{coh:.3f}'])

    table = ax5.table(cellText=table_data,
                      colLabels=['Motion', 'R²', 'Coherence'],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    ax5.set_title('E. Summary Statistics (Mean)', fontsize=14, y=0.85)

    plt.suptitle('Gastric Rhythm Contribution to Head Motion: Variance Explained Analysis',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")


##############################################################################
# Main Execution
##############################################################################

def run_variance_analysis():
    """Run complete variance explained analysis pipeline."""
    print("="*70)
    print("VARIANCE EXPLAINED ANALYSIS: Gastric Rhythm → Head Motion")
    print("="*70)

    # Load data
    print("\n[1/4] Loading subject data...")
    all_data = load_all_subject_data()
    if not all_data:
        print("ERROR: No valid data loaded. Exiting.")
        return
    print(f"  Loaded {len(all_data)} subject-run pairs")

    # Run analyses
    print("\n[2/4] Computing variance explained (R², Coherence)...")

    all_variance_results = []
    all_coherence_results = []

    for idx, ((subj, run), data) in enumerate(all_data.items()):
        print(f"  Processing {subj} run {run} ({idx+1}/{len(all_data)})...")

        gastric = data['gastric']
        motion_filt = data['motion_filtered'].values

        # 1A: Variance explained (R²)
        var_df = compute_variance_explained(gastric, motion_filt, MOTION_COLS)
        var_df['subject'] = subj
        var_df['run'] = run
        var_df['gastric_peak_freq'] = data['gastric_peak']
        all_variance_results.append(var_df)

        # 1B: Coherence
        coh_df = compute_coherence_analysis(gastric, motion_filt, MOTION_COLS)
        coh_df['subject'] = subj
        coh_df['run'] = run
        all_coherence_results.append(coh_df)

    # Combine results
    variance_df = pd.concat(all_variance_results, ignore_index=True)
    coherence_df = pd.concat(all_coherence_results, ignore_index=True)

    # Create summary statistics
    print("\n[3/4] Computing summary statistics...")
    summary_data = []
    for motion_param in MOTION_COLS:
        var_subset = variance_df[variance_df['motion_param'] == motion_param]
        coh_subset = coherence_df[coherence_df['motion_param'] == motion_param]

        summary_data.append({
            'motion_param': motion_param,
            'r2_full_mean': var_subset['r2_full'].mean(),
            'r2_full_std': var_subset['r2_full'].std(),
            'r2_phase_mean': var_subset['r2_phase_only'].mean(),
            'r2_phase_std': var_subset['r2_phase_only'].std(),
            'coherence_mean': coh_subset['coherence_gastric_band'].mean(),
            'coherence_std': coh_subset['coherence_gastric_band'].std(),
            'n_subjects': len(var_subset),
            'pct_significant_f': (var_subset['f_pvalue'] < 0.05).mean() * 100
        })

    summary_df = pd.DataFrame(summary_data)

    # Save results
    print("\n[4/4] Saving results...")
    variance_df.to_csv(OUTPUT_VARIANCE_CSV, index=False)
    coherence_df.to_csv(OUTPUT_COHERENCE_CSV, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"  - Variance results: {OUTPUT_VARIANCE_CSV}")
    print(f"  - Coherence results: {OUTPUT_COHERENCE_CSV}")
    print(f"  - Summary: {OUTPUT_SUMMARY_CSV}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Variance Explained by Gastric Rhythm")
    print("="*70)
    print(f"\n{'Motion Param':<12} {'R² Full':>10} {'R² Phase':>10} {'Coherence':>10} {'% Sig':>8}")
    print("-"*60)
    for _, row in summary_df.iterrows():
        print(f"{row['motion_param']:<12} {row['r2_full_mean']:>10.4f} "
              f"{row['r2_phase_mean']:>10.4f} {row['coherence_mean']:>10.4f} "
              f"{row['pct_significant_f']:>7.1f}%")

    # Generate plots
    plot_variance_results(variance_df, coherence_df, OUTPUT_PLOT_PATH)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nInterpretation:")
    print(f"  - R² < 0.05: Minimal gastric contribution to motion")
    print(f"  - R² 0.05-0.15: Moderate gastric contribution")
    print(f"  - R² > 0.15: Substantial gastric contribution")
    print(f"  - Coherence > 0.3: Strong frequency-specific coupling")


if __name__ == "__main__":
    run_variance_analysis()
