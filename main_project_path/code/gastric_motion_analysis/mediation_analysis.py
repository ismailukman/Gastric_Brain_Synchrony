#!/usr/bin/env python3
"""
MEDIATION ANALYSIS: Does the Brain Mediate Gastric-Motion Effects?

This script tests whether brain activity mediates the relationship between
gastric rhythm and head motion using classical mediation analysis and
Granger causality approaches.

Causal model tested:
    Gastric Rhythm (X) → Brain Activity (M) → Head Motion (Y)

Analyses included:
    1. Classical mediation (Baron & Kenny, 1986) with Sobel test
    2. Bootstrap mediation for robust confidence intervals
    3. Granger causality analysis for temporal precedence

Based on the analysis plan in GASTRIC_MOTION_VARIANCE_MEDIATION_ANALYSIS.txt
and building on egg_confounds_synchrony_v5.py data structures.

Usage:
    conda activate brain_gut
    python mediation_analysis.py

References:
    - Baron RM & Kenny DA (1986). JPSP - Mediation analysis
    - Preacher KJ & Hayes AF (2008). Behav Res Methods - Bootstrap mediation
    - Granger CWJ (1969). Econometrica - Granger causality
    - Rebollo I et al. (2018). eLife - Gastric-brain synchrony

"""

import os
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample, hilbert
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
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
OUTPUT_DIR = PARENT_DIR / "outputs" / "mediation_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MEDIATION_CSV = OUTPUT_DIR / "mediation_results.csv"
OUTPUT_BOOTSTRAP_CSV = OUTPUT_DIR / "bootstrap_mediation_results.csv"
OUTPUT_GRANGER_CSV = OUTPUT_DIR / "granger_causality_results.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "mediation_summary.csv"
OUTPUT_PLOT_PATH = OUTPUT_DIR / "mediation_analysis_plots.png"

# Analysis parameters
SAMPLE_RATE_FMRI = sample_rate_fmri
EGG_INTERMEDIATE_SFREQ = intermediate_sample_rate
MOTION_COLS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
GASTRIC_FREQ_BAND = (0.033, 0.067)  # Hz

# Mediation parameters
N_BOOTSTRAP = 5000  # Number of bootstrap iterations
GRANGER_MAX_LAG = 5  # Maximum lag for Granger causality test


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


def create_synthetic_brain_mediator(gastric_signal, motion_signal, noise_level=0.3):
    """
    Create a synthetic brain mediator signal for demonstration.

    In real analysis, this would be replaced with actual brain ROI timeseries
    (e.g., insula, somatosensory cortex, or regions showing gastric-brain PLV).

    The synthetic mediator is a weighted combination of:
    - Gastric phase (shifted to simulate processing delay)
    - Motion signal (reversed causality check)
    - Random noise

    Parameters:
        gastric_signal: Preprocessed gastric EGG signal
        motion_signal: Motion parameter timeseries
        noise_level: Standard deviation of added noise (relative to signal)

    Returns:
        Synthetic brain mediator signal
    """
    # Extract gastric phase
    analytic = hilbert(gastric_signal)
    gastric_phase = np.angle(analytic)

    # Phase-shifted gastric (simulating neural processing delay ~0.5-2s)
    shift_samples = int(1.0 * SAMPLE_RATE_FMRI)  # 1 second delay
    gastric_shifted = np.roll(np.sin(gastric_phase), shift_samples)

    # Combine with some motion influence
    brain_signal = 0.6 * gastric_shifted + 0.3 * (motion_signal / (np.std(motion_signal) + 1e-10))

    # Add noise
    noise = np.random.randn(len(brain_signal)) * noise_level * np.std(brain_signal)
    brain_signal = brain_signal + noise

    # Z-score normalize
    brain_signal = (brain_signal - np.mean(brain_signal)) / (np.std(brain_signal) + 1e-10)

    return brain_signal


##############################################################################
# Analysis 3A: Classical Mediation Analysis (Baron & Kenny, 1986)
##############################################################################

def mediation_analysis(X, M, Y):
    """
    Perform classical mediation analysis following Baron & Kenny (1986).

    Tests the causal chain: X → M → Y

    Steps:
        1. Path c: X predicts Y (total effect)
        2. Path a: X predicts M
        3. Path b: M predicts Y controlling for X
        4. Path c': X predicts Y controlling for M (direct effect)

    Mediation exists if:
        - c is significant (or proceed anyway for indirect effect test)
        - a is significant
        - b is significant
        - c' is smaller than c (partial mediation) or non-significant (full mediation)

    Parameters:
        X: Independent variable (gastric signal)
        M: Mediator (brain signal)
        Y: Dependent variable (motion)

    Returns:
        Dictionary with path coefficients, p-values, and Sobel test results
    """
    # Ensure 1D arrays
    X = np.asarray(X).flatten()
    M = np.asarray(M).flatten()
    Y = np.asarray(Y).flatten()

    # Path c: Total effect (X → Y)
    X_const = sm.add_constant(X)
    model_c = sm.OLS(Y, X_const).fit()
    c = model_c.params[1]
    c_se = model_c.bse[1]
    c_p = model_c.pvalues[1]

    # Path a: X → M
    model_a = sm.OLS(M, X_const).fit()
    a = model_a.params[1]
    a_se = model_a.bse[1]
    a_p = model_a.pvalues[1]

    # Paths b and c': M → Y and X → Y controlling for M
    XM_const = sm.add_constant(np.column_stack([X, M]))
    model_bc = sm.OLS(Y, XM_const).fit()
    c_prime = model_bc.params[1]  # Direct effect (X → Y | M)
    c_prime_se = model_bc.bse[1]
    c_prime_p = model_bc.pvalues[1]
    b = model_bc.params[2]  # M → Y | X
    b_se = model_bc.bse[2]
    b_p = model_bc.pvalues[2]

    # Indirect effect: a * b
    indirect = a * b

    # Sobel test for indirect effect significance
    # SE of indirect effect = sqrt(a²*SE_b² + b²*SE_a²)
    sobel_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else 0
    sobel_p = 2 * (1 - norm.cdf(np.abs(sobel_z)))

    # Proportion mediated
    if np.abs(c) > 1e-10:
        prop_mediated = indirect / c
    else:
        prop_mediated = np.nan

    # Model fit statistics
    r2_total = model_c.rsquared  # Variance explained by X alone
    r2_with_mediator = model_bc.rsquared  # Variance explained by X + M

    return {
        # Path coefficients
        'path_c_total': c,
        'path_c_total_se': c_se,
        'path_c_total_p': c_p,
        'path_a': a,
        'path_a_se': a_se,
        'path_a_p': a_p,
        'path_b': b,
        'path_b_se': b_se,
        'path_b_p': b_p,
        'path_c_prime_direct': c_prime,
        'path_c_prime_se': c_prime_se,
        'path_c_prime_p': c_prime_p,
        # Indirect effect
        'indirect_effect': indirect,
        'indirect_se': sobel_se,
        # Sobel test
        'sobel_z': sobel_z,
        'sobel_p': sobel_p,
        # Proportion mediated
        'proportion_mediated': prop_mediated,
        # Model fit
        'r2_total_effect': r2_total,
        'r2_with_mediator': r2_with_mediator,
        'r2_increase': r2_with_mediator - r2_total
    }


##############################################################################
# Analysis 3B: Bootstrap Mediation
##############################################################################

def bootstrap_mediation(X, M, Y, n_bootstrap=N_BOOTSTRAP, ci_level=0.95):
    """
    Bootstrap mediation analysis for robust confidence intervals.

    The bootstrap approach:
    1. Resample data with replacement
    2. Compute indirect effect (a*b) for each resample
    3. Construct percentile confidence intervals

    Parameters:
        X: Independent variable (gastric signal)
        M: Mediator (brain signal)
        Y: Dependent variable (motion)
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence interval level (default 0.95)

    Returns:
        Dictionary with bootstrap statistics and confidence intervals
    """
    X = np.asarray(X).flatten()
    M = np.asarray(M).flatten()
    Y = np.asarray(Y).flatten()
    n = len(X)

    # Storage for bootstrap estimates
    indirect_boot = np.zeros(n_bootstrap)
    a_boot = np.zeros(n_bootstrap)
    b_boot = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample indices with replacement
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        M_boot = M[idx]
        Y_boot = Y[idx]

        # Path a: X → M
        X_const = sm.add_constant(X_boot)
        try:
            model_a = sm.OLS(M_boot, X_const).fit()
            a = model_a.params[1]
        except:
            a = np.nan

        # Path b: M → Y | X
        XM_const = sm.add_constant(np.column_stack([X_boot, M_boot]))
        try:
            model_bc = sm.OLS(Y_boot, XM_const).fit()
            b = model_bc.params[2]
        except:
            b = np.nan

        a_boot[i] = a
        b_boot[i] = b
        indirect_boot[i] = a * b

    # Remove any NaN values
    valid_mask = ~np.isnan(indirect_boot)
    indirect_boot = indirect_boot[valid_mask]
    a_boot = a_boot[valid_mask]
    b_boot = b_boot[valid_mask]

    # Compute confidence intervals (percentile method)
    alpha = 1 - ci_level
    ci_lower = np.percentile(indirect_boot, 100 * alpha / 2)
    ci_upper = np.percentile(indirect_boot, 100 * (1 - alpha / 2))

    # Bias-corrected and accelerated (BCa) CI - simplified version
    # (Full BCa requires jackknife, here we use bias-corrected percentile)
    mean_boot = np.mean(indirect_boot)
    median_boot = np.median(indirect_boot)

    # Check if CI excludes zero (significant indirect effect)
    ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)

    return {
        'indirect_mean': mean_boot,
        'indirect_median': median_boot,
        'indirect_std': np.std(indirect_boot),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_level': ci_level,
        'ci_excludes_zero': ci_excludes_zero,
        'n_valid_bootstrap': len(indirect_boot),
        'a_mean': np.mean(a_boot),
        'a_std': np.std(a_boot),
        'b_mean': np.mean(b_boot),
        'b_std': np.std(b_boot)
    }


##############################################################################
# Analysis 3C: Granger Causality Analysis
##############################################################################

def granger_causality_analysis(signal_1, signal_2, max_lag=GRANGER_MAX_LAG):
    """
    Test Granger causality between two signals.

    Signal 1 "Granger-causes" Signal 2 if past values of Signal 1
    help predict Signal 2 beyond what Signal 2's own past predicts.

    Parameters:
        signal_1: First signal (potential cause)
        signal_2: Second signal (potential effect)
        max_lag: Maximum number of lags to test

    Returns:
        Dictionary with Granger causality test results for each lag
    """
    signal_1 = np.asarray(signal_1).flatten()
    signal_2 = np.asarray(signal_2).flatten()

    # Prepare data for Granger test (needs 2D array with columns)
    data = np.column_stack([signal_2, signal_1])  # [effect, cause] order for statsmodels

    results_dict = {}

    try:
        # Run Granger causality test
        gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        for lag in range(1, max_lag + 1):
            if lag in gc_results:
                # Extract F-test results
                f_test = gc_results[lag][0]['ssr_ftest']
                f_stat = f_test[0]
                f_p = f_test[1]
                df_num = f_test[2]
                df_denom = f_test[3]

                # Extract likelihood ratio test
                lr_test = gc_results[lag][0]['lrtest']
                lr_stat = lr_test[0]
                lr_p = lr_test[1]

                results_dict[f'lag_{lag}_f_stat'] = f_stat
                results_dict[f'lag_{lag}_f_p'] = f_p
                results_dict[f'lag_{lag}_lr_stat'] = lr_stat
                results_dict[f'lag_{lag}_lr_p'] = lr_p
                results_dict[f'lag_{lag}_significant'] = f_p < 0.05
    except Exception as e:
        # If Granger test fails, return NaN values
        for lag in range(1, max_lag + 1):
            results_dict[f'lag_{lag}_f_stat'] = np.nan
            results_dict[f'lag_{lag}_f_p'] = np.nan
            results_dict[f'lag_{lag}_lr_stat'] = np.nan
            results_dict[f'lag_{lag}_lr_p'] = np.nan
            results_dict[f'lag_{lag}_significant'] = False

    # Summary: minimum p-value across lags
    p_values = [results_dict.get(f'lag_{lag}_f_p', 1.0) for lag in range(1, max_lag + 1)]
    valid_p = [p for p in p_values if not np.isnan(p)]
    results_dict['min_p_value'] = min(valid_p) if valid_p else np.nan
    results_dict['best_lag'] = np.argmin(p_values) + 1 if valid_p else np.nan
    results_dict['any_significant'] = any(p < 0.05 for p in valid_p)

    return results_dict


def run_full_granger_analysis(gastric, brain, motion, motion_names):
    """
    Run comprehensive Granger causality analysis for all variable pairs.

    Tests:
    - Gastric → Brain (does gastric predict brain?)
    - Brain → Motion (does brain predict motion?)
    - Gastric → Motion (direct path)
    - Reverse directions to check for feedback
    """
    results = []

    # Gastric → Brain
    gc_gastric_brain = granger_causality_analysis(gastric, brain)
    results.append({
        'cause': 'gastric',
        'effect': 'brain',
        **gc_gastric_brain
    })

    # Brain → Gastric (reverse direction check)
    gc_brain_gastric = granger_causality_analysis(brain, gastric)
    results.append({
        'cause': 'brain',
        'effect': 'gastric',
        **gc_brain_gastric
    })

    # For each motion parameter
    for i, motion_name in enumerate(motion_names):
        motion_i = motion[:, i]

        # Brain → Motion
        gc_brain_motion = granger_causality_analysis(brain, motion_i)
        results.append({
            'cause': 'brain',
            'effect': motion_name,
            **gc_brain_motion
        })

        # Motion → Brain (reverse)
        gc_motion_brain = granger_causality_analysis(motion_i, brain)
        results.append({
            'cause': motion_name,
            'effect': 'brain',
            **gc_motion_brain
        })

        # Gastric → Motion (direct)
        gc_gastric_motion = granger_causality_analysis(gastric, motion_i)
        results.append({
            'cause': 'gastric',
            'effect': motion_name,
            **gc_gastric_motion
        })

        # Motion → Gastric (reverse)
        gc_motion_gastric = granger_causality_analysis(motion_i, gastric)
        results.append({
            'cause': motion_name,
            'effect': 'gastric',
            **gc_motion_gastric
        })

    return pd.DataFrame(results)


##############################################################################
# Visualization
##############################################################################

def plot_mediation_results(mediation_df, bootstrap_df, granger_df, output_path):
    """Create comprehensive visualization of mediation analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    motion_labels = {
        'trans_x': 'Trans X', 'trans_y': 'Trans Y',
        'trans_z': 'Trans Z', 'rot_x': 'Rot X',
        'rot_y': 'Rot Y', 'rot_z': 'Rot Z'
    }

    # Aggregate across subjects
    med_summary = mediation_df.groupby('motion_param').agg({
        'indirect_effect': ['mean', 'std'],
        'proportion_mediated': ['mean', 'std'],
        'path_a': ['mean', 'std'],
        'path_b': ['mean', 'std']
    }).reset_index()
    med_summary.columns = ['motion_param', 'indirect_mean', 'indirect_std',
                           'prop_mean', 'prop_std', 'a_mean', 'a_std', 'b_mean', 'b_std']

    boot_summary = bootstrap_df.groupby('motion_param').agg({
        'indirect_mean': ['mean', 'std'],
        'ci_excludes_zero': 'mean'
    }).reset_index()
    boot_summary.columns = ['motion_param', 'boot_indirect_mean', 'boot_indirect_std',
                            'pct_significant']

    x_pos = np.arange(len(MOTION_COLS))

    # Plot 1: Indirect Effects by Motion Parameter
    ax1 = axes[0, 0]
    indirect_vals = [med_summary[med_summary['motion_param'] == m]['indirect_mean'].values[0]
                     for m in MOTION_COLS]
    indirect_errs = [med_summary[med_summary['motion_param'] == m]['indirect_std'].values[0]
                     for m in MOTION_COLS]
    colors = ['green' if v > 0 else 'red' for v in indirect_vals]
    ax1.bar(x_pos, indirect_vals, yerr=indirect_errs, capsize=5, color=colors, alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([motion_labels[m] for m in MOTION_COLS], fontsize=10, rotation=45, ha='right')
    ax1.set_ylabel('Indirect Effect (a × b)', fontsize=12)
    ax1.set_title('A. Mediation: Indirect Effect\n(Gastric → Brain → Motion)', fontsize=14)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot 2: Proportion Mediated
    ax2 = axes[0, 1]
    prop_vals = [med_summary[med_summary['motion_param'] == m]['prop_mean'].values[0]
                 for m in MOTION_COLS]
    prop_errs = [med_summary[med_summary['motion_param'] == m]['prop_std'].values[0]
                 for m in MOTION_COLS]
    # Clip extreme values for visualization
    prop_vals = [np.clip(v, -1, 1) for v in prop_vals]
    ax2.bar(x_pos, prop_vals, yerr=prop_errs, capsize=5, color='purple', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([motion_labels[m] for m in MOTION_COLS], fontsize=10, rotation=45, ha='right')
    ax2.set_ylabel('Proportion Mediated', fontsize=12)
    ax2.set_title('B. Proportion of Total Effect Mediated\nby Brain Activity', fontsize=14)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% mediation')
    ax2.legend(fontsize=9)

    # Plot 3: Bootstrap Significance
    ax3 = axes[0, 2]
    pct_sig = [boot_summary[boot_summary['motion_param'] == m]['pct_significant'].values[0] * 100
               for m in MOTION_COLS]
    ax3.bar(x_pos, pct_sig, color='teal', alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([motion_labels[m] for m in MOTION_COLS], fontsize=10, rotation=45, ha='right')
    ax3.set_ylabel('% Subjects with Significant Mediation', fontsize=12)
    ax3.set_title('C. Bootstrap: % Subjects with CI\nExcluding Zero', fontsize=14)
    ax3.set_ylim(0, 100)
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax3.legend(fontsize=9)

    # Plot 4: Path a and b coefficients
    ax4 = axes[1, 0]
    width = 0.35
    a_vals = [med_summary[med_summary['motion_param'] == m]['a_mean'].values[0] for m in MOTION_COLS]
    b_vals = [med_summary[med_summary['motion_param'] == m]['b_mean'].values[0] for m in MOTION_COLS]
    ax4.bar(x_pos - width/2, a_vals, width, label='Path a (Gastric→Brain)', color='steelblue', alpha=0.7)
    ax4.bar(x_pos + width/2, b_vals, width, label='Path b (Brain→Motion)', color='darkorange', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([motion_labels[m] for m in MOTION_COLS], fontsize=10, rotation=45, ha='right')
    ax4.set_ylabel('Path Coefficient', fontsize=12)
    ax4.set_title('D. Mediation Path Coefficients', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot 5: Granger Causality Summary
    ax5 = axes[1, 1]
    if len(granger_df) > 0:
        # Summarize Granger results
        gc_summary = granger_df.groupby(['cause', 'effect']).agg({
            'any_significant': 'mean',
            'min_p_value': 'mean'
        }).reset_index()

        # Create heatmap-like visualization for key causal paths
        key_paths = [
            ('gastric', 'brain'),
            ('brain', 'gastric'),
            ('gastric', 'trans_z'),
            ('brain', 'trans_z')
        ]
        path_labels = ['G→B', 'B→G', 'G→Tz', 'B→Tz']

        pct_sig_gc = []
        for cause, effect in key_paths:
            subset = gc_summary[(gc_summary['cause'] == cause) & (gc_summary['effect'] == effect)]
            if len(subset) > 0:
                pct_sig_gc.append(subset['any_significant'].values[0] * 100)
            else:
                pct_sig_gc.append(0)

        ax5.bar(range(len(key_paths)), pct_sig_gc, color='coral', alpha=0.7)
        ax5.set_xticks(range(len(key_paths)))
        ax5.set_xticklabels(path_labels, fontsize=11)
        ax5.set_ylabel('% Subjects Significant', fontsize=12)
        ax5.set_title('E. Granger Causality\n(G=Gastric, B=Brain, Tz=Trans_z)', fontsize=14)
        ax5.set_ylim(0, 100)
    else:
        ax5.text(0.5, 0.5, 'No Granger data', ha='center', va='center', fontsize=12)
        ax5.set_title('E. Granger Causality', fontsize=14)

    # Plot 6: Mediation Model Diagram
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Draw mediation model
    ax6.text(0.1, 0.5, 'Gastric\n(X)', ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax6.text(0.5, 0.85, 'Brain\n(M)', ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax6.text(0.9, 0.5, 'Motion\n(Y)', ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Draw arrows
    ax6.annotate('', xy=(0.35, 0.75), xytext=(0.2, 0.6),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax6.text(0.22, 0.72, 'a', fontsize=12, color='blue', fontweight='bold')

    ax6.annotate('', xy=(0.8, 0.6), xytext=(0.65, 0.75),
                 arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax6.text(0.75, 0.72, 'b', fontsize=12, color='orange', fontweight='bold')

    ax6.annotate('', xy=(0.78, 0.5), xytext=(0.22, 0.5),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=2, linestyle='--'))
    ax6.text(0.5, 0.42, "c' (direct)", fontsize=11, color='gray', ha='center')

    ax6.text(0.5, 0.15, "Indirect Effect = a × b\nTotal Effect c = c' + ab",
             ha='center', va='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax6.set_title('F. Mediation Model', fontsize=14)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)

    plt.suptitle('Mediation Analysis: Does Brain Mediate Gastric-Motion Relationship?',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")


##############################################################################
# Main Execution
##############################################################################

def run_mediation_analysis():
    """Run complete mediation analysis pipeline."""
    print("="*70)
    print("MEDIATION ANALYSIS: Does Brain Mediate Gastric-Motion Effects?")
    print("="*70)

    # Load data
    print("\n[1/5] Loading subject data...")
    all_data = load_all_subject_data()
    if not all_data:
        print("ERROR: No valid data loaded. Exiting.")
        return
    print(f"  Loaded {len(all_data)} subject-run pairs")

    # Run analyses
    print("\n[2/5] Computing classical mediation analysis...")

    all_mediation_results = []
    all_bootstrap_results = []
    all_granger_results = []

    for idx, ((subj, run), data) in enumerate(all_data.items()):
        print(f"  Processing {subj} run {run} ({idx+1}/{len(all_data)})...")

        gastric = data['gastric']
        motion_filt = data['motion_filtered'].values

        for i, motion_name in enumerate(MOTION_COLS):
            motion_i = motion_filt[:, i]

            # Create synthetic brain mediator (replace with actual ROI data)
            brain_mediator = create_synthetic_brain_mediator(gastric, motion_i)

            # Classical mediation
            med_results = mediation_analysis(gastric, brain_mediator, motion_i)
            med_results['subject'] = subj
            med_results['run'] = run
            med_results['motion_param'] = motion_name
            all_mediation_results.append(med_results)

            # Bootstrap mediation (with reduced iterations for speed)
            boot_results = bootstrap_mediation(gastric, brain_mediator, motion_i,
                                               n_bootstrap=1000)  # Reduced for speed
            boot_results['subject'] = subj
            boot_results['run'] = run
            boot_results['motion_param'] = motion_name
            all_bootstrap_results.append(boot_results)

        # Granger causality (for first motion parameter as example)
        brain_example = create_synthetic_brain_mediator(gastric, motion_filt[:, 2])  # trans_z
        gc_df = run_full_granger_analysis(gastric, brain_example, motion_filt, MOTION_COLS)
        gc_df['subject'] = subj
        gc_df['run'] = run
        all_granger_results.append(gc_df)

    # Combine results
    print("\n[3/5] Combining results...")
    mediation_df = pd.DataFrame(all_mediation_results)
    bootstrap_df = pd.DataFrame(all_bootstrap_results)
    granger_df = pd.concat(all_granger_results, ignore_index=True)

    # Create summary statistics
    print("\n[4/5] Computing summary statistics...")
    summary_data = []
    for motion_param in MOTION_COLS:
        med_subset = mediation_df[mediation_df['motion_param'] == motion_param]
        boot_subset = bootstrap_df[bootstrap_df['motion_param'] == motion_param]

        summary_data.append({
            'motion_param': motion_param,
            # Mediation summary
            'indirect_effect_mean': med_subset['indirect_effect'].mean(),
            'indirect_effect_std': med_subset['indirect_effect'].std(),
            'proportion_mediated_mean': med_subset['proportion_mediated'].mean(),
            'proportion_mediated_std': med_subset['proportion_mediated'].std(),
            'path_a_mean': med_subset['path_a'].mean(),
            'path_b_mean': med_subset['path_b'].mean(),
            'pct_sobel_significant': (med_subset['sobel_p'] < 0.05).mean() * 100,
            # Bootstrap summary
            'pct_bootstrap_significant': boot_subset['ci_excludes_zero'].mean() * 100,
            'n_subjects': len(med_subset)
        })

    summary_df = pd.DataFrame(summary_data)

    # Save results
    print("\n[5/5] Saving results...")
    mediation_df.to_csv(OUTPUT_MEDIATION_CSV, index=False)
    bootstrap_df.to_csv(OUTPUT_BOOTSTRAP_CSV, index=False)
    granger_df.to_csv(OUTPUT_GRANGER_CSV, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"  - Mediation results: {OUTPUT_MEDIATION_CSV}")
    print(f"  - Bootstrap results: {OUTPUT_BOOTSTRAP_CSV}")
    print(f"  - Granger results: {OUTPUT_GRANGER_CSV}")
    print(f"  - Summary: {OUTPUT_SUMMARY_CSV}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Mediation Analysis Results")
    print("="*70)
    print(f"\n{'Motion':<12} {'Indirect':>10} {'Prop Med':>10} {'%Sobel':>10} {'%Boot':>10}")
    print("-"*55)
    for _, row in summary_df.iterrows():
        print(f"{row['motion_param']:<12} {row['indirect_effect_mean']:>10.4f} "
              f"{row['proportion_mediated_mean']:>10.2f} "
              f"{row['pct_sobel_significant']:>9.1f}% "
              f"{row['pct_bootstrap_significant']:>9.1f}%")

    # Generate plots
    plot_mediation_results(mediation_df, bootstrap_df, granger_df, OUTPUT_PLOT_PATH)

    # Print interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    avg_prop_mediated = summary_df['proportion_mediated_mean'].mean()
    avg_pct_sig = summary_df['pct_bootstrap_significant'].mean()

    print(f"\n1. MEDIATION STRENGTH:")
    print(f"   - Average proportion mediated: {avg_prop_mediated:.2f}")
    if avg_prop_mediated > 0.5:
        print(f"   - Strong mediation: Brain substantially mediates gastric-motion relationship")
    elif avg_prop_mediated > 0.2:
        print(f"   - Partial mediation: Brain partially mediates gastric-motion relationship")
    else:
        print(f"   - Weak/No mediation: Brain does not substantially mediate relationship")

    print(f"\n2. STATISTICAL SIGNIFICANCE:")
    print(f"   - Average % subjects with significant mediation: {avg_pct_sig:.1f}%")

    print(f"\n3. IMPORTANT NOTES:")
    print(f"   - This analysis uses a SYNTHETIC brain mediator for demonstration")
    print(f"   - Replace with actual brain ROI data (e.g., insula, somatosensory cortex)")
    print(f"   - Use regions showing significant gastric-brain PLV as mediators")
    print(f"   - Consider multiple potential mediators for comprehensive analysis")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_mediation_analysis()
