import os
import sys
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import resample, hilbert
from scipy.stats import false_discovery_control, mannwhitneyu
from mne.filter import filter_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

##############################################################################
# Configuration                                                              #
##############################################################################

PARENT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = PARENT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from config import main_project_path, clean_level, sample_rate_fmri, intermediate_sample_rate, bandpass_lim, filter_order, transition_width

META_DATAFRAME_PATH = PROJECT_ROOT / "dataframes" / "egg_brain_meta_data.csv"

MOTION_FILE_TEMPLATE = os.path.join(
    main_project_path,
    "BIDS_data",
    "sub_motion_files",
    "sub-{sub}_dfile.r0{run}.1D"
)

EGG_FILE_TEMPLATE = os.path.join(
    main_project_path,
    "derivatives",
    "brain_gast",
    "{sub}",
    "{sub}{run}",
    "gast_data_{sub}_run{run}{clean_level}.npy"
)

GASTRIC_FREQ_TEMPLATE = os.path.join(
    main_project_path,
    "derivatives",
    "brain_gast",
    "{sub}",
    "{sub}{run}",
    "max_freq{sub}_run{run}{clean_level}.npy"
)

SAMPLE_RATE_FMRI = sample_rate_fmri
BANDPASS_LIM = bandpass_lim
FILTER_ORDER = filter_order
TRANSITION_WIDTH = transition_width
EGG_INTERMEDIATE_SFREQ = intermediate_sample_rate

# Output paths - V5 filenames
OUTPUT_PLV_PATH = PROJECT_ROOT / "dataframes" / "plvs_egg_w_motion_v5.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "dataframes" / "motion_summary_v5.csv"
OUTPUT_POPULATION_PATH = PROJECT_ROOT / "dataframes" / "population_level_v5.csv"
OUTPUT_PLOT_PATH = PROJECT_ROOT.parent / "plots" / "plv_awplv_densities_v5.png"


##############################################################################
# Helper Functions                                                           #
##############################################################################
def bp_filter_confounds(df, gastric_peak, sample_rate=SAMPLE_RATE_FMRI,
                        bandpass_lim=BANDPASS_LIM, filter_order=FILTER_ORDER,
                        transition_width=TRANSITION_WIDTH, verbose=None):
    """
    Bandpass-filter each column in df around the subject-specific gastric_peak.
    """
    l_freq = gastric_peak - bandpass_lim
    h_freq = gastric_peak + bandpass_lim
    filter_length = int(filter_order * np.floor(sample_rate / (gastric_peak - bandpass_lim)))

    confound_filtered = filter_data(
        data=df.values.T,
        sfreq=sample_rate,
        l_freq=l_freq,
        h_freq=h_freq,
        filter_length=filter_length,
        l_trans_bandwidth=transition_width * (gastric_peak - bandpass_lim),
        h_trans_bandwidth=transition_width * (gastric_peak + bandpass_lim),
        n_jobs=1,
        method='fir',
        phase='zero-double',
        fir_window='hamming',
        fir_design='firwin2',
        verbose=verbose
    )
    return pd.DataFrame(confound_filtered.T, columns=df.columns, index=df.index)


def calc_plv(signal_a, signal_b):
    """
    Compute the Phase Locking Value (PLV) between two signals of equal length.
    Uses analytic signal (Hilbert) to extract instantaneous phase.
    """
    assert len(signal_a) == len(signal_b), "Signals must be the same length."
    a_phase = np.angle(hilbert(signal_a))
    b_phase = np.angle(hilbert(signal_b))
    plv = np.abs(np.mean(np.exp(1j * (a_phase - b_phase))))
    return plv


def calc_awplv(signal_a, signal_b):
    """
    Compute the Amplitude-Weighted Phase Locking Value (awPLV) between two signals.
    The phase-locking is weighted by the product of both amplitudes (gastric x motion).

    Formula: |Σ((A_gastric · A_motion / Σ(A_gastric · A_motion)) · e^(i·Δφ))|
    """
    assert len(signal_a) == len(signal_b), "Signals must be the same length."

    analytic_a = hilbert(signal_a)
    analytic_b = hilbert(signal_b)

    phase_a = np.angle(analytic_a)
    phase_b = np.angle(analytic_b)
    amplitude_a = np.abs(analytic_a)
    amplitude_b = np.abs(analytic_b)
    
    # Calculate phase difference
    phase_diff_complex = np.exp(1j * (phase_a - phase_b))

    # Calculate product of amplitudes for weighting
    product_amplitudes = amplitude_a * amplitude_b

    # Calculate normalized weights
    sum_product_amplitudes = np.sum(product_amplitudes)
    if sum_product_amplitudes == 0: # Avoid division by zero
        return 0.0
    weights = product_amplitudes / sum_product_amplitudes

    # Compute awPLV
    awplv = np.abs(np.sum(weights * phase_diff_complex))
    return awplv

def get_motion_column_names():
    return ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

def confound_summary(df=None, method='mean_abs'):
    measures = get_motion_column_names()
    out_dict = {}
    for m in measures:
        if m not in df.columns:
            continue
        if method == 'mean_abs':
            stat = np.mean(np.abs(df[m].values))
            out_dict[f"{m}_{method}"] = stat
        elif method == 'mean_abs_diff':
            stat = np.mean(np.abs(np.diff(df[m].values)))
            out_dict[f"{m}_{method}"] = stat
    return out_dict

def plot_plv_awplv_densities(df, pop_df, null_dists):
    """
    Plots Gaussian Kernel density estimates of PLV and awPLV for each motion parameter.
    Includes both empirical and null distributions in a nested 3x2 grid structure.
    """
    fig = plt.figure(figsize=(16, 18))

    # Create a 3x2 grid for the main layout with increased spacing
    outer_grid = gridspec.GridSpec(3, 2, figure=fig, wspace=0.3, hspace=0.5, top=0.9, bottom=0.15)

    # Add main column titles (doubled font size: 18 → 36)
    fig.text(0.3, 0.95, 'Translation', ha='center', va='center', fontsize=36, weight='bold')
    fig.text(0.72, 0.95, 'Rotation', ha='center', va='center', fontsize=36, weight='bold')

    # Add main row titles (doubled font size: 16 → 32)
    row_labels = ['X', 'Y', 'Z']
    y_coords = [0.75, 0.48, 0.21] # Adjusted for new layout
    for i, label in enumerate(row_labels):
        fig.text(0.04, y_coords[i], label, fontsize=32, rotation=90, ha='center', va='center', weight='bold')

    motion_params_map = {
        (0, 0): 'trans_x', (1, 0): 'trans_y', (2, 0): 'trans_z',
        (0, 1): 'rot_x', (1, 1): 'rot_y', (2, 1): 'rot_z'
    }

    for i in range(3):  # Rows
        for j in range(2):  # Cols
            param = motion_params_map[(i, j)]

            # Create a 1x2 nested grid for PLV and awPLV plots
            inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[i, j], wspace=0.25)

            param_data = df[df['motion_param'] == param]

            # --- PLV Subplot ---
            ax_plv = fig.add_subplot(inner_grid[0])
            plv_color = 'blue'
            sns.kdeplot(param_data['plv_empirical'], ax=ax_plv, color=plv_color, fill=True, alpha=0.3, label='Empirical')
            sns.kdeplot(null_dists[param]['PLV'], ax=ax_plv, color=plv_color, linestyle='--', label='Null')
            plv_p = pop_df.loc[(pop_df['motion_param'] == param) & (pop_df['metric'] == 'PLV'), 'p_fdr'].iloc[0]

            # Determine significance marker
            if plv_p < 0.001:
                sig_marker = '***'
            elif plv_p < 0.01:
                sig_marker = '**'
            elif plv_p < 0.05:
                sig_marker = '*'
            else:
                sig_marker = ''

            # Removed "PLV\n" from title, added significance asterisks
            ax_plv.set_title(f"q={plv_p:.3f} {sig_marker}", fontsize=20)
            # Show legend only in top-left subplot (i==0, j==0)
            if i == 0 and j == 0:
                ax_plv.legend(fontsize=20, loc='upper right')
            # Doubled font size for axis labels: default 10 → 20
            ax_plv.set_ylabel('Density', fontsize=20)
            ax_plv.set_xlabel('PLV', fontsize=20)
            ax_plv.tick_params(labelsize=16)

            # --- awPLV Subplot ---
            ax_awplv = fig.add_subplot(inner_grid[1])
            awplv_color = 'red'
            sns.kdeplot(param_data['awplv_empirical'], ax=ax_awplv, color=awplv_color, fill=True, alpha=0.3, label='Empirical')
            sns.kdeplot(null_dists[param]['awPLV'], ax=ax_awplv, color=awplv_color, linestyle='--', label='Null')
            awplv_p = pop_df.loc[(pop_df['motion_param'] == param) & (pop_df['metric'] == 'awPLV'), 'p_fdr'].iloc[0]

            # Determine significance marker
            if awplv_p < 0.001:
                sig_marker = '***'
            elif awplv_p < 0.01:
                sig_marker = '**'
            elif awplv_p < 0.05:
                sig_marker = '*'
            else:
                sig_marker = ''

            # Removed "awPLV\n" from title, added significance asterisks
            ax_awplv.set_title(f"q={awplv_p:.3f} {sig_marker}", fontsize=20)
            # Show legend only in top-left subplot (i==0, j==0)
            if i == 0 and j == 0:
                ax_awplv.legend(fontsize=20, loc='upper right')
            ax_awplv.set_ylabel('')
            # Doubled font size for axis labels: default 10 → 20
            ax_awplv.set_xlabel('awPLV', fontsize=20)
            ax_awplv.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax_awplv.tick_params(labelsize=16)

    # Doubled font size for caption: 12 → 24, split into multiple lines
    fig.text(0.5, 0.05,
             "Gaussian Kernel density estimates (smoothed histograms)\nof synchronization between head motions and the gastric rhythm.",
             ha='center', fontsize=24)

    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"\nPlot saved to {OUTPUT_PLOT_PATH}")


##############################################################################
# Main Analysis                                                              #
##############################################################################

def main():
    """
    Main function to compute EGG-Motion Synchrony using PLV and awPLV with mismatch null distribution.
    VERSION 5 CHANGES (from v4):
    - Added Amplitude-Weighted PLV (awPLV) analysis alongside PLV.
    - Individual-level results include null distribution statistics for both metrics.
    - Separate FDR and Bonferroni corrections for PLV and awPLV (6 tests each).
    - Population level analysis includes both PLV and awPLV with separate corrections.
    - Output CSV includes 24 columns: subject, run, motion_param, empirical values,
      null statistics (median, mean, std), p-values, corrected p-values, and
      significance flags for both PLV and awPLV, plus n_null_samples.
    - Added a 3x2 nested grid plot showing PLV and awPLV densities for each motion parameter.
    """
    # Load metadata
    record_meta_pd = pd.read_csv(META_DATAFRAME_PATH)
    if clean_level == 'strict_gs_cardiac':
        record_meta_pd = record_meta_pd.loc[(record_meta_pd['ppu_exclude'] == False) & (record_meta_pd['ppu_found'] == True)]

    subjects_runs = list(zip(record_meta_pd['subject'], record_meta_pd['run']))
    motion_cols = get_motion_column_names()

    print(f"Processing {len(subjects_runs)} subject-run pairs...")
    print("V5: Calculating PLV and awPLV with corrected null distribution.")

    # STEP 1: Load data
    all_data = {}
    for (subject_name, run) in subjects_runs:
        try:
            motion_path = MOTION_FILE_TEMPLATE.format(sub=subject_name, run=run)
            egg_file = EGG_FILE_TEMPLATE.format(sub=subject_name, run=run, clean_level=clean_level)
            freq_file = GASTRIC_FREQ_TEMPLATE.format(sub=subject_name, run=run, clean_level=clean_level)

            if not all(os.path.isfile(p) for p in [motion_path, egg_file, freq_file]):
                print(f"Missing file for {subject_name} run {run}")
                continue

            motion_data = np.loadtxt(motion_path)
            df_confound = pd.DataFrame(motion_data, columns=motion_cols)
            gastric_signal = np.load(egg_file)
            gastric_peak = float(np.load(freq_file).flatten()[0])

            n_points_fmri = int((len(gastric_signal) / EGG_INTERMEDIATE_SFREQ) * SAMPLE_RATE_FMRI)
            if n_points_fmri < 10: continue

            gastric_signal_resampled = resample(gastric_signal, n_points_fmri)
            min_length = min(len(gastric_signal_resampled), len(df_confound))
            gastric_signal_resampled = gastric_signal_resampled[:min_length]
            df_confound = df_confound.iloc[:min_length]

            df_confound_filt = bp_filter_confounds(df_confound, gastric_peak)

            all_data[(subject_name, run)] = {
                'subject': subject_name, # Storing subject name for null dist calculation
                'gastric': gastric_signal_resampled,
                'motion_filtered': df_confound_filt,
                'motion_raw': df_confound
            }
            print(f"✓ Loaded {subject_name} run {run}")
        except Exception as e:
            print(f"✗ Error loading {subject_name} run {run}: {e}")

    if not all_data:
        print("No valid data. Exiting.")
        return
    print(f"\nLoaded {len(all_data)} subject-run pairs")

    # STEP 2: Compute empirical and null PLV/awPLV
    print("\nComputing PLV and awPLV with mismatch null distribution...")
    results = []
    summary_list = []
    for idx, ((subj, run), data) in enumerate(all_data.items()):
        print(f"\nProcessing {subj} run {run} ({idx+1}/{len(all_data)})...")
        gastric_current = data['gastric']
        
        summary_dict = confound_summary(data['motion_filtered'], method='mean_abs')
        summary_dict.update({'subject': subj, 'run': run})
        summary_list.append(summary_dict)

        for motion_param in motion_cols:
            motion_signal = data['motion_filtered'][motion_param].values
            
            plv_empirical = calc_plv(gastric_current, motion_signal)
            awplv_empirical = calc_awplv(gastric_current, motion_signal)

            null_plvs, null_awplvs = [], []
            for (other_subj, _), other_data in all_data.items():
                if other_subj == subj: continue

                gastric_mismatch = other_data['gastric']
                min_len = min(len(motion_signal), len(gastric_mismatch))

                null_plvs.append(calc_plv(gastric_mismatch[:min_len], motion_signal[:min_len]))
                null_awplvs.append(calc_awplv(gastric_mismatch[:min_len], motion_signal[:min_len]))

            # Convert to arrays for statistical calculations
            null_plvs_arr = np.array(null_plvs)
            null_awplvs_arr = np.array(null_awplvs)

            # Calculate p-values and null distribution statistics
            p_value_plv = np.mean(null_plvs_arr >= plv_empirical)
            p_value_awplv = np.mean(null_awplvs_arr >= awplv_empirical)

            plv_null_median = np.median(null_plvs_arr)
            plv_null_mean = np.mean(null_plvs_arr)
            plv_null_std = np.std(null_plvs_arr)

            awplv_null_median = np.median(null_awplvs_arr)
            awplv_null_mean = np.mean(null_awplvs_arr)
            awplv_null_std = np.std(null_awplvs_arr)

            results.append({
                'subject': subj, 'run': run, 'motion_param': motion_param,
                'plv_empirical': plv_empirical,
                'plv_null_median': plv_null_median,
                'plv_null_mean': plv_null_mean,
                'plv_null_std': plv_null_std,
                'p_value_plv': p_value_plv,
                'awplv_empirical': awplv_empirical,
                'awplv_null_median': awplv_null_median,
                'awplv_null_mean': awplv_null_mean,
                'awplv_null_std': awplv_null_std,
                'p_value_awplv': p_value_awplv,
                'n_null_samples': len(null_plvs)
            })
            print(f"  {motion_param}: PLV={plv_empirical:.4f} (p={p_value_plv:.4f}), awPLV={awplv_empirical:.4f} (p={p_value_awplv:.4f})")

    results_df = pd.DataFrame(results)

    # Add individual-level multiple comparison corrections
    print("\nApplying individual-level FDR and Bonferroni corrections...")

    # Separate PLV and awPLV for independent correction (6 motion params each)
    all_plv_pvals = results_df['p_value_plv'].values
    all_awplv_pvals = results_df['p_value_awplv'].values

    # Apply FDR correction separately for PLV and awPLV
    results_df['p_fdr_plv'] = false_discovery_control(all_plv_pvals, method='bh')
    results_df['p_fdr_awplv'] = false_discovery_control(all_awplv_pvals, method='bh')

    # Apply Bonferroni correction separately for PLV and awPLV (6 motion parameters each)
    n_motion_params = 6
    results_df['p_bonferroni_plv'] = np.minimum(results_df['p_value_plv'] * n_motion_params, 1.0)
    results_df['p_bonferroni_awplv'] = np.minimum(results_df['p_value_awplv'] * n_motion_params, 1.0)

    # Add significance flags for PLV
    results_df['sig_uncorrected_plv'] = results_df['p_value_plv'] < 0.05
    results_df['sig_fdr_plv'] = results_df['p_fdr_plv'] < 0.05
    results_df['sig_bonferroni_plv'] = results_df['p_bonferroni_plv'] < 0.05

    # Add significance flags for awPLV
    results_df['sig_uncorrected_awplv'] = results_df['p_value_awplv'] < 0.05
    results_df['sig_fdr_awplv'] = results_df['p_fdr_awplv'] < 0.05
    results_df['sig_bonferroni_awplv'] = results_df['p_bonferroni_awplv'] < 0.05

    # STEP 2b: Population-level analysis
    print("\n" + "="*70 + "\nPOPULATION-LEVEL ANALYSIS\n" + "="*70)
    population_results = []
    all_null_distributions = {}
    for motion_param in motion_cols:
        all_null_distributions[motion_param] = {}
        for metric, empirical_key, p_key in [('PLV', 'plv_empirical', 'p_value_plv'), 
                                             ('awPLV', 'awplv_empirical', 'p_value_awplv')]:
            
            param_data = results_df[results_df['motion_param'] == motion_param]
            empirical_values = param_data[empirical_key].values

            all_null_values = []
            for idx, ((subj, run), data) in enumerate(all_data.items()):
                motion_signal = data['motion_filtered'][motion_param].values
                for (other_subj, _), other_data in all_data.items():
                    if other_subj == subj: continue
                    gastric_mismatch = other_data['gastric']
                    min_len = min(len(motion_signal), len(gastric_mismatch))
                    if metric == 'PLV':
                        all_null_values.append(calc_plv(gastric_mismatch[:min_len], motion_signal[:min_len]))
                    else:
                        all_null_values.append(calc_awplv(gastric_mismatch[:min_len], motion_signal[:min_len]))
            
            all_null_distributions[motion_param][metric] = all_null_values
            statistic, p_value_mw = mannwhitneyu(empirical_values, all_null_values, alternative='greater')
            
            population_results.append({
                'motion_param': motion_param, 
                'metric': metric,
                'n_empirical': len(empirical_values),
                'n_null': len(all_null_values),
                'mean_empirical': np.mean(empirical_values), 
                'std_empirical': np.std(empirical_values),
                'mean_null': np.mean(all_null_values), 
                'std_null': np.std(all_null_values),
                'effect_size': np.mean(empirical_values) - np.mean(all_null_values),
                'mann_whitney_u': statistic,
                'p_value': p_value_mw
            })

    population_df = pd.DataFrame(population_results)
    
    # Separate testing for PLV and awPLV (6 tests each)
    plv_pop_df = population_df[population_df['metric'] == 'PLV'].copy()
    awplv_pop_df = population_df[population_df['metric'] == 'awPLV'].copy()

    # Correct PLV p-values
    plv_p_values = plv_pop_df['p_value'].values
    plv_pop_df['p_fdr'] = false_discovery_control(plv_p_values, method='bh')
    plv_pop_df['p_bonferroni'] = np.minimum(plv_p_values * len(plv_p_values), 1.0)
    plv_pop_df['sig_uncorrected'] = plv_pop_df['p_value'] < 0.05
    plv_pop_df['sig_fdr'] = plv_pop_df['p_fdr'] < 0.05
    plv_pop_df['sig_bonferroni'] = plv_pop_df['p_bonferroni'] < 0.05

    # Correct awPLV p-values
    awplv_p_values = awplv_pop_df['p_value'].values
    awplv_pop_df['p_fdr'] = false_discovery_control(awplv_p_values, method='bh')
    awplv_pop_df['p_bonferroni'] = np.minimum(awplv_p_values * len(awplv_p_values), 1.0)
    awplv_pop_df['sig_uncorrected'] = awplv_pop_df['p_value'] < 0.05
    awplv_pop_df['sig_fdr'] = awplv_pop_df['p_fdr'] < 0.05
    awplv_pop_df['sig_bonferroni'] = awplv_pop_df['p_bonferroni'] < 0.05

    # Combine back into a single dataframe
    population_df = pd.concat([plv_pop_df, awplv_pop_df]).sort_values(by=['motion_param', 'metric'])
    
    print(population_df.to_string())

    # STEP 3 & 4: Save results
    print("\nSaving results...")
    results_df.to_csv(OUTPUT_PLV_PATH, index=False)
    if summary_list:
        pd.DataFrame(summary_list).to_csv(OUTPUT_SUMMARY_PATH, index=False)
    population_df.to_csv(OUTPUT_POPULATION_PATH, index=False)
    print(f"Individual results saved to {OUTPUT_PLV_PATH}")
    print(f"Population results saved to {OUTPUT_POPULATION_PATH}")
    
    # STEP 5: Plotting
    plot_plv_awplv_densities(results_df, population_df, all_null_distributions)

    print("\nDone! (Version 5 - PLV and awPLV analysis)")

if __name__ == "__main__":
    main()
