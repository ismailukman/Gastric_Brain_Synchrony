#!/usr/bin/env python3
"""
EGG (Electrogastrography) Preprocessing Pipeline

This script preprocesses raw EGG data recorded during fMRI sessions.
It supports both single-subject processing and batch processing of multiple subjects.

Main Processing Steps:
1. Load raw EGG data (.acq format from AcqKnowledge/Biopac)
2. Detect MRI trigger timing to align with fMRI acquisition
3. Slice the recording to match fMRI scan duration
4. Resample to intermediate sampling rate (10 Hz)
5. Perform spectral analysis to identify dominant gastric frequency
6. Apply bandpass filter around dominant frequency
7. Optionally apply z-score normalization
8. Save processed data and generate diagnostic plots

Usage:
    Single subject:
        python preprocess_gastric_data.py sub-01 1

    Batch processing:
        python preprocess_gastric_data.py --batch

    Batch with specific metadata file:
        python preprocess_gastric_data.py --batch --metadata path/to/metadata.csv

"""

import bioread
import pandas as pd
import scipy as sp
import numpy as np
import os
import sys
import pathlib
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from utils.gastric_utils import plot_signal, plot_trigger
from utils.spect_utils import powerspect
from mne.filter import filter_data
from scipy import stats

# Import configuration
from config import (
    egg_data_path, plots_path, derivatives_path, metadata_file,
    intermediate_sample_rate, trigger_channel, filter_order,
    bandpass_lim, transition_width, window, overlap, freq_range,
    zscore_flag, clean_level, multi_thread, num_threads
)


def preprocess_single_subject(subject_name, run, metadata_df=None, verbose=True):
    """
    Preprocess EGG data for a single subject/run.

    Parameters
    ----------
    subject_name : str
        Subject identifier (e.g., 'sub-01')
    run : str or int
        Run number
    metadata_df : pandas.DataFrame, optional
        Pre-loaded metadata DataFrame. If None, will load from metadata_file.
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    dict
        Processing results containing:
        - 'status': 'success' or 'error'
        - 'subject': subject name
        - 'run': run number
        - 'max_freq': dominant gastric frequency (if successful)
        - 'dominant_channel': dominant channel number (if successful)
        - 'error': error message (if failed)
    """
    run = str(run)
    result = {'subject': subject_name, 'run': run, 'status': 'error'}

    if verbose:
        print(f'\n{"="*60}')
        print(f'Processing subject: {subject_name}, run: {run}')
        print(f'{"="*60}')

    # STEP 1: Setup output directories
    plot_path = os.path.join(plots_path, subject_name, f'{subject_name}{run}')
    data_path = os.path.join(derivatives_path, subject_name, f'{subject_name}{run}')

    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    # STEP 2: Load metadata
    if metadata_df is None:
        if not os.path.exists(metadata_file):
            result['error'] = f'Metadata file not found: {metadata_file}'
            print(f"Error: {result['error']}")
            return result
        metadata_df = pd.read_csv(metadata_file)

    # Get metadata for this subject/run
    record_meta_df = metadata_df.loc[
        (metadata_df['subject'] == subject_name) &
        (metadata_df['run'] == int(run))
    ]

    if record_meta_df.empty:
        result['error'] = f'No metadata found for subject={subject_name}, run={run}'
        print(f"Error: {result['error']}")
        return result

    record_meta = record_meta_df.iloc[0].to_dict()

    # STEP 3: Load raw EGG data
    egg_file_path = os.path.join(
        egg_data_path, subject_name, 'egg', f'{subject_name}_rest{run}.acq'
    )

    if verbose:
        print(f'Reading EGG file: {egg_file_path}')

    if not os.path.exists(egg_file_path):
        # Try alternative path: {egg_data_path}/{subject}/{subject}_Acq/{subject}_EGG.acq
        alt_egg_file_path = os.path.join(
            egg_data_path, subject_name, f'{subject_name}_Acq', f'{subject_name}_EGG.acq'
        )
        if os.path.exists(alt_egg_file_path):
             print(f'File not found at default path. Found at: {alt_egg_file_path}')
             egg_file_path = alt_egg_file_path
        else:
            # Try another alternative path: {egg_data_path}/{subject}_Acq/{subject}_EGG.acq
            alt_egg_file_path_2 = os.path.join(
                egg_data_path, f'{subject_name}_Acq', f'{subject_name}_EGG.acq'
            )
            if os.path.exists(alt_egg_file_path_2):
                 print(f'File not found at default path. Found at: {alt_egg_file_path_2}')
                 egg_file_path = alt_egg_file_path_2
            else:
                result['error'] = f'File not found: {egg_file_path}'
                print(f"Error: {result['error']}")
                return result

    try:
        data = bioread.read_file(egg_file_path)
        if verbose:
            print(f"DEBUG: Actual number of channels in file: {len(data.channels)}")
            for i, ch in enumerate(data.channels):
                print(f"DEBUG: Channel {i}: {ch.name} (samples: {len(ch.data)})")
    except Exception as e:
        result['error'] = f'Failed to read EGG file: {str(e)}'
        print(f"Error: {result['error']}")
        return result

    # STEP 4: Extract recording parameters
    original_sample_rate = data.channels[0].samples_per_second
    signal_time = data.channels[0].time_index
    duration = original_sample_rate * record_meta['mri_length']
    num_gast = record_meta['num_channles']

    if verbose:
        print(f'Original sample rate: {original_sample_rate} Hz')
        print(f'MRI duration: {record_meta["mri_length"]} seconds')
        print(f'Number of EGG channels: {num_gast}')

    # STEP 5: Detect or set MRI trigger indices
    trigger = data.channels[trigger_channel].data.astype(int)

    if record_meta['trigger_start'] == 'auto':
        if trigger[0] == 0:
            action_idx = [np.where(trigger)[0][0], np.where(trigger)[0][0] + int(duration)]
        else:
            no_trigger_locs = np.where(trigger == 0)[0]
            trigger_locs = np.where(trigger >= 0.999)[0]
            str_loc = trigger_locs[trigger_locs > no_trigger_locs[0]][0]
            action_idx = [str_loc, str_loc + int(duration)]
            warnings.warn('Signal started with trigger > 0, skipped to next trigger')
    else:
        if verbose:
            print('Using manual trigger start')
        trigger_start = int(max(float(record_meta['trigger_start']), 0) * original_sample_rate)
        action_idx = [trigger_start, trigger_start + int(duration)]

    # STEP 6: Plot trigger signal
    plot_trigger(trigger, action_idx, original_sample_rate,
                 subject=subject_name, run=run, save_path=plot_path)

    # STEP 7: Slice recording according to MRI trigger timing
    signal_time = data.channels[0].time_index[:int(duration)]
    signal_egg = [data.channels[i].data[action_idx[0]:action_idx[1]] for i in range(num_gast)]

    # STEP 8: Plot sliced EGG signal
    plot_signal(signal_time, signal_egg, 'EGG sliced signal',
                'sliced_signal', f'{subject_name}_{run}', plot_path)

    # STEP 9: Resample to intermediate sampling rate (10Hz)
    resample_n_points = int((len(signal_time) / original_sample_rate) * intermediate_sample_rate)
    signal_egg = [sp.signal.resample(signal_egg[i], resample_n_points) for i in range(len(signal_egg))]
    signal_time = np.arange(0, resample_n_points / intermediate_sample_rate, 1.0 / intermediate_sample_rate)

    # STEP 10: Plot resampled signal
    plot_signal(signal_time, signal_egg, 'EGG signal after resampling',
                'post_first_resample_', f'{subject_name}_{run}', plot_path)

    # STEP 11: Perform Welch power spectral analysis
    max_freq, dominant_channel_num, power_spect_data_list = powerspect(
        signal_egg, window, overlap, intermediate_sample_rate,
        freq_range, True, subject_name, run, plot_path,
        dominant=record_meta['dominant_channel'],
        dominant_freq=record_meta['dominant_frequency']
    )

    # STEP 12: Override automatic detection if manual values specified
    if (record_meta['dominant_channel'] != 'auto') or (record_meta['dominant_frequency'] != 'auto'):
        if verbose:
            print('Manual channel/frequency override specified')
        if (record_meta['dominant_channel'] == 'auto') and (record_meta['dominant_frequency'] == 'auto'):
            raise Exception('Both "dominant_channel" and "dominant_frequency" must be set together')
        max_freq = float(record_meta['dominant_frequency'])
        dominant_channel_num = int(record_meta['dominant_channel'])

    if verbose:
        print(f'Dominant frequency: {max_freq:.4f} Hz')
        print(f'Dominant channel: {dominant_channel_num + 1}')

    # STEP 13: Apply bandpass filter around dominant frequency
    signal_egg_selected = filter_data(
        signal_egg[dominant_channel_num],
        sfreq=intermediate_sample_rate,
        l_freq=max_freq - bandpass_lim,
        h_freq=max_freq + bandpass_lim,
        picks=None, n_jobs=1, method='fir', phase='zero-double',
        filter_length=int(filter_order * np.floor(intermediate_sample_rate / (max_freq - bandpass_lim))),
        l_trans_bandwidth=transition_width * (max_freq - bandpass_lim),
        h_trans_bandwidth=transition_width * (max_freq + bandpass_lim),
        fir_window='hamming', fir_design='firwin2'
    )

    # STEP 14: Apply z-score normalization if specified
    if zscore_flag:
        signal_egg_selected = stats.zscore(signal_egg_selected)

    # STEP 15: Plot final filtered signal
    plot_signal(signal_time, [signal_egg_selected], 'EGG filtered',
                'egg_filtered', f'{subject_name}_{run}', plot_path)

    # STEP 16: Save processed data
    output_data_file = os.path.join(data_path, f'gast_data_{subject_name}_run{run}{clean_level}.npy')
    output_freq_file = os.path.join(data_path, f'max_freq{subject_name}_run{run}{clean_level}.npy')

    np.save(output_data_file, signal_egg_selected)
    np.save(output_freq_file, max_freq)

    if verbose:
        print(f'\nOutput files saved:')
        print(f'  - Data: {output_data_file}')
        print(f'  - Frequency: {output_freq_file}')
        print(f'  - Plots: {plot_path}/')

    result['status'] = 'success'
    result['max_freq'] = max_freq
    result['dominant_channel'] = dominant_channel_num
    result['output_data'] = output_data_file
    result['output_freq'] = output_freq_file

    return result


def batch_process(metadata_path=None, n_jobs=None):
    """
    Process multiple subjects/runs from metadata file.

    Parameters
    ----------
    metadata_path : str, optional
        Path to metadata CSV file. If None, uses default from config.
    n_jobs : int, optional
        Number of parallel jobs. If None, uses config setting.

    Returns
    -------
    list of dict
        Processing results for each subject/run
    """
    if metadata_path is None:
        metadata_path = metadata_file

    if n_jobs is None:
        n_jobs = num_threads if multi_thread else 1

    print(f'\n{"="*60}')
    print('BATCH PROCESSING MODE')
    print(f'{"="*60}')
    print(f'Metadata file: {metadata_path}')
    print(f'Parallel jobs: {n_jobs}')

    # Load metadata
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        return []

    metadata_df = pd.read_csv(metadata_path)
    subjects_runs = list(zip(metadata_df['subject'], metadata_df['run']))

    print(f'Total subjects/runs to process: {len(subjects_runs)}')
    print(f'{"="*60}\n')

    results = []

    if n_jobs == 1:
        # Sequential processing
        for subject, run in subjects_runs:
            result = preprocess_single_subject(subject, run, metadata_df, verbose=True)
            results.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(preprocess_single_subject, subject, run, metadata_df, verbose=False): (subject, run)
                for subject, run in subjects_runs
            }

            for future in as_completed(futures):
                subject, run = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = 'SUCCESS' if result['status'] == 'success' else 'FAILED'
                    print(f'[{status}] {subject} run {run}')
                except Exception as e:
                    print(f'[ERROR] {subject} run {run}: {str(e)}')
                    results.append({
                        'subject': subject,
                        'run': run,
                        'status': 'error',
                        'error': str(e)
                    })

    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    print(f'\n{"="*60}')
    print('BATCH PROCESSING COMPLETE')
    print(f'{"="*60}')
    print(f'Successful: {successful}/{len(results)}')
    print(f'Failed: {failed}/{len(results)}')

    if failed > 0:
        print('\nFailed subjects:')
        for r in results:
            if r['status'] == 'error':
                print(f"  - {r['subject']} run {r['run']}: {r.get('error', 'Unknown error')}")

    return results


def main():
    """Main entry point for command line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mutually exclusive: single subject or batch mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('subject', nargs='?', metavar='subject_name',
                       help='Subject identifier (e.g., sub-01)')
    group.add_argument('--batch', action='store_true',
                       help='Run batch processing for all subjects in metadata file')

    parser.add_argument('run', nargs='?', metavar='run_number',
                        help='Run number (required for single subject mode)')
    parser.add_argument('--metadata', metavar='FILE',
                        help='Path to metadata CSV file (default: from config)')
    parser.add_argument('--jobs', '-j', type=int, metavar='N',
                        help='Number of parallel jobs for batch processing')

    args = parser.parse_args()

    if args.batch:
        # Batch processing mode
        batch_process(
            metadata_path=args.metadata,
            n_jobs=args.jobs
        )
    else:
        # Single subject mode
        if args.run is None:
            parser.error('Run number is required for single subject processing')

        preprocess_single_subject(args.subject, args.run)


if __name__ == '__main__':
    main()
