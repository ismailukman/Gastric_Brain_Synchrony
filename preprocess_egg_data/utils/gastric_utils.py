"""
Utility functions for EGG signal visualization and processing.

This module provides functions for:
- Plotting EGG signals at various processing stages
- Visualizing trigger signals and timing
- Signal resampling with optional visualization
- Phase extraction and resampling
"""

from matplotlib import pyplot as plt
from scipy import signal
import numpy as np


def plot_signal(x_axis, data, main_title, prefix, id, save_path):
    """
    Plot multi-channel EGG signals.

    Parameters
    ----------
    x_axis : array-like
        Time axis values (in seconds)
    data : list of arrays
        List of signal arrays, one per channel
    main_title : str
        Main title for the plot
    prefix : str
        Prefix for the output filename
    id : str
        Identifier (e.g., subject_run) for the filename
    save_path : str
        Directory path to save the figure
    """
    num_chan = len(data)
    plt.figure(figsize=(30, 12))
    for chan in range(len(data)):
        plt.subplot(num_chan, 1, chan + 1)
        plt.plot(x_axis, data[chan])
        plt.xlabel('Time(sec)', fontsize=10)
        plt.ylabel('Voltage (mV)', fontsize=10)
        plt.title("Channel number " + str(chan + 1), fontsize=11)
    plt.tight_layout(pad=6.0, w_pad=6.0, h_pad=6.0)
    plt.suptitle(main_title, fontsize=18)
    plt.savefig(save_path + '/' + prefix + id + '.png')
    print('Figure saved to:', save_path)
    plt.close('all')


def roundup(x, to):
    """Round x up to the nearest multiple of 'to'."""
    return x if x % to == 0 else x + to - x % to


def resampling(x_axis, data, num_iter, subtitle, rows, columns, resampling_factor,
               main_title, plot_flag, id, save_path):
    """
    Resample signal data with optional visualization.

    Parameters
    ----------
    x_axis : array-like
        Time axis for resampled data
    data : list of arrays
        List of signal arrays to resample
    num_iter : int
        Number of channels to process
    subtitle : str
        Subtitle for each subplot
    rows : int
        Number of subplot rows
    columns : int
        Number of subplot columns
    resampling_factor : int
        Target number of samples after resampling
    main_title : str
        Main title for the plot
    plot_flag : bool
        Whether to generate and save plots
    id : str
        Identifier for the filename
    save_path : str
        Directory path to save the figure

    Returns
    -------
    y : list of arrays
        Resampled signal arrays
    """
    y = []
    if plot_flag:
        plt.figure(figsize=(30, 12))
    for chan in range(num_iter):
        y.append(signal.resample(data[chan], resampling_factor))
        if plot_flag:
            plt.subplot(rows, columns, chan + 1)
            plt.plot(x_axis, y[chan])
            plt.xlabel('Time(sec)', fontsize=10)
            plt.ylabel('Voltage (mV)', fontsize=10)
            try:
                max_x_label_value = roundup(int(x_axis[-1]), 100)
            except:
                max_x_label_value = 600
            plt.xlim(0, max_x_label_value)
            plt.title(subtitle + " " + str(chan + 1), fontsize=11)
            plt.tight_layout(pad=6.0, w_pad=6.0, h_pad=6.0)
            plt.suptitle(main_title, fontsize=18)
    if plot_flag:
        plt.savefig(save_path + '/Resampling_10Hz' + id + '.png')
        plt.close('all')
    return y


def plot_trigger(trigger, action_idx, sample_rate, subject='', run='',
                 save_path=None, show=False):
    """
    Plot the trigger signal with detected start/end points.

    Parameters
    ----------
    trigger : array-like
        Trigger signal data
    action_idx : list
        [start_index, end_index] for the recording segment
    sample_rate : float
        Sampling rate of the trigger signal (Hz)
    subject : str, optional
        Subject identifier
    run : str, optional
        Run identifier
    save_path : str, optional
        Directory path to save the figure
    show : bool, optional
        Whether to display the plot interactively
    """
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(0, len(trigger) / sample_rate, 1 / sample_rate), trigger, c='b')
    plt.axvline(x=action_idx[0] / sample_rate, c='g', label='sample start')
    plt.axvline(x=action_idx[1] / sample_rate, c='r', label='sample end')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Trigger Signal')
    plt.title(f'Trigger Signal - {subject} Run {run}')
    plt.legend(loc='upper right')
    if save_path is not None:
        plt.savefig(save_path + '/trigger_cut_' + subject + '_' + run + '.png')
    if show:
        plt.show()
    plt.close('all')


def to_phase_resampled(signal_1d, origin_sr, target_sr):
    """
    Extract instantaneous phase and resample to target sampling rate.

    Parameters
    ----------
    signal_1d : array-like
        Input signal (1D array)
    origin_sr : float
        Original sampling rate (Hz)
    target_sr : float
        Target sampling rate (Hz)

    Returns
    -------
    phase_resampled : array
        Resampled instantaneous phase signal
    """
    import scipy as sp
    analytic_signal = signal.hilbert(signal_1d)
    phase = np.angle(analytic_signal)
    # Resample to target sampling rate
    resample_n_points = int((len(signal_1d) / origin_sr) * target_sr)
    return sp.signal.resample(phase, resample_n_points)
