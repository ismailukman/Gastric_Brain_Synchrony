"""
Configuration file for EGG (Electrogastrography) preprocessing pipeline.

This file contains all adjustable parameters for the preprocessing workflow.
Modify these values according to your experimental setup and data characteristics.
"""

import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base path where raw EGG data is stored
# Structure expected: {egg_data_path}/{subject}/egg/{subject}_rest{run}.acq
egg_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'egg_data')

# Output paths for processed data and plots
output_path = os.path.join(os.path.dirname(__file__), 'output')
plots_path = os.path.join(output_path, 'plots')
derivatives_path = os.path.join(output_path, 'derivatives')

# Path to metadata CSV file
metadata_file = os.path.join(os.path.dirname(__file__), 'egg_metadata.csv')

# =============================================================================
# SAMPLING RATE CONFIGURATION
# =============================================================================

# Intermediate sampling rate after initial downsampling (Hz)
# EGG signals are typically recorded at high rates (e.g., 1000 Hz)
# and downsampled to this rate for processing
intermediate_sample_rate = 10

# =============================================================================
# TRIGGER CONFIGURATION
# =============================================================================

# Channel number containing the MRI trigger signal
# This is used to align EGG data with fMRI acquisition
trigger_channel = 8

# =============================================================================
# FILTERING PARAMETERS
# =============================================================================

# Filter order for the bandpass filter
filter_order = 5

# Bandwidth around the dominant gastric frequency (Hz)
# The bandpass filter will be: [dominant_freq - bandpass_lim, dominant_freq + bandpass_lim]
bandpass_lim = 0.015

# Transition width as a fraction of the filter cutoff frequency
# Controls the sharpness of the filter rolloff
transition_width = 15 / 100  # 15%

# =============================================================================
# SPECTRAL ANALYSIS PARAMETERS (Welch method)
# =============================================================================

# Window size for Welch power spectral density estimation (seconds)
window = 200

# Overlap between consecutive windows (seconds)
overlap = 100

# Frequency range of interest for gastric activity (Hz)
# Normal gastric slow wave frequency is typically 0.033-0.066 Hz (2-4 cycles per minute)
freq_range = [0.033, 0.066]

# =============================================================================
# SIGNAL PROCESSING OPTIONS
# =============================================================================

# Whether to apply z-score normalization to the filtered signal
zscore_flag = True

# Cleaning level identifier (used in output filenames)
clean_level = 'strict'

# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================

# Enable multiprocessing for batch processing
multi_thread = True

# Number of parallel threads for batch processing
num_threads = 8
