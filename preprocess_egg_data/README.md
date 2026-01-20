# EGG (Electrogastrography) Preprocessing Pipeline

This module provides a complete pipeline for preprocessing EGG (electrogastrography) data recorded during fMRI sessions. The pipeline extracts and filters gastric slow-wave signals for subsequent synchrony analysis with brain/motion data.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Organization](#data-organization)
- [Metadata File Format](#metadata-file-format)
- [Usage](#usage)
  - [Single Subject Processing](#single-subject-processing)
  - [Batch Processing](#batch-processing)
- [Processing Pipeline](#processing-pipeline)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

Electrogastrography (EGG) records the electrical activity of the stomach using surface electrodes placed on the abdomen. The gastric slow wave typically oscillates at **0.033-0.066 Hz** (2-4 cycles per minute), known as the normogastric frequency range.

This preprocessing pipeline:
1. Reads raw EGG data from AcqKnowledge/Biopac format (.acq files)
2. Aligns the EGG signal with fMRI acquisition using trigger signals
3. Identifies the dominant gastric frequency for each subject
4. Applies narrow bandpass filtering around the dominant frequency
5. Outputs cleaned, filtered gastric signals ready for synchrony analysis

---

## Requirements

### Python Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
mne>=0.24.0
bioread>=3.0.0
scikit-learn>=0.24.0
```

### Installation

```bash
# Create a virtual environment (recommended)
python -m venv egg_env
source egg_env/bin/activate  # On Windows: egg_env\Scripts\activate

# Install dependencies
pip install numpy scipy pandas matplotlib mne bioread scikit-learn
```

---

## Data Organization

### Input Data Structure

Place your raw EGG data in the `egg_data/` folder following this structure:

```
egg_data/
├── sub-01/
│   └── egg/
│       ├── sub-01_rest1.acq
│       └── sub-01_rest2.acq
├── sub-02/
│   └── egg/
│       ├── sub-02_rest1.acq
│       └── sub-02_rest2.acq
└── ...
```

### Expected .acq File Contents

Each `.acq` file should contain:
- **Channels 0-3**: EGG electrode signals (typically 4 channels)
- **Channel 8**: MRI trigger signal (configurable in `config.py`)

---

## Metadata File Format

Create a metadata CSV file (`egg_metadata.csv`) with the following columns:

| Column | Description | Example Values |
|--------|-------------|----------------|
| `subject` | Subject identifier | `sub-01`, `sub-02` |
| `run` | Run number | `1`, `2` |
| `mri_length` | fMRI scan duration in seconds | `600` |
| `num_channles` | Number of EGG channels | `4` |
| `trigger_start` | Trigger detection mode | `auto` or seconds (e.g., `10.5`) |
| `dominant_channel` | Which channel to use | `auto` or channel index (0-3) |
| `dominant_frequency` | Gastric frequency | `auto` or Hz (e.g., `0.05`) |

### Example Metadata File

```csv
subject,run,mri_length,num_channles,trigger_start,dominant_channel,dominant_frequency
sub-01,1,600,4,auto,auto,auto
sub-01,2,600,4,auto,auto,auto
sub-02,1,600,4,auto,auto,auto
sub-02,2,600,4,auto,1,0.05
```

A template file (`egg_metadata_template.csv`) is provided.

---

## Usage

### Single Subject Processing

Process one subject/run at a time:

```bash
# Basic usage
python preprocess_gastric_data.py sub-01 1

# This will:
# 1. Load sub-01's run 1 EGG data from egg_data/sub-01/egg/sub-01_rest1.acq
# 2. Process the signal
# 3. Save outputs to output/derivatives/sub-01/sub-011/
# 4. Save plots to output/plots/sub-01/sub-011/
```

**Example Output:**

```
============================================================
Processing subject: sub-01, run: 1
============================================================
Reading EGG file: /path/to/egg_data/sub-01/egg/sub-01_rest1.acq
Original sample rate: 1000.0 Hz
MRI duration: 600 seconds
Number of EGG channels: 4
Channel #1: frequency=0.048, height=2.5e-06, prominences=1.2e-06, curvature=-3.4e-04
Channel #2: frequency=0.051, height=1.8e-06, prominences=8.5e-07, curvature=-2.1e-04
Dominant frequency: 0.0480 Hz
Dominant channel: 1

Output files saved:
  - Data: output/derivatives/sub-01/sub-011/gast_data_sub-01_run1strict.npy
  - Frequency: output/derivatives/sub-01/sub-011/max_freqsub-01_run1strict.npy
  - Plots: output/plots/sub-01/sub-011/
```

### Batch Processing

Process all subjects defined in the metadata file:

```bash
# Process all subjects (uses default metadata file from config)
python preprocess_gastric_data.py --batch

# Use a specific metadata file
python preprocess_gastric_data.py --batch --metadata /path/to/my_metadata.csv

# Specify number of parallel jobs
python preprocess_gastric_data.py --batch --jobs 4

# Sequential processing (1 job)
python preprocess_gastric_data.py --batch --jobs 1
```

**Example Batch Output:**

```
============================================================
BATCH PROCESSING MODE
============================================================
Metadata file: /path/to/egg_metadata.csv
Parallel jobs: 8
Total subjects/runs to process: 84
============================================================

[SUCCESS] sub-01 run 1
[SUCCESS] sub-01 run 2
[SUCCESS] sub-02 run 1
[FAILED] sub-03 run 1
...

============================================================
BATCH PROCESSING COMPLETE
============================================================
Successful: 82/84
Failed: 2/84

Failed subjects:
  - sub-03 run 1: File not found: .../sub-03_rest1.acq
  - sub-15 run 2: No peaks found in normogastric range
```

---

## Processing Pipeline

The preprocessing pipeline consists of the following steps:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EGG PREPROCESSING PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘

Step 1: DATA LOADING
        ├── Read .acq file (Biopac/AcqKnowledge format)
        ├── Extract EGG channels (typically 4)
        └── Extract trigger channel

Step 2: TRIGGER DETECTION
        ├── Detect MRI trigger onset (auto or manual)
        └── Define recording segment aligned with fMRI

Step 3: SIGNAL SLICING
        ├── Extract EGG segment matching fMRI duration
        └── Generate sliced signal plot

Step 4: RESAMPLING
        ├── Downsample from original rate (e.g., 1000 Hz)
        └── to intermediate rate (10 Hz)

Step 5: SPECTRAL ANALYSIS (Welch PSD)
        ├── Compute power spectral density per channel
        ├── Identify peaks in normogastric range (0.033-0.066 Hz)
        ├── Select dominant channel (highest power peak)
        └── Determine dominant gastric frequency

Step 6: BANDPASS FILTERING
        ├── Apply FIR filter (Hamming window)
        ├── Passband: [dominant_freq ± 0.015 Hz]
        └── Zero-phase filtering (no phase distortion)

Step 7: NORMALIZATION (optional)
        └── Z-score standardization

Step 8: OUTPUT
        ├── Save filtered signal (.npy)
        ├── Save dominant frequency (.npy)
        └── Save diagnostic plots (.png)
```

---

## Output Files

### Directory Structure

After processing, the output folder will contain:

```
preprocess_egg_data/
└── output/
    ├── derivatives/
    │   ├── sub-01/
    │   │   ├── sub-011/
    │   │   │   ├── gast_data_sub-01_run1strict.npy    # Filtered EGG signal
    │   │   │   └── max_freqsub-01_run1strict.npy     # Dominant frequency
    │   │   └── sub-012/
    │   │       ├── gast_data_sub-01_run2strict.npy
    │   │       └── max_freqsub-01_run2strict.npy
    │   └── sub-02/
    │       └── ...
    └── plots/
        ├── sub-01/
        │   ├── sub-011/
        │   │   ├── trigger_cut_sub-01_1.png          # Trigger detection
        │   │   ├── sliced_signalsub-01_1.png         # Raw sliced signal
        │   │   ├── post_first_resample_sub-01_1.png  # After resampling
        │   │   ├── egg_power_spectral_density_sub-01_1.png  # PSD plot
        │   │   └── egg_filteredsub-01_1.png          # Final filtered signal
        │   └── ...
        └── ...
```

### Output Data Format

**Filtered Signal** (`gast_data_*.npy`):
- 1D NumPy array
- Sampling rate: 10 Hz (configurable)
- Duration matches fMRI scan
- Z-score normalized (if enabled)

```python
import numpy as np

# Load preprocessed EGG signal
egg_signal = np.load('output/derivatives/sub-01/sub-011/gast_data_sub-01_run1strict.npy')
print(f"Signal shape: {egg_signal.shape}")  # e.g., (6000,) for 600s at 10Hz
print(f"Signal range: [{egg_signal.min():.2f}, {egg_signal.max():.2f}]")
```

**Dominant Frequency** (`max_freq*.npy`):
- Single float value in Hz
- Typically in range 0.033-0.066 Hz

```python
# Load dominant gastric frequency
freq = np.load('output/derivatives/sub-01/sub-011/max_freqsub-01_run1strict.npy')
print(f"Dominant frequency: {freq:.4f} Hz ({freq*60:.2f} cycles/min)")
```

---

## Configuration

Edit `config.py` to customize the preprocessing parameters:

```python
# Sampling rates
intermediate_sample_rate = 10  # Hz (after downsampling)

# Trigger channel
trigger_channel = 8  # Channel containing MRI trigger

# Bandpass filter parameters
filter_order = 5
bandpass_lim = 0.015  # Hz (passband = dominant_freq ± this value)
transition_width = 0.15  # Fraction of cutoff frequency

# Spectral analysis (Welch method)
window = 200  # seconds
overlap = 100  # seconds
freq_range = [0.033, 0.066]  # Normogastric range (Hz)

# Processing options
zscore_flag = True  # Apply z-score normalization
clean_level = 'strict'  # Output filename suffix

# Batch processing
multi_thread = True
num_threads = 8
```

---

## Troubleshooting

### Common Issues

**1. "No metadata found for subject"**
- Ensure your metadata CSV contains the correct subject ID and run number
- Check for typos in subject names (case-sensitive)

**2. "File not found: *.acq"**
- Verify the data folder structure matches the expected format
- Check that `egg_data_path` in `config.py` points to the correct location

**3. "No peaks found in normogastric range"**
- The subject may not have clear gastric activity
- Try manually specifying `dominant_frequency` in the metadata
- Check the PSD plot to identify peaks outside the normal range

**4. "Signal started with trigger > 0"**
- Warning only; the pipeline handles this automatically
- If alignment is incorrect, manually specify `trigger_start` in metadata

**5. Import errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)

### Diagnostic Plots

Review the generated plots to verify preprocessing quality:

1. **trigger_cut_*.png**: Verify correct MRI trigger detection
2. **sliced_signal*.png**: Check raw signal quality
3. **egg_power_spectral_density_*.png**: Verify dominant frequency selection
4. **egg_filtered*.png**: Inspect final filtered signal

---

## License

MIT License

## Author

Ismail Ukman
