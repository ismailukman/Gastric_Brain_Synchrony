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
├── VITD0107_Acq/
│   └── VITD0107_EGG.acq
├── VITD0126_Acq/
│   └── VITD0126_EGG.acq
├── VITD0128_Acq/
│   └── VITD0128_EGG.acq
└── ...
```

### Expected .acq File Contents

Each `.acq` file should contain:

- **Channels 0-3**: EGG electrode signals (4 channels)
- **Channel 4**: EDA signal (optional)
- **Channels 5-6**: Digital trigger signals (MRI trigger on channel 6, configurable in `config.py`)

---

## Metadata File Format

Create a metadata CSV file (`egg_metadata.csv`) with the following columns:

| Column                 | Description                   | Example Values                       |
| ---------------------- | ----------------------------- | ------------------------------------ |
| `subject`            | Subject identifier            | `sub-01`, `sub-02`               |
| `run`                | Run number                    | `1`, `2`                         |
| `mri_length`         | fMRI scan duration in seconds | `600`                              |
| `num_channles`       | Number of EGG channels        | `4`                                |
| `trigger_start`      | Trigger detection mode        | `auto` or seconds (e.g., `10.5`) |
| `dominant_channel`   | Which channel to use          | `auto` or channel index (0-3)      |
| `dominant_frequency` | Gastric frequency             | `auto` or Hz (e.g., `0.05`)      |

### Example Metadata File

```csv
subject,run,mri_length,num_channles,trigger_start,dominant_channel,dominant_frequency
VITD0107,1,600,4,0,auto,auto
VITD0126,1,547,4,0,auto,auto
VITD0128,1,600,4,45,auto,auto
```

The metadata file `egg_metadata.csv` is located in the `preprocess_egg_data/` folder.

---

## Usage

### Single Subject Processing

Process one subject/run at a time:

```bash
cd preprocess_egg_data

# Basic usage
python preprocess_gastric_data.py VITD0107 1

# This will:
# 1. Load EGG data from egg_data/VITD0107_Acq/VITD0107_EGG.acq
# 2. Process the signal
# 3. Save outputs to output/derivatives/VITD0107/VITD01071/
# 4. Save plots to output/plots/VITD0107/VITD01071/
```

**Example Output:**

```
============================================================
Processing subject: VITD0107, run: 1
============================================================
Reading EGG file: egg_data/VITD0107_Acq/VITD0107_EGG.acq
Original sample rate: 2000.0 Hz
MRI duration: 600 seconds
Number of EGG channels: 4
Dominant frequency: 0.0480 Hz
Dominant channel: 1

Output files saved:
  - Data: output/derivatives/VITD0107/VITD01071/gast_data_VITD0107_run1strict.npy
  - Frequency: output/derivatives/VITD0107/VITD01071/max_freqVITD0107_run1strict.npy
  - Plots: output/plots/VITD0107/VITD01071/
```

### Batch Processing

Process all subjects defined in the metadata file:

```bash
cd preprocess_egg_data

# Process all subjects (uses default metadata file from config)
python preprocess_gastric_data.py --batch

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
Metadata file: preprocess_egg_data/egg_metadata.csv
Parallel jobs: 8
Total subjects/runs to process: 3
============================================================

[SUCCESS] VITD0107 run 1
[SUCCESS] VITD0126 run 1
[SUCCESS] VITD0128 run 1

============================================================
BATCH PROCESSING COMPLETE
============================================================
Successful: 3/3
Failed: 0/3
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

Step 7: NORMALIZATION
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
    │   ├── VITD0107/
    │   │   └── VITD01071/
    │   │       ├── gast_data_VITD0107_run1strict.npy    # Filtered EGG signal
    │   │       └── max_freqVITD0107_run1strict.npy      # Dominant frequency
    │   ├── VITD0126/
    │   │   └── VITD01261/
    │   │       └── ...
    │   └── VITD0128/
    │       └── ...
    └── plots/
        ├── VITD0107/
        │   └── VITD01071/
        │       ├── trigger_cut_VITD0107_1.png              # Trigger detection
        │       ├── sliced_signalVITD0107_1.png             # Raw sliced signal
        │       ├── post_first_resample_VITD0107_1.png      # After resampling
        │       ├── egg_power_spectral_density_VITD0107_1.png  # PSD plot
        │       └── egg_filteredVITD0107_1.png              # Final filtered signal
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
egg_signal = np.load('output/derivatives/VITD0107/VITD01071/gast_data_VITD0107_run1strict.npy')
print(f"Signal shape: {egg_signal.shape}")  # e.g., (6000,) for 600s at 10Hz
print(f"Signal range: [{egg_signal.min():.2f}, {egg_signal.max():.2f}]")
```

**Dominant Frequency** (`max_freq*.npy`):

- Single float value in Hz
- Typically in range 0.033-0.066 Hz

```python
# Load dominant gastric frequency
freq = np.load('output/derivatives/VITD0107/VITD01071/max_freqVITD0107_run1strict.npy')
print(f"Dominant frequency: {freq:.4f} Hz ({freq*60:.2f} cycles/min)")
```
