# Gastric-Brain-Motion Synchrony Analysis

**Investigating the coupling between gastric electrical rhythm and head motion during resting-state fMRI**

---

## Overview

This project quantifies phase synchronization between the gastric slow-wave rhythm (measured via electrogastrography, EGG) and head motion parameters during resting-state fMRI. Using dual metrics—Phase Locking Value (PLV) and Amplitude-Weighted PLV (awPLV)—we demonstrate that gastric rhythm significantly modulates head motion in multiple dimensions, providing evidence for gut-brain-motor integration.

**Key Finding**: Gastric rhythm significantly couples with head motion in anterior-posterior, lateral, pitch, and roll dimensions, but NOT superior-inferior motion.

---

## Dataset

- **Participants**: 43 healthy adults
- **Final Analysis**: 84 runs after quality control
- **Acquisition**: Simultaneous resting-state fMRI (TR=2s, ~6 min) and EGG recording
- **Motion Parameters**: 6 degrees of freedom
  - Translations: X (left-right), Y (anterior-posterior), Z (superior-inferior)
  - Rotations: X (pitch), Y (roll), Z (yaw)

---

## Analysis Pipeline

### 1. Preprocessing
- **fMRI**: Motion correction, normalization (MNI), smoothing (6mm FWHM) via AFNI/fMRIprep
- **EGG**: Cardiac artifact removal, bandpass filtering (0.02-0.08 Hz), resampled to 10 Hz
- **Motion**: Extracted from fMRI realignment, bandpass filtered at subject-specific gastric frequency ± 0.015 Hz

### 2. Synchrony Metrics
**Phase Locking Value (PLV):**
```
PLV = |1/T Σ exp(iΔφ(t))|
```
Measures pure phase synchronization (range: 0-1)

**Amplitude-Weighted PLV (awPLV):**
```
awPLV = |1/T Σ w(t)·exp(iΔφ(t))|
where w(t) = A_gastric(t)·A_motion(t) / Σ(A_gastric·A_motion)
```
Emphasizes high-amplitude coupling periods (range: 0-1)

### 3. Statistical Testing
- **Null Distribution**: Mismatch approach (N-1 pairings with gastric signals from other subjects)
- **Individual Level**: 504 tests (84 runs × 6 motion params) per metric
  - FDR correction (Benjamini-Hochberg, q<0.05)
  - Bonferroni correction
- **Population Level**: Mann-Whitney U test (empirical vs pooled null)
  - Separate FDR corrections for PLV and awPLV (6 tests each)

---

## Key Results

### Population-Level Significance (FDR-corrected, q<0.05)

| Motion Parameter | PLV | awPLV | Effect Size (PLV) | Effect Size (awPLV) |
|------------------|-----|-------|-------------------|---------------------|
| **Translation Y** (Ant-Post) | ✓ *** | ✓ *** | 0.042 | 0.059 |
| **Translation X** (Left-Right) | ✓ * | ✓ ** | 0.021 | 0.044 |
| **Rotation X** (Pitch) | ✓ ** | ✓ ** | 0.030 | 0.049 |
| **Rotation Y** (Roll) | ✓ ** | ✓ * | 0.033 | 0.043 |
| **Rotation Z** (Yaw) | ✗ | ✓ * | 0.020 | 0.045 |
| **Translation Z** (Sup-Inf) | ✗ | ✗ | -0.007 | 0.007 |

**Significance levels**: * q<0.05, ** q<0.01, *** q<0.001

### Individual-Level Significance
- **PLV**: 22/504 tests (4.4%) FDR-significant
- **awPLV**: 30/504 tests (6.0%) FDR-significant

### Interpretation
- **Strongest coupling**: Anterior-posterior translation (Translation Y)
- **Robust coupling**: Lateral translation, pitch, and roll
- **No coupling**: Superior-inferior translation
- **awPLV advantage**: Larger effect sizes, suggesting coupling is prominent during high-amplitude gastric oscillations

---

## Repository Structure

```
main_project_path/
├── code/
│   ├── synchrony_analysis/
│   │   ├── egg_confounds_synchrony_v5.py    # Main analysis script (PLV + awPLV)
│   │   ├── egg_confounds_synchrony_v6.py    # Latest version (if exists)
│   │   ├── signal_slicing.py                # Signal preprocessing utilities
│   │   └── voxel_based_analysis.py          # Voxel-wise brain analysis
│   ├── plot_subject_signals.py              # Comprehensive subject visualization (9 panels)
│   ├── plot_gastric_transx.py               # Simple gastric + Translation X plot
│   ├── regenerate_plot_v5.py                # Regenerate main density plots
│   ├── create_method_figure_v2.py           # Enhanced method schematic
│   ├── analysis_pipeline_flowchart.txt      # Detailed pipeline documentation
│   ├── analysis_pipeline_simplified.txt     # Simplified pipeline overview
│   └── dataframes/
│       ├── plvs_egg_w_motion_v5.csv         # Individual-level results (504 rows)
│       ├── population_level_v5.csv          # Population-level results (12 rows)
│       └── egg_brain_meta_data.csv          # Subject metadata
├── plots/
│   ├── egg_confounds_synchrony_v5.png       # Main results figure (3×2 grid)
│   ├── method_Figure1_v2.pdf/.png           # Enhanced methods schematic
│   ├── gastric_rhythm_subject_*.png         # Individual subject gastric plots
│   └── filtered_motion_subject_*.png        # Individual subject motion plots
├── derivatives/brain_gast/                  # Preprocessed EGG data
├── BIDS_data/sub_motion_files/              # Motion parameter files
├── config.py                                 # Configuration parameters
├── experiment_methods_results.txt            # Detailed methods & results text
└── README.md                                 # This file
```

---

## Key Scripts

### Main Analysis
- **`egg_confounds_synchrony_v5.py`**: Complete PLV/awPLV analysis pipeline
  - Loads EGG and motion data
  - Computes synchrony metrics with mismatch null distribution
  - Performs individual and population-level statistical testing
  - Generates density plot visualization

### Visualization
- **`plot_subject_signals.py`**: Individual subject detailed plots (gastric rhythm + 6 motion parameters)
- **`plot_gastric_transx.py`**: Simple 2-panel plot (gastric + Translation X)
- **`create_method_figure_v2.py`**: Professional methods schematic flowchart
- **`regenerate_plot_v5.py`**: Regenerate main results figure with updated formatting

### Documentation
- **`analysis_pipeline_flowchart.txt`**: Detailed step-by-step pipeline (8 steps)
- **`analysis_pipeline_simplified.txt`**: High-level overview (7 steps)
- **`experiment_methods_results.txt`**: Publication-ready methods and results text

---

## Running the Analysis

### Prerequisites
```bash
# Python 3.11+
pip install numpy scipy pandas matplotlib seaborn mne-python
```

### Main Analysis
```bash
cd code
python synchrony_analysis/egg_confounds_synchrony_v5.py
```

**Outputs:**
- `dataframes/plvs_egg_w_motion_v5.csv` - Individual-level results
- `dataframes/population_level_v5.csv` - Population-level statistics
- `plots/egg_confounds_synchrony_v5.png` - Main results visualization

### Subject-Specific Visualization
```bash
# Detailed 9-panel figure
python plot_subject_signals.py

# Simple 2-panel figure (gastric + Translation X)
python plot_gastric_transx.py
```

Edit `SUBJECT` and `RUN` variables in scripts to change subject.

### Regenerate Main Figure
```bash
python regenerate_plot_v5.py
```

---

## Configuration

Edit `config.py` to modify analysis parameters:
- `sample_rate_fmri`: fMRI sampling rate (default: 0.5 Hz)
- `intermediate_sample_rate`: EGG resampling rate (default: 10 Hz)
- `bandpass_lim`: Frequency bandwidth for filtering (default: 0.015 Hz)
- `clean_level`: EGG cleaning method (default: 'strict_gs_cardiac')

---

## Output Files

### Data Files
- **`plvs_egg_w_motion_v5.csv`** (504 rows): Individual-level results
  - Columns: subject, run, motion_param, plv_empirical, awplv_empirical, p-values, FDR-corrected p-values, significance flags

- **`population_level_v5.csv`** (12 rows): Population-level statistics
  - Columns: motion_param, metric, n_empirical, n_null, mean_empirical, mean_null, effect_size, p_fdr, sig_fdr

### Figures
- **Main Results**: 3×2 grid showing PLV and awPLV density distributions for all 6 motion parameters
  - Empirical (filled) vs Null (dashed) distributions
  - Significance markers: * (q<0.05), ** (q<0.01), *** (q<0.001)
  - Font sizes optimized for publication (2× standard)

- **Methods Schematic**: Professional flowchart showing complete analysis pipeline
  - Modern color palette (light cyan, steel blue)
  - Mathematical formulas included
  - Publication-ready at 300 DPI

---

## Biological Interpretation

The observed gastric-motion phase-locking provides evidence that:

1. **Gut-Brain-Motor Integration**: Gastric rhythm modulates postural control during rest
2. **Directional Specificity**:
   - Strong coupling in anterior-posterior, lateral, pitch, and roll
   - No coupling in superior-inferior (potential biomechanical constraint)
3. **Amplitude Dependence**: awPLV shows stronger effects, suggesting coupling is particularly prominent during high-amplitude gastric oscillations
4. **Interoceptive Pathway**: Physiologically meaningful connection between visceral state and motor control

### Implications for fMRI Preprocessing
Conventional motion regression may inadvertently remove physiologically meaningful interoceptive signals. Researchers should consider gastric-band motion fluctuations as potential physiological confounds rather than noise.

---

## Citation

If you use this code or analysis approach, please cite:

```
[Your publication details here]
```

---

## Software Dependencies

- **Python**: 3.11+
- **NumPy**: Array operations and numerical computing
- **SciPy**: Statistical testing (Mann-Whitney U, FDR correction)
- **Pandas**: Data manipulation and CSV handling
- **Matplotlib**: Visualization and figure generation
- **Seaborn**: Kernel density estimation and enhanced plotting
- **MNE-Python**: Signal filtering (FIR, Hamming window)

---

## License

[Add license information]

---

## Contact

[Add contact information]

---

## Changelog

### Version 5 (Current)
- ✓ Added amplitude-weighted PLV (awPLV) analysis
- ✓ Implemented mismatch null distribution approach
- ✓ Separate FDR corrections for PLV and awPLV
- ✓ Enhanced visualization with significance markers
- ✓ Doubled font sizes for publication quality
- ✓ Created comprehensive documentation and pipeline flowcharts
- ✓ Added subject-specific visualization scripts
- ✓ Enhanced methods schematic figure (v2)

### Version 4
- Circular permutation null distribution
- Single-metric PLV analysis
- Basic visualization

---

## Acknowledgments

- Data preprocessing: fMRIprep, AFNI
- Signal processing: MNE-Python
- Statistical methods: SciPy (Benjamini-Hochberg FDR)
- Visualization: Matplotlib, Seaborn

---

**Last Updated**: December 2024
