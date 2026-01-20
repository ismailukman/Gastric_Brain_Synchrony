"""
Utility modules for EGG preprocessing.

- gastric_utils: Signal visualization and processing functions
- spect_utils: Spectral analysis functions (Welch PSD, peak detection)
"""

from .gastric_utils import plot_signal, plot_trigger, to_phase_resampled
from .spect_utils import powerspect
