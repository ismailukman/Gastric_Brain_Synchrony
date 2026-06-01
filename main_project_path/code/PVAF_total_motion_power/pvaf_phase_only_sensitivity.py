#!/usr/bin/env python3
"""
Sensitivity check: PVAF with a phase-only design matrix.

The headline analysis (pvaf_total_motion.py) uses

    X(t) = [ g(t) , H[g](t) ]   ==   [ A(t) cos(phi(t)) , A(t) sin(phi(t)) ]

which captures both phase locking AND amplitude modulation of the
in-band gastric rhythm. As a sensitivity check we rerun the same
pipeline with

    X(t) = [ cos(phi(t)) , sin(phi(t)) ]

which captures pure phase locking only. This is the closest match to
the PLV statistic used in the OHBM analysis.

All outputs are written to outputs/v_phase_only/ so the headline
outputs in outputs/ are untouched.

Usage
-----
    conda activate brain_gut
    python pvaf_phase_only_sensitivity.py
"""

import sys
import pathlib

PARENT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(PARENT))

import pvaf_total_motion as pvm
import numpy as np
from scipy.signal import hilbert


def build_gastric_regressors_phase_only(gastric_bp):
    """
    Phase-only design matrix:

        X(t) = [ cos(phi(t)) , sin(phi(t)) ]

    Both columns are pure sinusoids at the gastric frequency. They
    recover any preferred phase (Fisher 1993) but do not scale with
    the gastric envelope A(t). This is the cleanest "pure phase
    locking" model and is the analogue of the OHBM PLV statistic.
    """
    z = hilbert(gastric_bp)
    phi = np.angle(z)
    X = np.column_stack([np.cos(phi), np.sin(phi)])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)
    return X


pvm.build_gastric_regressors = build_gastric_regressors_phase_only

OUT = PARENT / "outputs" / "v_phase_only"
OUT.mkdir(parents=True, exist_ok=True)
pvm.OUTPUT_DIR         = OUT
pvm.OUTPUT_PER_RUN     = OUT / "pvaf_per_run.csv"
pvm.OUTPUT_GROUP_LEVEL = OUT / "pvaf_group_level.csv"
pvm.OUTPUT_DECOMP      = OUT / "pvaf_decomposition.csv"
pvm.OUTPUT_FIGURE      = OUT / "pvaf_figure.png"
pvm.OUTPUT_DOC         = OUT / "pvaf_documentation.txt"
pvm.OUTPUT_LOG         = OUT / "pvaf_session_log.txt"


if __name__ == "__main__":
    pvm.main()
