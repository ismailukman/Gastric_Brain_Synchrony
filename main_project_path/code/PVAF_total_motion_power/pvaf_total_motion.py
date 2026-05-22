#!/usr/bin/env python3
"""
PVAF (Percent Variance Accounted For) of total head-motion power.

Usage
-----
    conda activate brain_gut
    python pvaf_total_motion.py

Outputs
-------
    outputs/pvaf_per_run.csv         per-(subject, run, axis) PVAF on 3 denominators
    outputs/pvaf_group_level.csv     median, IQR, Wilcoxon p, FDR
    outputs/pvaf_decomposition.csv   band-power fraction and in-band PVAF per axis
    outputs/pvaf_figure.png          summary figure
    outputs/pvaf_session_log.txt     run log
"""

import os
import sys
import pathlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample, welch, coherence, hilbert
from scipy.stats import wilcoxon, false_discovery_control
import statsmodels.api as sm
from mne.filter import filter_data

warnings.filterwarnings("ignore")

##############################################################################
# Configuration                                                              #
##############################################################################

PARENT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = PARENT_DIR.parent          # .../main_project_path/code
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from config import (main_project_path, clean_level, sample_rate_fmri,
                    intermediate_sample_rate, bandpass_lim, filter_order,
                    transition_width, freq_range)

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

OUTPUT_DIR = PARENT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PER_RUN     = OUTPUT_DIR / "pvaf_per_run.csv"
OUTPUT_GROUP_LEVEL = OUTPUT_DIR / "pvaf_group_level.csv"
OUTPUT_DECOMP      = OUTPUT_DIR / "pvaf_decomposition.csv"
OUTPUT_FIGURE      = OUTPUT_DIR / "pvaf_figure.png"
OUTPUT_LOG         = OUTPUT_DIR / "pvaf_session_log.txt"

# Column labels are kept identical to the existing pipeline. If the AFNI
# 1D file is actually in [roll, pitch, yaw, dS, dL, dP] order rather than
# the assumed [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z], the
# headline PVAF (averaged across all six axes) is unaffected; only the
# per-axis labels in the figure would need re-permuting. See README.md.
MOTION_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

RSFMRI_BAND = (0.01, 0.10)                 # Conventional rsfMRI band (Hz)
GASTRIC_BAND_NORMOGASTRIC = freq_range     # (0.033, 0.066) from config

SAMPLE_RATE_FMRI = sample_rate_fmri        # 0.5 Hz (TR = 2 s)
EGG_INTERMEDIATE_SFREQ = intermediate_sample_rate

# How many mismatched subjects to use per axis for the null distribution.
# Setting MAX_NULL=None uses ALL other subjects (matches OHBM null exactly
# but is O(N_runs * N_subjects * N_axes); ~84*42*6 ~ 21000 regressions).
MAX_NULL = None

##############################################################################
# Helpers                                                                    #
##############################################################################


def bp_filter_1d(x, sfreq, l_freq, h_freq, filt_order=filter_order,
                 trans_width=transition_width):
    """
    Symmetric (zero-phase) FIR bandpass filter using MNE.

    Identical filter family to the rest of the pipeline (Hamming window,
    firwin2, zero-double phase). Returns a 1-D array of the same length.
    """
    filter_length = int(filt_order * np.floor(sfreq / l_freq))
    out = filter_data(
        data=np.asarray(x, dtype=float).reshape(1, -1),
        sfreq=sfreq,
        l_freq=l_freq, h_freq=h_freq,
        filter_length=filter_length,
        l_trans_bandwidth=trans_width * l_freq,
        h_trans_bandwidth=trans_width * h_freq,
        n_jobs=1, method="fir", phase="zero-double",
        fir_window="hamming", fir_design="firwin2", verbose=False,
    )
    return out.flatten()


def build_gastric_regressors(gastric_bp, include_amp=True):
    """
    Build the design matrix from the bandpassed gastric signal.

        gastric_bp(t) = real part
        H[gastric_bp](t) = imaginary part of analytic signal
        phi(t) = arg(z), A(t) = |z|

    Returns columns:  cos(phi), sin(phi), A    (3 predictors; intercept added later)
    With include_amp=False returns only cos(phi), sin(phi)  (phase-only model).

    Standardising each predictor to unit variance keeps the OLS coefficients
    interpretable but does NOT change R^2.
    """
    z = hilbert(gastric_bp)
    phi = np.angle(z)
    A = np.abs(z)
    cols = [np.cos(phi), np.sin(phi)]
    if include_amp:
        cols.append(A)
    X = np.column_stack(cols)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)
    return X


def regression_ss(y, X):
    """
    Plain OLS fit. Returns (ss_total, ss_explained, beta).

        ss_total     = sum((y - mean(y))^2)
        ss_explained = ss_total - sum((y - y_hat)^2)
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    y_mean = y.mean()
    ss_total = float(np.sum((y - y_mean) ** 2))
    ss_resid = float(np.sum(model.resid ** 2))
    ss_explained = ss_total - ss_resid
    return ss_total, ss_explained, model.params


def band_power(x, sfreq, band, nperseg=None):
    """
    Variance (= total time-domain power) attributable to a frequency band,
    computed via Welch PSD. Uses 'spectrum' scaling so that
    sum(PSD) ~ var(signal).
    """
    if nperseg is None:
        nperseg = min(64, len(x) // 4)
        if nperseg < 16:
            nperseg = max(8, len(x) // 2)
    f, p = welch(x, fs=sfreq, nperseg=nperseg, scaling="spectrum")
    in_band = (f >= band[0]) & (f <= band[1])
    return float(np.sum(p[in_band])), float(np.sum(p)), f, p


def fd_power_2012(motion_df, radius_mm=50.0):
    """
    Framewise displacement (Power et al. 2012, eq. 2):

        FD(t) = |dx| + |dy| + |dz|
              + r * (|d_rotx| + |d_roty| + |d_rotz|)

    with rotations in RADIANS converted to mm via a 50 mm sphere. AFNI
    1D motion files report rotations in DEGREES, so we convert here.
    """
    m = motion_df.copy()
    for col in ["rot_x", "rot_y", "rot_z"]:
        m[col] = np.deg2rad(m[col]) * radius_mm
    diffs = m.diff().fillna(0).abs()
    return diffs.sum(axis=1).values


def load_all_subject_data():
    """Load and time-align gastric + motion for every available subject-run."""
    meta = pd.read_csv(META_DATAFRAME_PATH)
    if clean_level == "strict_gs_cardiac":
        meta = meta.loc[(meta["ppu_exclude"] == False) &
                        (meta["ppu_found"] == True)]
    sub_runs = list(zip(meta["subject"], meta["run"]))

    all_data = {}
    for (sub, run) in sub_runs:
        try:
            motion_path = MOTION_FILE_TEMPLATE.format(sub=sub, run=run)
            egg_path = EGG_FILE_TEMPLATE.format(
                sub=sub, run=run, clean_level=clean_level)
            freq_path = GASTRIC_FREQ_TEMPLATE.format(
                sub=sub, run=run, clean_level=clean_level)
            if not all(os.path.isfile(p) for p in (motion_path, egg_path, freq_path)):
                continue

            motion = np.loadtxt(motion_path)
            df_motion = pd.DataFrame(motion, columns=MOTION_COLS)

            gastric_egg = np.load(egg_path)                # already bp'd at peak, 10 Hz
            gastric_peak = float(np.load(freq_path).flatten()[0])

            n_fmri = int((len(gastric_egg) / EGG_INTERMEDIATE_SFREQ) * SAMPLE_RATE_FMRI)
            if n_fmri < 60:
                continue
            gastric_resampled = resample(gastric_egg, n_fmri)
            n = min(len(gastric_resampled), len(df_motion))
            gastric_resampled = gastric_resampled[:n]
            df_motion = df_motion.iloc[:n].reset_index(drop=True)

            all_data[(sub, run)] = {
                "subject": sub,
                "run": run,
                "gastric_peak": gastric_peak,
                "gastric_bp": gastric_resampled,           # already at peak +- 0.015 Hz
                "motion_raw": df_motion,
                "n": n,
            }
        except Exception as exc:
            print(f"  [load] {sub} run {run}: {exc}")
    return all_data


##############################################################################
# Per-run PVAF computation                                                   #
##############################################################################


def pvaf_for_run(data, all_data):
    """
    Compute the three PVAF measures (total / rsfMRI / gastric-band) for each
    of the 6 motion axes, plus framewise displacement, plus a per-run null
    distribution via mismatched-subject regressors.

    Returns one record per (axis OR FD-summary) for this subject-run.
    """
    sub, run, n = data["subject"], data["run"], data["n"]
    gastric_peak = data["gastric_peak"]
    motion_df = data["motion_raw"]

    # ---- Gastric regressors (empirical and from each other subject) ----
    X_emp = build_gastric_regressors(data["gastric_bp"], include_amp=True)
    other_subjs = [k for k in all_data
                   if all_data[k]["subject"] != sub
                   and all_data[k]["n"] >= n]
    if MAX_NULL is not None and len(other_subjs) > MAX_NULL:
        rng = np.random.default_rng(seed=hash((sub, run)) % (2**32))
        other_subjs = rng.choice(other_subjs, size=MAX_NULL, replace=False).tolist()
    X_null_list = [
        build_gastric_regressors(all_data[k]["gastric_bp"][:n], include_amp=True)
        for k in other_subjs
    ]

    # ---- Filter motion at the rsfMRI and gastric bands once ----
    motion_rsfMRI = pd.DataFrame(
        {c: bp_filter_1d(motion_df[c].values, SAMPLE_RATE_FMRI,
                         RSFMRI_BAND[0], RSFMRI_BAND[1])
         for c in MOTION_COLS}
    )
    l_g, h_g = gastric_peak - bandpass_lim, gastric_peak + bandpass_lim
    motion_gastric = pd.DataFrame(
        {c: bp_filter_1d(motion_df[c].values, SAMPLE_RATE_FMRI, l_g, h_g)
         for c in MOTION_COLS}
    )

    rows = []
    for axis in MOTION_COLS:
        y_raw   = motion_df[axis].values
        y_rsf   = motion_rsfMRI[axis].values
        y_gast  = motion_gastric[axis].values

        # SS_explained is computed against the RAW signal: this is the
        # variance attributable to the gastric regressors over the whole
        # broadband signal. The three PVAFs share this same numerator.
        ss_total_raw, ss_expl_raw, beta = regression_ss(y_raw, X_emp)
        ss_total_rsf,  _, _ = regression_ss(y_rsf, X_emp)
        ss_total_gast, _, _ = regression_ss(y_gast, X_emp)

        pvaf_total   = ss_expl_raw / ss_total_raw  if ss_total_raw  > 0 else np.nan
        pvaf_rsfMRI  = ss_expl_raw / ss_total_rsf  if ss_total_rsf  > 0 else np.nan
        pvaf_gastric = ss_expl_raw / ss_total_gast if ss_total_gast > 0 else np.nan

        # Spectral decomposition: fraction of motion power in each band.
        bp_rsf,   total_p, _, _ = band_power(y_raw, SAMPLE_RATE_FMRI, RSFMRI_BAND)
        bp_gast,  _,       _, _ = band_power(y_raw, SAMPLE_RATE_FMRI, (l_g, h_g))
        frac_in_rsfMRI  = bp_rsf  / total_p if total_p > 0 else np.nan
        frac_in_gastric = bp_gast / total_p if total_p > 0 else np.nan

        # Null distribution: same numerator with mismatched gastric.
        pvaf_total_null = np.array([
            (regression_ss(y_raw, Xn)[1] / ss_total_raw) if ss_total_raw > 0 else np.nan
            for Xn in X_null_list
        ])

        rows.append({
            "subject": sub, "run": run, "axis": axis,
            "gastric_peak_Hz": gastric_peak, "n_timepoints": n,
            "ss_total_raw": ss_total_raw,
            "ss_total_rsfMRI": ss_total_rsf,
            "ss_total_gastric_band": ss_total_gast,
            "ss_explained_by_gastric": ss_expl_raw,
            "pvaf_total_pct": 100 * pvaf_total,
            "pvaf_rsfMRI_pct": 100 * pvaf_rsfMRI,
            "pvaf_gastric_band_pct": 100 * pvaf_gastric,
            "motion_power_in_rsfMRI_band_frac": frac_in_rsfMRI,
            "motion_power_in_gastric_band_frac": frac_in_gastric,
            "pvaf_total_null_median_pct": 100 * np.nanmedian(pvaf_total_null),
            "pvaf_total_null_mean_pct":   100 * np.nanmean(pvaf_total_null),
            "pvaf_total_null_p95_pct":    100 * np.nanpercentile(pvaf_total_null, 95),
            "pvaf_total_excess_pct":      100 * (pvaf_total - np.nanmedian(pvaf_total_null)),
            "n_null":                     int(np.sum(~np.isnan(pvaf_total_null))),
        })

    # ---- Framewise displacement: one PVAF per run ----
    fd = fd_power_2012(motion_df)
    ss_total_raw, ss_expl_raw, _ = regression_ss(fd, X_emp)
    pvaf_fd_total = ss_expl_raw / ss_total_raw if ss_total_raw > 0 else np.nan
    pvaf_fd_null = np.array([
        (regression_ss(fd, Xn)[1] / ss_total_raw) if ss_total_raw > 0 else np.nan
        for Xn in X_null_list
    ])
    rows.append({
        "subject": sub, "run": run, "axis": "FD_Power2012",
        "gastric_peak_Hz": gastric_peak, "n_timepoints": n,
        "ss_total_raw": ss_total_raw,
        "ss_total_rsfMRI": np.nan,
        "ss_total_gastric_band": np.nan,
        "ss_explained_by_gastric": ss_expl_raw,
        "pvaf_total_pct": 100 * pvaf_fd_total,
        "pvaf_rsfMRI_pct": np.nan,
        "pvaf_gastric_band_pct": np.nan,
        "motion_power_in_rsfMRI_band_frac": np.nan,
        "motion_power_in_gastric_band_frac": np.nan,
        "pvaf_total_null_median_pct": 100 * np.nanmedian(pvaf_fd_null),
        "pvaf_total_null_mean_pct":   100 * np.nanmean(pvaf_fd_null),
        "pvaf_total_null_p95_pct":    100 * np.nanpercentile(pvaf_fd_null, 95),
        "pvaf_total_excess_pct":      100 * (pvaf_fd_total - np.nanmedian(pvaf_fd_null)),
        "n_null":                     int(np.sum(~np.isnan(pvaf_fd_null))),
    })

    return rows


##############################################################################
# Group-level statistics                                                     #
##############################################################################


def group_level(per_run_df):
    """
    Wilcoxon signed-rank empirical-vs-null-median, FDR across the 7 measures
    (6 motion axes + FD).
    """
    out = []
    axes = list(MOTION_COLS) + ["FD_Power2012"]
    for axis in axes:
        sub = per_run_df.query("axis == @axis")
        emp = sub["pvaf_total_pct"].values
        nul = sub["pvaf_total_null_median_pct"].values
        excess = emp - nul
        try:
            stat, p = wilcoxon(emp, nul, alternative="greater")
        except ValueError:
            stat, p = np.nan, np.nan
        out.append({
            "axis": axis,
            "n_runs": len(emp),
            "pvaf_total_median_pct": np.median(emp),
            "pvaf_total_iqr_pct":    np.percentile(emp, 75) - np.percentile(emp, 25),
            "pvaf_total_null_median_pct": np.median(nul),
            "pvaf_total_excess_median_pct": np.median(excess),
            "pvaf_rsfMRI_median_pct":     np.nanmedian(sub["pvaf_rsfMRI_pct"]),
            "pvaf_gastric_band_median_pct": np.nanmedian(sub["pvaf_gastric_band_pct"]),
            "motion_power_in_gastric_band_pct": 100 * np.nanmedian(sub["motion_power_in_gastric_band_frac"]),
            "motion_power_in_rsfMRI_band_pct":  100 * np.nanmedian(sub["motion_power_in_rsfMRI_band_frac"]),
            "wilcoxon_stat": stat,
            "wilcoxon_p_one_sided": p,
        })
    df = pd.DataFrame(out)
    df["wilcoxon_p_fdr"] = false_discovery_control(df["wilcoxon_p_one_sided"].values, method="bh")
    df["sig_fdr"] = df["wilcoxon_p_fdr"] < 0.05
    return df


##############################################################################
# Plotting                                                                   #
##############################################################################


def plot_pvaf(group_df, per_run_df, out_path):
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30, top=0.93, bottom=0.08)

    axes_order = list(MOTION_COLS) + ["FD_Power2012"]
    pretty = {"trans_x": "Trans X", "trans_y": "Trans Y", "trans_z": "Trans Z",
              "rot_x": "Rot X", "rot_y": "Rot Y", "rot_z": "Rot Z",
              "FD_Power2012": "FD"}
    x = np.arange(len(axes_order))

    # Panel A: PVAF on the three denominators
    ax = fig.add_subplot(gs[0, 0])
    pvaf_t = [group_df.set_index("axis").loc[a, "pvaf_total_median_pct"] for a in axes_order]
    pvaf_r = [group_df.set_index("axis").loc[a, "pvaf_rsfMRI_median_pct"] for a in axes_order]
    pvaf_g = [group_df.set_index("axis").loc[a, "pvaf_gastric_band_median_pct"] for a in axes_order]
    w = 0.27
    ax.bar(x - w, pvaf_t, w, color="#2c3e50", label="vs broadband motion")
    ax.bar(x,     pvaf_r, w, color="#3498db", label="vs rsfMRI band")
    ax.bar(x + w, pvaf_g, w, color="#e67e22", label="vs gastric band")
    ax.set_xticks(x); ax.set_xticklabels([pretty[a] for a in axes_order], rotation=30)
    ax.set_ylabel("PVAF (%)"); ax.set_title("A. PVAF by denominator (median across runs)")
    ax.legend(fontsize=9)

    # Panel B: empirical vs null PVAF_total with excess
    ax = fig.add_subplot(gs[0, 1])
    emp = [group_df.set_index("axis").loc[a, "pvaf_total_median_pct"] for a in axes_order]
    nul = [group_df.set_index("axis").loc[a, "pvaf_total_null_median_pct"] for a in axes_order]
    w = 0.35
    ax.bar(x - w/2, emp, w, color="#16a085", label="Empirical")
    ax.bar(x + w/2, nul, w, color="#bdc3c7", label="Mismatch null")
    sig = group_df.set_index("axis").loc[axes_order, "sig_fdr"].values
    for i, s in enumerate(sig):
        if s:
            ax.text(i, max(emp[i], nul[i]) * 1.05, "*", ha="center", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels([pretty[a] for a in axes_order], rotation=30)
    ax.set_ylabel("PVAF_total (%)"); ax.set_title("B. Empirical vs null PVAF_total\n(* = FDR q<0.05)")
    ax.legend(fontsize=9)

    # Panel C: where is the motion power? fraction in each band
    ax = fig.add_subplot(gs[0, 2])
    frac_g = [group_df.set_index("axis").loc[a, "motion_power_in_gastric_band_pct"]
              for a in MOTION_COLS]
    frac_r = [group_df.set_index("axis").loc[a, "motion_power_in_rsfMRI_band_pct"]
              for a in MOTION_COLS]
    xs = np.arange(len(MOTION_COLS)); w = 0.35
    ax.bar(xs - w/2, frac_r, w, color="#3498db", label="rsfMRI band")
    ax.bar(xs + w/2, frac_g, w, color="#e67e22", label="Gastric band")
    ax.set_xticks(xs); ax.set_xticklabels([pretty[a] for a in MOTION_COLS], rotation=30)
    ax.set_ylabel("Fraction of motion power (%)")
    ax.set_title("C. Where the motion power lives\n(median across runs)")
    ax.legend(fontsize=9)

    # Panel D: per-run boxplot of PVAF_total
    ax = fig.add_subplot(gs[1, 0])
    data_box = [per_run_df.query("axis == @a")["pvaf_total_pct"].values for a in axes_order]
    bp = ax.boxplot(data_box, labels=[pretty[a] for a in axes_order], showfliers=False,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2c3e50"); patch.set_alpha(0.7)
    ax.set_ylabel("PVAF_total per run (%)")
    ax.set_title("D. Per-run distribution (84 runs)")
    plt.setp(ax.get_xticklabels(), rotation=30)

    # Panel E: decomposition - PVAF_total = PVAF_gastric * band fraction
    ax = fig.add_subplot(gs[1, 1])
    pvaf_g_arr = np.array(pvaf_g[:6])
    frac_g_arr = np.array(frac_g) / 100.0
    predicted = pvaf_g_arr * frac_g_arr
    observed  = np.array(pvaf_t[:6])
    ax.scatter(predicted, observed, s=90, color="#c0392b", edgecolor="black")
    lim = max(np.nanmax(predicted), np.nanmax(observed)) * 1.1 + 1e-6
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5)
    for i, a in enumerate(MOTION_COLS):
        ax.annotate(pretty[a], (predicted[i], observed[i]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("PVAF_gastric * band-power-fraction (%)")
    ax.set_ylabel("PVAF_total observed (%)")
    ax.set_title("E. Spectral check:\nPVAF_total = PVAF_band * band fraction")

    # Panel F: numeric summary
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    fd_row = group_df.set_index("axis").loc["FD_Power2012"]
    motion_med_total = np.median([group_df.set_index("axis").loc[a, "pvaf_total_median_pct"]
                                  for a in MOTION_COLS])
    motion_med_rsf   = np.median([group_df.set_index("axis").loc[a, "pvaf_rsfMRI_median_pct"]
                                  for a in MOTION_COLS])
    motion_med_gast  = np.median([group_df.set_index("axis").loc[a, "pvaf_gastric_band_median_pct"]
                                  for a in MOTION_COLS])
    motion_med_band  = np.median([group_df.set_index("axis").loc[a, "motion_power_in_gastric_band_pct"]
                                  for a in MOTION_COLS])
    txt = (
        "SUMMARY\n"
        "==============================\n\n"
        f"Median across 6 axes:\n"
        f"  PVAF vs broadband motion : {motion_med_total:.3f}%\n"
        f"  PVAF vs rsfMRI band      : {motion_med_rsf:.3f}%\n"
        f"  PVAF vs gastric band     : {motion_med_gast:.3f}%\n\n"
        f"FD (Power 2012):\n"
        f"  PVAF vs broadband motion : {fd_row['pvaf_total_median_pct']:.3f}%\n"
        f"  Excess over null (median): {fd_row['pvaf_total_excess_median_pct']:+.3f}%\n\n"
        f"Where the motion lives:\n"
        f"  Median power in gastric band : {motion_med_band:.2f}%\n"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=11,
            family="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#f6f3ee", alpha=0.8))

    n_runs = per_run_df.query("axis == 'FD_Power2012'").shape[0]
    fig.suptitle("Percent Variance Accounted For (PVAF) of head motion by the gastric rhythm\n"
                 f"({n_runs} runs, mismatch-subjects null, FDR across 7 measures)",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


##############################################################################
# Main                                                                       #
##############################################################################


def main():
    log = []
    def log_print(msg):
        print(msg); log.append(msg)

    log_print(f"PVAF analysis started at {datetime.now().isoformat(timespec='seconds')}")
    log_print(f"clean_level={clean_level}  fs_fmri={SAMPLE_RATE_FMRI} Hz  "
              f"EGG fs={EGG_INTERMEDIATE_SFREQ} Hz")
    log_print(f"rsfMRI band: {RSFMRI_BAND}  gastric +-{bandpass_lim} Hz")
    log_print("[1/4] loading data...")
    all_data = load_all_subject_data()
    n_runs = len(all_data)
    n_subjects = len({d["subject"] for d in all_data.values()})
    log_print(f"  loaded {n_runs} runs from {n_subjects} subjects")
    if n_runs == 0:
        log_print("  no runs - aborting"); OUTPUT_LOG.write_text("\n".join(log)); return

    log_print("[2/4] per-run PVAF computation...")
    all_rows = []
    for idx, (k, data) in enumerate(all_data.items()):
        if idx % 10 == 0:
            log_print(f"  {data['subject']} run {data['run']}  ({idx+1}/{n_runs})")
        all_rows.extend(pvaf_for_run(data, all_data))
    per_run_df = pd.DataFrame(all_rows)
    per_run_df.to_csv(OUTPUT_PER_RUN, index=False)
    log_print(f"  wrote {OUTPUT_PER_RUN}")

    log_print("[3/4] group-level statistics + decomposition table...")
    group_df = group_level(per_run_df)
    group_df.to_csv(OUTPUT_GROUP_LEVEL, index=False)
    decomp = group_df[["axis", "pvaf_total_median_pct", "pvaf_rsfMRI_median_pct",
                       "pvaf_gastric_band_median_pct",
                       "motion_power_in_rsfMRI_band_pct",
                       "motion_power_in_gastric_band_pct"]].copy()
    decomp.to_csv(OUTPUT_DECOMP, index=False)
    log_print(f"  wrote {OUTPUT_GROUP_LEVEL}")
    log_print(f"  wrote {OUTPUT_DECOMP}")

    log_print("[4/4] figures...")
    plot_pvaf(group_df, per_run_df, OUTPUT_FIGURE)
    log_print(f"  wrote {OUTPUT_FIGURE}")

    log_print("done.")
    OUTPUT_LOG.write_text("\n".join(log))


if __name__ == "__main__":
    main()
