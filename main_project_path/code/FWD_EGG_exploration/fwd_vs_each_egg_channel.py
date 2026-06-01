#!/usr/bin/env python3
"""
FWD vs each EGG channel - exploratory test
==========================================

Companion script to the PVAF analysis. The OHBM 2025 pipeline collapses the
multi-electrode EGG to a single 'dominant channel' early in preprocessing
and never re-visits the other channels. This script asks two simple
sanity-check questions that are useful for the journal manuscript:

    Q1: How does framewise displacement (FD, Power et al. 2012) line up
        in time with each individual EGG channel for a few example runs?

    Q2: Across all runs, how does the FD~gastric coupling computed from
        the dominant channel compare with the coupling computed from
        every other electrode (correlation, coherence, phase-locking)?

It is intentionally exploratory rather than confirmatory: the goal is to
verify the FD computation, confirm the dominant-channel choice was
reasonable, and produce diagnostic plots that can be included in a
supplement.

FWD definition (Power et al. 2012, eq. 2)
-----------------------------------------
    FD(t) = |dx(t)| + |dy(t)| + |dz(t)|
          + r * (|d_pitch(t)| + |d_roll(t)| + |d_yaw(t)|)
with displacements in mm, rotations in RADIANS, and r = 50 mm (typical
sphere radius). AFNI 3dvolreg writes rotations in degrees, so we convert.

Outputs
-------
    outputs/fwd_per_channel_results.csv         per-(run, channel) stats
    outputs/fwd_overlay_<sub>_run<run>.png      example time courses
    outputs/fwd_channel_comparison.png          per-channel coupling boxplots
    outputs/fwd_dominant_vs_all.png             dominant vs other electrodes
    outputs/fwd_log.txt                         run log + FD sanity values

Usage
-----
    conda activate brain_gut
    python fwd_vs_each_egg_channel.py
"""

import os
import sys
import pathlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.signal import resample, coherence, hilbert
from scipy.stats import pearsonr, spearmanr
import bioread
from mne.filter import filter_data

warnings.filterwarnings("ignore")

##############################################################################
# Configuration                                                              #
##############################################################################

PARENT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = PARENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from config import (main_project_path, clean_level, sample_rate_fmri,
                    intermediate_sample_rate, trigger_channel, bandpass_lim,
                    filter_order, transition_width, freq_range)

META_DATAFRAME_PATH = PROJECT_ROOT / "dataframes" / "egg_brain_meta_data.csv"

MOTION_FILE_TEMPLATE = os.path.join(
    main_project_path, "BIDS_data", "sub_motion_files",
    "sub-{sub}_dfile.r0{run}.1D"
)
ACQ_FILE_TEMPLATE = os.path.join(
    main_project_path, "physio", "{sub}", "egg", "{sub}_rest{run}.acq"
)
GASTRIC_FREQ_TEMPLATE = os.path.join(
    main_project_path, "derivatives", "brain_gast", "{sub}", "{sub}{run}",
    "max_freq{sub}_run{run}{clean_level}.npy"
)

OUTPUT_DIR = PARENT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "fwd_per_channel_results.csv"
OUTPUT_BOXPLOT = OUTPUT_DIR / "fwd_channel_comparison.png"
OUTPUT_DOM = OUTPUT_DIR / "fwd_dominant_vs_all.png"
OUTPUT_LOG = OUTPUT_DIR / "fwd_log.txt"

MOTION_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
SAMPLE_RATE_FMRI = sample_rate_fmri
EGG_RESAMPLE_HZ = intermediate_sample_rate  # 10 Hz
TRIGGER_CHANNEL = trigger_channel
FD_RADIUS_MM = 50.0

# Number of example time-course overlay PNGs to emit (in addition to the
# group-level summary plots, which use every run).
N_EXAMPLE_OVERLAYS = 3


##############################################################################
# Helpers                                                                    #
##############################################################################


def bp_filter_1d(x, sfreq, l_freq, h_freq, filt_order=filter_order,
                 trans_width=transition_width):
    filter_length = int(filt_order * np.floor(sfreq / l_freq))
    out = filter_data(
        data=np.asarray(x, dtype=float).reshape(1, -1),
        sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
        filter_length=filter_length,
        l_trans_bandwidth=trans_width * l_freq,
        h_trans_bandwidth=trans_width * h_freq,
        n_jobs=1, method="fir", phase="zero-double",
        fir_window="hamming", fir_design="firwin2", verbose=False,
    )
    return out.flatten()


def fd_power_2012(motion_df, radius_mm=FD_RADIUS_MM):
    """
    FD(t) = sum_i |dx_i(t)| with rotations converted from deg -> rad -> mm.
    """
    m = motion_df.copy()
    for col in ["rot_x", "rot_y", "rot_z"]:
        m[col] = np.deg2rad(m[col]) * radius_mm
    diffs = m.diff().fillna(0).abs()
    return diffs.sum(axis=1).values


def plv_two_signals(a, b):
    pa = np.angle(hilbert(a))
    pb = np.angle(hilbert(b))
    return float(np.abs(np.mean(np.exp(1j * (pa - pb)))))


def coherence_in_band(a, b, fs, band, nperseg=None):
    if nperseg is None:
        nperseg = min(64, len(a) // 4)
        if nperseg < 16:
            nperseg = max(8, len(a) // 2)
    f, coh = coherence(a, b, fs=fs, nperseg=nperseg)
    in_band = (f >= band[0]) & (f <= band[1])
    if not np.any(in_band):
        idx = np.argmin(np.abs(f - np.mean(band)))
        return float(coh[idx]), float(coh[idx])
    return float(np.mean(coh[in_band])), float(np.max(coh[in_band]))


def load_egg_channels_from_acq(sub, run, record_meta_row):
    """
    Read the raw .acq, find the MRI trigger window, slice and resample each
    analog EGG channel to EGG_RESAMPLE_HZ.

    Returns (list[np.ndarray channels at EGG_RESAMPLE_HZ], original_sfreq).
    """
    path = ACQ_FILE_TEMPLATE.format(sub=sub, run=run)
    data = bioread.read_file(path)
    orig_sr = data.channels[0].samples_per_second
    duration_samples = int(orig_sr * record_meta_row["mri_length"])
    num_gast = int(record_meta_row["num_channles"])

    trigger = data.channels[TRIGGER_CHANNEL].data.astype(int)
    trigger_meta = record_meta_row["trigger_start"]
    if str(trigger_meta).strip() == "auto":
        if trigger[0] == 0:
            start = np.where(trigger)[0][0]
        else:
            no_trig = np.where(trigger == 0)[0]
            trig_on = np.where(trigger >= 0.999)[0]
            start = trig_on[trig_on > no_trig[0]][0]
    else:
        start = int(max(float(trigger_meta), 0) * orig_sr)
    end = start + duration_samples

    raw_channels = [data.channels[i].data[start:end] for i in range(num_gast)]

    n_resampled = int((len(raw_channels[0]) / orig_sr) * EGG_RESAMPLE_HZ)
    resampled = [sp_signal.resample(ch, n_resampled) for ch in raw_channels]
    return resampled, orig_sr


def align_to_fmri(channel_egg, motion_df):
    """
    Resample EGG (at EGG_RESAMPLE_HZ) down to SAMPLE_RATE_FMRI to match
    motion sampling. Truncate to the shorter of the two.
    """
    n_fmri = int(len(channel_egg) / EGG_RESAMPLE_HZ * SAMPLE_RATE_FMRI)
    egg_at_fmri = sp_signal.resample(channel_egg, n_fmri)
    n = min(len(egg_at_fmri), len(motion_df))
    return egg_at_fmri[:n], motion_df.iloc[:n].reset_index(drop=True)


##############################################################################
# Per-run analysis                                                           #
##############################################################################


def analyse_run(sub, run, record_meta_row):
    """
    Compute FD-vs-each-EGG-channel coupling for one run.

    Returns one record per (run, channel index).
    """
    motion_path = MOTION_FILE_TEMPLATE.format(sub=sub, run=run)
    if not os.path.isfile(motion_path):
        return []
    motion = np.loadtxt(motion_path)
    motion_df = pd.DataFrame(motion, columns=MOTION_COLS)

    try:
        raw_chs, orig_sr = load_egg_channels_from_acq(sub, run, record_meta_row)
    except Exception as exc:
        return [{"subject": sub, "run": int(run), "channel": -1, "error": str(exc)}]

    freq_path = GASTRIC_FREQ_TEMPLATE.format(sub=sub, run=run, clean_level=clean_level)
    if not os.path.isfile(freq_path):
        return [{"subject": sub, "run": int(run), "channel": -1,
                 "error": "no max_freq file"}]
    gastric_peak = float(np.load(freq_path).flatten()[0])
    band = (gastric_peak - bandpass_lim, gastric_peak + bandpass_lim)

    # Detect the dominant channel from the data itself: the channel whose
    # bandpassed signal (at EGG_RESAMPLE_HZ) has the highest variance. This
    # matches what preprocess_gastric.py picks via Welch peak power. The
    # metadata 'dominant_channel' column is "auto" for most runs and is not
    # back-filled, so we cannot rely on it.
    bp_powers = []
    for ch in raw_chs:
        try:
            ch_bp_10hz = bp_filter_1d(ch, EGG_RESAMPLE_HZ, band[0], band[1])
            bp_powers.append(float(np.var(ch_bp_10hz)))
        except Exception:
            bp_powers.append(np.nan)
    dom_idx = int(np.nanargmax(bp_powers)) if any(np.isfinite(bp_powers)) else -1

    rows = []
    fwd_at_fmri = None    # raw FD time course, computed once per run
    fwd_bp = None         # FD bandpassed at the gastric peak +/- 0.015 Hz
    overlay_payload = {"sub": sub, "run": run, "gastric_peak": gastric_peak,
                       "fd": None, "channels_bp_fmri": [], "time_s": None}

    for ch_idx, ch in enumerate(raw_chs):
        try:
            egg_at_fmri, mdf = align_to_fmri(ch, motion_df)
            if fwd_at_fmri is None:
                fwd_at_fmri = fd_power_2012(mdf)
                # Bandpass FD at the same gastric band the EGG was filtered
                # at, so the PLV / Pearson comparisons are between two
                # narrowband signals (Issue 1 + Issue 2 of the review).
                fwd_bp = bp_filter_1d(fwd_at_fmri, SAMPLE_RATE_FMRI,
                                      band[0], band[1])
                overlay_payload["fd"] = fwd_at_fmri
                overlay_payload["time_s"] = (
                    np.arange(len(fwd_at_fmri)) / SAMPLE_RATE_FMRI
                )

            ch_bp = bp_filter_1d(egg_at_fmri, SAMPLE_RATE_FMRI, band[0], band[1])
            overlay_payload["channels_bp_fmri"].append(ch_bp)

            n = min(len(ch_bp), len(fwd_bp))
            fd_n     = fwd_at_fmri[:n]   # raw FD (mm)
            fd_bp_n  = fwd_bp[:n]        # bandpassed FD (mm in the gastric band)
            ch_n     = ch_bp[:n]         # bandpassed EGG channel
            if (n < 30 or np.std(fd_bp_n) < 1e-12 or np.std(ch_n) < 1e-12):
                rows.append({"subject": sub, "run": int(run),
                             "channel": ch_idx,
                             "is_dominant": False,
                             "error": "too short or flat"})
                continue

            # PHASE-LOCKING COUPLING (narrowband vs narrowband, Issue 1)
            r_bp,  p_bp  = pearsonr(fd_bp_n, ch_n)
            plv = plv_two_signals(fd_bp_n, ch_n)
            coh_mean, coh_peak = coherence_in_band(
                fd_bp_n, ch_n, SAMPLE_RATE_FMRI, band
            )

            # AMPLITUDE COUPLING (raw FD vs EGG envelope, Issue 2 option b).
            # Both signals are slow and non-negative, so the Pearson /
            # Spearman correlation between them is interpretable as
            # "do bigger gastric oscillations come with more head motion".
            egg_envelope = np.abs(hilbert(ch_n))
            r_env_p, _   = pearsonr(fd_n, egg_envelope)
            r_env_s, _   = spearmanr(fd_n, egg_envelope)

            rows.append({
                "subject": sub, "run": int(run), "channel": ch_idx,
                "is_dominant": (ch_idx == dom_idx),
                "gastric_peak_Hz": gastric_peak,
                "n_timepoints": n,
                "fd_mean_mm": float(np.mean(fd_n)),
                "fd_max_mm":  float(np.max(fd_n)),
                "egg_band_rms": float(np.std(ch_n)),
                # Phase-locked coupling (bandpassed FD vs bandpassed EGG):
                "pearson_r_fd_vs_egg":  r_bp,
                "pearson_p_fd_vs_egg":  p_bp,
                "plv_fd_vs_egg":        plv,
                "coh_band_mean":        coh_mean,
                "coh_band_peak":        coh_peak,
                # Amplitude coupling (raw FD vs EGG envelope):
                "pearson_r_fd_vs_egg_envelope":  r_env_p,
                "spearman_r_fd_vs_egg_envelope": r_env_s,
                "error": "",
            })
        except Exception as exc:
            rows.append({"subject": sub, "run": int(run), "channel": ch_idx,
                         "error": str(exc)})

    return rows, overlay_payload


##############################################################################
# Plotting                                                                   #
##############################################################################


def plot_overlay(payload, out_path):
    fd = payload["fd"]
    chs = payload["channels_bp_fmri"]
    t = payload["time_s"]
    n_chs = len(chs)
    fig, axes = plt.subplots(n_chs + 1, 1, figsize=(12, 2.0 + 1.6 * n_chs),
                             sharex=True)
    axes[0].plot(t, fd, color="#c0392b", lw=1.2)
    axes[0].set_ylabel("FD (mm)")
    axes[0].set_title(f"{payload['sub']} run {payload['run']}  - gastric peak "
                      f"{payload['gastric_peak']:.4f} Hz")
    axes[0].grid(alpha=0.3)
    colors = ["#2c3e50", "#3498db", "#e67e22", "#16a085", "#8e44ad"]
    for i, ch in enumerate(chs):
        axes[i + 1].plot(t[:len(ch)], ch, color=colors[i % len(colors)], lw=1.0)
        axes[i + 1].set_ylabel(f"EGG ch{i}\n(band)")
        axes[i + 1].grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_channel_comparison(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [("plv_fd_vs_egg", "PLV (FD vs EGG channel)"),
               ("coh_band_mean", "Coherence in gastric band"),
               ("pearson_r_fd_vs_egg", "Pearson r")]
    channels = sorted(df["channel"].dropna().unique())
    for ax, (col, title) in zip(axes, metrics):
        data_box = [df.loc[df["channel"] == c, col].dropna().values for c in channels]
        bp = ax.boxplot(data_box, labels=[f"ch{int(c)}" for c in channels],
                        showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db"); patch.set_alpha(0.6)
        ax.set_title(title)
        ax.axhline(0, color="grey", lw=0.6)
        ax.grid(alpha=0.3)
    fig.suptitle("FD vs each EGG channel - all runs", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_dominant_vs_others(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [("plv_fd_vs_egg", "PLV"),
               ("coh_band_mean", "Coherence (gastric band)"),
               ("pearson_r_fd_vs_egg", "Pearson r")]
    for ax, (col, title) in zip(axes, metrics):
        dom_vals = df.loc[df["is_dominant"] == True, col].dropna().values
        oth_vals = df.loc[df["is_dominant"] == False, col].dropna().values
        if len(dom_vals) == 0:
            dom_vals = np.array([np.nan])
        bp = ax.boxplot([dom_vals, oth_vals],
                        labels=["dominant", "other electrodes"],
                        showfliers=False, patch_artist=True)
        bp["boxes"][0].set_facecolor("#e67e22"); bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#bdc3c7"); bp["boxes"][1].set_alpha(0.7)
        ax.set_title(title)
        ax.axhline(0, color="grey", lw=0.6)
        ax.grid(alpha=0.3)
    fig.suptitle("Does the dominant-channel choice carry the FD-EGG coupling?",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


##############################################################################
# Main                                                                       #
##############################################################################


def main():
    log = []
    def log_print(msg):
        print(msg); log.append(msg)

    log_print(f"FWD-vs-EGG exploration started {datetime.now().isoformat(timespec='seconds')}")
    meta = pd.read_csv(META_DATAFRAME_PATH)
    if clean_level == "strict_gs_cardiac":
        meta = meta.loc[(meta["ppu_exclude"] == False) &
                        (meta["ppu_found"] == True)]
    log_print(f"  {len(meta)} runs in metadata")

    all_rows = []
    overlay_payloads = []
    for _, row in meta.iterrows():
        sub = row["subject"]; run = str(row["run"])
        acq_path = ACQ_FILE_TEMPLATE.format(sub=sub, run=run)
        if not os.path.isfile(acq_path):
            log_print(f"  skip {sub} run {run}: no .acq")
            continue
        try:
            result = analyse_run(sub, run, row)
            if isinstance(result, tuple):
                rows, payload = result
                all_rows.extend(rows)
                overlay_payloads.append(payload)
            else:
                all_rows.extend(result)
        except Exception as exc:
            log_print(f"  fail {sub} run {run}: {exc}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    log_print(f"  wrote {OUTPUT_CSV}  ({len(df)} rows)")

    # FD sanity printout (Power 2012 typical mean ~0.1-0.3 mm in healthy adults)
    if "fd_mean_mm" in df.columns:
        log_print(f"  FD mean across runs: median={df['fd_mean_mm'].median():.4f} mm "
                  f"(IQR {df['fd_mean_mm'].quantile(0.25):.4f} - "
                  f"{df['fd_mean_mm'].quantile(0.75):.4f})")
        log_print(f"  FD max  across runs: median={df['fd_max_mm'].median():.4f} mm")

    if len(df) > 0:
        plot_channel_comparison(df.dropna(subset=["plv_fd_vs_egg"]), OUTPUT_BOXPLOT)
        plot_dominant_vs_others(df.dropna(subset=["plv_fd_vs_egg"]), OUTPUT_DOM)
        log_print(f"  wrote {OUTPUT_BOXPLOT}")
        log_print(f"  wrote {OUTPUT_DOM}")

    # Emit overlay PNGs for the first N_EXAMPLE_OVERLAYS runs
    for payload in overlay_payloads[:N_EXAMPLE_OVERLAYS]:
        if payload["fd"] is None or len(payload["channels_bp_fmri"]) == 0:
            continue
        png = OUTPUT_DIR / f"fwd_overlay_{payload['sub']}_run{payload['run']}.png"
        plot_overlay(payload, png)
        log_print(f"  wrote {png}")

    log_print("done.")
    OUTPUT_LOG.write_text("\n".join(log))


if __name__ == "__main__":
    main()
