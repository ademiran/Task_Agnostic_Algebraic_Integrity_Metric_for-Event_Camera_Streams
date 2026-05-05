# PCC-Event — Reproducibility Package

Companion code and figures for the preprint:

> **PCC-Event: A Task-Agnostic Algebraic Integrity Metric for Event-Camera Streams Toward SOTIF-Compliant Perception**
> Arthur de Miranda Neto — Universidade Federal de Lavras (UFLA), Brazil — Preprint, May 2026.

This repository contains the procedural-synthetic simulation, the figures used in the preprint, and a 25-second demonstration video showing the three PCC-Event metrics in action.

---

## What is in this package

```
.
├── README.md                    -- this file
├── simulate.py                  -- generates the synthetic event stream and the 6 figures
├── build_video.py               -- generates the 25-second demonstration video
├── figures/
│   ├── fig1_event_emission.png      -- Eq. 1, single-pixel illustration
│   ├── fig2_representations.png     -- Eqs. 2-4, TS / EF / VG side by side
│   ├── fig3_r2ef_mechanism.png      -- Eq. 11, r2-EF binarization and ROI
│   ├── fig4_pccvg_gating.png        -- Eq. 12, PCC-VG temporal gating
│   ├── fig5_pccts_integrity.png     -- Eq. 10, PCC-TS integrity monitor
│   ├── fig6_isomorphism.png         -- conceptual diagram, Eq. 8
│   └── metrics.txt                  -- numerical summary of the synthetic experiment
└── video/
    └── PCC-Event_demo.mp4           -- 1280x720, 24 fps, 25 s demonstration
```

---

## What is *procedural synthetic*?

The event stream used throughout this package is **not** drawn from any recorded dataset (DSEC, GEN1, N-CARS) and is **not** generated from real video by a video-to-events tool such as v2e. Instead, the events are produced by direct numerical simulation of the standard event-emission rule

```
e = (x, y, t, p),  emit if  | log L(x,y,t) - log L(x,y,t-Δt) | >= C
```

on a controlled synthetic scene (a moving disk over a textured background, with a 100-ms tunnel-dip episode in which the global illumination drops by ~85 %). This makes every figure and every numerical value in the package **fully reproducible from code**, but it also means the package is **illustrative**, not experimental in the sense of validation on real recordings. Validation on real data is the next stage of the research program (Objective OS1 in the paper).

---

## Reproducing the figures and the video

### Requirements

- Python ≥ 3.9
- `numpy`, `matplotlib`
- `ffmpeg` (in PATH, for the video)

```bash
pip install numpy matplotlib
sudo apt-get install ffmpeg     # or equivalent for your platform
```

### Reproduce the 6 figures

```bash
python3 simulate.py
```

Output: `figures/fig{1..6}_*.png` and `figures/metrics.txt`. Run-time: ~1 minute on a laptop.

### Reproduce the 25 s demonstration video

```bash
python3 build_video.py
```

Output: `video/PCC-Event_demo.mp4` (1280×720, 24 fps, ~4 MB). Run-time: ~3 minutes on a laptop.

The video shows, side by side:

- the synthetic scene `L(x, y, t)`,
- the Time Surface (Eq. 2),
- the Event Frame (Eq. 4) accumulating in a 30-ms sliding window,
- the ROI selected by `r2-EF` (Eq. 11),
- the rolling `r_C(t)` integrity-monitor curve (Eq. 10), with the integrity-alarm overlay activated when `r_C < 0.4`,
- the rolling `r_VG(t)` temporal-gating curve (Eq. 12), with a "PIPELINE TRIGGERED / GATED" status indicator.

### Numerical summary of the synthetic stream

| Quantity | Value |
|---|---|
| Sensor resolution | 240 × 180 |
| Duration | 500 ms |
| Contrast threshold C | 0.15 |
| Total events emitted | 893,793 |
| `f_VG` (windows discarded by gating) | ~9 % |
| `\|ROI_EF\|/\|Ω_act\|` at t = 150 ms | ~50 % |
| `r_C` outside the dip | ≈ 0.93 |
| `r_C` minimum during the dip | ≈ −0.09 |
| `r_C` recovery time after dip | ~30 ms |

These values are the ones reported in the preprint, labelled `[ILLUSTR.-SYNTH.]`.

---

## License

The code in this repository is released for academic reproducibility purposes. Please cite the preprint if you use the code or build upon the framework. A formal license file will be added in a future release.

---

## Roadmap

- **OS1 — empirical validation on DSEC, GEN1, N-CARS.** Replace synthetic figures with experimental ones; quantify `f_VG`, `|ROI_EF|/|Ω_act|`, and threshold-C drift detection on real recordings.
- **OS2 — SNN-PCC.** A separate paper: spiking neural network approximating PCC directly on the raw `(x, y, t, p)` stream on neuromorphic hardware (Loihi 2, DYNAP-CNN).
- **OS3 — credal occupancy grids.** Mapping `r_C(t)` to Dempster-Shafer mass functions for multi-sensor fusion under extreme uncertainty.

Issues and pull requests are welcome.
