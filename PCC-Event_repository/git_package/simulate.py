"""
PCC-Event: procedural synthetic event-stream generator and metrics computation.

This script generates a synthetic scene (moving disk + texture + a tunnel-like
luminance dip) on a 240x180 sensor at high temporal resolution, emits events
following the standard log-luminance threshold model |delta L| >= C, then
computes Time Surfaces, Event Frames, Voxel Grids, and the PCC-Event metrics
r-TS / r2-EF / r-VG.

All data is procedurally synthetic. No real dataset (DSEC / GEN1 / v2e on real
video) is used. Figures are explicitly labelled as such.

Outputs: PNG figures for the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

np.random.seed(42)
OUT = "/home/claude/paper/figures"
os.makedirs(OUT, exist_ok=True)

# ---------- sensor ----------
H, W = 180, 240                 # sensor resolution
DT_SIM = 1e-4                   # 100 us simulation tick
T_TOTAL = 0.5                   # 500 ms scene
N_TICKS = int(T_TOTAL / DT_SIM)
C = 0.15                        # contrast threshold (log units)


# ---------- scene ----------
def luminance_at(t, ticks_per_sec=int(1/DT_SIM)):
    """
    Build the luminance image L(x,y,t) for instant t.
    - background texture (static)
    - moving disk (translating across the sensor)
    - tunnel-like dip in global illumination around t in [0.20, 0.30]
    """
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # static low-frequency texture
    bg = 0.5 + 0.15 * np.sin(xx * 0.05) * np.cos(yy * 0.04)

    # moving disk
    cx = 30 + (W - 60) * (t / T_TOTAL)
    cy = H / 2 + 25 * np.sin(2 * np.pi * t * 4)
    r = 18
    dist = (xx - cx) ** 2 + (yy - cy) ** 2
    disk = np.where(dist <= r * r, 0.85, 0.0)

    L = bg + disk
    L = np.clip(L, 0.05, 1.5)

    # tunnel dip: global illumination drops between 0.20 s and 0.30 s
    if 0.20 < t < 0.30:
        dip = 1.0 - 0.85 * np.sin(np.pi * (t - 0.20) / 0.10)
        L = L * dip + 0.02

    # mild per-pixel noise
    L = L * (1.0 + 0.01 * np.random.randn(H, W))
    return np.clip(L, 0.01, None)


# ---------- event emission (Eq. 1) ----------
def simulate_events():
    """Emit events whenever |log L(t) - log L_ref| >= C, then update L_ref."""
    L0 = luminance_at(0.0)
    log_ref = np.log(L0)

    events = []  # list of (t, x, y, p)
    for k in range(1, N_TICKS):
        t = k * DT_SIM
        L = luminance_at(t)
        log_L = np.log(L)
        delta = log_L - log_ref

        pos = delta >= C
        neg = delta <= -C

        if pos.any():
            ys, xs = np.where(pos)
            for x, y in zip(xs, ys):
                events.append((t, int(x), int(y), +1))
            log_ref[pos] = log_L[pos]

        if neg.any():
            ys, xs = np.where(neg)
            for x, y in zip(xs, ys):
                events.append((t, int(x), int(y), -1))
            log_ref[neg] = log_L[neg]

    arr = np.array(events, dtype=[('t', 'f8'), ('x', 'i4'),
                                   ('y', 'i4'), ('p', 'i1')])
    print(f"Generated {len(arr)} events over {T_TOTAL*1000:.0f} ms.")
    return arr


# ---------- representations ----------
def time_surface(events, t_now, decay=0.05):
    """TS(x,y,t) = exp(-(t - tau(x,y))/delta), Eq. 2."""
    tau = -np.inf * np.ones((H, W))
    mask = events['t'] <= t_now
    for ev in events[mask]:
        tau[ev['y'], ev['x']] = ev['t']
    age = t_now - tau
    age[~np.isfinite(age)] = 1e6
    TS = np.exp(-age / decay)
    return TS


def event_frame(events, t0, T):
    """EF(x,y) = sum p_k for t_k in [t0, t0+T], Eq. 4."""
    EF = np.zeros((H, W), dtype=np.int32)
    mask = (events['t'] >= t0) & (events['t'] < t0 + T)
    sub = events[mask]
    for ev in sub:
        EF[ev['y'], ev['x']] += int(ev['p'])
    return EF


def voxel_grid(events, t0, T, B=5):
    """VG(x,y,b) approximation, Eq. 3 (simple bilinear in time)."""
    VG = np.zeros((B, H, W), dtype=np.float32)
    mask = (events['t'] >= t0) & (events['t'] < t0 + T)
    sub = events[mask]
    if len(sub) == 0:
        return VG
    norm_t = (sub['t'] - t0) / T * (B - 1)
    b_lo = np.floor(norm_t).astype(int)
    b_hi = np.clip(b_lo + 1, 0, B - 1)
    w_hi = norm_t - b_lo
    w_lo = 1.0 - w_hi
    for k in range(len(sub)):
        VG[b_lo[k], sub[k]['y'], sub[k]['x']] += sub[k]['p'] * w_lo[k]
        VG[b_hi[k], sub[k]['y'], sub[k]['x']] += sub[k]['p'] * w_hi[k]
    return VG


# ---------- PCC-Event metrics ----------
def pcc(a, b):
    """Pearson, Eq. 5."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    da = np.sqrt((a * a).sum())
    db = np.sqrt((b * b).sum())
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float((a * b).sum() / (da * db))


def r2_ef(EF):
    """r2_EF = sign(EF - mu_EF) on active pixels, Eq. 11."""
    active = EF != 0
    if active.sum() == 0:
        return np.zeros_like(EF), active, 0.0
    mu = EF[active].mean()
    r2 = np.zeros_like(EF, dtype=np.int8)
    r2[active] = np.sign(EF[active] - mu).astype(np.int8)
    roi = (r2 == -1) & active
    return r2, active, mu


# =========================================================================
# RUN SIMULATION
# =========================================================================
events = simulate_events()


# =========================================================================
# FIGURE 1 -- Event emission model on a single pixel (illustrate Eq. 1)
# =========================================================================
print("Building Fig. 1 ...")
px, py = 120, 90
ticks = np.arange(N_TICKS) * DT_SIM
L_pix = np.array([luminance_at(t)[py, px] for t in ticks[::5]])
t_pix = ticks[::5]
logL = np.log(L_pix)

# events for this pixel
mask_px = (events['x'] == px) & (events['y'] == py)
ev_px = events[mask_px]

fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                        gridspec_kw={'height_ratios': [3, 1]})
ax[0].plot(t_pix * 1000, logL, color='#1F3864', lw=1.5,
           label='log L(x,y,t)')
ax[0].axhline(logL[0], color='gray', lw=0.8, ls='--',
              label='log L_ref (initial)')
ax[0].fill_between(t_pix * 1000, logL[0] + C, logL[0] - C, alpha=0.1,
                   color='red', label=f'±C band (C={C})')
ax[0].set_ylabel('log L  [log units]')
ax[0].set_title(
    f'Synthetic per-pixel log-luminance and emitted events  (pixel x={px}, y={py})')
ax[0].legend(loc='lower left', fontsize=8)
ax[0].grid(alpha=0.3)

for ev in ev_px:
    color = 'red' if ev['p'] > 0 else 'blue'
    ax[1].vlines(ev['t'] * 1000, 0, ev['p'], colors=color, lw=1.2)
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_ylim(-1.5, 1.5)
ax[1].set_yticks([-1, 0, 1])
ax[1].set_ylabel('polarity p')
ax[1].set_xlabel('t  [ms]')
ax[1].grid(alpha=0.3)
ax[1].text(0.99, 0.95, f'{len(ev_px)} events on this pixel',
           transform=ax[1].transAxes, ha='right', va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUT}/fig1_event_emission.png', dpi=160,
            bbox_inches='tight')
plt.close()


# =========================================================================
# FIGURE 2 -- Three intermediate representations TS / EF / VG (Eqs. 2-4)
# =========================================================================
print("Building Fig. 2 ...")
t_view = 0.15
T_win = 0.030
TS = time_surface(events, t_view, decay=0.030)
EF = event_frame(events, t_view, T_win)
VG = voxel_grid(events, t_view, T_win, B=5)

fig, ax = plt.subplots(2, 3, figsize=(11, 6))
# raw scene
ax[0, 0].imshow(luminance_at(t_view), cmap='gray', vmin=0, vmax=1.5)
ax[0, 0].set_title(f'Synthetic scene L(x,y,t)  @ t={t_view*1000:.0f} ms')
ax[0, 0].axis('off')

# TS
im1 = ax[0, 1].imshow(TS, cmap='hot', vmin=0, vmax=1)
ax[0, 1].set_title('Time Surface (Eq. 2),  δ=30 ms')
ax[0, 1].axis('off')
plt.colorbar(im1, ax=ax[0, 1], fraction=0.04)

# EF
vmax = max(1, np.abs(EF).max())
im2 = ax[0, 2].imshow(EF, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax[0, 2].set_title(
    f'Event Frame (Eq. 4),  T={T_win*1000:.0f} ms\n'
    f'{(EF != 0).sum()} active pixels')
ax[0, 2].axis('off')
plt.colorbar(im2, ax=ax[0, 2], fraction=0.04)

# VG: show 3 of 5 bins
for i, b in enumerate([0, 2, 4]):
    bound = max(1, np.abs(VG[b]).max())
    im = ax[1, i].imshow(VG[b], cmap='RdBu_r', vmin=-bound, vmax=bound)
    ax[1, i].set_title(f'Voxel Grid (Eq. 3),  bin b={b}')
    ax[1, i].axis('off')
    plt.colorbar(im, ax=ax[1, i], fraction=0.04)

fig.suptitle(
    'Procedural synthetic event stream  —  representations TS / EF / VG',
    fontsize=11, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_representations.png', dpi=160,
            bbox_inches='tight')
plt.close()


# =========================================================================
# FIGURE 3 -- r2-EF mechanism: ROI selection (Eq. 11)
# =========================================================================
print("Building Fig. 3 ...")
EF = event_frame(events, t_view, T_win)
r2, active, mu = r2_ef(EF)
roi_count = ((r2 == -1) & active).sum()
act_count = active.sum()

fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
vmax = max(1, np.abs(EF).max())
ax[0].imshow(EF, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax[0].set_title(f'Event Frame  (Eq. 4)\nμ_EF = {mu:.2f}  '
                f'(over {act_count} active px)')
ax[0].axis('off')

ax[1].imshow(r2, cmap='RdBu_r', vmin=-1, vmax=1)
ax[1].set_title(
    'r₂-EF = sign(EF − μ_EF)   (Eq. 11)\n'
    'red = +1, blue = −1, white = inactive')
ax[1].axis('off')

roi_mask = (r2 == -1) & active
ax[2].imshow(luminance_at(t_view), cmap='gray', vmin=0, vmax=1.5)
overlay = np.zeros((*roi_mask.shape, 4))
overlay[roi_mask] = [1, 0, 0, 0.55]
ax[2].imshow(overlay)
ratio = roi_count / max(1, act_count)
ax[2].set_title(
    f'ROI_EF overlay on scene\n'
    f'|ROI_EF|/|Ω_act| = {ratio:.2%}   ({roi_count}/{act_count} px)')
ax[2].axis('off')

fig.suptitle(
    'r₂-EF mechanism  —  pure-integer ROI selection on synthetic event frame',
    fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_r2ef_mechanism.png', dpi=160,
            bbox_inches='tight')
plt.close()


# =========================================================================
# FIGURE 4 -- PCC-VG temporal coherence over time (Eq. 12) and the
# tunnel-dip alarm scenario
# =========================================================================
print("Building Fig. 4 ...")
T_win = 0.020
delta = 0.020
times = np.arange(0.020, T_TOTAL - delta, T_win / 2)
r_vg_vals = []
ef_density = []
for t0 in times:
    VG_a = voxel_grid(events, t0, T_win, B=5)
    VG_b = voxel_grid(events, t0 + delta, T_win, B=5)
    r_vg_vals.append(pcc(VG_a, VG_b))
    EF_t = event_frame(events, t0, T_win)
    ef_density.append((EF_t != 0).sum() / (H * W))
r_vg_vals = np.array(r_vg_vals)
ef_density = np.array(ef_density)

theta = 0.5  # gating threshold
trigger = r_vg_vals < theta
f_VG_estimate = 1.0 - trigger.mean()  # fraction of windows discarded

fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax[0].plot(times * 1000, r_vg_vals, color='#1F3864', lw=1.5,
           label='r_VG(t, t+Δ)  (Eq. 12)')
ax[0].axhline(theta, color='red', ls='--', lw=1,
              label=f'θ = {theta}')
ax[0].fill_between(times * 1000, theta, 1.05,
                   where=r_vg_vals >= theta, alpha=0.15,
                   color='green',
                   label='redundant — pipeline NOT triggered')
ax[0].fill_between(times * 1000, -1.05, theta,
                   where=r_vg_vals < theta, alpha=0.15,
                   color='red',
                   label='changing — pipeline triggered')
ax[0].axvspan(200, 300, color='orange', alpha=0.15,
              label='synthetic tunnel-dip episode')
ax[0].set_ylabel('r_VG  ∈ [−1, +1]')
ax[0].set_ylim(-1.05, 1.05)
ax[0].set_title(
    f'PCC-VG temporal coherence  —  estimated f_VG ≈ '
    f'{f_VG_estimate:.0%} of windows discarded')
ax[0].legend(loc='lower right', fontsize=8, ncol=2)
ax[0].grid(alpha=0.3)

ax[1].plot(times * 1000, ef_density * 100, color='#7F7F7F', lw=1.2)
ax[1].axvspan(200, 300, color='orange', alpha=0.15)
ax[1].set_ylabel('|Ω_act|/(H·W)  [%]')
ax[1].set_xlabel('t  [ms]')
ax[1].set_title('Event-frame activity density (sanity check)')
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/fig4_pccvg_gating.png', dpi=160,
            bbox_inches='tight')
plt.close()


# =========================================================================
# FIGURE 5 -- PCC-TS integrity monitor r_C(t) (Eq. 10) -- detect tunnel
# dip as an integrity event when prediction != observation
# =========================================================================
print("Building Fig. 5 ...")
# We mimic the BiasBench-style scenario: a *predicted* TS based on what
# the system "expected" given known ego-motion of the disk. We approximate
# the prediction by computing TS from the same scene WITHOUT the tunnel
# dip, so divergence between observed and predicted TS during 0.20-0.30 s
# triggers the integrity alarm.

def luminance_predicted(t):
    """Predicted scene without the tunnel dip."""
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    bg = 0.5 + 0.15 * np.sin(xx * 0.05) * np.cos(yy * 0.04)
    cx = 30 + (W - 60) * (t / T_TOTAL)
    cy = H / 2 + 25 * np.sin(2 * np.pi * t * 4)
    r = 18
    dist = (xx - cx) ** 2 + (yy - cy) ** 2
    disk = np.where(dist <= r * r, 0.85, 0.0)
    return np.clip(bg + disk, 0.05, 1.5)


def simulate_predicted_events():
    L0 = luminance_predicted(0.0)
    log_ref = np.log(L0)
    events = []
    for k in range(1, N_TICKS):
        t = k * DT_SIM
        log_L = np.log(luminance_predicted(t))
        delta = log_L - log_ref
        pos = delta >= C
        neg = delta <= -C
        if pos.any():
            ys, xs = np.where(pos)
            for x, y in zip(xs, ys):
                events.append((t, int(x), int(y), +1))
            log_ref[pos] = log_L[pos]
        if neg.any():
            ys, xs = np.where(neg)
            for x, y in zip(xs, ys):
                events.append((t, int(x), int(y), -1))
            log_ref[neg] = log_L[neg]
    return np.array(events, dtype=[('t', 'f8'), ('x', 'i4'),
                                    ('y', 'i4'), ('p', 'i1')])


print("  -- generating predicted (dip-free) reference stream ...")
events_pred = simulate_predicted_events()

times_rc = np.arange(0.030, T_TOTAL, 0.010)
r_c_vals = []
for tn in times_rc:
    TS_obs = time_surface(events, tn, decay=0.030)
    TS_pred = time_surface(events_pred, tn, decay=0.030)
    r_c_vals.append(pcc(TS_obs, TS_pred))
r_c_vals = np.array(r_c_vals)

theta_low, theta_high = 0.4, 0.7

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(times_rc * 1000, r_c_vals, color='#1F3864', lw=1.8,
        label='r_C(t) = r(TS_obs, TS_pred)   (Eq. 10)')
ax.axhline(theta_low, color='red', ls='--', lw=1,
           label=f'θ_low = {theta_low}')
ax.axhline(theta_high, color='green', ls='--', lw=1,
           label=f'θ_high = {theta_high}')
ax.fill_between(times_rc * 1000, -1.05, theta_low,
                where=r_c_vals < theta_low, alpha=0.25,
                color='red', label='ALARM: integrity violation')
ax.axvspan(200, 300, color='orange', alpha=0.15,
           label='synthetic tunnel-dip episode')
ax.set_ylabel('r_C(t)  ∈ [−1, +1]')
ax.set_xlabel('t  [ms]')
ax.set_ylim(-1.05, 1.05)
ax.set_title(
    'PCC-TS integrity monitor on synthetic stream  —  '
    'tunnel-dip detected as r_C drop')
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_pccts_integrity.png', dpi=160,
            bbox_inches='tight')
plt.close()


# =========================================================================
# FIGURE 6 -- structural isomorphism diagram (conceptual; Eq. 8)
# =========================================================================
print("Building Fig. 6 ...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# left box: PCC frame-based
ax.add_patch(Rectangle((0.3, 0.6), 4.4, 4.6, fc='#EAF1F8',
                        ec='#1F3864', lw=1.5))
ax.text(2.5, 4.8, 'PCC paradigm  [9, 10]',
        ha='center', fontsize=12, fontweight='bold', color='#1F3864')
ax.text(0.5, 4.2, 'Reference:  μ(scene)  — adaptive',
        fontsize=10)
ax.text(0.5, 3.6, 'Criterion:  r < θ  →  process',
        fontsize=10)
ax.text(0.5, 3.0, 'Granularity:  frame / ROI',
        fontsize=10)
ax.text(0.5, 2.4, 'Threshold:  θ  (adaptive, software)',
        fontsize=10)
ax.text(0.5, 1.8, 'Distribution:  free  (non-parametric)',
        fontsize=10)
ax.text(0.5, 1.2, 'Reference type:  RELATIONAL',
        fontsize=10, fontweight='bold', color='#C00000')

# right box: event camera
ax.add_patch(Rectangle((5.3, 0.6), 4.4, 4.6, fc='#F8EAEA',
                        ec='#C00000', lw=1.5))
ax.text(7.5, 4.8, 'Event-camera paradigm  [1]',
        ha='center', fontsize=12, fontweight='bold', color='#C00000')
ax.text(5.5, 4.2, 'Reference:  L(x,y,t−Δt)  — per-pixel',
        fontsize=10)
ax.text(5.5, 3.6, 'Criterion:  |ΔL| ≥ C  →  emit event',
        fontsize=10)
ax.text(5.5, 3.0, 'Granularity:  pixel, asynchronous',
        fontsize=10)
ax.text(5.5, 2.4, 'Threshold:  C  (fixed, hardware bias)',
        fontsize=10)
ax.text(5.5, 1.8, 'Distribution:  free',
        fontsize=10)
ax.text(5.5, 1.2, 'Reference type:  ABSOLUTE',
        fontsize=10, fontweight='bold', color='#C00000')

# bidirectional arrow
ax.annotate('', xy=(5.2, 3.0), xytext=(4.8, 3.0),
            arrowprops=dict(arrowstyle='<->', lw=2, color='black'))
ax.text(5.0, 3.4, 'isomorphism\n(Eq. 8)',
        ha='center', fontsize=9, fontweight='bold')

# bottom: synthesis
ax.add_patch(Rectangle((1.5, 0.0), 7.0, 0.5, fc='#FFF7E0',
                        ec='#666600', lw=1.0))
ax.text(5.0, 0.25,
        'PCC-Event = relational reference INSIDE the event paradigm  '
        '→  fills the BiasBench gap [6]',
        ha='center', fontsize=10, fontweight='bold')

ax.set_title(
    'Structural isomorphism between the PCC change criterion and the '
    'event-emission rule', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_isomorphism.png', dpi=160,
            bbox_inches='tight')
plt.close()


# =========================================================================
# Numerical summary
# =========================================================================
print("\n=== NUMERICAL RESULTS (procedural synthetic stream) ===")
print(f"Total events                : {len(events):,}")
print(f"Sensor                      : {W} x {H} px")
print(f"Duration                    : {T_TOTAL*1000:.0f} ms")
print(f"Contrast threshold C        : {C}")
print(f"f_VG (windows discarded)    : {f_VG_estimate:.1%}")
print(f"|ROI_EF|/|Ω_act|            : {ratio:.1%}  (single window @ t=150 ms)")
print(f"Events on probe pixel       : {len(ev_px)}")
print(f"r_C min during dip          : {r_c_vals[(times_rc>=0.20)&(times_rc<=0.30)].min():.3f}")
print(f"r_C mean outside dip        : "
      f"{r_c_vals[(times_rc<0.20)|(times_rc>0.30)].mean():.3f}")

# save metrics for the paper
with open(f'{OUT}/metrics.txt', 'w') as f:
    f.write(f"Total events: {len(events)}\n")
    f.write(f"Sensor: {W}x{H}\n")
    f.write(f"Duration: {T_TOTAL*1000:.0f} ms\n")
    f.write(f"C: {C}\n")
    f.write(f"f_VG: {f_VG_estimate:.3f}\n")
    f.write(f"ROI_EF ratio: {ratio:.3f}\n")
    f.write(f"r_C min during dip: "
            f"{r_c_vals[(times_rc>=0.20)&(times_rc<=0.30)].min():.3f}\n")
    f.write(f"r_C mean outside dip: "
            f"{r_c_vals[(times_rc<0.20)|(times_rc>0.30)].mean():.3f}\n")

print("\nAll figures saved to", OUT)
