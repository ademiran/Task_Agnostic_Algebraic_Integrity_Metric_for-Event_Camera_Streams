"""
Optimized PCC-Event demo video.
600 frames @ 24 fps = 25 seconds total.
Reuses the same matplotlib figure (only updates artists) for ~5x speedup.
"""
import os, subprocess, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch

OUT = "/home/claude/paper/video"
FRAMES = f"{OUT}/frames"
os.makedirs(FRAMES, exist_ok=True)

H, W = 180, 240
DT_SIM = 1e-4
T_TOTAL = 0.5
FPS = 24
N_FRAMES = 600
DT_FRAME = T_TOTAL / N_FRAMES
C = 0.15
EF_WIN = 0.030
TS_DECAY = 0.030
VG_BINS = 5

np.random.seed(42)


def luminance_at(t):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    bg = 0.5 + 0.15*np.sin(xx*0.05) * np.cos(yy*0.04)
    cx = 30 + (W-60) * (t/T_TOTAL)
    cy = H/2 + 25*np.sin(2*np.pi*t*4)
    r = 18
    dist = (xx-cx)**2 + (yy-cy)**2
    disk = np.where(dist <= r*r, 0.85, 0.0)
    L = bg + disk
    L = np.clip(L, 0.05, 1.5)
    if 0.20 < t < 0.30:
        dip = 1.0 - 0.85*np.sin(np.pi*(t-0.20)/0.10)
        L = L*dip + 0.02
    L = L*(1.0 + 0.01*np.random.randn(H, W))
    return np.clip(L, 0.01, None)

def luminance_predicted(t):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    bg = 0.5 + 0.15*np.sin(xx*0.05) * np.cos(yy*0.04)
    cx = 30 + (W-60) * (t/T_TOTAL)
    cy = H/2 + 25*np.sin(2*np.pi*t*4)
    r = 18
    dist = (xx-cx)**2 + (yy-cy)**2
    disk = np.where(dist <= r*r, 0.85, 0.0)
    return np.clip(bg + disk, 0.05, 1.5)


def gen_stream(lum_fn):
    L0 = lum_fn(0.0)
    log_ref = np.log(L0)
    n_ticks = int(T_TOTAL / DT_SIM)
    events = []
    for k in range(1, n_ticks):
        t = k*DT_SIM
        log_L = np.log(lum_fn(t))
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
    return np.array(events, dtype=[('t','f8'),('x','i4'),('y','i4'),('p','i1')])

print("Generating event streams ...")
events_obs = gen_stream(luminance_at)
events_pred = gen_stream(luminance_predicted)
print(f"  observed: {len(events_obs)} events")
print(f"  predicted: {len(events_pred)} events")


def time_surface_fast(events, t_now, decay=TS_DECAY):
    tau = -np.inf*np.ones((H, W))
    mask = events['t'] <= t_now
    sub = events[mask]
    if len(sub) > 0:
        tau[sub['y'], sub['x']] = sub['t']
    age = t_now - tau
    age[~np.isfinite(age)] = 1e6
    return np.exp(-age/decay)

def event_frame_fast(events, t0, T):
    EF = np.zeros((H, W), dtype=np.int32)
    mask = (events['t'] >= t0) & (events['t'] < t0+T)
    sub = events[mask]
    if len(sub) > 0:
        np.add.at(EF, (sub['y'], sub['x']), sub['p'].astype(np.int32))
    return EF

def voxel_grid_flat(events, t0, T, B=VG_BINS):
    VG = np.zeros((B, H, W), dtype=np.float32)
    mask = (events['t'] >= t0) & (events['t'] < t0+T)
    sub = events[mask]
    if len(sub) == 0: return VG
    norm_t = (sub['t']-t0)/T*(B-1)
    b_lo = np.floor(norm_t).astype(int)
    b_hi = np.clip(b_lo+1, 0, B-1)
    w_hi = norm_t-b_lo; w_lo = 1.0-w_hi
    np.add.at(VG, (b_lo, sub['y'], sub['x']), sub['p']*w_lo)
    np.add.at(VG, (b_hi, sub['y'], sub['x']), sub['p']*w_hi)
    return VG

def pcc(a, b):
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    da = np.sqrt((a*a).sum()); db = np.sqrt((b*b).sum())
    if da<1e-12 or db<1e-12: return 0.0
    return float((a*b).sum()/(da*db))


print("Pre-computing rolling curves ...")
curve_t = np.arange(0.030, T_TOTAL, 0.005)
curve_rC, curve_rVG = [], []
for tn in curve_t:
    TS_o = time_surface_fast(events_obs, tn)
    TS_p = time_surface_fast(events_pred, tn)
    curve_rC.append(pcc(TS_o, TS_p))
    if tn+0.020 < T_TOTAL:
        VGa = voxel_grid_flat(events_obs, tn, 0.020)
        VGb = voxel_grid_flat(events_obs, tn+0.020, 0.020)
        curve_rVG.append(pcc(VGa, VGb))
    else:
        curve_rVG.append(np.nan)
curve_rC = np.array(curve_rC); curve_rVG = np.array(curve_rVG)
print("  done")


# build figure ONCE
plt.rcParams['font.family'] = 'DejaVu Sans'
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor('#0E1A2B')

gs = gridspec.GridSpec(3, 4, figure=fig,
                       height_ratios=[0.18, 1.0, 1.0],
                       width_ratios=[1, 1, 1, 1],
                       hspace=0.35, wspace=0.30,
                       left=0.05, right=0.97, top=0.95, bottom=0.07)

ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off'); ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
ax_title.text(0.5, 0.7, 'PCC-Event  -  Procedural-Synthetic Demonstration',
              ha='center', va='center', fontsize=20, fontweight='bold', color='#FFFFFF')
title_status = ax_title.text(0.5, 0.20, '', ha='center', va='center',
                              fontsize=11, color='#A0C4E8', family='monospace')
title_dip_box = FancyBboxPatch((0.78, 0.05), 0.20, 0.85,
                               boxstyle="round,pad=0.02", linewidth=0,
                               facecolor='#FF4136', alpha=0.85, visible=False)
ax_title.add_patch(title_dip_box)
title_dip_text = ax_title.text(0.88, 0.48, 'TUNNEL DIP', ha='center', va='center',
                                fontsize=13, fontweight='bold', color='white', visible=False)

ax_scene = fig.add_subplot(gs[1, 0])
img_scene = ax_scene.imshow(np.zeros((H, W)), cmap='inferno', vmin=0, vmax=1.5,
                             interpolation='nearest')
ax_scene.set_title('Synthetic scene  L(x,y,t)', color='white', fontsize=11, pad=8)
ax_scene.axis('off')

ax_ts = fig.add_subplot(gs[1, 1])
img_ts = ax_ts.imshow(np.zeros((H, W)), cmap='hot', vmin=0, vmax=1,
                       interpolation='nearest')
ax_ts.set_title(f'Time Surface  (Eq. 2,  d={TS_DECAY*1000:.0f} ms)',
                color='white', fontsize=11, pad=8)
ax_ts.axis('off')

ax_ef = fig.add_subplot(gs[1, 2])
img_ef = ax_ef.imshow(np.zeros((H, W)), cmap='RdBu_r', vmin=-3, vmax=3,
                       interpolation='nearest')
title_ef = ax_ef.set_title('', color='white', fontsize=11, pad=8)
ax_ef.axis('off')

ax_roi = fig.add_subplot(gs[1, 3])
img_roi_bg = ax_roi.imshow(np.zeros((H, W)), cmap='gray', vmin=0, vmax=1.5, alpha=0.6,
                            interpolation='nearest')
img_roi_overlay = ax_roi.imshow(np.zeros((H, W, 4)), interpolation='nearest')
title_roi = ax_roi.set_title('', color='white', fontsize=11, pad=8)
ax_roi.axis('off')

ax_rc = fig.add_subplot(gs[2, 0:2])
ax_rc.set_facecolor('#1A2840')
ax_rc.axvspan(200, 300, color='#FF6B35', alpha=0.18)
line_rc, = ax_rc.plot([], [], color='#FFD700', lw=2.5)
scatter_rc = ax_rc.scatter([], [], s=120, color='#FFD700', ec='white', lw=1.5, zorder=5)
ax_rc.axhline(0.4, color='#FF4136', ls='--', lw=1, alpha=0.8)
ax_rc.axhline(0.7, color='#2ECC40', ls='--', lw=1, alpha=0.8)
ax_rc.text(498, 0.42, 'theta_low', color='#FF4136', fontsize=9, ha='right', va='bottom')
ax_rc.text(498, 0.72, 'theta_high', color='#2ECC40', fontsize=9, ha='right', va='bottom')
ax_rc.set_xlim(0, 500); ax_rc.set_ylim(-1.05, 1.10)
ax_rc.set_xlabel('t  [ms]', color='white', fontsize=10)
ax_rc.set_ylabel('r_C(t)', color='white', fontsize=10)
ax_rc.tick_params(colors='white', labelsize=9)
for s in ax_rc.spines.values(): s.set_color('#A0C4E8')
ax_rc.grid(alpha=0.2, color='white')
title_rc = ax_rc.set_title('', color='white', fontsize=12, pad=8, fontweight='bold')
alarm_rc = ax_rc.text(0.5, 0.92, '', transform=ax_rc.transAxes, ha='center',
                       fontsize=14, fontweight='bold', color='#FF4136', visible=False,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#1A2840',
                                 edgecolor='#FF4136', lw=2))

ax_vg = fig.add_subplot(gs[2, 2:4])
ax_vg.set_facecolor('#1A2840')
ax_vg.axvspan(200, 300, color='#FF6B35', alpha=0.18)
line_vg, = ax_vg.plot([], [], color='#00D4FF', lw=2.5)
scatter_vg = ax_vg.scatter([], [], s=120, color='#00D4FF', ec='white', lw=1.5, zorder=5)
ax_vg.axhline(0.5, color='#FF4136', ls='--', lw=1, alpha=0.8)
ax_vg.text(498, 0.52, 'theta', color='#FF4136', fontsize=9, ha='right', va='bottom')
ax_vg.set_xlim(0, 500); ax_vg.set_ylim(-1.05, 1.10)
ax_vg.set_xlabel('t  [ms]', color='white', fontsize=10)
ax_vg.set_ylabel('r_VG', color='white', fontsize=10)
ax_vg.tick_params(colors='white', labelsize=9)
for s in ax_vg.spines.values(): s.set_color('#A0C4E8')
ax_vg.grid(alpha=0.2, color='white')
title_vg = ax_vg.set_title('', color='white', fontsize=12, pad=8, fontweight='bold')
status_vg = ax_vg.text(0.5, 0.92, '', transform=ax_vg.transAxes, ha='center',
                        fontsize=11, fontweight='bold', color='#2ECC40',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A2840',
                                  edgecolor='#2ECC40', lw=1.5))

print(f"Rendering {N_FRAMES} frames ...")
t0 = time.time()
for i in range(N_FRAMES):
    t = i * DT_FRAME
    in_dip = 0.20 <= t <= 0.30

    L = luminance_at(t)
    TS = time_surface_fast(events_obs, t, TS_DECAY)
    EF = event_frame_fast(events_obs, max(0, t-EF_WIN), EF_WIN)

    active = EF != 0
    if active.sum() > 0:
        mu_EF = EF[active].mean()
        r2 = np.zeros_like(EF, dtype=np.int8)
        r2[active] = np.sign(EF[active] - mu_EF).astype(np.int8)
        roi_mask = (r2 == -1) & active
        roi_ratio = roi_mask.sum() / max(1, active.sum())
    else:
        roi_mask = np.zeros_like(EF, dtype=bool)
        roi_ratio = 0.0

    r_C_now = float(np.interp(t, curve_t, curve_rC))
    r_VG_now = float(np.interp(t, curve_t, curve_rVG))
    n_events_so_far = int((events_obs['t'] <= t).sum())

    title_status.set_text(
        f't = {t*1000:6.1f} ms     |     C = {C}    |    sensor 240x180    '
        f'|    {n_events_so_far:,} events emitted')
    title_dip_box.set_visible(in_dip)
    title_dip_text.set_visible(in_dip)

    img_scene.set_data(L)
    img_ts.set_data(TS)
    vmax = max(1, np.abs(EF).max())
    img_ef.set_data(EF); img_ef.set_clim(-vmax, vmax)
    title_ef.set_text(f'Event Frame  (Eq. 4)  -  {active.sum()} active px')

    img_roi_bg.set_data(L)
    overlay = np.zeros((H, W, 4))
    overlay[roi_mask] = [1, 0.2, 0.1, 0.85]
    img_roi_overlay.set_data(overlay)
    title_roi.set_text(f'ROI_EF  (Eq. 11)  -  {roi_ratio:.0%} of |Omega_act|')

    mask_hist = curve_t*1000 <= t*1000
    line_rc.set_data(curve_t[mask_hist]*1000, curve_rC[mask_hist])
    scatter_rc.set_offsets([[t*1000, r_C_now]])
    title_rc.set_text(f'PCC-TS integrity monitor   -   r_C = {r_C_now:+.2f}')
    if r_C_now < 0.4:
        alarm_rc.set_text('INTEGRITY ALARM')
        alarm_rc.set_visible(True)
    else:
        alarm_rc.set_visible(False)

    line_vg.set_data(curve_t[mask_hist]*1000, curve_rVG[mask_hist])
    scatter_vg.set_offsets([[t*1000, r_VG_now]])
    title_vg.set_text(f'PCC-VG temporal gating   -   r_VG = {r_VG_now:+.2f}')
    if r_VG_now < 0.5:
        status_vg.set_text('PIPELINE TRIGGERED')
        status_vg.set_color('#FF4136')
        status_vg.get_bbox_patch().set_edgecolor('#FF4136')
    else:
        status_vg.set_text('PIPELINE GATED (redundant)')
        status_vg.set_color('#2ECC40')
        status_vg.get_bbox_patch().set_edgecolor('#2ECC40')

    fig.savefig(f'{FRAMES}/frame_{i:05d}.png', dpi=100,
                facecolor=fig.get_facecolor())

    if (i+1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (i+1)/elapsed
        eta = (N_FRAMES-i-1)/rate
        print(f"  {i+1}/{N_FRAMES}  ({rate:.1f} fps render, ETA {eta:.0f}s)")

plt.close(fig)
print(f"All frames rendered in {time.time()-t0:.0f} s")

print("Encoding MP4 ...")
cmd = ['ffmpeg', '-y', '-framerate', str(FPS),
       '-i', f'{FRAMES}/frame_%05d.png',
       '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
       '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
       f'{OUT}/PCC-Event_demo.mp4']
subprocess.run(cmd, check=True, capture_output=True)
print("MP4:", os.path.getsize(f'{OUT}/PCC-Event_demo.mp4'), "bytes")
