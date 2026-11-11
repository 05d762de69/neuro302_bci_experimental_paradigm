"""61-channel EEG montage layout (10-5 system).

Ground: AFz. Reference: right earlobe (A2).
Electrode positions are computed on an ideal spherical head following the 10-5
construction (Oostenveld & Praamstra, 2001) and drawn with an azimuthal
equidistant projection, so the Fpz-T7-Oz-T8 circumference is the head outline.

Edit ROWS to change the montage; run `python make_montage.py`.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ montage
ROWS = [
    ['F3', 'F1', 'Fz', 'F2', 'F4'],
    ['FFC5h', 'FFC3h', 'FFC1h', 'FFC2h', 'FFC4h', 'FFC6h'],
    ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6'],
    ['FTT7h', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h', 'FCC6h', 'FTT8h'],
    ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
    ['TTP7h', 'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h', 'CCP4h', 'CCP6h', 'TTP8h'],
    ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
    ['CPP5h', 'CPP3h', 'CPP1h', 'CPP2h', 'CPP4h', 'CPP6h'],
    ['P3', 'P1', 'Pz', 'P2', 'P4'],
    ['PPO1h', 'PPO2h'],
]
EEG = [ch for row in ROWS for ch in row]
GROUND = 'AFz'
# standard 10-10 positions not recorded (drawn faintly for orientation)
UNUSED = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F6', 'F8',
          'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'P7', 'P5', 'P6', 'P8',
          'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
assert len(EEG) == 61 and len(set(EEG)) == 61 and GROUND not in EEG

# ------------------------------------------------------------------ 10-5 geometry on a unit sphere
CIRC = 72.0          # polar angle (deg from Cz) of the Fpz-T7-Oz-T8 circumference
# row -> (azimuth of its lateral circumference point, polar angle of its midline point; < 0 = posterior)
ROW_GEOM = {'AF': (36, 54), 'AFF': (45, 45), 'F': (54, 36), 'FFC': (63, 27), 'FC': (72, 18),
            'FCC': (81, 9), 'C': (90, 0), 'CCP': (99, -9), 'CP': (108, -18), 'CPP': (117, -27),
            'P': (126, -36), 'PPO': (135, -45), 'PO': (144, -54)}
PREFIX = {'Fp': 'Fp', 'AF': 'AF', 'AFF': 'AFF', 'F': 'F', 'FFC': 'FFC', 'FFT': 'FFC',
          'FC': 'FC', 'FT': 'FC', 'FCC': 'FCC', 'FTT': 'FCC', 'C': 'C', 'T': 'C',
          'CCP': 'CCP', 'TTP': 'CCP', 'CP': 'CP', 'TP': 'CP', 'CPP': 'CPP', 'TPP': 'CPP',
          'P': 'P', 'PPO': 'PPO', 'PO': 'PO', 'O': 'O'}
# steps (of 8) from the lateral circumference point towards the midline
COL = {'7': 0, '7h': 1, '5': 2, '5h': 3, '3': 4, '3h': 5, '1': 6, '1h': 7, 'z': 8,
       '8': 0, '8h': 1, '6': 2, '6h': 3, '4': 4, '4h': 5, '2': 6, '2h': 7}


def sph(theta, phi):
    t, p = np.radians(theta), np.radians(phi)
    return np.array([np.sin(t) * np.sin(p), np.sin(t) * np.cos(p), np.cos(t)])


def arc_point(L, M, R, frac):
    """Point at `frac` of the arc length from L to M on the circle through L, M, R."""
    n = np.cross(M - L, R - L); n /= np.linalg.norm(n)
    d = n @ L
    if d < 0:
        n, d = -n, -d
    c = d * n
    rho = np.sqrt(max(1 - d * d, 1e-12))
    u = (L - c) / rho
    w = np.cross(n, u)
    aM = np.arctan2((M - c) @ w, (M - c) @ u)
    a = frac * aM
    return c + rho * (np.cos(a) * u + np.sin(a) * w)


def pos3d(label):
    pre, col = re.match(r'^([A-Za-z]+?)(z|\d+h?)$', label).groups()
    row = PREFIX[pre]
    right = col != 'z' and int(col.rstrip('h')) % 2 == 0
    if row in ('Fp', 'O'):
        base = 0 if row == 'Fp' else 180
        off = {'z': 0, '1': -18, '2': 18}[col]
        return sph(CIRC, base + (off if row == 'Fp' else -off))
    a_lat, th_mid = ROW_GEOM[row]
    L, R = sph(CIRC, -a_lat), sph(CIRC, a_lat)
    M = sph(abs(th_mid), 0 if th_mid >= 0 else 180)
    p = arc_point(L, M, R, COL[col] / 8)
    if right:
        p[0] = -p[0]
    return p


def pos2d(label):
    x, y, z = pos3d(label)
    theta = np.degrees(np.arccos(np.clip(z, -1, 1)))
    rxy = np.hypot(x, y)
    r = theta / CIRC
    return (0.0, 0.0) if rxy < 1e-9 else (r * x / rxy, r * y / rxy)


# ------------------------------------------------------------------ drawing
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8983'
BLUE, BLUE_FILL = '#2a78d6', '#cde2fb'
ORANGE, ORANGE_EDGE = '#eb6834', '#a9461d'
AQUA, AQUA_EDGE = '#1baf7a', '#11724f'
GREY_EDGE = '#c3c2b7'
COND = FontProperties(fname='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf')
COND_B = FontProperties(fname='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})


def main(out='eeg_montage_61ch'):
    P = {ch: pos2d(ch) for ch in EEG + [GROUND] + UNUSED}
    pts = np.array([P[c] for c in EEG + [GROUND]])
    dmin = min(np.hypot(*(pts[i] - pts[j])) for i in range(len(pts)) for j in range(i))
    rad = 0.47 * dmin

    fig = plt.figure(figsize=(7.0, 7.25))
    ax = fig.add_axes([0.0, 0.085, 1.0, 0.915])
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_xlim(-1.38, 1.38); ax.set_ylim(-1.11, 1.20)

    # head, nose, ears
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.3, ec=INK2, zorder=1))
    ax.add_patch(Polygon([(-0.11, 0.994), (0, 1.135), (0.11, 0.994)], closed=False,
                         fill=False, lw=1.3, ec=INK2, joinstyle='round', zorder=1))
    t = np.linspace(-np.pi / 2, np.pi / 2, 100)
    for s in (-1, 1):
        ax.plot(s * (0.995 + 0.10 * np.cos(t)), -0.03 + 0.21 * np.sin(t), color=INK2, lw=1.3,
                solid_capstyle='round', zorder=1)
    ax.text(-1.2, 0.36, 'L', ha='center', va='center', fontsize=11, color=MUTED, fontweight='bold')
    ax.text(1.2, 0.36, 'R', ha='center', va='center', fontsize=11, color=MUTED, fontweight='bold')

    # unused 10-10 positions
    for ch in UNUSED:
        x, y = P[ch]
        ax.add_patch(Circle((x, y), rad, fc='white', ec=GREY_EDGE, lw=0.7, ls=(0, (2, 1.5)), zorder=2))
        ax.text(x, y, ch, ha='center', va='center', fontproperties=COND, fontsize=6.2, color=MUTED, zorder=3)

    # recorded EEG electrodes
    for ch in EEG:
        x, y = P[ch]
        ax.add_patch(Circle((x, y), rad, fc=BLUE_FILL, ec=BLUE, lw=0.9, zorder=2))
        ax.text(x, y, ch, ha='center', va='center', fontproperties=COND, fontsize=6.6, color=INK, zorder=3)

    # ground
    x, y = P[GROUND]
    ax.add_patch(Circle((x, y), rad, fc=ORANGE, ec=ORANGE_EDGE, lw=0.9, zorder=2))
    ax.text(x, y + 0.016, 'AFz', ha='center', va='center', fontproperties=COND_B, fontsize=6.2, color=INK, zorder=3)
    ax.text(x, y - 0.026, 'GND', ha='center', va='center', fontproperties=COND, fontsize=5.4, color=INK, zorder=3)

    # reference on the right earlobe
    ex, ey = 1.085, -0.175
    ax.add_patch(Circle((ex, ey), rad, fc=AQUA, ec=AQUA_EDGE, lw=0.9, zorder=4))
    ax.text(ex, ey, 'REF', ha='center', va='center', fontproperties=COND_B, fontsize=6.2, color=INK, zorder=5)
    ax.annotate('right earlobe', xy=(ex + 0.02, ey - rad), xytext=(1.19, -0.41), fontsize=7.5, color=INK2,
                ha='center', va='top',
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7, shrinkA=1, shrinkB=1))

    # legend
    def h(fc, ec, ls='-'):
        return Line2D([], [], ls='', marker='o', ms=11, mfc=fc, mec=ec, mew=0.9)
    handles = [h(BLUE_FILL, BLUE), h(ORANGE, ORANGE_EDGE), h(AQUA, AQUA_EDGE), h('white', GREY_EDGE)]
    labels = ['EEG electrode (n = 61, 10-5 positions)', 'GND: ground electrode at AFz',
              'REF: reference electrode, right earlobe', 'Standard 10-10 position, not recorded']
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.012), handletextpad=0.4, columnspacing=1.6, labelcolor=INK)

    for ext, kw in (('png', dict(dpi=600)), ('pdf', {})):
        fig.savefig(f'{out}.{ext}', facecolor='white', **kw)
    print('saved', out, '| electrode radius', round(rad, 4), '| min spacing', round(dmin, 4))


if __name__ == '__main__':
    main()
