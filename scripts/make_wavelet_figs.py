"""Generate figures for misc/wavelet_decomposition.md.

Outputs (all .jpg, per repo convention), written into ../assets:
  wavelet_tiling.jpg        -- time-frequency tiling: samples vs STFT vs wavelet
  wavelet_mother_scales.jpg -- one mother wavelet, dilated (scale) and translated
  wavelet_haar_1d.jpg       -- 1D Haar: signal -> average + detail, one and multi level
  wavelet_filterbank.jpg    -- Mallat cascade schematic (filters + downsampling)
  wavelet_2d_subbands.jpg   -- 2D DWT of a real image: LL/LH/HL/HH, one and two levels
  wavelet_denoise.jpg       -- coefficient thresholding removes noise
  wavelet_compression.jpg   -- reconstruct keeping only the largest k% of coefficients

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_wavelet_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import pywt
from skimage import data
from skimage.util import random_noise

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

C = {
    "blue":   "#6C8EBF",
    "purple": "#9673A6",
    "green":  "#82B366",
    "orange": "#D6795B",
    "red":    "#C0392B",
    "grey":   "#555555",
    "lgrey":  "#EDEDED",
    "yellow": "#E8B84B",
    "text":   "#222222",
}
DPI = 150
RNG = np.random.RandomState(0)


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", os.path.normpath(path))


# ---------------------------------------------------------------------------
# 1. Time-frequency tiling (Heisenberg boxes).
#    Left: raw samples -> perfect time, no frequency.
#    Middle: STFT -> a fixed grid, same box everywhere.
#    Right: wavelet -> tall thin boxes at high freq, short wide at low freq.
# ---------------------------------------------------------------------------
def fig_tiling():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # (a) samples: columns, full frequency height
    ax = axes[0]
    n = 8
    for i in range(n):
        ax.add_patch(Rectangle((i / n, 0), 1 / n, 1, facecolor=C["blue"],
                               edgecolor="white", alpha=0.55, lw=1.2))
    ax.set_title("(a) Raw samples\nexact time, no frequency", fontsize=11)

    # (b) STFT: uniform grid
    ax = axes[1]
    nt, nf = 4, 4
    for i in range(nt):
        for j in range(nf):
            ax.add_patch(Rectangle((i / nt, j / nf), 1 / nt, 1 / nf,
                                   facecolor=C["green"], edgecolor="white",
                                   alpha=0.55, lw=1.2))
    ax.set_title("(b) STFT / Fourier window\nsame box everywhere", fontsize=11)

    # (c) wavelet: dyadic tiling
    ax = axes[2]
    # frequency bands split dyadically: low band = 1 wide box, next = 2, then 4, then 8
    bands = [(0.0, 0.125, 1), (0.125, 0.25, 2), (0.25, 0.5, 4), (0.5, 1.0, 8)]
    cols = [C["purple"], C["blue"], C["orange"], C["red"]]
    for (y0, y1, ncols), col in zip(bands, cols):
        for k in range(ncols):
            ax.add_patch(Rectangle((k / ncols, y0), 1 / ncols, y1 - y0,
                                   facecolor=col, edgecolor="white",
                                   alpha=0.55, lw=1.2))
    ax.set_title("(c) Wavelet (multiresolution)\nwide+short low freq, thin+tall high freq",
                 fontsize=11)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("time / position", fontsize=10)
        ax.set_ylabel("frequency", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Each method carves the time-frequency plane differently "
                 "(every tile has the same area = uncertainty limit)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "wavelet_tiling.jpg")


# ---------------------------------------------------------------------------
# 2. One mother wavelet, dilated and translated.
# ---------------------------------------------------------------------------
def fig_mother_scales():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    t = np.linspace(-8, 8, 2000)

    def morlet(t, s=1.0, tau=0.0):
        z = (t - tau) / s
        return np.exp(-z**2 / 2) * np.cos(5 * z)

    # (a) mother wavelet
    ax = axes[0]
    ax.plot(t, morlet(t), color=C["blue"], lw=2)
    ax.axhline(0, color=C["grey"], lw=0.8)
    ax.set_title(r"(a) Mother wavelet $\psi(t)$" + "\nzero mean, localized bump", fontsize=11)

    # (b) scaling (dilation): small s = high freq, large s = low freq
    ax = axes[1]
    ax.plot(t, morlet(t, s=0.5), color=C["red"], lw=1.8, label="s=0.5 (squeezed narrow, high freq)")
    ax.plot(t, morlet(t, s=1.0), color=C["blue"], lw=1.8, label="s=1 (mother)")
    ax.plot(t, morlet(t, s=2.0), color=C["purple"], lw=1.8, label="s=2 (wide, low freq)")
    ax.axhline(0, color=C["grey"], lw=0.8)
    ax.set_title("(b) Scaling: stretch / squeeze\ntunes the frequency", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right")

    # (c) translation
    ax = axes[2]
    for tau, col in [(-4, C["green"]), (0, C["blue"]), (4, C["orange"])]:
        ax.plot(t, morlet(t, s=1.0, tau=tau), color=col, lw=1.8, label=fr"$\tau$={tau}")
    ax.axhline(0, color=C["grey"], lw=0.8)
    ax.set_title("(c) Translation: slide along\nselects the position", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

    for ax in axes:
        ax.set_xlabel("t", fontsize=10)
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "wavelet_mother_scales.jpg")


# ---------------------------------------------------------------------------
# 3. 1D Haar: average + detail.
# ---------------------------------------------------------------------------
def fig_haar_1d():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # a blocky-ish signal
    x = np.array([4, 6, 10, 12, 8, 6, 5, 5], dtype=float)
    idx = np.arange(len(x))

    ax = axes[0]
    ax.step(np.arange(len(x) + 1), np.append(x, x[-1]), where="post",
            color=C["blue"], lw=2)
    ax.set_title("(a) Signal (8 samples)", fontsize=11)
    ax.set_ylim(0, 14)

    # Haar: pairwise average and difference
    a = (x[0::2] + x[1::2]) / np.sqrt(2)   # approximation (scaled)
    d = (x[0::2] - x[1::2]) / np.sqrt(2)   # detail
    avg = (x[0::2] + x[1::2]) / 2          # plain average, easier to read
    dif = (x[0::2] - x[1::2]) / 2

    ax = axes[1]
    ax.step(np.arange(len(avg) + 1), np.append(avg, avg[-1]), where="post",
            color=C["green"], lw=2)
    ax.set_title("(b) Approximation\n(pairwise averages) - the blur", fontsize=11)
    ax.set_ylim(0, 14)

    ax = axes[2]
    ax.bar(np.arange(len(dif)), dif, color=C["orange"], width=0.6)
    ax.axhline(0, color=C["grey"], lw=0.8)
    ax.set_title("(c) Detail\n(pairwise differences) - the edges", fontsize=11)
    ax.set_ylim(-4, 4)

    for ax in axes:
        ax.set_xlabel("index", fontsize=10)
    fig.suptitle("One Haar step splits a signal into a coarse blur + fine detail "
                 "(half the samples each)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "wavelet_haar_1d.jpg")


# ---------------------------------------------------------------------------
# 4. Mallat cascade / filter bank schematic.
# ---------------------------------------------------------------------------
def fig_filterbank():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     facecolor=col, edgecolor=C["text"], lw=1.3, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9.5, color=C["text"])

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     arrowstyle="-|>", mutation_scale=14,
                     color=C["grey"], lw=1.5))

    # level 1
    box(0.2, 2.6, 1.4, 0.8, "signal\n$x$", C["lgrey"])
    # high pass branch (detail)
    box(2.6, 4.2, 1.5, 0.8, "high-pass $H$", C["orange"])
    box(4.6, 4.2, 1.2, 0.8, r"$\downarrow 2$", C["yellow"])
    box(6.4, 4.2, 1.4, 0.8, "detail $D_1$", C["red"])
    # low pass branch (approx)
    box(2.6, 1.0, 1.5, 0.8, "low-pass $L$", C["blue"])
    box(4.6, 1.0, 1.2, 0.8, r"$\downarrow 2$", C["yellow"])
    box(6.4, 1.0, 1.6, 0.8, "approx $A_1$", C["green"])

    arrow(1.6, 3.0, 2.6, 4.6)
    arrow(1.6, 3.0, 2.6, 1.4)
    arrow(4.1, 4.6, 4.6, 4.6)
    arrow(5.8, 4.6, 6.4, 4.6)
    arrow(4.1, 1.4, 4.6, 1.4)
    arrow(5.8, 1.4, 6.4, 1.4)

    # recurse on approximation
    box(8.6, 1.0, 3.0, 0.8, "repeat on $A_1$ ...", C["blue"])
    arrow(8.0, 1.4, 8.6, 1.4)
    ax.text(10.1, 2.3, "(only the approximation\nis split again)", ha="center",
            fontsize=8.5, color=C["grey"])

    ax.set_title("One level of the wavelet filter bank; the cascade repeats on the "
                 "low-pass output", fontsize=12)
    fig.tight_layout()
    save(fig, "wavelet_filterbank.jpg")


# ---------------------------------------------------------------------------
# helper: pretty subband display (log-magnitude, clipped) for details
# ---------------------------------------------------------------------------
def _norm_detail(band):
    b = np.abs(band)
    hi = np.percentile(b, 99.5) + 1e-8
    return np.clip(b / hi, 0, 1)


# ---------------------------------------------------------------------------
# 5. 2D DWT of a real image: one-level LL/LH/HL/HH, plus two-level mosaic.
# ---------------------------------------------------------------------------
def fig_2d_subbands():
    img = data.camera().astype(float) / 255.0

    # one level
    LL, (LH, HL, HH) = pywt.dwt2(img, "haar")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("original", fontsize=11)

    axes[0, 1].imshow(LL, cmap="gray")
    axes[0, 1].set_title("LL - approximation\n(half-size blur)", fontsize=11)

    axes[0, 2].imshow(_norm_detail(LH), cmap="magma")
    axes[0, 2].set_title("LH - horizontal edges", fontsize=11)

    axes[1, 0].imshow(_norm_detail(HL), cmap="magma")
    axes[1, 0].set_title("HL - vertical edges", fontsize=11)

    axes[1, 1].imshow(_norm_detail(HH), cmap="magma")
    axes[1, 1].set_title("HH - diagonal / corners", fontsize=11)

    # two-level mosaic in standard layout
    coeffs = pywt.wavedec2(img, "haar", level=2)
    arr, _ = pywt.coeffs_to_array(coeffs)
    disp = np.abs(arr)
    disp = np.clip(disp / (np.percentile(disp, 99.0) + 1e-8), 0, 1)
    # keep the top-left approximation bright/linear
    n = arr.shape[0] // 4
    disp[:n, :n] = (arr[:n, :n] - arr[:n, :n].min()) / (np.ptp(arr[:n, :n]) + 1e-8)
    axes[1, 2].imshow(disp, cmap="gray")
    axes[1, 2].set_title("two-level pyramid\n(standard subband layout)", fontsize=11)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("2D Haar wavelet transform: the image splits into a coarse blur "
                 "plus three directional detail bands", fontsize=12, y=0.98)
    fig.tight_layout()
    save(fig, "wavelet_2d_subbands.jpg")


# ---------------------------------------------------------------------------
# 6. Denoising by coefficient thresholding.
# ---------------------------------------------------------------------------
def _psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / (mse + 1e-12))


def fig_denoise():
    img = data.camera().astype(float) / 255.0
    noisy = np.clip(img + np.sqrt(0.02) * RNG.randn(*img.shape), 0, 1)

    coeffs = pywt.wavedec2(noisy, "db2", level=3)
    arr, slices = pywt.coeffs_to_array(coeffs)

    # universal-ish threshold from the finest detail band
    finest = coeffs[-1][0]
    sigma = np.median(np.abs(finest)) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(img.size))

    arr_t = np.sign(arr) * np.maximum(np.abs(arr) - thr, 0)  # soft threshold
    # do not threshold the coarse approximation block
    n = arr.shape[0]
    a0 = slices[0]
    arr_t[a0] = arr[a0]
    coeffs_t = pywt.array_to_coeffs(arr_t, slices, output_format="wavedec2")
    denoised = np.clip(pywt.waverec2(coeffs_t, "db2"), 0, 1)
    denoised = denoised[:img.shape[0], :img.shape[1]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("clean", fontsize=11)
    axes[1].imshow(noisy, cmap="gray")
    axes[1].set_title(f"noisy  (PSNR {_psnr(img, noisy):.1f} dB)", fontsize=11)
    axes[2].imshow(denoised, cmap="gray")
    axes[2].set_title(f"wavelet soft-threshold\n(PSNR {_psnr(img, denoised):.1f} dB)",
                      fontsize=11)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Denoising: shrink small detail coefficients toward zero, keep the "
                 "big ones (real edges)", fontsize=12, y=1.0)
    fig.tight_layout()
    save(fig, "wavelet_denoise.jpg")


# ---------------------------------------------------------------------------
# 7. Compression: keep only the largest k% of coefficients.
# ---------------------------------------------------------------------------
def fig_compression():
    img = data.camera().astype(float) / 255.0
    coeffs = pywt.wavedec2(img, "db2", level=4)
    arr, slices = pywt.coeffs_to_array(coeffs)
    flat = np.abs(arr).ravel()

    keeps = [1.0, 0.1, 0.02, 0.005]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, k in zip(axes, keeps):
        if k >= 1.0:
            kept = arr
        else:
            thr = np.percentile(flat, 100 * (1 - k))
            kept = np.where(np.abs(arr) >= thr, arr, 0.0)
        c2 = pywt.array_to_coeffs(kept, slices, output_format="wavedec2")
        rec = np.clip(pywt.waverec2(c2, "db2"), 0, 1)[:img.shape[0], :img.shape[1]]
        ax.imshow(rec, cmap="gray")
        ax.set_title(f"keep {k*100:g}% coeffs\nPSNR {_psnr(img, rec):.1f} dB", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Compression: most wavelet coefficients are ~0, so keeping only the "
                 "largest few still reconstructs the image", fontsize=12, y=1.0)
    fig.tight_layout()
    save(fig, "wavelet_compression.jpg")


if __name__ == "__main__":
    fig_tiling()
    fig_mother_scales()
    fig_haar_1d()
    fig_filterbank()
    fig_2d_subbands()
    fig_denoise()
    fig_compression()
    print("done")
