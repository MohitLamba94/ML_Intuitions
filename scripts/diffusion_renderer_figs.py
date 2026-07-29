"""Generate the figures used by diffusion_renderer.md.

All figures are synthesized from scratch (numpy + matplotlib) so nothing is a
verbatim copy of any blog/paper image. Where a concept needs a "render", we
shade an analytic sphere with a toy Cook-Torrance BRDF -- crude but faithful to
the intuition. Everything is written out as .jpg into assets/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

BG = "#f7f8fa"
INK = "#20242b"
MUTED = "#69707d"
BLUE = "#2878b5"
CYAN = "#55a9bd"
GREEN = "#4c956c"
AMBER = "#d99b2b"
RED = "#c9574f"
PURPLE = "#7b61a8"


def save(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=190, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
# Toy renderer helpers: an analytic sphere with a metallic/roughness BRDF.
# --------------------------------------------------------------------------- #
def sphere_geometry(res=320):
    """Return (mask, normals[H,W,3], view[3], y,x grid) for a unit sphere."""
    ys, xs = np.mgrid[-1:1:res * 1j, -1:1:res * 1j]
    r2 = xs ** 2 + ys ** 2
    mask = r2 <= 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    n = np.stack([xs, ys, z], axis=-1)  # points toward viewer (+z)
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n = n / norm
    return mask, n, xs, ys, z


def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-9)


def shade_sphere(base_color, roughness, metallic, res=320,
                 light_dir=(-0.5, 0.6, 0.8), light_color=(1.0, 1.0, 1.0),
                 ambient=0.06):
    """Very small GGX-ish shader, just for building intuition figures."""
    mask, n, xs, ys, z = sphere_geometry(res)
    L = _normalize(np.array(light_dir, float))
    V = np.array([0.0, 0.0, 1.0])
    H = _normalize(L + V)

    base = np.array(base_color, float)
    ndotl = np.clip(n @ L, 0.0, 1.0)
    ndoth = np.clip(n @ H, 0.0, 1.0)
    ndotv = np.clip(n @ V, 0.0, 1.0)

    a = max(roughness, 0.04) ** 2
    # GGX normal distribution term (drives the highlight tightness).
    denom = (ndoth ** 2 * (a ** 2 - 1.0) + 1.0)
    D = a ** 2 / (np.pi * denom ** 2 + 1e-9)
    # Schlick Fresnel; F0 is 0.04 for dielectrics, base color for metals.
    F0 = 0.04 * (1 - metallic) + metallic * base
    fres = F0[None, None, :] + (1 - F0)[None, None, :] * (1 - ndoth[..., None]) ** 5
    # Cheap smith-ish geometry attenuation.
    k = (roughness + 1) ** 2 / 8.0
    G = (ndotl / (ndotl * (1 - k) + k + 1e-9)) * (ndotv / (ndotv * (1 - k) + k + 1e-9))

    lc = np.array(light_color, float)
    spec = (D * G)[..., None] * fres * lc[None, None, :]
    spec = spec / (4 * ndotl[..., None] * ndotv[..., None] + 1e-3)

    kd = (1 - fres) * (1 - metallic)
    diffuse = kd * base[None, None, :] / np.pi * lc[None, None, :]

    col = (diffuse + spec) * ndotl[..., None]
    col = col + ambient * base[None, None, :]
    img = np.clip(col, 0.0, 1.0) ** (1 / 2.2)  # gamma

    out = np.ones((res, res, 3))
    out[~mask] = np.array([0.96, 0.965, 0.975])  # soft background
    out[mask] = img[mask]
    return out, mask, n, (xs, ys, z)


# --------------------------------------------------------------------------- #
# 1. Forward vs inverse rendering round trip
# --------------------------------------------------------------------------- #
def forward_inverse():
    fig, ax = plt.subplots(figsize=(12.5, 5.2), facecolor=BG)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                           facecolor=fc, edgecolor=INK, linewidth=1.6, mutation_scale=1)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="top",
                fontsize=13, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.9, lines, ha="center", va="top",
                fontsize=9.6, color=INK)

    # Scene description box (left)
    box(0.4, 1.2, 4.2, 3.0, "3D SCENE DESCRIPTION",
        "geometry (normals, depth)\n+ materials (albedo,\nroughness, metallic)\n+ lighting (env map)\n+ camera",
        "#e9f1f8")
    # Image box (right)
    box(7.9, 1.2, 4.2, 3.0, "2D IMAGE (RGB)",
        "what a camera\nactually records:\ncolor per pixel", "#fdf1e0")

    # Forward arrow (top)
    ar = FancyArrowPatch((4.7, 3.4), (7.8, 3.4), arrowstyle="-|>", mutation_scale=26,
                         linewidth=3, color=GREEN)
    ax.add_patch(ar)
    ax.text(6.25, 3.95, "FORWARD rendering", ha="center", fontsize=12.5,
            fontweight="bold", color=GREEN)
    ax.text(6.25, 3.55, "graphics:  scene → image", ha="center", fontsize=9.6, color=MUTED)

    # Inverse arrow (bottom)
    ar2 = FancyArrowPatch((7.8, 1.9), (4.7, 1.9), arrowstyle="-|>", mutation_scale=26,
                          linewidth=3, color=PURPLE)
    ax.add_patch(ar2)
    ax.text(6.25, 1.05, "INVERSE rendering", ha="center", fontsize=12.5,
            fontweight="bold", color=PURPLE)
    ax.text(6.25, 1.42, "vision:  image → scene", ha="center", fontsize=9.6, color=MUTED)

    ax.text(6.25, 4.85, "Rendering is a two-way street", ha="center",
            fontsize=15, fontweight="bold", color=INK)
    save(fig, "dr_forward_inverse.jpg")


# --------------------------------------------------------------------------- #
# 2. Surface normals: arrows + RGB encoding
# --------------------------------------------------------------------------- #
def surface_normals():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=BG)

    # (a) normals as arrows on a wavy surface
    ax = axes[0]
    ax.set_facecolor(BG)
    x = np.linspace(0, 2 * np.pi, 400)
    y = 0.6 * np.sin(x) + 2.0
    ax.plot(x, y, color=INK, lw=2.5)
    ax.fill_between(x, y, 0, color="#dfe6ee")
    xs = np.linspace(0.3, 2 * np.pi - 0.3, 9)
    ys = 0.6 * np.sin(xs) + 2.0
    dydx = 0.6 * np.cos(xs)
    nx, ny = -dydx, np.ones_like(xs)
    nn = np.hypot(nx, ny)
    nx, ny = nx / nn, ny / nn
    for xi, yi, nxi, nyi in zip(xs, ys, nx, ny):
        ax.add_patch(FancyArrowPatch((xi, yi), (xi + 0.55 * nxi, yi + 0.55 * nyi),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=RED, lw=2))
    ax.set_xlim(-0.2, 2 * np.pi + 0.2)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.set_title("Normal = unit vector\nperpendicular to the surface",
                 fontsize=12, color=INK)

    # (b) Lambert cosine: brightness = n . l
    ax = axes[1]
    ax.set_facecolor(BG)
    _, n, xs2, ys2, z2 = sphere_geometry(300)
    mask = xs2 ** 2 + ys2 ** 2 <= 1
    L = _normalize(np.array([-0.5, 0.55, 0.7]))
    shade = np.clip(n @ L, 0, 1)
    img = np.ones((300, 300))
    img[mask] = shade[mask]
    disp = np.dstack([img, img, img])
    disp[~mask] = [0.96, 0.965, 0.975]
    ax.imshow(disp, origin="lower")
    ax.axis("off")
    ax.set_title(r"Brightness $\propto \max(0,\ n\cdot l)$" + "\n(Lambert's cosine law)",
                 fontsize=12, color=INK)

    # (c) normal encoded as RGB
    ax = axes[2]
    ax.set_facecolor(BG)
    rgb = (n * 0.5 + 0.5)
    rgb_img = np.ones((300, 300, 3))
    rgb_img[mask] = rgb[mask]
    rgb_img[~mask] = [0.96, 0.965, 0.975]
    ax.imshow(rgb_img, origin="lower")
    ax.axis("off")
    ax.set_title("Same normals stored as RGB\n(x,y,z) → (R,G,B), the 'blue' normal map",
                 fontsize=12, color=INK)

    fig.suptitle("Surface normals: the direction a surface faces",
                 fontsize=15, fontweight="bold", color=INK, y=1.02)
    save(fig, "dr_surface_normals.jpg")


# --------------------------------------------------------------------------- #
# 3. Albedo vs fully shaded
# --------------------------------------------------------------------------- #
def albedo_vs_shaded():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG)
    base = (0.80, 0.32, 0.28)

    # flat albedo: no lighting, just the intrinsic color
    mask, n, xs, ys, z = sphere_geometry(320)
    flat = np.ones((320, 320, 3))
    flat[mask] = np.array(base) ** (1 / 2.2)
    flat[~mask] = [0.96, 0.965, 0.975]
    axes[0].imshow(flat, origin="lower")
    axes[0].axis("off")
    axes[0].set_title("ALBEDO / BASECOLOR\nintrinsic color, no lighting",
                      fontsize=12.5, fontweight="bold", color=INK)

    shaded, *_ = shade_sphere(base, roughness=0.35, metallic=0.0, res=320)
    axes[1].imshow(shaded, origin="lower")
    axes[1].axis("off")
    axes[1].set_title("SHADED RGB\n= albedo + lighting + shadow + highlight",
                      fontsize=12.5, fontweight="bold", color=INK)

    fig.suptitle("Albedo is what a surface looks like under flat white light "
                 "(lighting removed)", fontsize=13, color=MUTED, y=0.02)
    save(fig, "dr_albedo_vs_shaded.jpg")


# --------------------------------------------------------------------------- #
# 4. Material sphere grid: roughness (x) x metallic (y)
# --------------------------------------------------------------------------- #
def material_sphere_grid():
    roughs = [0.05, 0.25, 0.5, 0.8]
    metals = [0.0, 1.0]
    base = (0.62, 0.36, 0.30)  # coppery so metal reads well
    fig, axes = plt.subplots(len(metals), len(roughs),
                             figsize=(3.0 * len(roughs), 3.0 * len(metals)),
                             facecolor=BG)
    for i, m in enumerate(metals):
        for j, r in enumerate(roughs):
            img, *_ = shade_sphere(base, roughness=r, metallic=m, res=240)
            ax = axes[i, j]
            ax.imshow(img, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#c9ced6")
            if i == 0:
                ax.set_title(f"roughness = {r}", fontsize=11.5, color=INK)
            if j == 0:
                ax.set_ylabel("metallic = 0\n(dielectric)" if m == 0 else
                              "metallic = 1\n(metal)", fontsize=11.5, color=INK)
    fig.suptitle("Roughness (sharp → blurry highlight)  ×  Metallic "
                 "(colored diffuse → tinted mirror)",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.01)
    save(fig, "dr_material_sphere_grid.jpg")


# --------------------------------------------------------------------------- #
# 5. G-buffer montage: one scene decomposed into channels
# --------------------------------------------------------------------------- #
def gbuffer_montage():
    res = 260
    mask, n, xs, ys, z = sphere_geometry(res)
    base = (0.30, 0.55, 0.80)

    # final RGB
    rgb, *_ = shade_sphere(base, roughness=0.3, metallic=0.0, res=res)

    # basecolor
    bc = np.ones((res, res, 3)); bc[mask] = np.array(base) ** (1 / 2.2)
    bc[~mask] = [0.96, 0.965, 0.975]

    # normal (rgb)
    nm = np.ones((res, res, 3)); nm[mask] = (n * 0.5 + 0.5)[mask]
    nm[~mask] = [0.96, 0.965, 0.975]

    # depth: z from viewer, nearer = brighter
    dp = np.full((res, res), np.nan)
    dp[mask] = z[mask]
    depth_img = np.ones((res, res, 3)) * np.array([0.96, 0.965, 0.975])
    dd = np.zeros((res, res)); dd[mask] = (z[mask] - z[mask].min()) / (np.ptp(z[mask]) + 1e-9)
    for c in range(3):
        depth_img[..., c] = np.where(mask, dd, depth_img[..., c])

    # roughness map (constant here) and metallic map (constant 0)
    rough_val, metal_val = 0.3, 0.0
    rgh = np.ones((res, res, 3)) * np.array([0.96, 0.965, 0.975])
    for c in range(3):
        rgh[..., c] = np.where(mask, rough_val, rgh[..., c])
    mtl = np.ones((res, res, 3)) * np.array([0.96, 0.965, 0.975])
    for c in range(3):
        mtl[..., c] = np.where(mask, metal_val, mtl[..., c])

    panels = [(rgb, "final RGB\n(what we want)", True),
              (bc, "basecolor / albedo\n(3 ch)", False),
              (nm, "normal\n(3 ch)", False),
              (depth_img, "depth\n(1 ch)", False),
              (rgh, "roughness = 0.3\n(1 ch)", False),
              (mtl, "metallic = 0\n(1 ch)", False)]

    fig, axes = plt.subplots(1, 6, figsize=(16.5, 3.2), facecolor=BG)
    for ax, (img, title, hero) in zip(axes, panels):
        ax.imshow(img, origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        color = GREEN if hero else INK
        weight = "bold" if hero else "normal"
        ax.set_title(title, fontsize=11, color=color, fontweight=weight)
        for sp in ax.spines.values():
            sp.set_color(GREEN if hero else "#c9ced6")
            sp.set_linewidth(2.4 if hero else 1.0)
    fig.suptitle("A G-buffer: the final image decomposed into per-pixel intrinsic "
                 "channels (the 9-number interface both papers use)",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.12)
    save(fig, "dr_gbuffer_montage.jpg")


# --------------------------------------------------------------------------- #
# 6. Environment-map lighting: panorama -> lit sphere, LDR vs HDR
# --------------------------------------------------------------------------- #
def envmap_lighting():
    # synthesize a toy equirectangular panorama: sky gradient + bright sun + ground
    H, W = 120, 240
    vv, uu = np.mgrid[0:1:H * 1j, 0:1:W * 1j]   # v: 0 top..1 bottom, u: 0..1
    sky = np.stack([0.35 + 0.4 * (1 - vv),
                    0.55 + 0.3 * (1 - vv),
                    0.85 + 0.1 * (1 - vv)], axis=-1)
    ground = vv > 0.62
    sky[ground] = np.array([0.28, 0.24, 0.2])
    # a bright sun (HDR peak)
    su, sv = 0.28, 0.28
    dist = ((uu - su) * 2) ** 2 + ((vv - sv) * 3) ** 2
    sun = np.exp(-dist * 40)
    hdr = sky + sun[..., None] * np.array([6.0, 5.6, 4.6])  # >1 = HDR

    ldr = np.clip(hdr, 0, 1) ** (1 / 2.2)
    logv = np.log1p(hdr) / np.log1p(hdr.max())

    fig = plt.figure(figsize=(13.5, 5.6), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.4, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, :2])
    ax0.imshow(ldr, origin="upper", aspect="auto")
    ax0.axis("off")
    ax0.set_title("Environment map = equirectangular HDRI panorama "
                  "(all incoming light, 360°)", fontsize=12, color=INK)

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.imshow(logv, origin="upper", aspect="auto", cmap=None)
    ax1.axis("off")
    ax1.set_title("log-encoded\n(reveals HDR sun peak)", fontsize=10.5, color=MUTED)

    # light the sphere from the sun direction
    ax2 = fig.add_subplot(gs[1, 0])
    img, *_ = shade_sphere((0.8, 0.8, 0.82), roughness=0.5, metallic=0.0,
                           res=240, light_dir=(-0.5, 0.6, 0.7),
                           light_color=(1.0, 0.95, 0.85), ambient=0.12)
    ax2.imshow(img, origin="lower"); ax2.axis("off")
    ax2.set_title("diffuse object lit by it", fontsize=11, color=INK)

    ax3 = fig.add_subplot(gs[1, 1])
    img2, *_ = shade_sphere((0.95, 0.64, 0.32), roughness=0.12, metallic=1.0,
                            res=240, light_dir=(-0.5, 0.6, 0.7),
                            light_color=(1.0, 0.95, 0.85), ambient=0.05)
    ax3.imshow(img2, origin="lower"); ax3.axis("off")
    ax3.set_title("shiny metal reflects it", fontsize=11, color=INK)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    ax4.text(0.02, 0.9, "Why HDR matters:", fontsize=11.5, fontweight="bold",
             color=INK, va="top")
    ax4.text(0.02, 0.66,
             "The sun is thousands of\ntimes brighter than the sky.\n"
             "LDR (0–1) clips that peak;\nHDR keeps it, so highlights\n"
             "and reflections look right.\n\n"
             "DiffusionRenderer feeds the\nmodel LDR + log + direction\nmaps of the same panorama.",
             fontsize=10, color=MUTED, va="top")

    fig.suptitle("Image-based lighting: the whole environment is the light source",
                 fontsize=14, fontweight="bold", color=INK, y=0.99)
    save(fig, "dr_envmap_lighting.jpg")


# --------------------------------------------------------------------------- #
# 7. DiffusionRenderer pipeline
# --------------------------------------------------------------------------- #
def _rbox(ax, x, y, w, h, title, sub, fc, tcolor=INK):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                       facecolor=fc, edgecolor=INK, linewidth=1.5)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
            fontsize=11.5, fontweight="bold", color=tcolor)
    if sub:
        ax.text(x + w / 2, y + h - 0.62, sub, ha="center", va="top",
                fontsize=9, color=tcolor)


def _arrow(ax, p0, p1, color=INK, text=None, ty=0.15):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=20,
                                 lw=2.2, color=color))
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + ty, text, ha="center", fontsize=8.8, color=color)


def diffusionrenderer_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 6.0), facecolor=BG)
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.0); ax.axis("off")

    _rbox(ax, 0.3, 4.0, 2.3, 1.3, "Input video", "RGB frames\n(real capture)", "#fdf1e0")
    _rbox(ax, 3.6, 4.0, 3.0, 1.3, "INVERSE model", "video diffusion,\nRGB → G-buffers\n(1 attribute / pass)", "#efe6f4")
    _rbox(ax, 7.9, 3.7, 2.6, 1.9, "G-buffers", "basecolor, normal,\ndepth, roughness,\nmetallic", "#e9f1f8")

    _arrow(ax, (2.6, 4.65), (3.6, 4.65))
    _arrow(ax, (6.6, 4.65), (7.9, 4.65))

    # editing / new env map
    _rbox(ax, 7.9, 1.0, 2.6, 1.6, "Edit + new light", "swap env map,\nedit material/\ninsert object", "#fdeaea")
    _arrow(ax, (9.2, 3.7), (9.2, 2.6), color=RED)

    _rbox(ax, 3.6, 0.9, 3.0, 1.4, "FORWARD model", "video diffusion,\nG-buffers + env map → RGB", "#e6f2ec")
    _arrow(ax, (7.9, 1.8), (6.6, 1.8), color=GREEN)

    _rbox(ax, 0.3, 0.9, 2.3, 1.4, "Relit video", "photoreal RGB\nunder new light", "#fdf1e0")
    _arrow(ax, (3.6, 1.6), (2.6, 1.6), color=GREEN)

    ax.text(0.3, 5.7, "DiffusionRenderer: one video-diffusion backbone, two directions",
            fontsize=14, fontweight="bold", color=INK)
    ax.text(5.5, 3.15, "env map: LDR + log + direction maps →\ncross-attention into the UNet",
            fontsize=8.8, color=MUTED, ha="center")
    save(fig, "dr_diffusionrenderer_pipeline.jpg")


# --------------------------------------------------------------------------- #
# 8. FrameDiffuser autoregressive loop
# --------------------------------------------------------------------------- #
def framediffuser_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 5.6), facecolor=BG)
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 5.6); ax.axis("off")

    _rbox(ax, 0.3, 3.4, 2.7, 1.7, "G-buffer @ frame t", "basecolor, normal,\ndepth, roughness,\nmetallic (from engine)", "#e9f1f8")
    _rbox(ax, 0.3, 0.7, 2.7, 1.5, "Prev frame  F(t-1)\n+ irradiance", "model's own last\noutput (VAE-encoded)", "#fdf1e0")

    _rbox(ax, 4.0, 2.4, 3.1, 1.9, "SD 1.5 UNet", "ControlNet ← G-buffers\nControlLoRA ← prev frame\n10 denoise steps", "#efe6f4")

    _rbox(ax, 8.6, 2.6, 2.7, 1.6, "Frame  F(t)", "photoreal RGB\n(this frame only)", "#e6f2ec")

    _arrow(ax, (3.0, 4.0), (4.0, 3.7), color=BLUE, text="ControlNet")
    _arrow(ax, (3.0, 1.5), (4.0, 2.7), color=AMBER, text="ControlLoRA")
    _arrow(ax, (7.1, 3.35), (8.6, 3.35), color=GREEN)

    # feedback loop: route it below all the boxes so it never crosses text
    ax.plot([9.9, 9.9], [2.6, 0.35], color=PURPLE, lw=2.2)
    ax.plot([9.9, 1.65], [0.35, 0.35], color=PURPLE, lw=2.2)
    ax.add_patch(FancyArrowPatch((1.65, 0.35), (1.65, 0.68), arrowstyle="-|>",
                                 mutation_scale=20, lw=2.2, color=PURPLE))
    ax.text(5.7, 0.12, "feed F(t) back as prev frame for t+1  (autoregressive)",
            ha="center", fontsize=9.5, color=PURPLE)

    ax.text(0.3, 5.25, "FrameDiffuser: causal, one frame at a time (built for interactive/game rendering)",
            fontsize=13.5, fontweight="bold", color=INK)
    save(fig, "dr_framediffuser_pipeline.jpg")


if __name__ == "__main__":
    forward_inverse()
    surface_normals()
    albedo_vs_shaded()
    material_sphere_grid()
    gbuffer_montage()
    envmap_lighting()
    diffusionrenderer_pipeline()
    framediffuser_pipeline()
    print("done")
