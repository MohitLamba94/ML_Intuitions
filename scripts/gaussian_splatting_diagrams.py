"""Generate the conceptual diagrams used by gaussian-splatting.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle


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


def base_figure(width=12, height=5):
    fig = plt.figure(figsize=(width, height), facecolor=BG)
    return fig


def representation_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=BG)
    titles = ["Triangle mesh", "Neural radiance field", "3D Gaussian splats"]
    subtitles = ["Explicit surface", "Implicit function", "Explicit soft primitives"]

    for ax, title, subtitle in zip(axes, titles, subtitles):
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, color=INK, fontsize=15, fontweight="bold", pad=20)
        ax.text(0.5, 1.01, subtitle, ha="center", va="bottom", color=MUTED, fontsize=10)

    # A faceted surface: geometry first, then texture and lighting.
    ax = axes[0]
    pts = np.array([[0.13, 0.23], [0.28, 0.72], [0.50, 0.86], [0.76, 0.68],
                    [0.88, 0.28], [0.56, 0.15]])
    center = np.array([0.50, 0.48])
    colors = ["#78a8c7", "#5f98bd", "#79b7a0", "#d8ad58", "#cb765f", "#9b83b8"]
    for i in range(len(pts)):
        tri = Polygon([center, pts[i], pts[(i + 1) % len(pts)]], closed=True,
                      facecolor=colors[i], edgecolor="white", linewidth=1.8)
        ax.add_patch(tri)
    for p in pts:
        ax.plot([center[0], p[0]], [center[1], p[1]], color="#324756", lw=0.6, alpha=0.55)
    ax.text(0.5, 0.04, "Clean topology; easy to edit and relight", ha="center", color=MUTED, fontsize=9)

    # A coordinate/view direction enters an opaque network and returns density/color.
    ax = axes[1]
    ax.text(0.10, 0.63, "position x\nview d", ha="center", va="center", fontsize=10, color=INK)
    net_x = [0.35, 0.48, 0.61]
    layers = [[0.34, 0.50, 0.66], [0.29, 0.43, 0.57, 0.71], [0.34, 0.50, 0.66]]
    prev = [(0.20, 0.50)]
    for x, ys in zip(net_x, layers):
        curr = [(x, y) for y in ys]
        for p0 in prev:
            for p1 in curr:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#a5aab2", lw=0.7, zorder=1)
        for _, y in curr:
            ax.add_patch(Circle((x, y), 0.025, facecolor=PURPLE, edgecolor="white", lw=0.8, zorder=2))
        prev = curr
    for p0 in prev:
        ax.plot([p0[0], 0.79], [p0[1], 0.50], color="#a5aab2", lw=0.7)
    ax.add_patch(FancyArrowPatch((0.13, 0.53), (0.28, 0.53), arrowstyle="-|>", mutation_scale=12,
                                 color=INK, lw=1.2))
    ax.text(0.87, 0.54, "color c\ndensity sigma", ha="center", va="center", fontsize=10, color=INK)
    ax.text(0.5, 0.04, "Compact, but every pixel needs network queries", ha="center", color=MUTED, fontsize=9)

    # Soft ellipses show anisotropic, overlapping primitives.
    ax = axes[2]
    rng = np.random.default_rng(4)
    centers = rng.normal([0.52, 0.50], [0.17, 0.14], size=(28, 2))
    palette = [BLUE, CYAN, GREEN, AMBER, RED]
    for i, (x, y) in enumerate(centers):
        w = rng.uniform(0.07, 0.20)
        h = rng.uniform(0.025, 0.075)
        angle = rng.uniform(-70, 70)
        ax.add_patch(Ellipse((x, y), w, h, angle=angle, facecolor=palette[i % len(palette)],
                             edgecolor="none", alpha=0.34))
    ax.scatter(centers[:, 0], centers[:, 1], s=7, color=INK, alpha=0.55)
    ax.text(0.5, 0.04, "Fast to rasterize; appearance is directly optimized", ha="center", color=MUTED, fontsize=9)

    fig.subplots_adjust(wspace=0.12, left=0.02, right=0.98, top=0.84, bottom=0.10)
    save(fig, "gaussian_scene_representations.jpg")


def training_pipeline():
    fig = base_figure(14, 5.2)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, title, detail, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=2))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", color=INK,
                fontsize=10.5, fontweight="bold", linespacing=1.1)
        ax.text(x + w / 2, y + 0.32, detail, ha="center", va="bottom", color=MUTED, fontsize=9,
                linespacing=1.4)

    box(0.25, 2.05, 2.2, 1.55, "1. Capture", "Overlapping photos\nof a static scene", BLUE)
    box(3.05, 2.05, 2.2, 1.55, "2. Structure\nfrom Motion", "Camera poses +\nsparse point cloud", CYAN)
    box(5.85, 2.05, 2.2, 1.55, "3. Initialize splats", "One Gaussian near\neach SfM point", GREEN)
    box(8.65, 2.05, 2.2, 1.55, "4. Render a view", "Project, sort, and\nalpha-composite", AMBER)
    box(11.45, 2.05, 2.2, 1.55, "5. Compare", "Rendered image vs.\nknown photograph", RED)

    for x0, x1 in [(2.45, 3.05), (5.25, 5.85), (8.05, 8.65), (10.85, 11.45)]:
        ax.add_patch(FancyArrowPatch((x0, 2.83), (x1, 2.83), arrowstyle="-|>", mutation_scale=14,
                                     color=INK, lw=1.5))

    # Feedback loop.
    ax.add_patch(FancyArrowPatch((12.55, 1.97), (7.0, 0.72), connectionstyle="arc3,rad=-0.18",
                                 arrowstyle="-|>", mutation_scale=15, color=PURPLE, lw=2.2))
    ax.add_patch(FancyArrowPatch((7.0, 0.72), (6.95, 1.98), arrowstyle="-|>", mutation_scale=15,
                                 color=PURPLE, lw=2.2))
    ax.text(9.65, 0.47, "Backpropagate image error", ha="center", color=PURPLE,
            fontsize=11, fontweight="bold")
    ax.text(9.65, 0.16, "move, reshape, recolor, clone, split, or prune Gaussians", ha="center",
            color=MUTED, fontsize=9)

    ax.text(7, 4.62, "Training turns photographs into an explicit renderable scene",
            ha="center", color=INK, fontsize=17, fontweight="bold")
    ax.text(7, 4.22, "The renderer is differentiable, so pixel error can update the scene itself",
            ha="center", color=MUTED, fontsize=11)
    save(fig, "gaussian_training_pipeline.jpg")


def projection_and_compositing():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("A 3D Gaussian becomes a 2D footprint", color=INK, fontsize=15, fontweight="bold", pad=14)
    ax.add_patch(Polygon([[0.08, 0.22], [0.08, 0.78], [0.49, 0.66], [0.49, 0.34]], closed=True,
                         facecolor="#dceaf2", edgecolor=BLUE, lw=1.4, alpha=0.7))
    ax.add_patch(Ellipse((0.30, 0.50), 0.22, 0.10, angle=28, facecolor=GREEN,
                         edgecolor="#326749", lw=1.5, alpha=0.55))
    ax.scatter([0.30], [0.50], s=24, color=INK, zorder=4)
    ax.plot([0.02, 0.30], [0.50, 0.50], color=INK, lw=1.2)
    ax.add_patch(Polygon([[0.00, 0.46], [0.00, 0.54], [0.07, 0.50]], closed=True,
                         facecolor=INK, edgecolor=INK))
    ax.text(0.04, 0.60, "camera", fontsize=9, color=MUTED)
    ax.text(0.27, 0.62, "3D ellipsoid", fontsize=10, color=INK, fontweight="bold")
    ax.plot([0.30, 0.72], [0.50, 0.50], color=MUTED, lw=1, ls="--")
    ax.add_patch(Rectangle((0.72, 0.18), 0.045, 0.64, facecolor="white", edgecolor=INK, lw=1.5))
    ax.add_patch(Ellipse((0.742, 0.50), 0.035, 0.27, angle=0, facecolor=GREEN, edgecolor="#326749",
                         lw=1.3, alpha=0.65))
    ax.text(0.80, 0.55, "2D ellipse", fontsize=10, color=INK, fontweight="bold")
    ax.text(0.80, 0.46, "on image plane", fontsize=9, color=MUTED)
    ax.text(0.5, 0.06, "Position and covariance determine where the splat lands\nand how large, stretched, and rotated it looks.",
            ha="center", color=MUTED, fontsize=10, linespacing=1.4)

    ax = axes[1]
    ax.set_title("Front-to-back alpha compositing", color=INK, fontsize=15, fontweight="bold", pad=14)
    y = 0.52
    ax.add_patch(Polygon([[0.03, y - 0.035], [0.03, y + 0.035], [0.10, y]], closed=True,
                         facecolor=INK, edgecolor=INK))
    ax.plot([0.10, 0.91], [y, y], color=INK, lw=1.4, zorder=0)
    splats = [(0.27, 0.20, 0.42, BLUE, "1"), (0.47, 0.13, 0.52, AMBER, "2"),
              (0.66, 0.24, 0.38, RED, "3")]
    for x, w, alpha, color, label in splats:
        ax.add_patch(Ellipse((x, y), w, 0.30, facecolor=color, edgecolor="none", alpha=alpha))
        ax.scatter([x], [y], s=22, color=INK, zorder=3)
        ax.text(x, 0.73, f"splat {label}", ha="center", color=INK, fontsize=10, fontweight="bold")
        ax.text(x, 0.68, "opacity alpha", ha="center", color=MUTED, fontsize=8)
    ax.add_patch(Rectangle((0.91, 0.28), 0.035, 0.48, facecolor="white", edgecolor=INK, lw=1.5))
    ax.text(0.88, 0.20, "pixel", color=MUTED, fontsize=9)
    ax.text(0.5, 0.10, "Near splats contribute first. Far splats contribute only through\nthe transparency left by every splat in front of them.",
            ha="center", color=MUTED, fontsize=10, linespacing=1.4)

    fig.subplots_adjust(wspace=0.10, left=0.02, right=0.98, top=0.85, bottom=0.05)
    save(fig, "gaussian_projection_compositing.jpg")


def rasterization_vs_ray_tracing():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Rasterization: start from scene primitives and determine covered pixels.
    ax = axes[0]
    ax.set_title("Rasterization: primitives to pixels", color=INK, fontsize=15,
                 fontweight="bold", pad=14)
    ax.add_patch(Polygon([[0.05, 0.46], [0.05, 0.54], [0.12, 0.50]],
                         closed=True, facecolor=INK, edgecolor=INK))
    ax.text(0.07, 0.61, "camera", ha="center", color=MUTED, fontsize=9)
    plane_x = 0.28
    ax.add_patch(Rectangle((plane_x, 0.20), 0.035, 0.60, facecolor="white",
                           edgecolor=INK, lw=1.5))
    for y in np.linspace(0.24, 0.76, 9):
        ax.plot([plane_x, plane_x + 0.035], [y, y], color="#c3c7ce", lw=0.6)
    triangles = [
        ([[0.54, 0.28], [0.70, 0.43], [0.49, 0.50]], BLUE),
        ([[0.59, 0.57], [0.79, 0.72], [0.76, 0.47]], GREEN),
        ([[0.75, 0.28], [0.91, 0.40], [0.87, 0.60]], AMBER),
    ]
    for pts, color in triangles:
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=INK,
                             lw=1.0, alpha=0.75))
        centroid = np.mean(np.asarray(pts), axis=0)
        ax.add_patch(FancyArrowPatch((centroid[0] - 0.02, centroid[1]),
                                     (plane_x + 0.04, centroid[1]),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=color, lw=1.4, alpha=0.9))
    ax.text(0.66, 0.85, "scene triangles", ha="center", color=INK,
            fontsize=10, fontweight="bold")
    ax.text(0.5, 0.08,
            "Project each primitive, find the pixels it covers,\nthen shade those pixels.",
            ha="center", color=MUTED, fontsize=10, linespacing=1.4)

    # Ray tracing: start from pixels and search the scene for intersections.
    ax = axes[1]
    ax.set_title("Ray tracing: pixels to scene", color=INK, fontsize=15,
                 fontweight="bold", pad=14)
    ax.add_patch(Polygon([[0.05, 0.46], [0.05, 0.54], [0.12, 0.50]],
                         closed=True, facecolor=INK, edgecolor=INK))
    ax.text(0.07, 0.61, "camera", ha="center", color=MUTED, fontsize=9)
    ax.add_patch(Rectangle((0.20, 0.20), 0.035, 0.60, facecolor="white",
                           edgecolor=INK, lw=1.5))
    for y in np.linspace(0.24, 0.76, 9):
        ax.plot([0.20, 0.235], [y, y], color="#c3c7ce", lw=0.6)
    surface = Polygon([[0.60, 0.24], [0.88, 0.35], [0.79, 0.68], [0.56, 0.58]],
                      closed=True, facecolor="#d9a09b", edgecolor=INK, lw=1.0,
                      alpha=0.75)
    ax.add_patch(surface)
    light = Circle((0.88, 0.82), 0.045, facecolor=AMBER, edgecolor="#a57216", lw=1.2)
    ax.add_patch(light)
    ax.text(0.88, 0.90, "light", ha="center", color=MUTED, fontsize=9)
    rays = [(0.225, 0.34, 0.65, 0.36), (0.225, 0.49, 0.63, 0.49),
            (0.225, 0.64, 0.68, 0.60)]
    for x0, y0, x1, y1 in rays:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=11, color=BLUE, lw=1.4))
    hit = np.array([0.68, 0.60])
    ax.add_patch(FancyArrowPatch(tuple(hit), (0.85, 0.78), arrowstyle="-|>",
                                 mutation_scale=11, color=AMBER, lw=1.4,
                                 linestyle="--"))
    ax.text(0.48, 0.70, "primary rays", color=BLUE, fontsize=9)
    ax.text(0.69, 0.08,
            "Send a ray through each pixel, find what it hits,\nthen trace more rays for light, shadows, or reflections.",
            ha="center", color=MUTED, fontsize=10, linespacing=1.4)

    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.84, bottom=0.05)
    save(fig, "rasterization_vs_ray_tracing.jpg")


def sfm_colmap_pipeline():
    fig = base_figure(14, 6.4)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(7, 6.02, "Structure from Motion: recover cameras and sparse 3D points",
            ha="center", color=INK, fontsize=17, fontweight="bold")
    ax.text(7, 5.62,
            "The same visual feature appears at different 2D locations; geometry explains all observations together.",
            ha="center", color=MUTED, fontsize=10.5)

    def pipeline_box(x, title, detail, color):
        ax.add_patch(Rectangle((x, 0.45), 2.35, 1.18, facecolor="white",
                               edgecolor=color, linewidth=2))
        ax.text(x + 1.175, 1.34, title, ha="center", va="top", color=INK,
                fontsize=10.5, fontweight="bold")
        ax.text(x + 1.175, 0.67, detail, ha="center", va="bottom", color=MUTED,
                fontsize=8.7, linespacing=1.25)

    boxes = [
        (0.25, "1. Detect features", "Distinctive corners\nand local descriptors", BLUE),
        (3.05, "2. Match + verify", "Candidate correspondences\nthat obey camera geometry", CYAN),
        (5.85, "3. Bootstrap", "Initialize two cameras\nand triangulate points", GREEN),
        (8.65, "4. Grow model", "Register more cameras\nand triangulate tracks", AMBER),
        (11.45, "5. Bundle adjustment", "Jointly refine cameras\nand 3D points", RED),
    ]
    for args in boxes:
        pipeline_box(*args)
    for x0, x1 in [(2.60, 3.05), (5.40, 5.85), (8.20, 8.65), (11.00, 11.45)]:
        ax.add_patch(FancyArrowPatch((x0, 1.04), (x1, 1.04), arrowstyle="-|>",
                                     mutation_scale=13, color=INK, lw=1.4))

    # Three image planes observing the same scene point.
    image_xs = [1.25, 4.05, 6.85]
    feature_ys = [4.00, 4.40, 3.75]
    for idx, (x, fy) in enumerate(zip(image_xs, feature_ys), start=1):
        ax.add_patch(Rectangle((x, 2.73), 1.75, 2.15, facecolor="white",
                               edgecolor=BLUE, lw=1.5))
        ax.plot([x + 0.15, x + 1.60], [3.20, 4.60], color="#c4c9d1", lw=1)
        ax.plot([x + 0.10, x + 1.55], [4.52, 3.12], color="#d3d6dc", lw=1)
        ax.scatter([x + 0.82], [fy], s=50, color=RED, edgecolor="white",
                   linewidth=0.8, zorder=4)
        ax.scatter([x + 0.35, x + 1.35], [3.35, 4.55], s=25,
                   color=[GREEN, AMBER], zorder=3)
        ax.text(x + 0.875, 2.53, f"image {idx}", ha="center", color=MUTED, fontsize=9)
    ax.text(4.90, 5.08, "matching red observations form one feature track",
            ha="center", color=RED, fontsize=9.5, fontweight="bold")
    ax.plot([2.07, 4.87], [feature_ys[0], feature_ys[1]], color=RED, lw=1.1,
            linestyle="--", alpha=0.8)
    ax.plot([4.87, 7.67], [feature_ys[1], feature_ys[2]], color=RED, lw=1.1,
            linestyle="--", alpha=0.8)

    # Reconstructed cameras and a triangulated 3D point.
    point = (11.10, 4.00)
    camera_centers = [(9.10, 3.05), (9.55, 4.70), (12.85, 3.10)]
    for k, (cx, cy) in enumerate(camera_centers, start=1):
        direction = np.array(point) - np.array([cx, cy])
        direction = direction / np.linalg.norm(direction)
        perp = np.array([-direction[1], direction[0]])
        tip = np.array([cx, cy]) + direction * 0.38
        base1 = np.array([cx, cy]) + perp * 0.16
        base2 = np.array([cx, cy]) - perp * 0.16
        ax.add_patch(Polygon([base1, base2, tip], closed=True,
                             facecolor=BLUE, edgecolor=INK, lw=0.8, alpha=0.75))
        ax.plot([tip[0], point[0]], [tip[1], point[1]], color=RED, lw=1,
                linestyle="--", alpha=0.7)
        ax.text(cx, cy - 0.30, f"camera {k}", ha="center", color=MUTED, fontsize=8.5)
    ax.scatter([point[0]], [point[1]], s=90, color=RED, edgecolor="white",
               linewidth=1.2, zorder=5)
    ax.scatter([10.55, 11.55, 11.83, 10.80], [3.55, 3.35, 4.45, 4.62],
               s=25, color=[GREEN, AMBER, PURPLE, CYAN], alpha=0.85)
    ax.text(11.15, 5.08, "estimated poses + sparse point cloud",
            ha="center", color=INK, fontsize=10, fontweight="bold")

    save(fig, "sfm_colmap_pipeline.jpg")


if __name__ == "__main__":
    representation_comparison()
    training_pipeline()
    projection_and_compositing()
    rasterization_vs_ray_tracing()
    sfm_colmap_pipeline()
