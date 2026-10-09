"""Create the original conceptual JPG diagrams used by VideoJAM.md."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle, Rectangle
import numpy as np


ASSETS = Path(__file__).resolve().parents[1] / "assets"
INK = "#202528"
BLUE = "#247caa"
RED = "#bf5049"
GREEN = "#368775"
BACKGROUND = "#fafbf9"


def canvas(width, height):
    figure, axis = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    return figure, axis


def label(axis, xpos, ypos, text, size=12, color=INK):
    axis.text(xpos, ypos, text, ha="center", va="center", fontsize=size,
              color=color, linespacing=1.4)


def box(axis, xpos, ypos, width, height, text, color=BLUE):
    axis.add_patch(Rectangle((xpos, ypos), width, height,
                            facecolor="white", edgecolor=color, linewidth=1.7))
    label(axis, xpos + width / 2, ypos + height / 2, text, size=11)


def arrow(axis, start, end, color=INK):
    axis.annotate("", xy=end, xytext=start,
                  arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6})


def save(figure, name):
    ASSETS.mkdir(exist_ok=True)
    figure.savefig(ASSETS / name, dpi=180, bbox_inches="tight",
                   facecolor=BACKGROUND, pil_kwargs={"quality": 94})
    plt.close(figure)
    print(ASSETS / name)


def frame_order():
    figure, axis = canvas(11, 4.8)
    label(axis, 0.5, 0.94, "Same pictures, different temporal meaning", size=17)
    orders = [(0, 1, 2, 3), (0, 3, 1, 2)]
    for row, order in enumerate(orders):
        ypos = 0.57 - row * 0.34
        label(axis, 0.10, ypos + 0.08,
              "Ordered" if row == 0 else "Shuffled", color=GREEN if row == 0 else RED)
        for column, frame in enumerate(order):
            xpos = 0.23 + column * 0.185
            axis.add_patch(Rectangle((xpos, ypos), 0.15, 0.20,
                                    facecolor="white", edgecolor="#bac2c4"))
            axis.plot([xpos + 0.01, xpos + 0.14], [ypos + 0.045] * 2,
                      color="#bac2c4")
            axis.add_patch(Circle((xpos + 0.025 + frame * 0.032, ypos + 0.095),
                                  0.015, color=BLUE))
            label(axis, xpos + 0.075, ypos - 0.04, f"Frame {frame + 1}", size=10)
            if column < 3:
                arrow(axis, (xpos + 0.154, ypos + 0.1),
                      (xpos + 0.180, ypos + 0.1))
    label(axis, 0.5, 0.055,
          "Conceptual illustration: frame quality alone does not establish coherent motion.", size=10)
    save(figure, "videojam_frame_order.jpg")


def optical_flow():
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    figure.patch.set_facecolor(BACKGROUND)
    for axis in axes:
        axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
        axis.set_facecolor("white")
        axis.set_xticks([])
        axis.set_yticks([])
    for axis, xpos, title in zip(axes[:2], [0.30, 0.60], ["Frame 1", "Frame 2"]):
        axis.add_patch(Rectangle((xpos - 0.15, 0.35), 0.30, 0.30, color=BLUE))
        axis.set_title(title, fontsize=12)
    positions = np.linspace(0.36, 0.64, 5)
    horizontal, vertical = np.meshgrid(positions, positions)
    axes[2].quiver(horizontal, vertical, np.full_like(horizontal, 0.30),
                   np.zeros_like(vertical), angles="xy", scale_units="xy", scale=1,
                   color=RED, width=0.009)
    axes[2].set_title("Flow: displacement, not texture", fontsize=12)
    angles = np.linspace(0, 1, 256)
    color_strip = hsv_to_rgb(np.stack([angles, np.ones_like(angles),
                                     np.ones_like(angles)], axis=-1))[None, :, :]
    axes[2].imshow(color_strip, extent=(0.1, 0.9, 0.10, 0.17), aspect="auto")
    axes[2].text(0.5, 0.04, "Illustrative direction-to-color key", ha="center", fontsize=9)
    figure.suptitle("A moving object becomes a displacement field", fontsize=16)
    figure.text(0.5, 0.01,
                "Conceptual flow illustration; not RAFT output or the paper's exact color wheel.",
                ha="center", fontsize=9, color=INK)
    figure.tight_layout(rect=(0, 0.06, 1, 0.9))
    save(figure, "videojam_optical_flow.jpg")


def training():
    figure, axis = canvas(12, 5)
    label(axis, 0.5, 0.94, "One shared representation, two learning targets", size=17)
    box(axis, 0.02, 0.59, 0.17, 0.16, "Noisy video\nlatent x_t")
    box(axis, 0.02, 0.23, 0.17, 0.16, "Noisy RGB-flow\nlatent d_t", GREEN)
    box(axis, 0.26, 0.40, 0.18, 0.18, "Channel concat.\nExpanded input\nprojection")
    box(axis, 0.51, 0.40, 0.20, 0.18, "Shared DiT blocks\nSame hidden width")
    box(axis, 0.79, 0.59, 0.19, 0.16, "Video velocity\nhead + video loss")
    box(axis, 0.79, 0.23, 0.19, 0.16, "Flow velocity\nhead + flow loss", GREEN)
    arrow(axis, (0.19, 0.67), (0.26, 0.52))
    arrow(axis, (0.19, 0.31), (0.26, 0.46), GREEN)
    arrow(axis, (0.44, 0.49), (0.51, 0.49))
    arrow(axis, (0.71, 0.52), (0.79, 0.67))
    arrow(axis, (0.71, 0.46), (0.79, 0.31), GREEN)
    label(axis, 0.61, 0.72, "Text prompt + time", size=11)
    arrow(axis, (0.61, 0.67), (0.61, 0.58))
    label(axis, 0.5, 0.09,
          "Flow comes from training-video frame pairs. All model weights are fine-tuned.", size=11)
    save(figure, "videojam_joint_training.jpg")


def guidance():
    figure, axis = canvas(12, 6)
    label(axis, 0.5, 0.95, "Inner-Guidance: contrast full prediction with missing-condition branches", size=15)
    box(axis, 0.02, 0.42, 0.19, 0.20, "Current video x_t\nCurrent motion d_t\nPrompt y")
    branch_y = [0.71, 0.43, 0.15]
    branch_text = ["Full: video + motion + text", "No text: video + motion", "No motion: video + text"]
    branch_color = [BLUE, RED, GREEN]
    for ypos, text, color in zip(branch_y, branch_text, branch_color):
        box(axis, 0.30, ypos, 0.31, 0.15, text, color)
        arrow(axis, (0.21, 0.52), (0.30, ypos + 0.075), color)
        arrow(axis, (0.61, ypos + 0.075), (0.72, 0.52), color)
    box(axis, 0.72, 0.42, 0.26, 0.20, "Full + text contrast\n+ motion contrast\nUpdate joint state")
    arrow(axis, (0.85, 0.42), (0.85, 0.015))
    arrow(axis, (0.85, 0.015), (0.11, 0.015))
    arrow(axis, (0.11, 0.015), (0.11, 0.42))
    label(axis, 0.5, 0.075, "Repeat at the next sampling time", size=10)
    label(axis, 0.5, 0.90, "Same model weights in all three branches; motion evolves with the video", size=10)
    save(figure, "videojam_inner_guidance.jpg")


def related_methods():
    figure, axis = canvas(13, 6.8)
    label(axis, 0.5, 0.96, "Four ways to teach structure beyond appearance", size=17)
    label(axis, 0.265, 0.88, "Training supervision", size=11)
    label(axis, 0.58, 0.88, "Learning mechanism", size=11)
    label(axis, 0.885, 0.88, "What the model gains", size=11)
    rows = [
        ("VideoJAM", "Training video +\nRAFT optical flow",
         "Joint video-flow\nprediction task", "Joint representation\n+ generated motion", BLUE),
        ("MoAlign", "Clean training video\n+ flow labels",
         "Flow-supervised teacher\nthen relational alignment", "Motion-aware\nvideo features", GREEN),
        ("GeoVideo", "Training RGB + depth\n+ geometric estimates",
         "Joint RGB-depth learning\n+ reprojection loss", "Cross-view\ngeometric agreement", RED),
        ("Track4Gen", "Training video +\npoint correspondences",
         "Refiner + tracking loss\n+ feedback to generator", "Correspondence-rich\nvideo features", "#876c34"),
    ]
    for row, (name, supervision, mechanism, result, color) in enumerate(rows):
        ypos = 0.70 - row * 0.20
        label(axis, 0.065, ypos + 0.065, name, size=11, color=color)
        box(axis, 0.15, ypos, 0.23, 0.13, supervision, color)
        box(axis, 0.43, ypos, 0.30, 0.13, mechanism, color)
        box(axis, 0.78, ypos, 0.21, 0.13, result, color)
        arrow(axis, (0.38, ypos + 0.065), (0.43, ypos + 0.065), color)
        arrow(axis, (0.73, ypos + 0.065), (0.78, ypos + 0.065), color)
    label(axis, 0.5, 0.035,
          "Conceptual training comparison, not a performance ranking. Supervision is not necessarily an inference-time input.",
          size=9)
    save(figure, "videojam_related_methods.jpg")


if __name__ == "__main__":
    frame_order()
    optical_flow()
    training()
    guidance()
    related_methods()