"""Generate figures for euler_vs_quaternions.md.

Outputs (all .jpg, per repo convention):
  assets/rot_euler_order.jpg   -- same two 90 deg turns in opposite order -> different frame
  assets/rot_gimbal_lock.jpg   -- three gimbal rings: healthy vs middle-at-90 (axes collapse)
  assets/rot_axis_angle.jpg    -- Euler's theorem: one axis + one angle rotates the whole frame
  assets/rot_slerp_vs_lerp.jpg -- SLERP (constant angular speed) vs normalized-LERP (bunched)
  assets/rot_continuity.jpg    -- Euler wrap / quaternion sign-flip jump vs continuous 6D rep

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_rotation_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# A soft, consistent, brand-neutral palette (matches make_optimiser_figs.py).
C = {
    "blue":   "#6C8EBF",
    "purple": "#9673A6",
    "green":  "#82B366",
    "orange": "#D6795B",
    "grey":   "#555555",
    "lgrey":  "#EDEDED",
    "yellow": "#E8B84B",
    "text":   "#222222",
}

DPI = 150


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", os.path.normpath(path))


# ---------------------------------------------------------------------------
# Rotation-matrix helpers (right-handed, active rotations).
# ---------------------------------------------------------------------------
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


AXCOL = {"x": C["orange"], "y": C["green"], "z": C["blue"]}


def draw_frame(ax, R, alpha=1.0, lw=2.6, style="-", label=True, ls_len=1.0):
    """Draw the three basis arrows of a frame after rotation R."""
    names = ["x", "y", "z"]
    for i, nm in enumerate(names):
        v = R[:, i] * ls_len
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=AXCOL[nm], lw=lw,
                  arrow_length_ratio=0.16, alpha=alpha, linestyle=style)
        if label:
            p = v * 1.16
            ax.text(p[0], p[1], p[2], nm, color=AXCOL[nm], fontsize=11,
                    fontweight="bold", ha="center", va="center")


def style_3d(ax, lim=1.25):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=22, azim=35)
    # keep the panes light and unobtrusive
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor(C["lgrey"])
        pane.pane.set_alpha(1.0)


# ---------------------------------------------------------------------------
# 1. Order matters: R differs when you swap the order of two 90-degree turns.
# ---------------------------------------------------------------------------
def fig_euler_order():
    fig = plt.figure(figsize=(10.4, 5.6))
    a = np.pi / 2

    # Left: X-turn first, then Z-turn  (applied to a vector: Rz @ Rx).
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    style_3d(axL)
    draw_frame(axL, Rz(a) @ Rx(a))
    axL.set_title("Turn X 90°, then Z 90°", fontsize=12.5,
                  fontweight="bold", color=C["text"], pad=0)

    # Right: Z-turn first, then X-turn  (Rx @ Rz).
    axR = fig.add_subplot(1, 2, 2, projection="3d")
    style_3d(axR)
    draw_frame(axR, Rx(a) @ Rz(a))
    axR.set_title("Turn Z 90°, then X 90°", fontsize=12.5,
                  fontweight="bold", color=C["text"], pad=0)

    fig.suptitle("Same two turns, opposite order → different orientation",
                 fontsize=13, fontweight="bold", color=C["text"], y=0.99)
    fig.text(0.5, 0.03,
             "Rotations do not commute, so three Euler angles mean nothing "
             "without also fixing the axis order (and intrinsic vs. extrinsic).",
             ha="center", fontsize=9.5, style="italic", color=C["grey"])
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    save(fig, "rot_euler_order.jpg")


# ---------------------------------------------------------------------------
# 2. Gimbal lock: three rings. Healthy -> 3 independent axes. Middle at 90 deg
#    -> outer and inner rotation axes coincide, so a whole DOF is lost.
# ---------------------------------------------------------------------------
def _ring(ax, R, color, lw=3.0, alpha=1.0, scale=1.0):
    """A gimbal ring is a unit circle in the local XY plane, placed by R."""
    t = np.linspace(0, 2 * np.pi, 200)
    circ = scale * np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)])
    pts = R @ circ
    ax.plot(pts[0], pts[1], pts[2], color=color, lw=lw, alpha=alpha)


def _spin_axis(ax, R, color):
    """The axis a ring spins about = its local z (the circle's normal)."""
    n = R[:, 2]
    ax.quiver(0, 0, 0, n[0], n[1], n[2], color=color, lw=2.2,
              arrow_length_ratio=0.18)
    ax.quiver(0, 0, 0, -n[0], -n[1], -n[2], color=color, lw=2.2,
              arrow_length_ratio=0.18)


def fig_gimbal_lock():
    fig = plt.figure(figsize=(10.4, 5.8))

    # Outer ring spins about world Z; its circle lies in the XY plane.
    outer = Rz(0)                    # normal = z
    # Middle ring hangs inside the outer one, spinning about X (normal = x).
    #   place its circle in the YZ plane  -> rotate the base circle about Y by 90.
    middle_base = Ry(np.pi / 2)      # normal = x
    # Inner ring spins about Y in the healthy pose (normal = y): circle in XZ plane.
    inner_base = Rx(np.pi / 2)       # normal = y

    # ---- healthy: three mutually perpendicular spin axes = 3 DOF ----
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    style_3d(axL, lim=1.15)
    _ring(axL, outer, C["blue"])
    _ring(axL, middle_base, C["green"])
    _ring(axL, inner_base, C["orange"])
    _spin_axis(axL, outer, C["blue"])
    _spin_axis(axL, middle_base, C["green"])
    _spin_axis(axL, inner_base, C["orange"])
    axL.set_title("Healthy: 3 perpendicular axes = 3 DOF",
                  fontsize=12.5, fontweight="bold", color=C["text"], pad=0)

    # ---- locked: spin the MIDDLE ring 90 deg about ITS axis (x). That turn
    #      carries the inner ring so the inner spin axis (y) swings onto the
    #      outer (z) axis -> the two axes coincide. ----
    tilt = Rx(np.pi / 2)             # the middle ring's own 90-deg rotation (about x)
    axR = fig.add_subplot(1, 2, 2, projection="3d")
    style_3d(axR, lim=1.15)
    _ring(axR, outer, C["blue"])
    _ring(axR, middle_base, C["green"])                       # middle ring itself
    _ring(axR, tilt @ inner_base, C["orange"], scale=0.8)     # carried by the tilt
    _spin_axis(axR, outer, C["blue"])
    _spin_axis(axR, tilt @ inner_base, C["orange"])
    axR.set_title("Locked: middle ring at 90°",
                  fontsize=12.5, fontweight="bold", color=C["text"], pad=0)
    axR.text2D(0.5, -0.02,
               "outer (blue) & inner (orange) axes coincide → 1 DOF lost",
               transform=axR.transAxes, ha="center", fontsize=9.5,
               color=C["orange"], style="italic")

    fig.suptitle("Gimbal lock: a singularity of the three-angle scheme itself",
                 fontsize=13, fontweight="bold", color=C["text"], y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save(fig, "rot_gimbal_lock.jpg")


# ---------------------------------------------------------------------------
# 3. Axis-angle (Euler's rotation theorem): any orientation is ONE turn of
#    angle theta about ONE fixed axis n.
# ---------------------------------------------------------------------------
def fig_axis_angle():
    fig = plt.figure(figsize=(5.8, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    # Turn the empty bounding cube OFF so the tight bbox crops to the arrows,
    # not to a big blank box. Then zoom in hard.
    ax.set_axis_off()
    ax.view_init(elev=18, azim=35)
    # arrows are "top-heavy" (n-hat points up), so use an asymmetric z-range and
    # a shortened vertical box so the content fills the frame instead of leaving
    # the lower half of a symmetric cube empty.
    lim = 0.64
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-0.42, 0.80)
    ax.set_box_aspect([1.0, 1.0, 0.80])

    # the single rotation axis n (a slanted unit vector), drawn long
    n = np.array([0.3, 0.3, 1.0]); n = n / np.linalg.norm(n)
    ax.quiver(0, 0, 0, n[0], n[1], n[2], color=C["purple"], lw=3.2,
              arrow_length_ratio=0.13)
    ax.quiver(0, 0, 0, -n[0] * 0.35, -n[1] * 0.35, -n[2] * 0.35,
              color=C["purple"], lw=3.2, arrow_length_ratio=0.0)
    ax.text(n[0] - 0.16, n[1] - 0.16, n[2] + 0.02, r"$\hat n$",
            color=C["purple"], fontsize=15, fontweight="bold")

    theta = np.deg2rad(110)
    R = _axis_angle_matrix(n, theta)

    draw_frame(ax, np.eye(3), alpha=0.32, lw=2.0, style="--", label=False,
               ls_len=0.82)
    draw_frame(ax, R, lw=2.8, label=True, ls_len=0.82)

    # a little arc around n to suggest the angle theta
    _angle_arc(ax, n, theta)

    fig.suptitle("Euler's theorem: one axis $\\hat n$ + one angle "
                 "$\\theta$\nreach any orientation",
                 fontsize=12.5, fontweight="bold", color=C["text"], y=0.99)
    fig.text(0.5, 0.05,
             "Dashed = start frame, solid = after a single turn of "
             "$\\theta$ about $\\hat n$.",
             ha="center", fontsize=9.5, style="italic", color=C["grey"])
    # pull the 3D axes out to the edges to kill the surrounding margin
    ax.set_position([0.0, 0.02, 1.0, 0.93])
    save(fig, "rot_axis_angle.jpg")


def _axis_angle_matrix(n, theta):
    """Rodrigues' formula: rotation of angle theta about unit axis n."""
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _angle_arc(ax, n, theta):
    """Draw a small arc in the plane perpendicular to n to show theta."""
    # build two orthonormal vectors u, w spanning the plane perpendicular to n
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(n @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp); u /= np.linalg.norm(u)
    w = np.cross(n, u)
    r = 0.45
    ts = np.linspace(0, theta, 60)
    arc = np.array([r * (np.cos(t) * u + np.sin(t) * w) for t in ts]).T
    ax.plot(arc[0], arc[1], arc[2], color=C["grey"], lw=1.8)
    mid = arc[:, len(ts) // 2] * 1.25
    ax.text(mid[0], mid[1], mid[2], r"$\theta$", color=C["grey"],
            fontsize=13, fontweight="bold")


# ---------------------------------------------------------------------------
# 4. SLERP vs normalized-LERP: same endpoints, but SLERP moves at a constant
#    angular speed while nlerp bunches the samples toward the two ends.
# ---------------------------------------------------------------------------
def fig_slerp_vs_lerp():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.8, 5.2))

    a0, a1 = np.deg2rad(20), np.deg2rad(160)
    Om = a1 - a0
    q0 = np.array([np.cos(a0), np.sin(a0)])
    q1 = np.array([np.cos(a1), np.sin(a1)])
    ts = np.linspace(0, 1, 9)

    # ---- left: the geometric picture on the unit circle ----
    th = np.linspace(0, np.pi, 300)
    axL.plot(np.cos(th), np.sin(th), color=C["lgrey"], lw=8, solid_capstyle="round")

    slerp = np.array([(np.sin((1 - t) * Om) * q0 + np.sin(t * Om) * q1) / np.sin(Om)
                      for t in ts])
    nlerp = np.array([(v := (1 - t) * q0 + t * q1) / np.linalg.norm(v) for t in ts])

    axL.plot(slerp[:, 0], slerp[:, 1], "o", color=C["blue"], ms=9,
             label="SLERP (even spacing)", zorder=4)
    axL.plot(nlerp[:, 0] * 0.86, nlerp[:, 1] * 0.86, "s", color=C["orange"], ms=8,
             label="normalized LERP (bunched)", zorder=4)
    for q, txt, dx in [(q0, "$q_0$", -0.12), (q1, "$q_1$", 0.06)]:
        axL.plot([0, q[0]], [0, q[1]], color=C["grey"], lw=1.0, ls=":")
        axL.text(q[0] + dx, q[1] + 0.08, txt, fontsize=12, color=C["text"])
    axL.set_aspect("equal")
    axL.set_xlim(-1.15, 1.15); axL.set_ylim(-0.15, 1.2)
    axL.axis("off")
    axL.legend(loc="lower center", fontsize=9.5, frameon=False)
    axL.set_title("Interpolating between two orientations",
                  fontsize=12.5, fontweight="bold", color=C["text"])

    # ---- right: fraction of the turn completed vs t ----
    tt = np.linspace(0, 1, 200)
    frac_slerp = tt  # by construction, linear in the angle
    nl = np.array([(v := (1 - t) * q0 + t * q1) / np.linalg.norm(v) for t in tt])
    ang = (np.arctan2(nl[:, 1], nl[:, 0]) - a0) / Om
    axR.plot(tt, frac_slerp, color=C["blue"], lw=2.6, label="SLERP")
    axR.plot(tt, ang, color=C["orange"], lw=2.6, label="normalized LERP")
    axR.plot([0, 1], [0, 1], color=C["grey"], lw=1.0, ls=":")
    axR.set_xlabel("interpolation parameter  $t$", fontsize=10, color=C["text"])
    axR.set_ylabel("fraction of the rotation done", fontsize=10, color=C["text"])
    axR.set_title("SLERP has constant angular speed",
                  fontsize=12.5, fontweight="bold", color=C["text"])
    axR.legend(loc="upper left", fontsize=9.5, frameon=False)
    axR.spines[["top", "right"]].set_visible(False)
    axR.text(0.52, 0.12,
             "nlerp lags, then rushes:\nspeed is not constant",
             fontsize=9, color=C["orange"], style="italic")

    fig.tight_layout()
    save(fig, "rot_slerp_vs_lerp.jpg")


# ---------------------------------------------------------------------------
# 5. Continuity: sweep a rotation smoothly (spin about z from 0 to 360). The
#    Euler angle wraps and the (sign-canonical) quaternion flips -> jumps a
#    network must reproduce. The 6D rep (columns of R) stays perfectly smooth.
# ---------------------------------------------------------------------------
def fig_continuity():
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    phi = np.linspace(0, 360, 720)
    r = np.deg2rad(phi)

    # (a) Euler yaw kept in (-180, 180]: a hard wrap at 180 deg.
    yaw = ((phi + 180) % 360) - 180
    ax = axes[0]
    ax.plot(phi, yaw, color=C["orange"], lw=2.4)
    ax.set_title("Euler angle: wraps", fontsize=12, fontweight="bold",
                 color=C["text"])
    ax.annotate("jump", xy=(180, 0), xytext=(235, 90), fontsize=9.5,
                color=C["orange"], arrowprops=dict(arrowstyle="-|>",
                color=C["orange"], lw=1.2))

    # (b) quaternion for spin about z, canonicalized to w >= 0: qz flips sign.
    w = np.cos(r / 2)
    qz = np.sin(r / 2)
    flip = np.where(w < 0, -1.0, 1.0)
    w_c, qz_c = w * flip, qz * flip
    ax = axes[1]
    ax.plot(phi, w_c, color=C["blue"], lw=2.4, label="$w$")
    ax.plot(phi, qz_c, color=C["purple"], lw=2.4, label="$q_z$")
    ax.set_title("Quaternion: sign flip", fontsize=12, fontweight="bold",
                 color=C["text"])
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.annotate("jump", xy=(180, 1), xytext=(210, 0.2), fontsize=9.5,
                color=C["purple"], arrowprops=dict(arrowstyle="-|>",
                color=C["purple"], lw=1.2))

    # (c) 6D rep = first two columns of R_z(phi): pure cos/sin, fully smooth.
    ax = axes[2]
    ax.plot(phi, np.cos(r), color=C["green"], lw=2.4, label="col entries")
    ax.plot(phi, np.sin(r), color=C["green"], lw=2.4, alpha=0.55)
    ax.set_title("6D rep: continuous", fontsize=12, fontweight="bold",
                 color=C["green"])

    for ax in axes:
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("rotation swept (deg)", fontsize=9.5, color=C["text"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(180, color=C["lgrey"], lw=1.0, zorder=0)

    fig.suptitle("As the SAME rotation is swept smoothly, only the 6D "
                 "representation stays continuous",
                 fontsize=12.5, fontweight="bold", color=C["text"], y=1.02)
    fig.tight_layout()
    save(fig, "rot_continuity.jpg")


if __name__ == "__main__":
    fig_euler_order()
    fig_gimbal_lock()
    fig_axis_angle()
    fig_slerp_vs_lerp()
    fig_continuity()
    print("done.")
