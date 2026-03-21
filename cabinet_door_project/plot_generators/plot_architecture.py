import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(16, 14))
ax.set_xlim(0, 16)
ax.set_ylim(0, 14)
ax.axis("off")

C_OBS    = "#4A90D9"
C_UNET   = "#E8913A"
C_DIFF   = "#6BBF6B"
C_ACTION = "#D94A4A"
C_COND   = "#9B72CF"


def box(x, y, w, h, text, color, fontsize=9, bold=False, alpha=0.15):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=color, alpha=alpha, linewidth=2)
    ax.add_patch(rect)
    border = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor="none", edgecolor=color, linewidth=2)
    ax.add_patch(border)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="#222222")


def arrow(x1, y1, x2, y2, color="#555555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))


# ── Title ──
ax.text(8, 13.5, "Diffusion Policy Architecture — OpenCabinet", ha="center",
        fontsize=16, fontweight="bold", color="#222222")
ax.text(8, 13.1, "ConditionalUnet1D  |  DDPM  |  Low-Dim Observations  |  4.2M params",
        ha="center", fontsize=10, color="#666666")

# ===================== ROW 1: Inputs (y ≈ 11.5) =====================

# Observation inputs — left side
box(0.5, 11.3, 3.2, 1.2,
    "Robot State (7)\neef_pos (3) + eef_quat (4)", C_OBS, fontsize=9)
box(4.2, 11.3, 3.6, 1.2,
    "Handle Features (6)\nhandle_to_eef (3) + openness (1)\n+ xaxis_xy (2)", C_OBS, fontsize=9)

# Noisy actions — right side (separated clearly)
box(10.0, 11.3, 5.0, 1.2,
    "Noisy Action Trajectory\naₜ ~ N(0, I)  →  (16, 7)\nhorizon=16, action_dim=7", C_DIFF, fontsize=9)

# ===================== ROW 2: Merge obs (y ≈ 9.7) =====================

# Arrows from obs boxes down to context window
arrow(2.1, 11.3, 4.0, 10.5, C_OBS)
arrow(6.0, 11.3, 4.0, 10.5, C_OBS)

box(1.5, 9.7, 5.0, 0.8,
    "Obs Context Window: 2 steps → (2, 13)", C_OBS, fontsize=9, bold=True)

# ===================== ROW 3: Conditioning (y ≈ 8.3) =====================

# Flatten obs
arrow(4.0, 9.7, 4.0, 9.1, C_COND)
box(1.5, 8.3, 5.0, 0.8,
    "Flatten → Global Cond (26-dim)", C_COND, fontsize=9, bold=True)

# Diffusion timestep t — separate input, not from noisy actions
box(10.0, 9.7, 5.0, 0.8,
    "Timestep t  (integer 0–99)", C_DIFF, fontsize=9, bold=True)

arrow(12.5, 9.7, 12.5, 9.1, C_DIFF)
box(10.0, 8.3, 5.0, 0.8,
    "Diffusion Step Embed (128-dim)\nsinusoidal → MLP", C_DIFF, fontsize=9)

# ===================== ROW 4: UNet block (y ≈ 4.5–7.2) =====================

box(0.5, 4.5, 15.0, 3.2, "", C_UNET, alpha=0.06)
ax.text(8, 7.35, "ConditionalUnet1D", ha="center", fontsize=13,
        fontweight="bold", color=C_UNET)

# Conditioning arrows into UNet top
arrow(4.0, 8.3, 4.0, 7.7, C_COND)
ax.text(4.3, 8.0, "global cond", fontsize=8, color=C_COND)
arrow(12.5, 8.3, 12.5, 7.7, C_DIFF)
ax.text(12.8, 8.0, "step embed", fontsize=8, color=C_DIFF)

# Noisy action input arrow straight down into UNet
arrow(12.5, 11.3, 8.0, 7.7, "#888888")
ax.text(10.8, 9.7, "noisy input", fontsize=8, color="#888888", rotation=-30)

# Encoder
box(1.2, 4.9, 3.5, 2.0,
    "Encoder\n\n↓ Conv1D (k=5)\nGroupNorm(8)\n\n7 → 64\n64 → 128\n128 → 256", C_UNET, fontsize=9)

# Mid block
box(5.8, 4.9, 4.4, 2.0,
    "Mid Block\n\n256-dim\nConv1D + GroupNorm(8)\n+ Diffusion Step Cond\n+ Global Obs Cond", C_UNET, fontsize=9)

# Decoder
box(11.3, 4.9, 3.5, 2.0,
    "Decoder\n\n↑ Conv1D (k=5)\nGroupNorm(8)\n\n256 → 128\n128 → 64\n64 → 7 (ε pred)", C_UNET, fontsize=9)

# Internal encoder → mid → decoder arrows
arrow(4.7, 5.9, 5.8, 5.9, C_UNET)
arrow(10.2, 5.9, 11.3, 5.9, C_UNET)

# Skip connections (curved, below the blocks)
ax.annotate("", xy=(11.3, 5.5), xytext=(4.7, 5.5),
            arrowprops=dict(arrowstyle="-|>", color=C_UNET, lw=1.3,
                            connectionstyle="arc3,rad=-0.4", linestyle="dashed"))
ax.text(8, 4.55, "skip connections", fontsize=8, ha="center",
        color=C_UNET, style="italic")

# ===================== ROW 5: DDPM loop (y ≈ 2.5) =====================

arrow(8, 4.5, 8, 3.7, C_DIFF)

box(3.5, 2.5, 9.0, 1.1,
    "DDPM Iterative Denoising (T = 100 steps)\nβ-schedule: squared cosine  |  predict ε  |  clip sample",
    C_DIFF, fontsize=9, bold=True)

# Loop-back arrow — routed along the right edge, pointing to box side
ax.plot([12.5, 15.2], [3.0, 3.0], color=C_DIFF, lw=2.0, linestyle="dotted")
ax.plot([15.2, 15.2], [3.0, 11.9], color=C_DIFF, lw=2.0, linestyle="dotted")
ax.annotate(
    "", xy=(15.0, 11.9), xytext=(15.2, 11.9),
    arrowprops=dict(arrowstyle="-|>", color=C_DIFF, lw=2.0, linestyle="dotted"))
ax.text(15.5, 7.3, "iterate\n100×", fontsize=10, ha="center", color=C_DIFF,
        fontweight="bold")

# ===================== ROW 6: Output (y ≈ 0.7) =====================

arrow(8, 2.5, 8, 1.7, C_ACTION)

box(3.0, 0.6, 10.0, 1.0,
    "Predicted Actions (8 of 16 horizon steps executed)\nexpand 7-dim → 12-dim for env.step()",
    C_ACTION, fontsize=9, bold=True)

plt.savefig("../figures/architecture_diagram.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.show()
print("Saved architecture_diagram.png")