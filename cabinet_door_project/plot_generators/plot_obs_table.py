"""Generate a table summarizing kept vs dropped observation and action dimensions."""

import matplotlib.pyplot as plt

obs_data = [
    # (Source, Feature, Dims, Status, Reason)
    ("Robot State", "eef_pos", 3, "Kept", "End-effector position relative to base — directly controls arm reach"),
    ("Robot State", "eef_quat", 4, "Kept", "End-effector orientation — needed for rotation control"),
    ("Robot State", "base_pos", 3, "Dropped", "Varies with kitchen layout, not task behavior — adds noise"),
    ("Robot State", "base_quat", 4, "Dropped", "Varies with kitchen layout, not task behavior — adds noise"),
    ("Robot State", "gripper_qpos", 2, "Dropped", "Redundant with gripper action — not informative for planning"),
    ("Handle (augmented)", "handle_to_eef", 3, "Kept", "Relative vector to handle — key spatial signal for reaching"),
    ("Handle (augmented)", "door_openness", 1, "Kept", "How open the door is — task progress signal"),
    ("Handle (augmented)", "handle_xaxis_xy", 2, "Kept", "Door facing direction (xy) — needed for approach angle"),
    ("Handle (augmented)", "handle_xaxis_z", 1, "Dropped", "Z component always ~0 (handles are horizontal)"),
    ("Handle (augmented)", "handle_pos", 3, "Dropped", "World-frame position — varies with layout, redundant with handle_to_eef"),
    ("Handle (augmented)", "hinge_direction", 1, "Dropped", "Left/right opening — constant within each episode, no temporal signal"),
]

act_data = [
    ("Action", "dead dims [0–4]", 5, "Dropped", "Always [0,0,0,0,−1] — base/torso not needed for OpenCabinet"),
    ("Action", "arm pos delta", 3, "Kept", "Translational movement commands"),
    ("Action", "arm rot delta", 3, "Kept", "Rotational movement commands"),
    ("Action", "gripper", 1, "Kept", "Open/close gripper command"),
]

all_data = obs_data + act_data

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis("off")

col_labels = ["Source", "Feature", "Dims", "Status", "Reason"]
cell_text = [[src, feat, str(dims), status, reason]
             for src, feat, dims, status, reason in all_data]

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc="center",
    cellLoc="left",
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.8)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2a6cb8")
    cell.set_text_props(color="white", fontweight="bold", fontsize=10)

# Style rows
for i, (_, _, _, status, _) in enumerate(all_data, start=1):
    color = "#e8f4e8" if status == "Kept" else "#f5e6e6"
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(color)
        if j == 3:
            fc = "#2e7d32" if status == "Kept" else "#c62828"
            table[i, j].set_text_props(color=fc, fontweight="bold")

# Column widths
col_widths = [0.13, 0.12, 0.05, 0.07, 0.45]
for j, w in enumerate(col_widths):
    for i in range(len(all_data) + 1):
        table[i, j].set_width(w)

# Separator line between obs and action sections
for j in range(len(col_labels)):
    table[len(obs_data) + 1, j].set_edgecolor("#333333")

kept_obs = sum(d for _, _, d, s, _ in obs_data if s == "Kept")
dropped_obs = sum(d for _, _, d, s, _ in obs_data if s == "Dropped")
kept_act = sum(d for _, _, d, s, _ in act_data if s == "Kept")
dropped_act = sum(d for _, _, d, s, _ in act_data if s == "Dropped")

ax.set_title(
    f"Observation & Action Dimension Selection\n"
    f"Obs: {kept_obs} kept / {dropped_obs} dropped  |  "
    f"Actions: {kept_act} kept / {dropped_act} dropped  |  "
    f"Total input: {kept_obs} obs + {kept_act} act = {kept_obs + kept_act} dims",
    fontsize=13, fontweight="bold", pad=20,
)

plt.tight_layout()
plt.savefig("../figures/obs_action_table.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print("Saved obs_action_table.png")
