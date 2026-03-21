import matplotlib.pyplot as plt
import numpy as np

epochs =      [1,        10,       20,       30,       40,       50,       60,       70,       80,       90,       100]
train_loss =  [0.596689, 0.075011, 0.060781, 0.054927, 0.053271, 0.048833, 0.046503, 0.045018, 0.042442, 0.041309, 0.041315]
val_loss =    [np.nan,   0.098281, 0.063878, 0.057780, 0.054257, 0.055282, 0.053696, 0.055179, 0.060251, 0.060744, 0.059244]
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4")
ax.plot(epochs, val_loss,   "s-", label="Val Loss",   color="#ff7f0e")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title("OpenCabinet – Diffusion Policy Training (100 epochs, cosine LR)", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig("../figures/training_curves.png", dpi=150)
plt.show()
print("Saved training_curves.png")
