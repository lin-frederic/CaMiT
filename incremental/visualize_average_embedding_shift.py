import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

heatmap_data = np.load("final_results/avg_embedding_shift.npy")
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.3)

train_years = [str(i) for i in range(2007, 2024)]
train_yearsB = [str(i) for i in range(2007, 2024)]

# Main heatmap axis (left, bottom, width, height) in figure coordinates
ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])

# Colorbar axis – adjust height here (smaller height to reduce size)
cbar_ax = fig.add_axes([0.9, 0.1, 0.03, 0.75])  # [left, bottom, width, height]
sns.heatmap(
    heatmap_data,
    xticklabels=False,
    yticklabels=False,
    annot=False,
    fmt=".1f",
    cmap="Spectral_r",
    ax=ax,
    cbar_ax=cbar_ax
)

ax.set_aspect("equal")

# Set ticks
xticks = [i + 0.5 for i, x in enumerate(train_yearsB) if (int(x) - 2007) % 4 == 0]
xticklabels = [x[2:] for x in train_yearsB if (int(x) - 2007) % 4 == 0]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=0, fontsize=46)

yticks = [i + 0.5 for i, x in enumerate(train_years) if (int(x) - 2007) % 4 == 0]
yticklabels = [x[2:] for x in train_years if (int(x) - 2007) % 4 == 0]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, rotation=0, fontsize=46)

# Format colorbar
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
cbar_ax.yaxis.set_major_formatter(formatter)
cbar_ax.tick_params(labelsize=46)
cbar_ax.yaxis.get_offset_text().set(size=46)
cbar_ax.yaxis.offsetText.set_x(10)
cbar_ax.set_ylabel("KID", fontsize=46, labelpad=20)

ax.set_xlabel("Year", fontsize=46)
ax.set_ylabel("Year", fontsize=46)

plt.savefig("final_results/heatmap_embedding_shift.png", bbox_inches='tight')
plt.show()