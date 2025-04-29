# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt

# --- Synthetic trust data (we'll replace this with real extracted values if needed) ---

trust_scores = {
    'Worm Brain': [0.92, 0.90, 0.88, 0.86, 0.83, 0.81],
    'Random Graph': [0.85, 0.78, 0.68, 0.58, 0.45, 0.32],
    'Social Graph': [0.88, 0.83, 0.74, 0.65, 0.52, 0.4],
    'Knowledge Graph': [0.90, 0.89, 0.86, 0.82, 0.79, 0.74],
    'Protein Graph': [0.95, 0.93, 0.91, 0.88, 0.84, 0.8],
}

noise_levels = [0, 10, 20, 30, 40, 50]

# --- Plotting ---

fig, ax = plt.subplots(figsize=(10,6))

for label, scores in trust_scores.items():
    ax.plot(noise_levels, scores, marker='o', label=label)

ax.set_xlabel("Noise Level (%)", fontsize=14)
ax.set_ylabel("Trustworthiness Score", fontsize=14)
ax.set_title("Structural Resilience Comparison Across Graph Types", fontsize=18)
ax.grid(True)
ax.legend(fontsize=12)
plt.xticks(noise_levels)
plt.ylim(0,1)
plt.tight_layout()
plt.show()
