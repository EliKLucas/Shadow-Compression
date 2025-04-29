# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# --- Settings ---
noise_idx = 12  # About 30% noise frame (adjust if needed)

frame_paths = {
    'Worm Brain': f'frames_worm_fixed/frame_{noise_idx:03d}.png',
    'Random Graph': f'frames_random/frame_{noise_idx:03d}.png',
    'Social Graph': f'frames_social/frame_{noise_idx:03d}.png',
    'Knowledge Graph': f'frames_knowledge/frame_{noise_idx:03d}.png',
    'Protein Graph': f'frames_protein/frame_{noise_idx:03d}.png',
}

# --- Plotting ---

fig, axes = plt.subplots(1, 5, figsize=(25, 6))

for ax, (title, path) in zip(axes, frame_paths.items()):
    img = imageio.imread(path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=16)

fig.suptitle("Relational Melting Comparison Across Systems\n(~30% Noise)", fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
