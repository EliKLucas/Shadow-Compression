# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import umap
import random
import os
import imageio
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

# --- Set Up Workspace ---
save_folder = 'frames_better'
os.makedirs(save_folder, exist_ok=True)

# --- Step 1: Simulate Base Worm Brain Graph ---
G_base = nx.watts_strogatz_graph(300, 5, 0.1)

# --- Functions ---

def extract_snowflakes(G, n_snowflakes=40, k_neighbors=5):
    snowflakes = []
    candidates = [n for n in G.nodes if G.degree(n) >= k_neighbors]
    selected_centers = random.sample(candidates, n_snowflakes)
    for center in selected_centers:
        neighbors = list(G.neighbors(center))
        if len(neighbors) >= k_neighbors:
            selected_neighbors = random.sample(neighbors, k_neighbors)
            subgraph = G.subgraph([center] + selected_neighbors)
            coords = np.random.rand(len(subgraph.nodes), 2)
            adj = nx.to_numpy_array(subgraph)
            snowflakes.append((coords, adj))
    return snowflakes

def flatten_snowflake(coords, adj):
    return adj.flatten()

def perturb_snowflakes(snowflakes, perturb_prob=0.05):
    perturbed = []
    for coords, adj in snowflakes:
        adj_copy = adj.copy()
        n = adj.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if np.random.rand() < perturb_prob:
                    adj_copy[i,j] = 1 - adj_copy[i,j]
                    adj_copy[j,i] = 1 - adj_copy[i,j]
        perturbed.append((coords, adj_copy))
    return perturbed

def center_shadow(shadow):
    return shadow - np.mean(shadow, axis=0)

# --- Step 2: Base Feature Vectors ---
original_snowflakes = extract_snowflakes(G_base, n_snowflakes=40, k_neighbors=5)

# --- Step 3: Progressive Noise Injection + Frame Saving ---
noise_levels = np.linspace(0.0, 0.5, 21)  # 0% to 50%, fine-grained

trust_scores = []

for idx, noise in enumerate(noise_levels):
    print(f"Noise Level: {noise:.2f}")
    
    # Perturb
    noisy_snowflakes = perturb_snowflakes(original_snowflakes, perturb_prob=noise)
    noisy_features = np.array([flatten_snowflake(coords, adj) for coords, adj in noisy_snowflakes])
    
    # Project
    shadow = umap.UMAP(n_neighbors=10, min_dist=0.2, random_state=42).fit_transform(noisy_features)
    
    # Center the shadow
    shadow = center_shadow(shadow)
    
    # Compute trustworthiness
    trust = trustworthiness(noisy_features, shadow, n_neighbors=5)
    trust_scores.append(trust)
    
    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(shadow[:,0], shadow[:,1], c='dodgerblue', edgecolor='black', s=80, alpha=0.8)
    plt.title(f'Snowflake Relational Shadow\nNoise Level: {noise*100:.1f}%  |  Trust: {trust:.3f}', fontsize=16)
    plt.xlabel('Shadow Dimension 1')
    plt.ylabel('Shadow Dimension 2')
    plt.grid(True)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    
    # Save frame
    frame_path = os.path.join(save_folder, f'frame_{idx:03d}.png')
    plt.savefig(frame_path)
    plt.close()

# --- Step 4: Build Animation ---

frame_files = sorted([os.path.join(save_folder, f) for f in os.listdir(save_folder) if f.endswith('.png')])

with imageio.get_writer('better_snowflake_melting.gif', mode='I', duration=0.3) as writer:
    for filename in frame_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print("\n✅ Better animation saved as 'better_snowflake_melting.gif'!")
