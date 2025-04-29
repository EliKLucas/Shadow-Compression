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
save_folder = 'frames_protein'
os.makedirs(save_folder, exist_ok=True)

# --- Step 1: Simulate Folded Protein Structure Graph ---

G_base = nx.Graph()

# Create several tightly connected clusters (folded regions)
folded_clusters = {
    'helix1': [f'h1_{i}' for i in range(15)],
    'helix2': [f'h2_{i}' for i in range(15)],
    'sheet1': [f's1_{i}' for i in range(10)],
    'sheet2': [f's2_{i}' for i in range(10)],
    'coil': [f'c_{i}' for i in range(20)]
}

# Strong internal links within clusters
for group in folded_clusters.values():
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            if np.random.rand() < 0.6:  # 60% strong internal connection
                G_base.add_edge(group[i], group[j])

# Sparse inter-cluster links (simulating loops/flexible connectors)
cluster_names = list(folded_clusters.keys())
for _ in range(30):
    cluster_a, cluster_b = random.sample(cluster_names, 2)
    node_a = random.choice(folded_clusters[cluster_a])
    node_b = random.choice(folded_clusters[cluster_b])
    if not G_base.has_edge(node_a, node_b):
        G_base.add_edge(node_a, node_b)

print(f"Protein-like Graph: {G_base.number_of_nodes()} nodes, {G_base.number_of_edges()} edges")

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
noise_levels = np.linspace(0.0, 0.5, 21)

previous_shadow = None
trust_scores = []

for idx, noise in enumerate(noise_levels):
    print(f"Noise Level: {noise:.2f}")
    
    # Perturb
    noisy_snowflakes = perturb_snowflakes(original_snowflakes, perturb_prob=noise)
    noisy_features = np.array([flatten_snowflake(coords, adj) for coords, adj in noisy_snowflakes])
    
    # Project
    shadow = umap.UMAP(n_neighbors=10, min_dist=0.2, random_state=42).fit_transform(noisy_features)
    shadow = center_shadow(shadow)
    
    # Compute trustworthiness
    trust = trustworthiness(noisy_features, shadow, n_neighbors=5)
    trust_scores.append(trust)
    
    # Drift calculation
    if previous_shadow is None:
        drift = np.zeros(shadow.shape[0])
    else:
        drift = np.linalg.norm(shadow - previous_shadow, axis=1)
    
    previous_shadow = shadow.copy()
    
    drift_normalized = (drift - drift.min()) / (drift.max() - drift.min() + 1e-8)
    
    # Plot with drift colors
    plt.figure(figsize=(8,6))
    plt.scatter(
        shadow[:,0], shadow[:,1], 
        c=drift_normalized, cmap='coolwarm', edgecolor='black', s=80, alpha=0.9
    )
    plt.colorbar(label="Drift Speed (Red = fast melting)")
    plt.title(f'Protein-like Graph Relational Shadow\nNoise Level: {noise*100:.1f}%  |  Trust: {trust:.3f}', fontsize=16)
    plt.xlabel('Shadow Dimension 1')
    plt.ylabel('Shadow Dimension 2')
    plt.grid(True)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    
    frame_path = os.path.join(save_folder, f'frame_{idx:03d}.png')
    plt.savefig(frame_path)
    plt.close()

# --- Step 4: Build Animation ---

frame_files = sorted([os.path.join(save_folder, f) for f in os.listdir(save_folder) if f.endswith('.png')])

with imageio.get_writer('protein_graph_melting.gif', mode='I', duration=0.3) as writer:
    for filename in frame_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print("\n✅ Protein graph melting animation saved as 'protein_graph_melting.gif'!")
