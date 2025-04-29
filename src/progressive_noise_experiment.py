# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import umap
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
import random

# --- Step 0: Simulate Base Worm Brain Graph ---

G_base = nx.watts_strogatz_graph(300, 5, 0.1)

print(f"Simulated worm brain: {G_base.number_of_nodes()} neurons, {G_base.number_of_edges()} connections.")

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

# --- Step 1: Base Feature Vectors ---

original_snowflakes = extract_snowflakes(G_base, n_snowflakes=40, k_neighbors=5)
original_features = np.array([flatten_snowflake(coords, adj) for coords, adj in original_snowflakes])

# --- Step 2: Progressive Noise Injection ---

noise_levels = np.linspace(0.0, 0.5, 11)  # From 0% to 50% noise
trust_scores = []

for noise in noise_levels:
    print(f"Injecting {noise*100:.1f}% noise...")
    
    # Perturb snowflakes
    noisy_snowflakes = perturb_snowflakes(original_snowflakes, perturb_prob=noise)
    noisy_features = np.array([flatten_snowflake(coords, adj) for coords, adj in noisy_snowflakes])
    
    # Project with UMAP
    shadow = umap.UMAP(n_neighbors=10, min_dist=0.2, random_state=42).fit_transform(noisy_features)
    
    # Compute trustworthiness
    trust = trustworthiness(noisy_features, shadow, n_neighbors=5)
    trust_scores.append(trust)

# --- Step 3: Plot Trustworthiness vs Noise ---

plt.figure(figsize=(10,6))
plt.plot(noise_levels*100, trust_scores, marker='o', linestyle='-', color='royalblue')
plt.title('Trustworthiness vs Noise Level', fontsize=18)
plt.xlabel('Noise Level (%)', fontsize=14)
plt.ylabel('Trustworthiness Score', fontsize=14)
plt.grid(True)
plt.ylim(0,1.05)
plt.show()
