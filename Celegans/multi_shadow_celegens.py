# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import umap
import random
import copy

# --- Step 0: Load Local C. elegans Connectome ---

# Simulate a "worm brain" (small-world neural network)
G_base = nx.watts_strogatz_graph(300, 5, 0.1)

print(f"Simulated worm brain: {G_base.number_of_nodes()} neurons, {G_base.number_of_edges()} connections.")

# --- Step 1: Get User Input for Noise Level ---

noise_input = input("Enter noise level (e.g., 0.0 for no noise, 0.05 for 5%, 0.15 for 15%): ")
try:
    noise_level = float(noise_input)
except ValueError:
    noise_level = 0.0

print(f"Using noise level: {noise_level * 100:.1f}%")

# --- Step 2: Prepare Two Datasets ---

def perturb_graph(G, perturb_prob=0.05):
    G_copy = copy.deepcopy(G)
    edges = list(G_copy.edges())
    nodes = list(G_copy.nodes())
    
    for (u, v) in edges:
        if np.random.rand() < perturb_prob:
            G_copy.remove_edge(u, v)
            # Randomly pick two nodes to connect
            a, b = random.sample(nodes, 2)
            G_copy.add_edge(a, b)
    return G_copy

# Dataset 1: Original
G1 = G_base

# Dataset 2: Perturbed
G2 = perturb_graph(G_base, perturb_prob=noise_level)

# --- Step 3: Extract Snowflakes ---

def extract_snowflakes_from_graph(G, n_snowflakes=40, k_neighbors=5):
    snowflakes = []
    center_degrees = []
    
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
            center_degrees.append(G.degree(center))
    
    return snowflakes, center_degrees

snowflakes_1, center_degrees_1 = extract_snowflakes_from_graph(G1, n_snowflakes=40, k_neighbors=5)
snowflakes_2, center_degrees_2 = extract_snowflakes_from_graph(G2, n_snowflakes=40, k_neighbors=5)

# --- Step 4: Flatten Snowflakes ---

def flatten_snowflake(coords, adjacency):
    return adjacency.flatten()

feature_vectors_1 = np.array([flatten_snowflake(coords, adj) for coords, adj in snowflakes_1])
feature_vectors_2 = np.array([flatten_snowflake(coords, adj) for coords, adj in snowflakes_2])

# Stack them
feature_vectors = np.vstack([feature_vectors_1, feature_vectors_2])
dataset_labels = np.array([0]*len(feature_vectors_1) + [1]*len(feature_vectors_2))

print(f"Total snowflakes: {feature_vectors.shape[0]}")

# --- Step 5: Project to 2D ---

projector = umap.UMAP(n_neighbors=10, min_dist=0.2, metric='euclidean', random_state=42)
shadow = projector.fit_transform(feature_vectors)

print(f"Shadow projection shape: {shadow.shape}")

# --- Step 6: Visualize Combined Shadow ---

colors = ['deepskyblue', 'lightcoral']

plt.figure(figsize=(10,8))
for label in np.unique(dataset_labels):
    idx = dataset_labels == label
    plt.scatter(
        shadow[idx,0], shadow[idx,1],
        c=colors[label], edgecolor='black', s=90, alpha=0.7,
        label=f"Dataset {label + 1}"
    )
plt.title(f'Multi-Dataset Neural Snowflake Shadow (Noise {noise_level*100:.1f}%)', fontsize=18)
plt.xlabel('Shadow Dimension 1')
plt.ylabel('Shadow Dimension 2')
plt.legend()
plt.grid(True)
plt.show()
