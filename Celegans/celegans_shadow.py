# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import umap
import random
import os
import urllib.request

# --- Step 0: Load C. elegans Connectome ---

# --- REMOVED DOWNLOAD LOGIC ---
# def download_celegans():
#     url = 'https://raw.githubusercontent.com/briancohenlab/Connectomes/master/celegans_connectome.gpickle'
#     filename = 'celegans_connectome.gpickle'
#     if not os.path.exists(filename):
#         print("Downloading C. elegans connectome...")
#         urllib.request.urlretrieve(url, filename)
#     return filename
# 
# connectome_file = download_celegans()
# G = nx.read_gpickle(connectome_file)

# --- ADDED PLACEHOLDER GRAPH ---
# Fake but similar small-world graph (for immediate testing)
G = nx.watts_strogatz_graph(300, 5, 0.1)  # 300 nodes, 5 neighbors each, 10% rewiring


print(f"Loaded placeholder graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} connections.")

# --- Step 1: Extract Snowflakes ---

def extract_snowflakes_from_graph(G, n_snowflakes=80, k_neighbors=5):
    snowflakes = []
    center_degrees = []
    
    candidates = [n for n in G.nodes if len(list(G.neighbors(n))) >= k_neighbors]
    
    # Ensure we don't try to sample more candidates than available
    actual_n_snowflakes = min(n_snowflakes, len(candidates))
    if actual_n_snowflakes < n_snowflakes:
        print(f"Warning: Could only find {actual_n_snowflakes} candidate centers with at least {k_neighbors} neighbors.")
        
    if actual_n_snowflakes == 0:
        print("Error: No suitable snowflake centers found. Exiting.")
        return [], []
        
    selected_centers = random.sample(candidates, actual_n_snowflakes)
    
    for center in selected_centers:
        neighbors = list(G.neighbors(center))
        
        # This check is technically redundant due to the 'candidates' list filtering,
        # but kept for clarity and safety.
        if len(neighbors) >= k_neighbors:
            selected_neighbors = random.sample(neighbors, k_neighbors)  # Exactly k neighbors
            subgraph = G.subgraph([center] + selected_neighbors)
            
            # Ensure the subgraph has the expected number of nodes (k+1)
            # This handles potential edge cases where G.subgraph might behave unexpectedly
            # though unlikely with standard networkx graphs.
            if subgraph.number_of_nodes() == k_neighbors + 1:
                coords = np.random.rand(k_neighbors + 1, 2) # Fixed size layout placeholder
                adj = nx.to_numpy_array(subgraph)
                
                snowflakes.append((coords, adj))
                center_degrees.append(G.degree(center))
            # else:
            #     # Optionally handle cases where subgraph formation failed unexpectedly
            #     print(f"Warning: Subgraph for center {center} did not have {k_neighbors + 1} nodes. Skipping.")

    # Handle the case where no valid snowflakes could be formed after sampling
    if not snowflakes:
         print("Error: Failed to extract any valid snowflakes after processing candidates.")
         return [], []
            
    return snowflakes, center_degrees

n_snowflakes = 80
k_neighbors = 5

snowflakes, center_degrees = extract_snowflakes_from_graph(G, n_snowflakes, k_neighbors)

print(f"Extracted {len(snowflakes)} snowflakes.")

# --- Step 2: Flatten Snowflake Structures ---

def flatten_snowflake(coords, adjacency):
    return adjacency.flatten()

feature_vectors = np.array([flatten_snowflake(coords, adj) for coords, adj in snowflakes])

print(f"Feature matrix shape: {feature_vectors.shape}")

# --- Step 3: Project to 2D (Shadow) ---

projector = umap.UMAP(n_neighbors=10, min_dist=0.2, metric='euclidean', random_state=42)
shadow = projector.fit_transform(feature_vectors)

print(f"Shadow projection shape: {shadow.shape}")

# --- Step 4: Visualize the Shadow (Color by Degree) ---

plt.figure(figsize=(10,8))
scatter = plt.scatter(
    shadow[:,0], shadow[:,1], 
    c=center_degrees, cmap='viridis', edgecolor='black', s=100
)
plt.title('C. elegans Neural Snowflake Shadow Projection', fontsize=18)
plt.xlabel('Shadow Dimension 1')
plt.ylabel('Shadow Dimension 2')
cbar = plt.colorbar(scatter)
cbar.set_label('Center Neuron Degree (# connections)', fontsize=14)
plt.grid(True)
plt.show()

# --- Step 5: (Optional) Perturb Snowflakes and Animate ---

def perturb_snowflakes(snowflakes, perturb_prob=0.1):
    perturbed = []
    for coords, adj in snowflakes:
        adj_copy = adj.copy()
        n = adj.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if np.random.rand() < perturb_prob:
                    adj_copy[i,j] = 1 - adj_copy[i,j]  # Flip connection
                    adj_copy[j,i] = 1 - adj_copy[j,i]
        perturbed.append((coords, adj_copy))
    return perturbed

# Perturb and reproject
perturbed_snowflakes = perturb_snowflakes(snowflakes, perturb_prob=0.05)
perturbed_features = np.array([flatten_snowflake(coords, adj) for coords, adj in perturbed_snowflakes])
perturbed_shadow = projector.fit_transform(perturbed_features)

# Plot perturbed shadow vs original
plt.figure(figsize=(10,8))
plt.scatter(
    shadow[:,0], shadow[:,1], 
    c='blue', edgecolor='black', s=80, alpha=0.5, label='Original'
)
plt.scatter(
    perturbed_shadow[:,0], perturbed_shadow[:,1], 
    c='red', edgecolor='black', s=80, alpha=0.5, label='Perturbed'
)
plt.title('Shadow Drift After Small Perturbations', fontsize=18)
plt.xlabel('Shadow Dimension 1')
plt.ylabel('Shadow Dimension 2')
plt.legend()
plt.grid(True)
plt.show()
