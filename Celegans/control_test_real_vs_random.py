# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import umap
import trimap
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import random

# --- Function: Calculate Continuity Score ---
def continuity(high_neighbors, low_neighbors):
    n_samples, k = high_neighbors.shape
    cont_score = 1 - (2 / (n_samples * k * (2*n_samples - 3*k - 1))) * np.sum(
        [
            len(set(high_neighbors[i]).difference(set(low_neighbors[i])))
            for i in range(n_samples)
        ]
    )
    return cont_score

# --- Function: Full Evaluation ---
def evaluate_relational_projection(feature_vectors, dataset_labels=None, n_neighbors=5):
    print("Starting projection evaluation...\\n")

    # --- Step 1: Project with UMAP, t-SNE, TriMAP ---
    print("Projecting with UMAP...")
    shadow_umap = umap.UMAP(n_neighbors=10, min_dist=0.2, random_state=42).fit_transform(feature_vectors)

    print("Projecting with t-SNE...")
    shadow_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(feature_vectors)

    print("Projecting with TriMAP...")
    shadow_trimap = trimap.TRIMAP(n_dims=2).fit_transform(feature_vectors)

    # --- Step 2: Trustworthiness and Continuity ---
    print("\\nCalculating Trustworthiness and Continuity scores...")
    trust_umap = trustworthiness(feature_vectors, shadow_umap, n_neighbors=n_neighbors)
    trust_tsne = trustworthiness(feature_vectors, shadow_tsne, n_neighbors=n_neighbors)
    trust_trimap = trustworthiness(feature_vectors, shadow_trimap, n_neighbors=n_neighbors)

    nn_high = NearestNeighbors(n_neighbors=n_neighbors).fit(feature_vectors)
    high_neighbors = nn_high.kneighbors(return_distance=False)

    low_neighbors_umap = NearestNeighbors(n_neighbors=n_neighbors).fit(shadow_umap).kneighbors(return_distance=False)
    low_neighbors_tsne = NearestNeighbors(n_neighbors=n_neighbors).fit(shadow_tsne).kneighbors(return_distance=False)
    low_neighbors_trimap = NearestNeighbors(n_neighbors=n_neighbors).fit(shadow_trimap).kneighbors(return_distance=False)

    cont_umap = continuity(high_neighbors, low_neighbors_umap)
    cont_tsne = continuity(high_neighbors, low_neighbors_tsne)
    cont_trimap = continuity(high_neighbors, low_neighbors_trimap)

    # --- Step 3: Reconstruction Errors ---
    print("\\nCalculating Reconstruction Errors...")
    reconstructed_umap = NearestNeighbors(n_neighbors=n_neighbors).fit(shadow_umap).kneighbors_graph(shadow_umap).toarray()
    reconstructed_tsne = NearestNeighbors(n_neighbors=n_neighbors).fit(shadow_tsne).kneighbors_graph(shadow_tsne).toarray()
    reconstructed_trimap = NearestNeighbors(n_neighbors=n_neighbors).fit(shadow_trimap).kneighbors_graph(shadow_trimap).toarray()

    true_high = NearestNeighbors(n_neighbors=n_neighbors).fit(feature_vectors).kneighbors_graph(feature_vectors).toarray()

    recon_error_umap = mean_squared_error(true_high, reconstructed_umap)
    recon_error_tsne = mean_squared_error(true_high, reconstructed_tsne)
    recon_error_trimap = mean_squared_error(true_high, reconstructed_trimap)

    # --- Step 4: Print Full Report ---
    print("\\n=== Evaluation Report ===\\n")
    print(f"UMAP:\\n  Trustworthiness: {trust_umap:.4f}\\n  Continuity: {cont_umap:.4f}\\n  Reconstruction Error: {recon_error_umap:.6f}\\n")
    print(f"t-SNE:\\n  Trustworthiness: {trust_tsne:.4f}\\n  Continuity: {cont_tsne:.4f}\\n  Reconstruction Error: {recon_error_tsne:.6f}\\n")
    print(f"TriMAP:\\n  Trustworthiness: {trust_trimap:.4f}\\n  Continuity: {cont_trimap:.4f}\\n  Reconstruction Error: {recon_error_trimap:.6f}\\n")

    # --- Step 5: Plot All Projections ---
    print("Plotting projections...\\n")
    plt.figure(figsize=(18,6))

    plt.subplot(1,3,1)
    plt.scatter(shadow_umap[:,0], shadow_umap[:,1], c=dataset_labels, cmap='coolwarm', edgecolor='black', s=60)
    plt.title('UMAP Projection')

    plt.subplot(1,3,2)
    plt.scatter(shadow_tsne[:,0], shadow_tsne[:,1], c=dataset_labels, cmap='coolwarm', edgecolor='black', s=60)
    plt.title('t-SNE Projection')

    plt.subplot(1,3,3)
    plt.scatter(shadow_trimap[:,0], shadow_trimap[:,1], c=dataset_labels, cmap='coolwarm', edgecolor='black', s=60)
    plt.title('TriMAP Projection')

    plt.suptitle('Relational Shadows: UMAP vs t-SNE vs TriMAP', fontsize=18)
    plt.show()

# --- Step 0: Simulate Worm Brain Graph ---

# (you could load celegans.gml if you want, but simulating is easier for now)
G_real = nx.watts_strogatz_graph(300, 5, 0.1)

print(f"Simulated worm brain: {G_real.number_of_nodes()} neurons, {G_real.number_of_edges()} connections.")

# --- Step 1: Extract Real Snowflakes ---

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

real_snowflakes, real_degrees = extract_snowflakes_from_graph(G_real, n_snowflakes=40, k_neighbors=5)

# --- Step 2: Generate Random Snowflakes ---

def generate_random_snowflakes(n_snowflakes=40, n_nodes=6, connection_prob=0.3):
    snowflakes = []
    for _ in range(n_snowflakes):
        G_random = nx.erdos_renyi_graph(n_nodes, connection_prob)
        coords = np.random.rand(n_nodes, 2)
        adj = nx.to_numpy_array(G_random)
        snowflakes.append((coords, adj))
    return snowflakes

random_snowflakes = generate_random_snowflakes(n_snowflakes=40, n_nodes=6, connection_prob=0.3)

# --- Step 3: Flatten Snowflakes ---

def flatten_snowflake(coords, adjacency):
    return adjacency.flatten()

feature_vectors_real = np.array([flatten_snowflake(coords, adj) for coords, adj in real_snowflakes])
feature_vectors_random = np.array([flatten_snowflake(coords, adj) for coords, adj in random_snowflakes])

# --- Step 4: Stack and Project ---

feature_vectors = np.vstack([feature_vectors_real, feature_vectors_random])
dataset_labels = np.array([0]*len(feature_vectors_real) + [1]*len(feature_vectors_random))

print(f"Total snowflakes: {feature_vectors.shape[0]}")

# --- Step 5: Run Full Evaluation ---
evaluate_relational_projection(feature_vectors, dataset_labels, n_neighbors=5)

# --- Step 6: Visualize Real vs Random --- REMOVED as plotting is inside evaluate_relational_projection

# colors = ['mediumseagreen', 'tomato']

# plt.figure(figsize=(10,8))
# for label in np.unique(dataset_labels):
#     idx = dataset_labels == label
#     plt.scatter(
#         shadow[idx,0], shadow[idx,1],
#         c=colors[label], edgecolor='black', s=90, alpha=0.7,
#         label=f"{'Real Snowflakes' if label==0 else 'Random Snowflakes'}"
#     )
# plt.title('Control Test: Real vs Random Snowflakes', fontsize=18)
# plt.xlabel('Shadow Dimension 1')
# plt.ylabel('Shadow Dimension 2')
# plt.legend()
# plt.grid(True)
# plt.show() 