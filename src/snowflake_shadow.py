# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import umap
import networkx as nx

# --- Parameters ---
n_snowflakes = 80  # number of snowflakes
points_per_snowflake = 25  # number of nodes in each snowflake
connection_prob = 2.0  # probability two points are connected

np.random.seed(42)

# --- Step 1: Generate Random Snowflake Graphs ---
snowflakes = []

for _ in range(n_snowflakes):
    # Random 2D coordinates for points
    coords = np.random.rand(points_per_snowflake, 2)
    
    # Create random graph based on proximity or random wiring
    adjacency = np.zeros((points_per_snowflake, points_per_snowflake))
    
    for i in range(points_per_snowflake):
        for j in range(i+1, points_per_snowflake):
            if np.random.rand() < connection_prob:
                adjacency[i, j] = 1
                adjacency[j, i] = 1  # Symmetric

    # Store both the coordinates and the relational structure
    snowflakes.append((coords, adjacency))

print(f"Generated {len(snowflakes)} snowflakes.")

# --- Step 2: Flatten Snowflake Structures for Projection ---
def flatten_snowflake(coords, adjacency):
    # You could combine both position and connections into features
    # Here we'll just use adjacency for simplicity
    return adjacency.flatten()

feature_vectors = np.array([flatten_snowflake(coords, adj) for coords, adj in snowflakes])

print(f"Feature matrix shape: {feature_vectors.shape}")  # (n_snowflakes, features)

# --- Step 3: Project to 2D (Shadow) ---
projector = umap.UMAP(n_neighbors=10, min_dist=0.2, metric='euclidean', random_state=42)
shadow = projector.fit_transform(feature_vectors)

print(f"Shadow shape: {shadow.shape}")

# --- Step 4: Visualize the Shadow ---
plt.figure(figsize=(8,6))
plt.scatter(shadow[:,0], shadow[:,1], c='mediumslateblue', edgecolor='black', s=80)
plt.title('Relational Snowflake Shadow Projection (Layered)', fontsize=16)
plt.xlabel('Shadow Dimension 1')
plt.ylabel('Shadow Dimension 2')
plt.grid(True)
plt.show()

# --- (Optional) Step 5: Visualize Some Sample Snowflakes (Mini-Graphs) ---
def plot_sample_snowflakes(snowflakes, n_samples=5):
    fig, axs = plt.subplots(1, n_samples, figsize=(15, 3))
    selected = np.random.choice(len(snowflakes), n_samples, replace=False)
    
    for idx, snowflake_idx in enumerate(selected):
        coords, adj = snowflakes[snowflake_idx]
        G = nx.Graph()
        for i in range(coords.shape[0]):
            G.add_node(i, pos=coords[i])
        for i in range(coords.shape[0]):
            for j in range(i+1, coords.shape[0]):
                if adj[i, j] > 0:
                    G.add_edge(i, j)
        
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, ax=axs[idx], with_labels=False, node_color='deepskyblue', edge_color='gray')
        axs[idx].set_title(f"Snowflake {snowflake_idx}")
        axs[idx].axis('off')
    
    plt.suptitle('Sample Individual Snowflake Graphs', fontsize=16)
    plt.show()

# Uncomment to also plot some individual snowflakes
plot_sample_snowflakes(snowflakes, n_samples=5)
