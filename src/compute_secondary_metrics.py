import networkx as nx
import random

def generate_graph(graph_type, n_nodes=300):
    if graph_type == "worm":
        return nx.watts_strogatz_graph(n_nodes, 5, 0.1)
    elif graph_type == "random":
        return nx.erdos_renyi_graph(n_nodes, 0.05)
    elif graph_type == "social":
        return nx.barabasi_albert_graph(n_nodes, 3)
    elif graph_type == "knowledge":
        return generate_synthetic_knowledge_graph(n_nodes)
    elif graph_type == "protein":
        return generate_synthetic_protein_graph(n_nodes)
    else:
        raise ValueError(f"Unknown graph type {graph_type}")

def generate_synthetic_knowledge_graph(n_nodes):
    G = nx.Graph()
    cluster_size = n_nodes // 4
    for c in range(4):
        nodes = [f"c{c}_{i}" for i in range(cluster_size)]
        G.add_nodes_from(nodes)
        random.shuffle(nodes) # Correctly use random.shuffle
        for i in nodes:
            for j in nodes:
                if i != j and random.random() < 0.3: # Correctly use random.random()
                    G.add_edge(i, j)
    all_nodes = list(G.nodes) # Get list of nodes once
    if not all_nodes: # Handle case where no nodes were added
      return G
    for _ in range(50):
        a = random.choice(all_nodes) # Correctly use random.choice
        b = random.choice(all_nodes) # Correctly use random.choice
        if not G.has_edge(a, b) and a != b:
            G.add_edge(a, b)
    return G

def generate_synthetic_protein_graph(n_nodes):
    G = nx.Graph()
    folded_clusters = 5
    cluster_size = n_nodes // folded_clusters
    for c in range(folded_clusters):
        nodes = [f"f{c}_{i}" for i in range(cluster_size)]
        G.add_nodes_from(nodes)
        random.shuffle(nodes) # Correctly use random.shuffle
        for i in nodes:
            for j in nodes:
                if i != j and random.random() < 0.6: # Correctly use random.random()
                    G.add_edge(i, j)
    all_nodes = list(G.nodes) # Get list of nodes once
    if not all_nodes:
        return G
    for _ in range(30):
        a = random.choice(all_nodes) # Correctly use random.choice
        b = random.choice(all_nodes) # Correctly use random.choice
        if not G.has_edge(a, b) and a != b:
            G.add_edge(a, b)
    return G

def compute_metrics(G):
    # Handle empty graph case for clustering
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        clustering = 0.0
        modularity = None # Modularity is undefined for empty/edgeless graphs
        return clustering, modularity

    clustering = nx.average_clustering(G)
    try:
        from networkx.algorithms import community
        # Use label propagation as it's generally faster and handles disconnected components
        communities = list(nx.community.label_propagation_communities(G))
        if not communities or len(communities) <= 1: # Modularity is 0 or undefined if 0/1 communities
          modularity = 0.0
        else:
          modularity = community.modularity(G, communities)
    except Exception as e:
        print(f"Modularity computation failed: {e}") # Keep error message
        modularity = None
    return clustering, modularity

if __name__ == "__main__":
    graph_types = ['worm', 'random', 'social', 'knowledge', 'protein']

    results = {}
    for gtype in graph_types:
        print(f"--- {gtype.upper()} ---")
        G = generate_graph(gtype)
        clustering, modularity = compute_metrics(G)
        results[gtype] = {'clustering': clustering, 'modularity': modularity}
        print(f"Average Clustering Coefficient: {clustering:.3f}")
        if modularity is not None:
            print(f"Modularity: {modularity:.3f}")
        else:
            print("Modularity: (could not compute or undefined)")

    # Optional: Print summary table at the end
    print("\n--- SUMMARY ---")
    print("Graph Type      | Avg Clustering | Modularity")
    print("----------------|----------------|------------")
    for gtype, metrics in results.items():
        mod_str = f"{metrics['modularity']:.3f}" if metrics['modularity'] is not None else "N/A"
        print(f"{gtype:<15} | {metrics['clustering']:.3f}          | {mod_str}") 