from collections import defaultdict
import math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

#Algorithms
def shortest_path(graph_path: str):
    print(graph_path)
    #defining wanted parameters
    n_visited = 0
    shortest_length = 200 #TODO: change to some max value
    shortest_walk = []

    #reading the file 
    with open(graph_path, 'r') as f:
        # Read first line
        n, m, k, s, t = map(int, f.readline().split())

        # Read nodes
        nodes = {}
        for _ in range(n):
            node_id, x, y = f.readline().split()
            nodes[int(node_id)] = (float(x), float(y))

        # Build graph as adjacency list
        graph = defaultdict(list)
        for _ in range(m):
            u, v = map(int, f.readline().split())
            # Calculate weight (Euclidean distance)
            x1, y1 = nodes[u]
            x2, y2 = nodes[v]
            weight = math.hypot(x2 - x1, y2 - y1)
            # Since it's an undirected graph, add both directions
            graph[u].append((v, weight))
            graph[v].append((u, weight))

    #print(n_visited)
    #print(shortest_length)
    #print(shortest_walk)
    return graph, nodes

#Tools
def print_graph_info(graph, nodes):
    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of edges: {sum(len(neighbors) for neighbors in graph.values()) // 2}")

    print("Node coordinates:")
    for node_id, (x, y) in nodes.items():
        print(f"  Node {node_id}: ({x}, {y})")
    
    print("\n Adjacency list with edge weights:")
    for node_id, neighbors in graph.items():
        neighbor_str = ", ".join(f"{nbr} (weight={weight:.2f})" for nbr, weight in neighbors)
        print(f"  Node {node_id} -> {neighbor_str}")
		
def print_adjacency_matrix(graph, nodes):
    """
    Prints the adjacency matrix of the graph.

    Args:
        graph: dict of adjacency list with weights.
        nodes: dict of node_id -> (x, y).
    """
    node_ids = sorted(nodes.keys())
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    size = len(node_ids)

    # Initialize matrix with infinities
    matrix = np.full((size, size), np.inf)

    # Fill in the weights
    for u in graph:
        for v, weight in graph[u]:
            i, j = id_to_index[u], id_to_index[v]
            matrix[i][j] = weight

    # Optional: set diagonal to 0
    for i in range(size):
        matrix[i][i] = 0.0

    # Print header
    header = "    " + "  ".join(f"{nid:>3}" for nid in node_ids)
    print(header)
    print("   " + "-" * (len(header) - 3))
    
    # Print matrix rows
    for i, row in enumerate(matrix):
        row_str = f"{node_ids[i]:>3}|" + " ".join(
                f"{int(val):>5}" if val != np.inf and val.is_integer() else f"{val:>5.1f}" if val != np.inf else "  inf"
                for val in row
        )
        print(row_str)
		
def visualize_graph(graph, nodes, s=None, t=None, highlight_path=None):
    """
    Visualizes the graph using matplotlib and networkx.

    Args:
        graph: dict of adjacency list with weights.
        nodes: dict of node_id -> (x, y).
        s: starting node id (optional).
        t: target node id (optional).
        highlight_path: list of node ids representing the path to highlight (optional).
    """
    G = nx.Graph()

    # Add nodes with positions
    for node_id, (x, y) in nodes.items():
        G.add_node(node_id, pos=(x, y))

    # Add edges with weights
    for u, neighbors in graph.items():
        for v, weight in neighbors:
            if (v, u) not in G.edges:  # avoid adding the same undirected edge twice
                G.add_edge(u, v, weight=weight)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(8, 6))
    
    # Draw nodes
    node_colors = []
    for node in G.nodes:
        if node == s:
            node_colors.append('green')
        elif node == t:
            node_colors.append('red')
        else:
            node_colors.append('lightblue')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1)

    # Highlight path if given
    if highlight_path and len(highlight_path) > 1:
        path_edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=3)

    # Draw labels
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{d['weight']:.1f}" for e, d in G.edges.items()}, font_size=8)

    plt.title("Graph Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

graph, nodes = shortest_path("test1.txt")
print_graph_info(graph, nodes)
print("")
print("\n Adjacency matrix:")
print_adjacency_matrix(graph, nodes)

#Vizualize the whole graph
visualize_graph(graph, nodes, s=1, t=4)
