from collections import defaultdict
import math

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


graph, nodes = shortest_path("test1.txt")
print_graph_info(graph, nodes)