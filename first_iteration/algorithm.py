import sys
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time

class Vertex:
    def __init__(self, name):
        self.name = name
        self.visited = False
        self.adjacencies_list = []
        self.min_distance = float('inf')
        self.previous_vertex = None

    def add_edge(self, edge):
        self.adjacencies_list.append(edge)

    def __str__(self):
        return self.name


class Edge:
    def __init__(self, weight, start_vertex, target_vertex):
        self.weight = weight
        self.start_vertex = start_vertex
        self.target_vertex = target_vertex


class Graph:
    def __init__(self, vertices_count):
        self.Cena = "None"
        self.vertices_count = vertices_count
        self.vertex_list = [Vertex(str(i + 1)) for i in range(vertices_count)]
        self.edge_list = []
        self.coordinates = {i + 1: (0.0, 0.0) for i in range(vertices_count)}  # node_id: (x, y)

    def add_edge(self, from_idx, to_idx, cost):
        edge = Edge(cost, self.vertex_list[from_idx - 1], self.vertex_list[to_idx - 1])
        self.edge_list.append(edge)

    def shortest_path_internal(self, source_idx, target_idx):
        source_vertex = self.vertex_list[source_idx - 1]
        target_vertex = self.vertex_list[target_idx - 1]
        source_vertex.min_distance = 0

        # Bellman-Ford algorithm
        for _ in range(self.vertices_count - 1):
            for edge in self.edge_list:
                if edge.start_vertex.min_distance == float('inf'):
                    continue
                new_distance = edge.start_vertex.min_distance + edge.weight
                if new_distance < edge.target_vertex.min_distance:
                    edge.target_vertex.min_distance = new_distance
                    edge.target_vertex.previous_vertex = edge.start_vertex

        # Check for negative cycles
        for edge in self.edge_list:
            if edge.start_vertex.min_distance != float('-inf'):
                if self.has_cycle(edge):
                    print("Imamo negative cikelj")

        if target_vertex.min_distance == float('inf'):
            self.Cena = "None"
        else:
            self.Cena = str(int(target_vertex.min_distance))

    def has_cycle(self, edge):
        return (edge.start_vertex.min_distance + edge.weight) < edge.target_vertex.min_distance

    def adjacency_list(self):
        """
        Returns:
            A defaultdict(list) representing the adjacency list of the graph,
            where each key is a node ID and each value is a list of (neighbor, weight) tuples.
        """
        adj_list = defaultdict(list)
        for edge in self.edge_list:
            u = int(edge.start_vertex.name)
            v = int(edge.target_vertex.name)
            w = edge.weight
            adj_list[u].append((v, w))
        return adj_list

    def get_shortest_path(self, target_idx):
        path = []
        current_vertex = self.vertex_list[target_idx - 1]

        # If no path was found
        if current_vertex.min_distance == float('inf'):
            return []

        while current_vertex:
            path.append(int(current_vertex.name))  # Assuming names are str of indices
            current_vertex = current_vertex.previous_vertex

        return list(reversed(path))

    #Helper methods
    @classmethod
    def from_txt_file(cls, graph_path):
        with open(graph_path, 'r') as f:
            # Read graph meta-info
            n, m, k, s, t = map(int, f.readline().split())

            # Read node positions
            nodes = {}
            for _ in range(n):
                node_id, x, y = f.readline().split()
                nodes[int(node_id)] = (float(x), float(y))

            # Create the graph
            graph = cls(n)

            #populate self.coordinates:
            graph.coordinates = nodes

            # Read and add edges with Euclidean distances
            for _ in range(m):
                u, v = map(int, f.readline().split())
                x1, y1 = nodes[u]
                x2, y2 = nodes[v]
                weight = math.hypot(x2 - x1, y2 - y1)
                graph.add_edge(u, v, weight)
                graph.add_edge(v, u, weight)  # undirected

            return graph, s, t

    def to_txt_file(self, graph_path, k=0, s=1, t=1):
        with open(graph_path, 'w') as f:
            # Write n (nodes), m (edges), k, s, t
            f.write(f"{self.vertices_count} {len(self.edge_list) // 2} {k} {s} {t}\n")

            # Write node positions (dummy values, like (0.0, 0.0))
            for i in range(1, self.vertices_count + 1):
                f.write(f"{i} 0.0 0.0\n")

            # Track written edges to avoid writing both directions
            written = set()
            for edge in self.edge_list:
                u = int(edge.start_vertex.name)
                v = int(edge.target_vertex.name)
                if (v, u) not in written:  # Avoid duplicates in undirected graph
                    f.write(f"{u} {v}\n")
                    written.add((u, v))

    def print_graph_info(self):
        from collections import defaultdict
        adj_list = defaultdict(list)
        for edge in self.edge_list:
            u = int(edge.start_vertex.name)
            v = int(edge.target_vertex.name)
            w = edge.weight
            adj_list[u].append((v, w))

        print(f"Number of nodes: {self.vertices_count}")
        print(f"Number of edges: {len(self.edge_list) // 2}")
        print("Node coordinates:")
        for node_id, (x, y) in self.coordinates.items():
            print(f"  Node {node_id}: ({x}, {y})")

        print("\nAdjacency list with edge weights:")
        for node_id in sorted(adj_list.keys()):
            neighbors = ", ".join(f"{v} (weight={w:.2f})" for v, w in adj_list[node_id])
            print(f"  Node {node_id} -> {neighbors}")

    def print_adjacency_matrix(self):
        node_ids = sorted(self.coordinates.keys())
        id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        size = len(node_ids)
        matrix = np.full((size, size), np.inf)

        for edge in self.edge_list:
            u = int(edge.start_vertex.name)
            v = int(edge.target_vertex.name)
            i, j = id_to_index[u], id_to_index[v]
            matrix[i][j] = edge.weight

        for i in range(size):
            matrix[i][i] = 0.0

        header = "      " + "   ".join(f"{nid:>3}" for nid in node_ids)
        print(header)
        print("   " + "-" * (len(header) - 3))
        for i, row in enumerate(matrix):
            row_str = f"{node_ids[i]:>3}|" + " ".join(
                f"{int(val):>5}" if val != np.inf and val.is_integer() else f"{val:>5.1f}" if val != np.inf else "  inf"
                for val in row
            )
            print(row_str)

    def visualize_graph(self, s=None, t=None, highlight_path=None):
        G = nx.Graph()
        for node_id, (x, y) in self.coordinates.items():
            G.add_node(node_id, pos=(x, y))

        for edge in self.edge_list:
            u = int(edge.start_vertex.name)
            v = int(edge.target_vertex.name)
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=edge.weight)

        pos = nx.get_node_attributes(G, 'pos')
        edge_labels = nx.get_edge_attributes(G, 'weight')

        plt.figure(figsize=(8, 6))
        node_colors = ['green' if node == s else 'red' if node == t else 'lightblue' for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(G, pos, width=1)

        if highlight_path and len(highlight_path) > 1:
            path_edges = list(zip(highlight_path, highlight_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=3)

        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{d['weight']:.1f}" for e, d in G.edges.items()},
                                     font_size=8)

        plt.title("Graph Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":

    def shortest_path(graph_path: str):
        g, start, end = Graph.from_txt_file(sys.argv[1])

        graph = g.adjacency_list()
        nodes = g.coordinates

        return graph, nodes


    #shortest_path("test1.txt") #To satisfy header from navodila
    start_time = time.time()
    if len(sys.argv) > 1:
        g, start, end = Graph.from_txt_file(sys.argv[1])
    else:
        g = Graph(5)
        g.add_edge(1, 2, 1)
        g.add_edge(2, 5, 1)
        g.add_edge(2, 3, 1)
        g.add_edge(3, 4, 1)
        g.add_edge(4, 5, 1)

        start = 1
        end = 5

    #g.print_graph_info()
    #print("")
    #print("\n Adjacency matrix:")
    #g.print_adjacency_matrix()

    #g.to_txt_file("output_graph.txt", k=0, s=start, t=end) #If u want to save our graph to txt file
    g.shortest_path_internal(start, end)
    # print("Cena:", g.Cena)

    # Vizualize the whole graph
    #g.visualize_graph(s=1, t=4)

    # This will come usefull when our shortest_path algorithm will work, since we will be able to highlight shortest path
    #g.visualize_graph(s=1, t=4, highlight_path=g.get_shortest_path(5))

    print("Path:", g.get_shortest_path(end))  # Output like: [0, 2, 3, 4]
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

