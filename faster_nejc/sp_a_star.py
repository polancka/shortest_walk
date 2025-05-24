import heapq
import math
import os
import time
from heapdict import heapdict
import numpy as np
import matplotlib.pyplot as plt
import statistics
import re
import sys

# Store results
filenames = []
execution_times = []
path_lengths = []
nodes_visited = []

def edge_distance(x1, y1, x2, y2, k):
    '''
    Calculates distance between two points using D(u, v, k) = (|xu - xv| ^k + yu - yv |^k )^(1/k) formula
    '''
    if k == -1:
        return max(abs(x1 - x2), abs(y1 - y2)) 
    elif k == 0:
        return 1
    return (abs(x1 - x2)**k + abs(y1 - y2)**k)**(1.0 / k)

def read_graph(filename):
    '''
    Reads graph information from a txt file, undirected graph with non-negative weighted edges
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()

    n, m, k_str, s, t = lines[0].split()
    n, m, s, t = int(n), int(m), int(s), int(t)
    k = float(k_str)

    node_coords = {}
    for i in range(1, n + 1):
        node_id, x, y = lines[i].split()
        node_coords[int(node_id)] = (float(x), float(y))

    graph = {node_id: [] for node_id in node_coords}

    for i in range(n + 1, n + m + 1):
        u, v = map(int, lines[i].split())
        x1, y1 = node_coords[u]
        x2, y2 = node_coords[v]
        cost = edge_distance(x1, y1, x2, y2, k)
        graph[u].append((v, cost))
        graph[v].append((u, cost)) 

    return graph, s, t, n, node_coords

def heuristic(u, target_coords, node_coords):
    x1, y1 = node_coords[u]
    x2, y2 = target_coords
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def shortest_path_a_star(graph, s, t, n, node_coords):
    if s == t:
        return 1, 0.0, [s]

    # Initialize data structures
    dist_fwd = np.full(n + 1, np.inf)
    dist_bwd = np.full(n + 1, np.inf)
    dist_fwd[s] = 0
    dist_bwd[t] = 0

    pq_fwd = heapdict()
    pq_bwd = heapdict()
    pq_fwd[s] = 0 + heuristic(s, node_coords[t], node_coords)
    pq_bwd[t] = 0 + heuristic(t, node_coords[s], node_coords)

    pred_fwd = np.full(n + 1, -1, dtype=int)
    pred_bwd = np.full(n + 1, -1, dtype=int)
    visited_fwd = np.zeros(n + 1, dtype=bool)
    visited_bwd = np.zeros(n + 1, dtype=bool)

    best_length = np.inf
    meeting_node = -1
    total_visits = 0

    while pq_fwd and pq_bwd:
        # Early stopping condition
        if pq_fwd.peekitem()[1] + pq_bwd.peekitem()[1] >= best_length:
            break

        # Forward search
        u_fwd, _ = pq_fwd.popitem()
        if visited_fwd[u_fwd]:
            continue
        visited_fwd[u_fwd] = True
        total_visits += 1

        if visited_bwd[u_fwd]:
            new_length = dist_fwd[u_fwd] + dist_bwd[u_fwd]
            if new_length < best_length:
                best_length = new_length
                meeting_node = u_fwd

        for v, w in graph[u_fwd]:
            if dist_fwd[u_fwd] + w < dist_fwd[v]:
                dist_fwd[v] = dist_fwd[u_fwd] + w
                pred_fwd[v] = u_fwd
                pq_fwd[v] = dist_fwd[v] + heuristic(v, node_coords[t], node_coords)

        # Backward search
        u_bwd, _ = pq_bwd.popitem()
        if visited_bwd[u_bwd]:
            continue
        visited_bwd[u_bwd] = True
        total_visits += 1

        if visited_fwd[u_bwd]:
            new_length = dist_fwd[u_bwd] + dist_bwd[u_bwd]
            if new_length < best_length:
                best_length = new_length
                meeting_node = u_bwd

        for v, w in graph[u_bwd]:
            if dist_bwd[u_bwd] + w < dist_bwd[v]:
                dist_bwd[v] = dist_bwd[u_bwd] + w
                pred_bwd[v] = u_bwd
                pq_bwd[v] = dist_bwd[v] + heuristic(v, node_coords[s], node_coords)

    if meeting_node == -1:
        return total_visits, -1, []

    # Reconstruct path
    path = []
    node = meeting_node
    while node != -1:
        path.append(node)
        node = pred_fwd[node]
    path.reverse()
    node = pred_bwd[meeting_node]
    while node != -1:
        path.append(node)
        node = pred_bwd[node]

    return total_visits, best_length, path

def shortest_path(filename):
    graph, s, t, n, node_coords = read_graph(filename)
    if s == t:
        return 1, 0.0, [s]
    return shortest_path_a_star(graph, s, t, n, node_coords)

def numerical_sort_key(s):
    # Split the string into parts of digits and non-digits, convert digits to integers
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def process_file(filepath):
    start_time = time.time()
    visits, length, path = shortest_path(filepath)
    end_time = time.time()
    ex_time = end_time - start_time

    filenames.append(os.path.relpath(filepath))
    execution_times.append(ex_time)
    path_lengths.append(length)
    nodes_visited.append(visits)
    print(f"File: {filepath}")
    print(f"Time: {ex_time:.6f}s, Visits: {visits}, Length: {length}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        if not os.path.isdir(folder_path):
            print("Provided path is not a directory.")
            sys.exit(1)

        for root, dirs, files in os.walk(folder_path):
            dirs.sort(key=numerical_sort_key)
            files = sorted(files, key=numerical_sort_key)
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        process_file(file_path)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}\n")

        # Generate simple x-axis labels (1, 2, 3, ..., n)
        x_labels = list(range(1, len(execution_times) + 1))

        # Execution Time Histogram
        plt.figure(figsize=(10, 4))
        plt.bar(x_labels, execution_times, color='skyblue')
        plt.xlabel("Instance")
        plt.ylabel("Execution Time (s)")
        plt.title("Execution Time per Instance")
        plt.xticks(ticks=x_labels, labels=x_labels)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

        # Path Length Histogram
        plt.figure(figsize=(10, 4))
        plt.bar(x_labels, path_lengths, color='lightgreen')
        plt.xlabel("Instance")
        plt.ylabel("Path Length")
        plt.title("Path Length per Instance")
        plt.xticks(ticks=x_labels, labels=x_labels, rotation=90)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

        # Nodes Visited Histogram
        plt.figure(figsize=(10, 4))
        plt.bar(x_labels, nodes_visited, color='salmon')
        plt.xlabel("Instance")
        plt.ylabel("Nodes Visited")
        plt.title("Nodes Visited per Instance")
        plt.xticks(ticks=x_labels, labels=x_labels, rotation=90)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

        total_time = sum(execution_times)
        average_time = total_time / len(execution_times)
        std_dev_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0

        average_visits = sum(nodes_visited) / len(nodes_visited)
        average_length = sum(path_lengths) / len(path_lengths)

        print("\n--- Summary Statistics ---")
        print(f"Total execution time: {total_time:.6f} seconds")
        print(f"Average execution time per instance: {average_time:.6f} seconds")
        print(f"Standard deviation of execution times: {std_dev_time:.6f} seconds")
        print(f"Average number of nodes visited: {average_visits:.2f}")
        print(f"Average path length: {average_length:.2f}")

    else:
        print("Please provide the directory path: python sp.py path/to/folder")