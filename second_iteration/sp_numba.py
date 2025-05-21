import heapq
import time
import math
import sys
import numpy as np
from numba import njit

def edge_distance(x1, y1, x2, y2, k):
    '''
    Calculates distance between two points using D(u, v, k) = (|xu - xv| ^k + |yu - yv| ^k )^(1/k) formula
    '''
    if k == -1:
        return max(abs(x1 - x2), abs(y1 - y2)) 
    return (abs(x1 - x2)**k + abs(y1 - y2)**k)**(1.0 / k)

def read_graph(filename):
    '''
    Reads graph information from a txt file, undirected graph with non-negative weighted edges.
    Converts graph into CSR format for numba.
    Returns:
        n: number of nodes
        offsets: array of length n+1 with offsets into adj and costs arrays
        adj: concatenated adjacency lists
        costs: concatenated edge costs
        s, t: start and target nodes, zero-based
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()

    n, m, k_str, s, t = lines[0].split()
    n, m, s, t = int(n), int(m), int(s) - 1, int(t) - 1  # zero-based indexing
    k = float(k_str)

    node_coords = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        node_id, x, y = lines[i+1].split()
        node_coords[int(node_id) - 1] = [float(x), float(y)]

    # Build adjacency list in dict first
    adjacency = [[] for _ in range(n)]
    for i in range(m):
        u, v = map(int, lines[n + 1 + i].split())
        u -= 1
        v -= 1
        cost = edge_distance(node_coords[u][0], node_coords[u][1], node_coords[v][0], node_coords[v][1], k)
        adjacency[u].append((v, cost))
        adjacency[v].append((u, cost))

    # Convert adjacency list to CSR format arrays
    edge_count = sum(len(adj) for adj in adjacency)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        offsets[i + 1] = offsets[i] + len(adjacency[i])

    adj = np.zeros(edge_count, dtype=np.int64)
    costs = np.zeros(edge_count, dtype=np.float64)

    idx = 0
    for i in range(n):
        for (nbr, cost) in adjacency[i]:
            adj[idx] = nbr
            costs[idx] = cost
            idx += 1

    return n, offsets, adj, costs, s, t

@njit
def bidirectional_dijkstra_numba(n, offsets, adj, costs, s, t):
    dist_fwd = np.full(n, 1e20, dtype=np.float64)
    dist_bwd = np.full(n, 1e20, dtype=np.float64)
    dist_fwd[s] = 0.0
    dist_bwd[t] = 0.0

    pred_fwd = np.full(n, -1, dtype=np.int64)
    pred_bwd = np.full(n, -1, dtype=np.int64)

    visited_fwd = np.zeros(n, dtype=np.bool_)
    visited_bwd = np.zeros(n, dtype=np.bool_)

    # use Numba-compatible priority queues
    pq_fwd = [(0.0, s)]
    pq_bwd = [(0.0, t)]

    total_visits = 0
    best = 1e20
    meeting_node = -1

    while pq_fwd or pq_bwd:
        # Forward
        if pq_fwd:
            d_fwd, u_fwd = heapq.heappop(pq_fwd)
            if visited_fwd[u_fwd]:
                continue
            visited_fwd[u_fwd] = True
            total_visits += 1

            if visited_bwd[u_fwd]:
                alt_path = dist_fwd[u_fwd] + dist_bwd[u_fwd]
                if alt_path < best:
                    best = alt_path
                    meeting_node = u_fwd
                    break

            for i in range(offsets[u_fwd], offsets[u_fwd + 1]):
                v = adj[i]
                cost = costs[i]
                alt = dist_fwd[u_fwd] + cost
                if alt < dist_fwd[v]:
                    dist_fwd[v] = alt
                    pred_fwd[v] = u_fwd
                    heapq.heappush(pq_fwd, (alt, v))

        # Backward
        if pq_bwd:
            d_bwd, u_bwd = heapq.heappop(pq_bwd)
            if visited_bwd[u_bwd]:
                continue
            visited_bwd[u_bwd] = True
            total_visits += 1

            if visited_fwd[u_bwd]:
                alt_path = dist_fwd[u_bwd] + dist_bwd[u_bwd]
                if alt_path < best:
                    best = alt_path
                    meeting_node = u_bwd
                    break

            for i in range(offsets[u_bwd], offsets[u_bwd + 1]):
                v = adj[i]
                cost = costs[i]
                alt = dist_bwd[u_bwd] + cost
                if alt < dist_bwd[v]:
                    dist_bwd[v] = alt
                    pred_bwd[v] = u_bwd
                    heapq.heappush(pq_bwd, (alt, v))

    return total_visits, best, meeting_node, pred_fwd, pred_bwd

def reconstruct_path(meeting_node, pred_fwd, pred_bwd):
    if meeting_node == -1:
        return []

    path_fwd = []
    node = meeting_node
    while node != -1:
        path_fwd.append(node)
        node = pred_fwd[node]
    path_fwd.reverse()

    path_bwd = []
    node = pred_bwd[meeting_node]
    while node != -1:
        path_bwd.append(node)
        node = pred_bwd[node]

    # Convert zero-based indexing to one-based node IDs
    path = [x + 1 for x in path_fwd + path_bwd]
    return path

def shortest_path(filename):
    n, offsets, adj, costs, s, t = read_graph(filename)
    total_visits, best, meeting_node, pred_fwd, pred_bwd = bidirectional_dijkstra_numba(n, offsets, adj, costs, s, t)
    path = reconstruct_path(meeting_node, pred_fwd, pred_bwd)
    return total_visits, best, path

if __name__ == "__main__":
    start_time = time.time()
    if len(sys.argv) > 1:
        visits, length, path = shortest_path(sys.argv[1])
        print(visits)
        print(length)
        print(" ".join(map(str, path)))
    else:
        print("Please provide the graph txt file: python sp.py filename.txt")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
