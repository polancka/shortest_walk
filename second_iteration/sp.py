import heapq
import time
import math
import sys

#To make it faster
import numpy as np
from numba import jit, njit
from numba.typed import List


def edge_distance(x1, y1, x2, y2, k):
    '''
    Calculates distance between two points using D(u, v, k) = (|xu - xv| ^k + yu - yv |^k )^(1/k) formula
    '''
    if k == -1:
        return max(abs(x1 - x2), abs(y1 - y2)) 
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

    return graph, s, t


def reconstruct_path(meeting_node, pred_fwd, pred_bwd):
    '''
    Reconstructs the visited path from both sides
    '''
    path_fwd = []
    node = meeting_node
    while node is not None:
        path_fwd.append(node)
        node = pred_fwd.get(node)
    path_fwd.reverse()

    path_bwd = []
    node = pred_bwd.get(meeting_node)
    while node is not None:
        path_bwd.append(node)
        node = pred_bwd.get(node)

    return path_fwd + path_bwd

def bidirectional_dijkstra(graph, s, t):
    '''
    Core algorithm - Dijkstra bi-directional implementation of finding the shortest
    path from starting to target node, taking into account the weight of the edges
    '''
    if s == t:
        return 1, 0.0, [s]

    dist_fwd = {node: math.inf for node in graph}
    dist_bwd = {node: math.inf for node in graph}
    dist_fwd[s] = 0
    dist_bwd[t] = 0

    pred_fwd = {s: None}
    pred_bwd = {t: None}

    pq_fwd = [(0, s)]
    pq_bwd = [(0, t)]

    visited_fwd = set()
    visited_bwd = set()

    total_visits = 0
    best = math.inf
    meeting_node = None

    while pq_fwd or pq_bwd:
        # Forward search
        if pq_fwd:
            d_fwd, u_fwd = heapq.heappop(pq_fwd)
            total_visits += 1
            if u_fwd in visited_fwd:
                continue
            visited_fwd.add(u_fwd)

            if u_fwd in visited_bwd:
                if dist_fwd[u_fwd] + dist_bwd[u_fwd] < best:
                    best = dist_fwd[u_fwd] + dist_bwd[u_fwd]
                    meeting_node = u_fwd
                    break

            for v, weight in graph[u_fwd]:
                alt = dist_fwd[u_fwd] + weight
                if alt < dist_fwd[v]:
                    dist_fwd[v] = alt
                    pred_fwd[v] = u_fwd
                    heapq.heappush(pq_fwd, (alt, v))

        # Backward search
        if pq_bwd:
            d_bwd, u_bwd = heapq.heappop(pq_bwd)
            total_visits += 1
            if u_bwd in visited_bwd:
                continue
            visited_bwd.add(u_bwd)

            if u_bwd in visited_fwd:
                if dist_fwd[u_bwd] + dist_bwd[u_bwd] < best:
                    best = dist_fwd[u_bwd] + dist_bwd[u_bwd]
                    meeting_node = u_bwd
                    break

            for v, weight in graph[u_bwd]:
                alt = dist_bwd[u_bwd] + weight
                if alt < dist_bwd[v]:
                    dist_bwd[v] = alt
                    pred_bwd[v] = u_bwd
                    heapq.heappush(pq_bwd, (alt, v))

    if meeting_node is None:
        return total_visits, -1, []

    path = reconstruct_path(meeting_node, pred_fwd, pred_bwd)
    total_cost = dist_fwd[meeting_node] + dist_bwd[meeting_node]
    return total_visits, total_cost, path


def shortest_path(filename):
    '''
    Wrapper function to load graph from file and compute shortest path using Dijkstra bi-directional
    '''
    graph, s, t = read_graph(filename)
    return bidirectional_dijkstra(graph, s, t)


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
