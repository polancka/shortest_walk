import heapq
import time
import math
import sys


def edge_distance(x1, y1, x2, y2, k):
    if k == -1:
        return max(abs(x1 - x2), abs(y1 - y2))
    return (abs(x1 - x2)**k + abs(y1 - y2)**k)**(1.0 / k)


def read_graph(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    n, m, k_str, s_raw, t_raw = lines[0].split()
    n, m = int(n), int(m)
    k = float(k_str)

    # Step 1: Map real node IDs to contiguous indices
    node_id_map = {}
    node_coords_raw = {}

    for i in range(1, n + 1):
        node_id, x, y = lines[i].split()
        node_id = int(node_id)
        node_id_map[node_id] = len(node_id_map)
        node_coords_raw[node_id] = (float(x), float(y))

    # Translate s and t
    s = node_id_map[int(s_raw)]
    t = node_id_map[int(t_raw)]

    # Step 2: Now store coordinates using mapped indices
    coords = [None] * n
    for real_id, index in node_id_map.items():
        coords[index] = node_coords_raw[real_id]

    # Step 3: Build graph using mapped indices
    graph = [[] for _ in range(n)]

    for i in range(n + 1, n + m + 1):
        u_raw, v_raw = map(int, lines[i].split())
        u = node_id_map[u_raw]
        v = node_id_map[v_raw]
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        cost = edge_distance(x1, y1, x2, y2, k)
        graph[u].append((v, cost))
        graph[v].append((u, cost))

    return graph, s, t, n


def reconstruct_path(meeting_node, pred_fwd, pred_bwd):
    path = []

    node = meeting_node
    while node is not None:
        path.append(node)
        node = pred_fwd[node]
    path.reverse()

    node = pred_bwd[meeting_node]
    while node is not None:
        path.append(node)
        node = pred_bwd[node]

    return path


def bidirectional_dijkstra(graph, s, t, n):
    if s == t:
        return 1, 0.0, [s]

    dist_fwd = [math.inf] * n
    dist_bwd = [math.inf] * n
    pred_fwd = [None] * n
    pred_bwd = [None] * n

    dist_fwd[s] = 0
    dist_bwd[t] = 0

    pq_fwd = [(0, s)]
    pq_bwd = [(0, t)]

    visited_fwd = [False] * n
    visited_bwd = [False] * n

    total_visits = 0
    best = math.inf
    meeting_node = None

    while pq_fwd or pq_bwd:
        # Forward step
        if pq_fwd:
            d_u, u = heapq.heappop(pq_fwd)
            if visited_fwd[u]:
                continue
            visited_fwd[u] = True
            total_visits += 1

            if visited_bwd[u]:
                path_cost = dist_fwd[u] + dist_bwd[u]
                if path_cost < best:
                    best = path_cost
                    meeting_node = u
                    break

            for v, weight in graph[u]:
                alt = dist_fwd[u] + weight
                if alt < dist_fwd[v]:
                    dist_fwd[v] = alt
                    pred_fwd[v] = u
                    heapq.heappush(pq_fwd, (alt, v))

        # Backward step
        if pq_bwd:
            d_u, u = heapq.heappop(pq_bwd)
            if visited_bwd[u]:
                continue
            visited_bwd[u] = True
            total_visits += 1

            if visited_fwd[u]:
                path_cost = dist_fwd[u] + dist_bwd[u]
                if path_cost < best:
                    best = path_cost
                    meeting_node = u
                    break

            for v, weight in graph[u]:
                alt = dist_bwd[u] + weight
                if alt < dist_bwd[v]:
                    dist_bwd[v] = alt
                    pred_bwd[v] = u
                    heapq.heappush(pq_bwd, (alt, v))

    if meeting_node is None:
        return total_visits, -1, []

    path = reconstruct_path(meeting_node, pred_fwd, pred_bwd)
    return total_visits, best, path


def shortest_path(filename):
    graph, s, t, n = read_graph(filename)
    return bidirectional_dijkstra(graph, s, t, n)


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
