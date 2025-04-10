# Super fast walk trough the graph
The aim of this algorithm is to find the shortes walk trough the graph as fast as possible. 

## Reading format
The algorithm recieves data in format as follows, where n is number of nodes, m is number od edges, k is an arbitrary number, s is the id of the starting node and t is the id of the target node. 
```
1 | n m k s t
2 | id1 x1 y1
3 | id2 x2 y2
  | ...
n + 1 | idn xn yn
n + 2 | id1_u id1_v
n + 3 | id2_u id2_v
      | ...
n + m + 1 | idm_u idm_v
```

A short example of a graph with 5 nodes and 5 edges: 
```
5 5 1 1 4
1 1 1
2 1 2
3 1 3
4 2 3
5 2 2
1 2
2 3
2 5
3 4
4 5
```

## Output
The algorithm provides result to standard output in format as follows: 
```
- number of visitations to the nodes (multiple visits to one node count as multiple)
- lenght of shortest walk |p|
- ids of nodes in the shortest walk

```
An example output to standard output would be: 
```
5
5
1 2 5 4
```
