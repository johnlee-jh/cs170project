import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob


def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """
    length = len(G.nodes)
    if (length >= 20 and length <= 30):
        k = 15
        c = 1
    elif (length > 30 and length <= 50):
        k = 30
        c = 3
    else:
        k = 100
        c = 5
    H = G.copy()
    for u,v,d in H.edges(data=True):
        d['weight'] = 1 / d['weight']
    longest_path_nodes = nx.dijkstra_path(H, 0, len(G.nodes) - 1, weight='weight')
    size_of_edges = len(longest_path_nodes) - 1
    size_of_vertices = size_of_edges + 1
    if len(G.edges) - k <= size_of_edges and len(G.nodes) - c <= size_of_vertices: # i dont think this condition is correct 
        nodes_delete = []
        edges_delete = []
        for node in G.nodes:
            if node not in longest_path_nodes: # add all nodes that are not in the longest path
                nodes_delete.append(node)
        for u,v,d in G.edges():
            if u in longest_path_nodes:
                index = longest_path_nodes.index(u)
                if not (index + 1 < len(longest_path_nodes) - 1 and longest_path_nodes[index + 1] == v):
                    edges_delete.append((u, v, d)) # edge as a tuple
        return nodes_delete, edges_delete

    pass


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    c, k = solve(G)
    assert is_valid_solution(G, c, k)
    print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
    write_output_file(G, c, k, 'outputs/small-1.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)
