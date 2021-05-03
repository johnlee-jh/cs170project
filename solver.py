import stat

import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import os.path
import glob
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import tree
from networkx.algorithms.flow import dinitz
from networkx.algorithms.flow import edmonds_karp
MIN_VALUE = -10000000
MAX_VALUE = 10000000
k_val = 0
c_val = 0

def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """
    c1, k1 = naive1(G)
    c2, k2 = naive2(G)
    if calculate_score(G, c1, k1) > calculate_score(G, c2, k2):
        return c1, k1
    else:
        return c2, k2


def naive1(G):
    V_G = len(G.nodes)
    if (V_G >= 20 and V_G <= 30):
        k_val = 15
        c_val = 1
    elif (V_G > 30 and V_G <= 50):
        k_val = 30
        c_val = 3
    else:
        k_val = 100
        c_val = 5

    s = 0
    t = V_G - 1
    H = G.copy()
    delete_nodes = []
    delete_edges = []
    for i in range(c_val):
        least = MIN_VALUE
        delete_node = 0
        curr_short_path = nx.dijkstra_path(H, s, t, weight='weight')
        for node in curr_short_path:
            if node != s and node != t:
                edges = list(H.edges(node, data=True))
                H.remove_node(node)
                if nx.is_connected(H):
                    path = nx.dijkstra_path(H, s, t, weight='weight')
                    path_weight = nx.path_weight(H, path, weight='weight')
                    if path_weight > least:
                        least = path_weight
                        delete_node = node
                H.add_node(node)
                for e in edges:
                    H.add_edge(e[0], e[1], weight=e[2]['weight'])
        if delete_node != s and delete_node != t:
            delete_nodes.append(delete_node)
            H.remove_node(delete_node)

    for i in range(k_val):
        A = H.copy()
        for j in range(i + 1):
            current = nx.dijkstra_path(A, s, t, weight='weight')
            edges = []
            for a in range(len(current) - 1):
                u = current[a]
                v = current[a + 1]
                weight = {'weight': G[u][v]['weight']}
                edges.append((u, v, weight))

            least = MIN_VALUE
            edge_delete_one_iter = None
            for edge in edges:
                A.remove_edge(edge[0], edge[1])
                if nx.is_connected(H):
                    try:
                        current = nx.dijkstra_path(A, s, t, weight='weight')
                        current_weight = nx.path_weight(A, current, weight='weight')
                        if current_weight > least:
                            least = current_weight
                            edge_delete_one_iter = edge
                    except nx.NetworkXNoPath:
                        pass
                A.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
            if edge_delete_one_iter != None:
                A.remove_edge(edge_delete_one_iter[0], edge_delete_one_iter[1])
                if i == k_val - 1:
                    delete_edges.append((edge_delete_one_iter[0], edge_delete_one_iter[1]))

    if is_valid_solution(G, delete_nodes, delete_edges):
        return delete_nodes, delete_edges
    else:
        return [], []

def naive2(G):
    V_G = len(G.nodes)
    if (V_G >= 20 and V_G <= 30):
        k_val = 15
        c_val = 1
    elif (V_G > 30 and V_G <= 50):
        k_val = 30
        c_val = 3
    else:
        k_val = 100
        c_val = 5

    s = 0
    t = V_G - 1
    H = G.copy()
    delete_nodes = []
    delete_edges = []
    for i in range(c_val):
        least = MIN_VALUE
        delete_node = 0
        for node in H:
            if node != s and node != t:
                edges = list(H.edges(node, data=True))
                H.remove_node(node)
                if nx.is_connected(H):
                    path = nx.dijkstra_path(H, s, t, weight='weight')
                    path_weight = nx.path_weight(H, path, weight='weight')
                    if path_weight > least:
                        least = path_weight
                        delete_node = node
                H.add_node(node)
                for e in edges:
                    H.add_edge(e[0], e[1], weight=e[2]['weight'])
        if delete_node != s and delete_node != t:
            delete_nodes.append(delete_node)
            H.remove_node(delete_node)

    A = H.copy()
    for i in range(k_val):

        least = MIN_VALUE
        edge_delete_one_iter = None
        edges = list(A.edges(data=True))
        for edge in edges:
            A.remove_edge(edge[0], edge[1])
            if nx.is_connected(H):
                try:
                    current = nx.dijkstra_path(A, s, t, weight='weight')
                    current_weight = nx.path_weight(A, current, weight='weight')
                    if current_weight > least:
                        least = current_weight
                        edge_delete_one_iter = edge
                except nx.NetworkXNoPath:
                    pass
            A.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        if edge_delete_one_iter != None:
            A.remove_edge(edge_delete_one_iter[0], edge_delete_one_iter[1])
            delete_edges.append((edge_delete_one_iter[0], edge_delete_one_iter[1]))
    if is_valid_solution(G, delete_nodes, delete_edges):
        return delete_nodes, delete_edges
    else:
        return [], []

def drawGraph(G, filename, detail):
    """
    Draw the graph G, and store the drawing into the visualizations folder.
    G -> Graph to visualize
    filename -> Filepath under visualizations/ to store drawing in
    detail -> True if you want v & e numbers, False for a plain graph
    """
    if (detail == True):
        pos=nx.spring_layout(G)
        nx.draw_networkx(G,pos)
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        plt.savefig("visualizations/" + filename + ".jpg")
    else:
        nx.draw_networkx(G)
        plt.savefig("visualizations/" + filename + ".jpg")


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in
#
# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     # solve(G) #Delete this line
#     #Uncomment everything below this line
#     c, k = solve(G)
#     assert is_valid_solution(G, c, k)
#     print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
#     write_output_file(G, c, k, 'outputs/small-1.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
    # inputs = glob.glob('inputs\\small/*')
    # inputs.extend(glob.glob('inputs\\medium/*'))
    # # inputs.extend(glob.glob('inputs\\large/*'))
    # for input_path in inputs:
    #     output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
    #     G = read_input_file(input_path)
    #     c, k = solve(G)
    #     print(c)
    #     print('it went here')
    #     assert is_valid_solution(G, c, k)
    #     distance = calculate_score(G, c, k)
    #     write_output_file(G, c, k, output_path)

if __name__ == '__main__':
    assert len(sys.argv) == 1
    for i in range(1, 301):
        input_path = 'inputs/small/small-' + str(i) + '.in'
        output_path = 'outputs/small/small-' + str(i) + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
        write_output_file(G, c, k, output_path)
    for i in range(1, 301):
        input_path = 'inputs/medium/medium-' + str(i) + '.in'
        output_path = 'outputs/medium/medium-' + str(i) + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
        write_output_file(G, c, k, output_path)
    for i in range(1, 301):
        input_path = 'inputs/large/large-' + str(i) + '.in'
        output_path = 'outputs/large/large-' + str(i) + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
        write_output_file(G, c, k, output_path)

