import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
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
    V_G = len(G.nodes)
    E_G = len(G.edges)
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

    curr_short_path = nx.dijkstra_path(H, s, t, weight='weight')
    print(curr_short_path)
    for i in range(c_val):
        least = MIN_VALUE
        delete_node = 0
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
            curr_short_path = nx.dijkstra_path(H, s, t, weight='weight')

    A = H.copy()
    for i in range(k_val):
        current = nx.dijkstra_path(A, s, t, weight='weight')
        A = H.copy()
        for j in range(0, i + 1):

            edges = []
            for a in range(len(current) - 1):
                u = current[a]
                v = current[a + 1]
                weight = {'weight' : G[u][v]['weight']}
                edges.append((u,v,weight))

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
                current = nx.dijkstra_path(A, s, t, weight='weight')
                if i == k_val - 1:
                    delete_edges.append((edge_delete_one_iter[0], edge_delete_one_iter[1]))

    return delete_nodes, delete_edges
    # #Find approximated longest path in G as L.
    # L_path, L_weight = semi_longest_path(G, source=s, target=t, num_sample=3000)
    # # print(nx.path_weight(G, L_path, 'weight'))
    # # print(nx.path_weight(G, nx.dijkstra_path(G, s, t, weight='weight'), 'weight'))
    #
    # # print(L_path)
    # #Check whether k, c constraints are met
    # L_E = len(L_path) - 1
    # L_V = len(L_path)
    # # print(L_E)
    # # print(E_G)
    # # print(V_G)
    # # print(L_V)
    # # print(G.edges(0 ,data=True))
    # if E_G - k_val <= L_E and V_G - c_val <= L_V:
    #     delete_nodes = []
    #     delete_edges = []
    #     num = 0
    #     for v in G.nodes:
    #         if num < c_val:
    #             if v not in L_path:
    #                 delete_nodes.append(v)
    #                 num += 1
    #     num_e = 0
    #     for u,v,d in G.edges(data=True):
    #         if num_e < k_val:
    #             if u in L_path:
    #                 index = L_path.index(u)
    #                 if not (index + 1 < len(L_path) and L_path[index + 1] == v):
    #                     delete_edges.append((u,v))
    #                     num_e += 1
    #             elif v in L_path:
    #                 index = L_path.index(v)
    #                 if not (index + 1 < len(L_path) and L_path[index + 1] == u):
    #                     delete_edges.append((v, u))
    #                     num_e += 1
    #     return delete_nodes, delete_edges
    #
    # R = G.copy() # R is actually now the graph of the longest path that still contains the vertices, just disconnected
    #
    # edges_of_longest_path = []
    #
    # for i in range(len(L_path) - 1):
    #     u = L_path[i]
    #     v = L_path[i+1]
    #     w_uv = G[u][v]["weight"]
    #     edges_of_longest_path.append((u, v, {'weight': w_uv}))
    #     edges_of_longest_path.append((v, u, {'weight': w_uv}))
    # for u,v,d in G.edges(data=True):
    #     if (u,v,d) not in edges_of_longest_path:
    #         R.remove_edge(u, v)
    # new_L_path = L_path # after this it will include vertices so that it satisfies c constraint
    #
    # # print(R.edges)
    # # print(nx.dijkstra_path(R, s, t, weight='weight'))
    # if len(G.nodes) - len(L_path) > c_val:
    #     for a in range(len(G.nodes) - len(L_path) - c_val):
    #         nodes = np.setdiff1d(G.nodes, new_L_path)
    #         L_max = MIN_VALUE
    #         node_max = 0
    #         for v in nodes:
    #             edge_to_add = None
    #             node_edges = G.edges(v, data=True)
    #             for e in node_edges:
    #                 if e[0] in new_L_path or e[1] in new_L_path: # select only one edge to connect this vertice to the graph
    #                     R.add_edge(e[0], e[1], weight=e[2]['weight'])
    #                     edge_to_add = e
    #                     break
    #             # R.add_edges_from(node_edges)
    #             L_length = nx.path_weight(R, nx.dijkstra_path(R, s, t, weight='weight'), 'weight')
    #             if L_length > L_max:     # add the best vertices to add to the graph based on djikstras shortest path
    #                 L_max = L_length
    #                 node_max = v
    #             R.remove_edge(edge_to_add[0], edge_to_add[1])
    #         for e in G.edges(node_max, data=True):
    #             if e[0] in new_L_path or e[1] in new_L_path:
    #                 R.add_edge(e[0], e[1], weight=e[2]['weight'])
    #                 break
    #         new_L_path.append(node_max)
    #
    # edge_count = len(R.edges)
    # if len(G.edges) - edge_count > k_val: # if the vertice shifting does not fix edge count, select best edges to add to existing nodes
    #     for a in range(len(G.edges) - edge_count - k_val):
    #         max_edge = None
    #         L_max = MIN_VALUE
    #         edges_to_add = []
    #         for a in new_L_path:
    #             l = list(G.edges(a, data=True))
    #             for u, v, d in l:
    #                 if (u, v, d) not in edges_of_longest_path:
    #                     edges_to_add.append((u, v, d))
    #             # edges_to_add.extend(np.setdiff1d(l, ab)) # add all edges that are still stemming from the vertices in the longest path, dont include edges already in longest path
    #         for e in edges_to_add:
    #             R.add_edge(e[0], e[1], weight=e[2]['weight'])
    #             L_length = nx.path_weight(R, nx.dijkstra_path(R, s, t, weight='weight'), 'weight')
    #             if L_length > L_max:
    #                 L_max = L_length
    #                 max_edge = e
    #             R.remove_edge(e[0], e[1])
    #         R.add_edge(max_edge[0], max_edge[1], weight= max_edge[2]['weight']) # add edge that maximized djikstras to the graph R
    #         edges_of_longest_path.append((max_edge[0], max_edge[1], max_edge[2]))
    #         edges_of_longest_path.append((max_edge[1], max_edge[0], max_edge[2]))
    #
    #
    # delete_nodes = []
    # delete_edges = []
    # for v in G.nodes:
    #     if v not in new_L_path:
    #             delete_nodes.append(v)
    # for e in G.edges():
    #     R_edges = R.edges()
    #     if e not in R_edges:
    #         if e[0] not in delete_nodes and e[1] not in delete_nodes:
    #             delete_edges.append((e))
    # return delete_nodes, delete_edges

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

def semi_longest_path(graph, source, target, num_sample):
    """
    Randomly sample simple paths from s-t and return the longest
    path out of the sample.
    TODO: Make the random path selector more random (not relying on NX)
    """
    longest_path = []
    longest_path_length = 0
    simple_paths = nx.all_simple_paths(G, source=source, target=target)
    for path in nx.all_simple_paths(G, source=source, target=target):
        print(len(path))
        if num_sample == 0:
            break
        path_length = nx.path_weight(G, path, weight="weight")
        if path_length > longest_path_length:
            longest_path_length = path_length
            longest_path = path
        num_sample -= 1
    # print(longest_path_length)
    return longest_path, longest_path_length

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    # solve(G) #Delete this line
    #Uncomment everything below this line
    c, k = solve(G)
    print(c)
    print(k)
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