import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob
import matplotlib.pyplot as plt
from networkx.algorithms import tree
from networkx.algorithms.flow import dinitz
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.connectivity import minimum_st_edge_cut


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

    """Initialize variables (start)"""
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
    c = []
    k = []
    curr_c = 0
    curr_k = 0
    s = 0
    t = V_G - 1

    finalG = G.copy()
    """Initialize variables (end)"""

    """Remove Vertices (start)"""
    while(curr_c < c_val):
        w_to_v = {}
        w_list = []
        for v in finalG.nodes:
            if not(v == s or v == t):
                testG = finalG.copy()
                testG.remove_node(v)
                if (nx.has_path(testG, s, t)):
                    total_w, total_p = nx.single_source_dijkstra(testG, s, t, weight='weight')
                    w_list.append(total_w)
                    w_to_v[total_w] = v
        v_remove = w_to_v[max(w_list)]
        finalG.remove_node(v_remove)
        curr_c += 1
    """Remove Vertices (start)"""

    L = nx.Graph()
    L_path = semi_longest_path(finalG, source=s, target=t, num_sample=10000)
    
    for i in range(len(L_path) - 1):
        u = L_path[i]
        v = L_path[i+1]
        w_uv = G[u][v]["weight"]
        L.add_edge(u, v, weight=w_uv, capacity=w_uv)

    while(curr_k < k_val):
        print(curr_k)

        S = nx.Graph()
        S_val, S_path = nx.single_source_dijkstra(finalG, s, t, weight='weight')

        for i in range(len(S_path) - 1):
            u = S_path[i]
            v = S_path[i+1]
            w_uv = G[u][v]["weight"]
            S.add_edge(u, v, weight=w_uv, capacity=w_uv)

        currMaxDiff = 0
        currMaxE = list(finalG.edges)[0]
        falseCount = 0
        for e in S.edges:
            tempG = finalG.copy()
            tempG.remove_edge(*e)
            if (nx.has_path(tempG, s, t)):
                sp_weight, sp_path = nx.single_source_dijkstra(tempG, e[0], e[1], weight='weight')
                diff = sp_weight - finalG[e[0]][e[1]]['weight']
                if (diff >= currMaxDiff):
                    currMaxE = e
            else:
                falseCount += 1
        if (falseCount == len(S.edges)):
            break
        verify = finalG.copy()
        verify.remove_edge(*currMaxE)
        if (nx.has_path(verify, s, t)):
            finalG.remove_edge(*currMaxE)
            curr_k += 1
        else:
            break
    
    #print(nx.single_source_dijkstra(G, s, t, weight='weight'))
    #print(nx.single_source_dijkstra(finalG, s, t, weight='weight'))
    
    c = vertex_diff(G, finalG)
    k = edge_diff(G, finalG, c)
    #print(c,k)
    return c, k

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
        nx.draw_shell(G, with_labels = True)
        plt.savefig("visualizations/" + filename + ".jpg")

def semi_longest_path(graph, source, target, num_sample):
    """
    Randomly sample simple paths from s-t and return the longest
    path out of the sample.
    TODO: Make the random path selector more random (not relying on NX)
    """
    longest_path = []
    longest_path_length = 0
    simple_paths = nx.all_simple_paths(graph, source=source, target=target)
    for path in nx.all_simple_paths(graph, source=source, target=target):
        if num_sample == 0:
            break
        path_length = nx.path_weight(graph, path, weight="weight")
        if path_length > longest_path_length:
            longest_path_length = path_length
            longest_path = path
        num_sample -= 1
    #print(longest_path_length)
    return longest_path

def edge_diff(G1, G2, v_diff):
    #G1 should have more edges than G2.
    e_diff = []
    for e in G1.edges:
        if not (e[0] in v_diff) and not (e[1] in v_diff):
            if not G2.has_edge(*e):
                e_diff.append(e)
    return e_diff

def vertex_diff(G1, G2):
    #G1 should have more vertices than G2.
    v_diff = []
    for v in G1.nodes:
        if not G2.has_node(v):
            v_diff.append(v)
    return v_diff


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    #solve(G) #Delete this line
    #Uncomment everything below this line
    c, k = solve(G)
    assert is_valid_solution(G, c, k)
    print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
    #write_output_file(G, c, k, 'outputs/small-1.out')


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
