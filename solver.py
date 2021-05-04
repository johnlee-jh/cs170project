import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from random import random, randint
from os.path import basename, normpath
import glob
import matplotlib.pyplot as plt
from networkx.algorithms import tree
from networkx.algorithms.flow import dinitz
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.connectivity import minimum_st_edge_cut
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

    if (V_G >= 20 and V_G <= 30):
        c1, k1 = naive1(G) #small algorithm 1
        c2, k2 = naive2(G) #small algorithm 2
        if calculate_score(G, c1, k1) > calculate_score(G, c2, k2):
            return c1, k1
        else:
            return c2, k2
    elif (V_G > 30 and V_G <= 50):
        c3, k3 = randomized_helper(G, 150) #medium and large algorithm
        c4, k4 = large_and_medium(G)
        if calculate_score(G, c3, k3) > calculate_score(G, c4, k4):
            return c3, k3
        else:
            return c4, k4
    else:
        c3, k3 = large_and_medium(G) #medium and large algorithm
        return c3, k3
def randomized_helper(G, n):
    max_score = 0
    delete = (None, None)
    for i in range(n):
        c, k = randomized(G)
        score = calculate_score(G, c, k)
        if score > max_score:
            delete = (c, k)
            max_score = score
    return delete[0], delete[1]

def randomized(G):
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

    c = 0
    k = 0
    finalG = G.copy()
    H = G.copy()
    while c < c_val or k < k_val:
        choose = random()
        if choose < .30 and c < c_val:
            least = MIN_VALUE
            delete_node = 0
            idk, shortest_path = nx.single_source_dijkstra(H, s, t)
            for node in shortest_path:
                if node != s and node != t:
                    edges = list(H.edges(node, data=True))
                    H.remove_node(node)
                    if nx.is_connected(H):
                        dist, path = nx.single_source_dijkstra(H, s, t)
                        # path_weight = nx.path_weight(H, path, weight='weight')
                        if dist > least:
                            least = dist
                            delete_node = node
                    H.add_node(node)
                    for e in edges:
                        H.add_edge(e[0], e[1], weight=e[2]['weight'])
            if delete_node != s and delete_node != t:
                delete_nodes.append(delete_node)
                H.remove_node(delete_node)
                for edge in list(delete_edges):
                    if edge[0] == delete_node or edge[1] == delete_node:
                        delete_edges.remove(edge)
                        k = k - 1
            c += 1
        elif choose >= .30 and k < k_val:
            least = MIN_VALUE
            edge_delete_one_iter = None
            edges = []
            current = nx.dijkstra_path(H, s, t, weight='weight')
            for a in range(len(current) - 1):
                u = current[a]
                v = current[a + 1]
                weight = {'weight': G[u][v]['weight']}
                edges.append((u, v, weight))

            for edge in edges:
                H.remove_edge(edge[0], edge[1])
                if nx.is_connected(H):
                    try:
                        current = nx.dijkstra_path(H, s, t, weight='weight')
                        current_weight = nx.path_weight(H, current, weight='weight')
                        if current_weight > least:
                            least = current_weight
                            edge_delete_one_iter = edge
                    except nx.NetworkXNoPath:
                        pass
                H.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
            if edge_delete_one_iter != None:
                if edge_delete_one_iter[0] in delete_nodes or edge_delete_one_iter[1] in delete_nodes:
                    pass
                else:
                    H.remove_edge(edge_delete_one_iter[0], edge_delete_one_iter[1])
                    delete_edges.append((edge_delete_one_iter[0], edge_delete_one_iter[1]))
                    k += 1
    #print(c,k)
    if is_valid_solution(G, delete_nodes, delete_edges):
        return delete_nodes, delete_edges
    else:
        return [], []

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

def large_and_medium(G):
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

    """Find Longest Path (start) REMOVED
    L = nx.Graph()
    L_path = semi_longest_path(finalG, source=s, target=t, num_sample=10000)
    
    for i in range(len(L_path) - 1):
        u = L_path[i]
        v = L_path[i+1]
        w_uv = G[u][v]["weight"]
        L.add_edge(u, v, weight=w_uv, capacity=w_uv)
    """
    if (True): #Large
        while(curr_k < k_val):
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
                    sp_weight = nx.single_source_dijkstra(tempG, e[0], e[1], weight='weight')[0]
                    diff = sp_weight - finalG[e[0]][e[1]]['weight']
                    if (diff >= currMaxDiff):
                        currMaxE = e
                else:
                    falseCount += 1
            #print(falseCount)
            if (falseCount == len(S.edges)):
                break
            #print(curr_k)
            finalG.remove_edge(*currMaxE)
            curr_k += 1
    else: #Medium
        """ 
        while(curr_k < k_val):
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
                    sp_weight = nx.single_source_dijkstra(tempG, e[0], e[1], weight='weight')[0]
                    diff = sp_weight - finalG[e[0]][e[1]]['weight']
                    if (diff >= currMaxDiff):
                        currMaxE = e
                else:
                    falseCount += 1
            if (falseCount == len(S.edges)):
                break
            finalG.remove_edge(*currMaxE)
        """        
        #print("gsd")
        while (curr_k < k_val):
            if (True):
                #print("yeet1")
                toRemove = findEdge(finalG, finalG, 1, True, s, t, [])
                #print(toRemove)
                finalG.remove_edge(*toRemove)
                curr_k += 1
        

    #print(L_path)
    #print(nx.single_source_dijkstra(finalG, s, t, weight='weight')[0])
    
    #print(nx.single_source_dijkstra(G, s, t, weight='weight'))
    #print(nx.single_source_dijkstra(finalG, s, t, weight='weight'))
    
    c = vertex_diff(G, finalG)
    k = edge_diff(G, finalG, c)
    #print(c,k)
    return c, k






def findEdge(originalG, currG, depth, saveEdge, s, t, e_list):

    S = nx.Graph()
    S_val, S_path = nx.single_source_dijkstra(currG, s, t, weight='weight')

    for i in range(len(S_path) - 1):
        u = S_path[i]
        v = S_path[i+1]
        w_uv = currG[u][v]["weight"]
        S.add_edge(u, v, weight=w_uv)

    if (depth == 0):
        sp_weight = 0
        for i in range(len(e_list)):
            e = e_list[i]
            if (i == 0):
                sp_weight += nx.single_source_dijkstra(currG, e[0], e[1], weight='weight')[0]
                sp_weight -= originalG[e[0]][e[1]]['weight']
        return sp_weight

    currScore = 0
    optimalEdge = list(S.edges)[0]

    for e in S.edges:
        futureG = currG.copy()
        futureG.remove_edge(*e)
        if (nx.has_path(futureG, s, t)):
            e_list.append(e)
            score = findEdge(originalG, futureG, depth - 1, False, s, t, e_list)
            #score += heuristic(S, futureG, e, s, t)
            score = round(score, 4)
            #print(score)
            if (score > currScore):
                currScore = score
                optimalEdge = e

    if saveEdge:
        return optimalEdge
    else:
        return currScore

def heuristic(S, futureG, e, s, t):
    #sp_weight = nx.single_source_dijkstra(futureG, e[0], e[1], weight='weight')[0]
    #diff = sp_weight - currG[e[0]][e[1]]['weight']
    sp_weight = nx.single_source_dijkstra(futureG, e[0], e[1], weight='weight')[0]
    diff = sp_weight - S[e[0]][e[1]]['weight']
    return diff
    #return nx.single_source_dijkstra(currG, s, t, weight='weight')[0]

def n_highest(currG, edges):
    pass






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


# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     c, k = solve(G)
#     assert is_valid_solution(G, c, k)
#     print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
#     #write_output_file(G, c, k, 'outputs/small-1.out')

"""
# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)

if __name__ == '__main__':
    #inputs = []
    failed_large = [16,51,74,93,100,123,131,177,189,215,252,254]
    failed_medium = [31,46,51,123,186]
    for i in range(300, 301): #max(failed_medium)+1
        if (not (i in failed_medium)):
            print(i)
            input_path = "inputs/large/large-" + str(i) + ".in"    
            output_path = 'outputs/large/large-' + str(i) + '.out'
            G = read_input_file(input_path)
            c, k = solve(G)
            assert is_valid_solution(G, c, k)
            distance = calculate_score(G, c, k)
            write_output_file(G, c, k, output_path)
"""

if __name__ == '__main__':
    assert len(sys.argv) == 1
    # for i in range(1, 301):
    #     input_path = 'inputs/small/small-' + str(i) + '.in'
    #     output_path = 'outputs/small/small-' + str(i) + '.out'
    #     G = read_input_file(input_path)
    #     c, k = solve(G)
    #     assert is_valid_solution(G, c, k)
    #     print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
    #     write_output_file(G, c, k, output_path)
    for i in range(1, 301):
        input_path = 'inputs/medium/medium-' + str(i) + '.in'
        output_path = 'outputs/medium/medium-' + str(i) + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        print("Shortest Path Difference for medium-" + str(i) + ": {}".format(calculate_score(G, c, k)))
        write_output_file(G, c, k, output_path)
    # for i in range(1, 301):
    #     input_path = 'inputs/large/large-' + str(i) + '.in'
    #     output_path = 'outputs/large/large-' + str(i) + '.out'
    #     G = read_input_file(input_path)
    #     c, k = solve(G)
    #     assert is_valid_solution(G, c, k)
    #     print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
    #     write_output_file(G, c, k, output_path)