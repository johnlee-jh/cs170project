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

    c = []
    k = []

    s = 0
    t = V_G - 1

    finalG = G.copy()

    """ STEP 1: Find Longest Path """

    #Find approximated longest path in G as L.
    L = nx.Graph()
    L_path = semi_longest_path(finalG, source=s, target=t, num_sample=10000)
    #Check whether k, c constraints are met
    E_L = len(L_path) - 1
    V_L = len(L_path)
    if E_G - E_L <= k_val and V_G - V_L <= c_val:
        #k,c constraints are met. Return k, c values for G -> L
        print("return this thingy")
        return None
        #Return k, c values for G -> L

    #Construct L and R = G - L
    R = G.copy()
    for i in range(len(L_path) - 1):
        u = L_path[i]
        v = L_path[i+1]
        w_uv = G[u][v]["weight"]
        L.add_edge(u, v, weight=w_uv)
        R.remove_edge(u, v)

    """ STEP 2: Make Longest Path the Only Path """
    #Dinitz Algorithm go Brr
    #Construct R_prime, which is R but with all edge weights set as 1
    Rprime = nx.Graph()
    E_Rprime = [e for e in R.edges]
    Rprime.add_edges_from(E_Rprime, capacity=1, weight=1)
    R_mincut, R_partition = nx.minimum_cut(Rprime, s, t)
    reachable, non_reachable = R_partition
    cutset = []
    for u, nbrs in ((n, R[n]) for n in reachable):
        cutset += [(u, v) for v in nbrs if v in non_reachable]
    
    """
    for e in cutset:
        finalG.remove_edge(*e)

    for v in L_path:
        if (v==s or v==t):
            pass
        else:
            finalG.remove_node(v)

    #G = G-L-M
    finalG.add_nodes_from(L.nodes)
    finalG.add_edges_from(L.edges)
    for e in finalG.edges:
        u = e[0]
        v = e[1]
        w_uv = G[u][v]["weight"]
        #print(u, v, w_uv)
        finalG.add_edge(u, v, weight=w_uv)
    """
    dijkstra_G = nx.single_source_shortest_path(G, s)
    dijkstra_L = nx.single_source_shortest_path(L, s)
    #print(dijkstra_G)
    #print("------------")
    #print(dijkstra_L)

    shortest_path_L = {}
    for v in range(V_G):
        if (v in dijkstra_L):
            sp_L_v = nx.path_weight(L, dijkstra_L[v], weight="weight")
            sp_L_v = round(sp_L_v, 4)
            shortest_path_L[v] = sp_L_v
        else:
            shortest_path_L[v] = 0

    shortest_path_G = {}
    for v in range(V_G):
        if (v in dijkstra_G):
            sp_G_v = nx.path_weight(G, dijkstra_G[v], weight="weight")
            sp_G_v = round(sp_G_v, 4)
            shortest_path_G[v] = sp_G_v
        else:
            shortest_path_G[v] = 0

    path_diff = {}
    delta_list = []
    curr_G_V = list(finalG.nodes)
    curr_G_V.remove(s)
    curr_G_V.remove(t)
    curr_L_V = list(L.nodes)
    for v in curr_G_V:
        for u in curr_L_V:
            if G.has_edge(u, v):
                if not L.has_edge(u, v):
                    sp_u = shortest_path_G[u]
                    w_uv = G[u][v]["weight"]
                    sp_v = shortest_path_L[v]
                    delta = sp_u + w_uv - sp_v
                    path_diff[delta] = (u, v)
                    delta_list.append(delta)

    delta_list.sort(reverse=True)
    
    

    """
    counter = 0
    while (len(finalG.edges) < E_G - k_val):
        delta_add = delta_list[counter]
        counter += 1
        e_add = path_diff[delta_add]
        finalG.add_edge(*e_add)
    """
    drawGraph(finalG, "fGa", False)
    
    #print(len(shortest_path_G))
    #print(len(shortest_path_L))
    """
    print(edge_diff(G, finalG))
    print(vertex_diff(G, finalG))
    print(len(edge_diff(G, finalG)))
    print(len(vertex_diff(G, finalG)))
    """

    print(nx.single_source_dijkstra(G, s, t, weight='weight'))
    print(nx.single_source_dijkstra(finalG, s, t, weight='weight'))

    """ STEP 4: Minimize Loss """
    #Is this L O S S ?
    
    print(c,k)
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
    simple_paths = nx.all_simple_paths(G, source=source, target=target)
    for path in nx.all_simple_paths(G, source=source, target=target):
        if num_sample == 0:
            break
        path_length = nx.path_weight(G, path, weight="weight")
        if path_length > longest_path_length:
            longest_path_length = path_length
            longest_path = path
        num_sample -= 1
    #print(longest_path_length)
    return longest_path

def edge_diff(G1, G2):
    #G1 should have more edges than G2.
    e_diff = []
    for e in G1.edges:
        u = e[0]
        v = e[1]
        if not G2.has_edge(u,v):
            e_diff.append(e)
    return e_diff

def vertex_diff(G1, G2):
    #G1 should have more vertices than G2.
    v_diff = []
    for v in G1.nodes:
        if not G2.has_node(v):
            v_diff.append(e)
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
