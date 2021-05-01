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
        c = 3
    else:
        k_val = 100
        c_val = 5

    s = 0
    t = V_G - 1

    """ STEP 1: Find Longest Path """

    #Find approximated longest path in G as L.
    L = nx.Graph()
    L_path = semi_longest_path(G, source=s, target=t, num_sample=3000)
    print(L_path)
    #Check whether k, c constraints are met
    L_E = len(L_path) - 1
    L_V = len(L_path)
    if E_G - k_val <= L_E and V_G - c_val <= L_V:
        #k,c constraints are met. Return k, c values for G -> L
        print("GSDGSDGS")
        return None
        #Return k, c values for G -> L

    #Construct L and R = G - L
    R = G.copy()
    for i in range(len(L_path) - 1):
        u = L_path[i]
        v = L_path[i+1]
        w_uv = G[u][v]["weight"]
        #print(u, v, w_uv)
        L.add_edge(u, v, weight=w_uv)
        R.remove_edge(u, v)
    #drawGraph(L, "L", False)
    #drawGraph(R, "R", True)

    """ STEP 2: Make Longest Path the Only Path """
    #Dinitz Algorithm go Brr
    #Construct R_prime, which is R but with all edge weights set as 1
    R_prime = R.copy()
    nx.set_edge_attributes(R_prime, values = 1, name = 'weight')
    #drawGraph(R_prime, "R_prime", True)
    #TODO: Find mincut value... I tried but failed - John

    """ STEP 3: Convert `c` to additional `k` """
    #Oh yeah take out them vertices

    """ STEP 4: Minimize Loss """
    #Is this L O S S ?
    return None

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
        if num_sample == 0:
            break
        path_length = nx.path_weight(G, path, weight="weight")
        if path_length > longest_path_length:
            longest_path_length = path_length
            longest_path = path
        num_sample -= 1
    print(longest_path_length)
    return longest_path

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    solve(G) #Delete this line
    #Uncomment everything below this line
    #c, k = solve(G)
    #assert is_valid_solution(G, c, k)
    #print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
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
