import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob
import matplotlib.pyplot as plt
from networkx.algorithms import tree

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
        k_val = 15
        c_val = 1
    elif (V_G > 30 and V_G <= 50):
        k_val = 30
        c = 3
    else:
        k_val = 100
        c_val = 5

    """ STEP 1: Find Longest Path """
    #Gives an approximation of the longest path as L.
    L = nx.Graph()
    longest_path = longest_simple_paths(L, source=0, target=(V_G-1))
    L_E = len(longest_path) - 1
    L_V = len(longest_path)
    if len(G.edges) - k <= L_E and len(G.nodes) - c <= L_V:
        #k,c constraints are satisfied. Return G - L
        pass
    else:
        #k,c constraints aren't satisfied. Time for Step 2
        for i in range(len(longest_path) - 1):
            u = longest_path[i]
            v = longest_path[i+1]
            w_uv = G[u][v]["weight"]
            #print(u, v, w_uv)
            L.add_edge(u, v, weight=w_uv)
    drawGraph(L, "L", True)

    """ STEP 2: Make Longest Path the Only Path """
    
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

def longest_simple_paths(graph, source, target):
    """
    Randomly sample simple paths from s-t and return the longest
    path out of the sample.
    TODO: Make the random path selector more random (not relying on NX)
    """
    longest_path = []
    longest_path_length = 0
    approx_limit = 5000
    simple_paths = nx.all_simple_paths(G, source=source, target=target)
    for path in nx.all_simple_paths(G, source=source, target=target):
        if approx_limit == 0:
            break
        path_length = nx.path_weight(G, path, weight="weight")
        if path_length > longest_path_length:
            longest_path_length = path_length
            longest_path = path
        approx_limit -= 1
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
