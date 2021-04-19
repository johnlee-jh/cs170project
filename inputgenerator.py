# Author: John Lee, Sunay Dagli, Wilson Nguyen
import sys
import parse
# Choose "circle", "bipartite", "strongly_connected", or "random" for the type (must be a string)
# Random will generate a connected graph.
graph_type = 'yeetmoderandom'

# Choose number of vertices (must be an int).
size = 29

# Choose edge weights (defaultEdgeWeight will assign same weights to every edge)
# Setting randomWeights to True will override defaultEdgeWeight
defaultEdgeWeight = 2

lowerBound = 1
upperBound = 10

#----------------------Algorithm Starts----------------------#

import random

def rand(start=lowerBound, end=upperBound):
    return random.uniform(lowerBound, upperBound)

def get_vertices(lines):
    vertices = set()
    for (u, v, w) in lines:
        vertices.add(u)
        vertices.add(v)
    return len(vertices)

def solidify(lines, offset=0):
    for i in range(len(lines)):
        (u, v, w) = lines[i]
        lines[i] = (u + offset, v + offset, '%.3f' % w if not isinstance(w, str) else w)
    return lines

def merge(lines1, lines2):
    lines1 = solidify(lines1)
    lines2 = solidify(lines2, get_vertices(lines1))
    return solidify(lines1 + lines2 + [(lines1[-1][0], lines2[0][0], defaultEdgeWeight)])

def visualize(lines):
    # Prints out the lines to the console
    print(get_vertices(lines))
    for line in lines:
        print(*line)

def validate(path):
    parse.validate_file(path)
    parse.read_input_file(path)

def save(lines):
    # Writes the lines to a file
    path = './samples/' + str(get_vertices(lines)) + '-' + str(len(lines)) + '-custom.in'
    writer = open(path,"w")
    writer.write(str(get_vertices(lines)) + '\n')
    last_line = lines.pop()
    for line in lines:
        writer.write(' '.join(map(str, line)) + '\n')
    writer.write(' '.join(map(str, last_line)))
    writer.close()
    validate(path)

def graphgenerator(graph_type, defaultEdgeWeight, randomWeights, size, lowerBound, upperBound):

    lines = []

    def line(u, v, w):    # Must be a tuple (u, v, w) ---- w is weight
        lines.append((u, v, w))

    # O(|V|) time
    if graph_type == 'circle':
        for v in range(size):
            edgeWeight = rand() if randomWeights else defaultEdgeWeight
            line(v, v+1, edgeWeight)
        edgeWeight = rand() if randomWeights else defaultEdgeWeight
        line(0, size, edgeWeight)
    # O(|V|^2) time
    elif graph_type == 'randomXD':
        d_vertices = {}
        for v in range(size):
            d_vertices[v] = 0
        not_enough = True
        used = []
        while(not_enough):
            v1 = rand(0, size)
            while (v1 not in used):
                v1 = rand(0, size)
            v2 = rand(0, size)
            while (v2 not in used):
                v2 = rand(0, size)
            line(v1, v2, rand(0, 10000))
            d_vertices[v1] += 1
            d_vertices[v2] += 1
            for key in d_vertices:
                if d_vertices[key] < 2:
                    not_enough = True
                else:
                    used.append(v1)
                if (not not_enough):
                    break        
    elif graph_type == 'strongly_connected':
        for u in range(size):
            for v in range(u, size):
                edgeWeight = rand() if randomWeights else defaultEdgeWeight
                line(v, u, edgeWeight)
    elif graph_type == "random": #
        vertices = [v for v in range(size)]
        toPrint = []
        edges = {}
        currNode = 0
        while len(vertices) > 0: #makes sure graph is connected
            edgeWeight = rand()
            nextNodeIndex = rand(0, size - 1)
            nextNode = vertices.pop(nextNodeIndex)

            toPrint.append((currNode, nextNode, defaultEdgeWeight))
            
            edges.update({currNode: nextNode})
            currNode = nextNode
        newEdgeCount = rand(0, size ** 2)
        for e in range(newEdgeCount): #adds random edges to (u, v)
            u = rand(0, size)
            v = rand(0, size)
            if u != v and edges.get(u) != v:
                edgeWeight = random.randint(lowerBound, upperBound)
                edges.update({u: v})
                toPrint.append((currNode, nextNode, defaultEdgeWeight))
        
        # NOT BEING USED
        def firstNode(stringo):
            n = 0
            for i in len(stringo):
                if (stringo[i] == " "):
                    n = i
                    break
            return stringo[0:n]

        for i in toPrint:
            line(*i)    #*i means to spread i
    elif graph_type == 'yeetmodeuniform':
        vertices = size
        for v in range(vertices):
            for u in range(vertices):
                w = rand() if randomWeights else defaultEdgeWeight
                line(v, u, w)
    elif graph_type == 'yeetmoderandom':
        vertices = size
        temp_lines = [];
        for v in range(vertices):
            vertex_lines = []
            for u in range(vertices):
                w = rand() if randomWeights else defaultEdgeWeight
                vertex_lines.append((v, u, w))
            temp_lines.append(vertex_lines)
        
        # Randomly deletes
        for v in range(vertices):
            vertex_lines = temp_lines[v]
            random.shuffle(vertex_lines)
            temp_lines[v] = vertex_lines[0:random.randint(2, vertices)]
        
        for vertex in temp_lines:
            for l in vertex:
                line(*l)
    else:
        raise RuntimeError('Make sure you are assigning type to the correct value.')
    return lines


lines1 = graphgenerator('yeetmodeuniform', defaultEdgeWeight, False, size, lowerBound, upperBound)
lines2 = graphgenerator('yeetmoderandom', defaultEdgeWeight, True, size, lowerBound, upperBound)
lines = merge(lines1, lines2)       # Using merge causes graph to inflate by factor of 2
solidify(lines)
visualize(lines)
save(lines)