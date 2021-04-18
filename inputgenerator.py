# Author: John Lee

# Choose "circle", "bipartite", "strongly_connected", or "random" for the type (must be a string)
# Random will generate a connected graph.
type = "random"

# Choose number of vertices (must be an int).
size = 30

# Choose edge weights (defaultEdgeWeight will assign same weights to every edge)
# Setting randomWeights to True will override defaultEdgeWeight
defaultEdgeWeight = 2

randomWeights = True
lowerBound = 1
upperBound = 10

#----------------------Algorithm Starts----------------------#

import random

def graphgenerator():
    if (type == "circle"): # O(|V|) time
        if (randomWeights == False):
            for v in range(size):
                line = str(v) + " " + str(v+1) + " " + str(defaultEdgeWeight)
                print(line)
        else:
            for v in range(size):
                edgeWeight = random.randint(lowerBound, upperBound)
                line = str(v) + " " + str(v+1) + " " + str(edgeWeight)
                print(line)
    elif (type == "strongly_connected"): # O(|V^2|) time
        if (randomWeights == False):
            for u in range(size):
                for v in range(u, size):
                    line = str(v) + " " + str(u) + " " + str(defaultEdgeWeight)
                    print(line)
        else:
            for u in range(size):
                for v in range(u, size):
                    edgeWeight = random.randint(lowerBound, upperBound)
                    line = str(v) + " " + str(u) + " " + str(edgeWeight)
                    print(line)
    elif (type == "random"): #
        vertices = []
        toPrint = []
        edges = {}
        for v in range(size):
            vertices += [v]
        currNode = 0
        while (len(vertices) > 0): #makes sure graph is connected
            edgeWeight = random.randint(lowerBound, upperBound)
            nextNodeIndex = random.randint(0, len(vertices) - 1)
            nextNode = vertices.pop(nextNodeIndex)
            toPrint += [str(currNode) + " " + str(nextNode) + " " + str(defaultEdgeWeight)]
            edges.update({currNode: nextNode})
            currNode = nextNode
        newEdgeCount = random.randint(0, size ** 2)
        for e in range(newEdgeCount): #adds random edges to (u, v)
            u = random.randint(0, size)
            v = random.randint(0, size)
            if (u != v and edges.get(u) != v):
                edgeWeight = random.randint(lowerBound, upperBound)
                edges.update({u: v})
                toPrint += [str(u) + " " + str(v) + " " + str(defaultEdgeWeight)]
        def firstNode(stringo):
            n = 0
            for i in len(stringo):
                if (stringo[i] == " "):
                    n = i
                    break
            return stringo[0:n]
        for i in range(len(toPrint)):
            print(toPrint[i])
    else:
        print("Make sure you are assigning type to the correct value.")

graphgenerator()