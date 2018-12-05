# -*- coding: utf-8 -*-
"""

This module test the results on Amazon co-purchasing networks

Example:
    Execute the code to test on Amazon co-purchasing network::

        $ python amazon.py

Research article is available at:
   http://google.github.io/styleguide/pyguide.html

"""

import networkx as nx
import os
import time
from networkx.algorithms.community.modularity_max import * 
from networkx.algorithms.community import LFR_benchmark_graph
from generalized_modularity import multiscale_community_detection
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from heatmap import hist

def loadAmazon():
    # cache graph in .gpickle format for fast reloading
    path =  "../data/amazon/amazon.gpickle"
    if os.path.isfile(path): 
        G = nx.read_gpickle(path)
        return G
    else:
        G = nx.Graph(gnc = {}, membership = {}, top5000 = {}, top5000_membership = {})
        with open("../data/amazon/com-amazon.ungraph.txt", "r") as txt:
            for line in txt:
                if not line[0] == '#':
                    e = line.split()
                    G.add_edge(int(e[0]), int(e[1]))
        with open("../data/amazon/com-amazon.top5000.cmty.txt", "r") as txt:
            count = 0
            for line in txt:
                if not line[0] == '#':
                    e = line.split()
                    G.graph["top5000"][count] = [int(_) for _ in e]
                    for n in G.graph["top5000"][count]:
                        if n in G.graph["top5000_membership"]:
                            G.graph["top5000_membership"][n].append( count )
                        else:
                            G.graph["top5000_membership"][n] = [ count ]
                    count += 1
        with open("../data/amazon/com-amazon.all.dedup.cmty.txt", "r") as txt:
            count = 0
            for line in txt:
                if not line[0] == '#':
                    e = line.split()
                    G.graph["gnc"][count] = [int(_) for _ in e]
                    for n in G.graph["gnc"][count]:
                        if n in G.graph["membership"]:
                            G.graph["membership"][n].append( count )
                        else:
                            G.graph["membership"][n] = [ count ]
                    count += 1
        print("write gpickle file..")
        nx.write_gpickle(G, path)
        return G

def LFR(n, tau1, tau2, mu):
    path = "../data/LFR/LFR_%d_%.2f_%.2f_%.2f.gpickle" %(n, tau1, tau2, mu)
    if os.path.isfile(path): 
        G = nx.read_gpickle(path)
        return G
    else:
        G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree = 8, min_community = 5, seed=10)
        print("write gpickle file", path)
        nx.write_gpickle(G, path)
        return G

if __name__ == "__main__":

    verbose = False

    #G = loadAmazon()
    G = LFR(n = 5800, tau1 = 3, tau2 = 1.5, mu = 0.15)

    print(nx.info(G))

    print("get ground truth")
    gnc = {frozenset(G.nodes[v]['community']) for v in G}
    sizes = [len(i) for i in gnc]
    gnc_sizes = sorted(sizes)
    verbose and print(gnc_sizes)
    
    #============

    print("start naive community detection")
    start = time.time()

    comms0 = greedy_modularity_communities(G, resolution = 1.0)
    comms0_sizes = sorted([len(comms0[i]) for i in range(len(comms0))])
    verbose and print(comms0_sizes)

    end = time.time()
    print("naive modularity", end - start, "seconds")

    #============

    # check NMI
    map_comm = {v:i for i, c in enumerate(comms0) for v in c}
    a = [map_comm[k] for k in G.nodes()]
    map_comm2 = {v:i for i, c in enumerate(gnc) for v in c}
    b = [map_comm2[k] for k in G.nodes()]
    print("Modularity NMI=", metrics.adjusted_mutual_info_score(a, b))

    # multi-scale community detection
    print("start multi-scale community detection")
    start = time.time()

    comms1 = list(multiscale_community_detection(G, resolution = 0.4, threshold = 2.5, verbose = True))
    comms1_sizes = sorted([len(comms1[i]) for i in range(len(comms1))])
    verbose and print(comms1_sizes)

    end = time.time()
    print("multi-scale", end - start, "seconds")

    #============

    map_comm3 = {v:i for i, c in enumerate(comms1) for v in c}
    c = [map_comm3[k] for k in G.nodes()]
    print("Multiscale NMI=", metrics.adjusted_mutual_info_score(a, c))
    
    print("Plot histogram of community sizes")
    hist({"Ground Truth": gnc_sizes, "Modularity": comms0_sizes, "Multiscale": comms1_sizes})
