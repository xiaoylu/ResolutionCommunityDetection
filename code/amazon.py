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
from networkx.algorithms.community.modularity_max import * 
from networkx.algorithms.community import LFR_benchmark_graph
from sklearn import metrics


def loadAmazon():
    # cache graph in .gpickle format for fast reloading
    if os.path.isfile("./data/amazon/amazon.gpickle"): 
        G = nx.read_gpickle("./data/amazon/amazon.gpickle")
        return G
    else:
        G = nx.Graph(gnc = {}, membership = {}, top5000 = {}, top5000_membership = {})
        with open("data/amazon/com-amazon.ungraph.txt", "r") as txt:
            for line in txt:
                if not line[0] == '#':
                    e = line.split()
                    G.add_edge(int(e[0]), int(e[1]))
        with open("data/amazon/com-amazon.top5000.cmty.txt", "r") as txt:
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
        with open("data/amazon/com-amazon.all.dedup.cmty.txt", "r") as txt:
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
        nx.write_gpickle(G, "data/amazon/amazon.gpickle")
        return G

if __name__ == "__main__":
    #G = loadAmazon()
    
    n = 2250
    tau1 = 3
    tau2 = 1.5
    mu = 0.2
    G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=10, seed=10)
    print(nx.info(G))
    
    gnc = {frozenset(G.nodes[v]['community']) for v in G}
    sizes = [len(i) for i in gnc]
    print(sorted(sizes))
    import collections
    print(collections.Counter(sizes).most_common())
    
    print("start community detection")
    comms0 = greedy_modularity_communities(G, gamma = 1.5)
    sizes = [len(comms0[i]) for i in range(len(comms0))]
    print(collections.Counter(sizes).most_common())

    # community detection
    from generalized_modularity import multiscale_community_detection
    comms1 = list(multiscale_community_detection(G))
    sizes = [len(comms1[i]) for i in range(len(comms1))]
    print(collections.Counter(sizes).most_common())
    
    # check NMI
    map_comm = {v:i for i, c in enumerate(comms0) for v in c}
    a = [map_comm[k] for k in G.nodes()]
    map_comm2 = {v:i for i, c in enumerate(gnc) for v in c}
    b = [map_comm2[k] for k in G.nodes()]
    map_comm3 = {v:i for i, c in enumerate(comms1) for v in c}
    c = [map_comm3[k] for k in G.nodes()]
    
    print("NMI=", metrics.adjusted_mutual_info_score(a, b))
    print("NMI=", metrics.adjusted_mutual_info_score(a, c))