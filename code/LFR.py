0# -*- coding: utf-8 -*-
"""

This module test the results on LFR networks

Example:
    Execute the code to test on LFR network::

        $ python LFR.py

Research article is available at:
   http://...

"""

import networkx as nx
import os
import time
import pickle
from networkx.algorithms.community.modularity_max import * 
from networkx.algorithms.community import LFR_benchmark_graph
from generalized_modularity import multiscale_community_detection
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from heatmap import hist


def LFR(n, tau1, tau2, mu, min_com_size, force = False):
    # enforce regeneration if force==True  
    path = "../data/LFR/LFR_%d_%.2f_%.2f_%.2f.gpickle" %(n, tau1, tau2, mu)
    if not force and os.path.isfile(path): 
        G = nx.read_gpickle(path)
        return G
    else:
        G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree = 8, min_community = min_com_size, seed=0)
        print("write gpickle file", path)
        nx.write_gpickle(G, path)
        return G

if __name__ == "__main__":

    verbose = True 

    #=========== Global Parameters ===========#
    _network_size = 5000
    _resolution = 0.9
    _threshold = 1.25
    _min_com_size = 20
    _benchmark = True
    _verbose = False

    #=========== Generate Graph ==============#

    G = LFR(n = _network_size, tau1 = 3.0, tau2 = 2.0, mu = 0.25, min_com_size = _min_com_size, force = True)

    #G = LFR(n = 100, tau1 = 2.5, tau2 = 1.2, mu = 0.15)
    print(nx.info(G))

    print("get ground truth")
    gnc = {frozenset(G.nodes[v]['community']) for v in G}
    map_comm2 = {v:i for i, c in enumerate(gnc) for v in c}
    b = [map_comm2[k] for k in G.nodes()]

    sizes = [len(i) for i in gnc]
    gnc_sizes = sorted(sizes)
    verbose and print("ground truth community sizes=", gnc_sizes)

    #=========== Benchmark ===============#
    print("start naive community detection")
    start = time.time()
    comms0 = greedy_modularity_communities(G)
    end = time.time()

    comms0_sizes = sorted([len(comms0[i]) for i in range(len(comms0))])
    verbose and print(comms0_sizes)

    map_comm = {v:i for i, c in enumerate(comms0) for v in c}
    a = [map_comm[k] for k in G.nodes()]
    print("FastGreedy Algorithm ARI=", adjusted_rand_score(a, b), "NMI=", normalized_mutual_info_score(a, b))

    print("which takes", end - start, "seconds")

    #=========== Multi-scale Community Detection ===============#

    print("Start Multi-scale Community Detection")
    start = time.time()
    comms1 = list(multiscale_community_detection(G, resolution = _resolution, threshold = _threshold, min_com_size = _min_com_size, verbose = _verbose))
    end = time.time()

    comms1_sizes = sorted([len(comms1[i]) for i in range(len(comms1))])
    verbose and print(comms1_sizes)

    map_comm3 = {v:i for i, c in enumerate(comms1) for v in c}
    c = [map_comm3[k] for k in G.nodes()]
    print("Multi-scale Algorithm ARI=", adjusted_rand_score(b, c), "NMI=", normalized_mutual_info_score(b, c))
    print("which takes", end - start, "seconds")

    #============ Plot community sizes ==============#

    print("Plot histogram of community sizes")
    sizes_distri = {"Ground Truth": gnc_sizes, "Modularity": comms0_sizes, "Multiscale": comms1_sizes}

    pickle.dump(sizes_distri, open('save%d.p' % _network_size, 'wb'))
    hist(sizes_distri, _network_size)
