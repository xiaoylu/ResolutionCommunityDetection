# -*- coding: utf-8 -*-
"""

This module implements the multi-scale community detection with generalized 
modularity maximization and log-likelihood ratio tests. 

Example:
    Execute the code to test on American Football network::

        $ python example_google.py

Research article is available at:
   http://google.github.io/styleguide/pyguide.html

"""

import numpy as np 
from collections import defaultdict, Counter
import itertools
import time

import networkx as nx 
#from networkx.algorithms.community import modularity, greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community.modularity_max import * 

from sklearn import metrics
from heatmap import draw, heatmap, hist


def loadG(path):
  G = nx.Graph()
  with open(path, "rb") as txt:
    for line in txt: 
      if len(line) > 1 and line[0]!='#':
        e = line.split()
        G.add_edge(int(e[0]), int(e[1]))
  G = nx.convert_node_labels_to_integers(G)
  return G 

def loadGNC(path):
  gnc = {}
  with open(path, "r") as fconf:
    for i, line in enumerate(fconf):
      gnc[i] = int(line.strip())
  return gnc

def mle_paras(G, comm): 
  ''' 
  Degree-corrected planted partition model (PPM) posterior estimation 
  
  Args:
      G: input networkx graph instance
      comm: a dict where comm[i] is the community label of node i   
  
  Returns:
      win, wout: the mixing parameter of PPM
  ''' 

  N, E = G.number_of_nodes(), G.number_of_edges()

  k_r, k_r_in = defaultdict(int), defaultdict(int)
  for i in comm: 
    k_r[comm[i]] += G.degree(i)
  for u, v in G.edges():
    if comm[u] == comm[v]:
      k_r_in[comm[u]] += 2 

  win =    sum(k_r_in.values()) \
        / (sum([np.power(sumdeg, 2) for sumdeg in k_r.values()]) / 2./ E)

  wout = (2.0 * E - sum(k_r_in.values())) \
         / (2. * E - ( sum([np.power(_, 2) for _ in k_r.values()]) / 2./ E ))

  return win, wout

def _2ll(G, comms):
  '''
  Log-likelihood ratio test (LR Test).
  H0 is the configuration model.
  H1 is the degree-corrected planted partition model.

  Args:
      G: input networkx graph instance
      comms: partition of network, list of lists 

  Returns:
      the log-likelihood ratio test statistic 
  '''

  # community sizes
  E = G.number_of_edges()

  # win, wout: the MLE mixing parameters of PPM
  map_comm = {v:i for i, c in enumerate(comms) for v in c}
  win, wout = mle_paras(G, map_comm) # the MLE win and wout
  gamma = (win - wout) / (np.log(win) - np.log(wout)) # the MLE gamme

  # modularity: modularity of the graph G under partition comm
  mod = modularity(G, comms, gamma)

  # constansts
  B = E * (np.log(win) - np.log(wout)) 
  C = E * (np.log(wout) - wout)

  return 2. * (B * mod + C + E)

def pvalue(G, comms, LLRtest, L = 3000):
  '''
  Compute the distribution of the test statistic in a range 

  Args:
      G: input networkx graph instance
      comms: the suggested partiton, list of lists
      LLRtest: the log-likelihood ratio test statistics 
      L: number of synthetic null networks (default 3000)

  Returns:
      (float): pvalue of LLRtest
  '''

  if LLRtest > 50: # skip the obvious cases
    return 0
  # WARN: fast check (skip p-value part)
  else:
    return 1

  node_seq, deg_seq = zip(*list(G.degree()))
  index = {n:i for i, n in enumerate(node_seq)}
  
  # nodes should be index-0 
  comms = [list(map(index.get, c)) for c in comms]
  
  null_distri = []
  for niter in range(L):
    # debug
    if niter % 100 == 1: print("iter", niter)

    # generate configuration network
    F = nx.Graph(nx.configuration_model(deg_seq))

    # obtain test statistic on null network
    LRnull = _2ll(F, comms)
    null_distri.append( LRnull )

  pval = sum([LLRnull > LLRtest for LLRnull in null_distri]) / float(len(null_distri))

  # plot
  print("plotting")
  hist(null_distri, LLRtest, "%d_%d" % (len(comms), G.number_of_nodes()), pval) 

  return pval 

def multiscale_community_detection(G, depth = 1, gamma = 0.8):
  '''
  Multi-scale community detection. Stop when hypothesis testing fails.
  Otherwise keep splitting a community into sub-communities.

  Args:
      G: input networkx graph instance

  Returns:
      list: the final partition of the network
  '''
  print("\t" * depth, int(G.number_of_nodes()), G.nodes())

  comms = greedy_modularity_communities(G, gamma)

  if len(comms) == 1:
    print("\t" * depth, "*")
    return [list(G.nodes())]

  LR_test = _2ll(G, comms)

  pval = pvalue(G, comms, LR_test)

  print("\t" * depth, "LR=", LR_test, "Pval=", pval)

  if LR_test < 5.0: # stop
    print("\t" * depth, "*")
    return [list(G.nodes())]

  return itertools.chain.from_iterable( \
      multiscale_community_detection(G.subgraph(c), depth + 1) \
      for c in comms)  

def football():
  path = "../data/football/football.txt"
  gnc_path = "../data/football/footballTSEinputConference.clu"
  name = 'American College Football Network'

  # load network and ground-truth communities
  G = loadG(path)
  G.graph['name'] = name 
  gnc = loadGNC(gnc_path)
  print(nx.info(G))

  # community detection
  comms = list(multiscale_community_detection(G))

  # check NMI
  map_comm = {v:i for i, c in enumerate(comms) for v in c}
  a = [map_comm[k] for k in G.nodes()]
  b = [gnc[k] for k in G.nodes()]
  print("NMI=", metrics.adjusted_mutual_info_score(a, b))

  print("#Comm=", len(comms))
  print(comms)

  # draw topology
  #draw(G, map_comm)

  # draw heatmap
  heatmap(G, comms)


#def lesmis():
#  name = 'Les Miserable'
#  path = "../data/lesmis/lesmis.txt"

def synthetic():
  # n communities of size m
  n, m = 20, 20
  sizes = [m] * n

  # mixing matrix
  probs = np.ones((n, n)) * 0.05
  for i in range(n):
      probs[i][i] = 0.8 
  #probs[:20][:20] += 0.1

  G = nx.stochastic_block_model(sizes, probs, seed=0)
  G.graph['name'] = 'synthetic'
  print(nx.info(G))

  ############## what if the graph is large ##################
  # the experiment shows that Wilk's theorem is true?!
  # in that way, n->+inf, the distribution is indeed chi-squared
  #comms = [list(range(i * m, i * m + m)) for i in range(n)]
  import random
  X = list(range(n * m))
  random.shuffle(X)
  tmp = {i:x for i, x in enumerate(X)}
  indices = [list(range(i * m, i * m + m)) for i in range(n)]
  comms = [list(map(tmp.get, row)) for row in indices]

  LR_test = _2ll(G, comms)
  print(LR_test)
  exit(1)
  pvalue(G, comms, LR_test, L = 3000)
  exit(1)
  ############# end of this experiment #######################

  # community detection
  comms = list(multiscale_community_detection(G, gamma = 0.3))
  map_comm = {v:i for i, c in enumerate(comms) for v in c}

  # check NMI
  a = [map_comm[k] for k in G.nodes()]
  b = [k//m for k in G.nodes()]
  print("NMI=", metrics.adjusted_mutual_info_score(a, b))

  print("#Comm=", len(comms))
  print(comms)

  comms = greedy_modularity_communities(G)
  print("#Comm=", len(comms))
  print(comms)


#===============================================================================
if __name__ == "__main__":
  #football()

  synthetic()
