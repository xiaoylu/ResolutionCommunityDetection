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

# treat warnings as error so can use try... except
import warnings
warnings.filterwarnings("error")


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
  posterior MLE of the degree-corrected planted partition model (PPM)
  
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

  win = sum(k_r_in.values()) \
        / (sum([np.power(sumdeg, 2) for sumdeg in k_r.values()]) / 2./ E)

  wout = (2.0 * E - sum(k_r_in.values())) \
         / (2. * E - ( sum([np.power(_, 2) for _ in k_r.values()]) / 2./ E ))

  return win, wout

def _2ll(G, comms):
  '''
  Log-likelihood ratio test (LR Test) normalized by the number of edges.
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
  try:
    map_comm = {v:i for i, c in enumerate(comms) for v in c}
    win, wout = mle_paras(G, map_comm) # the MLE win and wout
    gamma = (win - wout) / (np.log(win) - np.log(wout)) # the MLE gamme
  except RuntimeWarning: 
    #print("RuntimeWarning", list(G.edges()), comms, win, wout)
    return 0.

  # modularity: modularity of the graph G under partition comm
  mod = modularity(G, comms, gamma)

  # constansts
  B = E * (np.log(win) - np.log(wout)) 
  C = E * (np.log(wout) - wout)

  return 2. * (B * mod + C + E) / E # normalized by the number of edges ???

def pvalue(G, comms, LLRtest, L = 3000, plothist = False):
  '''
  Compute the distribution of the test statistic in a range.
  Warning: Enumerating all Null networks is very slow.  
           Do not call this function in practical community detection. 
           Use threshold on the LR test statistic directly.

  Args:
      G: input networkx graph instance
      comms: the suggested partiton, list of lists
      LLRtest: the log-likelihood ratio test statistics 
      L: number of synthetic null networks (default 3000)

  Returns:
      (float): pvalue of LLRtest
  '''

  node_seq, deg_seq = zip(*list(G.degree()))
  index = {n:i for i, n in enumerate(node_seq)}
  
  # nodes should be index-0 
  comms = [list(map(index.get, c)) for c in comms]
  
  null_distri = []
  for niter in range(L):
    # debug
    if niter % 100 == 1: print("iter", niter, "H0 LLR", LRnull)

    # generate configuration network
    F = nx.Graph(nx.configuration_model(deg_seq))

    # obtain test statistic on null network
    LRnull = _2ll(F, comms)
    null_distri.append( LRnull )

  pval = sum([LLRnull > LLRtest for LLRnull in null_distri]) / float(len(null_distri))

  # plot
  if plothist:
    print("plotting")
    hist(null_distri, LLRtest, "%d_%d" % (len(comms), G.number_of_nodes()), pval) 

  return pval 

def multiscale_community_detection(G, depth = 1, resolution = 0.5, threshold = 1.2, min_com_size = 5, verbose = False, force_pvalue_sampling = False):
  '''
  Multi-scale community detection. Stop when hypothesis testing fails.
  Otherwise keep splitting a community into sub-communities.

  Args:
      G: input networkx graph instance
      depth: the current depth of the recursion
      resolution: the resolution parameter, desired value is smaller than 1
      threshold: terminate the recursion when the log-likelihood ratio (LLR) becomes smaller than threshold * E

  Returns:
      list: the final partition of the network
  '''
  verbose and print("\t" * depth, "D%d," % depth, "%d nodes" % int(G.number_of_nodes()))
   
  # community too small
  if G.number_of_nodes() <= min_com_size:
    return [list(G.nodes())]

  comms = greedy_modularity_communities(G, resolution = resolution)

  # found only one with current resolution
  if len(comms) == 1:
    verbose and print("\t" * depth, "==")
    return [list(G.nodes())]

  LR_test = _2ll(G, comms)

  verbose and print("\t" * depth, "LR=", LR_test)

  # if we enforce the test based on pvalue
  if force_pvalue_sampling:
    pval = pvalue(G, comms, LR_test, L = 3000, plothist = False)
    verbose and print("\t" * depth, "pvalue=", pval)
    if pval > 0.005:
      return [list(G.nodes())]

  # otherwise, we just compare LR_test to a threshold
  else:
    # Accept null hypothesis H0 that it's indeed one community
    if LR_test < threshold: # stop
      verbose and print("\t" * depth, "**")
      return [list(G.nodes())]

  # otherwise, we accept H1, and partition at the next level
  return itertools.chain.from_iterable( \
      multiscale_community_detection(G.subgraph(c), depth + 1, resolution, threshold, min_com_size, verbose, force_pvalue_sampling) \
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
  # Either Choice 1
  comms = list(multiscale_community_detection(G, resolution = 0.9, verbose = True, force_pvalue_sampling = True))
  # Or Choice 2
  #comms = greedy_modularity_communities(G)

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


def synthetic(n = 20, m = 20):
  # n communities of size m
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
  gnc = [list(range(i * m, i * m + m)) for i in range(n)]

  import random
  X = list(range(n * m))
  random.shuffle(X)
  tmp = {i:x for i, x in enumerate(X)}
  indices = [list(range(i * m, i * m + m)) for i in range(n)]
  # a completly random partition
  rand_comms = [list(map(tmp.get, row)) for row in indices]
  print("\n".join(map(str, rand_comms)))

  LR_test = _2ll(G, rand_comms)
  print("random communities", LR_test)
  
  LR_test = _2ll(G, gnc)
  print("ground truth", LR_test)
  
  pvalue(G, gnc, LR_test, L = 3000, plothist = True)
  return
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
  football()

  #synthetic(20, 10)
  #synthetic(20, 20)
  #synthetic(20, 30)
  #synthetic(20, 40)

  #synthetic(10, 20)
  #synthetic(20, 20)
  #synthetic(30, 20)
  #synthetic(40, 20)

  #synthetic(20, 20)
