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

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


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

def draw(G, colormap):
  plt.clf()
  pos=nx.spring_layout(G)
  nx.draw_networkx_nodes(G,pos,node_size=[30*G.degree(k) for k,v in pos.items()],node_shape='o',node_color=list(map(colormap.get, G.nodes())))
  labels=nx.draw_networkx_labels(G,\
      pos={k:v+np.array([0.05,0.05]) for k,v in pos.items()},\
      labels={k:"%d"%k for k,v in pos.items()}, font_size=14)
  nx.draw_networkx_edges(G,pos,width=1,edge_color='black')
  plt.axis('off')
  plt.savefig("network.png", bbox_inches="tight")
  print("Save figure to", "network.png") 

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

def _2ll(G, comm, modularity, win, wout):
  '''
  Log-likelihood ratio test (LR Test).
  H0 is the configuration model.
  H1 is the degree-corrected planted partition model.

  Args:
      G: input networkx graph instance
      comm: a dict where comm[i] is the community label of node i   
      modularity: modularity of the graph G under partition comm
      win, wout: the mixing parameters of PPM

  Returns:
      the log-likelihood ratio test statistic 
  '''

  # community sizes
  ni = Counter(comm.values())
  E = G.number_of_edges()

  B = E * (np.log(win) - np.log(wout)) 
  C = E * (np.log(wout) - wout)

  return 2. * (B * modularity + C + E)

def multiscale_community_detection(G, depth = 1):
  '''
  Multi-scale community detection. Stop when hypothesis testing fails.
  Otherwise keep splitting a community into sub-communities.

  Args:
      G: input networkx graph instance

  Returns:
      list: the final partition of the network
  '''
  print("\t" * depth, G.nodes())

  comms = greedy_modularity_communities(G, gamma = 0.8)

  if len(comms) == 1:
    print("\t" * depth, "*")
    return [list(G.nodes())]

  map_comm = {v:i for i, c in enumerate(comms) for v in c}
  win, wout = mle_paras(G, map_comm)
  gamma = (win - wout) / (np.log(win) - np.log(wout))

  mod = modularity(G, comms, gamma)

  LR_test = _2ll(G, map_comm, mod, win, wout)
  print("\t" * depth, LR_test)

  if LR_test < 3.0: # stop
    print("\t" * depth, "*")
    return [list(G.nodes())]

  return itertools.chain.from_iterable( \
      multiscale_community_detection(G.subgraph(c), depth + 1) \
      for c in comms)  

def tests(network = 'football'):
  if network == 'football':
    path = "../data/football/football.txt"
    gnc_path = "../data/football/footballTSEinputConference.clu"
    output_path = "../data/football/"
    name = 'American College Football Network'

  elif network == 'lesmis':
    name = 'Les Miserable'
    path = "../data/lesmis/lesmis.txt"

  # load network and ground-truth communities
  G = loadG(path)
  G.graph['name'] = name 
  #gnc = loadGNC(gnc_path)
  print(nx.info(G))

  comms = list(multiscale_community_detection(G))
  map_comm = {v:i for i, c in enumerate(comms) for v in c}

  # check NMI
  #a = [map_comm[k] for k in G.nodes()]
  #b = [gnc[k] for k in G.nodes()]
  #print("NMI=", metrics.adjusted_mutual_info_score(a, b))

  print("#Comm=", len(comms))
  print(comms)
  draw(G, map_comm)


#def synthetic():
#  for n in range(10, 80, 10):
#      sizes = [50] * n 
#      probs = np.ones((n, n)) * 0.05
#      for i in range(n):
#          probs[i][i] = 0.3
#      G = nx.stochastic_block_model(sizes, probs, seed=0)
#      
#      start_time = time.time()
#      for c in greedy_modularity_communities(G, gamma = 0.2):
#          #print(c)
#          pass
#      print(G.number_of_nodes(), '\t', G.number_of_edges(), '\t', time.time() - start_time)
 

#===============================================================================
if __name__ == "__main__":
  tests(network = 'football')
  #tests(network = 'lesmis')
