import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import random
random.seed(11)
import numpy.random
np.random.seed(10)

def c2r1(nC=6, nS=30, kS=6):
  G = nx.fast_gnp_random_graph(nS, float(kS)/nS)
  H = nx.caveman_graph(1, nC)
  mapping1 = {_:_ + nS for _ in H.nodes()}
  mapping2 = {_:_ + nS +nC for _ in H.nodes()}
  S1 = nx.relabel_nodes(H,mapping1)
  S2 = nx.relabel_nodes(H,mapping2)
  G = nx.compose(G,S1)
  G = nx.compose(G,S2)
  G.add_edge(0,nS)
  G.add_edge(1,nS+nC)
  G.add_edge(nS+1,nS+nC+1)
  print nx.info(G)
  gnc = {_:0 for _ in range(nS)}
  gnc.update({_+nS:1 for _ in range(nC)})
  gnc.update({_+nS+nC:2 for _ in range(nC)})
  return G,gnc

def draw(G, colormap):
  plt.clf()
  pos=nx.spring_layout(G)
  nx.draw_networkx_nodes(G,pos,node_size=[30*G.degree(k) for k,v in pos.items()],node_shape='o',node_color=map(colormap.get, G.nodes()))
  labels=nx.draw_networkx_labels(G,\
      pos={k:v+np.array([0.05,0.05]) for k,v in pos.items()},\
      labels={k:"%d"%k for k,v in pos.items()}, font_size=14)
  nx.draw_networkx_edges(G,pos,width=1,edge_color='black')
  plt.axis('off')
  plt.savefig("network.png", bbox_inches="tight")
  print "Save figure to", "network.png" 

# Generate a degree-corrected SBM given the parameters
# @n_comm : number of communities
# @comm_size : a list of community sizes
# @k : approx degree of each node (the theta parameter of degree-corrected SBM) 
# Note that the expected #edge between nodes i and j is: w*k_i*k_j / (2*m)
#      rather the original definition
def dcSBM(block, w, k): 
  #n_comm = 3 
  #comm_size = [10,10,80]
  #k = [5 + int(nx.utils.powerlaw_sequence(1, 2.5)[0]) for _ in range(N)]
  #block = {i:int(i/(N/n_comm)) for i in range(N)}

  N = len(block) 
  n_comm = len(set(block.values()))
  m = sum(k) / 2.
  print N, n_comm, m

  # prior values for generation
  print "Prior", w

  G = nx.Graph()
  for i in range(N):
    G.add_node(i)
  for i in range(N):
    for j in range(N):
      if i != j:
        lam = w[block[i]][block[j]] * k[i] * k[j] / 2. / m
        if (np.random.poisson(lam) > 0):
          G.add_edge(i,j)

  return G

if __name__=="__main__":
  G,gnc = c2r1()
  draw(G,gnc)
