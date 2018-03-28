import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn import metrics
from synthetic import draw, dcSBM, c2r1
from util import NMI_dict, resolution_limit, modularity
from block_standard import Hierar

def two_regions():
  N = 40
  n_comm = 4 
  comm_size = [10] * 4 
  k = [3 + int(nx.utils.powerlaw_sequence(1, 2.5)[0]) for _ in range(N)]

  block = {}
  base = 0
  for t in range(len(comm_size)):
    block.update( {base+i:t for i in range(comm_size[t])} )
    base += comm_size[t]
  w = np.ones((n_comm,n_comm))
  #a1,a0 = 20,4
  a1,a0 = 0.,0.
  b1,b0 = 10.,0.1
  w0 = 0.0
  w =np.array( [ [a1,a0,w0,w0],
                 [a0,a1,w0,w0],
                 [w0,w0,b1,b0],
                 [w0,w0,b0,b1] ] )
  G = dcSBM(block, w, k)
  Omega  = w

  h = Hierar(G, 2)
  vals = np.linspace(0.3,20,num=20)
  nmi_list, ncomm_list = h.try_gamma(vals, block, flag=True)
  print vals
  print "NMI scores=", nmi_list
  print "#comm=", ncomm_list 

def test_c2r1(): 
  for nS in range(200, 300, 10):  
    G,block = c2r1(nC=13, nS=nS, kS=100)
    n_comm = len(set(block.values()))

    kp = defaultdict(float)
    k = np.zeros(G.number_of_nodes())
    for v, deg in G.degree():
      kp[block[v]] += deg
      k[v] = deg
    m = np.zeros((n_comm,n_comm))
    for e0, e1 in G.edges():
      m[block[e0]][block[e1]] += 2 
      m[block[e1]][block[e0]] += 2 
    E = float(G.number_of_edges())

    print kp
    print "edges:\n", m

    # prior values for generation
    w = np.zeros((n_comm,n_comm))
    for r in range(n_comm):
      for s in range(n_comm):
        w[r][s] = (m[r][s] / kp[r]) * (2. * E / kp[s]) # percent of in-degree
    print "omega:"
    np.set_printoptions(precision=1)
    print(w)
    #G = dcSBM(block, w, k)

    #draw(G,block)
    #a = nx.adjacency_matrix(G).todense()
    #plt.imshow(a, cmap='hot', interpolation='nearest')
    #plt.savefig("adjacency_matrix.png")
    #print "Draw adjacency_matrix.png"

    #h = Hierar(G, least_num_of_comm = 2, gamma = 1.0)
    #comm = h.hierar(stop_at_max_modularity=True)
    #print "NMI=", NMI_dict(comm, block)

    #vals = np.linspace(0.1,10,num=5)
    #nmi_list, ncomm_list = h.try_gamma(vals, block, flag=True)
    #print vals
    #print "NMI scores=", nmi_list
    #print "#comm=", ncomm_list 

    for gamma in np.linspace(0, 10, num=10):
      gnc = block
      gnc_mergeC = {k: int(v>0)  for k, v in gnc.items()}
      gnc_splitS = {k: v if v > 0 else 3+np.random.randint(2) for k, v in gnc.items()}
      sp, sd, mg = modularity(G,gnc_splitS,gamma=gamma), modularity(G,gnc,gamma=gamma), modularity(G,gnc_mergeC,gamma=gamma)

    #print w[0][0],w[1][2], sd>sp, sd>mg
    #print w[0][0],w[1][2]

#two_regions()
test_c2r1()
