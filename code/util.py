import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import random
random.seed(10)
from synthetic import draw, dcSBM, c2r1  
from sklearn import metrics
from collections import defaultdict
from block_standard import Hierar

#from networkx.algorithms.community.quality import modularity

# util func to compute the sum of (internal or total) degree of nodes in blocks
def cal_kappa(G, gnc): 
  kp_in, kp = defaultdict(float), defaultdict(float)
  m = G.number_of_edges()
  for v, deg in G.degree(): 
    kp[gnc[v]] += deg 
  for e0,e1 in G.edges():
    if gnc[e0] == gnc[e1]:
      kp_in[gnc[e0]] += 2 
  return kp_in, kp
 
# generalized modularity
def modularity(G,gnc,gamma=1.0): 
  kp_in, kp = cal_kappa(G, gnc)
  m = float(G.number_of_edges())
  ret = 0.
  for b in kp.keys():
    ret += (kp_in[b]) / (2. * m) - gamma * np.power((kp[b]) / (2. * m), 2)
  return ret

# modularity with corrected community density
def g_modularity(G,gnc,gamma,beta,w_out): 
  kp_in, kp = cal_kappa(G, gnc)
  m = float(G.number_of_edges())
  ret = 0.
  for b in kp.keys():
    if b in beta:
      ret += beta[b] * ( (kp_in[b]) / (2. * m) - gamma[b] * np.power((kp[b]) / (2. * m), 2) )
    else: #kp_in[b] == 0
      ret += w_out * np.power((kp[b]) / (2. * m), 2) 
  return ret

# maximum likelihood estimates of the Omegas (for planted parition model and its extension)
def mle(G,gnc,mode="modularity"): 
  kp_in, kp = cal_kappa(G, gnc)
  m = float(G.number_of_edges())
  sum_kp_in = sum(kp_in.values())
  if (sum_kp_in == 0): print("No communities given (but a set of individual nodes); Quit.");exit(1)
  sum_kp_sqr = sum([np.power(_, 2) for _ in kp.values()])
  if mode == "modularity":
    w_in = sum_kp_in / (sum_kp_sqr / (2.*m))
    w_out = (2.*m - sum_kp_in) / ( 2.*m - (sum_kp_sqr / (2.*m)) )
  elif mode == "density":
    w_in = {} 
    for b in kp_in.keys():
      w_in[b] = kp_in[b] / (np.power(kp[b], 2) / (2.*m))
    w_out = (2.*m - sum_kp_in) / ( 2.*m - (sum_kp_sqr / (2.*m)) )
  return w_in, w_out 

# the exact log-likelihood for unweighted graphs
def ll(G, gnc, mode="modularity"):
  w_in, w_out = mle(G, gnc, mode=mode)
  m = float(G.number_of_edges())
  C = m * (np.log(w_out) - w_out - np.log(2.*m)) + sum([ki * np.log(ki) for v,ki in G.degree()])  
  if mode == "modularity":
    B = (np.log(w_in) - np.log(w_out))
    gamma = (w_in - w_out) / (np.log(w_in) - np.log(w_out)) 
    mo = modularity(G, gnc, gamma)
    return mo, B * mo + C 
  elif mode == "density":
    gamma = {b:((w_in[b] - w_out) / (np.log(w_in[b]) - np.log(w_out))) for b in w_in.keys()}
    beta = {b:(np.log(w_in[b]) - np.log(w_out)) for b in w_in.keys()}
    g_mo = g_modularity(G, gnc, gamma, beta, w_out)
    return g_mo, m * g_mo + C

# the example of 2 cliques and 1 random graph in the resolution limit paper
def resolution_limit(): 
  gamma = 1.5
  for nS in range(100,300,10):
    G, gnc = c2r1(nC=13, nS=nS, kS=100)
    gnc_mergeC = {k: int(v>0)  for k, v in gnc.items()}
    gnc_splitS = {k: v if v > 0 else 3+np.random.randint(2) for k, v in gnc.items()}
    print(modularity(G,gnc_splitS,gamma=gamma), modularity(G,gnc,gamma=gamma), modularity(G,gnc_mergeC,gamma=gamma))

# the example of 2 cliques and 1 random graph in the resolution limit paper
def infer():
  nS = 200
  G, gnc = c2r1(nC=13, nS=nS, kS=100)
  gnc_mergeC = {k: int(v>0)  for k, v in gnc.items()}
  gnc_splitS = {k: v if v > 0 else 3+np.random.randint(2) for k, v in gnc.items()}
  print(ll(G,gnc_splitS,mode="modularity"), ll(G,gnc_mergeC,mode="modularity"), ll(G,gnc,mode="modularity"))

def NMI_dict(comm, gnc): 
  a = [comm[k] for k in comm.keys()]
  b = [gnc[k] for k in comm.keys()]
  return metrics.adjusted_mutual_info_score(a, b)

def test_resolution_limit():
  mo, l = ll(G,gnc,mode="modularity")
  print("Generalized Modularity=", mo)
  
  mo, l = ll(G,gnc,mode="density")
  print("Our model=", mo)
  
  G, gnc, Omega = dcSBM()
  
  a = nx.adjacency_matrix(G).todense()
  plt.imshow(a, cmap='hot', interpolation='nearest')
  plt.savefig("adjacency_matrix.png")
  h = Hierar(G, 2)
  comm = h.mdl()
  print(comm)
  
  for i in comm.keys():
    if comm[i] == 0:
      comm[i] = 1 
  print(mle(G, comm, mode="density"))

#infer()
#resolution_limit()
