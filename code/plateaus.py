import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np
import scipy
from scipy.stats import gaussian_kde
import networkx as nx
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn import metrics
from block_standard import Hierar, loadG, loadGNC
from util import modularity

def football(): 
  path = "../data/football/football.txt"
  gnc_path = "../data/football/footballTSEinputConference.clu"
  output_path = "../data/football/"
  G = loadG(path)
  gnc = loadGNC(gnc_path)
  gnc_list = defaultdict(list)
  for key, value in gnc.items():
      gnc_list[value].append(key)
  print "\n".join([str(_) for _ in gnc_list.values()])

  comm_list = pll(G)
  comm = {node:idx \
          for idx, nodes in enumerate(comm_list)\
          for node in nodes\
          }
  a = [comm[k] for k in comm.keys()]
  b = [gnc[k] for k in comm.keys()]
  print "NMI=", metrics.adjusted_mutual_info_score(a, b)

def pll(G):
  print "Loading..."
  print nx.info(G)
  print "Start"
  comm_list = pll_helper(G, gamma=.5)
  print len(comm_list), "Comms Detected."
  print "\n".join([str(_) for _ in comm_list])
  return comm_list

def pll_helper(G, gamma, indent=0, inflation_rate=1., low_gamma_risky=5.): 
  if G.number_of_nodes() <= 1: return [G.nodes()]
  h = Hierar(G, 2, gamma=gamma)
  comm = h.hierar(stop_at_max_modularity=True,verbose=False)
  print "\t"*indent, "L%d"%indent, "split", len(G.nodes()), "nodes into #blocks=", len(h.active_bIDs), "gamma=", gamma 

  if len(comm) == 0: # empty dict comm, because gamma is too high that every node is a comm
    # return the subgraph as one community
    print "\t"*indent, "*-*-* fInAL comm", G.nodes(), "gamma=", gamma
    return [list(G.nodes())]
  else: # else if LR hypo test
    win, wout = h.mle_paras(comm)
    twicell = h._2ll(comm, win, wout)
    print "\t"*indent, "-2ll=", twicell,
    pval = pvalue(G.copy(),comm,twicell,gamma,indent)
    #if (twicell) < 3.: 
    if (pval) > .3: 
      print "Stop with p-value=", pval
      print "\t"*indent, "*-*-* fInAL comm", G.nodes(), "gamma=", gamma
      return [list(G.nodes())]
    else:
      print "Proceed with p-value=", pval

  # recursion
  ret_comm = []
  pll_paras = []
  for bID in h.active_bIDs:
    #if h.blocks[bID].omega_in() < low_gamma_risky * gamma: 
      H = G.subgraph(h.blocks[bID].nodes)
      pll_paras.append((H, inflation_rate*gamma, indent+1)) # parameters
    #else:
    #  print "\t"*indent, "*-*-* fInAL comm", bID, h.blocks[bID].nodes, h.blocks[bID].omega_in(),\
    #                     "gamma=", gamma,\
    #                     "density=",h.blocks[bID].omega_in()
    #  ret_comm += [list(h.blocks[bID].nodes)]
  if len(pll_paras) > 0: # next level of the tree
    if indent == 0: # parallel on the first layer of recursion 
                    # as multiprocess does not support nested parallelization. :(
      ret_comm += reduce(lambda x,y:x+y,\
                         Parallel(n_jobs=1)( \
                         delayed(pll_helper)(*para) for para in pll_paras) \
                         )
    else:
      ret_comm += reduce(lambda x,y:x+y, [pll_helper(*para) for para in pll_paras])
    #print "\t"*indent, indent, "RET:", ret_comm
  return ret_comm

def pvalue(G,comm,twicell,gamma,indent):
  plt.clf()
  E = G.number_of_edges()
  deg_seq = [deg for n,deg in G.degree()]
  nullcomm = {i:comm[n] for i,(n,deg) in enumerate(G.degree())}
  L = 1000 
  null_distri = []
  for _ in range(L):
    F = nx.Graph(nx.configuration_model(deg_seq))
    E = F.number_of_edges() 
    win, wout = Hierar(F, 2, gamma=gamma).mle_paras(nullcomm)
    Q = modularity(F, nullcomm, gamma=(win - wout) / (np.log(win) - np.log(wout)) )
    B = E * (np.log(win) - np.log(wout)) 
    C = E * (np.log(wout) - wout) 
    null_distri.append( (B*Q + C + E) )
  plt.hist(null_distri, 35, normed=1, facecolor='green', alpha=0.5)
  kde = scipy.stats.gaussian_kde(null_distri,bw_method=None) 
  t_range = np.linspace(-1,3,100)
  plt.plot(t_range,kde(t_range),lw=2, label='KDE')
  plt.xlabel(r"$\Lambda_{\bf \hat{g}}$",size=45)
  plt.ylabel(r"Probablity Density",size=25)
  plt.xlim([-1,3])
  plt.tight_layout()
  plt.text(0.8, 0.8, r"-2LL=%.1f" % twicell,fontsize=35)
  plt.savefig("pvalue%d.png" % indent)
  print "savefig pvalue%d.png" % indent
  if (indent == 4): exit(1) 
  return 1.*sum([_>twicell for _ in null_distri])/L

#pll_test()
#pll()
football()
