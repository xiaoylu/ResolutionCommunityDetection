import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import random
random.seed(10)
from collections import defaultdict
import numpy.random
np.random.seed(10)
from block_standard import Hierar, loadG


def real_test():
  ##path = "../data/eu_core/email-Eu-core.txt"
  ##output_path = "../data/eu_core/"
  ##path = "../data/jazz/jazz.txt"
  ##output_path = "../data/jazz/"

  #path = "../data/karate/karate.txt"
  #output_path = "../data/karate/"
  #gamma, n_comm = 0.78, 2

  #path = "../data/lesmis/lesmis.txt"
  #output_path = "../data/lesmis/"
  #gamma, n_comm = 1.36, 6 

  #path = "../data/dolphin/dolphins.txt"
  #output_path = "../data/dolphin/"
  #gamma, n_comm = 0.59, 2 

  path = "../data/football/football.txt"
  output_path = "../data/football/"
  gamma, n_comm = 2.27, 11

  G = loadG(path)

  # results using optimal gamma
  h = Hierar(G, n_comm, gamma=gamma)
  comm = h.hierar(stop_at_max_modularity=False,verbose=False)
  print len(G.nodes()), "nodes into #blocks=", len(h.active_bIDs), "gamma=", gamma 
  w1 = min([h.blocks[bID].omega_in() for bID in h.active_bIDs])
  n_n, w0 = h.mle_paras(comm)
  print "(%.2f, %.2f)" % (w0, w1)

  # try different gammas
  gnc = comm
  vals = np.linspace(0., w1*1.5, num=80)
  nmi_list, ncomm_list = h.try_gamma(vals, gnc, flag=False)

  exit(1)

  plt.axvline(x=w0,c='k',linestyle='--',linewidth=3)
  plt.axvline(x=w1,c='r',linestyle='--',linewidth=3)
  WM, HM = 0.25, 0.1

  plt.ylim(ymax=1.1)
  plt.text(w0-WM, HM, r"$\omega_0$",fontsize=38)
  plt.text(w1-WM, HM, r"$\omega_1$",fontsize=38)
  plt.plot(vals, nmi_list, marker='8', markersize=7)
  plt.tick_params(axis='both', which='major', labelsize=25)

  plt.xlabel(r"$\gamma$",size="40")
  plt.ylabel("NMI score",size="30")
  plt.savefig("nmi.png", bbox_inches="tight")

if __name__=="__main__":
  real_test()
