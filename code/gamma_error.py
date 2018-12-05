import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import random
random.seed(10)
from synthetic import draw, dcSBM, c2r1  
from collections import defaultdict
import numpy.random
np.random.seed(10)
from block_standard import Hierar
from util import mle  
from networkx.algorithms.community.modularity_max import * 
from sklearn import metrics


# save the graph in GDF format for gephi visualization
def save_gdf(filename, G, labels):
  with open("{}.gdf".format(filename), "w+") as txt:
      txt.write("nodedef>name VARCHAR,label VARCHAR")
      keys = labels.keys()
      for key in keys: txt.write(",{} VARCHAR".format(key))
      txt.write("\n")
      for i in G.nodes():
          txt.write("{},{}".format(i, i))
          for key in keys: txt.write(",{}".format(labels[key][i])) 
          txt.write("\n")

      txt.write("edgedef>node1 VARCHAR,node2 VARCHAR")
      for key in keys: txt.write(",{} VARCHAR".format(key))
      txt.write("\n")

      def edge_case(i, j):
          if i == j: return i 
          else: return -1

      for i, j in G.edges():
          txt.write("{},{}".format(i, j))
          for key in keys: txt.write(",{}".format(edge_case(labels[key][i], labels[key][j]))) 
          txt.write("\n")

def gamma_error_fig(): 
  N = 100
  n_comm = 10 
  comm_size = [10] * 10
  k = [3 + int(nx.utils.powerlaw_sequence(1, 2.5)[0]) for _ in range(N)]

  block = {}
  base = 0
  for t in range(len(comm_size)):
    for i in range(comm_size[t]):
      block[base+i] = t 
    base += comm_size[t]

  # prior values for generation
  #w = np.array([[10,1],[1,10]])
  sum_deg = np.zeros(n_comm) 
  sum_deg_in = np.zeros(n_comm) 
  for v, deg in enumerate(k):
    sum_deg[block[v]] += deg
  for _ in range(n_comm):
    sum_deg_in[_] = .9 * sum_deg[_]
  m = sum(sum_deg) / 2.

  # the estimation of the community's density 
  w = np.ones((n_comm, n_comm)) * \
      ((2.* m - sum(sum_deg_in))/(2.* m - sum(sum_deg_in[:]*sum_deg_in[:])/(2.*m)))
  for _ in range(n_comm):
    w[_][_] = (sum_deg_in[_] / sum_deg[_]) * (2. * m / sum_deg[_]) # percent of in-degree

  G = dcSBM(block, w, k)
  save_gdf("fig2_graph", G, {'comm': block}) 
  

  #w_in, w_out = mle(G, gnc, mode="density")
  #Omega = np.ones((n_comm, n_comm)) * w_out 
  #for _ in range(n_comm):
  #  Omega[_][_] = w_in[_] 

  ## posterior MLE 
  #sum_deg = np.zeros(n_comm) 
  #sum_deg_in = np.zeros(n_comm) 
  #for v, deg in G.degree():
  #  sum_deg[block[v]] += deg
  #for u, v in G.edges():
  #  if block[v] == block[u]:
  #    sum_deg_in[block[v]] += 2 
  #m = sum(sum_deg) / 2.
  #w = np.ones((n_comm, n_comm)) * \
  #    ((2.* m - sum(sum_deg_in))/(2.* m - sum(sum_deg_in[:]*sum_deg_in[:])/(2.*m)))
  #for _ in range(n_comm):
  #  w[_][_] = (sum_deg_in[_] / sum_deg[_]) * (2. * m / sum_deg[_]) # percent of in-degree
  Omega = w

  #h = Hierar(G, 2)
  #vals = np.linspace(0.3,12,num=40)
  #nmi_list, ncomm_list = h.try_gamma(vals, block, flag=True)
 
  # updated version
  nmi_list, ncomm_list = [], []
  print(nx.info(G))
  vals = np.linspace(0.3,15,num=40)
  for gamma in vals:
    comms = greedy_modularity_communities(G, resolution=gamma)
    a = list(map({v:i for i, c in enumerate(comms) for v in c}.get, list(G.nodes())))
    b = list(map(block.get, list(G.nodes())))

    nmi_list.append( metrics.adjusted_mutual_info_score(a, b) )
    ncomm_list.append(len(comms))
  print(nmi_list)
  print(ncomm_list)


  print("NMI scores=", nmi_list)
  gamma_error_fig_helper(vals, nmi_list, ncomm_list, Omega)
  print("Posterior Estimation of Omega, given the ground truth communities.")
  print(Omega)
  print("Program Done")

def gamma_error_fig_helper(vals, nmi_list, ncomm_list, Omega):
  plt.clf()
  fig, ax1 = plt.subplots()
  w0 = Omega[1][0]
  plt.axvline(x=w0,c='k',linestyle='--',linewidth=3, label=r"background $\omega_0$")
  WM, HM = 1.4, 0.1
  plt.text(w0-WM, HM, r"$\omega_0$",fontsize=25)
  omegas, indices = zip(*sorted([(Omega[i][i], i) for i in range(len(Omega))]))
  print("Omega rank of comms:", indices)
  plt.text(omegas[0]-WM, HM, r"$\omega_1$",fontsize=25)
  plt.text(omegas[1]-WM, HM, r"$\omega_2$",fontsize=25)
  plt.text(omegas[2]-WM, HM, r"$\omega_3$",fontsize=25)
  plt.text(omegas[3], HM, r"$\ldots$",fontsize=30)
  plt.axvline(x=Omega[0][0],c='r',linestyle='--',linewidth=2, label=r"communities $\omega_r$")
  for _ in range(1, Omega.shape[0], 1):
    plt.axvline(x=Omega[_][_],c='r',linestyle='--',linewidth=2)
  plt.legend(prop={'size': 19}, bbox_to_anchor=(-0.1, 1.1), ncol = 2, borderaxespad=0, frameon=False, loc='lower left')


  plt.tick_params(axis='both', which='major', labelsize=25)

  ax1.plot(vals,nmi_list,c='g',marker='8',markersize=10)
  ax2 = ax1.twinx()
  ax2.plot(vals,ncomm_list,c='b',marker='^',markersize=10)
  print("#comm=", ncomm_list)

  plt.tick_params(axis='both', which='major', labelsize=25)

  plt.xlim([-1.5,12])
  #ax1.set_xlabel(r"Resolution parameter $\gamma$",size="22")

  ax1.set_ylim([0.,1.1])
  #ax1.set_ylabel(r"NMI score",size="25", color='g')
  ax1.tick_params('y', colors='g')
  ax2.set_ylim(ymax=50)
  #ax2.set_ylabel(r"#Communities",size="25", color='b')
  ax2.tick_params('y', colors='b')

  plt.savefig("gamma_error.pdf", bbox_inches="tight")
  print("Draw gamma_error.pdf")


if __name__=="__main__":
  gamma_error_fig()
