from collections import Counter, defaultdict
import numpy as np
import networkx as nx
from math import sqrt
import pickle
from sklearn import metrics
from itertools import combinations
from scipy import stats


class Block:
  def __init__(self, ID, nodes, edges, GAMMA, E):
    self.E = E

    self.ID = ID #unique ID as hash
    self.nodes = nodes
    self.edges = edges
    self.GAMMA = GAMMA

    self.kappa = sum([_ for _ in edges.values()])  
    self.active = True

  def merge(self, block): # merge with blocks' ougoing edges
    if self.ID >= block.ID: return 
    self.nodes = self.nodes.union(block.nodes)
    self.kappa = self.kappa + block.kappa
    self.edges = self.edges + block.edges
    if self.edges[block.ID] > 0: # merge the edges between them 
      self.edges.update({self.ID:self.edges[block.ID]})
      del self.edges[block.ID]

  def redirect(self, i, j): # merge two blocks i and j's ingoing edges
    i, j = min(i,j), max(i,j)
    self.edges.update({i:self.edges[j]})
    del self.edges[j]
     
  def merge_benefit(self, block):  
    merge_kappa = self.kappa + block.kappa
    return 1.0 * self.edges[block.ID] / self.E - self.GAMMA * self.kappa * block.kappa / (2.0 * self.E * self.E)

  def block_ll(self, edges = None):
    ret = (self.edges[self.ID] / self.E) - self.GAMMA * np.power(( self.kappa  / 2.0 / self.E ), 2)
    return ret 

  def omega_in(self): 
    return (1. * self.edges[self.ID] / self.kappa) * (2. * self.E / self.kappa) # percent of in-degree

  def describe(self):
    #return len(self.nodes), 1.0*self.edges[self.ID]/sum(self.edges.values())
    return len(self.nodes), 1.0*self.edges[self.ID]/sum(self.edges.values()), self.nodes

  def __hash__(self):
    return self.ID 

  def __eq__(self, other):
    return self.ID == other.ID

class Hierar:
  def __init__(self, G, least_num_of_comm = 2, gamma = 1.0):
    self.G = G
    self.N = G.number_of_nodes()
    self.E = G.number_of_edges()
    if self.N == 1 or self.E == 0: print("Input network has 1 node or 0 edges. Exit."); exit(1) 
    self.reset(gamma)
    self.least_num_of_comm = least_num_of_comm
    # constant used for the exact log-likelihood in unweighted graphs  
    self.D = - self.E * np.log(2. * self.E)\
             + sum([float(deg)*np.log(1.*deg) for _, deg in self.G.degree()])  

  def reset(self, gamma):
    #print "-" * 200
    #print "GRAPH:", self.N, "nodes", self.E, "edges"
    self.GAMMA = gamma 

    # every node is a block
    self.blocks = {} 
    for node in self.G.nodes():
      block = Block(node, set([node]), Counter(self.G.neighbors(node)), GAMMA=self.GAMMA, E=self.E)
      self.blocks[node] = block
    self.active_bIDs = set(list(self.G.nodes()))

    #print "\t" * 5, "(re)set parameters to GAMMA=", self.GAMMA, ",num of initial blocks=", len(self.active_bIDs)

  def merge(self, i, j):
    if i == j: return
    i, j = min(i,j), max(i,j)
    self.blocks[i].merge(self.blocks[j]) # merge blocks i and j

    for b in set(self.blocks[j].edges):
      self.blocks[b].redirect(i, j) # redirect edges b--j to be b--i

    self.blocks[j].active = False
    self.active_bIDs.remove(j) #delete j
    #print len(self.active_bIDs), "Blocks remaining"

  def ll(self):
    return sum([self.blocks[b].block_ll() for b in self.active_bIDs])

  # change in log-l upon merging blocks i and j
  def benefit(self, i, j):
    if i == j: return
    i, j = min(i,j), max(i,j)
    return self.blocks[i].merge_benefit(self.blocks[j]) 

  # planted partition network posterior estimation 
  def mle_paras(self, comm): 
    k_r, k_r_in = defaultdict(float), defaultdict(float)
    for i, r in comm.items():
      k_r[r] += self.G.degree(i)
    for e in self.G.edges():
      if comm[e[0]] == comm[e[1]]:
        k_r_in[comm[e[0]]] += 2 

    win = sum(k_r_in.values()) / ( sum([np.power(_, 2) for _ in k_r.values()]) / 2./ self.E )
    wout = (2.0 * self.E - sum(k_r_in.values())) \
           / (2. * self.E - ( sum([np.power(_, 2) for _ in k_r.values()]) / 2./ self.E ))
    return win, wout

  def init_ranking(self):
    ranking = {}
    for i in self.active_bIDs:
      for j in list(self.blocks[i].edges): # neighbors only
        if i < j: # the ranking keys (i,j) s.t. i < j
          ranking[(i,j)] = self.benefit(i,j)
    return ranking

  # log-lieklihood ratio test (-2LL)
  def _2ll(self, comm, win, wout):
    ni  = defaultdict(int)
    for i in comm.keys(): ni[comm[i]] += 1
    B = self.E * (np.log(win) - np.log(wout)) 
    C = self.E * (np.log(wout) - wout) 
    #print (np.log(win) - np.log(wout)), (np.log(wout) - wout) 
    model_length = float(self.N) * stats.entropy(1.*np.array(ni.values()) / self.N)
    exact_log_l = B * self.max_modularity + C + self.D
    #return model_length, exact_log_l, self.max_modularity, model_length+exact_log_l
    #return exact_log_l, - self.E + self.D
    return 2.*(exact_log_l - self.D + self.E)
    #return model_length+exact_log_l

  # the statistical inference of gamma and communities
  def stat_infer(self, gamma = 1.0, rule = "standard"): 
    self.reset(gamma) # reset parameters and create blocks
    comm = self.hierar()
    print("First trial gamma=",gamma, "resulting #comm=", len(Counter(comm.values())))
    new_gamma = gamma
    for _ in range(15): # number of iterations
      win, wout = self.mle_paras(comm)
      print("-2ll=", self._2ll(comm, win, wout))
      print()
      if rule == "standard":
        new_gamma = (win - wout) / (np.log(win) - np.log(wout))
      elif rule == "bound":
        new_gamma = (win + wout) / 2.5 
      print("update win=", win, "wout=", wout, "(s.t. next gamma=%f)" % new_gamma)
      self.reset(new_gamma) # consistant parameters now 
      comm = self.hierar(stop_at_max_modularity=True, verbose=False)
      print("resulting #comm=", len(Counter(comm.values())))
    return comm 

  # iterate over different gamma values
  def try_gamma(self, vals, gnc, flag=False): 
    from sklearn import metrics
    nmi_list, ncomm_list, accu_comm = [], [], []
    for gamma in vals:
      self.reset(gamma) # reset parameters and create blocks
      comm = self.hierar(stop_at_max_modularity=flag)
      if len(comm) < 1: nmi_list.append( 0.0 )
      else:
        a = [comm[k] for k in comm.keys()]
        b = [gnc[k] for k in comm.keys()]
        nmi_list.append( metrics.adjusted_mutual_info_score(a, b) )
      ncomm_list.append( len(set(comm.values())) )
    return nmi_list, ncomm_list 

  # subroutine to find best partition by modularity maximization
  # merging the best pair of comms to increase modularity
  # @stop_at_max_modularity: False, then the algorithm merge blocks until two blocks remain
  #                          True, then return immediately after modularity decreases   
  def hierar(self, stop_at_max_modularity = False, verbose=False):
    ranking = self.init_ranking()

    if verbose: print("start merging")

    cur_ll= self.ll()
    max_ll = 0. 
    gsbm_comm = {}

    while len(self.active_bIDs) > 1: 
      B0 = len(self.active_bIDs)
      B1 = B0 - 1 # = int(B0 / 1.2) #speedup the merging process (radically)

      if ( len(ranking.keys()) < 1 ): ranking = self.init_ranking()
      if ( len(ranking.keys()) < 1 ): #print "Error: isolated graph";
        break 

      top_ops = sorted(ranking.items(), key=lambda x:x[1])[-(B0 - B1):]
      for (i,j), delta_Q in top_ops:
        #print delta_Q
        del ranking[(i,j)]
        if self.blocks[i].active and self.blocks[j].active:
          # to prevent the system being too sensitive that -0.00000001 stops the program 
          if delta_Q < -1e-4 \
               or ( len(self.active_bIDs) == self.least_num_of_comm ): # or this is the desired/least number of communities
            if cur_ll > max_ll:
              max_ll = cur_ll
              self.max_modularity = max_ll

              # save to pickle
              for idx, b in enumerate(self.active_bIDs):
                for node in self.blocks[b].nodes:
                  gsbm_comm[node] = idx

              # command line display
              if verbose:
                print("=" * 20)
                print("ll=", cur_ll, "+", delta_Q, "=", cur_ll + delta_Q)
                print("Save result max_ll=", max_ll, "number of communities=", len(self.active_bIDs))
                print(len(gsbm_comm), "nodes")
                print(len(set(gsbm_comm.values())), "comms")
                print("=" * 20)

            if stop_at_max_modularity: return gsbm_comm # empty if every single node is a community

          if ( len(self.active_bIDs) == self.least_num_of_comm ): 
            print("Quit with least number of community = ", len(self.active_bIDs), "cur_ll=", cur_ll)
            for idx, b in enumerate(self.active_bIDs):
              for node in self.blocks[b].nodes:
                gsbm_comm[node] = idx
            return gsbm_comm # or this is the least number of communities

          # merge two blocks i and j
          self.merge(i, j)
          #print "ll=", cur_ll, "+", delta_Q, "=", cur_ll + delta_Q
          cur_ll += delta_Q

          for b in set(self.blocks[i].edges):
            if b != i and b != j:
              # b is the neighbor of i or j
              # recompute change of modularity upon joining b-->i
              if b < i: ranking[(b,i)] = self.benefit(b,i)
              else: ranking[(i,b)] = self.benefit(i,b)
        
          for b in set(self.blocks[j].edges):
            if b != i and b != j:
              del ranking[(min(b,j),max(b,j))]

    if max_ll > 0.:  
      print("final max_ll=", max_ll)
    else:
      print("Failed with inappriopate parameters gamma=", self.GAMMA)
    return gsbm_comm

  # test if the change of log-l upon merging two blocks is always correct
  def test1(self):
    # would be the same as function benefit() returns
    for _ in range(100):
      i, j = np.random.choice(list(self.active_bIDs)), np.random.choice(list(self.active_bIDs))
      if (i >= j): continue
      print('<' * 20)
      print(i, j, "benefit")
      print(self.blocks[i].nodes, self.blocks[i].edges, self.blocks[i].block_ll())
      print(self.blocks[j].nodes, self.blocks[j].edges, self.blocks[j].block_ll())
      be = self.benefit(i,j)
      oldl = self.ll()
      self.merge(i, j)
      newl = self.ll()
      dl = newl - oldl
      print('>' * 20)
      if (dl - be) > 1e-5:
        print("Test failed. Change of Log-l", dl, "!= calculated by benefit() which is", be)
        print(self.blocks[i].nodes, self.blocks[i].edges, self.blocks[i].block_ll())
        exit(1)
    print("Test Succeed")

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

def test():
  gnc = {}
  gnc_path = ""
  ##path = "../data/eu_core/email-Eu-core.txt"
  ##output_path = "../data/eu_core/"
  ##path = "../data/jazz/jazz.txt"
  ##output_path = "../data/jazz/"

  #path = "../data/karate/karate.txt"
  #output_path = "../data/karate/"
  #path = "../data/lesmis/lesmis.txt"
  #output_path = "../data/lesmis/"
  #path = "../data/dolphin/dolphins.txt"
  #output_path = "../data/dolphin/"
  path = "../data/football/football.txt"
  output_path = "../data/football/"
  gnc_path = "../data/football/footballTSEinputConference.clu"

  G = loadG(path)
  if len(gnc_path) > 0: gnc = loadGNC(gnc_path)
  h = Hierar(G, least_num_of_comm = 2)
  print("Graph Loaded")
  print(nx.info(G))

  label_set = [] 
  for rule in ["standard", "bound"]:
    comm = h.stat_infer(gamma=.5,rule=rule)

    label_set.append(comm)

    if len(gnc) > 0:
      a = [comm[k] for k in comm.keys()]
      b = [gnc[k] for k in comm.keys()]
      print("Ground-truth NMI=", metrics.adjusted_mutual_info_score(a, b))
    exit(1)

  for x,y in combinations(label_set, 2): 
    a = [x[k] for k in x.keys()]
    b = [y[k] for k in x.keys()]
    print("Mutual NMI=", metrics.adjusted_mutual_info_score(a, b))

if __name__=="__main__":
  test()
  #pass
