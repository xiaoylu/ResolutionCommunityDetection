# code: utf-8
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
from math import sqrt
import pickle


#def xlogx(x):
#  if x < 1e-6: return 0.0
#  return x * np.log(x)

def pairs_without_selfloop(alist): 
  for r in alist: 
    for s in alist: 
      if r != s:
        yield (r,s)

class Para:
  GAMMA = 3   # Newman's scaling parameters

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

  def describe(self):
    #return len(self.nodes), 1.0*self.edges[self.ID]/sum(self.edges.values())
    return len(self.nodes), 1.0*self.edges[self.ID]/sum(self.edges.values()), self.nodes

  def __hash__(self):
    return self.ID 

  def __eq__(self, other):
    return self.ID == other.ID

class Hierar:
  def __init__(self, G, number_of_comm):
    self.G = G
    self.N = G.number_of_nodes()
    self.E = G.number_of_edges()
    print "GRAPH:", self.N, "nodes", self.E, "edges"
    self.reset()
    self.number_of_comm = number_of_comm

  def reset(self):
    self.GAMMA = Para.GAMMA
    self.blocks = []
    for node in self.G.nodes():
      block = Block(node, set([node]), Counter(self.G.neighbors(node)), GAMMA=self.GAMMA, E=self.E)
      self.blocks.append(block)
    self.active_bIDs = set(list(range(self.N))) 
    print "                                       (re)set parameters to GAMMA=", self.GAMMA

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

  def benefit(self, i, j):
    if i == j: return
    i, j = min(i,j), max(i,j)
    # change in log-l upon merging blocks i and j
    return self.blocks[i].merge_benefit(self.blocks[j]) 

  def mle_paras(self, comm): 
    k_r, k_r_in = defaultdict(float), defaultdict(float)
    for i, r in comm.items():
      k_r[r] += self.G.degree(i)
    for e in self.G.edges():
      if comm[e[0]] == comm[e[1]]:
        k_r_in[comm[e[0]]] += 2 

    win = sum(k_r_in.values()) / ( sum([np.power(_, 2) for _ in k_r.values()]) / 2./ self.E )
    print "posterior mle win=", win
    wout = (2.0 * self.E - sum(k_r_in.values())) \
           / (2. * self.E - ( sum([np.power(_, 2) for _ in k_r.values()]) / 2./ self.E ))
    print "posterior mle wout=", wout
    return win, wout

  def init_ranking(self):
    ranking = {}
    for i in self.active_bIDs:
      for j in list(self.blocks[i].edges): # neighbors only
        if i < j:
          ranking[(i,j)] = self.benefit(i,j)
    return ranking

  def mdl(self): 
    print 
    print 
    print 
    self.reset() # reset parameters and create blocks
    comm = self.hierar()
    print "Debug", Counter(comm.values())

    for _ in range(15):
      win, wout = self.mle_paras(comm)
      Para.GAMMA = (win - wout) / (np.log(win) - np.log(wout))
      self.reset() # consistant parameters now 
      comm = self.hierar()
      print "Debug", Counter(comm.values())

    ret = 0.
    for i in range(self.N):
      for j in range(self.N):
        Aij = int(self.G.has_edge(i,j))
        ki, kj = self.G.degree(i), self.G.degree(j)
        if comm[i] == comm[j]:
          w = win * ki * kj / 2. / self.E
        else: 
          w = wout * ki * kj / 2. / self.E
        ret += (Aij * np.log(w) - w)

    wi = [win * i * i / 2. / self.E for i in range(1,20,1)]
    wo = [wout * i * i / 2. / self.E for i in range(1,20,1)]

    #print " " * 70, "MDL=", -ret / 2.0
    print "with win=", win, "wout=", wout, "(s.t. gamma=%f)" % Para.GAMMA 
    #return -ret / 2.0, wi, wo, len(set(comm.values()))
    return comm 

    #pickle.dump(comm, open(output_path + "/gsbm_comm.p", "wb"))
    #print "pickle dump at", output_path + "/gsbm_comm.p"

  def try_gamma(self, vals, gnc): 
    from sklearn import metrics
    nmi_list, ncomm_list, accu_comm = [], [], []
    for _ in vals:
      Para.GAMMA = _
      self.reset() # reset parameters and create blocks
      comm = self.hierar()
      a = [comm[k] for k in comm.keys()]
      b = [gnc[k] for k in comm.keys()]
      nmi_list.append( metrics.adjusted_mutual_info_score(a, b) )
      ncomm_list.append( len(set(comm.values())) )
    return nmi_list, ncomm_list 

  # subroutine to find best partition by modularity maximization
  def hierar(self):
    ranking = self.init_ranking()

    print "-" * 200
    print "start merging"
    cur_ll= self.ll()
    max_ll = 0. 
    gsbm_comm = {}

    while len(self.active_bIDs) > 1: 
      B0 = len(self.active_bIDs)
      #B1 = int(B0 / 1.2)
      B1 = B0 - 1 
      
      #if len(self.active_bIDs) == 30:
      #  print "A"
      #  exit(1)

      #print B0,B1
      if ( len(ranking.keys()) < 1 ): ranking = self.init_ranking()
      if ( len(ranking.keys()) < 1 ): print "these are isolated strong connected components"; break 

      top_ops = sorted(ranking.items(), key=lambda x:x[1])[-(B0 - B1):]
      for (i,j), delta_Q in top_ops:
        #print delta_Q
        del ranking[(i,j)]
        if self.blocks[i].active and self.blocks[j].active:
          # to prevent the system being too sensitive that -0.00000001 stops the program 
          if delta_Q < -1e-4 \
               or ( len(self.active_bIDs) == self.number_of_comm ): # or this is the desired/least number of communities
            if cur_ll > max_ll:
              max_ll = cur_ll

              # save to pickle
              for idx, b in enumerate(self.active_bIDs):
                for node in self.blocks[b].nodes:
                  gsbm_comm[node] = idx

              # command line display
              print "=" * 20
              print "ll=", cur_ll, "+", delta_Q, "=", cur_ll + delta_Q
              print "Save result max_ll=", max_ll, "number of communities=", len(self.active_bIDs)
              print len(gsbm_comm), "nodes"
              print len(set(gsbm_comm.values())), "comms"
              print "=" * 20

          if ( len(self.active_bIDs) == self.number_of_comm ): return gsbm_comm # or this is the desired number of communities

          # merge two blocks i and j
          self.merge(i, j)
          #print "ll=", cur_ll, "+", delta_Q, "=", cur_ll + delta_Q
          #print
          cur_ll += delta_Q

          for b in set(self.blocks[i].edges):
            if b != i and b != j:
              # the neighbors of i or j
              # recompute b-->i
              if b < i: ranking[(b,i)] = self.benefit(b,i)
              else: ranking[(i,b)] = self.benefit(i,b)
        
          for b in set(self.blocks[j].edges):
            if b != i and b != j:
              del ranking[(min(b,j),max(b,j))]

    if max_ll > 0.:  
      print "final max_ll=", max_ll 
    else:
      print "Failed with inappriopate parameters gamma=", self.GAMMA
    return gsbm_comm

  def test1(self):
    # test if the change of log-l upon merging two blocks 
    # would be the same as function benefit() returns
    for _ in range(100):
      i, j = np.random.choice(list(self.active_bIDs)), np.random.choice(list(self.active_bIDs))
      if (i >= j): continue
      print '<' * 20
      print i, j, "benefit"
      print self.blocks[i].nodes, self.blocks[i].edges, self.blocks[i].block_ll()
      print self.blocks[j].nodes, self.blocks[j].edges, self.blocks[j].block_ll()
      be = self.benefit(i,j)
      oldl = self.ll()
      self.merge(i, j)
      newl = self.ll()
      dl = newl - oldl
      print '>' * 20
      if (dl - be) > 1e-5:
        print "Test failed. Change of Log-l", dl, "!= benefit", be
        print self.blocks[i].nodes, self.blocks[i].edges, self.blocks[i].block_ll()
        exit(1)
    print "Test Succeed"

def loadG(path):
  G = nx.Graph()
  with open(path, "rb") as txt:
    for line in txt: 
      if len(line) > 1 and line[0]!='#':
        e = line.split()
        G.add_edge(int(e[0]), int(e[1]))
  G = nx.convert_node_labels_to_integers(G)
  return G 

def test():
  #M, S = 5, 10 
  #G = nx.planted_partition_graph(M, S, 0.8, 0.05,seed=42)
  #G = nx.convert_node_labels_to_integers(G)

  #G = nx.Graph()
  #for j in range(2): G.add_edges_from([(10*j,10*j + i + 1) for i in range(4)])
  #G.add_edge(0,10)
  #G = nx.convert_node_labels_to_integers(G)
  #print G.edges()

  #path = "../data/eu_core/email-Eu-core.txt"
  #output_path = "../data/eu_core/"
  #path = "../data/karate/karate.txt"
  #output_path = "../data/karate/"
  #path = "../data/lesmis/lesmis.txt"
  #output_path = "../data/lesmis/"
  #path = "../data/dolphin/dolphins.txt"
  #output_path = "../data/dolphin/"
  #path = "../data/jazz/jazz.txt"
  #output_path = "../data/jazz/"
  path = "../data/football/football.txt"
  output_path = "../data/football/"
  G = loadG(path)
   
  #M, S = 10, 2 
  #G = nx.planted_partition_graph(M, S, 0.9, 0.1,seed=42)

  h = Hierar(G, 2)
  print "Graph Loaded"
  #h.test1()

  Para.GAMMA = 1.0; 
  h.mdl()

  #ret = []
  #for Para.GAMMA in np.linspace(0.4, 1.5, num=100): 
  #    ret.append(h.mdl())
  #ret.append(h.mdl())
  #print "MDL=", min(ret, key=lambda x:x[0])
  
if __name__=="__main__":
  test()
