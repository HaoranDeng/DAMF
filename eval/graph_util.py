import pickle
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import networkx as nx
import random
from random import randint
import itertools
import time
from time import time
import pdb
import os.path
import math
import sklearn
from sklearn import preprocessing
import pickle
from algo_list import *

APP="app"
VERSE="verse"
AROPE="arope"
NRP="nrp"
GA="ga"
RARE="rare"
PRUNE="prune"
GRAPHGAN="graphgan"
GRAPHWAVE="graphwave"
LINE="line"
DEEPWALK="deepwalk"
N2V="n2v"
PBG="pbg"
NETSMF='netsmf'
STRAP="strap"
DNGR="dngr"
PRONE="prone"
ADJ="adj"
GLODYNE="glodyne"
DYADJ="dyadj"
DAMF="damf"
DAMF1="damf1"
LINE="line"

def saveResult(task, data, algo, res):
    assert task in ["NC", "LP", "GR"]
    if not os.path.exists("results"):
        os.makedirs("results")

    file_name = "results/"+task+"_"+data+".pkl"
    print(f"Wrting result to file: {file_name}")
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            A = pickle.load(f)
    else:
        A = {}
    
    A[algo] = res

    with open(file_name, "wb") as f:
        pickle.dump(A, f)

def loadGraphAttr(file_name):
    if not file_name or not os.path.exists(file_name):
        raise Exception("attr file not exist!")
    with open(file_name) as fin:
        n = int(fin.readline().split("=")[1])
        m = int(fin.readline().split("=")[1])
        directed = (fin.readline().strip()=="directed")
    return n, m, directed

def loadGraphFromEdgeListTxt(file_name, directed):
    if not file_name or not os.path.exists(file_name):
        raise Exception("edgelist file not exist!")
    t1 = time()
    with open(file_name, 'r') as f:
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for line in f:
            edge = line.strip().split()
            G.add_edge(int(edge[0]), int(edge[1]))

    print('%fs taken for loading graph' % (time() - t1))
    show_graph_info(G)
    return G

def loadNodesEdgesFromEdgeListTxt(file_name, directed):
    if not file_name or not os.path.exists(file_name):
        raise Exception("edgelist file not exist!")
    t1 = time()
    nodes = set()
    edges = []
    with open(file_name, 'r') as f:
        for line in f:
            edge = line.strip().split()
            u, v = int(edge[0]), int(edge[1])
            nodes.add(u)
            nodes.add(v)
            edges.append( (u, v) )
            if not directed:
                edges.append( (v, u) )

    print('%fs taken for loading graph' % (time() - t1))
    return nodes, edges


def loadEdgeListFromTxt(file_name):
    if not file_name or not os.path.exists(file_name):
        print(f"file_name: {file_name}")
        raise Exception("edgelist file not exist!")
    edges = []
    with open(file_name, 'r') as f:
        for line in f:
            edge = line.strip().split()
            edges.append((int(edge[0]), int(edge[1])))
    return edges

def loadLabels(file_name, n):
    if not file_name or not os.path.exists(file_name):
        raise Exception("label file not exist!")
    N = set() # labelled nodes
    L = set() # labels
    Y = [set() for i in xrange(n)]
    is_multiple = False
    with open(file_name, 'r') as f:
        for line in f:
            s = line.strip().split()
            node = int(s[0])
            if node>=n:
                break
            if len(s)>1:
                N.add(node)
                for label in s[1:]:
                    label = int(label)
                    L.add(label)
                    Y[node].add(label)
                if is_multiple==False and len(Y[node])>1:
                    is_multiple = True
                    
    print("number of labels: %d, multilabel=%s, num of labelled nodes: %d"%(len(L), str(is_multiple), len(N)))
    return L, N, Y, is_multiple

def show_graph_info(G):
    print('Num of nodes: %d, num of edges: %d, Avg degree: %f, Directed:%s' % \
        (G.number_of_nodes(), G.number_of_edges(), G.number_of_edges()*1./G.number_of_nodes(), str(nx.is_directed(G)) ))


def splitDiGraphToTrainTest(file_name, num_test_edge):
    edges = loadEdgeListFromTxt(file_name)
    print("edgelist loaded")
    random.shuffle(edges)
    test_edges = edges[:num_test_edge]
    train_edges = edges[num_test_edge:]

    return (train_edges, test_edges)

def selectNegativeEdges(file_name, n, num_negative, directed):
    if not file_name or not os.path.exists(file_name):
        raise Exception("edgelist file not exist!")
    edges = [ [] for i in range(n)]

    with open(file_name, 'r') as f:
        if directed==False:
            for line in f:
                edge = line.strip().split()
                s, t = int(edge[0]), int(edge[1])
                if s<t:
                    edges[s].append(t)
                else:
                    edges[t].append(s)
        else:
            for line in f:
                edge = line.strip().split()
                s, t = int(edge[0]), int(edge[1])
                edges[s].append(t)
    
    print("edgelist loaded")

    negative_edges = []

    if directed==False:
        for i in range(num_negative):
            u = random.randint(0,n-1)
            while True:
                v = random.randint(0,n-1)
                if u<v:
                    if v not in edges[u]:
                        edges[u].append(v)
                        negative_edges.append( (u, v) )
                        break
                    else:
                        continue
                else:
                    if u not in edges[v]:
                        edges[v].append(u)
                        negative_edges.append( (v, u) )
                        break
                    else:
                        continue

    else:
        for i in range(num_negative):
            u = random.randint(0,n-1)
            while True:
                v = random.randint(0,n-1)
                if v not in edges[u]:
                    edges[u].append(v)
                    negative_edges.append( (u, v) )
                    break
                elif u not in edges[v]:
                    edges[v].append(u)
                    negative_edges.append( (v, u) )
                    break
                else:
                    continue

    
    del edges

    if len(negative_edges)<num_negative:
        print("no enough negative edges are selected...")
    return negative_edges


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Embedding:
    def __init__(self, data, algo, d, suffix="", full=True):
        self.d=d
        self.algo=algo
        self.data=data
        self.suffix = suffix

        fattr = f'../data/{data}/attr.txt'
        print(f"Attr file: {fattr}")

        self.n, self.m, self.directed = loadGraphAttr(fattr)

        print("number of nodes: %d, number of edges: %d, directed: %r"%(self.n, self.m, self.directed))

        if full==True:
            self.loadEmbedding4FullGraph()
        else:
            self.loadEmbedding4TrainGraph()

    def loadEmbedding4FullGraph(self):
        self.fgraph = '../data/%s/edgelist.txt'%self.data
        self.fembed = '../embds/%s/%s.%s.bin'%(self.data, self.algo+self.suffix, str(self.d))
        print(f"Graph file: {self.fgraph}")
        print(f"Embedding file: {self.fembed}")
        print("number of nodes: %d, number of edges: %d"%(self.n, self.m))

        # loadGraphFromEdgeListTxt(fgraph, self.directed)

        if self.algo==VERSE:
            self.loadEmbedding(datatype=np.float32)
        elif self.algo==APP:
            self.loadSrcTgtEmbedding(datatype=np.float64, mode="txt")
        elif self.algo==RARE:
            self.loadRanking()
            self.loadEmbedding(datatype=np.float64, mode="txt")
        elif self.algo==N2V or self.algo==PBG or self.algo == LINE or self.algo==DEEPWALK:
            print(":::load embedding from text")
            self.loadEmbedding(datatype=np.float64, mode="txt")
        elif self.algo in algo_list_XY:
            self.loadSrcTgtEmbedding(datatype=np.float64)
        else:
            self.loadEmbedding(datatype=np.float64)

    def loadEmbedding4TrainGraph(self):
        ftraingraph = '../data/%s/edgelist.train.txt'%self.data
        ftestgraph = '../data/%s/edgelist.test.txt'%self.data
        fnegativegraph = '../data/%s/edgelist.negative.txt'%self.data
        self.fembed = '../embds/%s/%s.%s.train.bin'%(self.data, self.algo, str(self.d))
        
        train_nodes, self.train_edges = loadNodesEdgesFromEdgeListTxt(ftraingraph, self.directed)
        self.n_train = len(self.train_edges)

        self.test_edges = loadEdgeListFromTxt(ftestgraph)
        self.n_test = len(self.test_edges)

        self.train_edges = loadEdgeListFromTxt(ftraingraph)
        self.n_train = len(self.train_edges)

        self.negative_edges = loadEdgeListFromTxt(fnegativegraph)
        self.n_negative = len(self.negative_edges)

        print("train edges: %d, test edges: %d" % (self.n_train, self.n_test))

        if self.algo==VERSE:
            self.loadEmbedding(datatype=np.float32, nonhidden_nodes=train_nodes)
        elif self.algo==APP:
            self.loadSrcTgtEmbedding(datatype=np.float64, nonhidden_nodes=train_nodes, mode="txt")
        elif self.algo==RARE:
            self.loadRanking(nonhidden_nodes=train_nodes)
            self.loadEmbedding(datatype=np.float64, nonhidden_nodes=train_nodes, mode="txt")
        elif self.algo==N2V or self.algo==PBG or self.algo==DEEPWALK:
            self.loadEmbedding(datatype=np.float64, nonhidden_nodes=train_nodes, mode="txt")
        elif self.algo in algo_list_XY:
            self.loadSrcTgtEmbedding(datatype=np.float64, nonhidden_nodes=train_nodes)
        else:
            self.loadEmbedding(datatype=np.float64, nonhidden_nodes=train_nodes)
        del train_nodes
    
    def get_d(self):
        return self.d
    
    def get_algo(self):
        return self.algo

    def is_directed(self):
        return (self.directed==True)
    
    def get_train_edges(self):
        return self.train_edges

    def get_test_edges(self):
        return self.test_edges

    def get_negative_edges(self):
        return self.negative_edges

    def get_n_train(self):
        return self.n_train

    def get_n_test(self):
        return self.n_test
    
    def get_n_negative(self):
        return self.n_negative
    
    def get_n(self):
        return self.n

    def get_m(self):
        return self.m

    def get_combined_embedding(self):
        if self.algo==RARE:
            return self.X
        #elif self.algo==APP or self.algo==AROPE or self.algo==GA or self.algo==STRAP:
        #    return np.hstack([self.X, self.Y])
        elif self.algo==APP or self.algo==AROPE or self.algo==GA or self.algo==STRAP:
            XX = preprocessing.normalize(self.X, norm='l2', axis=1)
            YY = preprocessing.normalize(self.Y, norm='l2', axis=1)
            return np.hstack([XX, YY])
            #return np.hstack([self.X, self.Y])
        elif self.algo==NRP or self.algo==TMP:
            XX = self.X
            YY = self.Y
            #XX = preprocessing.power_transform(XX, method='yeo-johnson', standardize=False)
            #YY = preprocessing.power_transform(YY, method='yeo-johnson', standardize=False)
            XX = preprocessing.normalize(XX, norm='l2', axis=1)
            YY = preprocessing.normalize(YY, norm='l2', axis=1)
            XY = np.hstack([XX, YY])
            return XY
        else:
            return self.X

    def proximity(self, s, t):
        if self.algo==DAMF or self.algo==TMP or self.algo==DPRMF:
            if self.directed:
                return np.dot(self.X[s, :], self.Y[t, :])
            else:
                return max(np.dot(self.X[s, :], self.Y[t, :]), np.dot(self.X[t, :], self.Y[s, :]))
        elif self.algo == GLODYNE:
            A, B = self.X[s, :], self.X[t, :]
            return np.dot(A, B)
        else:
            return np.dot(self.X[s, :], self.X[t, :])
    
    def link_prob(self, s, t):
        score = self.proximity(s, t)
        # score = sigmoid(self.proximity(s, t))
        return score
    
    def loadFromTxt(self, fname, dim, datatype, nonhidden_nodes):
        if not fname or not os.path.exists(fname):
            raise Exception("%s embedding txt file not exist!"%fname)

        X = np.zeros((self.n, dim), dtype=datatype)
        with open(fname, 'rb') as f:
            lines = f.readlines()
            if nonhidden_nodes!=None:
                max_nid = max(nonhidden_nodes)
            else:
                max_nid = self.n-1
            if len(lines)==self.n:
                print("full embeddings")
                # X = arr.reshape(self.n, dim)
                i=0
                for line in lines:
                    values = line.strip().split()
                    X[i, :] = [float(value) for value in values]
                    i+=1
            elif len(lines)==max_nid+1:
                print("partial embeddings type 1")
                i=0
                for line in lines:
                    values = line.strip().split()
                    X[i, :] = [float(value) for value in values]
                    i+=1
                print("total number of nodes in original graph: %d, total number of nodes in embedded graph: %d"%(self.n, max_nid+1))
            else:
                number_of_nonhidden_nodes = len(nonhidden_nodes)
                id2node = dict(zip(range(number_of_nonhidden_nodes), sorted(map(int, nonhidden_nodes))))
                if len(lines)==number_of_nonhidden_nodes:
                    print("partial embeddings type 2")
                    i=0
                    for line in lines:
                        values = line.strip().split()
                        srcid = id2node[i]
                        X[srcid, :] = [float(value) for value in values]
                        i+=1
                else:
                    print("partial embeddings type 3")
                    print("error: only %d nodes are embedded"%(len(lines)))
                print("total number of nodes in original graph: %d, total number of nodes in nonhidden graph: %d"%(self.n, number_of_nonhidden_nodes))
        
        print("embedding matrxi shape:", X.shape)
        return X

    def loadFromBin(self, fname, dim, datatype, nonhidden_nodes):
        if not fname or not os.path.exists(fname):
            raise Exception("%s embedding bin file not exist!"%fname)

        with open(fname, 'rb') as f:
            arr = np.fromfile(f, dtype=datatype)

        X = np.zeros((self.n, dim), dtype=datatype)
        if nonhidden_nodes!=None:
            max_nid = max(nonhidden_nodes)
        else:
            max_nid = self.n-1  
        print("arr len: %d, n: %d, max-node-id: %d"%(len(arr), self.n, max_nid) )
        if len(arr)==self.n*dim:
            print("full embeddings")
            X = arr.reshape(self.n, dim)
        elif len(arr)==(max_nid+1)*dim:
            print("partial embeddings type 1")
            for i in range(max_nid+1):
                X[i, :] = arr[i*dim:(i+1)*dim]
            print("total number of nodes in original graph: %d, total number of nodes in embedded graph: %d"%(self.n, max_nid+1))
        else:
            number_of_nonhidden_nodes = len(nonhidden_nodes)
            id2node = dict(zip(range(number_of_nonhidden_nodes), sorted(map(int, nonhidden_nodes))))
            if len(arr)==number_of_nonhidden_nodes*dim:
                print("partial embeddings type 2")
                for i in range(number_of_nonhidden_nodes):
                    srcid = id2node[i]
                    X[srcid, :] = arr[i*dim:(i+1)*dim]
            else:
                print("partial embeddings type 3")
                print("error: only %d nodes are embedded"%(len(arr)/dim))
            print("total number of nodes in original graph: %d, total number of nodes in nonhidden graph: %d"%(self.n, number_of_nonhidden_nodes))
        
        print("embedding matrxi shape:", X.shape)
        return X


    def loadEmbedding(self, datatype=np.float64, nonhidden_nodes=None, mode="bin"):

        if mode=="bin":
            self.X = self.loadFromBin(self.fembed, self.d, datatype, nonhidden_nodes)
        else:
            self.X = self.loadFromTxt(self.fembed, self.d, datatype, nonhidden_nodes)

    
    def loadSrcTgtEmbedding(self, datatype=np.float64, nonhidden_nodes=None, mode="bin"):

        if mode=="bin":
            self.X = self.loadFromBin(self.fembed+".src", self.d//2, datatype, nonhidden_nodes)
            self.Y = self.loadFromBin(self.fembed+".tgt", self.d//2, datatype, nonhidden_nodes)
        else:
            self.X = self.loadFromTxt(self.fembed+".src", self.d//2, datatype, nonhidden_nodes)
            self.Y = self.loadFromTxt(self.fembed+".tgt", self.d//2, datatype, nonhidden_nodes)
        
        # print self.X[0]
        # print self.Y[0]

    
    def loadRanking(self, nonhidden_nodes=None):
        fname = self.fembed+".r"
        if not fname or not os.path.exists(fname):
            raise Exception("%s ranking txt file not exist!"%fname)

        self.ranking = [0]*self.n
        with open(fname, 'rb') as f:
            lines = f.readlines()
            if nonhidden_nodes!=None:
                max_nid = max(nonhidden_nodes)
            else:
                max_nid = self.n-1
            if len(lines)==self.n:
                print("full embeddings")
                i=0
                for line in lines:
                    value = float(line.strip())
                    self.ranking[i] = value
                    i+=1
            elif len(lines)==max_nid+1:
                print("partial embeddings type 1")
                i=0
                for line in lines:
                    value = float(line.strip())
                    self.ranking[i] = value
                    i+=1
                print("total number of nodes in original graph: %d, total number of nodes in embedded graph: %d"%(self.n, max_nid+1))
            else:
                number_of_nonhidden_nodes = len(nonhidden_nodes)
                id2node = dict(zip(range(number_of_nonhidden_nodes), sorted(map(int, nonhidden_nodes))))
                if len(lines)==number_of_nonhidden_nodes:
                    print("partial embeddings type 2")
                    i=0
                    for line in lines:
                        value = float(line.strip())
                        srcid = id2node[i]
                        self.ranking[srcid] = value
                        i+=1
                else:
                    print("partial embeddings type 3")
                    print("error: only %d nodes are embedded"%(len(lines)))
                print("total number of nodes in original graph: %d, total number of nodes in nonhidden graph: %d"%(self.n, number_of_nonhidden_nodes))
