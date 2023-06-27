#########################################################################
# File Name: graphreconstruc_util.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Tue 13 Nov 2018 01:48:15 PM +08
#########################################################################
#!/usr/bin/env/ python

import os
import operator
import graph_util as gutil
import numpy as np
import networkx as nx
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import random
import multiprocessing
import argparse
import os
import os.path

from algo_list import gr_classifier, gr_raw

def sample(data, ratio):
    fattr = '../data/%s/attr.txt'%data
    fgraph = '../data/%s/edgelist.txt'%data
    filename = '../data/%s/nodepairs.txt'%data

    n, m, directed = gutil.loadGraphAttr(fattr)

    G = gutil.loadGraphFromEdgeListTxt(fgraph, directed)

    if directed==False:
        m = m*2

    total_num = n * (n-1)
    print("real edge ratio: %f" % (m*1.0/total_num))
    num = int( total_num * ratio)
    print("sampling %d node pairs and writing to file: %s" % (num, filename))
    num_p = 0
    with open(filename, 'w') as fout:
        for i in range(num):
            u = random.randint(0,n-1)
            v = random.randint(0,n-1)
            p = 0
            if G.has_edge(u, v):
                p = 1
                num_p += 1
            fout.write(str(u)+" "+str(v)+" "+str(p)+"\n")
    print("positive samples: ", num_p)



class GraphReconstructionEval:
    def __init__(self, data, algo, d):
        self.embed = gutil.Embedding(data, algo, d, full=True)
        self.filename = '../data/%s/nodepairs.txt'%data

    def evalWithClassifier(self, node_pairs, node_pair_labels, negative_edges):
        embX = self.embed.get_combined_embedding()
        X_train= []
        Y_true = []

        edges = gutil.loadEdgeListFromTxt(self.embed.fgraph)

        if self.embed.directed:
            for (u, v) in edges:
                X_train.append( np.hstack([embX[u,:], embX[v,:]]) )
                Y_true.append(1)
        else:
            for (u, v) in edges:
                X_train.append( embX[u,:] * embX[v,:] )
                Y_true.append(1)

        if self.embed.directed:
            for (u, v) in negative_edges:
                X_train.append( np.hstack([embX[u,:], embX[v,:]]) )
                Y_true.append(0)
        else:
            for (u, v) in negative_edges:
                X_train.append( embX[u,:] * embX[v,:] )
                Y_true.append(0)

        print("training classifier...")
        classifier = LogisticRegression(random_state=1, solver='lbfgs')
        classifier.fit(X_train, Y_true)

        print("predicting link probability...")
        i=0
        pred_socres = {}
        if self.embed.directed:
            for (u, v) in node_pairs:
                x_u_v = np.hstack([embX[u,:], embX[v,:]])
                prob = classifier.predict_proba([x_u_v])
                pred_socres[i] = prob[0,1] #self.embed.link_prob(u, v)
                i+=1
        else:
            for (u, v) in node_pairs:
                x_u_v = embX[u,:] * embX[v,:]
                prob = classifier.predict_proba([x_u_v])
                pred_socres[i] = prob[0,1] #self.embed.link_prob(u, v)
                i+=1
        
        return pred_socres


    def eval(self):
        node_pairs = []
        node_pair_labels = []
        negative_edges = []
        i=0
        with open(self.filename, 'r') as fin:
            for line in fin:
                u, v, p = line.split()
                u, v, p = int(u), int(v), int(p)
                node_pairs.append( (u, v) )
                node_pair_labels.append(p)
                if p==0 and i<self.embed.get_m():
                    negative_edges.append( (u, v) )
                    i = i+1

        pred_socres = {}
        if self.embed.algo in gr_classifier:
            pred_socres = self.evalWithClassifier(node_pairs, node_pair_labels, negative_edges)
        elif self.embed.algo in gr_raw:
            i=0
            for (u, v) in node_pairs:
                pred_socres[i] = self.embed.link_prob(u, v)
                i+=1
        else:
            raise NotImplementedError

        print("compute precision@K...")
        sorted_scores = sorted(pred_socres.items(), key=operator.itemgetter(1), reverse=True)
        
        ks = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        pred_success_list = []
        for (key, score) in sorted_scores[0:ks[-1]]:
            p = 0
            if node_pair_labels[key]>0:
                p=1
            pred_success_list.append(p)

        precisions = []
        for k in ks:
            pred_success_ratio = sum(pred_success_list[0:k])*1.0/k
            precisions.append(pred_success_ratio)
            print("precision@%d: %f" % (k, pred_success_ratio))
        
        print(precisions)
        gutil.saveResult("GR", args.data, args.algo, precisions)
                

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

PARALLEL_LEVEL = multiprocessing.cpu_count()

print("CPU count: %d"%PARALLEL_LEVEL)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--algo', type=str, help='algorithm name')
    parser.add_argument('--d', default=128, type=int, help='embedding dimensionality')
    parser.add_argument('--data', type=str, help='graph dataset name')
    args = parser.parse_args()

    # sample(args.data, 0.01)

    gr = GraphReconstructionEval(args.data, args.algo, args.d)
    gr.eval()
