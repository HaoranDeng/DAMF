#########################################################################
# File Name: linkpred_util.py
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
from scipy import spatial
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


class LinkPredictionEval:
    def __init__(self, data, algo, d):
        self.data = data
        self.algo = algo
        self.embed = gutil.Embedding(data, algo, d, full=False)
        
    def eval(self):
        pred_node_score = {}
        pred_test = []
        i = 0
        labels = []
        for (u, v) in self.embed.get_test_edges():
            pred_score = self.embed.link_prob(u, v)
            pred_test.append(pred_score)
            pred_node_score[i] = pred_score
            i+=1
            labels.append(1)

        pred_negative = []
        for (u, v) in self.embed.get_negative_edges():
            pred_score = self.embed.link_prob(u, v)
            pred_negative.append(pred_score)
            pred_node_score[i] = pred_score
            i+=1
            labels.append(0)

        sorted_idx = sorted(pred_node_score.items(), key=operator.itemgetter(1), reverse=True)
        neg_num = 0
        k = 5000
        for (idx, score) in sorted_idx[0:k]:
            if idx >= len(pred_test):
                neg_num+=1
        
        y = []
        for (idx, score) in sorted_idx:
            y.append(score)

        
        print("negative pred: %d, precision@%d: %f"%(neg_num, k, 1.0-neg_num*1.0/k))

        
        pred_labels = np.hstack([pred_test, pred_negative])
        true_labels = labels

        auc_score = roc_auc_score(true_labels, pred_labels)

        ap_score = average_precision_score(true_labels, pred_labels)

        median = np.median(pred_labels)
        index_pos = pred_labels > median
        index_neg = pred_labels <= median
        print("positive preds: %d, negative preds: %d"%(len(index_pos), len(index_neg)))
        pred_labels[index_pos] = 1
        pred_labels[index_neg] = 0
        acc_score = accuracy_score(true_labels, pred_labels)

        print("AUC: %f, AP: %f, Accuracy: %f" % (auc_score, ap_score, acc_score))
        res = {
            "AUC": auc_score,
            "AP": ap_score,
            "ACC": acc_score
        }
        gutil.saveResult("LP", self.data, self.algo, res)
