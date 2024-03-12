#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

import time
import numpy as np
import argparse
import sys
import warnings

import graph_util as gutil
from algo_list import algo_list_X, algo_list_XY



warnings.filterwarnings("ignore")

clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='wiki', help='Data name.')
parser.add_argument('--algo', default='damf1', help='Algorithm name.')
parser.add_argument('--d', default=128, type=int, help="dimension of embedding.")
args = parser.parse_args()


# In[2]:


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)


class Classifier(object):
    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=False)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        print(Y.shape)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ['micro', 'macro']
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)

        return results

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed):
        state = np.random.get_state()
        training_size = int(train_precent * len(X))
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)

def read_node_label(filename):
    fin = open(filename, "r")
    X = []
    Y = []
    while True:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X.append(int(vec[0]))
        Y.append(vec[1:])
    fin.close()
    return X, Y

def load_sup_info(filename):
    fin = open(filename, "r")
    X_sup = []
    while True:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X_sup.append(int(vec[0]))
    return X_sup


# In[3]:
# label_suffix = "" if args.noshuffle else ".shuffled" 
label_suffix = ""
label_file = '../data/' + args.data + '/labels' + label_suffix + '.txt'
X, Y = read_node_label(label_file)
print(len(X))

# In[4]:
suffix = ""
embed = gutil.Embedding(args.data, args.algo, args.d, suffix=suffix, full=True)
print(embed)


if args.algo in algo_list_X:
    # vectors = normalize(embed.X, norm='l2')
    vectors = embed.X
    # print(len(X))
    # vectors = vectors[X]
    # print(vectors.shape)
    # X = list(range(int(vectors.shape[0])))
    # print("1111111111")
    # print(len(X), len(Y))
elif args.algo in algo_list_XY:
    vectors1, vectors2 = embed.X, embed.Y
    # vectors1 = vectors1[X]
    # vectors2 = vectors2[X]
    vectors1 = normalize(vectors1, norm='l2')
    vectors2 = normalize(vectors2, norm='l2')
    vectors = np.concatenate((vectors1, vectors2), axis=1)
    # X = list(range(int(vectors.shape[0])))
    # print(len(X))
else:
    raise NotImplementedError

# In[ ]:

output_results = []
for ratio in clf_ratio:
    print('Training classifier using {:.2f}% nodes...'.format(ratio * 100))
    clf = Classifier(vectors=vectors, clf=LogisticRegression())
    results = []
    results_macro = []
    for i in range(5):
        tmp = clf.split_train_evaluate(X, Y, ratio, i)
        results.append(float(tmp['micro']))
        results_macro.append(float(tmp['macro']))
    print("Micro F1 score: {:.4f}".format(np.mean(results) * 100))
    print("Macro F1 score: {:.4f}".format(np.mean(results_macro) * 100))
    output_results.append(np.mean(results))

gutil.saveResult("NC", args.data, args.algo, output_results)

