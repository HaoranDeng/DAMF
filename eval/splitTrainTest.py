#########################################################################
# File Name: splitTrainTest.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Tue 13 Nov 2018 02:29:51 PM +08
#########################################################################
# !/usr/bin/env/ python

import argparse
import numpy as np
import os.path
import graph_util as gutil
import scipy.sparse
import h5py

parser = argparse.ArgumentParser(description='Process...')
parser.add_argument('--data', type=str, help='graph dataset name')
parser.add_argument('--ratio', type=float, default=0.3, help='test data ratio')
parser.add_argument('--action', type=str, default="split", help='action: "split" or "select" ')
args = parser.parse_args()

fgraph = '../data/%s/edgelist.txt' % args.data
fattr = '../data/%s/attr.txt' % args.data
ftraingraph = '../data/%s/edgelist.train.txt' % args.data
ftestgraph = '../data/%s/edgelist.test.txt' % args.data
fnegativegraph = '../data/%s/edgelist.negative.txt' % args.data
ftrainmat = '../data/%s/train.mat' % args.data


def save_adjacency_matrix(A, file_name):
    if not isinstance(A, scipy.sparse.csc_matrix):
        raise NotImplementedError
    A = A.astype(np.float64)
    ir = A.indices
    jc = A.indptr
    data = A.data
    with h5py.File(file_name, "w") as f:
        grp = f.create_group("A")
        grp.create_dataset("ir", data=ir)
        grp.create_dataset("jc", data=jc)
        grp.create_dataset("data", data=data)

if args.action == "select":
    n, m, directed = gutil.loadGraphAttr(fattr)
    print('graph attributes: n=%d, m=%d, directed=%r' % (n, m, directed))
    num_test_edge = int(m * args.ratio)
    print("selecting negative edges...")
    negative_edges = gutil.selectNegativeEdges(fgraph, n, num_test_edge, directed)

    print("writing %s" % fnegativegraph)
    with open(fnegativegraph, 'w') as fout:
        for (u, v) in negative_edges:
            fout.write(str(u) + " " + str(v) + "\n")
elif args.action == "split":
    n, m, directed = gutil.loadGraphAttr(fattr)
    print("graph attributes: n=%d, m=%d, directed=%r" % (n, m, directed))

    num_test_edge = int(m * args.ratio)
    train_edges, test_edges = gutil.splitDiGraphToTrainTest(fgraph, num_test_edge)
    print("train edges: %d, test edges: %d" % (len(train_edges), len(test_edges)))

    print("writing %s" % ftestgraph)
    with open(ftestgraph, 'w') as fout:
        for (s, t) in test_edges:
            fout.write(str(s) + " " + str(t) + "\n")

    print("writing %s" % ftraingraph)
    with open(ftraingraph, 'w') as fout:
        for (s, t) in train_edges:
            fout.write(str(s) + " " + str(t) + "\n")
    
    row, col = np.array(train_edges).T
    data = np.ones_like(row, dtype=np.float64)
    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float64)
    A += scipy.sparse.eye(A.shape[0])
    A[A>1] = 1

    A = A.tocsc()
    if directed==False:
        A = A + A.T
        for i in range(A.shape[0]):
            if A[i, i] >= 2:
                A[i, i] = 1
 

    print("writing %s" % ftrainmat)
    save_adjacency_matrix(A, ftrainmat)

else:
    raise NotImplementedError

print("Done.")
