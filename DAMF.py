#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm
import scipy
import scipy.sparse
from scipy.sparse.linalg import svds
import os
import sys
import math
import time
import h5py
import argparse
from typing import List
import utils

from matrices import SparseRowMatrix, OrthogonalMatrix
from models import DAMF
from utils import load_adjacency_matrix, mul_OM_SRM
import embedding_propagation as ep


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="flickr")
parser.add_argument("--type", default="full")
parser.add_argument("--d", default=128)
parser.add_argument("--epsilon", type=float, default=1e-5)
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--init", type=int, default=1000)
args = parser.parse_args()

start = args.init
step = 1


# In[2]:


A = load_adjacency_matrix(f"data/{args.data}/full.mat")


# In[3]:


class DPRMF(DAMF):
    def __init__(self, A0, dim=128, max_rows=100000):
        start_time = time.perf_counter()
        print(f"Initialize network embedding (SVD)... ", file=sys.stderr, end=" ", flush=True)
        self.num_node_ = A0.shape[0]
        
        U0, S0, V0 = scipy.sparse.linalg.svds(A0, dim)
        V0 = V0.T
        self.U = OrthogonalMatrix(U0, max_rows)
        self.S = S0
        self.V = OrthogonalMatrix(V0, max_rows)
        print(f"Done. [{time.perf_counter() - start_time:.2f}s]", file=sys.stderr, flush=True)
        
        start_time = time.perf_counter()
        print(f"Initialize network embedding (PPR)... ", file=sys.stderr, end=" ", flush=True)
        ep.init()
        # Swap row and column since PPR propagation in the inverse direction
        edge_index = np.vstack((A0.tocoo().col, A0.tocoo().row)).astype(np.int64)

        signal_init = self.U.U.astype(np.float32)
        self.ep = ep.StreamPPR(signal_init, edge_index, float(args.alpha), float(args.epsilon))
        self.ep.basic_propagation()
        self.dim_ = dim

        self.U.set_ep(self.ep)
        print(f"Done. [{time.perf_counter() - start_time:.2f}s]", file=sys.stderr, flush=True)
        
        
    def add_node(self, edge_list_row, edge_list_col, ):
        for i in edge_list_row:
            self.ep.add_edge(self.num_node_, i.item())
        for i in edge_list_col:
            self.ep.add_edge(i.item(), self.num_node_)
        super().add_node(edge_list_row, edge_list_col)
        self.ep.basic_propagation()
        return
    def get_embedding(self):
        X = self.ep.get_embedding()
        X = X @ self.U.K @ scipy.sparse.diags(np.sqrt(self.S))
        X = X * (1.0-args.alpha) * args.alpha
        Y = scipy.sparse.diags(np.sqrt(self.S)).dot(self.V.T).T
        return X, Y


# In[ ]:


input_file = f"data/{args.data}/{args.type}.mat"
print(f"Input file: {input_file}", flush=True)
A = load_adjacency_matrix(input_file)

model = DPRMF(A[:start, :start], dim=args.d//2, max_rows=A.shape[0])
A_row = A.tocsr()
A_col = A.tocsc()

print(args, flush=True)

t0 = time.perf_counter()
cur_row = start
num_steps = math.ceil( (A.shape[0]-cur_row) / step )
len_U, len_V = cur_row, cur_row
pbar = tqdm(range(num_steps))
for i in pbar:
    last_row = cur_row
    cur_row = min(cur_row+step, A.shape[0])
    C_row = A_row[last_row: cur_row, :last_row]
    C_col = A_col[:cur_row, last_row: cur_row]

    model.add_node(C_row.indices, C_col.indices)
    pbar.set_postfix(t0=time.perf_counter()-t0)


# In[ ]:


X, Y = model.get_embedding()

utils.save_embedding(args.data, "damf", args.d, args.type, X, Y)
print("Done.")
