#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import h5py
import scipy
import scipy.sparse
import scipy.sparse.linalg
import networkx as nx
from tqdm import tqdm
import argparse

start = 1000
D = 1


# In[8]:
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="wiki")
parser.add_argument("--type", default="full")
parser.add_argument("--d", default=128)
args = parser.parse_args()
data_name = args.data


def load_adjacency_matrix(file, variable_name="network"):
    M = h5py.File(file, "r")['A']
    data, ir, jc = M['data'], M['ir'], M['jc']
    M = scipy.sparse.csc_matrix((data, ir, jc))
    return M


A = load_adjacency_matrix(f"data/{data_name}/{args.type}.mat").tocsr()

# In[9]:




A_init = A[:start, :start]
# Initialize F
F = np.zeros((A.shape[0], 128), dtype=np.float64)
eigvals, eigvectors = scipy.sparse.linalg.eigsh(A_init, 128)
F[:A_init.shape[0], :] = eigvectors

# Initialize Graph
G = nx.Graph()
A_init = A_init.tocsr().tocoo()
row, col = A_init.row, A_init.col
for i in range(start):
    G.add_node(i)
for i in range(len(row)):
    if row[i] <= col[i]:
        G.add_edge(row[i], col[i])


# In[10]:


def InfluencedVerticesIdentification(G, D, v):
    k = 0
    Rk = set([v])
    ret = set()
    while k <= D:
        k = k+1
        Rk1 = set()
        for v in Rk:
            for u in G.neighbors(v):
                Puv = 1. / G.degree(u)
                r = np.random.choice(2, p=[1-Puv, Puv])
                if r == 1:
                    Rk1.add(u)
            ret.update(Rk1)
        Rk = Rk1.copy()
    return ret


# In[11]:


pbar = tqdm(range(1000, A.shape[0]))
# pbar = tqdm(range(1000, 1050))
for i in pbar:
    delta_A = A[i, :i]
    
    # Update graph.
    G.add_node(i)
    for u in delta_A.indices:
        G.add_edge(i, u)
        
    # Calculate representation for new vertex
    I = InfluencedVerticesIdentification(G, D, i)
    for u in I:
        F[i] += F[u] / len(I)
    
    if len(I) > 0:
        alpha = 1 - np.sqrt(1-1/len(I))
        for u in I:
            F[u] -= alpha * F[i]


# In[12]:

train_suffix = ".train" if args.type=="train" else ""
output_file = f"embds/{data_name}/la.128{train_suffix}.bin"
print(f"Embedding file: {output_file}")
F.tofile(output_file)
print("Done.")


