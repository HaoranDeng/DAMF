#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
from scipy.sparse import csgraph
import scipy
from scipy.sparse.linalg import svds
import os
import sys
import math
import time
import h5py
import argparse

import utils
from matrices import SparseRowMatrix, OrthogonalMatrix

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="wiki")
parser.add_argument("--type", default="full")
parser.add_argument("--d", default=128, type=int)
parser.add_argument("--init", default=1000, type=int)
parser.add_argument("--store_USV", action="store_true")
args = parser.parse_args()

data_name = args.data
dim = args.d

t1, t2, t3, t4, t5 = 0., 0., 0., 0., 0.

# num_threads = "8"
# os.environ['OMP_NUM_THREADS'] = num_threads
# os.environ['OPENBLAS_NUM_THREADS'] = num_threads
# os.environ['MKL_NUM_THREADS'] = num_threads
# os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
# os.environ['NUMEXPR_NUM_THREADS'] = num_threads

import numpy as np

start = args.init
step = 1

class StreamNE:
    def __init__(self, A0, dim=128, max_rows=100000):
        start_time = time.perf_counter()
        print(f"Initialize network embedding... ", file=sys.stderr, end=" ", flush=True)
        
        self.num_node_ = A0.shape[0]
        
        U0, S0, V0 = scipy.sparse.linalg.svds(A0, dim)
        V0 = V0.T
        
        self.dim_ = dim
        self.U = OrthogonalMatrix(U0, max_rows)
        self.S = S0
        self.V = OrthogonalMatrix(V0, max_rows)
        print(f"Done. [{time.perf_counter() - start_time:.2f}s]", file=sys.stderr, flush=True)
        
    
    @staticmethod
    def add_row(C, U, S, V, dim):   
        # Calculate 2-norm of (I-VV^T)C
        global t1, t2, t3
        def FastNorm(V, C):
            tmp_C = SparseRowMatrix(C)
            E = utils.mul_OM_SRM(V, tmp_C)
            ret_1 = tmp_C.column_product(0, 0)
            ret_2 = np.dot(E[:, 0], E[:, 0])
            return np.array( [[np.sqrt(ret_1 - ret_2) ]] )
        
        dn, m = C.shape
    
        # Normalize (I-VV^T)C.T
        time_0 = time.perf_counter()
        R = FastNorm(V, C.T)
        time_1 = time.perf_counter()

        Mu = np.concatenate((np.diag(S), np.zeros((dim, dn))), axis=-1)
        Md = np.concatenate((C@V, R.T), axis=-1)
        M = np.concatenate((Mu, Md))

        F, s, G = np.linalg.svd(M)
        G = G.T
        F = F[:, :dim]
        G = G[:, :dim]
        S[:] = s[:dim]
        time_2 = time.perf_counter()

        # Update U
        U.column_transform(F[:dim])
        U.add_row(F[dim])

        # S = O
        if (R[0, 0] == 0):
            V.column_transform(G[:dim])
            return 
        # Update V
        E = -(C@V).T/R[0, 0]
        
        V.column_transform(G[:dim] + E @ G[dim:])
        tmp = SparseRowMatrix(C.T/R[0, 0]) 
        tmp.matmul(G[dim:])
        V += tmp
        time_3 = time.perf_counter()
        
        t1 += time_1 - time_0
        t2 += time_2 - time_1
        t3 += time_3 - time_2
 
    
    def add_node(self, edge_list_row, edge_list_col):
        col = np.array(edge_list_row)
        row = np.zeros_like(col)
        data = np.ones_like(col)
        
        C = scipy.sparse.coo_matrix((data, (row, col)), shape=(1, self.num_node_))
        C = C.tocsr()
        self.add_row(C, self.U, self.S, self.V, self.dim_)
        
        self.num_node_ += 1
        
        col = np.array(edge_list_col)  
        row = np.zeros_like(col)
        data = np.ones_like(col)
        
        C = scipy.sparse.coo_matrix((data, (row, col)), shape=(1, self.num_node_))
        C = C.tocsr()
        self.add_row(C, self.V, self.S, self.U, self.dim_)
        
    def get_embedding(self):
        t0 = time.perf_counter()
        X = scipy.sparse.diags(np.sqrt(self.S)).dot(self.U.T).T
        Y = scipy.sparse.diags(np.sqrt(self.S)).dot(self.V.T).T
        return X, Y


if __name__ == "__main__":
    print(args)

    data_type = "train" if args.type=="train" else "full"
    input_file_name = f"data/{data_name}/{data_type}.mat" 
    print(f"input_file_name: {input_file_name}")
    A = utils.load_adjacency_matrix(input_file_name)
    model = StreamNE(A[:start, :start], dim=dim//2, max_rows=A.shape[0])

    A_row = A.tocsr()
    A_col = A.tocsc()

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
        pbar.set_postfix(t1=t1, t0=time.perf_counter()-t0, t2=t2, t3=t3)

    X, Y = model.get_embedding()

    if args.store_USV:
        U = model.U.T.T
        V = model.V.T.T
        S = model.S
        np.save(f"matrix_factorization/DAMFUSV/{args.data}.U.npy", U)
        np.save(f"matrix_factorization/DAMFUSV/{args.data}.S.npy", S)
        np.save(f"matrix_factorization/DAMFUSV/{args.data}.V.npy", V)

    utils.save_embedding(data_name, "damf1", args.d, args.type, X, Y)
    print("Done.")
