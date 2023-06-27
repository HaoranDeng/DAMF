import time
import numpy as np
import sys
import scipy
import scipy.sparse
from scipy.sparse.linalg import svds

from matrices import *
from utils import mul_OM_SRM
import embedding_propagation as ep


class DAMF:
    def __init__(self, A0, dim=128, max_rows=100000):
        start_time = time.perf_counter()
        print(f"Initialize network embedding... ", file=sys.stderr, end=" ", flush=True)
        
        self.num_node_ = A0.shape[0]
        A0 = A0.toarray()
        
        U0, S0, V0 = scipy.sparse.linalg.svds(A0, dim)
        V0 = V0.T
        assert np.linalg.matrix_rank(A0) >= dim
        
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
            E = mul_OM_SRM(V, tmp_C)
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
        
        # t1 += time_1 - time_0
        # t2 += time_2 - time_1
        # t3 += time_3 - time_2
 
    
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
        X = scipy.sparse.diags(np.sqrt(self.S)).dot(self.U.T).T
        Y = scipy.sparse.diags(np.sqrt(self.S)).dot(self.V.T).T
        return X, Y
