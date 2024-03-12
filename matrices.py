import numpy as np
import scipy
import scipy.sparse

class SparseRowMatrix(object):
    def __init__(self, A=None, nzr=None, n=None, d=None):
        if A is not None:
            if isinstance(A, np.ndarray):
                self.n, self.d = A.shape
                self.nnzr = self.n
                self.A = A.copy()
                self.nzr = np.arange(self.n, dtype=np.int64)
            else:
                assert isinstance(A, scipy.sparse.csc_matrix)
                n, d = A.shape
                self.n, self.d = n, d
                self.nnzr = len(A.data) // d # number of non-zero rows
                self.A = A.data.copy().reshape(d, -1).T
                self.nzr = A.indices[:self.nnzr].copy()
        else:
            assert isinstance(nzr, np.ndarray)
            assert (d is not None)
            self.n, self.d = n, d
            self.A = np.zeros( (len(nzr), d), dtype=np.float64)
            self.nzr = nzr
            self.nnzr = len(nzr)

    def column_product(self, i, j):
        return np.dot(self.A[:, i], self.A[:, j])

    def column_subtract(self, i, j, factor):
        self.A[:, i] -= factor * self.A[:, j]

    def column_div(self, i, factor):
        self.A[:, i] /= factor

    def __rmatmul__(self, B):
        return B[self.nzr] @ self.A

    def col(self, i):
        return self.A[:, i]

    def matmul(self, B):
        self.A = self.A @ B
        self.d = B.shape[1]

    @property
    def shape(self):
        return (self.n, self.d)

    def to_sparse(self):
        indices = np.tile(self.nzr, self.d)
        indptr = np.arange(0, self.d*self.nnzr+1, self.nnzr)
        data = self.A.T.flatten()
        return scipy.sparse.csc_matrix((data, indices, indptr), shape=(self.n, self.d))
    
    def to_dense(self):
        return self.to_sparse().todense()


class OrthogonalMatrix:
    def __init__(self, U, max_rows=100000):
        self.res = 0
        self.cur_row, self.dim = U.shape
        self.K = np.eye(self.dim, dtype=np.float64)
        self.U = np.zeros((max_rows, self.dim), dtype=np.float64)
        self.U[:self.cur_row] = U
        self.tot = 0
        self.ep = None

    def set_ep(self, ep=None):
        self.ep = ep
        
    def __getitem__(self, index):
        return self.U[index] @ self.K
        
    def __rmatmul__(self, B):
        # return B@U@K
        ret =  B @ self.U[:self.cur_row] @ self.K
        return ret
    
    def column_transform(self, Kp):
        self.K = self.K @ Kp
        if np.linalg.cond(self.K) > 5 and False:
            self.U[:self.cur_row] = self.U[:self.cur_row] @ self.K
            if self.ep is not None:
                self.ep.rotate_space(self.K.astype(np.float32))
            self.K = np.eye(self.dim)
            self.tot += 1
    def add_row(self, B=None):
        if B is not None:
            B = B @ np.linalg.inv(self.K)
            self.U[self.cur_row] = B
        if self.ep is not None:
            self.ep.add_node_signal(self.cur_row, self.U[self.cur_row].astype(np.float32))
        self.cur_row += 1
        
    def __iadd__(self, B):
        assert isinstance(B, SparseRowMatrix)
        inv_K = np.asarray(np.linalg.inv(self.K))
        B.matmul(inv_K)
        for i, j in enumerate(B.nzr):
            self.U[j] += B.A[i]
            if self.ep is not None:
                self.ep.add_node_signal(j.item(), B.A[i].astype(np.float32))
        return self
    
    @property
    def shape(self):
        return (self.cur_row, self.dim)
    
    @property
    def T(self):
        return (self.U[:self.cur_row]@self.K).T

    @property
    def data(self):
        return (self.U[:self.cur_row]@self.K)

    def __str__(self):
        return str(self.U[:self.cur_row]@self.K)
    
    def test(self):
        tmp = self.U[:self.cur_row]@self.K
        return np.linalg.norm(tmp.T@tmp-np.eye(self.dim), ord=2)
