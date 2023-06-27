import os
import h5py
import scipy.sparse
import numpy as np
from matrices import OrthogonalMatrix, SparseRowMatrix

def load_adjacency_matrix(file, variable_name="network"):
    M = h5py.File(file, "r")['A']
    data, ir, jc = M['data'], M['ir'], M['jc']
    M = scipy.sparse.csc_matrix((data, ir, jc))
    return M

def save_adjacency_matrix(A, file_name):
    print(f"Output adjacency matrix file: {file_name}")
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

def save_embedding(data, algo, dim, type, X, Y):
    if not os.path.exists("embds"):
        os.mkdir("embds")
    if not os.path.exists(f"embds/{data}"):
        os.mkdir(f"embds/{data}")

    train_suffix = ".train" if type=="train" else ""

    x_file = f"embds/{data}/{algo}.{dim}{train_suffix}.bin.src"
    y_file = f"embds/{data}/{algo}.{dim}{train_suffix}.bin.tgt"
    print(f"embedding_file_x: {x_file}")
    print(f"embedding_file_y: {y_file}")

    X.tofile(x_file)
    Y.tofile(y_file)

# Return A.T @ B
def mul_OM_SRM(A:OrthogonalMatrix, B:SparseRowMatrix) -> np.ndarray:
    return A.K.T @ A.U[B.nzr].T @ B.A