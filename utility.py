import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.io import mmread

def read_sparse_matrix(file_path: str) -> csc_matrix:
    """
    Reads a sparse matrix from a Matrix Market file (.mtx) and returns it as a csc_matrix.
    """
    return mmread(file_path).tocsc()

def init_zero_vector(size: int) -> np.ndarray:
    """
    Initializes a zero vector of the given size.
    """
    return np.zeros(size)

def relative_residual_norm(A: csc_matrix, x: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the relative residual norm ||Ax - b|| / ||b||.
    """
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)
