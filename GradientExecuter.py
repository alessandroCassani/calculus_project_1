import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from Executer import Executer  # Assuming Executer is defined in Executer.py

class GradientExecuter(Executer):
    def __init__(self, matrix: csc_matrix, tol: float, max_iter: int = 20000):
        super().__init__(matrix, tol, max_iter)
    
    
    def update_function(self):
        r = self.b - self.matrix.dot(self.x)  # Compute residual r(k) = b - A * x(k)
        Ar = self.matrix.dot(r)  # Compute A * r
        r_dot_r = np.dot(r, r)  # Compute r^T * r
        alpha = r_dot_r / np.dot(r, Ar)  # Compute alpha = (r^T * r) / (r^T * A * r)
        self.x += alpha * r  # Update x(k) in-place: x(k+1) = x(k) + alpha * r
        
        return self.x, r
