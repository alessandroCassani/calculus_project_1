import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from Executer import Executer  # Assuming Executer is defined in Executer.py

class GradientExecuter(Executer):
    def __init__(self, matrix: csc_matrix, tol: float, max_iter: int = 20000):
        super().__init__(matrix, tol, max_iter)
    
    
    def update_function(self):
        r = self.b - self.matrix.dot(self.x)  # Compute residual r(k) = b - A * x(k)
        alpha = np.dot(r, r) / np.dot(r, self.matrix.dot(r)) # Compute alpha = (r^T * r) / (r^T * y)
        xk = self.x + (alpha * r)  # Update x(k) = x(k) + alpha * r
    
        return xk, r
