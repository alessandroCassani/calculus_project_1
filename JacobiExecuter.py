from scipy.sparse import csc_matrix
from Executer import Executer
import numpy as np

class JacobiExecuter(Executer):
    def __init__(self, matrix: csc_matrix, tol: float, iterations: int = 20000):
        super().__init__(matrix, tol, iterations)
        self.diagonal_inv = 1.0 / matrix.diagonal()  # Inverse of the diagonal elements
    
    def update_function(self):
        r = self.b - self.matrix.dot(self.x)  # Compute residual r(k) = b - A * x(k)
        self.x += self.diagonal_inv * r  # Update x(k) in-place: x(k+1) = x(k) + D^-1 * r(k)
        return self.x, r
