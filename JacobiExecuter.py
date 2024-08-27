from scipy.sparse import csc_matrix
from Executer import Executer
import numpy as np

class JacobiExecuter(Executer):
    def __init__(self, matrix: csc_matrix, tol: float, iterations: int):
        super().__init__(matrix, tol, iterations)
        
        # Precompute the inverse of the diagonal elements
        # Use matrix.diagonal() to directly get the diagonal elements as a 1D array
        diag_elements = matrix.diagonal()
        self.diagonal_inv = np.reciprocal(diag_elements, where=diag_elements!=0)  # Avoid division by zero
        
    def update_function(self):
        # Compute the residual r(k) = b - A * x(k)
        r = self.b - self.matrix.dot(self.x)
        
        # Update x(k+1) = x(k) + D^-1 * r(k) (in-place update)
        np.multiply(self.diagonal_inv, r, out=r)  # r becomes D^-1 * r(k)
        self.x += r  # x(k+1) = x(k) + D^-1 * r(k)
        
        return self.x, r
