from scipy.sparse import csc_matrix
from Executer import Executer
import numpy as np

class JacobiExecuter(Executer):
    def __init__(self, matrix: csc_matrix, tol: float, iterations: int = 20000):
        super().__init__(matrix, tol, iterations)
        self.diagonal_inv = 1.0 / matrix.diagonal()  # Inverse of the diagonal elements
    
    def update_function(self):
        r = self.b - self.matrix.dot(self.x) #r(k) = b - A * x(k)
        return self.x + self.diagonal_inv * r, r  #x(k+1) = x(k) + P^-1 * r(k), 
