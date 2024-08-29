import numpy as np
from scipy.sparse import csc_matrix, tril
from scipy.sparse.linalg import spsolve
from Executer import Executer

class GaussSeidelExecuter(Executer):
    def __init__(self, matrix: csc_matrix, tol: float, max_iter: int = 20000):
        super().__init__(matrix, tol, max_iter)
        self.triang_inf = tril(matrix).tocsc()  # Sparse lower triangular matrix
    
    def update_function(self):
        r = self.b - self.matrix.dot(self.x)  # Compute residual r(k) = b - A * x(k)
        y = spsolve(self.triang_inf, r) 
        self.x += y
        return self.x, r  
