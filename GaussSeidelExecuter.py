import numpy as np
from scipy.sparse import csc_matrix, tril
from Executer import Executer

class GaussSeidelExecuter(Executer):
    def __init__(self, matrix: csc_matrix, tol: float, max_iter: int = 20000):
        super().__init__(matrix, tol, max_iter)
        self.triang_inf = tril(matrix).tocsc()  
    
    def forward_substitution(self, L: csc_matrix, b: np.ndarray):
        """ Perform forward substitution to solve Ly = b for a lower triangular matrix L. """
        n = L.shape[0]
        y = np.zeros(n)
        
        for i in range(n):
            y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
        
        return y
    
    def update_function(self):
        r = self.b - self.matrix.dot(self.x)  # Compute residual r(k) = b - A * x(k)
        y = self.forward_substitution(self.triang_inf, r)  # Solve P * y = r(k) using forward substitution
        
        # self.x +y scope: update x(k) = x(k) + y
        return self.x + y, r
    