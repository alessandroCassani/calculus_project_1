import numpy as np
from scipy.sparse import csc_matrix, tril
from scipy.sparse.linalg import splu
from Executer import Executer

class GaussSeidelExecuter(Executer):
    def __init__(self, matrix: csc_matrix, tol: float, max_iter: int = 20000):
        super().__init__(matrix, tol, max_iter)
        self.triang_inf = tril(matrix).tocsc()  # Sparse lower triangular matrix
        self.LU = splu(self.triang_inf)  # Precompute LU decomposition for the lower triangular part

    def forward_substitution(self, L: csc_matrix, b: np.ndarray):
        """Perform forward substitution to solve Ly = b for a lower triangular matrix L."""
        return self.LU.solve(b)  # Use LU solver for optimized forward substitution
    
    def update_function(self):
        r = self.b - self.matrix.dot(self.x)  # Compute residual r(k) = b - A * x(k)
        y = self.forward_substitution(self.triang_inf, r)  # Solve P * y = r(k) using optimized forward substitution
        
        # Update x(k) = x(k) + y in-place
        self.x += y
        return self.x, r  # Return updated x and residual r
