import numpy as np
from scipy.sparse import csc_matrix
from Executer import Executer 

class ConjugateGradientExecuter(Executer):
    def __init__(self, matrix: csc_matrix, tol: float, max_iter: int = 20000):
        super().__init__(matrix, tol, max_iter)
        self.r_old = None  
        self.p_old = None  

    def update_function(self):
        # Compute residual
        self.residual = self.b - self.matrix.dot(self.x)
        
        # Initialization of p(0)
        if self.r_old is None or self.p_old is None:
            p = self.residual.copy()
        else:
            # Compute beta
            beta = np.dot(self.residual, self.residual) / np.dot(self.r_old, self.r_old)
            # Update p(k) = r(k) + beta * p(k-1)
            p = self.residual + beta * self.p_old
        
        # Compute A * p
        Ap = self.matrix.dot(p)
        
        # Compute alpha = (r^T * r) / (p^T * A * p)
        alpha = np.dot(self.residual, self.residual) / np.dot(p, Ap)
        
        # Update x(k+1) = x(k) + alpha * p
        self.x = self.x + alpha * p
        
        # Store values for next iteration
        self.r_old = self.residual
        self.p_old = p
        
        return self.x, self.residual

