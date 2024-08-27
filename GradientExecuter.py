import numpy as np
from scipy.sparse import csc_matrix
from Executer import Executer  # Assuming Executer is defined in Executer.py

class GradientExecuter(Executer):
    def __init__(self, matrix: csc_matrix, tol: float, max_iter: int):
        super().__init__(matrix, tol, max_iter)

    def update_function(self):
        """
        Perform a single iteration of the gradient descent method to update the solution vector.
        """
        # Compute residual r(k) = b - A * x(k)
        r = self.b - self.matrix.dot(self.x)
        
        # Compute A * r
        Ar = self.matrix.dot(r)
        
        # Compute r^T * r (squared norm of the residual)
        r_dot_r = np.dot(r, r)
        
        # Compute alpha = (r^T * r) / (r^T * A * r)
        r_Ar_dot = np.dot(r, Ar)  # r^T * (A * r)
        if r_Ar_dot == 0:  # Prevent division by zero
            alpha = 0
        else:
            alpha = r_dot_r / r_Ar_dot
        
        # Update x(k+1) = x(k) + alpha * r (in-place update)
        self.x += alpha * r
        
        # Return the updated solution and the residual for the next iteration
        return self.x, r
