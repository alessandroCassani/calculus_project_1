import numpy as np
from scipy.sparse import csc_matrix
import time
from abc import ABC, abstractmethod

class Executer(ABC):
    def __init__(self, matrix: csc_matrix, tol: float, iterations: int = 20000):
        self.matrix = matrix
        self.x = np.zeros(matrix.shape[0])  # Correct shape initialization
        self.tol = tol
        self.iterations = iterations
        self.b = np.ones(matrix.shape[0])  # Correct shape initialization
        self.counter = 0
        self.b = np.array(self.b, dtype=np.float64)
        self.x = np.array(self.x, dtype=np.float64)
        
        # Assume self.diagonal_inv is precomputed and correctly initialized
        self.diagonal_inv = 1.0 / self.matrix.diagonal()

    def methodExecution(self):
        start = time.time()
        for self.counter in range(self.iterations):
            #print(f"Iteration {self.counter}: x shape = {self.x.shape}")
            self.x, residual_vector = self.update_function()  # Update x and get residual
            #print(f"Iteration {self.counter}: updated x shape = {self.x.shape}")
            self.counter += 1
            residual_norm = np.linalg.norm(residual_vector)
            if residual_norm < self.tol:
                print(f'{self.__class__.__name__}: Converged with residual norm {residual_norm}')
                break
            if self.counter >= self.iterations:
                print('Maximum iteration number exceeded')
                break
        else:
            print(f'{self.__class__.__name__}: Did not converge within {self.iterations} iterations')
        
        end = time.time() - start
        self.execution_timer = end

    @abstractmethod
    def update_function(self):
        pass
