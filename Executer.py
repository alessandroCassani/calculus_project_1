from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csc_matrix
import time
from Utility import Utility

class Executer(ABC):
    def __init__(self, matrix: csc_matrix, tol: float, iterations: int = 20000):
        self.matrix = matrix
        self.x = np.zeros(matrix.shape[0])  
        self.tol = tol
        self.iterations = iterations
        self.b = np.ones(matrix.shape[0])  
        self.counter = 0
        Utility.check_sizes(self.matrix, self.b)

    def methodExecution(self):
        start = time.time()
        for self.counter in range(self.iterations):
            self.x = self.update_function()  # Update x based on solver-specific logic
            self.counter += 1
            residual_norm = np.linalg.norm(self.b - self.matrix.dot(self.x))
            if residual_norm < self.tol:
                print(f'{self.__class__.__name__}: Converged with residual norm {residual_norm:.6f}')
                break
            if self.counter > self.iterations:
                print('maximum iteration number exceeded')
                break
        else:
            print(f'{self.__class__.__name__}: Did not converge within {self.iterations} iterations')
        
        end = time.time() - start
        self.execution_timer = end

    @abstractmethod
    def update_function(self):
        pass
