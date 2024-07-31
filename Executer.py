import numpy as np
from scipy.sparse import csc_matrix
import time
from abc import ABC, abstractmethod

class Executer(ABC):
    def __init__(self, matrix: csc_matrix, tol: float, iterations: int = 20000):
        self.matrix = matrix
        self.x = np.zeros(matrix.shape[0])  # Correct shape initialization
        self.x_optimal = np.ones(matrix.shape[0])
        self.tol = tol
        self.iterations = iterations
        self.b = np.ones(matrix.shape[0])  # Correct shape initialization
        self.counter = 0
        self.b = np.array(self.b, dtype=np.float64)
        self.x = np.array(self.x, dtype=np.float64)
        self.diagonal_inv = 1.0 / self.matrix.diagonal()

    def method_execution(self):
        residual_norm = np.inf  
        list_of_residual = []

        for self.counter in range(self.iterations):
            self.x, residual_vector = self.update_function()
            residual_norm = np.linalg.norm(residual_vector) / np.linalg.norm(self.b)
            list_of_residual.append(residual_norm)  # Corrected to store residual norms
            if residual_norm < self.tol:
                print(f'{self.__class__.__name__}: Converged with residual norm {residual_norm}')
                break

        return self.counter, residual_norm, list_of_residual

    @abstractmethod
    def update_function(self):
        pass
    
    def get_error(self):
         return np.divide(np.linalg.norm(self.x - self.x_optimal), np.linalg.norm(self.x))