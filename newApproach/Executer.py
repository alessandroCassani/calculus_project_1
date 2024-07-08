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
            self.x = self.update_function()
            self.counter += 1
            if np.linalg.norm(self.b - self.matrix.dot(self.x)) < self.tol:
                print('The method does not converge')
                break
            if self.counter > self.iterations:
                print('maximum iteration number exceeded')
                break
        end = time.time() - start
        self.execution_timer = end

    @abstractmethod
    def update_function(self):
        pass
