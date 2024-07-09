import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, lil_matrix
from scipy.io import mmread
import os

class Utility:
    
    @staticmethod
    def check_sizes(A: csc_matrix, b: np.ndarray):
        if A.shape[0] != b.shape[0]:
            raise ValueError("Matrix A and vector b must have the same number of rows.")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")
    
    @staticmethod
    def read(filename):
        try:
            # Read the matrix market file
            matrix = mmread(filename).tocsc()
            print('Matrix read correctly! \n')
            return matrix
        except Exception as e:
            # Print an error message if the file cannot be opened
            print(f"Error: Could not open file {filename}. {e}")
            return None
        
    def build_matrix_paths_list(directory):
        matrici = []
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f) and f.endswith('.mtx'):
                matrici.append(f)
        return matrici
