import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, lil_matrix
from scipy.io import mmread
import os
import csv  

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
    
    @staticmethod
    def write_usage_csv(file_path, data):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Matrix', 'Solver', 'Tolerance', 'Memory Usage (MB)', 'Time Usage (seconds)'])
            writer.writeheader()
            for entry in data:
                writer.writerow(entry)
