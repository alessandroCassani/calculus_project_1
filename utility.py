import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.io import mmread
import os
import csv  
from scipy.linalg import cholesky, LinAlgError

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
        
    @staticmethod
    def get_matrix_paths(directory):
        matrici = []
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f) and f.endswith('.mtx'):
                matrici.append(f)
        return matrici
    
    @staticmethod
    def write_usage_csv(file_path, results):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Matrix', 'Solver', 'Tolerance', 'Iterations', 'Residual', 'Time Usage (seconds)'])
        
            for matrix, matrix_results in results.items():
                writer.writerow([])
                writer.writerow([f'                     MATRIX: {matrix.split(os.sep)[-1]}'])
                for tol, tol_results in matrix_results.items():
                    writer.writerow([])
                    writer.writerow([f'TOLERANCE : {tol}'])
                    
                    for solver, metrics in tol_results.items():
                        writer.writerow([
                            solver, 
                            'Iterations:', metrics['Iterations'], 
                            'Residual:', metrics['Residual'],
                            'Time Usage (seconds):', f"{metrics['Time Usage (seconds)']:.2f}"
                        ])
                writer.writerow(['-'*80]) 
                writer.writerow([])
     
    @staticmethod           
    def is_positive_definite(matrix: csc_matrix):
        """Check if a sparse matrix is positive definite."""
        try:
            # Attempt Cholesky decomposition
            _ = cholesky(matrix.toarray(), lower=True)
            return True
        except LinAlgError:
            return False

    @staticmethod
    def is_symmetric(matrix: csc_matrix):
        """Check if a sparse matrix is symmetric."""
        if not matrix.shape[0] == matrix.shape[1]:
            return False  # Sparse matrix must be square to be symmetric
    
        # Check if the matrix is identical to its transpose
        transpose = matrix.transpose()
        return matrix.shape == transpose.shape and np.allclose(matrix.toarray(), transpose.toarray())
    
    @staticmethod
    def matrix_checks(matrix: csc_matrix):
        return Utility.is_positive_definite(matrix) and Utility.is_symmetric(matrix)
