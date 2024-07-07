import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix

class utility:
    
    @staticmethod
    def is_symmetric(A: csc_matrix) -> bool:
        """
        Check if the matrix A is symmetric.
        
        Parameters:
        - A (csc_matrix): The sparse matrix to check.
        
        Returns:
        - bool: True if A is symmetric, False otherwise.
        """
        return (A != A.T).nnz == 0

    @staticmethod
    def is_positive_definite(A: csc_matrix) -> bool:
        """
        Check if the matrix A is positive definite.
        
        Parameters:
        - A (csc_matrix): The sparse matrix to check.
        
        Returns:
        - bool: True if A is positive definite, False otherwise.
        """
        try:
            # Try to compute Cholesky decomposition
            sp.linalg.cholesky(A)
            return True
        except sp.linalg.LinAlgError:
            return False

    @staticmethod
    def init_zero_vector(size: int) -> np.ndarray:
        """
        Initialize a zero vector of given size.
        
        Parameters:
        - size (int): The size of the vector.
        
        Returns:
        - np.ndarray: The initialized zero vector.
        """
        return np.zeros(size)

    @staticmethod
    def relative_residual_norm(A: csc_matrix, x: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the relative residual norm ||Ax - b|| / ||b||.
        
        Parameters:
        - A (csc_matrix): The sparse matrix.
        - x (np.ndarray): The current solution vector.
        - b (np.ndarray): The right-hand side vector.
        
        Returns:
        - float: The relative residual norm.
        """
        return np.linalg.norm(A @ x - b) / np.linalg.norm(b)


    @staticmethod
    def is_symmetric(A: csc_matrix) -> bool:
        """
        Check if the matrix A is symmetric.
        
        Parameters:
        - A (csc_matrix): The sparse matrix to check.
        
        Returns:
        - bool: True if A is symmetric, False otherwise.
        """
        return (A != A.T).nnz == 0
