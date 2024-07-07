import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.io import mmread

class Utility:

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
    
    @staticmethod
    def read_sparse_matrix(file_path):
        """
        Reads a sparse matrix from a file and returns it as a scipy.sparse.csc_matrix.
    
        Parameters:
        file_path (str): The path to the file containing the sparse matrix.
    
        Returns:
        csc_matrix: The sparse matrix in CSC (Compressed Sparse Column) format.
        """
        # Initialize empty lists to hold the row indices, column indices, and values
        rows_index = []
        cols_index = []
        values = []

        try:
            # Open the file for reading
            with open(file_path, 'r') as file:
                # Read all lines from the file
                lines = file.readlines()
                # Iterate over the lines starting from the second line (skip the first line)
                for line in lines[1:]:
                    # Skip lines that start with '%'
                    if line.startswith('%'):
                        continue
                    # Split the line into components (row, column, value)
                    row, col, value = line.split()
                    # Append the parsed row index (convert to zero-based index)
                    rows_index.append(int(row) - 1)
                    # Append the parsed column index (convert to zero-based index)
                    cols_index.append(int(col) - 1)
                    # Append the parsed value
                    values.append(float(value))
        except Exception as e:
            # Print an error message if the file cannot be opened
            print(f"Error: Could not open file. {e}")
            return None

        # Convert lists to numpy arrays
        rows_index = np.array(rows_index, dtype=np.uint32)
        cols_index = np.array(cols_index, dtype=np.uint32)
        values = np.array(values, dtype=np.float64)

        # Create the sparse matrix in CSC format using the row indices, column indices, and values
        sparse_matrix = csc_matrix((values, (rows_index, cols_index)))

        return sparse_matrix  # Return the created sparse matrix

def cholesky_decomposition_sparse(A):
    """
    Perform Cholesky decomposition for a sparse matrix A in CSR format.

    Parameters:
    A : scipy.sparse.csr_matrix
        Input sparse matrix to decompose.

    Returns:
    L : scipy.sparse.csr_matrix
        Lower triangular matrix L such that A = L * L^T.
    """
    # Check if A is a CSR matrix
    if not isinstance(A, csr_matrix):
        raise ValueError("Input matrix A must be a scipy.sparse.csr_matrix")
    
    # Get the size of the matrix
    n = A.shape[0]
    
    # Check if A is square
    if n != A.shape[1]:
        raise ValueError("Matrix A must be square")
    
    # Lists to store the sparse matrix entries
    rows_index = []
    cols_index = []
    values = []

    # Dense matrix R to store intermediate calculations
    R = np.zeros((n, n), dtype=np.float64)
    
    # Copy of A to perform operations without modifying the original
    A_star = A.copy()

    # Perform Cholesky decomposition
    for k in range(n):
        # Calculate the diagonal element of L
        rkk = np.sqrt(A_star[k, k])
        # Store the diagonal element in L
        rows_index.append(k)
        cols_index.append(k)
        values.append(rkk)

        # Calculate and store the elements below the diagonal in L
        for j in range(k+1, n):
            rkj = A_star[k, j] / rkk
            rows_index.append(k)
            cols_index.append(j)
            values.append(rkj)
        
        # Update A_star for the next iteration
        rho = 1.0 / A_star[k, k]
        for j in range(k+1, n):
            for i in range(k+1, n):
                A_star[i, j] = A_star[i, j] - rho * A_star[i, k] * A_star[k, j]
    
    # Create the sparse matrix L from the lists of entries
    L = csr_matrix((values, (rows_index, cols_index)), shape=(n, n))
    return L
