import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, lil_matrix
from scipy.io import mmread

class Utility:
    
    @staticmethod
    def check_sizes(A: csc_matrix, b: np.ndarray):
        if A.shape[0] != b.shape[0]:
            raise ValueError("Matrix A and vector b must have the same number of rows.")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")
    
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
        
        print('Matrix read correctly! \n')

        return sparse_matrix  # Return the created sparse matrix


def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition for a sparse matrix A in CSC format.

    Parameters:
    A : scipy.sparse.csc_matrix
        Input sparse matrix to decompose.

    Returns:
    L : scipy.sparse.csc_matrix
        Lower triangular matrix L such that A = L * L^T.
    """
    # Check if A is a CSC matrix
    if not isinstance(A, csc_matrix):
        raise ValueError("Input matrix A must be a scipy.sparse.csc_matrix")
    
    # Get the size of the matrix
    n = A.shape[0]
    
    # Check if A is square
    if n != A.shape[1]:
        raise ValueError("Matrix A must be square")
    
    # Create an empty L matrix in LIL format
    L = lil_matrix((n, n), dtype=np.float64)

    # Copy of A to perform operations without modifying the original
    A_star = A.copy().tolil()

    print('Performing cholesky decomposition... \n')
    # Perform Cholesky decomposition
    for k in range(n):
        # Calculate the diagonal element of L
        rkk = np.sqrt(A_star[k, k])
        # Store the diagonal element in L
        L[k, k] = rkk

        # Calculate and store the elements below the diagonal in L
        for i in range(k+1, n):
            L[i, k] = A_star[i, k] / rkk
        
        # Update A_star for the next iteration
        rho = 1.0 / A_star[k, k]
        for j in range(k+1, n):
            for i in range(k+1, n):
                A_star[i, j] = A_star[i, j] - rho * A_star[i, k] * A_star[j, k]
    
    # Convert L to CSC format
    return L.tocsc()

