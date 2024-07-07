import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvals
from PIL import Image

class Utils:
    
    @staticmethod
    def read_sparse_matrix(file_path: str) -> csc_matrix:
        """
        Reads a sparse matrix from a file and returns it as a csc_matrix.

        Parameters:
        - file_path (str): The path to the file containing the sparse matrix.

        Returns:
        - csc_matrix: The sparse matrix read from the file.
        """
        rows_index = []
        cols_index = []
        values = []

        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith('%'):  # Skip comment lines
                        continue
                    row, col, value = map(float, line.split())
                    rows_index.append(int(row) - 1)
                    cols_index.append(int(col) - 1)
                    values.append(value)
        except Exception as e:
            print("Error: Could not open file.")
            raise e

        return csc_matrix((values, (rows_index, cols_index)), dtype=np.float64)

    @staticmethod
    def PartialPivot(A: csc_matrix) -> int:
        """
        Finds the row index with the largest absolute value in the first column of the matrix A.

        Parameters:
        - A (csc_matrix): The sparse matrix.

        Returns:
        - int: The row index with the largest absolute value in the first column.
        """
        return abs(A[:, 0]).argmax()

    @staticmethod
    def TotalPivot(A: csc_matrix):
        """
        Finds the indices of the row and column with the largest absolute value in the matrix A.

        Parameters:
        - A (csc_matrix): The sparse matrix.

        Returns:
        - (int, int): The row and column indices with the largest absolute value.
        """
        max_index = abs(A).argmax()
        return np.unravel_index(max_index, A.shape)

    @staticmethod
    def swapRow(A: csc_matrix, i: int, j: int):
        """
        Swaps rows i and j in the matrix A.

        Parameters:
        - A (csc_matrix): The sparse matrix.
        - i, j (int): Indices of the rows to swap.

        Raises:
        - IndexError: If i or j is out of bounds.
        """
        if i >= A.shape[0] or j >= A.shape[0] or i < 0 or j < 0:
            raise IndexError("Index out of bounds")

        A = A.tocsr()
        A[[i, j], :] = A[[j, i], :]
        return A.tocsc()

    @staticmethod
    def swapColumn(A: csc_matrix, i: int, j: int):
        """
        Swaps columns i and j in the matrix A.

        Parameters:
        - A (csc_matrix): The sparse matrix.
        - i, j (int): Indices of the columns to swap.

        Raises:
        - IndexError: If i or j is out of bounds.
        """
        if i >= A.shape[1] or j >= A.shape[1] or i < 0 or j < 0:
            raise IndexError("Index out of bounds")

        A = A.tocsc()
        A[:, [i, j]] = A[:, [j, i]]
        return A

    @staticmethod
    def swapVectorPosition(x: np.ndarray, i: int, j: int):
        """
        Swaps elements i and j in the vector x.

        Parameters:
        - x (np.ndarray): The vector.
        - i, j (int): Indices of the elements to swap.

        Raises:
        - IndexError: If i or j is out of bounds.
        """
        if i >= len(x) or j >= len(x) or i < 0 or j < 0:
            raise IndexError("Index out of bounds")

        x[i], x[j] = x[j], x[i]

    @staticmethod
    def check_sizes(A: csc_matrix, b: np.ndarray):
        """
        Checks if the matrix A and vector b have compatible sizes.

        Parameters:
        - A (csc_matrix): The sparse matrix.
        - b (np.ndarray): The vector.

        Raises:
        - ValueError: If A and b do not have compatible sizes.
        """
        if A.shape[0] != b.shape[0] or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A and vector b must have the same number of rows")

    @staticmethod
    def optimal_alpha_richardson(P: csc_matrix) -> float:
        """
        Computes the optimal relaxation parameter alpha for the Richardson method.

        Parameters:
        - P (csc_matrix): The preconditioner matrix.

        Returns:
        - float: The optimal relaxation parameter alpha.
        """
        P_dense = P.toarray()  # Convert to dense array for eigenvalue computation
        if P_dense.shape[0] > 100:
            max_eg = np.max(eigs(P_dense, k=1, which='LM', return_eigenvectors=False).real)
            min_eg = np.min(eigs(P_dense, k=1, which='SM', return_eigenvectors=False).real)
        else:
            eigs_vals = eigvals(P_dense)
            max_eg = np.max(eigs_vals).real
            min_eg = np.min(eigs_vals).real

        return 2 / (max_eg + min_eg)

    @staticmethod
    def is_diagonally_dominant(A: np.ndarray) -> bool:
        """
        Checks if the matrix A is diagonally dominant.

        Parameters:
        - A (np.ndarray): The matrix.

        Returns:
        - bool: True if the matrix is diagonally dominant, False otherwise.
        """
        for i in range(A.shape[0]):
            if np.abs(A[i, i]) < np.sum(np.abs(A[i])) - np.abs(A[i, i]):
                return False
        return True

    @staticmethod
    def gen_random_matrix(rows: int, cols: int) -> np.ndarray:
        """
        Generates a random matrix with specified dimensions.

        Parameters:
        - rows, cols (int): The number of rows and columns.

        Returns:
        - np.ndarray: The generated random matrix.
        """
        np.random.seed(0)  # For reproducibility
        return np.random.rand(rows, cols)

    @staticmethod
    def GenBmpImage(row: int, col: int) -> np.ndarray:
        """
        Generates a random grayscale image.

        Parameters:
        - row, col (int): The dimensions of the image.

        Returns:
        - np.ndarray: The generated random grayscale image.
        """
        return np.random.rand(row, col) * 255

    @staticmethod
    def LoadBmpImage(path: str) -> np.ndarray:
        """
        Loads a BMP image from the specified path and converts it to a numpy array.

        Parameters:
        - path (str): The path to the BMP image.

        Returns:
        - np.ndarray: The loaded image as a numpy array.
        """
        img = Image.open(path).convert('L')  # Convert to grayscale
        return np.array(img, dtype=np.float32)

    @staticmethod
    def SaveBmpImage(img: np.ndarray, path: str):
        """
        Saves a numpy array as a BMP image to the specified path.

        Parameters:
        - img (np.ndarray): The image to save.
        - path (str): The path to save the image.
        """
        img = Image.fromarray(img.astype('uint8'))  # Convert to uint8
        img.save(path)
