import numpy as np

def gaussian_elimination(A, b):
    
    """
    Perform Gaussian Elimination to solve the linear system Ax = b.

    Parameters:
    A (numpy.ndarray): Coefficient matrix A of shape (n, n)
    b (numpy.ndarray): Right-hand side vector b of shape (n,)

    Returns:
    numpy.ndarray: Solution vector x of shape (n,)
    """

    n = len(b)
    U = A.copy()
    b_prime = b.copy()
    
    for k in range(n - 1):
        # Find the pivot row
        max_index = np.argmax(np.abs(U[k:, k])) + k
        
        # Swap rows if necessary
        if k != max_index:
            U[[k, max_index]] = U[[max_index, k]]
            b_prime[[k, max_index]] = b_prime[[max_index, k]]
        
        # Eliminate entries below the pivot
        for i in range(k + 1, n):
            if U[k, k] == 0:
                raise ValueError("Matrix is singular and cannot be solved.")
            m = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - m * U[k, k:]
            b_prime[i] = b_prime[i] - m * b_prime[k]
    
    # Back substitution to solve Ux = b'
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Matrix is singular and cannot be solved.")
        x[i] = (b_prime[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    
    return x

# Example usage:
A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

x = gaussian_elimination(A, b)
print("Solution:", x)
