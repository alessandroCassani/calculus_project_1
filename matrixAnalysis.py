import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse.linalg import norm
from Utility import Utility

def compute_condition_number(matrix):
    """
    Function to compute the condition number for sparse matrices.
    """
    try:
        cond_number = np.linalg.cond(matrix.toarray())  # Convert to dense array for condition number
        print(f'Numero di condizionamento: {cond_number:.2e}')
        return cond_number
    except Exception as e:
        print(f"Error in computing condition number: {e}")
        return None

def plot_sparsity_pattern(matrix, title):
    """
    Function to plot the sparsity pattern of the matrix.
    """
    plt.figure()
    plt.spy(matrix, markersize=1)
    plt.title(title)
    plt.show()

def plot_value_distribution(matrix, title):
    """
    Function to plot the distribution of values in the matrix.
    """
    plt.figure()
    plt.hist(matrix.data, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Valore')
    plt.ylabel('Frequenza')
    plt.show()

def main():
    """
    Main function to execute analysis and plot graphs for each matrix.
    """
    PATH = 'matrici'
    
    # Get list of matrix file paths
    matrix_files = Utility.get_matrix_paths(PATH)

    for i, matrix_file in enumerate(matrix_files):
        # Read each matrix
        A = Utility.read(matrix_file)
        if A is not None:
            matrix_name = matrix_file.split('/')[-1].split('.')[0]  # Extract the matrix name from the filename
            print(f"Analisi della matrice: {matrix_name}")

            # Calculate and print the condition number
            compute_condition_number(A)

            # Plot the sparsity pattern of the matrix
            plot_sparsity_pattern(A, f'Pattern di Sparsità della Matrice {matrix_name}')

            # Plot the distribution of the non-zero values of the matrix
            plot_value_distribution(A, f'Distribuzione dei Valori degli Elementi della Matrice {matrix_name}')

if __name__ == "__main__":
    main()
