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
        print(f'Condition Number: {cond_number:.2e}')
        return cond_number
    except Exception as e:
        print(f"Error in computing condition number: {e}")
        return None

def plot_sparsity_pattern(matrix, title):
    """
    Function to plot the sparsity pattern of the matrix.
    """
    plt.figure(figsize=(8, 8))
    plt.spy(matrix, markersize=0.0000005)  # Decrease marker size for better granularity
    plt.title(title)
    plt.show()

def plot_value_distribution(matrix, title):
    """
    Function to plot the distribution of values in the matrix.
    """
    plt.figure(figsize=(3, 5))

    # Extract non-zero values and calculate zero counts
    zero_count = matrix.shape[0] * matrix.shape[1] - matrix.nnz
    non_zero_values = matrix.data

    # Plot histogram for non-zero values
    plt.hist(non_zero_values, bins=20, color='blue', alpha=0.7, label='Non-Zero Values')

    # Calculate the width of the bins to align zero count bar properly
    bin_width = (max(non_zero_values) - min(non_zero_values)) / 20 if non_zero_values.size > 0 else 1
    zero_bar_position = min(non_zero_values) - bin_width  # Position the zero value bar just before the first bin

    # Add a bar for zero values
    plt.bar(zero_bar_position, zero_count, width=bin_width, color='red', alpha=0.7, label='Zero Values')

    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Remove x-axis ticks and labels
    plt.xticks([])

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
            print(f"Analyzing matrix: {matrix_name}")

            # Calculate and print the condition number
            compute_condition_number(A)

            # Plot the sparsity pattern of the matrix
            plot_sparsity_pattern(A, f'Sparsity Pattern of Matrix {matrix_name}')

            # Plot the distribution of the non-zero values of the matrix
            plot_value_distribution(A, f'Distribution of Non-Zero Values in Matrix {matrix_name}')

if __name__ == "__main__":
    main()
