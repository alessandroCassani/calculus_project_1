from Utility import Utility
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from memory_profiler import memory_usage
from JacobiExecuter import JacobiExecuter
from GaussSeidelExecuter import GaussSeidelExecuter
from GradientExecuter import GradientExecuter
from ConjugateGradientExecuter import ConjugateGradientExecuter
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns

PATH = 'matrici'

def run_matrix_solvers():
    matrix_files = Utility.get_matrix_paths(PATH)
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    max_iterations = 20000
    results = {}

    for matrix in matrix_files:
        A = Utility.read(matrix)
        
        if Utility.matrix_checks(A) is False:
            print('incorrect matrix')
            return
        
        if A is None:
            continue 
        
        matrix_results = {}
        for tol in tolerances:
            tol_results = {}

            solvers = [
                ('Jacobi', JacobiExecuter(A, tol, max_iterations)),
                #('Gauss-Seidel', GaussSeidelExecuter(A, tol, max_iterations)),
                #('Gradient Descent', GradientExecuter(A, tol, max_iterations)),
                ('Conjugate Gradient', ConjugateGradientExecuter(A, tol, max_iterations))
            ]
            
            for solver_name, solver in solvers:
                print(f'\nSolving {matrix} with tolerance {tol} using {solver_name}')
                
                start_time = time.time()
                counter, residual_norm = solver.method_execution()
                end_time = time.time()

                tol_results[solver_name] = {
                    'Iterations': counter,
                    'Residual': residual_norm,
                    'Time Usage (seconds)': end_time - start_time
                }

            matrix_results[tol] = tol_results
        results[matrix] = matrix_results

    # Write the results to CSV
    Utility.write_usage_csv('results1/usage_data1.csv', results)
    
    return results

def parse_data(results):
    parsed_data = []
    for matrix, tol_results in results.items():
        for tol, solver_results in tol_results.items():
            for solver, metrics in solver_results.items():
                parsed_data.append({
                    'Matrix': matrix,
                    'Solver': solver,
                    'Tolerance': tol,
                    'Iterations': metrics['Iterations'],
                    'Residual': metrics['Residual'],
                    'Time Usage (seconds)': metrics['Time Usage (seconds)']
                })
    return pd.DataFrame(parsed_data)

def plot_results(df):
    # Create the root window
    root = tk.Tk()
    root.title('Matrix Solver Results')
    root.geometry('1200x800')

    # Create a frame for the canvas and scrollbar
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas in the frame
    canvas = tk.Canvas(frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add a scrollbar to the canvas
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create another frame inside the canvas
    scrollable_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox('all'))
        if scrollable_frame.winfo_width() != canvas.winfo_width():
            canvas.itemconfigure('window', width=canvas.winfo_width())

    scrollable_frame.bind('<Configure>', on_configure)

    sns.set(style="whitegrid")

    # Get unique matrices
    matrices = df['Matrix'].unique()

    for idx, matrix in enumerate(matrices):
        matrix_df = df[df['Matrix'] == matrix]

        fig, axs = plt.subplots(3, 1, figsize=(12, 24))
        fig.suptitle(f'Results for Matrix: {matrix}', fontsize=16)

        # Sort the tolerances in descending order
        matrix_df = matrix_df.sort_values(by='Tolerance', ascending=False)

        # Plot time usage
        max_time = matrix_df['Time Usage (seconds)'].max()
        barplot = sns.barplot(x='Tolerance', y='Time Usage (seconds)', hue='Solver', data=matrix_df, ax=axs[0])
        axs[0].set_title('Time Usage by Solver and Tolerance')
        axs[0].set_ylabel('Time Usage (seconds)')
        axs[0].set_xlabel('Tolerance')
        axs[0].set_ylim(0, max_time * 1.1)  # Set ylim slightly above max time for better visualization

        for container in axs[0].containers:
            axs[0].bar_label(container)

        for patch in barplot.patches:
            patch.set_edgecolor(patch.get_facecolor())

        # Plot residual
        max_residual = matrix_df['Residual'].max()
        min_residual = matrix_df['Residual'].min()
        barplot = sns.barplot(x='Tolerance', y='Residual', hue='Solver', data=matrix_df, ax=axs[1])
        axs[1].set_title('Residual by Solver and Tolerance')
        axs[1].set_ylabel('Residual')
        axs[1].set_xlabel('Tolerance')
        axs[1].set_yscale('log')
        axs[1].set_ylim(min_residual * 0.1, max_residual * 1.1)
        axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1e}'))

        for container in axs[1].containers:
            axs[1].bar_label(container)

        for patch in barplot.patches:
            patch.set_edgecolor(patch.get_facecolor())

        # Plot iterations
        max_iterations = matrix_df['Iterations'].max()
        barplot = sns.barplot(x='Tolerance', y='Iterations', hue='Solver', data=matrix_df, ax=axs[2])
        axs[2].set_title('Iterations by Solver and Tolerance')
        axs[2].set_ylabel('Iterations')
        axs[2].set_xlabel('Tolerance')
        axs[2].set_ylim(0, max_iterations * 1.1)  # Set ylim slightly above max iterations for better visualization

        for container in axs[2].containers:
            axs[2].bar_label(container)

        for patch in barplot.patches:
            patch.set_edgecolor(patch.get_facecolor())

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Embed the plot into the Tkinter frame
        canvas_plot = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    root.mainloop()


if __name__ == "__main__":
    results = run_matrix_solvers()
    
    data = parse_data(results)
    plot_results(data)
