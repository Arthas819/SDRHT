"""
    Sinkhorn Distributionally Robust Hyperthesis Testing for Gaussian Mixture Dataset.
"""

import os
import time

os.environ["MPLBACKEND"] = "Agg"

from Models.StandardBaselines import *
from Models.WDRO import *
from Models.SinkhornDRO import *

from Sinkhorn.ICNN import *
from Sinkhorn.HyperICNN import *

from DataGenerators.Data_Gaussian_Mixture import *

import cvxpy as cp
import pandas as pd
from Plotting.Plotting import *

# Choose Dataset.
datasets = ['Gaussian_mixture']
# Choose Methods to solve the optimization problem.
methods = ['SDRO_H', 'SDRO_C', 'WDRO', 'GMM', 'SVM', '3NN']
# Total trials for different batch sizes
total_trials = 10
# Record computation time
computational_time = True
convergence_tol = 1e-3
max_sdro_epochs = 10
max_baseline_iter = 1000

# Scale of the dataset (number of samples, dimension of features, etc.).
N_train_set = [10]
N_test = 1000
# Dimension of features
dim_set = [4, 10, 30, 50, 70, 100]
# Testing batch size
observations_set = list(range(1, 11)) # [1, 2, ..., 10]

# Choose COPT to solve the WDRO model.
solver = cp.COPT
# Device for training networks in Sinkhorn DRO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Solve the optimization problem for each dataset and each baseline, and record the results in the table.
for data in datasets:
    # All results will be recorded in a table for comparison.
    output_path = f'Results/{data}'
    figure_output_path = f'Results/{data}'

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(figure_output_path, exist_ok=True)
    computation_time_summary = []

    for N_train in N_train_set:
        for d in dim_set:

            print(' -------N_train=', N_train, ', d=', d, '-----')
            file_name = f'all_results_{data}_sample={N_train}_d={d}.xlsx'
            figure_file_name = f'all_results_{data}_sample={N_train}_d={d}_error_rate.pdf'

            all_results = []
            computation_time_results = []

            for trial in range(total_trials):

                print(' ======= Trial: ', trial, '========')

                # Get data from the selected dataset, X1 belongs to P1 and X2 belongs to P2.
                X1, X2 = generate_Gaussian_mixture_data(data, N_train, d)

                # Normalization for X1 and X2
                mean_all = np.concatenate((X1, X2), axis=0).mean(axis=0)
                std_all = np.concatenate((X1, X2), axis=0).std(axis=0)
                X1 = (X1 - mean_all) / (std_all + 1e-8)
                X2 = (X2 - mean_all) / (std_all + 1e-8)
                # Transform to tensor for Sinkhorn methods
                X1_tensor = torch.tensor(X1, dtype=torch.float64, device=device)
                X2_tensor = torch.tensor(X2, dtype=torch.float64, device=device)

                # For each method, train the corresponding model and record the results.
                for method in methods:
                    print('-----', method, '-----')

                    # Average error rate for all methods
                    method_error_rate = []

                    # Solve the optimization problem for the selected method and record the results.
                    train_start_time = time.perf_counter()
                    if method == 'SDRO_H':
                        if d < 8:
                            input_dim, hidden_dim, output_dim = d, [4], 1
                        elif d >= 8 and d <= 16:
                            input_dim, hidden_dim, output_dim = d, [8], 1
                        elif d > 16 and d <= 64:
                            input_dim, hidden_dim, output_dim = d, [32], 1
                        else:
                            input_dim, hidden_dim, output_dim = d, [64], 1
                        # Initialize the HyperICNNs for Sinkhorn DRO
                        icnn_0 = HyCNN(input_dim, hidden_dim, output_dim).to(device).double()
                        icnn_1 = HyCNN(input_dim, hidden_dim, output_dim).to(device).double()

                        # Train the HyperICNNs by the Sinkhorn DRO solver
                        Sinkhorn_DRO(icnn_0, icnn_1, X1_tensor, X2_tensor, d,
                                     epsilon_k=1, lambda_k=10, num_epochs=max_sdro_epochs,
                                     sample_size=5000, mini_batch_size=1000, 
                                     learning_rate=5e-3, weight_decay=1e-2,
                                     early_stopping=True, tol=convergence_tol,
                                     verbose=False)

                    elif method == 'SDRO_C':
                        if d <= 64:
                            input_dim, hidden_dim, output_dim = d, 8, 1
                        else:
                            input_dim, hidden_dim, output_dim = d, 16, 1
                        # Initialize the BasicICNNs for Sinkhorn DRO
                        icnn_0 = ICNN(input_dim, hidden_dim, output_dim).to(device).double()
                        icnn_1 = ICNN(input_dim, hidden_dim, output_dim).to(device).double()
                        # Train the BasicICNNs by the Sinkhorn DRO solver
                        Sinkhorn_DRO(icnn_0, icnn_1, X1_tensor, X2_tensor, d,
                                     epsilon_k=1.2, lambda_k=30, num_epochs=max_sdro_epochs,
                                     sample_size=500, mini_batch_size=100,
                                     learning_rate=5e-3, weight_decay=1e-2,
                                     early_stopping=True, tol=convergence_tol,
                                     verbose=False)

                    elif method == 'WDRO':
                        # Cross validation
                        cross_validation = True
                        if cross_validation:
                            rho, bandwidth = cross_validation_WDRO(X1, X2,
                                                                   N_train, solver,
                                                                   radius_set=[0.001, 0.005, 0.01, 0.05],
                                                                   bandwidth_set=[2, 3, 4, 5, 6])

                        # Get the optimal value of WDRO and the least favorable distributions (supported by the training set).
                        value, p1_lfd, p_2lfd = WDRO(X1, X2, N_train, rho, flag=False, solver=solver)

                    else:
                        # train ML baselines 
                        model = trainBaseline(method, X1, X2, max_iter=max_baseline_iter, tol=convergence_tol)

                    # Record running time for each method
                    train_elapsed_time = time.perf_counter() - train_start_time
                    if computational_time:
                        computation_time_results.append({
                            'Trial': trial,
                            'method': method,
                            'computation_time': train_elapsed_time,
                        })

                    # For different value of observations, calculate the error rate and record the results in the table.
                    for observation_size in observations_set:

                        # Generate the testing data
                        X_test_1, X_test_2 = generate_Gaussian_mixture_data(data, N_test * observation_size, d)
                        # Normalization for testing data
                        X_test_1 = (X_test_1 - mean_all) / (std_all + 1e-8)
                        X_test_2 = (X_test_2 - mean_all) / (std_all + 1e-8)
                        def test_current_method(A, B):
                            A_tensor = torch.tensor(A, dtype=torch.float64, device=device)
                            B_tensor = torch.tensor(B, dtype=torch.float64, device=device)

                            if method == 'SDRO_H':
                                accuracy, _, _ = evaluate_testing_set_without_label(
                                    icnn_0, icnn_1, X1_tensor, X2_tensor,
                                    A_tensor, B_tensor, observation_size, epsilon_k=1, d=d
                                )
                                return 1 - accuracy

                            if method == 'SDRO_C':
                                accuracy, _, _ = evaluate_testing_set_without_label(
                                    icnn_0, icnn_1, X1_tensor, X2_tensor,
                                    A_tensor, B_tensor, observation_size, epsilon_k=1.2, d=d
                                )
                                return 1 - accuracy

                            if method == 'WDRO':
                                error_rate, _, _ = calculateErrors(
                                    X1, X2, A, B, p1_lfd, p_2lfd, bandwidth, observation_size
                                )
                                return error_rate

                            return testBaseline(method, model, A, B, observation_size)

                        error_rate = test_current_method(X_test_1, X_test_2)
                        all_results.append({'Trial': trial, 'observations': observation_size,
                                            'method': method, 'error_rate': error_rate})

            if not all_results:
                continue

            df_raw = pd.DataFrame(all_results)

            # Sheet 1: Average error rates for each batch size and baseline
            df_mean = df_raw.pivot_table(index='observations',
                                         columns='method',
                                         values='error_rate',
                                         aggfunc='mean')
            # Follow the order in baseline list
            df_mean = df_mean.reindex(columns=methods)

            # Sheet 2: All results
            df_detailed = df_raw.pivot_table(index=['observations', 'Trial'],
                                             columns='method',
                                             values='error_rate')
            df_detailed = df_detailed.reindex(columns=methods)

            with pd.ExcelWriter(os.path.join(output_path, file_name)) as writer:
                df_mean.to_excel(writer, sheet_name='Average_Results', index=True)
                df_detailed.to_excel(writer, sheet_name='All_Trials_Detail', index=True)

            # Plotting
            plot_error_rate_curves(
                df_mean,
                'Dataset: Gaussian Mixture',
                d,
                os.path.join(figure_output_path, figure_file_name)
            )

            # Compute average computation time
            if computational_time:
                df_time_raw = pd.DataFrame(computation_time_results)
                mean_times = (
                    df_time_raw.groupby('method')['computation_time']
                    .mean()
                    .reindex(methods)
                )
                time_row = {'N_train': N_train, 'd': d}
                time_row.update(mean_times.to_dict())
                computation_time_summary.append(time_row)

    # Save the computation time sheet
    if computational_time:
        df_time_summary = pd.DataFrame(computation_time_summary, columns=['N_train', 'd'] + methods)
        df_time_summary.to_excel(
            os.path.join(output_path, f'average_computation_time_{data}.xlsx'),
            index=False,
        )
