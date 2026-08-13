"""
    Sinkhorn Distributionally Robust Hyperthesis Testing for MNIST Dataset.
"""

import os
import time

os.environ["MPLBACKEND"] = "Agg"

from Models.SinkhornDRO import *
from Models.WDRO import *
from Models.FDRO import *
from Models.StandardBaselines import *

from DataGenerators.Data_MNIST import *

from Sinkhorn.HyperICNN import *

import cvxpy as cp
import pandas as pd
from Plotting.Plotting import *

# Choose Dataset.
datasets = ['MNIST']
# Choose Methods to solve the optimization problem.
methods = ['SDRO', 'FDRO', 'WDRO', 'LR', 'SVM', '3NN']
# Total trials for different batch sizes
total_trials = 100
computational_time = True
convergence_tol = 1e-3
max_sdro_epochs = 5
max_fdro_batches = 5
max_baseline_iter = 1000

# Scale of the dataset (number of samples, dimension of features, etc.).
N_train_set = [2, 5, 10] 
N_test = 500
# Dimension of features
dim_set = [28*28]
# Testing batch size
observations_set = list(range(1, 11)) # [1, 2, ..., 10]

# Element number in each class
class_count = 5   # equals 5 if select all 10 digits
use_odd_even_split = False  # False: Use randomly selected digits; True: Use odd/even digits for two classes.
mnist_group_seed = 42
mnist_test_seed = 42

# Choose COPT to solve the WDRO model.
solver = cp.COPT
# Device for training the ICNNs in Sinkhorn DRO
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

                # Get digit split
                class1_digits, class2_digits = get_mnist_digit_groups(
                    use_odd_even_split=use_odd_even_split,
                    random_seed=mnist_group_seed,
                    class_count=class_count,
                )

                # Get data from the selected dataset, X1 belongs to P1 and X2 belongs to P2.
                X1, X2 = generate_MNIST_train(N_train, class1_digits, class2_digits)
                train_group_size = X1.shape[0]
                max_observation_size = max(observations_set)
                X_test_1_panel, X_test_2_panel = generate_MNIST_test_class_observations(
                    N_test,
                    max_observation_size,
                    class1_digits,
                    class2_digits,
                    random_seed=mnist_test_seed + trial,
                )
                # Tensor for training NN
                X1_tensor = torch.tensor(X1, dtype=torch.float64, device=device)
                X2_tensor = torch.tensor(X2, dtype=torch.float64, device=device)

                # For each method, train the corresponding model and record the results.
                for method in methods:
                    print('-----', method, '-----')

                    # Average error rate for all methods
                    method_error_rate = []

                    # Solve the optimization problem for the selected method and record the results.
                    train_start_time = time.perf_counter()
                    if method == 'SDRO':
                        input_dim, hidden_dim, output_dim = d, [32], 1

                        # Train the HyperICNNs by the Sinkhorn DRO solver
                        if use_odd_even_split:
                            icnn_0 = HyCNN(input_dim, hidden_dim, output_dim, tau=10, quad_coefficient=1e-2).to(device).double()
                            icnn_1 = HyCNN(input_dim, hidden_dim, output_dim, tau=10, quad_coefficient=1e-2).to(device).double()
                            Sinkhorn_DRO(icnn_0, icnn_1, X1_tensor, X2_tensor, d,
                                         epsilon_k=0.1, lambda_k=1, num_epochs=max_sdro_epochs,
                                         sample_size=2000, mini_batch_size=500,
                                         learning_rate=5e-3, weight_decay=1e-3,
                                         early_stopping=True, tol=convergence_tol,
                                         verbose=False)
                        else:
                            icnn_0 = HyCNN(input_dim, hidden_dim, output_dim, tau=10, quad_coefficient=0.2).to(device).double()
                            icnn_1 = HyCNN(input_dim, hidden_dim, output_dim, tau=10, quad_coefficient=0.2).to(device).double()
                            Sinkhorn_DRO(icnn_0, icnn_1, X1_tensor, X2_tensor, d,
                                         epsilon_k=0.1, lambda_k=0.5, num_epochs=max_sdro_epochs,
                                         sample_size=2000, mini_batch_size=500,
                                         learning_rate=5e-3, weight_decay=1e-3,
                                         early_stopping=True, tol=convergence_tol,
                                         verbose=False)

                    elif method == 'FDRO':
                        fdro_model = FDRO(
                            X1, X2, batch_tot=max_fdro_batches, batch_size=512,
                            gamma=5, FRM_steps=3, lr_flow=1e-4, lr_cnn=1e-4,
                            device=device, dtype=torch.float64, seed=trial,
                            early_stopping=True, tol=convergence_tol)

                    elif method == 'WDRO':
                        # Cross validation has been done.
                        rho = [0.01, 0.01]
                        bandwidth = 1.5
                        # Get the optimal value of WDRO and the least favorable distributions.
                        value, p1_lfd, p_2lfd = WDRO(X1, X2, N_train * class_count, rho, flag=False, solver=solver)

                    else:
                        # train ML models
                        model = trainBaseline(method, X1, X2, max_iter=max_baseline_iter, tol=convergence_tol)

                    # Record computation time for each method
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
                        X_test_1 = select_observation_prefix(
                            X_test_1_panel, observation_size, max_observation_size
                        )
                        X_test_2 = select_observation_prefix(
                            X_test_2_panel, observation_size, max_observation_size
                        )

                        def test_current_method(A, B):
                            if method == 'SDRO':
                                epsilon = 0.1 if use_odd_even_split else 0.1
                                A_tensor = torch.tensor(A, dtype=torch.float64, device=device)
                                B_tensor = torch.tensor(B, dtype=torch.float64, device=device)
                                accuracy, _, _ = evaluate_testing_set_without_label(
                                    icnn_0, icnn_1, X1_tensor, X2_tensor,
                                    A_tensor, B_tensor, observation_size, epsilon_k=epsilon, d=d)
                                return 1 - accuracy

                            if method == 'FDRO':
                                return test_FDRO(fdro_model, A, B, observation_size)

                            if method == 'WDRO':
                                error_rate, _, _ = calculateErrors(
                                    X1, X2, A, B, p1_lfd, p_2lfd, bandwidth, observation_size)
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
            plot_error_rate_curves(
                df_mean,
                'Dataset: MNIST',
                d,
                os.path.join(figure_output_path, figure_file_name)
            )

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

    if computational_time:
        df_time_summary = pd.DataFrame(computation_time_summary, columns=['N_train', 'd'] + methods)
        df_time_summary.to_excel(
            os.path.join(output_path, f'average_computation_time_{data}.xlsx'),
            index=False,
        )
