"""
    Sinkhorn Distributionally Robust Hyperthesis Testing for Higgs Dataset.
"""

import os
import time

os.environ["MPLBACKEND"] = "Agg"

from Models.SinkhornDRO import *
from Models.FDRO import *
from Models.StandardBaselines import *

from DataGenerators.Data_Higgs import *

from Sinkhorn.HyperICNN import *

import pandas as pd
import torch
from Plotting.Plotting import *


# Choose Dataset.
datasets = ["Higgs"]
# Choose Methods to solve the optimization problem.
methods = ["SDRO", "FDRO", "LR", "SVM", "3NN"]
# Total trials for different batch sizes.
total_trials = 5
# Record computation time.
computational_time = False

# Scale of the dataset.
N_train_set = list(range(1000, 10001, 1000))
N_test = 500
# Dimension of features.
dim_set = [21]
# Testing batch size.
observations_set = list(range(1, 11))

# Parameters for SDRO and FDRO.
convergence_tol = 1e-3
max_sdro_epochs = 5
max_baseline_iter = 1000
fdro_params = {
    "batch_tot": 0,
    "batch_size": 512,
    "gamma": 200,
    "FRM_steps": 1,
    "lr_flow": 1e-5,
    "lr_cnn": 3e-4,
    "tabular_hidden_dim": 256,
    "classifier_pretrain_steps": 1600,
}

higgs_data_path = r"E:\Datasets\higgs\HIGGS.csv"
higgs_pool_size_per_class = 50000

# Device for training networks.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Solve the optimization problem for each dataset and each baseline, and record the results in the table.
for data in datasets:
    output_path = f"Results/{data}"
    figure_output_path = f"Results/{data}"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(figure_output_path, exist_ok=True)

    # Load once and reuse the same HIGGS pools for all training sizes.
    X1_pool, X2_pool = load_higgs_class_pools(
        higgs_data_path,
        pool_size_per_class=higgs_pool_size_per_class,
    )

    all_n_train_results = []
    computation_time_summary = []
    max_train = max(N_train_set)
    max_observation_size = max(observations_set)

    for N_train in N_train_set:
        for d in dim_set:

            print(" -------N_train=", N_train, ", d=", d, "-----")
            file_name = f"all_results_{data}_sample={N_train}_d={d}.xlsx"
            figure_file_name = f"all_results_{data}_sample={N_train}_d={d}_error_rate.pdf"

            all_results = []
            computation_time_results = []

            for trial in range(total_trials):

                print(" ======= Trial: ", trial, "========")

                # Get data from HIGGS. Training data and test panels are normalized by the data generator.
                X1, X2, X_test_1_panel, X_test_2_panel = get_higgs_train_test(X1_pool, X2_pool, N_train, max_train, N_test, max_observation_size, trial)

                # Transform to tensor for Sinkhorn methods.
                X1_tensor = torch.tensor(X1, dtype=torch.float64, device=device)
                X2_tensor = torch.tensor(X2, dtype=torch.float64, device=device)

                # For each method, train the corresponding model and record the results.
                for method in methods:
                    print("-----", method, "-----")

                    # Solve the optimization problem for the selected method.
                    train_start_time = time.perf_counter()
                    if method == "SDRO":
                        input_dim, hidden_dim, output_dim = d, [32, 32], 1
                        epsilon = 1e-2
                        icnn_0 = HyCNN(input_dim, hidden_dim, output_dim, tau=1e1, quad_coefficient=0.2).to(device).double()
                        icnn_1 = HyCNN(input_dim, hidden_dim, output_dim, tau=1e1, quad_coefficient=0.2).to(device).double()

                        # Train the HyperICNNs by the Sinkhorn DRO solver.
                        Sinkhorn_DRO(icnn_0, icnn_1, X1_tensor, X2_tensor, d,
                            epsilon_k=epsilon, lambda_k=1, num_epochs=max_sdro_epochs,
                            sample_size=1000, mini_batch_size=1000, learning_rate=5e-3, weight_decay=1e-3,
                            pre_train=True, clip_norm=False, early_stopping=False, tol=convergence_tol, verbose=False)

                    elif method == "FDRO":
                        fdro_model = FDRO(X1, X2, dtype=torch.float32, seed=trial,
                            verbose=False, image_shape=None, early_stopping=False,
                            tol=convergence_tol, **fdro_params)

                    else:
                        # Train ML baselines.
                        model = trainBaseline(method, X1, X2, max_iter=max_baseline_iter, tol=convergence_tol)

                    # Record running time for each method.
                    train_elapsed_time = time.perf_counter() - train_start_time
                    if computational_time:
                        computation_time_results.append({
                            "Trial": trial,
                            "method": method,
                            "computation_time": train_elapsed_time,
                        })

                    # For different observations, calculate the error rate.
                    for observation_size in observations_set:
                        X_test_1 = select_observation_prefix(
                            X_test_1_panel, observation_size, max_observation_size
                        )
                        X_test_2 = select_observation_prefix(
                            X_test_2_panel, observation_size, max_observation_size
                        )

                        def test_current_method(A, B):
                            if method == "SDRO":
                                A_tensor = torch.tensor(A, dtype=torch.float64, device=device)
                                B_tensor = torch.tensor(B, dtype=torch.float64, device=device)
                                accuracy, _, _ = evaluate_testing_set_without_label(
                                    icnn_0, icnn_1, X1_tensor, X2_tensor,
                                    A_tensor, B_tensor, observation_size,
                                    epsilon_k=epsilon, d=d
                                )
                                return 1 - accuracy

                            if method == "FDRO":
                                return test_FDRO(fdro_model, A, B, observation_size)

                            return testBaseline(method, model, A, B, observation_size)

                        error_rate = test_current_method(X_test_1, X_test_2)
                        all_results.append({
                            "N_train": N_train,
                            "Trial": trial,
                            "observations": observation_size,
                            "method": method,
                            "error_rate": error_rate,
                        })

            if not all_results:
                continue

            df_raw = pd.DataFrame(all_results)
            all_n_train_results.extend(all_results)

            # Sheet 1: Average error rates for each batch size and baseline.
            df_mean = df_raw.pivot_table(index="observations",
                                         columns="method",
                                         values="error_rate",
                                         aggfunc="mean")
            df_mean = df_mean.reindex(columns=methods)

            # Sheet 2: All results.
            df_detailed = df_raw.pivot_table(index=["observations", "Trial"],
                                             columns="method",
                                             values="error_rate")
            df_detailed = df_detailed.reindex(columns=methods)

            with pd.ExcelWriter(os.path.join(output_path, file_name)) as writer:
                df_mean.to_excel(writer, sheet_name="Average_Results", index=True)
                df_detailed.to_excel(writer, sheet_name="All_Trials_Detail", index=True)
                df_raw.to_excel(writer, sheet_name="All_Results_Long", index=False)

            plot_error_rate_curves(df_mean, "Dataset: Higgs", d, os.path.join(figure_output_path, figure_file_name))

            if computational_time:
                df_time_raw = pd.DataFrame(computation_time_results)
                mean_times = (
                    df_time_raw.groupby("method")["computation_time"]
                    .mean()
                    .reindex(methods)
                )
                time_row = {"N_train": N_train, "d": d}
                time_row.update(mean_times.to_dict())
                computation_time_summary.append(time_row)

    # Summary over all training sizes, averaged over trials and observation sizes.
    if all_n_train_results:
        df_all = pd.DataFrame(all_n_train_results)
        df_all_mean = df_all.pivot_table(index="N_train",
                                         columns="method",
                                         values="error_rate",
                                         aggfunc="mean")
        df_all_mean = df_all_mean.reindex(index=N_train_set, columns=methods)

        with pd.ExcelWriter(os.path.join(output_path, f"all_results_{data}_all_N_train.xlsx")) as writer:
            df_all.to_excel(writer, sheet_name="All_Results_Long", index=False)
            df_all_mean.to_excel(writer, sheet_name="Average_By_N_Train", index=True)

        df_plot = df_all_mean.copy()
        df_plot.index = [n_train / 1000 for n_train in df_plot.index]
        plot_error_rate_curves(
            df_plot,
            "Dataset: Higgs",
            dim_set[0],
            os.path.join(figure_output_path, f"all_results_{data}_average_error_rate_by_N_train.pdf"),
            x_label=r"Training Sample Size ($\times 10^3$)",
            x_ticks=list(df_plot.index),
        )

    if computational_time:
        df_time_summary = pd.DataFrame(computation_time_summary, columns=["N_train", "d"] + methods)
        df_time_summary.to_excel(
            os.path.join(output_path, f"average_computation_time_{data}.xlsx"),
            index=False,
        )
