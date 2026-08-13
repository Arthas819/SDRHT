"""
    This file is a WDRO solver. 
"""

import cvxpy as cp
import numpy as np
import scipy.spatial


def WDRO(X1, X2, N, rho, flag, solver):

    '''
        X1: The feature data of the first class, a numpy array of shape (N, d).
        X2: The feature data of the second class, a numpy array of shape (N, d).
        N: The number of samples in the dataset, an integer.
        rho: The radius of the Wasserstein ball, a list of two floats [rho_1, rho_2].
        flag: A boolean indicating whether to print the optimization results.
    '''

    Data_XY = np.concatenate((X1, X2), axis=0)
    
    D = scipy.spatial.distance.cdist(Data_XY, Data_XY, metric='euclidean')

    rho_1, rho_2 = rho[0], rho[1]

    p_1 = cp.Variable(N*2)
    p_2 = cp.Variable(N*2)

    Gamma_1 = cp.Variable((N*2, N*2), nonneg=True)
    Gamma_2 = cp.Variable((N*2, N*2), nonneg=True)

    Proj_1_Gamma_1 = np.concatenate((np.ones(N)/N, np.zeros(N)))
    Proj_1_Gamma_2 = np.concatenate((np.zeros(N), np.ones(N)/N))

    # Generating function: (t+1)^+
    objective = cp.Maximize(cp.sum(cp.minimum(p_1, p_2)) * 2)

    constraints = [p_1 >= 0, p_2 >= 0, cp.sum(p_1) == 1, cp.sum(p_2) == 1]
    constraints += [cp.sum(cp.multiply(Gamma_1, D)) <= rho_1, cp.sum(cp.multiply(Gamma_2, D)) <= rho_2]
    constraints += [cp.sum(Gamma_1, axis=1) == Proj_1_Gamma_1,
                    cp.sum(Gamma_2, axis=1) == Proj_1_Gamma_2]
    constraints += [cp.sum(Gamma_1, axis=0) == p_1, cp.sum(Gamma_2, axis=0) == p_2]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    if flag == True:
        print("Optimal value:", problem.value)
        print("Optimal p_1:", p_1.value)
        print("Optimal p_2:", p_2.value)

    return problem.value, p_1.value, p_2.value


#  Cross-validation procedure for selecting the radii of the ambiguity sets and smoothing bandwidth.   
def cross_validation_WDRO(X1, X2, N_train, solver, radius_set, bandwidth_set):
    # record the best radii and bandwidth 
    error_rate_min = float('inf')
    best_rho = None
    best_bandwidth = None

    for rho in radius_set:
        for bandwidth in bandwidth_set:

            # Seperate the training set into training and validation sets
            N_training_size = int(N_train * 0.7)

            # Random indices
            indices = np.random.permutation(N_train)
            train_idx = indices[:N_training_size]
            val_idx = indices[N_training_size:]
            
            # Seperate training sets
            X_train_1, X_val_1 = X1[train_idx], X1[val_idx]
            X_train_2, X_val_2 = X2[train_idx], X2[val_idx]

            # Radii
            all_rho = [rho, rho]
            _, p1_lfd, p_2lfd = WDRO(X_train_1, X_train_2, N_training_size, all_rho, False, solver)
            
            # Calculate the error rate
            error_rate, e1, e2 = calculateErrors(X_train_1, X_train_2, X_val_1, X_val_2, p1_lfd, p_2lfd, bandwidth, 1)
            
            # Compare the error rate
            if error_rate < error_rate_min:
                error_rate_min = error_rate
                best_rho= rho
                best_bandwidth = bandwidth

    # print("Best rho:", best_rho)
    # print("Best bandwidth:", best_bandwidth)

    return [best_rho, best_rho], best_bandwidth

# Evaluate the error rate of the WDRO model on the testing set. 
def calculateErrors(X1, X2, X_test_1, X_test_2, p1_lfd, p2_lfd, bandwidth, batch_size):

    # A batch test contains m obseravations and make decision by "majority rule"
    batch_num = int(X_test_1.shape[0] / batch_size)

    start_index = 0

    # For all batches, 
    t1_error_num = 0
    t2_error_num = 0

    for i in range(batch_num):

        X_test_1_batch = X_test_1[start_index : start_index + batch_size, :]
        X_test_2_batch = X_test_2[start_index : start_index + batch_size, :]

        start_index += batch_size

        # Calculate the probability of accepting the null hypothesis H0 for each testing data point
        prob_H0_1 = kernel_smoothing(X1, X2, X_test_1_batch, p1_lfd, p2_lfd, bandwidth)
        prob_H0_2 = kernel_smoothing(X1, X2, X_test_2_batch, p1_lfd, p2_lfd, bandwidth)

        # Majority rule
        pi_1 = np.mean(prob_H0_1)
        if pi_1 < 0.5:
            t1_error_num += 1

        pi_2 = np.mean(prob_H0_2)
        if pi_2 > 0.5:
            t2_error_num += 1

    type_1_error_rate = t1_error_num / batch_num
    type_2_error_rate = t2_error_num / batch_num
        
    # error rate
    error_rate = (type_1_error_rate + type_2_error_rate) / 2

    return error_rate, type_1_error_rate, type_2_error_rate

# Extend the LFD of WDRO to the whole space by kernel smoothing.
def kernel_smoothing(X1, X2, X_test, p1_lfd, p2_lfd, bandwidth):

    X_train = np.vstack([X1, X2])
    
    # Distance and kernel weights between the training data and the testing data
    distances = scipy.spatial.distance.cdist(X_train, X_test, metric='euclidean')

    # kernel weights shape: (N_train, N_test), where each element represents the weight of the corresponding training data point for the corresponding testing data point.
    kernel_weights = np.exp(-distances**2 / (2 * bandwidth**2))

    # Create the density of testing data 
    # [N_train,] -> [N_train, 1], [N_train, 1]*[N_train, N_test] -> [N_train, N_test] -> [N_test,]
    p1_test = np.sum(p1_lfd[:, np.newaxis] * kernel_weights, axis=0)
    p2_test = np.sum(p2_lfd[:, np.newaxis] * kernel_weights, axis=0)

    # The probability of accepting the null hypothesis H0.
    # prob_H0: (N_test, ), where each element represents the probability of accepting the null hypothesis H0 for the corresponding testing data point.
    prob_H0 = p1_test / (p1_test + p2_test + 1e-10)

    # prob_H0 is a numpy array of shape (N_test,), where each element represents the probability of accepting the null hypothesis H0 for the corresponding testing data point.
    return prob_H0 