'''
    This file provides the evaluation procedures for ICNN / HyperICNN
'''

from Sinkhorn.Objective_estimation import *

# Evaluation functions for testing set, X_test_0_tensor and X_test_1_tensor imply lables 0 and 1, respectively. 
def evaluate_testing_set_without_label(icnn_0, icnn_1, 
                                       X0_tensor, X1_tensor, X_test_0_tensor, X_test_1_tensor,
                                       observation, epsilon_k, d, score_clip=None):
    
    icnn_0.eval()
    icnn_1.eval()

    # For testing data from distribution 0, trace their original points in the latent space, and compute their log-density
    # First, get z = (\nabla \varphi)^{-1} (X_test), z: [N_test, d]
    z0_test = invert_icnn_map(icnn_0, X_test_0_tensor)
    z1_test = invert_icnn_map(icnn_1, X_test_0_tensor)

    # Compute the log-density of z
    log_density_0_test = eval_log_prob_reference(z0_test, X0_tensor, epsilon_k) - get_log_det_hessian(icnn_0, z0_test, d)
    log_density_1_test = eval_log_prob_reference(z1_test, X1_tensor, epsilon_k) - get_log_det_hessian(icnn_1, z1_test, d)

    with torch.no_grad():
        # Use the optimal detector to make decision phi*(omega) = 1/2 * (log dP_0 - log dP_1)
        detector_scores_0 = 0.5 * (log_density_0_test - log_density_1_test)
        if score_clip is not None:
            detector_scores_0 = torch.clamp(detector_scores_0, min=-score_clip, max=score_clip)
        # [N_test * observation] -> [N_test, observation] -> [N_test] (mean)
        group_scores_0 = detector_scores_0.view(-1, observation).mean(dim=1)
        # For the N_test groups, make decision by their average score.
        group_predictions_0 = torch.where(group_scores_0 > 0, 0, 1)
        # How many groups are correctly classified. Accuracy = correct_groups_0 / N_test
        accuracy_0 = (group_predictions_0 == 0).sum().item() / group_scores_0.size(0)

    # For testing data from distribution 1...
    z0_test = invert_icnn_map(icnn_0, X_test_1_tensor)
    z1_test = invert_icnn_map(icnn_1, X_test_1_tensor)

    log_density_0_test = eval_log_prob_reference(z0_test, X0_tensor, epsilon_k) - get_log_det_hessian(icnn_0, z0_test, d)
    log_density_1_test = eval_log_prob_reference(z1_test, X1_tensor, epsilon_k) - get_log_det_hessian(icnn_1, z1_test, d)

    with torch.no_grad():
        detector_scores_1 = 0.5 * (log_density_0_test - log_density_1_test)
        if score_clip is not None:
            detector_scores_1 = torch.clamp(detector_scores_1, min=-score_clip, max=score_clip)
        group_scores_1 = detector_scores_1.view(-1, observation).mean(dim=1)
        group_predictions_1 = torch.where(group_scores_1 > 0, 0, 1)
        accuracy_1 = (group_predictions_1 == 1).sum().item() / group_scores_1.size(0)
        accuracy = (accuracy_0 + accuracy_1) / 2

    # Back to training mode for the next round of training. 
    icnn_0.train()
    icnn_1.train()

    # print("accuracy: ", accuracy_0, accuracy_1)
    # print(f"Mean score for Class 0: {detector_scores_0.mean().item():.3f}")
    # print(f"Mean score for Class 1: {detector_scores_1.mean().item():.3f}")

    return accuracy, accuracy_0, accuracy_1

def evaluate_testing_panel_without_label(
    icnn_0,
    icnn_1,
    X0_tensor,
    X1_tensor,
    X_test_0_panel_tensor,
    X_test_1_panel_tensor,
    max_observation,
    observations_set,
    epsilon_k,
    d,
    trial=None,
):
    icnn_0.eval()
    icnn_1.eval()

    z0_test = invert_icnn_map(icnn_0, X_test_0_panel_tensor)
    z1_test = invert_icnn_map(icnn_1, X_test_0_panel_tensor)

    log_density_0_test = eval_log_prob_reference(z0_test, X0_tensor, epsilon_k) - get_log_det_hessian(icnn_0, z0_test, d)
    log_density_1_test = eval_log_prob_reference(z1_test, X1_tensor, epsilon_k) - get_log_det_hessian(icnn_1, z1_test, d)
    detector_scores_0 = 0.5 * (log_density_0_test - log_density_1_test)

    z0_test = invert_icnn_map(icnn_0, X_test_1_panel_tensor)
    z1_test = invert_icnn_map(icnn_1, X_test_1_panel_tensor)

    log_density_0_test = eval_log_prob_reference(z0_test, X0_tensor, epsilon_k) - get_log_det_hessian(icnn_0, z0_test, d)
    log_density_1_test = eval_log_prob_reference(z1_test, X1_tensor, epsilon_k) - get_log_det_hessian(icnn_1, z1_test, d)
    detector_scores_1 = 0.5 * (log_density_1_test - log_density_0_test)

    with torch.no_grad():
        score_panel_0 = detector_scores_0.view(-1, max_observation)
        score_panel_1 = detector_scores_1.view(-1, max_observation)
        results = []

        for observation in observations_set:
            group_scores_0 = score_panel_0[:, :observation].mean(dim=1)
            group_predictions_0 = torch.where(group_scores_0 > 0, 0, 1)
            accuracy_0 = (group_predictions_0 == 0).sum().item() / group_scores_0.size(0)
            positive_rate_0 = (group_scores_0 > 0).sum().item() / group_scores_0.size(0)

            group_scores_1 = score_panel_1[:, :observation].mean(dim=1)
            group_predictions_1 = torch.where(group_scores_1 > 0, 1, 0)
            accuracy_1 = (group_predictions_1 == 1).sum().item() / group_scores_1.size(0)
            positive_rate_1 = (group_scores_1 > 0).sum().item() / group_scores_1.size(0)
            accuracy = (accuracy_0 + accuracy_1) / 2
            error_rate = 1 - accuracy

            results.append(
                {
                    "observations": observation,
                    "accuracy": accuracy,
                    "accuracy_0": accuracy_0,
                    "accuracy_1": accuracy_1,
                }
            )

    icnn_0.train()
    icnn_1.train()

    return results

# Evaluation function for training set, X0_tensor and X1_tensor imply lables 0 and 1, respectively. 
# eval_ratio = 30% is the ratio of samples used for evaluation. 
def evaluate_training_set_without_label(icnn_0, icnn_1, X0_tensor, X1_tensor, epsilon_k, d, eval_ratio):
    N0 = X0_tensor.shape[0]
    N1 = X1_tensor.shape[0]
    num_test_0 = max(1, int(N0 * eval_ratio))
    num_test_1 = max(1, int(N1 * eval_ratio))

    # Generate indice and extract these samples as evaluation sets. 
    idx_0 = torch.randperm(N0, device=X0_tensor.device)[:num_test_0]
    idx_1 = torch.randperm(N1, device=X1_tensor.device)[:num_test_1]
    X_eval_0_tensor = X0_tensor[idx_0]
    X_eval_1_tensor = X1_tensor[idx_1]

    icnn_0.eval()
    icnn_1.eval()

    z0_test = invert_icnn_map(icnn_0, X_eval_0_tensor)
    z1_test = invert_icnn_map(icnn_1, X_eval_0_tensor)
    log_density_0_test = eval_log_prob_reference(z0_test, X0_tensor, epsilon_k) - get_log_det_hessian(icnn_0, z0_test, d)
    log_density_1_test = eval_log_prob_reference(z1_test, X1_tensor, epsilon_k) - get_log_det_hessian(icnn_1, z1_test, d)

    with torch.no_grad():
        detector_scores = 0.5 * (log_density_0_test - log_density_1_test)
        predictions = torch.where(detector_scores > 0, 0, 1)
        accuracy_0 = (predictions == 0).sum().item() / len(X_eval_0_tensor)

    z0_test = invert_icnn_map(icnn_0, X_eval_1_tensor)
    z1_test = invert_icnn_map(icnn_1, X_eval_1_tensor)
    log_density_0_test = eval_log_prob_reference(z0_test, X0_tensor, epsilon_k) - get_log_det_hessian(icnn_0, z0_test, d)
    log_density_1_test = eval_log_prob_reference(z1_test, X1_tensor, epsilon_k) - get_log_det_hessian(icnn_1, z1_test, d)

    with torch.no_grad():
        detector_scores = 0.5 * (log_density_1_test - log_density_0_test)
        predictions = torch.where(detector_scores > 0, 1, 0)
        accuracy_1 = (predictions == 1).sum().item() / len(X_eval_1_tensor)
        accuracy = (accuracy_0 + accuracy_1) / 2

    icnn_0.train()
    icnn_1.train()

    return accuracy, accuracy_0, accuracy_1
