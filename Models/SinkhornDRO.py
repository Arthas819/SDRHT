"""
    This file is the implementation of the Sinkhorn Distributionally Robust Optimization solver.
"""

from Sinkhorn.Evaluation import *
from Sinkhorn.HyperICNN import *
from Sinkhorn.Objective_estimation import *

from Plotting.Plotting import *

'''
    Sinkhorn DRO solver.
'''
def Sinkhorn_DRO(icnn_0, icnn_1, X1_tensor, X2_tensor, d,
                 epsilon_k, lambda_k, num_epochs, 
                 sample_size, mini_batch_size, 
                 learning_rate, weight_decay, pre_train = True, clip_norm = True,
                 objective_T=1.0, risk_clip=None, verbose=True,
                 early_stopping=False, tol=1e-3, min_epochs=2,
                 return_history=False):
    """
        icnn： (Hyper) Input Convex Neural Networks. 
        X1_tensor, X2_tensor: Training data for two distributions, should be turned to tensor and normalized.
    """

    # Parameters for training
    num_batches = sample_size // mini_batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optimizers
    optimizer_0 = torch.optim.Adam(icnn_0.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_1 = torch.optim.Adam(icnn_1.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Pre-Train the HyCNNs to approximate identity maps
    if pre_train == True:
        for _ in range(10):
            optimizer_0.zero_grad()
            optimizer_1.zero_grad()

            z_pre = torch.randn(256, d, dtype=torch.float64, device=device).requires_grad_(True)

            phi0_pre = icnn_0(z_pre)
            omega0_pre = torch.autograd.grad(phi0_pre.sum(), z_pre, create_graph=True)[0]
            loss_pre_0 = torch.nn.functional.mse_loss(omega0_pre, z_pre)

            phi1_pre = icnn_1(z_pre)
            omega1_pre = torch.autograd.grad(phi1_pre.sum(), z_pre, create_graph=True)[0]
            loss_pre_1 = torch.nn.functional.mse_loss(omega1_pre, z_pre)

            (loss_pre_0 + loss_pre_1).backward()
            optimizer_0.step()
            optimizer_1.step()

    history = {
        "epoch_loss": [],
        "epochs_run": 0,
        "converged": False,
    }
    previous_epoch_loss = None

    # Formal training
    for epoch in range(num_epochs):
        if verbose:
            print(" === Epoch === ", epoch)

        epoch_loss = 0.0

        for _ in range(num_batches):
            optimizer_0.zero_grad()
            optimizer_1.zero_grad()

            loss = objective_estimator(icnn_0, icnn_1, X1_tensor, X2_tensor, mini_batch_size, lambda_k, epsilon_k,
                T=objective_T, risk_clip=risk_clip, verbose=verbose)

            optimizer_0.zero_grad()
            optimizer_1.zero_grad()
            loss.backward()

            if verbose:
                print('------ loss:', loss.item(), ' -------')

            # Clip the gradients to prevent exploding
            if clip_norm == True:
                torch.nn.utils.clip_grad_norm_(icnn_0.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(icnn_1.parameters(), max_norm=5.0)

            optimizer_0.step()
            optimizer_1.step()

            epoch_loss += loss.item()

        epoch_loss /= max(num_batches, 1)
        history["epoch_loss"].append(epoch_loss)
        history["epochs_run"] = epoch + 1

        if early_stopping and previous_epoch_loss is not None and epoch + 1 >= min_epochs:
            relative_change = abs(epoch_loss - previous_epoch_loss) / max(abs(previous_epoch_loss), 1e-12)
            if relative_change <= tol:
                history["converged"] = True
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1}/{num_epochs}, "
                        f"relative loss change = {relative_change:.2e}"
                    )
                break

        previous_epoch_loss = epoch_loss

    if return_history:
        return history
