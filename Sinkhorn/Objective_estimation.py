"""
    This file provides the objective estimation procedures for ICNN / HyperICNN.
"""
import torch
import numpy as np
from torch.func import hessian, vmap

from Sinkhorn.logdet_estimators import stochastic_lanczos_quadrature, conjugate_gradient

# Constants for log-determinant estimation by stochastic Lanczos quadrature (SLQ, Huang et al. 2020, https://arxiv.org/abs/2012.05942)
SLQ_PROBES = 8  # Number of random probes for SLQ
SLQ_STEPS = 20  # Number of Lanczos steps for SLQ
SLQ_EVAL_PROBES = 4  # Number of random probes for SLQ when evaluating log-det during training
SLQ_EVAL_STEPS = 20  # Number of Lanczos steps for SLQ when evaluating log-det during training 
LOGDET_DAMPING = 1e-4  # Damping term added to the Hessian for numerical stability in SLQ estimation
EXACT_LOGDET_DIM_THRESHOLD = 32  # If dim \le 32, we compute log-det-Hessian by exact method; otherwise, we use SLQ estimation
HYICNN_QUAD_COEFF = 0.2  # Quadratic coefficient in HyCNN

'''
    Part 1: Total objective function estimation (risk + Sinkhorn discrepancy).
'''
def objective_estimator(icnn_0, icnn_1, X1_tensor, X2_tensor, mini_batch_size, lambda_k, epsilon_k, T=1.0, risk_clip=None, verbose=True):

    ### Part 1.1: Risk function
    ## Part 1.1.1: log-densities of P0 and P1 at the generated samples omega0 and omega1

    # Sample from P_k^{\epsilon}, and return its log-density
    # z0 and z1 are nodes in the computation graph, and their log-densities only have gradients with respect to z0 and z1
    # z: [sample_size, d],  log_density: [sample_size]
    z0, log_p_ref0 = sample_and_log_prob_reference(X1_tensor, mini_batch_size, epsilon_k)
    z1, log_p_ref1 = sample_and_log_prob_reference(X2_tensor, mini_batch_size, epsilon_k)
    # z: [sample_size, d],  log_density: [sample_size]
    # To compute \omega = \nabla (\varphi) z 
    z0.requires_grad_(True)
    z1.requires_grad_(True)

    ## Get the outputs of the ICNNs and compute the gradients
    phi0_z = icnn_0(z0) # phi0_z = varphi_0 (z0), [sample_size, 1] 
    phi1_z = icnn_1(z1)
    # omega0 = \nabla varphi_0 (z0) : [sample_size, d]
    omega0 = torch.autograd.grad(phi0_z.sum(), z0, create_graph=True)[0]
    omega1 = torch.autograd.grad(phi1_z.sum(), z1, create_graph=True)[0]

    # Compute the value of log-det-Hessian without gradient (.detach()). 
    log_det_0_val = get_log_det_hessian_without_gradient(icnn_0, z0, z0.shape[1])
    log_det_1_val = get_log_det_hessian_without_gradient(icnn_1, z1, z1.shape[1])
    # Surrogate gradient for log-det-Hessian, based on Huang et al. (2020).
    surrogate_0 = cg_surrogate_gradient(icnn_0, z0)
    surrogate_1 = cg_surrogate_gradient(icnn_1, z1)
    # Attach the surrogate gradient to the log-det value.
    log_det_0 = log_det_0_val + (surrogate_0 - surrogate_0.detach())
    log_det_1 = log_det_1_val + (surrogate_1 - surrogate_1.detach())
    # log-density
    log_dP0_at_omega0 = log_p_ref0 - log_det_0
    log_dP1_at_omega1 = log_p_ref1 - log_det_1

    # Compute the density of omega_0 in P1 and the density of omega_1 in P0
    # Find the origin sample of omega_0 in P_1^{\epsilon}, denoted as z01
    z01 = invert_icnn_map(icnn_1, omega0.detach())  # Now z01 is a leaf node in computational graph, as omega0.detach()
    # z01 need to be differentiable at omega0
    log_dP1_at_omega0 = implicit_log_density_at_omega(icnn_1, z01, omega0, X2_tensor, epsilon_k)

    z10 = invert_icnn_map(icnn_0, omega1.detach())
    log_dP0_at_omega1 = implicit_log_density_at_omega(icnn_0, z10, omega1, X1_tensor, epsilon_k)

    # Risk of detectors 
    diff_0 = 0.5 * (log_dP1_at_omega0 - log_dP0_at_omega0)
    diff_1 = 0.5 * (log_dP0_at_omega1 - log_dP1_at_omega1)

    diff_0_for_risk = diff_0
    diff_1_for_risk = diff_1
    if risk_clip is not None:
        diff_0_for_risk = torch.clamp(diff_0_for_risk, min=-risk_clip, max=risk_clip)
        diff_1_for_risk = torch.clamp(diff_1_for_risk, min=-risk_clip, max=risk_clip)

    # print("log_dP1_at_omega0", log_dP1_at_omega0.min().item(), log_dP1_at_omega0.max().item())
    # print("log_dP0_at_omega0", log_dP0_at_omega0.min().item(), log_dP0_at_omega0.max().item())

    risk_term_0 = torch.exp(diff_0_for_risk / T).mean()
    risk_term_1 = torch.exp(diff_1_for_risk / T).mean()
    detector_risk = risk_term_0 + risk_term_1

    # Sinkhorn discrepancy estimation. 
    S0 = sinkhorn_loss_with_extension(X1_tensor, omega0, epsilon_k)
    S1 = sinkhorn_loss_with_extension(X2_tensor, omega1, epsilon_k)

    if verbose:
        print(f"Diff 0 Min: {diff_0.min().item():.2f}, Max: {diff_0.max().item():.2f}")
        print(f"Risk 0: {risk_term_0.item():.4f}, Risk 1: {risk_term_1.item():.4f}")
        print(f"Sinkhorn 0: {S0.item():.4f}, Sinkhorn 1: {S1.item():.4f}")
        print(f"Log Det 0 Mean: {log_det_0.mean().item():.4f}, Log Det 1 Mean: {log_det_1.mean().item():.4f}")

    return -(detector_risk - lambda_k * S0 - lambda_k * S1)


'''
    Part 2: Estimation for the risk fuction. 
'''

## Reference distribution sampling and log-density evaluation
def sample_and_log_prob_reference(X_k, sample_size, epsilon):
    # Sample for P_k^{\epsilon} and compute log-density log dP_k^{\epsilon}(z)
    # X_k: (n_k, d) training data for distribution P_k
    n_k, d = X_k.shape
    device = X_k.device
    # Get indices (with replacement) from the training data
    indices = torch.randint(0, n_k, (sample_size,), device=device)
    centers = X_k[indices]
     # Generate random z from P_k^{\epsilon}
    z = centers + np.sqrt(epsilon) * torch.randn(sample_size, d, device=device, dtype=X_k.dtype)
    # z: [sample_size, d],  log_density: [sample_size]
    # This log-density has no gradient.
    return z, eval_log_prob_reference(z, X_k, epsilon)

# Evaluate the log-density of z under the reference distribution P_k^{\epsilon}
def eval_log_prob_reference(z, X_k, epsilon):
    n_k, d = X_k.shape
    # Follow the definition of transport cost
    dist_sq = torch.cdist(z, X_k, p=2) ** 2
    log_const = - 0.5 * d * np.log(2 * np.pi * epsilon)
    log_component_probs = log_const - (0.5 / epsilon) * dist_sq
    log_n_k = torch.log(torch.tensor(float(n_k), device=z.device, dtype=z.dtype))
    return torch.logsumexp(log_component_probs, dim=1) - log_n_k

# Attach the implicit gradient dz / d(omega) = Hessian(varphi)^{-1}. See the Appendix of our paper. 
def implicit_log_density_at_omega(icnn, z_inverse, omega, X_ref, epsilon, cg_iters=10, rtol=1e-3, atol=1e-4):
    z_inverse = z_inverse.detach().requires_grad_(True)

    # Compute log-density of z_inverse in P_k^{\epsilon}.
    log_p_ref_at_omega = eval_log_prob_reference(z_inverse, X_ref, epsilon)
    # Estimate the value of log-det-Hessian at omega without gradient.
    log_det_at_omega_val = get_log_det_hessian_without_gradient(icnn, z_inverse, z_inverse.shape[1])
    # Compute the surrogate gradient for log-det-Hessian at omega, based on Huang et al. (2020).
    surrogate_at_omega = cg_surrogate_gradient(icnn, z_inverse)
    log_det_at_omega = log_det_at_omega_val + (
        surrogate_at_omega - surrogate_at_omega.detach()
    )
    log_density_at_omega_val = log_p_ref_at_omega - log_det_at_omega

    phi_z_inverse = icnn(z_inverse)
    grad_phi_z_inverse = torch.autograd.grad(phi_z_inverse.sum(), z_inverse, create_graph=True)[0]
    g_z = torch.autograd.grad(log_density_at_omega_val.sum(), z_inverse, retain_graph=True)[0]

    # Hessian-Vector Product operator for Hessian(varphi)(z_inverse).
    def hvp_fun(v):
        hvp = torch.autograd.grad(
            grad_phi_z_inverse, z_inverse, grad_outputs=v,
            retain_graph=True, only_inputs=True
        )[0]
        return hvp + LOGDET_DAMPING * v

    # Solve Hessian(varphi)(z_inverse) v = grad_z log_density(z_inverse).
    with torch.no_grad():
        v_star = conjugate_gradient(hvp_fun, g_z, m=cg_iters, rtol=rtol, atol=atol)

    # For different parameters, the gradient is different. 
    surrogate_joint = (v_star.detach() * (omega - grad_phi_z_inverse)).sum(dim=1)
    return log_density_at_omega_val + surrogate_joint - surrogate_joint.detach()

# Get the value of log-det-Hessian with gradient.
def get_log_det_hessian(model, z, d):
    if d <= EXACT_LOGDET_DIM_THRESHOLD:
        return exact_logdet_value(model, z).detach()

    return safe_highdim_logdet_value(model, z, d)

# Get the value of log-det-Hessian without gradient.
def get_log_det_hessian_without_gradient(model, z, d):
    if d <= EXACT_LOGDET_DIM_THRESHOLD:
        return exact_logdet_value(model, z.detach()).detach()
    return safe_highdim_logdet_value(model, z, d)

# Exact log-det-Hessian value by batch-wise computation. 
def exact_logdet_value(model, z):
    def scalar_out(z_single):
        return model(z_single.unsqueeze(0)).sum()

    h_batch = vmap(hessian(scalar_out))(z.detach())
    return torch.logdet(h_batch)

# SLQ method
def safe_highdim_logdet_value(model, z, d):
    value = highdim_logdet_value(model, z)

    if torch.isfinite(value).all():
        return value

    fallback = slq_logdet_value(model, z, probes=1, steps=10, deterministic=True).detach()
    finite_fallback = torch.isfinite(fallback)

    if finite_fallback.all():
        return fallback

    quad_logdet = d * np.log(HYICNN_QUAD_COEFF + LOGDET_DAMPING)
    quad_logdet = torch.tensor(quad_logdet, dtype=z.dtype, device=z.device)
    fallback = torch.where(finite_fallback, fallback, quad_logdet.expand_as(fallback))
    return fallback

def highdim_logdet_value(model, z):
    was_training = model.training
    model.eval()
    try:
        _, value = model.forward_transform_stochastic(z.detach())
    finally:
        model.train(was_training)
    return value.detach()

def slq_logdet_value(model, z, probes=4, steps=20, deterministic=False):
    values = []
    probe_bank = deterministic_unit_rademacher(probes, z.shape, z.device, z.dtype) if deterministic else None
    z_req = z.detach().clone().requires_grad_(True)
    phi = model(z_req)
    grad_phi = torch.autograd.grad(phi.sum(), z_req, create_graph=True)[0]

    def hvp_fun(v):
        hv = torch.autograd.grad(grad_phi, z_req, grad_outputs=v, retain_graph=True)[0]
        return hv + LOGDET_DAMPING * v

    for probe_idx in range(probes):
        v = probe_bank[probe_idx] if deterministic else sample_unit_rademacher(z_req.shape, z_req.device, z_req.dtype)
        values.append(stochastic_lanczos_quadrature(hvp_fun, v, m=steps))
    return torch.stack(values, dim=0).mean(dim=0)

def sample_unit_rademacher(shape, device, dtype):
    v = (torch.randint(0, 2, shape, device=device, dtype=torch.int64).to(dtype) * 2.0) - 1.0
    return torch.nn.functional.normalize(v, dim=-1)


def deterministic_unit_rademacher(probes, sample_shape, device, dtype):
    batch, dim = sample_shape
    idx = torch.arange(batch * dim, device=device, dtype=dtype).reshape(batch, dim)
    values = []
    for probe_idx in range(probes):
        freq = 2 * probe_idx + 1
        phase = (probe_idx + 1) * 0.37
        v = torch.cos((idx + 0.5) * freq * np.pi / max(1, dim) + phase)
        v = v - v.mean(dim=1, keepdim=True)
        values.append(torch.nn.functional.normalize(v, dim=-1))
    q, _ = torch.linalg.qr(torch.stack(values, dim=1).transpose(1, 2), mode="reduced")
    return q.transpose(1, 2).transpose(0, 1)

## For given observated testing data omega, compute z = (\nabla \varphi)^{-1} (omega)
## The inverse map (\nabla \varphi)^{-1} is equivalent to the convex conjugate of convex function \varphi. 
def invert_icnn_map(icnn, omega, max_iter=100, tol=1e-7):
    # Initialize z as omega (indentity map)
    z = omega.clone().detach().requires_grad_(True)
    # L-BFGS optimizer 
    optimizer = torch.optim.LBFGS(
        [z],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        phi_z = icnn(z)
        loss = (phi_z.squeeze() - (omega * z).sum(dim=1)).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    # Return inverted z with gradient 
    return z.detach().requires_grad_(True)

# Estimate the gradient of log-det-Hessian by conjugate gradient method, based on Huang et al. (2020). 
def cg_surrogate_gradient(icnn, z, cg_iters=10):
    z = z.requires_grad_(True)
    # Compute f(z) = \nabla \varphi(z)
    f_z = torch.autograd.grad(icnn(z).sum(), z, create_graph=True)[0]
    # Rademacher vectors 
    v = (torch.randint_like(z, 0, 2).to(z.dtype) * 2.0) - 1.0

    # Hessian-Vector Product operator
    def hvp(vec, create_graph=False):
        return torch.autograd.grad(f_z, z, grad_outputs=vec, retain_graph=True, create_graph=create_graph)[0]

    # Conjugate gradient method
    with torch.no_grad():
        z_star = torch.zeros_like(v)
        r = v.clone()
        p = r.clone()
        rs_old = torch.sum(r * r, dim=1, keepdim=True)

        for _ in range(cg_iters):
            Ap = hvp(p)
            alpha = rs_old / (torch.sum(p * Ap, dim=1, keepdim=True) + 1e-8)
            z_star += alpha * p
            r -= alpha * Ap
            rs_new = torch.sum(r * r, dim=1, keepdim=True)
            if torch.max(rs_new) < 1e-6:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

    return torch.sum(z_star.detach() * hvp(v, create_graph=True), dim=1)


'''
    Part 3: Estimation for the Sinkhorn discrepancy. 
'''
def sinkhorn_loss_with_extension(X_ref, omega_gen, epsilon, max_iter=100):
    # In this algorithm, we extend the discrete potential g to general continuous space.
    # This will not impact the value of the estimator, but ensure it is differentiable.
    device = X_ref.device
    dtype = X_ref.dtype
    n_ref = X_ref.shape[0]
    n_gen = omega_gen.shape[0]

    log_n = torch.log(torch.tensor(float(n_ref), device=device, dtype=dtype))
    log_N = torch.log(torch.tensor(float(n_gen), device=device, dtype=dtype))

    # Update f and g without recording gradients, as we only need the optimal f due to the envelop theorem
    with torch.no_grad():
        # detach the distance computation
        C_detached = 0.5 * torch.cdist(X_ref, omega_gen.detach(), p=2) ** 2
        f = torch.zeros(n_ref, device=device, dtype=dtype)
        # Iteration of the Sinkhorn algorithm
        for _ in range(max_iter):
            # g: [log_N]; f: [n_ref]
            g = -epsilon * torch.logsumexp((f.unsqueeze(1) - C_detached) / epsilon, dim=0) + epsilon * log_n
            f = -epsilon * torch.logsumexp((g.unsqueeze(0) - C_detached) / epsilon, dim=1) + epsilon * log_N
    
    # Distance in computation graph
    C_attached = 0.5 * torch.cdist(X_ref, omega_gen, p=2) ** 2
    
    # Extend g to general space, note that this will not impact the value, but make it differentiable.
    g_continuous = -epsilon * torch.logsumexp((f.unsqueeze(1) - C_attached) / epsilon, dim=0) + epsilon * log_n
    # Return differentiable Sinkhorn discrepancy estimator
    return f.mean() + g_continuous.mean()
