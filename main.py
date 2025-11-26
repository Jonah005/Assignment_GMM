import numpy as np
import pandas as pd


def gaussian_pdf(X, mu, Sigma):

    X = np.atleast_2d(X)
    D = X.shape[1]

    Sigma_reg = Sigma + 1e-6 * np.eye(D)
    inv_Sigma = np.linalg.inv(Sigma_reg)
    det_Sigma = np.linalg.det(Sigma_reg)

    diff = X - mu
    exponent = -0.5 * np.sum(diff @ inv_Sigma * diff, axis=1)
    norm_const = np.sqrt((2.0 * np.pi) ** D * det_Sigma)

    return np.exp(exponent) / norm_const


def init_double_helix_params(X, K=48):

    N, D = X.shape
    assert D == 3
    K_per = K // 2


    radii = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    R_s = radii.mean()                 # staircase radius
    z_min = X[:, 2].min()
    z_max = X[:, 2].max()
    H = z_max - z_min                  # approximate total height


    a = H / (4.0 * np.pi)

    theta_step = 4.0 * np.pi / K_per
    thetas = (np.arange(K_per) + 0.5) * theta_step

    means = np.zeros((K, 3))
    staircase_ids = np.zeros(K, dtype=int)

    for idx, th in enumerate(thetas):
        z = z_min + a * th

        # Staircase 1
        means[idx] = np.array([R_s * np.cos(th),
                               R_s * np.sin(th),
                               z])
        staircase_ids[idx] = 1

        # Staircase 2 (phase shift π in x-y plane)
        k2 = K_per + idx
        means[k2] = np.array([R_s * np.cos(th + np.pi),
                              R_s * np.sin(th + np.pi),
                              z])
        staircase_ids[k2] = 2

    # Local variances (in tangent / radial / binormal frame)
    sigma_t = 3.0  # along stairs
    sigma_r = 1.0  # across stair width
    sigma_b = 1.0
    Lambda = np.diag([sigma_t ** 2, sigma_r ** 2, sigma_b ** 2])

    covs = np.zeros((K, 3, 3))

    def rotation_matrix(theta, staircase):

        if staircase == 1:
            cos_th, sin_th = np.cos(theta), np.sin(theta)
        else:
            # staircase 2 is rotated by π in x-y plane
            cos_th, sin_th = np.cos(theta + np.pi), np.sin(theta + np.pi)

        # Tangent vector (derivative of helix wrt theta)
        t = np.array([-R_s * sin_th, R_s * cos_th, a])
        t /= np.linalg.norm(t)

        # Radial vector (from axis outward)
        r = np.array([cos_th, sin_th, 0.0])
        r /= np.linalg.norm(r)

        # Binormal completes an orthonormal basis
        b = np.cross(t, r)
        b /= np.linalg.norm(b)

        return np.stack([t, r, b], axis=1)

    # Initial covariances: same local Λ, rotated according to R(theta)
    for k in range(K):
        th = thetas[k % K_per]
        staircase = staircase_ids[k]
        Rk = rotation_matrix(th, staircase)
        covs[k] = Rk @ Lambda @ Rk.T

    # Equal mixture weights initially
    weights = np.full(K, 1.0 / K)

    return weights, means, covs, staircase_ids, R_s, z_min, a, Lambda




def project_mean_to_helix(mu_star, staircase, R_s, z_min, a):
    
    x, y, z = mu_star
    phi = np.arctan2(y, x)  # angle in x-y plane

    if staircase == 1:
        theta = phi
    else:

        theta = phi - np.pi

    # Wrap theta into [0, 4π) to keep it stable numerically
    theta = (theta + 8.0 * np.pi) % (4.0 * np.pi)

    z_new = z_min + a * theta
    if staircase == 1:
        x_new = R_s * np.cos(theta)
        y_new = R_s * np.sin(theta)
    else:
        x_new = R_s * np.cos(theta + np.pi)
        y_new = R_s * np.sin(theta + np.pi)

    return np.array([x_new, y_new, z_new]), theta




def constrained_em_double_helix(X, K=48, max_iter=100, tol=1e-4, verbose=True):

    N, D = X.shape
    weights, means, covs, staircase_ids, R_s, z_min, a, Lambda = \
        init_double_helix_params(X, K)

    def rotation_matrix(theta, staircase):
        if staircase == 1:
            cos_th, sin_th = np.cos(theta), np.sin(theta)
        else:
            cos_th, sin_th = np.cos(theta + np.pi), np.sin(theta + np.pi)
        t = np.array([-R_s * sin_th, R_s * cos_th, a])
        t /= np.linalg.norm(t)
        r = np.array([cos_th, sin_th, 0.0])
        r /= np.linalg.norm(r)
        b = np.cross(t, r)
        b /= np.linalg.norm(b)
        return np.stack([t, r, b], axis=1)

    def compute_log_likelihood():
        pdf_vals = np.zeros((N, K))
        for k in range(K):
            pdf_vals[:, k] = weights[k] * gaussian_pdf(X, means[k], covs[k])
        px = pdf_vals.sum(axis=1) + 1e-12
        return np.sum(np.log(px))

    # Initial log-likelihood under geometric initialisation
    prev_ll = compute_log_likelihood()
    if verbose:
        print(f"Initial log-likelihood: {prev_ll:.3f}")

    for it in range(max_iter):

        # ------------------ E-step ------------------
        resp = np.zeros((N, K))
        for k in range(K):
            resp[:, k] = weights[k] * gaussian_pdf(X, means[k], covs[k])
        resp_sum = resp.sum(axis=1, keepdims=True) + 1e-12
        resp /= resp_sum                    # γ_ik
        N_k = resp.sum(axis=0)              # effective counts

        # ------------------ M-step (unconstrained) ------------------
        weights_new = N_k / N
        means_star = np.zeros_like(means)
        covs_star = np.zeros_like(covs)

        for k in range(K):
            if N_k[k] < 1e-8:
                # Avoid numerical issues for empty components
                means_star[k] = means[k]
                covs_star[k] = covs[k]
                continue

            # Mean
            means_star[k] = (resp[:, k][:, None] * X).sum(axis=0) / N_k[k]

            # Covariance
            diff = X - means_star[k]
            covs_star[k] = (resp[:, k][:, None] * diff).T @ diff / N_k[k]
            covs_star[k] += 1e-6 * np.eye(D)

        # ------------------ Apply staircase constraints ------------------
        means_new = np.zeros_like(means)
        covs_new = np.zeros_like(covs)

        for k in range(K):
            staircase = staircase_ids[k]
            mu_proj, theta = project_mean_to_helix(
                means_star[k], staircase, R_s, z_min, a
            )
            means_new[k] = mu_proj

            Rk = rotation_matrix(theta, staircase)
            covs_new[k] = Rk @ Lambda @ Rk.T  # helix-aligned covariance

        # Update parameters
        means = means_new
        covs = covs_new
        weights = weights_new

        # ------------------ Check log-likelihood ------------------
        ll = compute_log_likelihood()
        if verbose:
            print(f"Iter {it + 1:3d}: log-likelihood = {ll:.3f}")

        if abs(ll - prev_ll) < tol:
            if verbose:
                print("Converged.")
            break
        prev_ll = ll

    return weights, means, covs, resp




if __name__ == "__main__":
    # Load the 1,000-point synthetic dataset
    df = pd.read_csv("double_helix_data_1000.csv")
    X = df[["x", "y", "z"]].values

    # Run constrained EM
    K = 48
    weights, means, covs, resp = constrained_em_double_helix(
        X,
        K=K,
        max_iter=100,
        tol=1e-3,
        verbose=True,
    )

    # Example: print first few means and mixture weights
    print("\nFinal mixture weights p_k (first 10):")
    print(weights[:10])

    print("\nFinal component means μ_k (first 5):")
    print(means[:5])
