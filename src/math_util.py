import numpy as np


def compute_class_means(X, y):
    """Computes the class means (μ_k) for each class k.
    The means are computed as the average of the samples per claass.

    Args:
        X (np.ndarray): training data (features).
        y (np.ndarray): class labels of the training data.

    Returns:
        means: dict of class means {class_label_k: μ_k} for each class k.
    """
    n_features = X.shape[1]
    means = {k: np.zeros(n_features) for k in np.unique(y)}

    for k in means.keys():
        # means[k] = np.mean(X[y == k], axis=0)
        means[k] = np.sum(X[y == k], axis=0) / np.sum(y == k)
    return means


def compute_class_covariances(X, y, means):
    """Computes the covariance matrices (Σ_k) for each class k.
    The covariance is computed as the average of the outer product of the
    difference between each sample and the class mean.
    The covariance is computed as Σ_k = 1/(n_k - 1) * Σ_{x_i ∈ ω_k} (x_i - μ_k)(x_i - μ_k)^T.

    Args:
        X (np.ndarray): the training data (features).
        y (np.ndarray): the class labels of the training data.
        means (dict): the class means {class_label_k: μ_k} for each class k.

    Returns:
        covariances: dict of class covariances {class_label_k: Σ_k} for each class k.
    """
    
    # init covariance dict
    n_features = X.shape[1]
    covariances = {k: np.zeros((n_features, n_features)) for k in means.keys()}
    # compute covariance per class
    for k in covariances.keys():
        X_k = X[y == k]
        n_k = X_k.shape[0]
        diff = X_k - means[k]
        covariances[k] = (diff.T @ diff) / (n_k - 1)

    return covariances


def compute_class_priors(y):
    """Computes P̂(ω_k)=n_k/n for each class on the training data (the prior).

    Args:
        y (np.ndarray): the class labels of the training data.

    Returns:
        dict: A dictionary {class_label: P̂(ω_k)} for each class.
    """
    unique, counts = np.unique(y, return_counts=True)
    priors = {k: n / len(y) for k, n in zip(unique, counts)}
    return priors