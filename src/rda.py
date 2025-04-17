import numpy as np
from util import check_X_y
from math_util import compute_class_means, compute_class_priors, compute_class_covariances
import matplotlib.pyplot as plt

class RDA:
    def __init__(self):
        self.lambda_ = 0.5  # Set values later
        self.gamma_ = 0.5   # Set values later
        self.means_ = None
        self.priors_ = None
        self.cov_ = {}
        self.classes_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)
        self.means_ = compute_class_means(X, y)
        self.priors_ = compute_class_priors(y)

        # Compute regularized covariances for RDA
        self.cov_ = self.compute_regularized_covariances(X, y)
        
        return self
    
    def compute_pooled_covariance(X, y):
        classes = np.unique(y)
        covariances = [np.cov(X[y == cls], rowvar=False) for cls in classes]
        pooled_cov = np.sum([cov * len(X[y == cls]) for cov, cls in zip(covariances, classes)], axis=0)
        pooled_cov /= len(X)

        return pooled_cov

    def compute_regularized_covariances(self, X, y):
        class_cov = compute_class_covariances(X, y, self.means_)
        pooled_cov = self.compute_pooled_covariance(X, y)

        regularized_covs = {}
        for c in self.classes_:
            sigma_k_lambda = (1 - self.lambda_) * class_cov[c] + \
            self.lambda_ * pooled_cov

            """
            d (sigma_k_lambda.shape[0]) - number of features, as rows == columns
            in covariance matrix
            """
            regularized_covs[c] = (1 - self.gamma_) * sigma_k_lambda + \
            (self.gamma / sigma_k_lambda.shape[0]) * np.trace(sigma_k_lambda) * np.eye(sigma_k_lambda.shape[0])

        return regularized_covs


