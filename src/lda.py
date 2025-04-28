from discriminant_analysis import DiscriminantAnalysis
from math_util import ClassLabel, compute_class_covariances, Number
from typing import override
import numpy as np


class LDA(DiscriminantAnalysis):
    def __init__(self, theoretical=False):
        super().__init__(theoretical=theoretical)
        self.classifier_name = "LDA"

    def _method_applicable(self, X, y):
        if X.shape[1] > X.shape[0]:
            raise ValueError(f"LDA cannot be applied: d={X.shape[1]} > n={X.shape[0]}. Use RDA instead.")

    def _compute_covariances(self, X, y):
        class_covs = compute_class_covariances(X, y, self.means_)
        weighted_cov_sum = sum((len(X[y == k]) - 1) * class_covs[k] for k in self.classes_)
        self._cov = weighted_cov_sum / (len(X) - len(self.classes_))
        return self._cov

    def _get_covariance(self, k: ClassLabel = None):
        return self._cov  # same for all classes in LDA

    @override
    def _log_likelihood(self, x, k) -> Number:
        """
        Compute log p(x | ω_k) for the LDA model.
        Toggled between:
            - Theoretical version (slides): uses explicit matrix inverse Σ^{-1}.
            - Numerical version (recommended): uses solve for Σ^{-1}μ_k and Σ^{-1}x.
        """
        mean = self.means_[k]
        cov = self._get_covariance()  # pooled covariance (shared across all classes)

        if self.theoretical:
            # Theory mode with explicit matrix inverse
            inv_cov = np.linalg.inv(cov)
            # β_k = Σ^{-1} μ_k
            beta = inv_cov @ mean
            # β_k0 = -1/2 μ_k^T Σ^{-1} μ_k
            beta0 = -0.5 * mean.T @ inv_cov @ mean
        else:
            # Numerical mode using solve linear system Σ * y = μ_k and Σ * y = x
            # np.linalg.solve(cov, mean) computes Σ^{-1} μ_k by solving Σ * y = mean
            # np.linalg.solve(cov, x) computes Σ^{-1} x by solving Σ * y = x
            # Avoids explicit inversion and is more numerically stable
            # Internally may use LU decomposition
            beta = np.linalg.solve(cov, mean)
            beta0 = -0.5 * mean.T @ np.linalg.solve(cov, mean)

        # log p(x | ω_k) = β_k^T x + β_k0
        loglikelihood = beta.T @ x + beta0

        return loglikelihood

if __name__ == "__main__":
    from test import TestDiscriminantAnalysis, DatasetsEnum
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA

    test_instance = TestDiscriminantAnalysis(LDA, SkLDA, theoretical=True, dataset=DatasetsEnum.FISH)

    test_instance.dataset_preview()
    test_instance.test_classifier()
