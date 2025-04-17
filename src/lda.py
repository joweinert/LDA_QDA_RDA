from discriminant_analysis import DiscriminantAnalysis
from math_util import ClassLabel, compute_class_covariances


class LDA(DiscriminantAnalysis):
    def __init__(self):
        super().__init__()

    def _compute_covariances(self, X, y):
        class_covs = compute_class_covariances(X, y, self.means_)
        weighted_cov_sum = sum((len(X[y == k]) - 1) * class_covs[k] for k in self.classes_)
        self._cov = weighted_cov_sum / (len(X) - len(self.classes_))
        return self._cov

    def _get_covariance(self, k: ClassLabel = None):
        return self._cov  # same for all classes in LDA
