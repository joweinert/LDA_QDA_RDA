from discriminant_analysis import DiscriminantAnalysis
from math_util import compute_class_covariances
import numpy as np


class QDA(DiscriminantAnalysis):
    def __init__(self, theoretical=False):
        super().__init__(theoretical=theoretical)
        self.classifier_name = "QDA"

    def _method_applicable(self, X, y):
        for cls in np.unique(y):
            n_cls = np.sum(y == cls)
            if X.shape[1] > n_cls:
                raise ValueError(
                    f"QDA cannot be applied: d={X.shape[1]} > n_class={n_cls} for class {cls}. Use LDA or RDA instead."
                )

    def _compute_covariances(self, X, y):
        self.cov_ = compute_class_covariances(X, y, self.means_)
        return self.cov_

    def _get_covariance(self, k):
        return self.cov_[k]


if __name__ == "__main__":
    from test import TestDiscriminantAnalysis, DatasetsEnum
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA

    test_instance = TestDiscriminantAnalysis(QDA, SkQDA, theoretical=True, dataset=DatasetsEnum.FISH)

    test_instance.dataset_preview()
    test_instance.test_classifier()
