from discriminant_analysis import DiscriminantAnalysis
from math_util import ClassLabel, compute_class_covariances
import numpy as np
from warnings import warn


class RDA(DiscriminantAnalysis):
    def __init__(self, lambda_=0.3, gamma=0.2, theoretical=False):
        super().__init__(theoretical=theoretical)
        if not (0 <= lambda_ <= 1):
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}.")
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
        self.lambda_ = lambda_
        self.gamma = gamma
        self.classifier_name = "RDA"
        if self.lambda_ == 0 and self.gamma == 0:
            warn("RDA is equivalent to LDA when lambda=0 and gamma=0. Consider using LDA instead for clarity.")
        elif self.lambda_ == 1 and self.gamma == 0:
            warn("RDA is equivalent to QDA when lambda=1 and gamma=0. Consider using QDA instead for clarity.")

    def _method_applicable(self, X, y):
        # RDA can always be applied because it regularizes
        pass

    def _compute_covariances(self, X, y):
        raw_covs = compute_class_covariances(X, y, self.means_)
        pooled = sum((len(X[y == k]) - 1) * raw_covs[k] for k in np.unique(y)) / (len(X) - len(np.unique(y)))

        self.cov_ = {}
        for k in raw_covs:
            reg_cov = (1 - self.lambda_) * raw_covs[k] + self.lambda_ * pooled
            if self.gamma > 0:
                trace = np.trace(reg_cov)
                d = reg_cov.shape[0]
                reg_cov = (1 - self.gamma) * reg_cov + self.gamma * (trace / d) * np.eye(d)
            self.cov_[k] = reg_cov
        return self.cov_

    def _get_covariance(self, k: ClassLabel):
        return self.cov_[k]


if __name__ == "__main__":
    from test import TestDiscriminantAnalysis, DatasetsEnum
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA

    test_instance_lambda = TestDiscriminantAnalysis(
        RDA,
        SkQDA,
        theoretical=True,
        classifier_kwargs={"lambda_": 0.3, "gamma": 0},
        compare_to_kwargs={"reg_param": 0.3},
        dataset=DatasetsEnum.FISH,
    )

    test_instance_gamma = TestDiscriminantAnalysis(
        RDA,
        SkLDA,
        theoretical=True,
        classifier_kwargs={"lambda_": 0, "gamma": 0.3},
        compare_to_kwargs={"solver": "lsqr", "shrinkage": 0.3},
        dataset=DatasetsEnum.FISH,
    )

    test_instance_gamma_2 = TestDiscriminantAnalysis(
        RDA,
        SkQDA,
        theoretical=True,
        classifier_kwargs={"lambda_": 0, "gamma": 0.30},
        compare_to_kwargs={"reg_param": 0.30},
        dataset=DatasetsEnum.FISH,
    )

    test_instance_lambda.dataset_preview()
    test_instance_lambda.test_classifier()
    test_instance_gamma.test_classifier()
    test_instance_gamma_2.test_classifier()
    print("Well, Sklearn doesnt implement it the same way it seems, life goes onnn")
