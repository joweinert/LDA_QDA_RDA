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
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA
    from util import DatasetsEnum, TestData
    import numpy as np

    X_train, y_train, X_test, y_test = TestData(DatasetsEnum.FISH).get_train_test_data()
    rda = RDA(lambda_=0.6, gamma=0.3).fit(X_train, y_train)
    print("Dataset preview:\nX_test head:\n", X_test[:5])
    print("\ny_test head:\n", y_test[:5])

    rda_preds = rda.predict(X_test)
    rda_probs = rda.predict_proba(X_test)

    print("\nRDA predictions:", rda_preds[:5])

    print("\nRDA probabilities:\n", rda_probs[:5])
