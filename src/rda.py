from discriminant_analysis import DiscriminantAnalysis
from math_util import ClassLabel, compute_class_covariances
import numpy as np


class RDA(DiscriminantAnalysis):
    def __init__(self, lambda_ = 0.3, gamma = 0.2):
        super().__init__()
        self.lambda_ = lambda_
        self.gamma = gamma

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
