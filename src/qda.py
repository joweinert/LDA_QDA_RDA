from discriminant_analysis import DiscriminantAnalysis
from math_util import compute_class_covariances


class QDA(DiscriminantAnalysis):
    def __init__(self):
        super().__init__()

    def _compute_covariances(self, X, y):
        self.cov_ = compute_class_covariances(X, y, self.means_)
        return self.cov_

    def _get_covariance(self, k):
        return self.cov_[k]


if __name__ == "__main__":
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA
    from util import DatasetsEnum, TestData
    import numpy as np

    X_train, y_train, X_test, y_test = TestData(DatasetsEnum.FISH).get_train_test_data()
    qda = QDA().fit(X_train, y_train)
    sk_qda = SkQDA().fit(X_train, y_train)
    print("Dataset preview:\nX_test head:\n", X_test[:5])
    print("\ny_test head:\n", y_test[:5])

    qda_preds = qda.predict(X_test)
    sk_qda_preds = sk_qda.predict(X_test)
    qda_probs = qda.predict_proba(X_test)
    sk_qda_probs = sk_qda.predict_proba(X_test)

    print("\nQDA predictions:", qda_preds[:5])
    print("\nSklearn QDA predictions:", sk_qda_preds[:5])
    print("\nSame predictions:", np.all(qda_preds == sk_qda_preds))

    print("\nQDA probabilities:\n", qda_probs[:5])
    print("\nSklearn QDA probabilities:\n", sk_qda_probs[:5])
    print("\nSame probabilities:", np.allclose(qda_probs, sk_qda_probs, atol=1e-6))
