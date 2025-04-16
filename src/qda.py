# qda.py
import numpy as np
from math_util import compute_class_means, compute_class_covariances, compute_class_priors
from util import check_X_y, check_X, plot_qda_pdf_contours
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class QDA:
    def __init__(self):
        self.means_ = None  # μ_k
        self.priors_ = None  # P(ω_k)
        self.cov_ = {}  # dictionary of Σₖ
        self.classes_ = None  # unique class labels

    def fit(self, X, y):
        """Fit QDA model to data.
        Model data using class means, covaiances, and priors. Saves it to the QDA classifier instance -> ready for prediction.

        Args:
            X (array-like of shape (n_samples, n_features)): Training data. (E.g. list of lists, numpy array, pandas DataFrame)
            y (array-like of shape (n_samples,)): Class labels. (E.g. list of labels, numpy array, pandas Series)

        Returns:
            self: QDA instance with fitted parameters.
        """
        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)
        self.means_ = compute_class_means(X, y)
        self.priors_ = compute_class_priors(y)
        self.cov_ = compute_class_covariances(X, y, self.means_)

        return self

    def _log_likelihood(self, x, k):
        mean = self.means_[k]  # μ_k
        cov = self.cov_[k]  # Σ_k

        sign, logdet = np.linalg.slogdet(cov)  # ln |Σ_k|
        if sign <= 0:
            raise ValueError(f"Covariance matrix for class {k} is not positive definite.")

        # Mahalanobis dist
        diff = x - mean  # (x - μ_k)
        # np.linalg.solve(cov, diff) computes Σ_k^{-1} (x - μ_k) by solving the linear system Σ_k * y = (x - μ_k)
        # diff @ np.linalg.solve(cov, diff) computes (x - μ_k)^T Σ_k^{-1} (x - μ_k) which is the squared Mahalanobis distance
        # This is equivalent to using scipy.spatial.distance.mahalanobis(x, mean, cov) or manually computing the equation
        # but avoids computing the inverse of the covariance matrix directly
        # therefore its numerically more stable
        # Internally, it may use SVD, LU, or Cholesky decomposition.
        mahalanobis = diff @ np.linalg.solve(cov, diff)  # (x - μ_k)^T Σ_k^{-1} (x - μ_k)
        # log P(x | ω_k) = -1/2 * ( ln |Σ_k| + (x - μ_k)^T Σ_k^{-1} (x - μ_k) )
        loglikelihood = -0.5 * (logdet + mahalanobis)

        return loglikelihood

    def _discriminant_function(self, x, k):
        """Compute the discriminant function for class k at point x.

        Args:
            x (array-like of shape (n_features,)): Sample point.
            k (int): Class label.

        Returns:
            float: Discriminant function value for class k at point x.
        """
        # putting the pieces together ->> prior + loglik -> Discriminant function
        prior = self.priors_[k]  # P(ω_k)
        # ln P(ω_k) - 1/2 * ( ln |Σ_k| + (x - μ_k)^T Σ_k^{-1} (x - μ_k) )
        gk = np.log(prior) + self._log_likelihood(x, k)  # g_k(x) = ln P(ω_k) + loglikelihood

        return gk

    def _discriminant_scores(self, X, X_checked=False):
        """Compute discriminant scores g_k(x) for each x in X.

        Args:
            X (np.ndarray): array of shape (n_samples, n_features)

        Returns:
            list of dicts: [{class_label: g_k(x)} for each x]
        """
        X = check_X(X) if not X_checked else X
        results = []

        for x in X:
            scores = {}
            for k in self.classes_:
                gk = self._discriminant_function(x, k)
                scores[k] = gk
            results.append(scores)

        return results

    def predict(self, X):
        """Predict class labels for samples in X using the QDA quadratic discriminant function for class ω_k.
        g_k(x) = ln P(ω_k) - 1/2 * ( ln |Σ_k| + (x - μ_k)^T Σ_k^{-1} (x - μ_k) )

        Args:
            X (array-like of shape (n_samples, n_features)): E.g. list of lists, numpy array, pandas DataFrame

        Returns:
            predictions: np.ndarray of shape (n_samples,)
                Predicted class labels for each sample in X.
        """
        X = check_X(X)
        all_scores = self._discriminant_scores(X, X_checked=True)

        preds = [max(score_dict, key=score_dict.get) for score_dict in all_scores]
        return np.array(preds)

    def predict_proba(self, X):
        """Predict class probabilities for samples in X using the QDA discriminant function.
        The probabilities are computed using the softmax function on the discriminant scores.

        Args:
            X (array-like of shape (n_samples, n_features)): E.g. list of lists, numpy array, pandas DataFrame

        Returns:
            probabilities: np.ndarray of shape (n_samples, n_classes)
                Predicted class probabilities for each sample in X.
        """
        X = check_X(X)
        all_scores = self._discriminant_scores(X, X_checked=True)

        # softmax function: exp(g_k(x)) / SUM_{j}(exp(g_j(x)) for all j)
        exp_scores = np.exp(np.array([list(score_k.values()) for score_k in all_scores]))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probabilities

    ############## SOME HELPER FUNCTIONS ##############

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        X, y = check_X_y(X, y)
        return np.mean(self.predict(X) == y)

    def confusion(self, X, y_true):
        """Returns confusion matrix between true labels and predictions."""
        return confusion_matrix(y_true, self.predict(X))

    def summary(self):
        """Print model parameters for inspection."""
        for k in self.classes_:
            print(f"\nClass {k}:")
            print("  Prior:", self.priors_[k])
            print("  Mean:", self.means_[k])
            print("  Covariance:\n", self.cov_[k])

    def plot(self, X, y, title="QDA Decision Boundaries"):
        """Plot the QDA decision boundaries."""
        plot_qda_pdf_contours(self, X, y, title=title)
        plt.show()


if __name__ == "__main__":
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA
    from util import DatasetsEnum, TestData, plot_qda_pdf_contours, plot_qda_decision_surface

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

    plot_qda_pdf_contours(qda, X_train, y_train)
    plot_qda_pdf_contours(qda, X_test, y_test)

    plot_qda_decision_surface(sk_qda, X_train, y_train)
    plot_qda_decision_surface(qda, X_test, y_test)
