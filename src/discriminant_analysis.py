##### WE AIMED TO MIRROR THE SLIDES AS CLOSE AS POSSIBLE #####


from abc import ABC, abstractmethod
import numpy as np
from math_util import Number, ClassLabel
from math_util import compute_class_means, compute_class_priors
from util import check_X, check_X_y
from sklearn.metrics import confusion_matrix


class DiscriminantAnalysis(ABC):
    """
    Abstract base class for Discriminant Analysis.
    This class provides a template for implementing QDA/LDA/RDA.
    It defines the basic structure and methods that should be implemented by any subclass.
    """

    def __init__(self, theoretical=False):
        """Initialize the DiscriminantAnalysis class.

        Args:
            theoretical (bool, optional): Use theoretical loglik computation (using inverse). Defaults to False.
        """
        self.theoretical = theoretical
        self.classifier_name = None  # name of the classifier (e.g. QDA, LDA, RDA)
        self._estimator_type = "classifier"
        self.means_ = None  # μ_k
        self.priors_ = None  # P(ω_k)
        self.cov_ = None  # either dictionary of Σₖ, Σ (pooled covariance) or regularized Σₖ
        self.classes_ = None  # unique class labels

    @abstractmethod
    def _method_applicable(self, X, y):
        """Abstract method to check if the method is applicable to the data."""
        pass

    @abstractmethod
    def _compute_covariances(self, X, y):
        """Abstract method to compute covariances.
        This method should be implemented in subclasses to compute class-specific covariances.
        """
        pass

    @abstractmethod
    def _get_covariance(self, k: ClassLabel = None) -> np.ndarray:
        """Abstract method to get covariance for class k."""
        pass

    def fit(self, X, y):
        """Fits the model to data.
        Model data using class means, covaiances, and priors. Saves it to the classifier instance -> ready for prediction.

        Args:
            X (array-like of shape (n_samples, n_features)): Training data. (E.g. list of lists, numpy array, pandas DataFrame)
            y (array-like of shape (n_samples,)): Class labels. (E.g. list of labels, numpy array, pandas Series)

        Returns:
            self: QDA | LDA | RDA instance with fitted parameters.
        """
        X, y = check_X_y(X, y)
        self._method_applicable(X, y)

        self.classes_ = np.unique(y)
        self.means_ = compute_class_means(X, y)
        self.priors_ = compute_class_priors(y)
        self._cov = self._compute_covariances(X, y)

        return self

    def _log_likelihood(self, x, k) -> Number:
        """
        Compute log p(x | ω_k).
        Toggled between:
            - Theoretical version (slides): uses explicit inverse and determinant.
            - Numerical version (recommended): uses slogdet and solve.
        """
        mean = self.means_[k]
        cov = self._get_covariance(k)
        diff = x - mean

        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError(
                f"""
                Covariance matrix for class {k} is not positive definite.
                {self.classifier_name} cannot be applied to this data (covariance singular).
                Consider using RDA instead.
                """
            )

        diff = x - mean

        if self.theoretical:
            # Theory mode wiht explicit matrix inverse
            inv_cov = np.linalg.inv(cov)
            mahalanobis = diff.T @ inv_cov @ diff  # (x - μ_k)^T Σ_k^{-1} (x - μ_k)
        else:
            # Numerical mode using solve linear system Σ_k * y = (x-μ)
            # np.linalg.solve(cov, diff) computes Σ_k^{-1} (x - μ_k) by solving the linear system Σ_k * y = (x - μ_k)
            # diff @ np.linalg.solve(cov, diff) computes (x - μ_k)^T Σ_k^{-1} (x - μ_k) which is the squared Mahalanobis distance
            # This is equivalent to using scipy.spatial.distance.mahalanobis(x, mean, cov) or manually computing the equation
            # but avoids computing the inverse of the covariance matrix directly
            # therefore its numerically more stable
            # Internally, it may uses LU.
            mahalanobis = diff @ np.linalg.solve(cov, diff)  # (x - μ_k)^T Σ_k^{-1} (x - μ_k)
            # log P(x | ω_k) = -1/2 * ( ln |Σ_k| + (x - μ_k)^T Σ_k^{-1} (x - μ_k) )

        loglikelihood = -0.5 * (logdet + mahalanobis)
        return loglikelihood
    
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
        all_scores = self._discriminant_scores(X)

        # softmax function: exp(g_k(x)) / SUM_{j}(exp(g_j(x)) for all j)
        exp_scores = np.exp(np.array([[score_k[c] for c in self.classes_] for score_k in all_scores]))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probabilities

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

    def _discriminant_scores(self, X):
        """Compute discriminant scores g_k(x) for each x in X.

        Args:
            X (np.ndarray): array of shape (n_samples, n_features)

        Returns:
            list of dicts: [{class_label: g_k(x)} for each x]
        """
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
        all_scores = self._discriminant_scores(X)

        preds = [max(score_dict, key=score_dict.get) for score_dict in all_scores]
        return np.array(preds)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        X, y = check_X_y(X, y)
        return sum(self.predict(X) == y) / X.shape[0]

    def confusion(self, X, y_true):
        """Returns confusion matrix between true labels and predictions."""
        return confusion_matrix(y_true, self.predict(X))

    def summary(self):
        """Print model parameters for inspection."""
        for k in self.classes_:
            print(f"\nClass {k}:")
            print("  Prior:", self.priors_[k])
            print("  Mean:", self.means_[k])
            print("  Covariance:\n", self._get_covariance(k))

    def plot_proba(self, X, y):
        """Plot class probabilities."""
        from sklearn.inspection import DecisionBoundaryDisplay
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        n_features = X.shape[1]
        if n_features > 2:
            print(f"X has {n_features} features, only two can be used plotting")
            print("In order to be consistent with the class structure, pls:")
            print("Refit the model with only two features")
            print("Use PCA n=2 and refit the model")
            raise ValueError("X has more than 2 features")

        labels = np.unique(y)
        n_targets = len(labels)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_targets,
            figsize=(n_targets * 3, 3),
        )
        for label in labels:
            # plot the probability estimate provided by the classifier
            disp = DecisionBoundaryDisplay.from_estimator(
                self,
                X,
                response_method="predict_proba",
                class_of_interest=label,
                ax=axes[label],
                vmin=0,
                vmax=1,
            )
            axes[label].set_title(f"Class {label}")
            # plot data predicted to belong to given class
            mask_y_true = y == label
            axes[label].scatter(X[mask_y_true, 0], X[mask_y_true, 1], marker="o", c="w", edgecolor="k")
            axes[label].set(xticks=(), yticks=())
        axes[0].set_ylabel(self.classifier_name)

        ax = plt.axes([0.15, 0.04, 0.7, 0.02])
        plt.title("Probability")
        _ = plt.colorbar(cm.ScalarMappable(norm=None, cmap="viridis"), cax=ax, orientation="horizontal")
        plt.suptitle(f"{self.classifier_name} Class Probabilitys")
        plt.show()
