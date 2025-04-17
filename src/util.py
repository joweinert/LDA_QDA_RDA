import numpy as np
import pandas as pd
from enum import Enum
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


class DatasetsEnum(str, Enum):
    """ "Enum for datasets."""

    FISH = "fish"
    IRIS = "iris"
    WINE = "wine"
    CANCER = "cancer"
    DIGITS = "digits"
    SYNTHETIC_BLOBS = "blobs"


class TestData:
    """Class to load datasets."""

    def __init__(
        self,
        dataset: DatasetsEnum = DatasetsEnum.FISH,
        test_size: float = 0.2,
    ):
        self.dataset = dataset
        self.X = None
        self.y = None
        self.load_dataset()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=42,
            stratify=self.y,  # for balanced classes
        )

    def load_dataset(self):
        if self.dataset == DatasetsEnum.FISH:
            X, y = self.load_fish()
        elif self.dataset == DatasetsEnum.IRIS:
            X, y = load_iris(return_X_y=True)
        elif self.dataset == DatasetsEnum.WINE:
            X, y = load_wine(return_X_y=True)
        elif self.dataset == DatasetsEnum.CANCER:
            X, y = load_breast_cancer(return_X_y=True)
        elif self.dataset == DatasetsEnum.DIGITS:
            X, y = load_digits(return_X_y=True)
        else:
            raise NotImplementedError(f"Dataset {self.dataset} is not yet implemented")

        self.X = X
        self.y = y

    def get_data(self):
        return self.X, self.y

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_train_test_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    @staticmethod
    def load_fish():
        """Load fish dataset."""
        df = pd.read_csv("../data/salmon_seabass.csv")
        df = df.drop(columns=["index"], errors="ignore")

        X = df.drop(columns=["species"])
        y = df["species"]

        return np.asarray(X), np.asarray(y)


def check_X_y(X, y):
    """Checks if handed X and y are valid for LDA/QDA/RDA.
    Continous features are expected in X and discrete labels in y.
    The function checks the following:
        1. X is 2D and y is 1D
        2. X and y have the same number of samples
        3. X and y do not contain NaNs or infs
        4. Each class in y has at least 2 samples

    Args:
        X (array-like of shape (n_samples, n_features)): Training data. (E.g. list of lists, numpy array, pandas DataFrame)
        y (array-like of shape (n_samples,)): Class labels. (E.g. list of labels, numpy array, pandas Series)

    Raises:
        ValueError: If any of the above conditions are not met.

    Returns:
        X: numpy.ndarray of shape (n_samples, n_features)
        y: numpy.ndarray of shape (n_samples,)
    """
    try:
        X = np.asarray(X)
        y = np.asarray(y)
    except:
        raise ValueError("X and y must be convertible to numpy arrays")
    # dim checks (1. and 2.)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (got shape {X.shape})")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (got shape {y.shape})")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples (got {X.shape[0]} and {y.shape[0]})")

    # checking nans and infs (3.)
    if not np.isfinite(X).all():
        raise ValueError("X contains NaNs or Infs")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaNs or Infs")

    # checking sample size (4.)
    unique, counts = np.unique(y, return_counts=True)
    for c, n in zip(unique, counts):
        if n < 2:
            raise ValueError(f"Class {c} has only {n} sample(s); need at least 2 to compute covariance")

    return X, y


def check_X(X):
    """Checks if handed X is valid for LDA/QDA/RDA prediction.
    Continous features are expected in X.
    The function checks the following:
        1. X is 2D
        2. X does not contain NaNs or infs

    Args:
        X (array-like of shape (n_samples, n_features)): Training data. (E.g. list of lists, numpy array, pandas DataFrame)

    Raises:
        ValueError: If any of the above conditions are not met.

    Returns:
        X: numpy.ndarray of shape (n_samples, n_features)
    """
    try:
        X = np.asarray(X)
    except:
        raise ValueError("X must be convertible to numpy arrays")

    # dim checks (1.)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (got shape {X.shape})")

    # checking nans and infs (3.)
    if not np.isfinite(X).all():
        raise ValueError("X contains NaNs or Infs")

    return X
