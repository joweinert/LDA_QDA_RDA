import numpy as np
import matplotlib.pyplot as plt



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


def plot_decision_surface(model, X, y, ax=None, title=None, h=0.02, alpha=0.30, cmap="coolwarm"):
    """
    Visualise a 2‑D decision surface.

    Parameters
    ----------
    model  : fitted object with a .predict(X) method
    X, y   : ndarray, shape (n_samples, 2)   training data
    ax     : existing matplotlib Axes or None (creates new figure)
    title  : str  title for the plot
    h      : float grid step in the same units as X
    alpha  : float fill transparency for regions
    cmap   : matplotlib colormap name
    """
    if X.shape[1] != 2:
        raise ValueError("X must be 2‑D for boundary plotting.")

    # 1 ─── mesh grid covering the data range
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 2 ─── model predictions on the grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    # 3 ─── draw filled contours + scatter of training points
    if ax is None:
        fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=alpha, cmap=cmap, levels=np.arange(Z.max() + 2) - 0.5)  # one colour per class
    scat = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor="k", s=25)

    # cosmetics
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title or model.__class__.__name__)
    ax.set_aspect("equal", adjustable="box")
    return ax