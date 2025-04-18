import numpy as np
from numpy.random import default_rng
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from util import plot_decision_surface
from sklearn.model_selection import KFold
from lda import LDA
from qda import QDA
from rda import RDA

def spherical_identical_sigma(n):
    X, y = make_blobs(n_samples=2*n,
                          n_features=d,
                          centers=[[0]*d, [2,0]+[0]*(d-2)],
                          cluster_std=1.0,
                          random_state=0)
    
    return X, y


def elliptical_identical_sigma(n):
    cov = np.array([[2.0, 1.0],
                [1.0, 1.5]])
    mean1, mean2 = np.zeros(d), np.array([2, 0])
    X1 = rng.multivariate_normal(mean1, cov, n)
    X2 = rng.multivariate_normal(mean2, cov, n)

    return np.vstack([X1, X2]), np.hstack([np.zeros(n), np.ones(n)])


def elliptical_different_sigma(n):
    cov1 = np.array([[1.5,  0.8],
                 [0.8,  1.2]])
    cov2 = np.array([[1.8, -0.4],
                    [-0.4, 0.7]])
    mean1, mean2 = np.zeros(d), np.array([2, 0])
    X1 = rng.multivariate_normal(mean1, cov1, n)
    X2 = rng.multivariate_normal(mean2, cov2, n)

    return np.vstack([X1, X2]), np.hstack([np.zeros(n), np.ones(n)])


#  -  60 dimensions, only 20 samples per class
#  -  Σ1 and Σ2 differ in a *few* eigenvalues
#  -  mean shift just 1.2 units in dim‑0
def high_dimensional():
    rng = np.random.default_rng(0)
    d = 60
    n = 20
    eps = 1e-5

    # ------- class means (small shift along 1st coordinate only) -------
    mu1 = np.zeros(d)
    mu2 = np.zeros(d)
    mu2[0] = 1.8

    # Well‑behaved eigen‑values  (no zeros, not too tiny)
    eig1 = np.linspace(0.5, 2.0, d)
    eig2 = eig1 * rng.uniform(0.7, 1.3, d)

    # random orthogonal basis
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))

    cov1 = Q @ np.diag(eig1) @ Q.T + eps*np.eye(d)   # add ridge
    cov2 = Q @ np.diag(eig2) @ Q.T + eps*np.eye(d)

    X1 = rng.multivariate_normal(mu1, cov1, n)
    X2 = rng.multivariate_normal(mu2, cov2, n)

    return np.vstack([X1, X2]), np.hstack([np.zeros(n), np.ones(n)])


def cv_accuracy(clf, X, y, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    scores = []
    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        clf_ = clf.__class__(*getattr(clf, "init_args", []))
        clf_.fit(Xtr, ytr)
        scores.append((clf_.predict(Xte) == yte).mean())
    return np.mean(scores)


if __name__ == "__main__":
    samples = 250
    d = 2   # Dimensionality
    rng = default_rng(0)    # Repeatability

    # Spherical & identical Σ  (straight‑line Bayes boundary -> LDA)
    X_sph, y_sph = spherical_identical_sigma(samples)

    # Elliptical & identical Σ  (same Σ but correlated -> LDA coping with correlation)
    X_ell, y_ell = elliptical_identical_sigma(samples)

    # Elliptical $ different, class-specific Σ_k (true quadratic boundary -> QDA)
    X_diff, y_diff = elliptical_different_sigma(samples)

    # High-dimensional, ill-conditioned 
    # Unequal variances of d > n, QDA overfits, LDA stable
    # RDA is a tradeoff between them
    X_hi, y_hi = high_dimensional()

    # plt.scatter(X_sph[:,0], X_sph[:,1], c=y_sph, cmap='coolwarm', s=15)
    # plt.title("Spherical & identical Σ example"); plt.show()

    lda = LDA().fit(X_diff, y_diff)
    qda = QDA().fit(X_diff, y_diff)
    rda = RDA().fit(X_diff, y_diff)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    plot_decision_surface(lda, X_diff, y_diff, ax=axes[0], title="LDA")
    plot_decision_surface(qda, X_diff, y_diff, ax=axes[1], title="QDA")
    plot_decision_surface(rda, X_diff, y_diff, ax=axes[2],
                        title=r"RDA  ($\lambda=0.3,\;\gamma=0.2$)")
    fig.tight_layout();  plt.show()

    # gamma = 1e-3 for RDA

    print("Before CV")
    for name, clf in [("LDA", lda), ("QDA", qda), ("RDA", rda)]:
        cv_acc = cv_accuracy(clf, X_diff, y_diff, k=10).mean()
        print(f"{name:>3}: 10‑fold CV accuracy = {cv_acc:.3f}")