import uuid
import pickle
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd

from lda import LDA
from qda import QDA
from rda import RDA

PICKLE_FILE = "demo_runs.pkl"
PRESETS = [
    # LDA shines: spherical perfect
    {
        "struct": "Spherical, identical Σ",
        "n_per": 100,
        "shift": 3.0,
        "min_scale": 1.0,
        "max_scale": 1.0,
        "n_classes": 3,
        "unbalanced": False,
        "min_per": None,
        "max_per": None,
        "seed": 42,
        "lam": 0.3,
        "gam": 0.2,
        "dim": 2,
        "ill_conditioned": False,
    },
    # LDA decent: elliptical identical
    {
        "struct": "Elliptical, identical Σ",
        "n_per": 100,
        "shift": 2.5,
        "min_scale": 1.0,
        "max_scale": 1.0,
        "n_classes": 3,
        "unbalanced": False,
        "min_per": None,
        "max_per": None,
        "seed": 7,
        "lam": 0.3,
        "gam": 0.3,
        "dim": 2,
        "ill_conditioned": False,
    },
    # QDA strong: different covariances, nice samples
    {
        "struct": "Elliptical, different Σ",
        "n_per": 100,
        "shift": 2.0,
        "min_scale": 2.0,
        "max_scale": 2.0,
        "n_classes": 3,
        "unbalanced": False,
        "min_per": None,
        "max_per": None,
        "seed": 21,
        "lam": 0.2,
        "gam": 0.2,
        "dim": 2,
        "ill_conditioned": False,
    },
    # RDA necessary: different Σ + unbalanced
    {
        "struct": "Elliptical, different Σ",
        "n_per": 60,
        "shift": 2.5,
        "min_scale": 1.5,
        "max_scale": 1.5,
        "n_classes": 4,
        "unbalanced": True,
        "min_per": 30,
        "max_per": 80,
        "seed": 11,
        "lam": 0.5,
        "gam": 0.5,
        "dim": 2,
        "ill_conditioned": False,
    },
    # RDA needed: high-dim identical Σ
    {
        "struct": "Elliptical, identical Σ",
        "n_per": 80,
        "shift": 3.0,
        "min_scale": 1.0,
        "max_scale": 1.0,
        "n_classes": 3,
        "unbalanced": False,
        "min_per": None,
        "max_per": None,
        "seed": 5,
        "lam": 0.4,
        "gam": 0.6,
        "dim": 50,
        "ill_conditioned": False,
    },
    # RDA strong: ill-conditioned Σ
    {
        "struct": "Elliptical, different Σ",
        "n_per": 40,
        "shift": 2.0,
        "min_scale": 1.5,
        "max_scale": 1.5,
        "n_classes": 3,
        "unbalanced": False,
        "min_per": None,
        "max_per": None,
        "seed": 17,
        "lam": 0.6,
        "gam": 0.5,
        "dim": 30,
        "ill_conditioned": True,
    },
    {
        "struct": "Elliptical, different Σ",
        "n_per": 50,
        "shift": 2.3,
        "min_scale": 3.5,
        "max_scale": 3.5,
        "n_classes": 4,
        "unbalanced": False,
        "min_per": None,
        "max_per": None,
        "seed": 0,
        "lam": 0.4,
        "gam": 0.4,
        "dim": 2,
        "ill_conditioned": False,
    },
]


@dataclass
class Run:
    id: str
    config: Dict
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    figs_boundary: Dict[str, Dict[str, plt.Figure]]
    figs_proba: Dict[str, Dict[str, plt.Figure]]
    conf_mats: Dict[str, Dict[str, np.ndarray]]
    metrics: Dict[str, Dict[str, float]]
    fig_data: Dict[str, plt.Figure]
    fitting_errors: Dict[str, str] = None


## Dataset generation
def make_dataset(
    struct: str,
    n_per: int,
    rng: np.random.Generator,
    shift: float,
    min_scale: float,
    max_scale: float,
    n_classes: int = 2,
    outlier_mode: bool = False,
    jitter_mode: bool = False,
    unbalanced: bool = False,
    min_per: Optional[int] = None,
    max_per: Optional[int] = None,
    dim: int = 2,
    ill_conditioned: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset with specified covariance structure.

    Args:
        struct (str): Covariance structure type.
        n_per (int): Number of samples per class.
        rng (np.random.Generator): Random number generator.
        shift (float): Class-mean separation.
        min_scale (float): Covariance scale.
        max_scale (float): Covariance scale. Actual scales is uniformly sampled between min_scale and max_scale.
        n_classes (int, optional): Number of classes. Defaults to 2.
        outlier_mode (bool, optional): If True, add randomness.
        jitter_mode (bool, optional): If True, add noise to class means.
        unbalanced (bool, optional): If True, generate unbalanced classes. Defaults to False.
        min_per (Optional[int], optional): Minimum samples per class if unbalanced. Defaults to None.
        max_per (Optional[int], optional): Maximum samples per class if unbalanced. Defaults to None.
        dim (int, optional): Dimension of the data. Defaults to 2.
        ill_conditioned (bool, optional): If True, generate ill-conditioned data. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: Generated features and labels.
    """
    # determine per-class counts
    if unbalanced and min_per is not None and max_per is not None:
        counts = np.linspace(min_per, max_per, n_classes, dtype=int)
    else:
        counts = np.full(n_classes, n_per, dtype=int)

    # Means #########
    mus = []
    if dim == 2:
        offset = rng.uniform(0, 2 * np.pi)  # rndom angle offset for spice
        angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False) + offset
        if jitter_mode:
            angles += rng.uniform(-0.2, 0.2, n_classes)
        for a in angles:
            mus.append(shift * np.array([np.cos(a), np.sin(a)]))
        if jitter_mode:
            for i in range(len(mus)):
                mus[i] += rng.normal(0, 0.4 * shift, 2)
    else:
        # high‑dimensional, picking a random unit direction per class
        for _ in range(n_classes):
            v = rng.normal(size=dim)
            v /= np.linalg.norm(v)  # unit length
            mus.append(shift * v)

    # Covs #########
    covs = []
    # Pre‑compute one ill‑conditioned matrix if we must keep them identical
    common_cov = None
    local_scale = rng.uniform(min_scale, max_scale)
    if ill_conditioned and "identical Σ" in struct:
        eig = np.logspace(0, -6, dim)
        Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        common_cov = local_scale * Q @ np.diag(eig) @ Q.T

    for k in range(n_classes):
        local_scale = rng.uniform(min_scale, max_scale)
        if ill_conditioned:
            if common_cov is not None:
                cov = common_cov
            else:  # different Σ and ill‑conditioned⇒ generate per class
                eig = np.logspace(0, -6, dim)
                Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
                cov = local_scale * Q @ np.diag(eig) @ Q.T
        else:
            if struct.startswith("Spherical"):
                cov = local_scale * np.eye(dim)
            elif "identical Σ" in struct:
                base = np.eye(dim)
                if dim == 2:
                    base = np.array([[2.0, 1.0], [1.0, 1.5]])
                cov = local_scale * base
            else:  # Elliptical, different Σ without ill‑conditioning
                if dim == 2:
                    cov = local_scale * (
                        np.array([[1.5, 0.8], [0.8, 1.2]]) if k % 2 == 0 else np.array([[1.8, -0.4], [-0.4, 0.7]])
                    )
                else:
                    M = rng.standard_normal((dim, dim))
                    cov = local_scale * (M @ M.T)
        covs.append(cov)

    # Sample points
    X, y = [], []
    for cls, (mu, cov, cnt) in enumerate(zip(mus, covs, counts)):
        X.append(rng.multivariate_normal(mu, cov, cnt))
        y.append(np.full(cnt, cls))

    X = np.vstack(X)
    y = np.concatenate(y)

    if outlier_mode:
        # Add random outliers (2% of total size)
        n_outliers = int(0.02 * X.shape[0])
        if n_outliers > 0:
            spread = np.std(X, axis=0).mean()
            X_out = rng.normal(0, 3 * spread, size=(n_outliers, dim))
            y_out = rng.integers(0, n_classes, size=n_outliers)
            X = np.vstack([X, X_out])
            y = np.concatenate([y, y_out])

        # Add label noise (5% of labels flipped)
        n_flip = int(0.05 * len(y))
        if n_flip > 0:
            idx = rng.choice(len(y), n_flip, replace=False)
            y[idx] = rng.integers(0, n_classes, size=n_flip)

    return X, y, mus, covs


def render_boundaries(X, y, models, cmap, hard_cmap, suffix):
    figs_b = {}
    for name, clf in models.items():
        fig, ax = plt.subplots(figsize=(3, 3))
        DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", cmap=hard_cmap, ax=ax)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=hard_cmap, edgecolor="k", s=20)
        ax.set_title(f"{name} {suffix}")
        ax.set(xticks=[], yticks=[])
        ax.set_aspect("equal")
        figs_b[name] = fig
    return figs_b


def render_probability_maps(X, y, models, cmap, suffix):
    figs_p = {}
    K = len(np.unique(y))
    class_colors = cmap.colors[:K]
    for name, clf in models.items():
        fig, axes = plt.subplots(K, 1, figsize=(4, 2.5 * K), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for cls, ax in enumerate(axes):
            DecisionBoundaryDisplay.from_estimator(
                clf, X, response_method="predict_proba", class_of_interest=cls, cmap="viridis", ax=ax
            )
            ax.scatter(X[y == cls, 0], X[y == cls, 1], c=[class_colors[cls]], edgecolor="k", s=20)
            ax.set_title(f"P(class={cls})", fontsize="small")
            ax.set(xticks=[], yticks=[])
            ax.set_aspect("equal")
        fig.suptitle(f"{name} {suffix}", fontsize="small")
        fig.tight_layout()
        figs_p[name] = fig
    return figs_p


def render_data_distribution(X, y, mus, covs, cmap, suffix):
    K = len(np.unique(y))
    class_colors = cmap.colors[:K]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    ax0 = axs[0]

    # Scatter with ellipses
    ax0.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.3, edgecolor="k", s=15)
    for cls in range(K):
        mu = mus[cls][:2]
        cov = covs[cls][:2, :2]
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)
        ell = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            edgecolor=class_colors[cls],
            facecolor=class_colors[cls],
            alpha=0.2,
        )
        ax0.add_patch(ell)
        ax0.scatter(mu[0], mu[1], c=[class_colors[cls]], marker="X", edgecolor="k", s=80)
    ax0.set_title(f"Ellipses & Samples {suffix}")
    ax0.set_aspect("equal")

    # Histograms
    for i, ax in enumerate(axs[1:]):
        data = [X[y == cls, i] for cls in range(K)]
        ax.hist(data, bins=30, density=True, alpha=0.3, color=class_colors, label=[f"Class {cls}" for cls in range(K)])
        xs = np.linspace(min(d.min() for d in data) - 1, max(d.max() for d in data) + 1, 200)
        for cls in range(K):
            mu = mus[cls][i]
            var = covs[cls][i, i]
            pdf = 1 / np.sqrt(2 * np.pi * var) * np.exp(-((xs - mu) ** 2) / (2 * var))
            ax.plot(xs, pdf)
        ax.set_title(f"Feature {i} PDF {suffix}")
        ax.legend(fontsize="small")

    fig.tight_layout()
    return fig


def _render_split(X, y, models, mus, covs, suffix):
    K = len(np.unique(y))
    cmap = plt.get_cmap("tab10", K)
    hard_cmap = ListedColormap(cmap.colors[:K])

    figs_b = render_boundaries(X, y, models, cmap, hard_cmap, suffix)
    figs_p = render_probability_maps(X, y, models, cmap, suffix)
    figd = render_data_distribution(X, y, mus, covs, cmap, suffix)

    confs = {name: confusion_matrix(y, clf.predict(X)) for name, clf in models.items()}
    metrics = {name: accuracy_score(y, clf.predict(X)) for name, clf in models.items()}

    return figs_b, figs_p, confs, metrics, figd


def render_run(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    gam: float,
    seed: int,
    dim: int,
    mus: list,
    covs: list,
) -> tuple:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    models_full = {"LDA": LDA(), "QDA": QDA(), "RDA": RDA(lambda_=lam, gamma=gam)}
    fitting_errors = {}

    for name, clf in models_full.items():
        try:
            clf.fit(X_tr, y_tr)
        except ValueError as e:
            fitting_errors[name] = str(e)
            models_full[name] = None  # mark that this model failed

    cms_tr, mets_tr, cms_te, mets_te = {}, {}, {}, {}
    for name, clf in models_full.items():
        if clf is not None:
            preds_tr = clf.predict(X_tr)
            preds_te = clf.predict(X_te)
            cms_tr[name] = confusion_matrix(y_tr, preds_tr)
            mets_tr[name] = accuracy_score(y_tr, preds_tr)
            cms_te[name] = confusion_matrix(y_te, preds_te)
            mets_te[name] = accuracy_score(y_te, preds_te)
        else:
            cms_tr[name] = None
            cms_te[name] = None
            mets_tr[name] = None
            mets_te[name] = None

    # Visualization
    if dim > 2:
        pca = PCA(n_components=2, random_state=seed)
        X_tr_vis = pca.fit_transform(X_tr)
        X_te_vis = pca.transform(X_te)
    else:
        X_tr_vis, X_te_vis = X_tr, X_te

    # Fit visual models (for decision boundary plots)
    models_vis = {"LDA": LDA(), "QDA": QDA(), "RDA": RDA(lambda_=lam, gamma=gam)}
    for name, clf in models_vis.items():
        try:
            clf.fit(X_tr_vis, y_tr)
        except Exception:
            models_vis[name] = None  # mark broken

    figs_b_tr, figs_p_tr, _, _, figd_tr = _render_split(X_tr_vis, y_tr, models_vis, mus, covs, "(train)")
    figs_b_te, figs_p_te, _, _, figd_te = _render_split(X_te_vis, y_te, models_vis, mus, covs, "(test)")

    return (
        {"train": figs_b_tr, "test": figs_b_te},
        {"train": figs_p_tr, "test": figs_p_te},
        {"train": cms_tr, "test": cms_te},
        {"train": mets_tr, "test": mets_te},
        {"train": figd_tr, "test": figd_te},
        X_tr,
        y_tr,
        X_te,
        y_te,
        fitting_errors,
    )


def _build_run_header(run: Run, dataset: str, idx: int, n_runs: int) -> str:
    cfg_copy = run.config.copy()
    cfg_copy["n_samp"] = getattr(run, f"X_{dataset}").shape[0]
    if run.config["unbalanced"]:
        cfg_copy["n_samp"] = (
            f"{cfg_copy["n_samp"]} (min_per: {run.config['min_per']}, max_per: {run.config['max_per']})"
        )
        del cfg_copy["min_per"], cfg_copy["max_per"], cfg_copy["unbalanced"]

    if cfg_copy["min_scale"] == cfg_copy["max_scale"]:
        cfg_copy["scale"] = cfg_copy["min_scale"]
        del cfg_copy["min_scale"], cfg_copy["max_scale"]
    header = f"Run {n_runs - idx}: {run.config['struct']} | "
    del cfg_copy["struct"]
    for key, value in cfg_copy.items():
        if isinstance(value, bool):
            header += f"{key} | " if value else ""
        else:
            header += f"{key}={value} | "
    return header[:-3]  # remove last " | "


def init_presets():
    try:
        with open(PICKLE_FILE, "rb") as f:
            st.session_state.runs = pickle.load(f)
    except Exception:
        st.session_state.runs = []
        for cfg in PRESETS:
            rng = np.random.default_rng(cfg["seed"])
            X, y, mus, covs = make_dataset(
                struct=cfg["struct"],
                n_per=cfg["n_per"],
                rng=rng,
                shift=cfg["shift"],
                min_scale=cfg["min_scale"],
                max_scale=cfg["max_scale"],
                n_classes=cfg["n_classes"],
                outlier_mode=cfg["outlier_mode"],
                jitter_mode=cfg["jitter_mode"],
                unbalanced=cfg["unbalanced"],
                min_per=cfg["min_per"],
                max_per=cfg["max_per"],
                dim=cfg["dim"],
                ill_conditioned=cfg["ill_conditioned"],
            )
            figs_b, figs_p, cms, mets, fd, Xtr, ytr, Xte, yte, fitting_errors = render_run(
                X, y, cfg["lam"], cfg["gam"], cfg["seed"], dim=cfg["dim"], mus=mus, covs=covs
            )
            run = Run(
                id=str(uuid.uuid4())[:8],
                config=cfg,
                X_train=Xtr,
                y_train=ytr,
                X_test=Xte,
                y_test=yte,
                figs_boundary=figs_b,
                figs_proba=figs_p,
                conf_mats=cms,
                metrics=mets,
                fig_data=fd,
                fitting_errors=fitting_errors,
            )
            st.session_state.runs.append(run)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(st.session_state.runs, f)


### ON STARTUP --- MAIN ###
st.set_page_config(page_title="LDA/QDA/RDA - Playground", layout="wide")
if "runs" not in st.session_state:
    # first run: load or generate preset runs
    init_presets()
    st.session_state.show_cm = {}

# Sidebar controls
st.sidebar.header("Data generation")
structure = st.sidebar.selectbox(
    "Covariance structure",
    ["Spherical, identical Σ", "Elliptical, identical Σ", "Elliptical, different Σ"],
)
n_classes = st.sidebar.number_input("Number of classes", 2, 10, 2)
unbalanced = st.sidebar.checkbox("Unbalanced classes", False)
min_per = st.sidebar.slider("Min samples per class", 10, 500, 10, 5) if unbalanced else None
max_per = st.sidebar.slider("Max samples per class", 10, 500, 100, 5) if unbalanced else None
n_per = st.sidebar.slider("Samples per class", 10, 500, 100, 10, disabled=unbalanced)
shift = st.sidebar.slider("Class-mean separation", 0.0, 10.0, 2.0, 0.1)
if "identical" in structure:
    min_scale = st.sidebar.slider("Covariance scale", 0.1, 5.0, 1.0, 0.1)
    max_scale = min_scale
else:
    min_scale = st.sidebar.slider("Minimum covariance scale", 0.1, 5.0, 0.5, 0.1)
    if min_scale < 5.0:
        max_scale = st.sidebar.slider("Maximum covariance scale", min_scale, 5.0, min_scale + 0.5, 0.1)
    else:
        max_scale = min_scale
outlier_mode = st.sidebar.checkbox("Add outliers", False)
jitter_mode = st.sidebar.checkbox("Add mean sep noise", False)
if structure != "Spherical, identical Σ":
    ill_conditioned = st.sidebar.checkbox("Ill conditioned covariance", False)
else:
    ill_conditioned = False
dim = st.sidebar.slider("Feature dimension d", 2, 200, 2, step=1)
seed = st.sidebar.number_input("Random seed", 0, step=1)

st.sidebar.markdown("---")
st.sidebar.header("RDA hyper-parameters")
lam = st.sidebar.slider("λ (blend pooled / class Σ)", 0.0, 1.0, 0.3, 0.05)
gam = st.sidebar.slider("γ (diagonal shrinkage)", 0.0, 1.0, 0.2, 0.05)
run_btn = st.sidebar.button("Generate / add run")

# Generate new run
if run_btn:
    status = st.sidebar.empty()
    progress = st.sidebar.progress(0)
    status.text("Sampling…")

    X, y, mus, covs = make_dataset(
        structure,
        n_per,
        np.random.default_rng(int(seed)),
        shift,
        min_scale,
        max_scale,
        n_classes=n_classes,
        outlier_mode=outlier_mode,
        jitter_mode=jitter_mode,
        unbalanced=unbalanced,
        min_per=min_per,
        max_per=max_per,
        dim=dim,
        ill_conditioned=ill_conditioned,
    )
    progress.progress(10)
    status.text("Render models…")

    figs_b, figs_p, cms, mets, fd, Xtr, ytr, Xte, yte, fitting_errors = render_run(
        X, y, lam, gam, int(seed), dim, mus, covs
    )

    progress.progress(100)
    status.text("Done!")
    cfg = {
        "struct": structure,
        "n_per": n_per,
        "shift": shift,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "dim": dim,
        "ill_conditioned": ill_conditioned,
        "outlier_mode": outlier_mode,
        "jitter_mode": jitter_mode,
        "n_classes": n_classes,
        "unbalanced": unbalanced,
        "min_per": min_per,
        "max_per": max_per,
        "seed": int(seed),
        "lam": lam,
        "gam": gam,
    }
    new_run = Run(
        id=str(uuid.uuid4())[:8],
        config=cfg,
        X_train=Xtr,
        y_train=ytr,
        X_test=Xte,
        y_test=yte,
        figs_boundary=figs_b,
        figs_proba=figs_p,
        conf_mats=cms,
        metrics=mets,
        fig_data=fd,
        fitting_errors=fitting_errors,
    )
    st.session_state.runs.insert(0, new_run)
    st.session_state.show_cm[new_run.id] = False

# Display runs
dataset = st.radio("Dataset:", ("train", "test"), horizontal=True)
show_mode = st.radio("Show:", ("Decision boundary", "Probability heatmap", "Data distribution"), horizontal=True)
n_runs = len(st.session_state.runs)
for idx, run in enumerate(st.session_state.runs):
    header = _build_run_header(run, dataset, idx, n_runs)

    with st.expander(header, expanded=(idx == 0)):
        if show_mode == "Data distribution":
            st.pyplot(run.fig_data[dataset], use_container_width=True)
        else:
            cols = st.columns(3)  # Always 3 columns for LDA, QDA, RDA
            for col, model_name in zip(cols, ["LDA", "QDA", "RDA"]):
                if run.fitting_errors.get(model_name):
                    col.warning(f"{model_name} failed:\n{run.fitting_errors[model_name]}")
                else:
                    figs = run.figs_boundary if show_mode == "Decision boundary" else run.figs_proba
                    fig = figs[dataset][model_name]
                    col.pyplot(fig, use_container_width=True)
                    col.caption(f"Acc ({model_name}): {run.metrics[dataset][model_name]:.2f}")

        if st.checkbox("Show confusion matrices", key=f"cm_{run.id}_{dataset}"):
            for name in ["LDA", "QDA", "RDA"]:
                conf = run.conf_mats[dataset][name]  # <<< GET the matrix here
                if conf is not None:
                    st.write(f"**{name}**")
                    classes = np.arange(conf.shape[0])  # assume classes are 0..K-1
                    conf_df = pd.DataFrame(
                        conf, index=[f"True {c}" for c in classes], columns=[f"Pred {c}" for c in classes]
                    )
                    st.dataframe(conf_df)
                else:
                    st.warning(f"No confusion matrix for {name} (model fitting failed).")

    st.markdown("---")
