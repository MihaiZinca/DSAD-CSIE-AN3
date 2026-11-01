import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def nan_replace_df(X: pd.DataFrame):
    """Inlocuieste valorile lipsa cu media coloanei."""
    x_ = X.copy()
    x_ = x_.fillna(x_.mean())
    return x_


def plot_variance(eigenvalues: np.ndarray):
    """Plot varianta (scree plot) cu pragul criteriului Kaiser."""
    labels = ["C" + str(i + 1) for i in range(len(eigenvalues))]
    f = plt.figure(figsize=(9, 6))
    ax = f.add_subplot(1, 1, 1)

    ax.plot(labels, eigenvalues)
    ax.scatter(labels, eigenvalues, c="r", alpha=0.5)
    ax.axhline(1, c="g", label="Criteriul Kaiser")
    ax.set_title("Plot varianta", color="b", fontsize=16)
    ax.set_xlabel("Componenta", fontsize=12)
    ax.set_ylabel("Varianta", fontsize=12)
    ax.legend()

    plt.show()


def plot_correlation_circle(r_xc: pd.DataFrame, k1: str = "C1", k2: str = "C2"):
    """Cercul corelatiilor: fiecare variabila este punctul (r_xc[k1], r_xc[k2])."""
    f = plt.figure(figsize=(7, 7))
    ax = f.add_subplot(1, 1, 1)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), c="b")
    ax.axhline(0, c="gray")
    ax.axvline(0, c="gray")
    ax.scatter(r_xc[k1], r_xc[k2], c="r")
    for variable in r_xc.index:
        ax.annotate(variable, (r_xc.loc[variable, k1], r_xc.loc[variable, k2]))

    ax.set_title("Cercul corelatiilor (" + k1 + ", " + k2 + ")", color="b", fontsize=16)
    ax.set_xlabel(k1, fontsize=12)
    ax.set_ylabel(k2, fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect(1)

    plt.show()


def plot_observations(C: pd.DataFrame, labels: pd.Series, k1: str = "C1", k2: str = "C2"):
    """Observatiile in spatiul componentelor k1 si k2, etichetate cu labels."""
    f = plt.figure(figsize=(11, 8))
    ax = f.add_subplot(1, 1, 1)

    ax.axhline(0, c="gray")
    ax.axvline(0, c="gray")
    ax.scatter(C[k1], C[k2], c="r", alpha=0.5)
    for code in C.index:
        ax.annotate(labels[code], (C.loc[code, k1], C.loc[code, k2]), fontsize=8)

    ax.set_title("Plot observatii (" + k1 + ", " + k2 + ")", color="b", fontsize=16)
    ax.set_xlabel(k1, fontsize=12)
    ax.set_ylabel(k2, fontsize=12)

    plt.show()


def plot_corr_heatmap(
    corr: pd.DataFrame,
    title="Correlation matrix",
    annot=True,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    figsize=None,
    ax=None,
):
    n_rows, n_cols = corr.shape
    if figsize is None:
        figsize = (max(6, 0.7 * n_cols + 2), max(5, 0.7 * n_rows + 1))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(corr.values, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n_cols), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows), labels=corr.index)
    ax.set_title(title)

    if annot:
        for i in range(n_rows):
            for j in range(n_cols):
                value = corr.iat[i, j]
                color = (
                    "white" if abs(value) > 0.6 * max(abs(vmin), abs(vmax)) else "black"
                )
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

    fig.tight_layout()
    plt.show()
    return ax
