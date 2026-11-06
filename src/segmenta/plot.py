"""Matplotlib visualizations of clustering results.

The object-oriented Figure API is used instead of pyplot, so no GUI
backend is needed and plots can be saved in CI or over SSH.
"""

from pathlib import Path

from matplotlib.figure import Figure

from segmenta.aliases import Point


def _save(fig: Figure, path: str | Path) -> None:
    """Save a figure, creating parent directories if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)


def plot_clusters(
    points: list[Point],
    labels: list[int],
    centroids: list[Point],
    path: str | Path,
) -> None:
    """Save a scatter plot of 2D points colored by cluster with centroids."""
    fig = Figure(figsize=(7, 6), dpi=120)
    ax = fig.add_subplot()
    ax.scatter(
        [p[0] for p in points],
        [p[1] for p in points],
        c=labels,
        cmap="tab10",
        vmin=0,
        vmax=9,
        s=14,
        alpha=0.7,
    )
    ax.scatter(
        [c[0] for c in centroids],
        [c[1] for c in centroids],
        c="black",
        marker="X",
        s=200,
        label="centroids",
    )
    ax.set_title(f"k-means: {len(centroids)} clusters")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    _save(fig, path)


def plot_elbow(k_values: list[int], inertias: list[float], path: str | Path) -> None:
    """Save a line plot of inertia versus number of clusters."""
    fig = Figure(figsize=(7, 4.5), dpi=120)
    ax = fig.add_subplot()
    ax.plot(k_values, inertias, marker="o")
    ax.set_title("Elbow method")
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("inertia")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.3)
    _save(fig, path)
