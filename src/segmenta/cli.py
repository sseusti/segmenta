"""Command-line interface for segmenta."""

import argparse

from segmenta.core import KMeans
from segmenta.data import make_blobs, make_centers


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the kmeans command."""
    parser = argparse.ArgumentParser(
        prog="kmeans",
        description="Generate synthetic 2D data and cluster it with k-means.",
    )
    parser.add_argument(
        "--samples", type=int, default=300, help="number of points (default: 300)"
    )
    parser.add_argument(
        "--k", type=int, default=3, help="number of clusters to find (default: 3)"
    )
    parser.add_argument(
        "--centers",
        type=int,
        default=3,
        help="number of true centers used to generate data (default: 3)",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=100,
        dest="max_iter",
        help="maximum number of iterations (default: 100)",
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="spread of each blob (default: 1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )
    return parser


def run(
    samples: int,
    k: int,
    centers: int = 3,
    max_iter: int = 100,
    sigma: float = 1.0,
    seed: int | None = None,
) -> str:
    """Generate data, fit k-means and return a text report.

    Raises:
        ValueError: If any parameter is invalid.
    """
    true_centers = make_centers(centers, seed=seed)
    points, _ = make_blobs(samples, true_centers, sigma=sigma, seed=seed)
    model = KMeans(n_clusters=k, max_iter=max_iter, seed=seed)
    labels = model.fit(points)

    lines = [
        f"Clustered {len(points)} points into {k} clusters "
        f"in {len(model.inertia_history_)} iterations"
    ]
    for j, center in enumerate(model.centroids_):
        coords = ", ".join(f"{x:.2f}" for x in center)
        lines.append(f"  cluster {j}: {labels.count(j)} points, center ({coords})")
    lines.append(f"Inertia: {model.inertia(points, labels):.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(
            samples=args.samples,
            k=args.k,
            centers=args.centers,
            max_iter=args.max_iter,
            sigma=args.sigma,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(report)
    return 0
