"""Synthetic data generation for clustering experiments."""

import random

from segmenta.aliases import Point


def make_centers(
    n_centers: int,
    dim: int = 2,
    spread: float = 10.0,
    seed: int | None = None,
) -> list[Point]:
    """Generate random cluster centers in a hypercube.

    Args:
        n_centers: Number of centers to generate.
        dim: Dimension of each center.
        spread: Coordinates are drawn uniformly from [-spread, spread].
        seed: Random seed for reproducibility.

    Returns:
        List of centers.

    Raises:
        ValueError: If n_centers or dim is not positive.

    """
    if n_centers <= 0:
        raise ValueError("n_center must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    rng = random.Random(seed)
    return [
        [rng.uniform(-spread, spread) for _ in range(dim)] for _ in range(n_centers)
    ]


def make_blobs(
    n_samples: int,
    centers: list[Point],
    sigma: float = 1.0,
    seed: int | None = None,
) -> tuple[list[Point], list[int]]:
    """Generate points scattered around given centers with Gaussian noise.

    Points are assigned to centers in round-robin order, so cluster
    sizes differ by at most one.

    Args:
        n_samples: Total number of points.
        centers: Cluster centers, all of the same dimension.
        sigma: Standard deviation of the noise along each axis.
        seed: Random seed for reproducibility.

    Returns:
        A pair (points, labels) where labels[i] is the index of the
        center that generated points[i].

    Raises:
        ValueError: If arguments are invalid.

    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not centers:
        raise ValueError("centres must not be empty")
    dim = len(centers[0])
    if dim == 0 or any(len(c) != dim for c in centers):
        raise ValueError("all centers must have the same non-zero dimension")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = random.Random(seed)
    points: list[Point] = []
    labels: list[int] = []
    for i in range(n_samples):
        label = i % len(centers)
        points.append([rng.gauss(coord, sigma) for coord in centers[label]])
        labels.append(label)
    return points, labels
