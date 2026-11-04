"""Core k-means implementation in pure Python."""

from segmenta.aliases import Point


def sq_dist(a: Point, b: Point) -> float:
    """Return the squared Euclidean distance between two points.

    Raises:
        ValueError: If the points have different dimensions.
    """
    if len(a) != len(b):
        raise ValueError("points must have the same dimension")
    res = 0.0
    for i in range(len(a)):
        res += (a[i] - b[i]) ** 2
    return res


def mean(points: list[Point]) -> Point:
    """Return the coordinate-wise mean (centroid) of points.

    Raises:
        ValueError: If points is empty.
    """
    if not points:
        raise ValueError("cannot compute mean of empty list")
    dim = len(points[0])
    sums = [0.0] * dim
    for p in points:
        for i in range(dim):
            sums[i] += p[i]
    return [s / len(points) for s in sums]
