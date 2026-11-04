"""Core k-means implementation in pure Python."""

import random

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


class KMeans:
    """K-means clustering (Lloyd's algorithm) in pure Python."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        seed: int | None = None,
        tol: float = 1e-6,
    ) -> None:
        """Create a model; see class docstring for the algorithm.

        Raises:
            ValueError: If any parameter is out of range.
        """
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.tol = tol
        self.centroids_: list[Point] = []
        self.labels_: list[int] = []
        self.inertia_history_: list[float] = []

    def _init_centroids(self, points: list[Point]) -> list[Point]:
        """Pick n_clusters distinct random points as initial centroids.

        Raises:
            ValueError: If there are fewer points than clusters.
        """
        if len(points) < self.n_clusters:
            raise ValueError("number of points must be at least n_clusters")
        rng = random.Random(self.seed)
        chosen = rng.sample(points, self.n_clusters)
        return [p[:] for p in chosen]

    def _assign(self, points: list[Point]) -> list[int]:
        """Return the index of the nearest centroid for each point."""
        labels: list[int] = []
        for p in points:
            best_j = 0
            best_d = sq_dist(p, self.centroids_[0])
            for j in range(1, len(self.centroids_)):
                d = sq_dist(p, self.centroids_[j])
                if d < best_d:
                    best_j, best_d = j, d
            labels.append(best_j)
        return labels

    def _update(self, points: list[Point], labels: list[int]) -> list[Point]:
        """Return new centroids as means of their assigned points.

        A cluster that received no points keeps its previous centroid.
        """
        new_centroids: list[Point] = []
        for j in range(self.n_clusters):
            members = [p for p, lab in zip(points, labels, strict=True) if lab == j]
            if members:
                new_centroids.append(mean(members))
            else:
                new_centroids.append(self.centroids_[j][:])
        return new_centroids

    def inertia(self, points: list[Point], labels: list[int]) -> float:
        """Return the sum of squared distances to assigned centroids."""
        total = 0.0
        for p, lab in zip(points, labels, strict=True):
            total += sq_dist(p, self.centroids_[lab])
        return total

    def fit(self, points: list[Point]) -> list[int]:
        """Run Lloyd's iterations until convergence and return labels.

        Iteration stops after max_iter steps or when the largest squared
        shift of any centroid is at most tol. The inertia after every
        update step is stored in inertia_history_.
        """
        self.centroids_ = self._init_centroids(points)
        self.inertia_history_ = []
        for _ in range(self.max_iter):
            labels = self._assign(points)
            new_centroids = self._update(points, labels)
            shift = max(
                sq_dist(old, new)
                for old, new in zip(self.centroids_, new_centroids, strict=True)
            )
            self.centroids_ = new_centroids
            self.inertia_history_.append(self.inertia(points, labels))
            if shift <= self.tol:
                break
        self.labels_ = self._assign(points)
        return self.labels_

    def predict(self, points: list[Point]) -> list[int]:
        """Return the nearest learned centroid for each new point.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self.centroids_:
            raise RuntimeError("model is not fitted; call fit() first")
        return self._assign(points)
