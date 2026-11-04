import math

import pytest

from segmenta.data import make_blobs, make_centers

CENTERS = [[0.0, 0.0], [10.0, 10.0], [-10.0, 5.0]]


def test_blobs_shapes() -> None:
    points, labels = make_blobs(30, CENTERS, seed=1)
    assert len(points) == len(labels) == 30
    assert all(len(p) == 2 for p in points)
    assert set(labels) == {0, 1, 2}


def test_blobs_reproducible() -> None:
    assert make_blobs(50, CENTERS, seed=42) == make_blobs(50, CENTERS, seed=42)
    assert make_blobs(50, CENTERS, seed=1) != make_blobs(50, CENTERS, seed=2)


def test_blobs_balanced() -> None:
    _, labels = make_blobs(10, CENTERS, seed=0)
    sizes = [labels.count(k) for k in range(3)]
    assert max(sizes) - min(sizes) <= 1


def test_blobs_near_centers() -> None:
    points, labels = make_blobs(300, CENTERS, sigma=0.5, seed=7)
    for k, center in enumerate(CENTERS):
        cluster = [p for p, lab in zip(points, labels, strict=True) if lab == k]
        mean = [sum(coord) / len(cluster) for coord in zip(*cluster, strict=True)]
        assert math.dist(mean, center) < 0.3


def test_zero_sigma_gives_exact_centers() -> None:
    points, labels = make_blobs(6, CENTERS, sigma=0.0, seed=0)
    assert all(p == CENTERS[lab] for p, lab in zip(points, labels, strict=True))


@pytest.mark.parametrize(
    ("n_samples", "centers", "sigma"),
    [
        (0, CENTERS, 1.0),
        (10, [], 1.0),
        (10, [[0.0, 0.0], [1.0]], 1.0),
        (10, CENTERS, -1.0),
    ],
)
def test_blobs_invalid_args(
    n_samples: int, centers: list[list[float]], sigma: float
) -> None:
    with pytest.raises(ValueError):
        make_blobs(n_samples, centers, sigma)


def test_centers_shape_and_range() -> None:
    centers = make_centers(4, dim=3, spread=5.0, seed=0)
    assert len(centers) == 4
    assert all(len(c) == 3 for c in centers)
    assert all(-5.0 <= x <= 5.0 for c in centers for x in c)


def test_centers_invalid() -> None:
    with pytest.raises(ValueError):
        make_centers(0)
