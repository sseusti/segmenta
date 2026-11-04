import pytest

from segmenta.core import KMeans, mean, sq_dist
from segmenta.data import make_blobs


def test_sq_dist_known_value() -> None:
    assert sq_dist([0.0, 0.0], [3.0, 4.0]) == 25.0


def test_sq_dist_same_point_is_zero() -> None:
    p = [1.5, -2.0, 7.0]
    assert sq_dist(p, p) == 0.0


def test_sq_dist_symmetric() -> None:
    a, b = [1.0, 2.0], [-3.0, 5.0]
    assert sq_dist(a, b) == sq_dist(b, a)


def test_sq_dist_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        sq_dist([1.0, 2.0], [1.0])


def test_mean_known_value() -> None:
    assert mean([[0.0, 0.0], [2.0, 4.0], [4.0, 2.0]]) == pytest.approx([2.0, 2.0])


def test_mean_single_point() -> None:
    assert mean([[3.0, -1.0]]) == [3.0, -1.0]


def test_mean_empty_raises() -> None:
    with pytest.raises(ValueError):
        mean([])


TRUE_CENTERS = [[0.0, 0.0], [10.0, 10.0], [-10.0, 5.0]]


def _blobs(seed: int = 42) -> list[list[float]]:
    points, _ = make_blobs(300, TRUE_CENTERS, sigma=1.0, seed=seed)
    return points


def test_kmeans_finds_known_centers() -> None:
    model = KMeans(n_clusters=3, seed=0)
    model.fit(_blobs())
    # Cluster indices are arbitrary, so match each true center to the nearest one.
    for true in TRUE_CENTERS:
        nearest = min(sq_dist(true, c) for c in model.centroids_)
        assert nearest < 0.5**2


def test_kmeans_inertia_non_increasing() -> None:
    model = KMeans(n_clusters=3, seed=1)
    model.fit(_blobs())
    history = model.inertia_history_
    assert len(history) >= 1
    assert all(b <= a + 1e-9 for a, b in zip(history, history[1:], strict=False))


def test_kmeans_same_seed_same_result() -> None:
    points = _blobs()
    a, b = KMeans(n_clusters=3, seed=7), KMeans(n_clusters=3, seed=7)
    assert a.fit(points) == b.fit(points)
    assert a.centroids_ == b.centroids_


@pytest.mark.parametrize("k", [1, 2, 4, 5])
def test_kmeans_number_of_clusters(k: int) -> None:
    model = KMeans(n_clusters=k, seed=0)
    labels = model.fit(_blobs())
    assert len(model.centroids_) == k
    assert set(labels) == set(range(k))


def test_predict_matches_fit_labels() -> None:
    points = _blobs()
    model = KMeans(n_clusters=3, seed=0)
    labels = model.fit(points)
    assert model.predict(points) == labels


def test_predict_new_points() -> None:
    model = KMeans(n_clusters=3, seed=0)
    model.fit(_blobs())
    labels = model.predict([[0.1, -0.2], [0.3, 0.1]])
    assert labels[0] == labels[1]


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        KMeans().predict([[0.0, 0.0]])


@pytest.mark.parametrize(
    ("n_clusters", "max_iter", "tol"),
    [(0, 100, 1e-6), (3, 0, 1e-6), (3, 100, -1.0)],
)
def test_invalid_params_raise(n_clusters: int, max_iter: int, tol: float) -> None:
    with pytest.raises(ValueError):
        KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol)


def test_fewer_points_than_clusters_raises() -> None:
    with pytest.raises(ValueError):
        KMeans(n_clusters=3).fit([[0.0, 0.0], [1.0, 1.0]])


def test_empty_cluster_keeps_running() -> None:
    # Two identical points become two centroids; one cluster gets no points.
    points = [[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]]
    model = KMeans(n_clusters=2, seed=0, max_iter=10)
    model.centroids_ = [[0.0, 0.0], [0.0, 0.0]]
    labels = [0, 0, 0]
    new = model._update(points, labels)
    assert new[1] == [0.0, 0.0]
    assert len(model.fit(points)) == 3
