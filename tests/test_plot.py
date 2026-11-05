from pathlib import Path

from segmenta.core import KMeans
from segmenta.data import make_blobs
from segmenta.plot import plot_clusters, plot_elbow

PNG_SIGNATURE = b"\x89PNG"


def test_plot_clusters_writes_png(tmp_path: Path) -> None:
    points, _ = make_blobs(60, [[0.0, 0.0], [5.0, 5.0]], seed=0)
    model = KMeans(n_clusters=2, seed=0)
    labels = model.fit(points)
    out = tmp_path / "clusters.png"
    plot_clusters(points, labels, model.centroids_, out)
    assert out.read_bytes().startswith(PNG_SIGNATURE)


def test_plot_elbow_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "elbow.png"
    plot_elbow([1, 2, 3], [100.0, 40.0, 30.0], out)
    assert out.read_bytes().startswith(PNG_SIGNATURE)
