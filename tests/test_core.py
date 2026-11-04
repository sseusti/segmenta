import pytest

from segmenta.core import mean, sq_dist


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
