from pathlib import Path

import pytest

from segmenta.cli import main, run


def test_run_report_contents() -> None:
    report = run(samples=60, k=3, seed=0)
    assert "Clustered 60 points into 3 clusters" in report
    assert "cluster 2:" in report
    assert "Inertia:" in report


def test_main_success(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--samples", "90", "--k", "3", "--seed", "42"])
    assert code == 0
    assert "Inertia:" in capsys.readouterr().out


def test_main_is_reproducible(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--seed", "1"])
    first = capsys.readouterr().out
    main(["--seed", "1"])
    assert capsys.readouterr().out == first


@pytest.mark.parametrize(
    "argv",
    [
        ["--k", "0"],
        ["--samples", "0"],
        ["--samples", "2", "--k", "3"],
        ["--sigma", "-1"],
        ["--k", "abc"],
    ],
)
def test_main_bad_args_exit_cleanly(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_main_saves_cluster_plot(tmp_path: Path) -> None:
    out = tmp_path / "clusters.png"
    assert main(["--samples", "60", "--seed", "0", "--plot", str(out)]) == 0
    assert out.exists()


def test_main_elbow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "elbow.png"
    code = main(["--samples", "60", "--seed", "0", "--elbow", str(out), "--k-max", "5"])
    assert code == 0
    assert out.exists()
    stdout = capsys.readouterr().out
    assert "k=1:" in stdout
    assert "k=5:" in stdout


def test_main_elbow_bad_k_max() -> None:
    with pytest.raises(SystemExit):
        main(["--elbow", "x.png", "--k-max", "0"])
