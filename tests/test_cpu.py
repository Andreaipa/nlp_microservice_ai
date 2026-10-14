"""Pruebas del cálculo de la CPU disponible dentro de un contenedor."""
from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.core.cpu import cgroup_cpu_quota, effective_cpu_count


def _files(tmp_path: Path, v2: str | None = None, v1: tuple[str, str] | None = None):
    paths = {"v2": tmp_path / "cpu.max", "v1_quota": tmp_path / "quota",
             "v1_period": tmp_path / "period"}
    if v2 is not None:
        paths["v2"].write_text(v2)
    if v1 is not None:
        paths["v1_quota"].write_text(v1[0])
        paths["v1_period"].write_text(v1[1])
    return paths


def test_cgroup_v2_quota(tmp_path: Path):
    # Docker --cpus 1 / Fargate 1 vCPU.
    assert cgroup_cpu_quota(**_files(tmp_path, v2="100000 100000\n")) == 1.0
    assert cgroup_cpu_quota(**_files(tmp_path, v2="50000 100000\n")) == 0.5


def test_cgroup_v2_without_limit(tmp_path: Path):
    assert cgroup_cpu_quota(**_files(tmp_path, v2="max 100000\n")) is None


def test_cgroup_v1_quota(tmp_path: Path):
    assert cgroup_cpu_quota(**_files(tmp_path, v1=("200000", "100000"))) == 2.0
    assert cgroup_cpu_quota(**_files(tmp_path, v1=("-1", "100000"))) is None


def test_no_cgroup_files(tmp_path: Path):
    assert cgroup_cpu_quota(**_files(tmp_path)) is None


def test_effective_cpu_count_is_at_least_one():
    assert effective_cpu_count() >= 1


def test_inference_threads_override():
    assert Settings(_env_file=None, ocr_threads=3).inference_threads == 3
    assert Settings(_env_file=None, ocr_threads=0).inference_threads == effective_cpu_count()
