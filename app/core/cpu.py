"""CPU disponible para el proceso.

En un contenedor, `os.cpu_count()` devuelve los núcleos de la máquina, no los
que el contenedor puede usar. ONNX Runtime y OpenCV crean un hilo por núcleo
visto: con 18 núcleos visibles y un límite de 1 vCPU, los 18 hilos compiten
por esa única CPU y el sistema los frena una y otra vez. Medido con este
servicio: 35 s por foto en lugar de 3,7 s.

Aquí se calcula la CPU que el proceso puede usar de verdad, teniendo en
cuenta la afinidad y la cuota del cgroup (Docker, ECS/Fargate, Kubernetes).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_CGROUP_V2 = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def cgroup_cpu_quota(
    v2: Path = _CGROUP_V2,
    v1_quota: Path = _CGROUP_V1_QUOTA,
    v1_period: Path = _CGROUP_V1_PERIOD,
) -> float | None:
    """CPUs que permite la cuota del cgroup, o None si no hay límite."""
    try:
        quota, period = v2.read_text().split()[:2]
        if quota == "max":
            return None
        return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota_us = int(v1_quota.read_text())
        period_us = int(v1_period.read_text())
        return quota_us / period_us if quota_us > 0 and period_us > 0 else None
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=1)
def effective_cpu_count() -> int:
    """Núcleos que el proceso puede aprovechar (como mínimo 1)."""
    if hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:  # macOS
        count = os.cpu_count() or 1
    quota = cgroup_cpu_quota()
    if quota is not None:
        # Hacia abajo: con 1,5 vCPU, un segundo hilo ya se vería frenado.
        count = min(count, int(quota))
    return max(1, count)
