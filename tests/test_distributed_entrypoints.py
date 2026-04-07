"""Regression tests for torchrun worker entrypoint shims."""

from __future__ import annotations

from importlib import import_module


def test_distributed_measure_shim_points_to_ddp_worker() -> None:
    """DDP shim should export the measurement worker main.

    Returns:
        None. The assertion verifies the torchrun import target exists.
    """

    module = import_module("simplesft.distributed_measure")
    from simplesft.measurement.distributed import main as worker_main

    assert module.main is worker_main


def test_distributed_zero2_measure_shim_points_to_zero_worker() -> None:
    """ZeRO shim should export the shared ZeRO worker main.

    Returns:
        None. The assertion verifies the torchrun import target exists.
    """

    module = import_module("simplesft.distributed_zero2_measure")
    from simplesft.measurement.distributed_zero2 import main as worker_main

    assert module.main is worker_main
