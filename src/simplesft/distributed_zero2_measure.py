"""Torchrun entrypoint shim for ZeRO-2 and ZeRO-3 measurement workers."""

from __future__ import annotations

from .measurement.distributed_zero2 import main


if __name__ == "__main__":
    main()
