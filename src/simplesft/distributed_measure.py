"""Torchrun entrypoint shim for DDP measurement workers."""

from __future__ import annotations

from .measurement.distributed import main


if __name__ == "__main__":
    main()
