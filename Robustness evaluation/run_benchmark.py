"""Run one CertiRank robustness experiment.

This file is intentionally a stable command-line entry point. Dataset/model
construction, attacks, aggregation, checkpointing, and evaluation live in the
``benchmark`` package, primarily ``benchmark/runner.py``. Keeping the launcher
small allows both direct runs and manifest workers to use the same tested
implementation.
"""

from benchmark.runner import main


if __name__ == "__main__":
    main()
