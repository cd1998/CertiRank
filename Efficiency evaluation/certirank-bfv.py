"""Command-line entry point for the CertiRank-BFV comparison.

The complete shared protocol implementation is in
``certirank_benchmark.py``. This launcher selects native BFV batching and
standard BFV parameters; it does not invoke SEncode.
"""

from certirank_benchmark import main as run_benchmark


def main() -> None:
    """Run the shared CertiRank workflow with the BFV backend."""

    run_benchmark("bfv")


if __name__ == "__main__":
    main()
