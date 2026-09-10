"""Command-line entry point for the CertiRank efficiency benchmark.

The complete shared protocol implementation is in
``certirank_benchmark.py``. Keeping the two launchers small ensures that
CertiRank and CertiRank-BFV execute exactly the same protocol steps and differ
only in their encoding/encryption backend.
"""

from certirank_benchmark import main as run_benchmark


def main() -> None:
    """Run CertiRank with SEncode and the RLWE-AHE parameterization."""

    run_benchmark("rlwe-ahe")


if __name__ == "__main__":
    main()
