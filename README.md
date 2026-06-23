# SecureFRL

This repository contains the core implementation and efficiency-evaluation
artifact for **SecureFRL: Efficient Privacy-Preserving Byzantine-Robust
Federated Learning via Joint Training and Encryption Adaptation**.

SecureFRL is a privacy-preserving Byzantine-robust federated learning scheme.
It replaces real-valued encrypted model updates with integer ranking vectors,
then performs verification and aggregation through lightweight additive
homomorphic operations. The artifact includes a customized
Microsoft SEAL/SEAL-Python stack and scripts for evaluating cryptographic and
secure-aggregation overhead.

## Artifact Scope

This repository provides:

- a customized SEAL/SEAL-Python binding for SecureFRL;
- primitive benchmarks for encoding, encryption, addition, decryption,
  decoding, and rotation;
- aggregation-efficiency scripts for SecureFRL and secure baselines.

## Repository Structure

```text
SecureFRL/
|-- README.md
|-- Efficiency evaluation/
|   |-- securefrl.py              # SecureFRL aggregation-efficiency script
|   |-- securefrl-bfv.py          # SecureFRL with standard BFV
|   |-- bcpbfl.py                 # CKKS-based BCPBFL-style baseline
|   |-- rvfl.py                   # Paillier/HEU-based RVPFL-style baseline
|   |-- paillier_test.py          # Paillier primitive benchmark
|   |-- comparison.py             # RLWE-AHE/BGV/BFV/CKKS primitive benchmark
|   |-- rotation.py               # Rotation benchmark and plot generation
|   |-- cleaned_results_*.csv     # Cleaned rotation results
|   `-- seal.cpython-310-x86_64-linux-gnu.so
`-- SEAL-Python/
    |-- SEAL/                     # Customized Microsoft SEAL source tree
    |-- pybind11/
    |-- src/wrapper.cpp           # Python binding definitions
    |-- setup.py
    `-- Dockerfile
```

Key implementation files:

- `SEAL-Python/SEAL/native/src/seal/encryptionparams.h`: adds
  `encoding_method`;
- `SEAL-Python/SEAL/native/src/seal/batchencoder.cpp`: implements the custom
  `encoding_method.pm` path;
- `SEAL-Python/src/wrapper.cpp`: exposes the customized SEAL APIs to Python.

## Requirements

The paper experiments were run on Ubuntu 22.04.1 LTS with an Intel Xeon
Platinum 8481C CPU, an NVIDIA RTX 4090 GPU, and 90 GB RAM.

For the efficiency artifact, a Linux x86_64 environment is recommended because
the repository includes a prebuilt Python 3.10 Linux shared object for `seal`.
The GPU is not required for the provided efficiency scripts.

Required software:

- Python 3.10 recommended;
- CMake >= 3.16;
- GCC/G++ >= 9.4 or Clang++ >= 10.0;
- Python packages: `numpy`, `tqdm`, `matplotlib`, `psutil`, `scipy`;
- optional for Paillier/RVPFL baselines: SecretFlow HEU, importable as
  `from heu import phe`;
- optional for full FL training experiments: PyTorch.

Install common Python dependencies:

```bash
python3 -m pip install -U pip
python3 -m pip install numpy tqdm matplotlib psutil scipy
```

## Setup

### Use the bundled binding

On Linux x86_64 with Python 3.10:

```bash
cd "Efficiency evaluation"
python3 -c "from seal import *; print('SEAL-Python is ready')"
```

### Build SEAL-Python from source

Use this if the bundled `seal*.so` is incompatible with your environment.

```bash
cd SEAL-Python
python3 -m pip install numpy pybind11 wheel setuptools

cd SEAL
cmake -S . -B build \
  -DSEAL_USE_MSGSL=OFF \
  -DSEAL_USE_ZLIB=OFF \
  -DSEAL_USE_ZSTD=OFF
cmake --build build -j

cd ..
python3 setup.py build_ext -i
cp seal*.so "../Efficiency evaluation/"
```

## Running Benchmarks

Run scripts from `Efficiency evaluation/` so that Python can import the local
`seal` module.

```bash
cd "Efficiency evaluation"
```

| Script | Purpose |
| --- | --- |
| `python3 comparison.py` | Primitive benchmark for RLWE-AHE, BGV, BFV, and CKKS |
| `python3 rotation.py` | Rotation benchmark; writes `rotation_time_*.png/.pdf` and `cleaned_results_*.csv` |
| `python3 securefrl.py` | SecureFRL aggregation-efficiency benchmark |
| `python3 securefrl-bfv.py` | SecureFRL variant with standard BFV |
| `python3 bcpbfl.py` | CKKS-based BCPBFL-style baseline |
| `python3 paillier_test.py` | Paillier primitive benchmark, requires HEU |
| `python3 rvfl.py` | Paillier/HEU-based RVPFL-style baseline, requires HEU |

The scripts print timing summaries such as per-client encoding/encryption
time, robust aggregation time, and decryption/decoding time.

## Experimental Parameters

The main script constants can be changed at the top of each file:

- `NUM_CLIENTS`: number of participating clients;
- `POLY_MOD_DEGREE`: polynomial modulus degree;
- `PLAIN_MOD_BIT_SIZE`: plaintext modulus bit size;
- `REP`: number of primitive-benchmark repetitions;
- `mnist_layer`, `svhn_layer`, `cifar10_layer`: model-layer profiles.

Paper settings:

- datasets: MNIST, SVHN, CIFAR10;
- data distribution: 500 clients, uniformly partitioned;
- selected clients per round: `U = 5`, `15`, or `25` for efficiency tests;
- robustness tests: `U = 25`, 300 global iterations;
- models: LeNet for MNIST, Conv8 for SVHN, ResNet18 for CIFAR10;
- RLWE-based polynomial degree: `N = 8192`;
- plaintext modulus: 35 bits;
- RLWE-AHE ciphertext modulus: 60-bit prime modulus;
- CKKS scale: `2^40`;
- Paillier modulus: 2048 bits.


## Notes

- Random seeds are not fixed by default. Add `numpy.random.seed(...)` for
  deterministic local runs.
- `comparison.py` and `rotation.py` set CPU affinity to logical core 64. Change
  or remove this line if your machine has fewer cores.
- Large profiles, especially CIFAR10/ResNet18, can require substantial runtime
  and memory. Reduce `NUM_CLIENTS`, `REP`, or the active layer profile for a
  quick smoke test.

## License and Citation

Third-party components retain their original licenses, including Microsoft
SEAL, pybind11, and SEAL-Python. Project-level citation information will be
added after de-anonymization.
