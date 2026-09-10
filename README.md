# CertiRank

This repository contains the implementation and evaluation artifact for
**CertiRank**, a privacy-preserving Byzantine-robust federated rank-learning
framework.

CertiRank combines three components:

1. **CMGRA** (Certified Membership-Gated Ranking Aggregation), which uses
   top-$K$ membership support to certify selected--unselected boundary changes
   and aggregated Borda scores to preserve fine-grained ordering;
2. **SEncode**, a sign-aware coefficient encoder for joint ranking and
   membership values; and
3. **RLWE-AHE**, a lightweight additive homomorphic-encryption construction
   supporting the additions and rotations required by the protocol.

The repository provides both the cryptographic/system-efficiency artifact and
the plaintext robustness simulator used for the experiments in the paper.

## Repository structure

```text
CertiRank/
|-- README.md
|-- Efficiency evaluation/
|   |-- README.md
|   |-- certirank.py             # CertiRank launcher (SEncode + RLWE-AHE)
|   |-- certirank-bfv.py         # CertiRank-BFV launcher
|   |-- certirank_benchmark.py   # complete shared secure workflow
|   |-- bcpbfl.py                # CKKS-based BCPBFL baseline
|   |-- rvfl.py                  # Paillier/HEU-based RVPFL baseline
|   |-- comparison.py            # cryptographic primitive benchmarks
|   |-- rotation.py              # rotation benchmark
|   `-- seal*.so                 # bundled Linux/Python 3.10 binding
|-- Robustness evaluation/
|   |-- README.md
|   |-- run_benchmark.py
|   |-- cmgra.py
|   |-- VEM.py
|   |-- benchmark/               # datasets, models, attacks, and aggregators
|   |-- tests/
|   `-- requirements.txt
`-- SEAL-Python/                 # customized Microsoft SEAL binding
```

## Experimental scope

The robustness evaluation uses the following matched dataset/model pairs:

| Dataset | Model | Ranked parameters |
| --- | --- | ---: |
| MNIST | Conv2 | 1,682,496 |
| SVHN | Conv8 | 5,275,840 |
| CIFAR10 | ResNet18 | 11,164,352 |

Each dataset is partitioned among 1,000 clients with a Dirichlet distribution
($\alpha=1.0$). In each global round, 25 clients participate. The reported
experiments run for 500 global rounds and consider 10%, 20%, and 30% malicious
clients.

The artifact implements FedAvg, BCPBFL, RVPFL, FRL, and CMGRA. It supports
Label Flipping, Gradient Ascent, Pixel Backdoor, and the ranking-specific VEM
attack. VEM is applicable only to FRL and CMGRA.

## Quick start: robustness experiments

```bash
cd "Robustness evaluation"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 -m benchmark.vem_partition \
  --dataset cifar10 --n-clients 1000 --alpha 1.0 --seed 0 \
  --root benchmark_data --download \
  --output partitions/cifar10_dirichlet_a1_seed0.pkl

# Example: CIFAR10 + ResNet18, CMGRA, 20% VEM clients, 500 rounds
python3 run_benchmark.py \
  --dataset cifar10 \
  --model resnet18 \
  --method cmgra \
  --attack vem \
  --malicious-fraction 0.2 \
  --n-clients 1000 \
  --round-clients 25 \
  --rounds 500 \
  --data-partition legacy-vem \
  --partition-file partitions/cifar10_dirichlet_a1_seed0.pkl \
  --local-epochs 5 \
  --keep-ratio 0.5 \
  --device cuda
```

Checkpoints, `metrics.csv`, `summary.json`, and accuracy plots are written under
the selected output directory. See
[`Robustness evaluation/README.md`](Robustness%20evaluation/README.md) for the
complete commands and attack settings.

The canonical 168-run matrix for Table 8 (15 clean runs, 135 common-attack
runs, and 18 VEM runs) can be generated with:

```bash
python3 -m benchmark.manifest --output paper_table8_manifest.jsonl
```

Every manifest row contains the complete command-line argument list, including
the dataset-specific model, density, local epochs, optimizer settings,
partition file, and CMGRA options.

## Quick start: efficiency experiments

The efficiency artifact targets Linux x86_64 and Python 3.10. A GPU is not
required.

```bash
cd "Efficiency evaluation"
python3 -c "from seal import *; print('SEAL-Python is ready')"
python3 certirank.py
python3 certirank-bfv.py
```

CertiRank uses SEncode and RLWE-AHE. CertiRank-BFV encodes the same logical
joint ranking--membership vector with BFV's native batching encoder and then
uses BFV encryption; it retains CMGRA and the same high-level protocol
workflow, but it does **not** use SEncode.

See [`Efficiency evaluation/README.md`](Efficiency%20evaluation/README.md) for
the build procedure and individual benchmark commands.

## Software requirements

Robustness experiments require Python 3.10 or newer, PyTorch, torchvision,
NumPy, SciPy, Pillow, and Matplotlib. Cryptographic experiments additionally
require the customized SEAL-Python binding. HEU is optional and is needed only
for the Paillier/RVPFL efficiency baseline.

## Reproducibility notes

- Datasets are downloaded by torchvision and are not included in this
  repository.
- Random seeds, client partitions, sampled clients, and attack parameters are
  recorded in every run's `config.json`.
- Rank-based experiments optimize Edge-Popup scores over fixed weights. The
  number of communicated ranks equals the number of fixed network weights.
- The bundled `seal*.so` is platform-specific. Rebuild the binding from
  `SEAL-Python/` when using a different Python or operating-system version.

## Third-party software

Microsoft SEAL, pybind11, and SEAL-Python retain their respective upstream
licenses. Project citation metadata will be added after de-anonymization.
