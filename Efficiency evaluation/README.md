# CertiRank efficiency evaluation

This directory contains the scripts used to evaluate cryptographic primitives
and per-round secure-aggregation overhead.

## CertiRank variants

- `certirank.py`: CertiRank command-line entry point.
- `certirank-bfv.py`: BFV native batch encoding and BFV encryption over the
  same logical joint ranking--membership representation and CMGRA workflow.
  This variant does not invoke SEncode.
- `certirank_benchmark.py`: the shared, complete implementation of client
  packing/encryption, S1 obfuscation and aggregation, S2 validation and
  decryption, and client-side aggregate recovery. Both launchers call this
  module so that their protocol paths cannot drift apart.

In both variants, a client first derives a $K$-hot membership vector from each
ranking and packs the pair logically as

```text
h[i] = lambda * rank[i] + membership[i],  lambda > number_of_clients.
```

After aggregation, `h mod lambda` gives the membership count and integer
division by `lambda` gives the Borda-score sum.

## Other scripts

| Command | Purpose |
| --- | --- |
| `python3 comparison.py` | RLWE-AHE, BGV, BFV, and CKKS primitive benchmark |
| `python3 rotation.py` | ciphertext-rotation benchmark and plots |
| `python3 bcpbfl.py` | CKKS-based BCPBFL-style baseline |
| `python3 paillier_test.py` | Paillier primitive benchmark (requires HEU) |
| `python3 rvfl.py` | Paillier/HEU-based RVPFL-style baseline (requires HEU) |

Run the scripts from this directory so that Python can import the bundled
`seal` module.

## Build the binding

The bundled shared object targets Linux x86_64 and Python 3.10. To rebuild:

```bash
cd ../SEAL-Python
python3 -m pip install numpy pybind11 wheel setuptools
cmake -S SEAL -B SEAL/build \
  -DSEAL_USE_MSGSL=OFF \
  -DSEAL_USE_ZLIB=OFF \
  -DSEAL_USE_ZSTD=OFF
cmake --build SEAL/build -j
python3 setup.py build_ext -i
cp seal*.so "../Efficiency evaluation/"
```

The model-layer profiles used by the paper are:

- MNIST/Conv2: 1,682,496 ranked parameters;
- SVHN/Conv8: 5,275,840 ranked parameters;
- CIFAR10/ResNet18: 11,164,352 ranked parameters.

Select the dataset profile and number of participating clients through the
command line. For example:

```bash
python3 certirank.py --dataset cifar10 --clients 25 --repetitions 3
python3 certirank-bfv.py --dataset cifar10 --clients 25 --repetitions 3
```

Use `--help` to list all available parameters. No source-code constants need
to be edited to reproduce a table entry.
