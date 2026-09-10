# CertiRank robustness evaluation

This directory contains the plaintext federated-learning simulator used for
the robustness results. It implements the three dataset/model pairs in the
paper, five aggregation methods, and four reported poisoning attacks.
The values reported in the paper are included in `paper_results.csv`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Methods and attacks

Methods:

- `fedavg`
- `bcpbfl`
- `rvpfl`
- `frl`
- `cmgra`

Reported attacks:

- `label_flip`: class $l$ is changed to $f-l-1$;
- `grad_ascent`: malicious local optimization follows the ascent direction;
- `pixel_backdoor`: the FRL-paper artificial backdoor; each malicious client
  receives the same nine examples stamped with a fixed $5\times5$ F-shaped
  trigger and relabeled as class 2;
- `vem`: the source-compatible VEM attack for rank-based methods.

## Main configurations

| Dataset | Model | Keep ratio | Local epochs | Clients | Clients/round | Rounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | Conv2 | 0.2 | 1 | 1,000 | 25 | 500 |
| SVHN | Conv8 | 0.5 | 5 | 1,000 | 25 | 500 |
| CIFAR10 | ResNet18 | 0.5 | 5 | 1,000 | 25 | 500 |

All main experiments use a Dirichlet partition with $\alpha=1.0$ and seed 0.
Use the same partition file and seed across methods for paired comparisons.

Generate a source-compatible partition before the first run, for example:

```bash
python3 -m benchmark.vem_partition \
  --dataset mnist --n-clients 1000 --alpha 1.0 --seed 0 \
  --root benchmark_data --download \
  --output partitions/mnist_dirichlet_a1_seed0.pkl
```

Repeat the command with `--dataset svhn` and `--dataset cifar10`, changing the
output filename accordingly. The canonical Table-8 manifest expects:

```text
partitions/mnist_dirichlet_a1_seed0.pkl
partitions/svhn_dirichlet_a1_seed0.pkl
partitions/cifar10_dirichlet_a1_seed0.pkl
```

## Single-run examples

CMGRA under VEM:

```bash
python3 run_benchmark.py \
  --dataset mnist --model conv2 \
  --method cmgra --attack vem --malicious-fraction 0.2 \
  --n-clients 1000 --round-clients 25 --rounds 500 \
  --data-partition legacy-vem \
  --partition-file partitions/mnist_dirichlet_a1_seed0.pkl \
  --local-epochs 1 --keep-ratio 0.2 --seed 0 --partition-seed 0 \
  --device cuda
```

FRL under the same attack:

```bash
python3 run_benchmark.py \
  --dataset mnist --model conv2 \
  --method frl --attack vem --malicious-fraction 0.2 \
  --n-clients 1000 --round-clients 25 --rounds 500 \
  --data-partition legacy-vem \
  --partition-file partitions/mnist_dirichlet_a1_seed0.pkl \
  --local-epochs 1 --keep-ratio 0.2 --seed 0 --partition-seed 0 \
  --device cuda
```

Clean gradient baseline:

```bash
python3 run_benchmark.py \
  --dataset cifar10 --model resnet18 \
  --method fedavg --attack clean --malicious-fraction 0 \
  --n-clients 1000 --round-clients 25 --rounds 500 \
  --data-partition legacy-vem \
  --partition-file partitions/cifar10_dirichlet_a1_seed0.pkl \
  --local-epochs 5 --seed 0 --partition-seed 0 \
  --device cuda
```

## Reproduce Table 8

Generate the complete experiment manifest:

```bash
python3 -m benchmark.manifest --output paper_table8_manifest.jsonl
```

The manifest contains 168 runs: five clean baselines per dataset; all five
methods under Label Flipping, Gradient Ascent, and Pixel Backdoor at 10%, 20%,
and 30%; and FRL/CMGRA under VEM at the same three ratios. Every row contains
the complete dataset-specific argument list. Launch one worker per GPU, for
example:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m benchmark.worker \
  --manifest paper_table8_manifest.jsonl --device cuda
```

The simulator automatically resumes from its latest checkpoint when the same
output directory and configuration are reused.

## Outputs

Each run writes:

- `config.json`: complete configuration;
- `metrics.csv`: round-wise accuracy, loss, timing, and backdoor ASR;
- `summary.json`: final and best metrics;
- periodic checkpoints for resumption.

Generate a plot from a completed run with:

```bash
python3 -m benchmark.plotting PATH_TO_RUN_DIRECTORY
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```
