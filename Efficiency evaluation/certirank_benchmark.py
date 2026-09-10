"""System-efficiency benchmark shared by CertiRank and CertiRank-BFV.

The two variants use the same joint representation and high-level protocol.
CertiRank uses SEncode with the lightweight RLWE-AHE parameterization, whereas
CertiRank-BFV uses BFV's native BatchEncoder and standard BFV encryption.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from seal import *


LAYER_PROFILES = {
    "mnist": [576, 73728, 1605632, 2560],
    "svhn": [
        1728, 36864, 73728, 147456, 294912, 589824,
        1179648, 2359296, 524288, 65536, 2560,
    ],
    "cifar10": [
        1728, 36864, 36864, 36864, 36864,
        73728, 147456, 8192, 147456, 147456,
        294912, 589824, 32768, 589824, 589824,
        1179648, 2359296, 131072, 2359296, 2359296, 5120,
    ],
}
KEEP_RATIOS = {"mnist": 0.2, "svhn": 0.5, "cifar10": 0.5}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=LAYER_PROFILES, default="mnist")
    parser.add_argument("--clients", type=int, default=25)
    parser.add_argument("--keep-ratio", type=float)
    parser.add_argument("--poly-degree", type=int, default=8192)
    parser.add_argument("--plain-modulus-bits", type=int, default=35)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.clients <= 0:
        parser.error("--clients must be positive")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.keep_ratio is not None and not 0.0 <= args.keep_ratio <= 1.0:
        parser.error("--keep-ratio must be in [0, 1]")
    return args


def _setup(backend: str, degree: int, plain_bits: int):
    parms = EncryptionParameters(scheme_type.bfv)
    parms.set_poly_modulus_degree(degree)
    if backend == "rlwe-ahe":
        parms.set_coeff_modulus(CoeffModulus.Create(degree, [60]))
    else:
        parms.set_coeff_modulus(CoeffModulus.BFVDefault(degree))
    parms.set_plain_modulus(PlainModulus.Batching(degree, plain_bits))
    if backend == "rlwe-ahe":
        parms.set_encoding_method(encoding_method.pm)

    context = SEALContext(parms)
    keygen = KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.create_public_key()
    galois_keys = (
        keygen.create_galois_keys() if backend == "bfv" else None
    )
    encoder = BatchEncoder(context)
    return {
        "encryptor": Encryptor(context, public_key),
        "evaluator": Evaluator(context),
        "decryptor": Decryptor(context, secret_key),
        "encoder": encoder,
        "galois_keys": galois_keys,
        "slot_count": encoder.slot_count(),
    }


def _packing_base(client_count: int) -> int:
    return 1 << math.ceil(math.log2(client_count + 1))


def _joint_layer(
    size: int,
    keep_ratio: float,
    packing_base: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ranks = rng.permutation(size).astype(np.int64, copy=False)
    keep = int(math.ceil(keep_ratio * size))
    membership = (ranks >= size - keep).astype(np.int64)
    return packing_base * ranks + membership


def _chunk_metadata(layer_sizes: list[int], slot_count: int):
    metadata = []
    for layer_index, size in enumerate(layer_sizes):
        chunks = math.ceil(size / slot_count)
        for chunk_index in range(chunks):
            valid = min(slot_count, size - chunk_index * slot_count)
            metadata.append((layer_index, valid))
    return metadata


def _encode(
    values: np.ndarray,
    shift: int,
    backend: str,
    encoder,
):
    packed = values.copy()
    # SEncode compensates for negacyclic wrap-around before x^shift rotation.
    if backend == "rlwe-ahe" and shift:
        packed[-shift:] *= -1
    return encoder.encode(packed)


def _obfuscate(ciphertext, shift: int, backend: str, crypto, monomial):
    if not shift:
        return ciphertext
    if backend == "rlwe-ahe":
        return crypto["evaluator"].multiply_plain(ciphertext, monomial)
    return crypto["evaluator"].rotate_rows(
        ciphertext, -shift, crypto["galois_keys"]
    )


def _remove_padding(values: np.ndarray, padding: int) -> np.ndarray | None:
    if not padding:
        return values
    zero_positions = np.flatnonzero(values == 0)
    if zero_positions.size < padding:
        return None
    keep = np.ones(values.size, dtype=bool)
    keep[zero_positions[:padding]] = False
    return values[keep]


def _valid_joint_layer(
    values: np.ndarray,
    size: int,
    keep_ratio: float,
    packing_base: int,
) -> bool:
    if values.size != size:
        return False
    membership = values % packing_base
    ranks = (values - membership) // packing_base
    keep = int(math.ceil(keep_ratio * size))
    return bool(
        np.all((membership == 0) | (membership == 1))
        and np.all((0 <= ranks) & (ranks < size))
        and np.unique(ranks).size == size
        and np.array_equal(membership, ranks >= size - keep)
        and int(membership.sum()) == keep
    )


def _run_once(backend: str, args: argparse.Namespace, crypto, rng):
    layer_sizes = LAYER_PROFILES[args.dataset]
    keep_ratio = (
        args.keep_ratio
        if args.keep_ratio is not None
        else KEEP_RATIOS[args.dataset]
    )
    packing_base = _packing_base(args.clients)
    slots = crypto["slot_count"]
    metadata = _chunk_metadata(layer_sizes, slots)
    shift_limit = slots if backend == "rlwe-ahe" else slots // 2
    shifts = rng.integers(0, shift_limit, size=len(metadata), dtype=np.int64)

    monomials = []
    if backend == "rlwe-ahe":
        for shift in shifts:
            vector = np.zeros(slots, dtype=np.int64)
            vector[int(shift)] = 1
            monomials.append(crypto["encoder"].encode(vector))
    else:
        monomials = [None] * len(metadata)

    # Client-side joint construction, encoding, and encryption.
    encrypted_clients = []
    start = time.perf_counter()
    for _ in range(args.clients):
        ciphertexts = []
        meta_index = 0
        for size in layer_sizes:
            joint = _joint_layer(size, keep_ratio, packing_base, rng)
            for offset in range(0, size, slots):
                chunk = joint[offset : offset + slots]
                if chunk.size < slots:
                    chunk = np.pad(chunk, (0, slots - chunk.size))
                plain = _encode(
                    chunk,
                    int(shifts[meta_index]),
                    backend,
                    crypto["encoder"],
                )
                ciphertexts.append(crypto["encryptor"].encrypt(plain))
                meta_index += 1
        encrypted_clients.append(ciphertexts)
    submit_time = time.perf_counter() - start

    # S1 obfuscates a copy in two stages; S2 decrypts and validates the joint
    # representation.  Chunks are shuffled independently for every client.
    # The layer label and valid length are retained, but the original chunk
    # position within that layer is not exposed to S2.
    start = time.perf_counter()
    qualified = []
    for client_index, ciphertexts in enumerate(encrypted_clients):
        recovered = [[] for _ in layer_sizes]
        valid = True
        shuffled_indices = []
        for layer_index in range(len(layer_sizes)):
            layer_indices = np.asarray(
                [
                    index
                    for index, (owner, _) in enumerate(metadata)
                    if owner == layer_index
                ],
                dtype=np.int64,
            )
            shuffled_indices.extend(rng.permutation(layer_indices).tolist())
        for index in shuffled_indices:
            layer_index, valid_length = metadata[index]
            shift = shifts[index]
            obfuscated = _obfuscate(
                ciphertexts[index],
                int(shift),
                backend,
                crypto,
                monomials[index],
            )
            plain = crypto["decryptor"].decrypt(obfuscated)
            values = np.asarray(crypto["encoder"].decode(plain), dtype=np.int64)
            values = _remove_padding(values, slots - valid_length)
            if values is None:
                valid = False
                break
            recovered[layer_index].append(values)
        if valid:
            valid = all(
                _valid_joint_layer(
                    np.concatenate(parts),
                    size,
                    keep_ratio,
                    packing_base,
                )
                for parts, size in zip(recovered, layer_sizes)
            )
        if valid:
            qualified.append(client_index)

    if not qualified:
        raise RuntimeError("no client contribution passed validation")

    # S1 aggregates original ciphertexts, masks each aggregate once, and S2
    # decrypts the masked aggregates. This is the current CertiRank workflow;
    # it does not perform the old per-client decrypt/re-encrypt key conversion.
    masked_plaintexts = []
    masks = []
    for index in range(len(metadata)):
        aggregate = encrypted_clients[qualified[0]][index]
        for client_index in qualified[1:]:
            crypto["evaluator"].add_inplace(
                aggregate, encrypted_clients[client_index][index]
            )
        mask = rng.integers(0, 1 << 10, size=slots, dtype=np.int64)
        mask_plain = crypto["encoder"].encode(mask)
        masked = crypto["evaluator"].add_plain(aggregate, mask_plain)
        masked_plaintexts.append(crypto["decryptor"].decrypt(masked))
        masks.append(mask)
    server_time = time.perf_counter() - start

    # Each client receives the masked plaintexts and masks, removes the masks,
    # and separates membership counts from Borda sums. CMGRA itself is a
    # plaintext integer-vector operation and is intentionally reported apart.
    start = time.perf_counter()
    for plain, mask, shift in zip(masked_plaintexts, masks, shifts):
        values = np.asarray(crypto["encoder"].decode(plain), dtype=np.int64)
        aggregate_joint = values - mask
        if backend == "rlwe-ahe" and shift:
            aggregate_joint[-int(shift) :] *= -1
        _membership_counts = aggregate_joint % packing_base
        _borda_sums = (aggregate_joint - _membership_counts) // packing_base
    recovery_time = time.perf_counter() - start

    return submit_time / args.clients, server_time, recovery_time, len(qualified)


def main(backend: str) -> None:
    if backend not in {"rlwe-ahe", "bfv"}:
        raise ValueError("backend must be 'rlwe-ahe' or 'bfv'")
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    crypto = _setup(backend, args.poly_degree, args.plain_modulus_bits)
    measurements = [
        _run_once(backend, args, crypto, rng) for _ in range(args.repetitions)
    ]
    means = np.mean(np.asarray([row[:3] for row in measurements]), axis=0)
    print(f"backend: {backend}")
    print(f"dataset: {args.dataset}")
    print(f"ranked parameters: {sum(LAYER_PROFILES[args.dataset]):,}")
    print(f"qualified clients: {measurements[-1][3]}/{args.clients}")
    print(f"client encode+encrypt: {means[0]:.6f} s")
    print(f"server verification+aggregation: {means[1]:.6f} s")
    print(f"client aggregate recovery: {means[2]:.6f} s")
