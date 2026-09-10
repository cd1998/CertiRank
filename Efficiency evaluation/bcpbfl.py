
from seal import *
import numpy as np
import math
from tqdm import tqdm
import time

NUM_CLIENTS = 25
POLY_MOD_DEGREE = 8192
scale = 2.0 ** 40

# CKKS
parms_ckks = EncryptionParameters(scheme_type.ckks)
parms_ckks.set_poly_modulus_degree(POLY_MOD_DEGREE)
parms_ckks.set_coeff_modulus(
    CoeffModulus.Create(POLY_MOD_DEGREE, [60, 40, 40, 60])
)
context_ckks = SEALContext(parms_ckks)


keygen_ckks = KeyGenerator(context_ckks)
sk_ckks = keygen_ckks.secret_key()
pk_ckks = keygen_ckks.create_public_key()
rk_ckks = keygen_ckks.create_relin_keys()
gk_ckks = keygen_ckks.create_galois_keys()
enc_ckks = Encryptor(context_ckks, pk_ckks)
eval_ckks = Evaluator(context_ckks)
dec_ckks = Decryptor(context_ckks, sk_ckks)
batch_ckks = CKKSEncoder(context_ckks)

slot_count = batch_ckks.slot_count()


mnist_layer = [576, 73728, 1605632, 2560]
layer_number = [math.ceil(x / slot_count) for x in mnist_layer]

# data
mnist_ranking_clients = []

for cid in range(NUM_CLIENTS):
    client_ranking = []
    for size in mnist_layer:
        vec = np.random.uniform(-1, 1, size)

        norm = np.linalg.norm(vec, ord=2)
        if norm > 0:
            vec = vec / norm

        client_ranking.append(vec)

    mnist_ranking_clients.append(client_ranking)

server_ranking = []
for size in mnist_layer:
    vec = np.random.uniform(-1, 1, size)
    norm = np.linalg.norm(vec, ord=2)
    if norm > 0:
        vec = vec / norm
    server_ranking.append(vec)



# print(np.linalg.norm(server_ranking[0], ord=2))
# print(np.linalg.norm(server_ranking[1], ord=2))
# print(np.linalg.norm(server_ranking[2], ord=2))
# time.sleep(100)
# ciphertext

ciphertexts_ckks_mnist = [[] for _ in range(NUM_CLIENTS)]

num_ciphertexts = sum(
    math.ceil(len(num_list) / slot_count)
    for num_list in mnist_ranking_clients[0]
)

rotation_steps = np.random.randint(
    low=0,
    high=slot_count,
    size=num_ciphertexts
)

def L_2(cipher):
    cip_mul = eval_ckks.multiply(cipher, cipher)
    eval_ckks.relinearize_inplace(cip_mul, rk_ckks)
    for i in range(int(np.log2(slot_count))):
        c = eval_ckks.rotate_vector(cip_mul, 2**i, gk_ckks)
        eval_ckks.add_inplace(cip_mul, c)
    return cip_mul

def inner(cipher,plain):
    cip_mul = eval_ckks.multiply_plain(cipher, plain)
    # eval_ckks.relinearize_inplace(cip_mul, rk_ckks)
    for i in range(int(np.log2(slot_count))):
        c = eval_ckks.rotate_vector(cip_mul, 2**i, gk_ckks)
        eval_ckks.add_inplace(cip_mul, c)
    return cip_mul

# total_chunks = 0
# for cid in range(NUM_CLIENTS):
#     for num_list in mnist_ranking_clients[cid]:
#         total_chunks += math.ceil(len(num_list) / slot_count)
# pbar = tqdm(total=total_chunks, desc="CKKS encode+encrypt", unit="chunk")

t1 = time.perf_counter()
for cid in range(NUM_CLIENTS):
    for num_list in mnist_ranking_clients[cid]:
        num_chunks = math.ceil(len(num_list) / slot_count)

        for i in range(num_chunks):
            start = i * slot_count
            end = min(start + slot_count, len(num_list))
            chunk = num_list[start:end]

            # padding
            if len(chunk) < slot_count:
                chunk = np.pad(chunk, (0, slot_count - len(chunk)))

            # encode & encrypt
            plain = batch_ckks.encode(chunk,scale)
            ciphertexts_ckks_mnist[cid].append(enc_ckks.encrypt(plain))

#             pbar.update(1)
# pbar.close()
t2 = time.perf_counter()

server_ranking_ploy = []
for num_list in server_ranking:
    num_chunks = math.ceil(len(num_list) / slot_count)

    for i in range(num_chunks):
        start = i * slot_count
        end = min(start + slot_count, len(num_list))
        chunk = num_list[start:end]

        # padding
        if len(chunk) < slot_count:
            chunk = np.pad(chunk, (0, slot_count - len(chunk)))

        # encode 
        plain = batch_ckks.encode(chunk,scale)
        server_ranking_ploy.append(plain)

t3 = time.perf_counter()
# computer L2
valid_clients = []
layer_sizes = mnist_layer
for cid in range(NUM_CLIENTS):
    ct_idx = 0

    for num in layer_number:
        layers_L2 = []
        needed = num
        collected = []

        while needed > 0:
            cipher = L_2(ciphertexts_ckks_mnist[cid][ct_idx])
            plain = dec_ckks.decrypt(cipher)
            vec = batch_ckks.decode(plain)
            l2 = vec[0]
            needed -= 1
            ct_idx += 1
            layers_L2.append(l2)

        layer_L2_sum = sum(layers_L2)

        is_valid = True
        if not (0.95 <= layer_L2_sum <= 1.05):
            is_valid = False
            break

    if is_valid:
        valid_clients.append(cid)
print(valid_clients)


# computer inner
for cid in valid_clients:
    ct_idx = 0
    for num in layer_number:
        layers_inner = []
        needed = num
        collected = []
        while needed > 0:
            cipher = inner(ciphertexts_ckks_mnist[cid][ct_idx],server_ranking_ploy[ct_idx])
            plain = dec_ckks.decrypt(cipher)
            vec = batch_ckks.decode(plain)
            l2 = vec[0]
            needed -= 1
            ct_idx += 1
            layers_inner.append(l2)

        layer_L2_sum = sum(layers_inner)
        print('summ',layer_L2_sum)


# ip1 = np.dot(mnist_ranking_clients[0][0], server_ranking[0])
# ip2 = np.dot(mnist_ranking_clients[0][1], server_ranking[1])
# ip3 = np.dot(mnist_ranking_clients[0][2], server_ranking[2])
# ip4 = np.dot(mnist_ranking_clients[0][3], server_ranking[3])
# print(ip1,ip2,ip3,ip4)

# add mask
masked_ciphertexts = {cid: [] for cid in valid_clients}
random_masks = {cid: [] for cid in valid_clients}

for cid in valid_clients:
    for k in range(num_ciphertexts):
        rand_vec = np.random.uniform(-1, 1, slot_count)
        pt_rand = batch_ckks.encode(rand_vec,scale)

        ct = ciphertexts_ckks_mnist[cid][k]
        ct_add = eval_ckks.add_plain(ct, pt_rand)

        masked_ciphertexts[cid].append(ct_add)
        random_masks[cid].append(pt_rand)

# decrypt → re-encrypt
refreshed_ciphertexts = {cid: [] for cid in valid_clients}

for cid in valid_clients:
    for k in range(num_ciphertexts):
        pt = dec_ckks.decrypt(masked_ciphertexts[cid][k])
        ct = enc_ckks.encrypt(pt)
        refreshed_ciphertexts[cid].append(ct)

# remove mask
final_client_ciphertexts = {cid: [] for cid in valid_clients}

for cid in valid_clients:
    for k in range(num_ciphertexts):
        ct = refreshed_ciphertexts[cid][k]
        ct_sub = eval_ckks.sub_plain(ct, random_masks[cid][k])
        final_client_ciphertexts[cid].append(ct_sub)


#element-wise add
aggregated_ckks_ciphertexts = []

for k in range(num_ciphertexts):
    agg_ct = None
    for cid in valid_clients:
        ct = final_client_ciphertexts[cid][k]
        if agg_ct is None:
            agg_ct = ct
        else:
            eval_ckks.add_inplace(agg_ct, ct)
    aggregated_ckks_ciphertexts.append(agg_ct)
t4 = time.perf_counter()
print("ckks complete robust aggregation finished.")
t5 = time.perf_counter()
aggregated_ckks_plaintexts = {cid: [] for cid in valid_clients}
for cid in valid_clients:
    for k in range(num_ciphertexts):
        pt = dec_ckks.decrypt(aggregated_ckks_ciphertexts[k])
        p = batch_ckks.decode(pt)
        aggregated_ckks_plaintexts[cid].append(p)
t6 = time.perf_counter()


# print(aggregated_ckks_plaintexts[0][0])
# print(len(aggregated_ckks_plaintexts[0][0]))

# first_layer_vectors = [
#     mnist_ranking_clients[cid][0] for cid in range(NUM_CLIENTS)
# ]

# # element-wise addition
# first_layer_sum = np.sum(first_layer_vectors, axis=0)

# print(first_layer_sum)
# print(len(first_layer_sum))
print('encrypt+encode',(t2-t1)/NUM_CLIENTS)
print('robust aggregation',t4-t3)
print('decrypt+decode',(t6-t5)/NUM_CLIENTS)







