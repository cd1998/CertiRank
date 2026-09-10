
import numpy as np
from tqdm import tqdm
from heu import phe
import time

NUM_CLIENTS = 1
mnist_layer = [576, 73728, 1605632, 2560]
TOTAL_DIM = sum(mnist_layer)

# Paillier init
kit = phe.setup(phe.SchemaType.ZPaillier, 2048)
enc = kit.encryptor()
dec = kit.decryptor()
eval = kit.evaluator()
encoder = phe.FloatEncoder(phe.SchemaType.ZPaillier,scale=100)

# client plain
clients_plain = []
for _ in range(NUM_CLIENTS):
    vec = np.random.uniform(-1, 1, TOTAL_DIM)
    clients_plain.append(vec)
clients_plain_median = np.median(clients_plain, axis=0)
# print('client_median',clients_plain_median)

# client ciphertetxt
clients_cipher = []
t1 = time.perf_counter()
for cid in range(NUM_CLIENTS):
    enc_vec = []
    for x in tqdm(clients_plain[cid], desc=f"Client {cid}", leave=False):
        pt = encoder.encode(x)
        enc_vec.append(enc.encrypt(pt))
    clients_cipher.append(enc_vec)
t2 = time.perf_counter()
print('t2-t1',t2-t1)

t3 = time.perf_counter()
# element-wise median compute
mask_r = np.random.randint(1, 10**6, TOTAL_DIM)
masked_cipher = [[] for _ in range(NUM_CLIENTS)]
for cid in tqdm(range(NUM_CLIENTS), desc="Applying first-stage mask"):
    for j in range(TOTAL_DIM):
        pt_r = encoder.encode(mask_r[j])
        ct = eval.add(clients_cipher[cid][j] , pt_r)
        masked_cipher[cid].append(ct)
t4 = time.perf_counter()
print('t4-t3',t4-t3)


t5 = time.perf_counter()
masked_plain = np.zeros((NUM_CLIENTS, TOTAL_DIM))
for cid in tqdm(range(NUM_CLIENTS), desc="Decrypting masked models"):
    for j in range(TOTAL_DIM):
        masked_plain[cid, j] = encoder.decode(
            dec.decrypt(masked_cipher[cid][j])
        )
t6 = time.perf_counter()
print('t6-t5',t6-t5)


t7 = time.perf_counter()
median_plain = np.median(masked_plain, axis=0)
median_cipher = []
for j in range(TOTAL_DIM):
    ct = enc.encrypt(encoder.encode(median_plain[j]))
    ct_sub = eval.sub(ct,encoder.encode(mask_r[j])) 
    median_cipher.append(ct_sub)
t8 = time.perf_counter()
print('t8-t7',t8-t7)

# chen = []
# for j in range(TOTAL_DIM):
#     pl = dec.decrypt(median_cipher[j])
#     plain = encoder.decode(pl)
#     chen.append(plain)
# print(chen)

# encryption and median mask
pair_mask = np.random.randint(
    1, 10**6, size=(NUM_CLIENTS, TOTAL_DIM)
)

masked_client_pair = [[] for _ in range(NUM_CLIENTS)]
masked_median_pair = [[] for _ in range(NUM_CLIENTS)]

t9 = time.perf_counter()
for cid in tqdm(range(NUM_CLIENTS), desc="Applying pairwise mask"):
    for j in range(TOTAL_DIM):
        pt_s = encoder.encode(pair_mask[cid][j])
        ct_client = eval.add(clients_cipher[cid][j] , pt_s)
        ct_server = eval.add(median_cipher[j] , pt_s)
        masked_client_pair[cid].append(ct_client)
        masked_median_pair[cid].append(ct_server)
t10 = time.perf_counter()
print('t10-t9',t10-t9)


# decrypt result used robust aggregation
client_plain_pair = np.zeros((NUM_CLIENTS, TOTAL_DIM))
median_plain_pair = np.zeros((NUM_CLIENTS, TOTAL_DIM))

t11 = time.perf_counter()
for cid in tqdm(range(NUM_CLIENTS), desc="Decrypting client-median pairs"):
    for j in range(TOTAL_DIM):
        client_plain_pair[cid, j] = encoder.decode(
            dec.decrypt(masked_client_pair[cid][j])
        )
        median_plain_pair[cid, j] = encoder.decode(
            dec.decrypt(masked_median_pair[cid][j])
        )
t12 = time.perf_counter()
print('t12-t11',t12-t11)

# weighted aggregation
weights = np.random.rand(NUM_CLIENTS)
weights = weights / np.sum(weights)

t13 = time.perf_counter()
weighted_cipher = [[] for _ in range(NUM_CLIENTS)]
for cid in tqdm(range(NUM_CLIENTS), desc="Applying weighted aggregation"):
    for j in range(TOTAL_DIM):
        pt_w = encoder.encode(weights[cid])
        ct = eval.mul(clients_cipher[cid][j] , pt_w)
        weighted_cipher[cid].append(ct)

# element-wise 相加
aggregated_cipher = []

for j in range(TOTAL_DIM):
    ct_sum = weighted_cipher[0][j]
    for cid in range(1, NUM_CLIENTS):
        eval.add_inplace(ct_sum,weighted_cipher[cid][j])
        # ct_sum += weighted_cipher[cid][j]
    aggregated_cipher.append(ct_sum)
t14 = time.perf_counter()
print('t14-t13',t14-t13)


# client decrypt
t15 = time.perf_counter()
aggregated_plain = np.zeros(TOTAL_DIM)

for j in tqdm(range(TOTAL_DIM), desc="Decrypting final aggregated model"):
    aggregated_plain[j] = encoder.decode(
        dec.decrypt(aggregated_cipher[j])
    )
t16 = time.perf_counter()
print('t16-t15',t16-t15)





