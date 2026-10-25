import urllib.request
import gzip
import struct
import io
import math
import random
import json
import time

print("1. MNIST veri seti indiriliyor...")
def get_data(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        return resp.read()

train_img_gz = get_data('https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz')
train_lbl_gz = get_data('https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz')

with gzip.GzipFile(fileobj=io.BytesIO(train_img_gz)) as f:
    _, n_img, rows, cols = struct.unpack('>IIII', f.read(16))
    N_TRAIN = 15000  # 15,000 el yazısı örneği
    raw_imgs = f.read(N_TRAIN * 784)

with gzip.GzipFile(fileobj=io.BytesIO(train_lbl_gz)) as f:
    _, n_lbl = struct.unpack('>II', f.read(8))
    raw_lbls = f.read(N_TRAIN)

print(f"2. {N_TRAIN} veri örneği belleğe alındı. MLP modeli eğitiliyor...")

X = []
Y = []
for i in range(N_TRAIN):
    lbl = raw_lbls[i]
    Y.append(lbl)
    img = [raw_imgs[i * 784 + j] / 255.0 for j in range(784)]
    X.append(img)

# Mimari: 784 -> 64 -> 10 (ReLU + Softmax)
INPUT_DIM = 784
HIDDEN_DIM = 64
OUTPUT_DIM = 10

# He / Xavier Başlatma
random.seed(42)
scale1 = math.sqrt(2.0 / INPUT_DIM)
W1 = [[(random.random() * 2 - 1) * scale1 for _ in range(INPUT_DIM)] for _ in range(HIDDEN_DIM)]
b1 = [0.0] * HIDDEN_DIM

scale2 = math.sqrt(2.0 / HIDDEN_DIM)
W2 = [[(random.random() * 2 - 1) * scale2 for _ in range(HIDDEN_DIM)] for _ in range(OUTPUT_DIM)]
b2 = [0.0] * OUTPUT_DIM

# Mini-batch SGD eğitimi (Adam optimizer ile hızlı yakınsama)
# Adam parametreleri
lr = 0.003
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

mW1 = [[0.0] * INPUT_DIM for _ in range(HIDDEN_DIM)]
vW1 = [[0.0] * INPUT_DIM for _ in range(HIDDEN_DIM)]
mb1 = [0.0] * HIDDEN_DIM
vb1 = [0.0] * HIDDEN_DIM

mW2 = [[0.0] * HIDDEN_DIM for _ in range(OUTPUT_DIM)]
vW2 = [[0.0] * HIDDEN_DIM for _ in range(OUTPUT_DIM)]
mb2 = [0.0] * OUTPUT_DIM
vb2 = [0.0] * OUTPUT_DIM

EPOCHS = 6
BATCH_SIZE = 64
t_step = 0

print("Eğitim başladı...")
start_time = time.time()

for epoch in range(EPOCHS):
    # Karıştır
    indices = list(range(N_TRAIN))
    random.shuffle(indices)
    
    total_loss = 0.0
    correct = 0

    for b_start in range(0, N_TRAIN, BATCH_SIZE):
        b_idx = indices[b_start : b_start + BATCH_SIZE]
        actual_bs = len(b_idx)
        
        grad_W1 = [[0.0] * INPUT_DIM for _ in range(HIDDEN_DIM)]
        grad_b1 = [0.0] * HIDDEN_DIM
        grad_W2 = [[0.0] * HIDDEN_DIM for _ in range(OUTPUT_DIM)]
        grad_b2 = [0.0] * OUTPUT_DIM
        
        for idx in b_idx:
            x = X[idx]
            y = Y[idx]
            
            # Forward pass: 784 -> 64
            h = [0.0] * HIDDEN_DIM
            h_relu = [0.0] * HIDDEN_DIM
            for j in range(HIDDEN_DIM):
                dot = b1[j]
                for k in range(INPUT_DIM):
                    if x[k] > 0: # seyreklik optimizasyonu
                        dot += W1[j][k] * x[k]
                h[j] = dot
                h_relu[j] = dot if dot > 0 else 0.0
                
            # 64 -> 10
            logits = [0.0] * OUTPUT_DIM
            for j in range(OUTPUT_DIM):
                dot = b2[j]
                for k in range(HIDDEN_DIM):
                    dot += W2[j][k] * h_relu[k]
                logits[j] = dot
                
            # Softmax
            max_l = max(logits)
            exps = [math.exp(l - max_l) for l in logits]
            sum_exps = sum(exps)
            probs = [e / sum_exps for e in exps]
            
            # Loss & Accuracy
            pred = probs.index(max(probs))
            if pred == y:
                correct += 1
            loss = -math.log(max(probs[y], 1e-12))
            total_loss += loss
            
            # Backward pass: dlogits = probs - one_hot
            dlogits = list(probs)
            dlogits[y] -= 1.0
            
            # Gradients W2, b2
            for j in range(OUTPUT_DIM):
                dl = dlogits[j]
                grad_b2[j] += dl
                for k in range(HIDDEN_DIM):
                    grad_W2[j][k] += dl * h_relu[k]
                    
            # Backprop to hidden
            dh = [0.0] * HIDDEN_DIM
            for k in range(HIDDEN_DIM):
                dh_k = 0.0
                for j in range(OUTPUT_DIM):
                    dh_k += dlogits[j] * W2[j][k]
                dh[k] = dh_k * (1.0 if h[j] > 0 else 0.0)
                
            # Gradients W1, b1
            for j in range(HIDDEN_DIM):
                if dh[j] != 0:
                    grad_b1[j] += dh[j]
                    for k in range(INPUT_DIM):
                        if x[k] > 0:
                            grad_W1[j][k] += dh[j] * x[k]
                            
        # Adam Güncellemesi
        t_step += 1
        for j in range(OUTPUT_DIM):
            mb2[j] = beta1 * mb2[j] + (1 - beta1) * (grad_b2[j] / actual_bs)
            vb2[j] = beta2 * vb2[j] + (1 - beta2) * ((grad_b2[j] / actual_bs) ** 2)
            m_hat = mb2[j] / (1 - beta1 ** t_step)
            v_hat = vb2[j] / (1 - beta2 ** t_step)
            b2[j] -= lr * m_hat / (math.sqrt(v_hat) + eps)
            
            for k in range(HIDDEN_DIM):
                g = grad_W2[j][k] / actual_bs
                mW2[j][k] = beta1 * mW2[j][k] + (1 - beta1) * g
                vW2[j][k] = beta2 * vW2[j][k] + (1 - beta2) * (g ** 2)
                m_hat = mW2[j][k] / (1 - beta1 ** t_step)
                v_hat = vW2[j][k] / (1 - beta2 ** t_step)
                W2[j][k] -= lr * m_hat / (math.sqrt(v_hat) + eps)
                
        for j in range(HIDDEN_DIM):
            mb1[j] = beta1 * mb1[j] + (1 - beta1) * (grad_b1[j] / actual_bs)
            vb1[j] = beta2 * vb1[j] + (1 - beta2) * ((grad_b1[j] / actual_bs) ** 2)
            m_hat = mb1[j] / (1 - beta1 ** t_step)
            v_hat = vb1[j] / (1 - beta2 ** t_step)
            b1[j] -= lr * m_hat / (math.sqrt(v_hat) + eps)
            
            for k in range(INPUT_DIM):
                g = grad_W1[j][k] / actual_bs
                mW1[j][k] = beta1 * mW1[j][k] + (1 - beta1) * g
                vW1[j][k] = beta2 * vW1[j][k] + (1 - beta2) * (g ** 2)
                m_hat = mW1[j][k] / (1 - beta1 ** t_step)
                v_hat = vW1[j][k] / (1 - beta2 ** t_step)
                W1[j][k] -= lr * m_hat / (math.sqrt(v_hat) + eps)

    acc = (correct / N_TRAIN) * 100
    print(f"Epoch {epoch+1}/{EPOCHS} - Doğruluk: %{acc:.2f} - Süre: {time.time()-start_time:.1f}s")

# Ağırlıkları yuvarlayarak JS formatına kaydet
print("3. Model JS dosyasına ihraç ediliyor...")
# 4 ondalık basamağa yuvarlayarak dosya boyutunu küçült
W1_round = [[round(val, 4) for val in row] for row in W1]
b1_round = [round(val, 4) for val in b1]
W2_round = [[round(val, 4) for val in row] for row in W2]
b2_round = [round(val, 4) for val in b2]

model_data = {
    "W1": W1_round,
    "b1": b1_round,
    "W2": W2_round,
    "b2": b2_round
}

with open("trained_model.json", "w") as f:
    json.dump(model_data, f)

print("Tamamlandı! Model kaydedildi.")
