const fs = require('fs');

console.log('1. Loading uncompressed MNIST binary data in Node.js V8 engine...');
const rawImages = fs.readFileSync('train_images.bin');
const rawLabels = fs.readFileSync('train_labels.bin');

// 40,000 eğitim örneği
const N_TRAIN = 40000;
const INPUT_DIM = 784;
const HIDDEN_DIM = 96; // 64 yerine 96 kapasite (daha zengin filtreler)
const OUTPUT_DIM = 10;

console.log(`2. Preparing ${N_TRAIN} samples with augmentation for digit 9 and digit 5...`);

// Float32Array flat buffers for speed
const images = [];
const labels = [];

for (let i = 0; i < N_TRAIN; i++) {
  const lbl = rawLabels[i];
  const offset = i * INPUT_DIM;
  const img = new Float32Array(INPUT_DIM);
  for (let j = 0; j < INPUT_DIM; j++) {
    img[j] = rawImages[offset + j] / 255.0;
  }
  images.push(img);
  labels.push(lbl);

  // 9 ve 4/7/3 ayrımı için veri zenginleştirme (shift ve hafif rotasyon)
  if (lbl === 9) {
    // 1 piksel yukarı
    const imgUp = new Float32Array(INPUT_DIM);
    for (let y = 1; y < 28; y++) {
      for (let x = 0; x < 28; x++) imgUp[(y - 1) * 28 + x] = img[y * 28 + x];
    }
    images.push(imgUp);
    labels.push(9);

    // 1 piksel aşağı
    const imgDown = new Float32Array(INPUT_DIM);
    for (let y = 0; y < 27; y++) {
      for (let x = 0; x < 28; x++) imgDown[(y + 1) * 28 + x] = img[y * 28 + x];
    }
    images.push(imgDown);
    labels.push(9);
  }
}

const TOTAL_SAMPLES = images.length;
console.log(`Total training samples after augmentation: ${TOTAL_SAMPLES}`);

// He / Kaiming Normalization initialization
const W1 = new Float32Array(HIDDEN_DIM * INPUT_DIM);
const b1 = new Float32Array(HIDDEN_DIM);
const W2 = new Float32Array(OUTPUT_DIM * HIDDEN_DIM);
const b2 = new Float32Array(OUTPUT_DIM);

const scale1 = Math.sqrt(2.0 / INPUT_DIM);
for (let i = 0; i < W1.length; i++) {
  W1[i] = (Math.random() * 2 - 1) * scale1;
}

const scale2 = Math.sqrt(2.0 / HIDDEN_DIM);
for (let i = 0; i < W2.length; i++) {
  W2[i] = (Math.random() * 2 - 1) * scale2;
}

// Adam Optimizer State
const mW1 = new Float32Array(W1.length);
const vW1 = new Float32Array(W1.length);
const mb1 = new Float32Array(b1.length);
const vb1 = new Float32Array(b1.length);

const mW2 = new Float32Array(W2.length);
const vW2 = new Float32Array(W2.length);
const mb2 = new Float32Array(b2.length);
const vb2 = new Float32Array(b2.length);

const lr = 0.0025;
const beta1 = 0.9;
const beta2 = 0.999;
const eps = 1e-8;
let t = 0;

const BATCH_SIZE = 64;
const EPOCHS = 8;

console.log('3. Training Neural Network (784 -> 96 ReLU -> 10 Softmax)...');
const startTime = Date.now();

// Pre-allocate buffers for forward/backward
const h = new Float32Array(HIDDEN_DIM);
const h_relu = new Float32Array(HIDDEN_DIM);
const logits = new Float32Array(OUTPUT_DIM);
const probs = new Float32Array(OUTPUT_DIM);
const dlogits = new Float32Array(OUTPUT_DIM);
const dh = new Float32Array(HIDDEN_DIM);

const grad_W1 = new Float32Array(W1.length);
const grad_b1 = new Float32Array(b1.length);
const grad_W2 = new Float32Array(W2.length);
const grad_b2 = new Float32Array(b2.length);

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  // Karıştır
  const indices = new Int32Array(TOTAL_SAMPLES);
  for (let i = 0; i < TOTAL_SAMPLES; i++) indices[i] = i;
  for (let i = TOTAL_SAMPLES - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }

  let correct = 0;
  let totalLoss = 0;

  for (let bStart = 0; bStart < TOTAL_SAMPLES; bStart += BATCH_SIZE) {
    const bEnd = Math.min(bStart + BATCH_SIZE, TOTAL_SAMPLES);
    const bs = bEnd - bStart;

    grad_W1.fill(0);
    grad_b1.fill(0);
    grad_W2.fill(0);
    grad_b2.fill(0);

    for (let k = bStart; k < bEnd; k++) {
      const idx = indices[k];
      const x = images[idx];
      const y = labels[idx];

      // Forward: 784 -> 96
      for (let j = 0; j < HIDDEN_DIM; j++) {
        let dot = b1[j];
        const rowOff = j * INPUT_DIM;
        for (let i = 0; i < INPUT_DIM; i++) {
          const val = x[i];
          if (val > 0) dot += W1[rowOff + i] * val;
        }
        h[j] = dot;
        h_relu[j] = dot > 0 ? dot : 0;
      }

      // 96 -> 10
      let maxL = -Infinity;
      for (let j = 0; j < OUTPUT_DIM; j++) {
        let dot = b2[j];
        const rowOff = j * HIDDEN_DIM;
        for (let i = 0; i < HIDDEN_DIM; i++) {
          dot += W2[rowOff + i] * h_relu[i];
        }
        logits[j] = dot;
        if (dot > maxL) maxL = dot;
      }

      // Softmax
      let sumExp = 0;
      for (let j = 0; j < OUTPUT_DIM; j++) {
        const e = Math.exp(logits[j] - maxL);
        probs[j] = e;
        sumExp += e;
      }
      for (let j = 0; j < OUTPUT_DIM; j++) {
        probs[j] /= sumExp;
      }

      // Loss & accuracy
      let pred = 0;
      let maxP = probs[0];
      for (let j = 1; j < OUTPUT_DIM; j++) {
        if (probs[j] > maxP) {
          maxP = probs[j];
          pred = j;
        }
      }
      if (pred === y) correct++;

      // Backward: dlogits = probs - one_hot
      for (let j = 0; j < OUTPUT_DIM; j++) {
        dlogits[j] = probs[j] - (j === y ? 1.0 : 0.0);
      }

      // Gradients W2, b2
      for (let j = 0; j < OUTPUT_DIM; j++) {
        const dl = dlogits[j];
        grad_b2[j] += dl;
        const rowOff = j * HIDDEN_DIM;
        for (let i = 0; i < HIDDEN_DIM; i++) {
          grad_W2[rowOff + i] += dl * h_relu[i];
        }
      }

      // Backprop to hidden
      for (let i = 0; i < HIDDEN_DIM; i++) {
        if (h_relu[i] > 0) {
          let sum = 0;
          for (let j = 0; j < OUTPUT_DIM; j++) {
            sum += dlogits[j] * W2[j * HIDDEN_DIM + i];
          }
          dh[i] = sum;
        } else {
          dh[i] = 0;
        }
      }

      // Gradients W1, b1
      for (let j = 0; j < HIDDEN_DIM; j++) {
        const dj = dh[j];
        if (dj !== 0) {
          grad_b1[j] += dj;
          const rowOff = j * INPUT_DIM;
          for (let i = 0; i < INPUT_DIM; i++) {
            const xi = x[i];
            if (xi > 0) grad_W1[rowOff + i] += dj * xi;
          }
        }
      }
    }

    // Adam Update
    t++;
    const corr1 = 1 - Math.pow(beta1, t);
    const corr2 = 1 - Math.pow(beta2, t);

    // Update W2, b2
    for (let j = 0; j < OUTPUT_DIM; j++) {
      const gb = grad_b2[j] / bs;
      mb2[j] = beta1 * mb2[j] + (1 - beta1) * gb;
      vb2[j] = beta2 * vb2[j] + (1 - beta2) * gb * gb;
      b2[j] -= lr * (mb2[j] / corr1) / (Math.sqrt(vb2[j] / corr2) + eps);

      const rowOff = j * HIDDEN_DIM;
      for (let i = 0; i < HIDDEN_DIM; i++) {
        const idx = rowOff + i;
        const g = grad_W2[idx] / bs;
        mW2[idx] = beta1 * mW2[idx] + (1 - beta1) * g;
        vW2[idx] = beta2 * vW2[idx] + (1 - beta2) * g * g;
        W2[idx] -= lr * (mW2[idx] / corr1) / (Math.sqrt(vW2[idx] / corr2) + eps);
      }
    }

    // Update W1, b1
    for (let j = 0; j < HIDDEN_DIM; j++) {
      const gb = grad_b1[j] / bs;
      mb1[j] = beta1 * mb1[j] + (1 - beta1) * gb;
      vb1[j] = beta2 * vb1[j] + (1 - beta2) * gb * gb;
      b1[j] -= lr * (mb1[j] / corr1) / (Math.sqrt(vb1[j] / corr2) + eps);

      const rowOff = j * INPUT_DIM;
      for (let i = 0; i < INPUT_DIM; i++) {
        const idx = rowOff + i;
        const g = grad_W1[idx] / bs;
        mW1[idx] = beta1 * mW1[idx] + (1 - beta1) * g;
        vW1[idx] = beta2 * vW1[idx] + (1 - beta2) * g * g;
        W1[idx] -= lr * (mW1[idx] / corr1) / (Math.sqrt(vW1[idx] / corr2) + eps);
      }
    }
  }

  const acc = ((correct / TOTAL_SAMPLES) * 100).toFixed(2);
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`Epoch ${epoch + 1}/${EPOCHS} - Accuracy: %${acc} (Elapsed: ${elapsed}s)`);
}

// Convert 1D buffers to 2D arrays for JSON export
const W1_2d = [];
for (let j = 0; j < HIDDEN_DIM; j++) {
  const row = [];
  const off = j * INPUT_DIM;
  for (let i = 0; i < INPUT_DIM; i++) {
    row.push(parseFloat(W1[off + i].toFixed(4)));
  }
  W1_2d.push(row);
}

const b1_arr = Array.from(b1).map(v => parseFloat(v.toFixed(4)));

const W2_2d = [];
for (let j = 0; j < OUTPUT_DIM; j++) {
  const row = [];
  const off = j * HIDDEN_DIM;
  for (let i = 0; i < HIDDEN_DIM; i++) {
    row.push(parseFloat(W2[off + i].toFixed(4)));
  }
  W2_2d.push(row);
}

const b2_arr = Array.from(b2).map(v => parseFloat(v.toFixed(4)));

const modelOut = {
  W1: W1_2d,
  b1: b1_arr,
  W2: W2_2d,
  b2: b2_arr
};

fs.writeFileSync('trained_model.json', JSON.stringify(modelOut));
console.log('4. Successfully saved trained_model.json with 96 hidden units!');
