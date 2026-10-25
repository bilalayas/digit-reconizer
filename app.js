/**
 * Main Application Logic
 * Canvas çizim etkileşimi, 28x28 normalizasyon, her iki motorun çalıştırılması ve UI güncellemesi.
 */

window.addEventListener('DOMContentLoaded', () => {
  const paintCanvas = document.getElementById('paintCanvas');
  const paintCtx = paintCanvas.getContext('2d', { willReadFrequently: true });

  const previewCanvas = document.getElementById('previewCanvas');
  const previewCtx = previewCanvas.getContext('2d', { willReadFrequently: true });

  const clearBtn = document.getElementById('clearBtn');
  const brushSizeInput = document.getElementById('brushSize');
  const toggleGridBtn = document.getElementById('toggleGridBtn');
  const gridOverlay = document.getElementById('gridOverlay');

  // UI Elementleri
  const aiDigitEl = document.getElementById('aiDigit');
  const aiConfidenceEl = document.getElementById('aiConfidence');
  const aiBarsEl = document.getElementById('aiBars');
  const aiExplanationEl = document.getElementById('aiExplanation');

  const ruleDigitEl = document.getElementById('ruleDigit');
  const ruleConfidenceEl = document.getElementById('ruleConfidence');
  const ruleExplanationEl = document.getElementById('ruleExplanation');

  const featLoopsEl = document.getElementById('featLoops');
  const featEndpointsEl = document.getElementById('featEndpoints');
  const featIntersectionsEl = document.getElementById('featIntersections');
  const featMassRatioEl = document.getElementById('featMassRatio');
  const featAspectEl = document.getElementById('featAspect');

  const statPixelsEl = document.getElementById('statPixels');
  const statCenterEl = document.getElementById('statCenter');
  const statBBoxEl = document.getElementById('statBBox');

  // AI Barlarını İlk Oluşturma
  function setupProbabilityBars() {
    aiBarsEl.innerHTML = '';
    for (let i = 0; i <= 9; i++) {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <span class="prob-label">${i}</span>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" id="barFill_${i}"></div>
        </div>
        <span class="prob-val" id="barVal_${i}">%0</span>
      `;
      aiBarsEl.appendChild(row);
    }
  }
  setupProbabilityBars();

  // Çizim Değişkenleri
  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;

  const canvasWrapper = document.getElementById('canvasWrapper');
  const brushValBadge = document.getElementById('brushVal');

  // Tuval Çözünürlüğünü Wrapper Boyutuna Eşitleme (Büyük ve Net Çizim Alanı)
  function resizeCanvas() {
    if (!canvasWrapper) return;
    const rect = canvasWrapper.getBoundingClientRect();
    const newSize = Math.round(rect.width);
    if (newSize > 0 && (paintCanvas.width !== newSize || paintCanvas.height !== newSize)) {
      // Çizim varsa yedekle
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = paintCanvas.width;
      tempCanvas.height = paintCanvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(paintCanvas, 0, 0);

      paintCanvas.width = newSize;
      paintCanvas.height = newSize;

      paintCtx.fillStyle = '#03060f';
      paintCtx.fillRect(0, 0, paintCanvas.width, paintCanvas.height);
      paintCtx.lineCap = 'round';
      paintCtx.lineJoin = 'round';
      paintCtx.strokeStyle = '#ffffff';

      // Eski çizimi ölçekleyerek geri çiz
      if (tempCanvas.width > 0 && tempCanvas.height > 0) {
        paintCtx.drawImage(tempCanvas, 0, 0, newSize, newSize);
      }
    }
  }

  function initCanvas() {
    if (canvasWrapper) {
      const rect = canvasWrapper.getBoundingClientRect();
      const s = Math.round(rect.width) || 400;
      paintCanvas.width = s;
      paintCanvas.height = s;
    } else {
      paintCanvas.width = 400;
      paintCanvas.height = 400;
    }
    paintCtx.fillStyle = '#03060f';
    paintCtx.fillRect(0, 0, paintCanvas.width, paintCanvas.height);
    paintCtx.lineCap = 'round';
    paintCtx.lineJoin = 'round';
    paintCtx.strokeStyle = '#ffffff';
  }
  initCanvas();

  window.addEventListener('resize', () => {
    resizeCanvas();
  });

  const canvasHint = document.getElementById('canvasHint');
  const infoToggleBtn = document.getElementById('infoToggleBtn');
  const closeDrawerBtn = document.getElementById('closeDrawerBtn');
  const detailsDrawer = document.getElementById('detailsDrawer');

  if (infoToggleBtn && detailsDrawer) {
    infoToggleBtn.addEventListener('click', () => {
      detailsDrawer.classList.toggle('open');
      // Sidebar açıldığında veya kapandığında canvas sahnesi yeniden boyutlanır
      setTimeout(resizeCanvas, 360);
    });
  }
  if (closeDrawerBtn && detailsDrawer) {
    closeDrawerBtn.addEventListener('click', () => {
      detailsDrawer.classList.remove('open');
      setTimeout(resizeCanvas, 360);
    });
  }

  if (brushSizeInput && brushValBadge) {
    brushSizeInput.addEventListener('input', (e) => {
      brushValBadge.textContent = `${e.target.value}px`;
    });
  }

  // Fare & Dokunmatik Olayları (Duyarlı Ölçekleme)
  function getCoordinates(e) {
    const rect = paintCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    
    // CSS boyutu ile Canvas piksel çözünürlüğü arasındaki oran
    const scaleX = paintCanvas.width / rect.width;
    const scaleY = paintCanvas.height / rect.height;

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  }

  function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    if (canvasHint) canvasHint.classList.add('hidden');
    const { x, y } = getCoordinates(e);
    lastX = x;
    lastY = y;
    drawDot(x, y);
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const { x, y } = getCoordinates(e);

    paintCtx.beginPath();
    paintCtx.lineWidth = parseInt(brushSizeInput.value, 10);
    paintCtx.moveTo(lastX, lastY);
    paintCtx.lineTo(x, y);
    paintCtx.stroke();

    lastX = x;
    lastY = y;

    debouncedPredict();
  }

  function drawDot(x, y) {
    paintCtx.beginPath();
    paintCtx.fillStyle = '#ffffff';
    paintCtx.arc(x, y, parseInt(brushSizeInput.value, 10) / 2, 0, Math.PI * 2);
    paintCtx.fill();
    debouncedPredict();
  }

  function stopDrawing() {
    if (isDrawing) {
      isDrawing = false;
      runAnalysis();
    }
  }

  paintCanvas.addEventListener('mousedown', startDrawing);
  paintCanvas.addEventListener('mousemove', draw);
  window.addEventListener('mouseup', stopDrawing);

  paintCanvas.addEventListener('touchstart', startDrawing, { passive: false });
  paintCanvas.addEventListener('touchmove', draw, { passive: false });
  window.addEventListener('touchend', stopDrawing);

  clearBtn.addEventListener('click', () => {
    initCanvas();
    resetOutputs();
    if (canvasHint) canvasHint.classList.remove('hidden');
  });

  toggleGridBtn.addEventListener('click', () => {
    gridOverlay.classList.toggle('active');
  });

  // Debounce yardımcı
  let predictTimeout = null;
  function debouncedPredict() {
    if (predictTimeout) clearTimeout(predictTimeout);
    predictTimeout = setTimeout(runAnalysis, 40);
  }

  function resetOutputs() {
    previewCtx.fillStyle = '#000000';
    previewCtx.fillRect(0, 0, 28, 28);

    aiDigitEl.textContent = '-';
    aiConfidenceEl.textContent = '%0 Güven';
    aiExplanationEl.textContent = 'Çizim bekleniyor...';

    ruleDigitEl.textContent = '-';
    ruleConfidenceEl.textContent = 'Kural Güveni: -';
    ruleExplanationEl.textContent = 'Çizim bekleniyor...';

    featLoopsEl.textContent = '-';
    featEndpointsEl.textContent = '-';
    featIntersectionsEl.textContent = '-';
    featMassRatioEl.textContent = '-';
    featAspectEl.textContent = '-';

    statPixelsEl.textContent = '0%';
    statCenterEl.textContent = '-';
    statBBoxEl.textContent = '-';

    for (let i = 0; i <= 9; i++) {
      const fill = document.getElementById(`barFill_${i}`);
      const val = document.getElementById(`barVal_${i}`);
      if (fill) fill.style.width = '0%';
      if (fill) fill.classList.remove('highest');
      if (val) val.textContent = '%0';
    }
  }

  /**
   * Çizimi 28x28 standart formata normalize etme (MNIST standardı: Bounding Box kırpma + Merkezleme)
   */
  function processInputTo28x28() {
    const rawData = paintCtx.getImageData(0, 0, paintCanvas.width, paintCanvas.height);
    const data = rawData.data;

    // Sınır kutusu bulma
    let minX = paintCanvas.width, maxX = 0;
    let minY = paintCanvas.height, maxY = 0;
    let activePixels = 0;

    for (let y = 0; y < paintCanvas.height; y++) {
      for (let x = 0; x < paintCanvas.width; x++) {
        const alpha = data[(y * paintCanvas.width + x) * 4]; // R kanalı (beyaz çizim)
        if (alpha > 40) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          activePixels++;
        }
      }
    }

    if (activePixels < 60) {
      return null; // Yetersiz çizim
    }

    // 20x20 içine en-boy oranı korunarak sığdırma
    const boxW = Math.max(1, maxX - minX + 1);
    const boxH = Math.max(1, maxY - minY + 1);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = boxW;
    tempCanvas.height = boxH;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(paintCanvas, minX, minY, boxW, boxH, 0, 0, boxW, boxH);

    // Orantılı ölçekleme (en fazla 20x20 piksele indir)
    const targetSize = 20;
    const scale = targetSize / Math.max(boxW, boxH);
    const scaledW = Math.max(1, Math.round(boxW * scale));
    const scaledH = Math.max(1, Math.round(boxH * scale));

    const scaledCanvas = document.createElement('canvas');
    scaledCanvas.width = scaledW;
    scaledCanvas.height = scaledH;
    const scaledCtx = scaledCanvas.getContext('2d');
    scaledCtx.drawImage(tempCanvas, 0, 0, scaledW, scaledH);

    // İlk olarak geçici 28x28 tuvale geometrik ortada çizelim
    const initOffsetX = Math.floor((28 - scaledW) / 2);
    const initOffsetY = Math.floor((28 - scaledH) / 2);

    const scratchCanvas = document.createElement('canvas');
    scratchCanvas.width = 28;
    scratchCanvas.height = 28;
    const scratchCtx = scratchCanvas.getContext('2d');
    scratchCtx.fillStyle = '#000000';
    scratchCtx.fillRect(0, 0, 28, 28);
    scratchCtx.drawImage(scaledCanvas, initOffsetX, initOffsetY);

    // MNIST Standardı: Center of Mass (Kütle Merkezi) hesaplama
    const sData = scratchCtx.getImageData(0, 0, 28, 28).data;
    let sumMass = 0;
    let sumX = 0;
    let sumY = 0;

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const val = sData[(y * 28 + x) * 4] / 255.0;
        if (val > 0.05) {
          sumMass += val;
          sumX += x * val;
          sumY += y * val;
        }
      }
    }

    // Ağırlık merkezinin (13.5, 13.5) yani 28x28'in tam ortasına olan kaymasını bul
    let shiftX = 0;
    let shiftY = 0;
    if (sumMass > 0) {
      const cX = sumX / sumMass;
      const cY = sumY / sumMass;
      shiftX = Math.round(13.5 - cX);
      shiftY = Math.round(13.5 - cY);
    }

    // Nihai 28x28 önizleme tuvaline kütle merkezi hizalı olarak çiz
    previewCtx.fillStyle = '#000000';
    previewCtx.fillRect(0, 0, 28, 28);
    previewCtx.drawImage(scratchCanvas, shiftX, shiftY);

    // 28x28 piksel float dizisi oluştur
    const pData = previewCtx.getImageData(0, 0, 28, 28).data;
    const normalized = new Float32Array(784);
    let countOn = 0;

    for (let i = 0; i < 784; i++) {
      const val = pData[i * 4] / 255.0;
      normalized[i] = val;
      if (val > 0.1) countOn++;
    }

    // İstatistikler
    statPixelsEl.textContent = `${Math.round((countOn / 784) * 100)}%`;
    statCenterEl.textContent = `(${Math.round(13.5 - shiftX)}, ${Math.round(13.5 - shiftY)})`;
    statBBoxEl.textContent = `${boxW}x${boxH} px`;

    return normalized;
  }

  /**
   * İki yöntemi de çalıştırıp arayüze basan ana fonksiyon
   */
  function runAnalysis() {
    const inputPixels = processInputTo28x28();
    if (!inputPixels) {
      resetOutputs();
      return;
    }

    // 1. YAKLAŞIM: Yapay Zeka Modeli Tahmini
    const aiResult = predictWithAI(inputPixels);
    aiDigitEl.textContent = aiResult.digit;
    aiConfidenceEl.textContent = `%${aiResult.confidence} Güven`;
    aiExplanationEl.textContent = aiResult.explanation;

    // AI Olasılık Barlarını Güncelle
    aiResult.probabilities.forEach((prob, digit) => {
      const percent = Math.round(prob * 100);
      const fill = document.getElementById(`barFill_${digit}`);
      const val = document.getElementById(`barVal_${digit}`);
      if (fill && val) {
        fill.style.width = `${percent}%`;
        val.textContent = `%${percent}`;
        if (digit === aiResult.digit) {
          fill.classList.add('highest');
        } else {
          fill.classList.remove('highest');
        }
      }
    });

    // 2. YAKLAŞIM: Kural Tabanlı / El Yapımı Algoritma Tahmini
    const features = extractTopologyAndGeometry(inputPixels, 28, 28);
    const ruleResult = predictWithRules(features);

    ruleDigitEl.textContent = ruleResult.digit !== null ? ruleResult.digit : '?';
    ruleConfidenceEl.textContent = `Güven: ${ruleResult.confidence}`;
    ruleExplanationEl.textContent = ruleResult.explanation;

    // Çıkarılan geometrik özellikleri göster
    if (ruleResult.features) {
      featLoopsEl.textContent = ruleResult.features.loops !== undefined ? ruleResult.features.loops : '-';
      featEndpointsEl.textContent = ruleResult.features.endpoints !== undefined ? ruleResult.features.endpoints : '-';
      featIntersectionsEl.textContent = ruleResult.features.crossings !== undefined ? ruleResult.features.crossings : '-';
      featMassRatioEl.textContent = ruleResult.features.massRatio !== undefined ? ruleResult.features.massRatio : '-';
      featAspectEl.textContent = ruleResult.features.aspectRatio !== undefined ? ruleResult.features.aspectRatio : '-';
    }
  }
});
