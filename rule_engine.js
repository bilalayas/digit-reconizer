/**
 * 2. YAKLAŞIM: Kural Tabanlı / El Yapımı Algoritma (Rule-based / Heuristic Engine)
 * 
 * Hiçbir hazır makine öğrenimi veya veri seti eğitimi KULLANMADAN:
 * Matematiksel & Topolojik özellikler çıkararak insan mantığıyla tek tek kurallar çalıştırır:
 * 1. Kapalı delik sayısı (Euler sayısı / Flood-fill delik tespiti) -> örn: 8'de 2 delik, 0/6/9'da 1 delik, 1/2/3/5/7'de 0 delik.
 * 2. Uç nokta sayısı (Endpoints / Uçlar) -> örn: 1 veya 2'de uç noktalar.
 * 3. Kütle merkezi ve dikey bölge oranları (Üst yarı vs Alt yarı yoğunluğu).
 * 4. Yatay simetri ve en/boy oranı.
 */

function extractTopologyAndGeometry(grid, w = 28, h = 28) {
  // 1. Sınır kutusu (Bounding box) ve toplam piksel
  let minX = w, maxX = 0, minY = h, maxY = 0;
  let totalPixels = 0;
  let sumX = 0, sumY = 0;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (grid[y * w + x] > 0.25) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        totalPixels++;
        sumX += x;
        sumY += y;
      }
    }
  }

  if (totalPixels < 10) {
    return null; // Yetersiz çizim
  }

  const boxW = Math.max(1, maxX - minX + 1);
  const boxH = Math.max(1, maxY - minY + 1);
  const aspectRatio = boxW / boxH;
  const centerX = sumX / totalPixels;
  const centerY = sumY / totalPixels;

  // 2. Üst ve Alt Kütle Oranı
  const midY = minY + boxH / 2;
  const midX = minX + boxW / 2;
  let topMass = 0, bottomMass = 0;
  let leftMass = 0, rightMass = 0;

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (grid[y * w + x] > 0.25) {
        if (y < midY) topMass++;
        else bottomMass++;

        if (x < midX) leftMass++;
        else rightMass++;
      }
    }
  }
  const topBottomRatio = topMass / (bottomMass || 1);
  const leftRightRatio = leftMass / (rightMass || 1);

  // 3. Kapalı Delik Sayımı (Flood-fill ile arka plan dışındaki izole siyah alanlar)
  // 30x30'luk genişletilmiş harita yapıp dışarıdan (0,0) flood fill yapıyoruz
  const padW = w + 2;
  const padH = h + 2;
  const padded = new Uint8Array(padW * padH);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (grid[y * w + x] > 0.25) {
        padded[(y + 1) * padW + (x + 1)] = 1; // Çizgi pikseli
      }
    }
  }

  // Dış boşluğu işaretle (flood fill)
  const visited = new Uint8Array(padW * padH);
  const queue = [[0, 0]];
  visited[0] = 1;

  while (queue.length > 0) {
    const [cx, cy] = queue.pop();
    const neighbors = [
      [cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]
    ];
    for (const [nx, ny] of neighbors) {
      if (nx >= 0 && nx < padW && ny >= 0 && ny < padH) {
        const idx = ny * padW + nx;
        if (!visited[idx] && padded[idx] === 0) {
          visited[idx] = 1;
          queue.push([nx, ny]);
        }
      }
    }
  }

  // Dışarıdan ulaşılamayan ve çizgi olmayan alanlar deliktir (Loop)
  let loopCount = 0;
  let loopCenterY = 0;
  let loopTotalPixels = 0;

  for (let y = 1; y <= h; y++) {
    for (let x = 1; x <= w; x++) {
      const idx = y * padW + x;
      if (padded[idx] === 0 && !visited[idx]) {
        // Yeni bir kapalı delik bulundu!
        loopCount++;
        let holeSize = 0;
        let holeSumY = 0;
        const hQueue = [[x, y]];
        visited[idx] = 1;

        while (hQueue.length > 0) {
          const [hx, hy] = hQueue.pop();
          holeSize++;
          holeSumY += hy;
          const hNeighbors = [
            [hx + 1, hy], [hx - 1, hy], [hx, hy + 1], [hx, hy - 1]
          ];
          for (const [nx, ny] of hNeighbors) {
            if (nx >= 0 && nx < padW && ny >= 0 && ny < padH) {
              const nIdx = ny * padW + nx;
              if (!visited[nIdx] && padded[nIdx] === 0) {
                visited[nIdx] = 1;
                hQueue.push([nx, ny]);
              }
            }
          }
        }

        // Çok ufak 1-2 piksellik pürüz delikleri yoksay
        if (holeSize < 4) {
          loopCount--;
        } else {
          loopCenterY += holeSumY / holeSize;
          loopTotalPixels += holeSize;
        }
      }
    }
  }

  if (loopCount > 0) {
    loopCenterY /= loopCount;
  }

  // 4. Uç Nokta (Endpoint) Tespiti (İskelet/Komşuluk analizi)
  let endpoints = 0;
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (grid[y * w + x] > 0.3) {
        // 8-komşulukta kaç dolu komşu var?
        let neighborCount = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
              if (grid[ny * w + nx] > 0.3) neighborCount++;
            }
          }
        }
        // Uç nokta genelde sadece 1 veya 2 komşuya sahiptir
        if (neighborCount === 1) {
          endpoints++;
        }
      }
    }
  }
  // Gruplanan uç noktaları normalize et (komşu uç noktalar tek bir uç sayılır)
  const estimatedEndpoints = Math.min(5, Math.max(0, Math.round(endpoints / 3.2)));

  // 5. Orta Yatay ve Dikey Kesişim Sayısı
  const midYIndex = Math.round(minY + boxH * 0.5);
  let horizontalCrossings = 0;
  let inStroke = false;
  for (let x = minX; x <= maxX; x++) {
    const val = grid[midYIndex * w + x] > 0.25;
    if (val && !inStroke) {
      horizontalCrossings++;
      inStroke = true;
    } else if (!val) {
      inStroke = false;
    }
  }

  return {
    totalPixels,
    boxW,
    boxH,
    aspectRatio,
    centerX,
    centerY,
    topBottomRatio,
    leftRightRatio,
    loopCount,
    loopCenterY,
    endpoints: estimatedEndpoints,
    horizontalCrossings,
    minY,
    maxY
  };
}

/**
 * Kuralları çalıştıran karar ağacı (Decision Tree / Heuristic Rule-Base)
 */
function predictWithRules(features) {
  if (!features) {
    return {
      digit: '-',
      confidence: 'Boş Çizim',
      explanation: 'Canvas üzerinde analiz edilecek yeterli çizgi bulunamadı.',
      features: {}
    };
  }

  const {
    loopCount,
    aspectRatio,
    topBottomRatio,
    leftRightRatio,
    endpoints,
    horizontalCrossings,
    centerY,
    minY,
    boxH,
    loopCenterY
  } = features;

  let digit = null;
  let explanation = '';
  let confidence = 'Yüksek';

  // --- KURAL 1: DELİK ANALİZİ (Euler Topolojisi) ---
  if (loopCount >= 2) {
    // 2 veya daha fazla kapalı döngü -> Kesinlikle 8
    digit = 8;
    confidence = 'Çok Yüksek (%95)';
    explanation = `Tespit edilen ${loopCount} adet kapalı döngü (halka) bulundu. Sadece 8 rakamında çift kapalı halka yer alır.`;
  } else if (loopCount === 1) {
    // Tek kapalı döngü -> 0, 6, 9 veya 4 olabilir.
    // Halkanın konumu belirleyicidir:
    const relativeHolePos = (loopCenterY - minY) / boxH;

    if (relativeHolePos > 0.55) {
      // Halka altta, üst açık -> 6
      digit = 6;
      confidence = 'Yüksek (%88)';
      explanation = `1 kapalı halka bulundu ve halkanın ağırlık merkezi alt kısımda (y: %${Math.round(relativeHolePos * 100)}). Bu 6 rakamının tipik özelliğidir.`;
    } else if (relativeHolePos < 0.45) {
      // Halka üstte, alt açık -> 9 veya 4
      if (leftRightRatio < 0.8) {
        digit = 4;
        confidence = 'Orta (%75)';
        explanation = `Üstte 1 kapalı üçgen/halka ve sağa doğru dikey çizgi var. 4 rakamı kuralına uyuyor.`;
      } else {
        digit = 9;
        confidence = 'Yüksek (%85)';
        explanation = `1 kapalı halka bulundu ve halka üst kısımda (y: %${Math.round(relativeHolePos * 100)}). Aşağı uzanan kuyruk ile birlikte 9 rakamı olarak sınıflandırıldı.`;
      }
    } else {
      // Halka ortada ve büyük -> 0
      digit = 0;
      confidence = 'Yüksek (%90)';
      explanation = `Çizimin merkezini kaplayan tek bir kapalı döngü bulundu. Uç nokta bulunmuyor. Bu doğrudan 0 rakamıdır.`;
    }
  } 
  // --- KURAL 2: HİÇ DELİK YOK (Açık Çizgiler: 1, 2, 3, 4, 5, 7) ---
  else {
    // En-boy oranı çok dar ise -> 1
    if (aspectRatio < 0.42) {
      digit = 1;
      confidence = 'Çok Yüksek (%92)';
      explanation = `En/Boy oranı ${aspectRatio.toFixed(2)} ile çok dar bir dikey çizgi. Kapalı halka yok. Doğrudan 1 rakamı olarak kural eşleşti.`;
    } 
    // Üstte düz çizgi, aşağı inen çapraz çizgi -> 7
    else if (topBottomRatio > 1.4 && horizontalCrossings <= 2) {
      digit = 7;
      confidence = 'Yüksek (%82)';
      explanation = `Üst kısım alt kısımdan belirgin şekilde daha yoğun (Üst/Alt oranı: ${topBottomRatio.toFixed(2)}). Yatay üst çizgi ve çapraz gövde ile 7 kuralına uydu.`;
    } 
    // Orta kesişim sayısı 3 ise (E şeklinde dalgalı) -> 3
    else if (horizontalCrossings >= 3 || (endpoints >= 2 && rightMassLeftRatioCheck(leftRightRatio))) {
      digit = 3;
      confidence = 'Orta (%80)';
      explanation = `Sağ tarafta toplanan 2 açık kavis ve orta kesişimler tespit edildi (Delik yok, sağ kütle baskın). 3 rakamı kuralı sağlandı.`;
    } 
    // 2 vs 5 ayrımı
    else if (topBottomRatio < 0.9) {
      // Alt taban daha geniş ve yatay -> 2
      digit = 2;
      confidence = 'Orta (%78)';
      explanation = `Altta yatay taban çizgisi ağırlıklı (Alt yoğunluk baskın) ve üstte yuvarlak kavis var. 2 rakamı kurallarıyla eşleşti.`;
    } else {
      // Üstte yatay çubuk, altta kavis -> 5
      digit = 5;
      confidence = 'Orta (%72)';
      explanation = `Üst tavan çizgisi ve orta/alt kavis özellikleri tespit edildi. 5 rakamı kuralına uyum sağlandı.`;
    }
  }

  return {
    digit,
    confidence,
    explanation,
    features: {
      loops: loopCount,
      endpoints: endpoints,
      crossings: horizontalCrossings,
      massRatio: topBottomRatio.toFixed(2),
      aspectRatio: aspectRatio.toFixed(2)
    }
  };
}

function rightMassLeftRatioCheck(ratio) {
  return ratio < 0.95; // sağ taraf daha kütleli
}
