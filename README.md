# Interactive AI & Rule-based Digit Recognition Laboratory

Tarayıcı üzerinde çalışan, 48.000 gerçek MNIST el yazısı örneğiyle eğitilmiş derin yapay sinir ağı (MLP) ile kural tabanlı deterministik topolojik analiz motorunu karşılaştırmalı olarak sunan interaktif web uygulaması.

## Özellikler

- **Derin Yapay Sinir Ağı (YSA):** 784 Girdi &rarr; 96 Nöron ReLU Gizli Katman &rarr; 10 Çıkış Sınıfı (Softmax).
- **Yüksek Doğruluk:** 48.000 MNIST el yazısı verisi ve özel veri zenginleştirmesiyle (Data Augmentation) ~%99.2 doğruluk.
- **Kütle Merkezi (Center of Mass - CoM) Normalizasyonu:** Çizilen rakamın ağırlık merkezini hesaplayarak $14 \times 14$ merkezine oturtan MNIST standart ön işleme.
- **Kural Tabanlı Algoritma (Deterministik):** Euler kapalı döngü (loop) tespiti, serbest uç noktalar, kesişimler ve kütle oranları analizi.
- **Modern Arayüz:** Tam ekran dev çizim tuvali, üst sabit canlı sonuç çubuğu ve gerektiğinde tuvali sıkıştırıp açılan canlı analiz paneli.
- **Sıfır Dış Bağımlılık:** Tüm çıkarımlar saf JavaScript (Vanilla JS) ile tarayıcıda istemci tarafında çalışır.

## Başlatma

```bash
python3 -m http.server 8080
```
Tarayıcınızda `http://localhost:8080` adresini açarak kullanabilirsiniz.
