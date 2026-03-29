# Chroma AI

Kıyafet görsellerinden stil sınıflandırması ve renk analizi yapan, transfer learning tabanlı bir görüntü tanıma sistemi.

---

## Genel Bakış

Chroma, yüklenen kıyafet görselini analiz ederek iki temel çıktı üretir: modelin öğrendiği stil sınıflarından birine atama yapar ve görseldeki renk paletini renk teorisi çerçevesinde yorumlar. Model EfficientNetV2B2 mimarisi üzerine inşa edilmiş olup 2 fazlı transfer learning stratejisiyle eğitilmiştir.

---

## Mimari

### Model

- **Base model:** EfficientNetV2B2 (ImageNet ağırlıkları)
- **Input:** 224×224 RGB görsel
- **Head:** GlobalAveragePooling → BatchNorm → Dense(512) → LayerNorm → Dropout(0.3) → Dense(256) → Dropout(0.2) → Softmax
- **Precision:** Mixed float16 (eğitim), float32 (çıktı katmanı)
- **Format:** ONNX (inference), Keras .keras (eğitim)

### Eğitim Stratejisi

**Faz 1 — Feature Extraction**
- Base model dondurulur, yalnızca head eğitilir
- LR: `1e-3`, EarlyStopping patience: 3
- ReduceLROnPlateau ile adaptif öğrenme hızı

**Faz 2 — Fine-Tuning**
- `block6` ve sonrası açılır, BatchNorm katmanları dondurulur
- LR: `5e-5` → `~3e-6` (epoch başına %15 azalma)
- EarlyStopping patience: 5

### Augmentation Pipeline

Augmentation GPU yerine CPU'da çalışarak GPU'nun sürekli beslenmesini sağlar:

```
RandomFlip → RandomBrightness → RandomContrast → rot90 → RandomCrop
```

### Renk Analizi

1. **Arka plan temizleme** — rembg (u2netp modeli) ile kıyafet maskesi oluşturulur
2. **Dominant renkler** — k-means (k=5) ile baskın renkler tespit edilir
3. **Nötr filtresi** — siyah, beyaz, gri renkler uyum hesabından ayrı tutulur
4. **Renk teorisi uyumu** — hue açı farkları üzerinden Monokromatik / Analog / Komplementer / Split-Komplementer / Triadik / Karma sınıflandırması yapılır
5. **Stil tahmini** — doygunluk, parlaklık ve renk çeşitliliğine göre sezon/stil önerisi üretilir

---


## Kurulum

### Gereksinimler

```
fastapi>=0.110.0
uvicorn>=0.29.0
python-multipart==0.0.9
opencv-python-headless==4.8.1.78
numpy<2.0.0
Pillow>=10.0.0
rembg==2.0.57
onnxruntime==1.16.3
```

### Lokal Çalıştırma

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t chroma-ai .
docker run -p 7860:7860 chroma-ai
```

---

## Proje Yapısı

```
├── main.py              # FastAPI backend
├── requirements.txt
├── Dockerfile
└── saved_model/
    ├── Chroma.onnx      # Inference modeli
    └── class_names.json # Sınıf isimleri
```

---

## Performans

| Metrik | Değer |
|---|---|
| Validation Accuracy | ~%93.85 |
| Inference Süresi | ~300ms (CPU) |
| Model Boyutu | ~20MB (ONNX) |

---

## Notlar

- Model `.keras` formatında eğitilip `tf2onnx` ile ONNX'e dönüştürülmüştür. Bu sayede inference için TensorFlow bağımlılığı kaldırılmış, `onnxruntime` ile hafif ve taşınabilir bir deployment sağlanmıştır.
- rembg, arka plan temizleme için `u2netp` modelini kullanır. İlk çalıştırmada model otomatik indirilir (~45MB).
- Augmentation yalnızca eğitim sırasında aktiftir; inference'ta devre dışı kalır.
