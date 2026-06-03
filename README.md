---
title: Teraliteskin
emoji: 🩺
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.15.2
python_version: '3.13'
app_file: app.py
pinned: false
---

# 🩺 Teralit — Skin Disease Classifier

> Deteksi penyakit kulit otomatis berbasis deep learning.  
> **Coding Camp 2026 × DBS Foundation** · Tim CC26-PSU247 · Tema: *Healthy Lives & Well-being*

---

## 📌 Tentang Proyek

**Teralit** (*Teralit Skin*) adalah sistem klasifikasi penyakit kulit yang dibangun menggunakan arsitektur **YOLOv8 Classification** yang dibungkus dengan **Keras Functional API**.

Model dirancang untuk mengenali berbagai jenis penyakit kulit dari citra dengan pendekatan hybrid:
- **YOLOv8** sebagai backbone klasifikasi utama
- **MobileNetV2 + Channel Attention (SE Block)** sebagai Keras wrapper untuk inferensi fleksibel

---

## 🧠 Arsitektur Model

| Komponen | Detail |
|---|---|
| Backbone Utama | YOLOv8m Classification (`yolov8m-cls.pt`) |
| Keras Wrapper | MobileNetV2 + ChannelAttentionLayer (SE Block) |
| Custom Layer | `ChannelAttentionLayer` — Squeeze & Excitation |
| Custom Loss | `FocalCrossEntropyLoss` — menangani class imbalance |
| Custom Callback | `TrainingLogger` — logging training per epoch |
| Input Size | 224 × 224 × 3 |
| Training Loop | Standard Keras + `tf.GradientTape` (custom loop) |

---

## 📦 Dataset

- **Format**: COCO JSON (dari Roboflow)
- **Sumber**: `Skin_desease_(Perbaikan)_dataset.coco.zip`
- **Split**: 70% Train / 15% Validation / 15% Test (stratified)
- **Preprocessing**: Resize → Normalize [0, 1]

---

## ⚙️ Stack Teknologi

```
Python 3.13
TensorFlow / Keras
Ultralytics YOLOv8
PyTorch (runtime YOLO)
Gradio (web interface)
scikit-learn · pandas · matplotlib · seaborn
```

---

## 🚀 Cara Pakai (Inference)

### Via Gradio App
Unggah gambar kulit melalui antarmuka web → model menampilkan top-3 prediksi beserta confidence score.

### Via Python
```python
import json
import numpy as np
from PIL import Image
import keras

# Load model & class mapping
model = keras.models.load_model(
    "best_model.h5",
    custom_objects={"ChannelAttentionLayer": ChannelAttentionLayer}
)
with open("class_map_20260531_133333.json") as f:
    class_map = json.load(f)

# Prediksi
img = np.array(Image.open("foto_kulit.jpg").resize((224, 224))) / 255.0
probs = model.predict(np.expand_dims(img, 0))[0]
idx = np.argmax(probs)
print(f"Prediksi: {class_map['idx_to_class'][str(idx)]}  ({probs[idx]*100:.2f}%)")
```

---

## 📥 Download Model

| File | Deskripsi | Link |
|---|---|---|
| `best_model.h5` | Keras model (HDF5) | [Google Drive](https://drive.google.com/file/d/130ro2n2uzTcw1qAQkCveFDAAbUVr9OPb/view?usp=sharing) |
| `class_map_*.json` | Label mapping | Tersedia di repo |

---

## 📁 Struktur Repository

```
AI-Terlait/
├── app.py                          # Gradio app entry point
├── best_model.h5 / .keras          # Model weights
├── class_map_20260531_133333.json  # Label & class mapping
├── training_log_20260531_133333.csv # Training history
├── requirements.txt
└── README.md
```

---

## 👥 Tim

| Info | Detail |
|---|---|
| ID Tim | CC26-PSU247 |
| Program | Coding Camp 2026 powered by DBS Foundation |
| Tema | Healthy Lives & Well-being |
| Role | AI Engineer |
