import os
import tensorflow as tf
import gradio as gr
import numpy as np
from PIL import Image

# ==============================================================================
# 1. DEFINISI CUSTOM LAYER (Inisialisasi di __init__ untuk Keras 3)
# ==============================================================================
class ChannelAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
        # Berdasarkan hasil diagnosismu (1280 channel):
        # fc1 = 1280 // 8 = 160 unit
        self.fc1 = tf.keras.layers.Dense(
            160, 
            activation='relu', 
            use_bias=False, 
            name="fc1"
        )
        # fc2 = 1280 unit
        self.fc2 = tf.keras.layers.Dense(
            1280, 
            use_bias=False, 
            name="fc2"
        )

    def call(self, inputs):
        # Menggunakan Global Average Pooling dan Max Pooling secara manual
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        
        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))
        
        scale = tf.nn.sigmoid(avg_out + max_out)
        return inputs * scale

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

# ==============================================================================
# 2. PROSES MEMUAT MODEL 
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_keras_20260531_133333.keras")

model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={"ChannelAttentionLayer": ChannelAttentionLayer}, 
    compile=False
)

# ==================== KONFIGURASI TARGET OUTPUT ====================
IMG_SIZE = (224, 224) 
LABELS = ["chicken pox", "eczema", "hives", "melonoma"] 
# ===================================================================

# ==============================================================================
# 3. FUNGSI PREDIKSI & GRADIO INTERFACE
# ==============================================================================
def predict_skin(img):
    if img is None:
        return None
        
    # Preprocessing Gambar
    img = Image.fromarray(img).resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  
    
    # Prediksi
    predictions = model.predict(img_array)[0]
    
    return {LABELS[i]: float(predictions[i]) for i in range(len(LABELS))}

demo = gr.Interface(
    fn=predict_skin,
    inputs=gr.Image(type="numpy", label="Unggah Foto Sampel Kulit"),
    outputs=gr.Label(num_top_classes=len(LABELS), label="Hasil Analisis"),
    title="Aplikasi Analisis Penyakit Kulit - Teraliteskin",
    description="Unggah foto kulit untuk mendapatkan hasil analisis prediksi dari model AI Keras secara real-time."
)

demo.launch()