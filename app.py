from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
# CORS diperlukan agar file HTML (frontend) diizinkan mengambil data dari Flask (backend)
CORS(app)

# Load model klasifikasi anime Anda
model = tf.keras.models.load_model('model_predik_anime.h5')


def prepare_image(img):
    # Mengacu pada notebook, gambar dikonversi ke Grayscale ("L")
    img = img.convert('L')
    
    # Sesuaikan ukuran resolusi input yang digunakan saat model di-training (misal 150x150)
    img = img.resize((128, 128)) 
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar'}), 400
    
    file = request.files['file']
    img = Image.open(file.stream)
    processed_img = prepare_image(img)
    
    # Lakukan prediksi menggunakan model Keras
    prediction = model.predict(processed_img)
    
    # Ambil nilai skor hasil prediksi
    score = float(prediction[0][0])
    
    # LOGIKA YANG DISESUAIKAN DENGAN NOTEBOOK:
    # Score mendekati 1 (> 0.5) = Female (Perempuan)
    # Score mendekati 0 (<= 0.5) = Male (Laki-laki)
    
    if score > 0.5:
        predicted_class = 'perempuan'
        confidence = score
    else:
        predicted_class = 'laki-laki'
        confidence = 1.0 - score

    result = {
        'class': predicted_class,
        'confidence': confidence
    }
    
    return jsonify(result)

import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))   # Render akan memberikan PORT secara otomatis
    app.run(host='0.0.0.0', port=port, debug=False)
    #app.run(debug=True, port=5000)