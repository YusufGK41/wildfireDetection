from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Model yükle
MODEL_PATH = 'results/fire_detection_model.h5'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Model yüklendi!")
    else:
        print("❌ Model bulunamadı! Lütfen önce train.py çalıştırın.")

# Model yükle
load_model()

def preprocess_image(image_file):
    """Resmi model için hazırla"""
    img = Image.open(io.BytesIO(image_file.read()))
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Model yüklenmedi! Lütfen train.py çalıştırın.'
        })
    
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı!'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi!'})
    
    try:
        # Resmi işle
        img_array = preprocess_image(file)
        
        # Tahmin yap
        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        # Not: flow_from_directory alfabetik sırayla yükler
        # Fire (0), Non_Fire (1) - Bu yüzden ters mantık
        if confidence < 0.5:  # Fire class (0)
            result = "🔥 YANGIN VAR!"
            percentage = (1 - confidence) * 100
        else:  # Non_Fire class (1)
            result = "✅ YANGIN YOK"
            percentage = confidence * 100
        
        return jsonify({
            'result': result,
            'confidence': f"{percentage:.2f}%"
        })
    
    except Exception as e:
        return jsonify({'error': f'Hata: {str(e)}'})

if __name__ == '__main__':
    print("🔥 Orman Yangını Tespit Sistemi Başlatılıyor...")
    print("🌐 http://localhost:5000")
    app.run(debug=True, port=5000)