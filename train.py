"""
Orman Yangını Tespit Sistemi
CNN ile görüntü sınıflandırma projesi
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ===============================
# 1. VERİ SETİ KONTROLÜ
# ===============================
def check_dataset(path):
    """Veri setindeki görüntü sayısını kontrol eder"""
    if not os.path.exists(path):
        print(f"❌ HATA: {path} klasörü bulunamadı!")
        return False
    
    print(f"\n📂 {path} klasörü kontrol ediliyor...")
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            img_count = len(os.listdir(class_path))
            print(f"   ✅ {class_name}: {img_count} görüntü")
    return True

# ===============================
# 2. VERİ HAZIRLAMA
# ===============================
def prepare_data(train_path, test_path, img_size=224, batch_size=32):
    """ImageDataGenerator ile veri yükler"""
    
    # Data Augmentation (Eğitim verisi için)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Test verisi için sadece normalize
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Veri setlerini yükle
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, test_generator

# ===============================
# 3. MODEL OLUŞTURMA
# ===============================
def build_model(img_size=224):
    """CNN modeli oluşturur"""
    
    model = Sequential([
        # Katman 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        MaxPooling2D(2, 2),
        
        # Katman 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Katman 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Katman 4
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Sınıflandırma katmanları
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary: Yangın Var/Yok
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ===============================
# 4. EĞİTİM
# ===============================
def train_model(model, train_gen, test_gen, epochs=10):
    """Modeli eğitir"""
    
    print("\n🚀 Model eğitimi başlıyor...")
    print(f"   Epoch sayısı: {epochs}")
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        verbose=1
    )
    
    print("\n✅ Eğitim tamamlandı!")
    return history

# ===============================
# 5. GRAFİK OLUŞTURMA
# ===============================
def plot_training_history(history, save_path='training_history.png'):
    """Eğitim grafiklerini çizer ve kaydeder"""
    
    plt.figure(figsize=(14, 5))
    
    # Accuracy grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Grafikler kaydedildi: {save_path}")
    # plt.show() kaldırıldı - GUI olmadan çalışması için

# ===============================
# 6. SONUÇLARI KAYDETME
# ===============================
def save_results(history, model, save_dir='results'):
    """Modeli ve sonuçları kaydeder"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Modeli kaydet
    model.save(f'{save_dir}/fire_detection_model.h5')
    print(f"💾 Model kaydedildi: {save_dir}/fire_detection_model.h5")
    
    # Sonuçları CSV olarak kaydet
    results_df = pd.DataFrame({
        'epoch': range(1, len(history.history['accuracy']) + 1),
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })
    results_df.to_csv(f'{save_dir}/training_results.csv', index=False)
    print(f"📄 Sonuçlar kaydedildi: {save_dir}/training_results.csv")

# ===============================
# ANA PROGRAM
# ===============================
if __name__ == "__main__":
    
    print("="*60)
    print("🔥 ORMAN YANGINI TESPİT SİSTEMİ")
    print("="*60)
    
    # GPU kontrolü
    print(f"\n💻 TensorFlow Version: {tf.__version__}")
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    print(f"🎮 GPU Kullanılabilir: {'✅ Evet' if gpu_available else '❌ Hayır (CPU kullanılacak)'}")
    
    # Veri yolları
    # Kaggle'dan indirilen orijinal klasör isimleri
    TRAIN_PATH = 'Train_Data'
    TEST_PATH = 'Test_Data'
    
    # Parametreler
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # Veri kontrolü
    if not check_dataset(TRAIN_PATH) or not check_dataset(TEST_PATH):
        print("\n❌ Veri seti bulunamadı!")
        print("   Lütfen Kaggle'dan veri setini indirip 'Train_Data' ve 'Test_Data' klasörlerini buraya yerleştirin.")
        print("   Link: https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images/data")
        exit(1)
    
    # Veriyi hazırla
    print("\n📦 Veri yükleniyor...")
    train_gen, test_gen = prepare_data(TRAIN_PATH, TEST_PATH, IMG_SIZE, BATCH_SIZE)
    
    # Model oluştur
    print("\n🏗️  Model oluşturuluyor...")
    model = build_model(IMG_SIZE)
    model.summary()
    
    # Eğitim
    history = train_model(model, train_gen, test_gen, EPOCHS)
    
    # Grafikleri çiz
    plot_training_history(history)
    
    # Sonuçları kaydet
    save_results(history, model)
    
    # Final test accuracy
    final_acc = history.history['val_accuracy'][-1]
    print(f"\n🎯 Final Test Accuracy: {final_acc*100:.2f}%")
    print("\n" + "="*60)
    print("✅ İşlem tamamlandı!")
    print("="*60)