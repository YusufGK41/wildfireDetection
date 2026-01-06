"""
Bozuk resimleri tespit edip silen script
"""
import os
from PIL import Image

def check_and_clean_images(folder_path):
    """Klasördeki tüm resimleri kontrol eder ve bozuk olanları siler"""
    
    total_images = 0
    corrupted_images = 0
    
    print(f"\n🔍 {folder_path} klasörü kontrol ediliyor...")
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                total_images += 1
                file_path = os.path.join(root, filename)
                
                try:
                    # Resmi açmayı dene
                    img = Image.open(file_path)
                    img.verify()  # Resmin geçerli olup olmadığını kontrol et
                    img.close()
                    
                except Exception as e:
                    # Bozuk resim bulundu
                    print(f"❌ BOZUK: {file_path}")
                    print(f"   Hata: {str(e)}")
                    
                    # Bozuk resmi sil
                    os.remove(file_path)
                    corrupted_images += 1
                    print(f"   🗑️  Silindi!")
    
    print(f"\n📊 SONUÇ:")
    print(f"   ✅ Toplam resim: {total_images}")
    print(f"   ❌ Bozuk resim: {corrupted_images}")
    print(f"   ✅ Temiz resim: {total_images - corrupted_images}")
    
    return corrupted_images

if __name__ == "__main__":
    print("="*60)
    print("🧹 VERİ SETİ TEMİZLEME ARACI")
    print("="*60)
    
    # Train ve test klasörlerini temizle (Kaggle orijinal isimleri)
    train_corrupted = check_and_clean_images('Train_Data')
    test_corrupted = check_and_clean_images('Test_Data')
    
    total_corrupted = train_corrupted + test_corrupted
    
    print("\n" + "="*60)
    if total_corrupted == 0:
        print("✅ Tüm resimler temiz! Eğitime başlayabilirsiniz!")
    else:
        print(f"🗑️  {total_corrupted} bozuk resim silindi!")
        print("✅ Artık eğitime başlayabilirsiniz!")
    print("="*60)