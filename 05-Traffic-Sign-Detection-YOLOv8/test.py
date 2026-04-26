# ----------------------------------------------------------------------------
# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# ----------------------------------------------------------------------------
import cv2                        # görüntü işleme ve çizim işlemleri için
import json                       # yapılandırma dosyasını okumak için
import os                         # dosya yolu işlemleri için
from ultralytics import YOLO      # eğitilmiş YOLOv8 modelini çalıştırmak için

# ----------------------------------------------------------------------------
# 2. YAPILANDIRMA DOSYASININ YÜKLENMESİ (CONFIG LOADING)
# ----------------------------------------------------------------------------
# Kodun çalıştığı dizini baz alarak config.json dosyasını yükle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, 'config.json')

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# JSON içerisinden gerekli yolları çek
MODEL_PATH = os.path.join(BASE_DIR, config['paths']['best_weights'])
IMAGE_PATH = os.path.join(BASE_DIR, config['paths']['test_image'])
OUTPUT_PATH = config['paths'].get('output_image', 'prediction_result.jpg')

# ----------------------------------------------------------------------------
# 3. MODELİN VE GÖRSELİN YÜKLENMESİ (MODEL & IMAGE LOADING)
# ----------------------------------------------------------------------------
# Eğitilmiş özel YOLO modelini sisteme yükleme
model = YOLO(MODEL_PATH)

# Test edilecek görseli OpenCV ile belleğe okuma
image = cv2.imread(IMAGE_PATH)

# ----------------------------------------------------------------------------
# 4. TAHMİN YÜRÜTME (MODEL INFERENCE)
# ----------------------------------------------------------------------------
# Görseli modele gönderip sonuç kümesini alma
results = model(IMAGE_PATH)[0]
print(f"Tespit edilen nesne sayısı: {len(results.boxes)}")

# ----------------------------------------------------------------------------
# 5. TESPİTLERİN GÖRSELLEŞTİRİLMESİ (VISUALIZATION & BOX DRAWING)
# ----------------------------------------------------------------------------
for box in results.boxes:
    # Koordinat bilgilerini alma (xyxy formatı)
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Sınıf ID ve güven skoru (confidence) bilgilerini alma
    cls_id = int(box.cls[0])
    confidence = float(box.conf[0])

    # Etiket metnini oluşturma (Örn: Stop 0.95)
    label = f"{model.names[cls_id]} {confidence:.2f}"

    # Görsel üzerine yeşil sınırlayıcı kutu çizme
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Etiketi kutunun 10 piksel üzerine konumlandırma
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ----------------------------------------------------------------------------
# 6. SONUÇLARIN GÖSTERİLMESİ VE KAYDEDİLMESİ (DISPLAY & SAVE)
# ----------------------------------------------------------------------------
# Sonucu yeni bir pencerede göster
cv2.imshow("Prediction", image)

# İşlenmiş görseli kaydet
cv2.imwrite(OUTPUT_PATH, image)

# Kullanıcı bir tuşa basana kadar bekle ve pencereleri kapat
cv2.waitKey(0)
cv2.destroyAllWindows()