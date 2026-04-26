"""
PROJE: Trafik Levhası Tespiti (Traffic Sign Detection)
YOLO = You Only Look Once
EO SENSOR: Kamera verisi üzerinden trafik kuralları ve trafik işaretlerinin tanınması
OTONOM GÖREV: Aracın en temel görevi olan çevreyi tanıma ve levha tespiti
"""

# ----------------------------------------------------------------------------
# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# ----------------------------------------------------------------------------
import json
import os
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# 2. YAPILANDIRMA DOSYASININ YÜKLENMESİ (CONFIG LOADING)
# ----------------------------------------------------------------------------
# Kodun çalıştığı klasörü baz alarak config.json dosyasını oku
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, 'config.json')

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# JSON içerisinden yolları ve parametreleri değişkenlere ata
DATA_PATH = os.path.join(BASE_DIR, config['paths']['data_yaml'])
MODEL_NAME = config['paths']['pretrained_model']
TRAIN_CFG = config['train_params']

# ----------------------------------------------------------------------------
# 3. MODELİN TANIMLANMASI VE YÜKLENMESİ (MODEL INITIALIZATION)
# ----------------------------------------------------------------------------
model = YOLO(MODEL_NAME)

# ----------------------------------------------------------------------------
# 4. MODELİN EĞİTİLMESİ VE PARAMETRELER (MODEL TRAINING)
# ----------------------------------------------------------------------------
model.train(
    # Veri seti yolu (Artık config.json üzerinden otomatik geliyor)
    data = DATA_PATH,

    # Eğitim Parametreleri (JSON üzerinden çekiliyor)
    epochs = TRAIN_CFG['epochs'],
    imgsz = TRAIN_CFG['imgsz'],
    batch = TRAIN_CFG['batch'], # 16 üzerine çıkma yoksa pcde donmalar veya mavi ekranla karşılaşabilirsin
    name = config['paths'].get('output_name', 'traffic-sign-model'),

    # Optimizasyon ve Öğrenme Ayarları
    lr0 = TRAIN_CFG['lr0'],
    optimizer = TRAIN_CFG['optimizer'],
    weight_decay = TRAIN_CFG['weight_decay'],
    momentum = TRAIN_CFG['momentum'],

    # Eğitim Kontrolü
    patience = 50,
    workers = 2,
    device = TRAIN_CFG['device'],

    # Kayıt ve İzleme
    save = True,
    save_period = 1,
    val = True,
    verbose = True
)