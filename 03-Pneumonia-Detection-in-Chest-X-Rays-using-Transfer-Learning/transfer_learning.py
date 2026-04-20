"""
PROJE: Zatürre Sınıflandırması için Transfer Öğrenme Uygulaması
VERİ SETİ: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
KULLANILAN MODEL: DenseNet121 (Transfer Learning)
"""

# ----------------------------------------------------------------------------
# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# ----------------------------------------------------------------------------
from keras.src.legacy.preprocessing.image import ImageDataGenerator  # görüntü verisi yükleme ve data augmentation
from keras.applications import DenseNet121                            # önceden eğitilmiş model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout       # model katmanları
from keras.models import Model                                        # model oluşturma
from keras.optimizers import Adam                                     # optimizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # callback'ler

import matplotlib.pyplot as plt                                       # görüntü gösterme
import numpy as np                                                    # sayısal işlemler için
import os                                                             # dosya işlemleri için
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # karışıklık matrisi ve görselleştirme

# ----------------------------------------------------------------------------
# 2. GLOBAL PARAMETRELER VE AYARLAR (GLOBAL CONFIGURATION)
# ----------------------------------------------------------------------------
DATA_DIR   = "chest_xray"  # veri seti dizini
IMAGE_SIZE = (224, 224)    # kullanılacak modelin beklediği input boyutu. Bizimki DenseNet121 olduğu için 224x224 kullandık
BATCH_SIZE = 64
CLASS_MODE = "binary"      # ikili sınıflandırma

# ----------------------------------------------------------------------------
# 3. VERİ ARTIRMA VE ÖN İŞLEME (DATA AUGMENTATION & PREPROCESSING)
# ----------------------------------------------------------------------------
train_datagenaration = ImageDataGenerator(
    rescale=1 / 255.0,           # normalizasyon
    horizontal_flip=True,        # yatayda çevirme
    rotation_range=10,           # +-10 derece döndürme
    brightness_range=[0.8, 1.2], # parlaklık ayarı
    validation_split=0.1         # validation için %10 ayırma
)

test_datagenaration = ImageDataGenerator(rescale=1 / 255.0)  # sadece test için normalizasyon

# ----------------------------------------------------------------------------
# 4. VERİ SETİNİN HAZIRLANMASI (DATA PIPELINE)
# ----------------------------------------------------------------------------
train_genaration = train_datagenaration.flow_from_directory(
    os.path.join(DATA_DIR, "train"),  # eğitim verisinin bulunduğu klasör
    target_size=IMAGE_SIZE,           # görüntüler IMAGE_SIZE botuna getirildi
    batch_size=BATCH_SIZE,            # batch boyutu
    class_mode=CLASS_MODE,            # (zatüre var/yok) şeklinde ikili bir sınıflandırma yapılacak
    subset="training",                # eğitim verisi
    shuffle=True                      # yukarıdaki eğitim verisini karıştırma
)

val_genaration = train_datagenaration.flow_from_directory(
    os.path.join(DATA_DIR, "train"),  # validation verisinin bulunduğu klasör
    target_size=IMAGE_SIZE,           # görüntüler IMAGE_SIZE botuna getirildi
    batch_size=BATCH_SIZE,            # batch boyutu
    class_mode=CLASS_MODE,            # ikili bir sınıflandırma
    subset="validation",              # validation verisi
    shuffle=False                     # veri sıralı olmalı
)

test_genaration = test_datagenaration.flow_from_directory(
    os.path.join(DATA_DIR, "test"),   # test verisinin bulunduğu klasör
    target_size=IMAGE_SIZE,           # görüntüler IMAGE_SIZE botuna getirildi
    batch_size=BATCH_SIZE,            # batch boyutu
    class_mode=CLASS_MODE,            # ikili bir sınıflandırma
    shuffle=False                     # veri sıralı olmalı
)

# ----------------------------------------------------------------------------
# 5. VERİ SETİNİN GÖRSEL OLARAK İNCELENMESİ (DATA VISUALIZATION)
# ----------------------------------------------------------------------------
class_names = list(train_genaration.class_indices.keys())  # sınıf isimleri [normal, pneumonia]
images, labels = next(train_genaration)                    # bir batch veri al

plt.figure(figsize=(10, 4))
for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    ax.imshow(images[i])
    ax.set_title(class_names[int(labels[i])])
    ax.axis("off")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 6. TRANSFER ÖĞRENME MODELİNİN TANIMLANMASI (MODEL ARCHITECTURE)
# ----------------------------------------------------------------------------
base_model = DenseNet121(
    weights="imagenet",  # önceden eğitilmiş model ağırlıkları
    include_top=False,  # son katmanı dahil etme, buraya kendi katmanımızı ekleyeceğiz
    input_shape=(*IMAGE_SIZE, 3)  # input boyutu (224, 224, 3)
)

base_model.trainable = False  # base modeli dondur, yani base model train edilmeyecek
x = base_model.output  # base model çıktısı
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)  # 128 nöronlu gizli katman
x = Dropout(0.5)(x)
prediction = Dense(1, activation="sigmoid")(x)  # ikili sınıflandırma yapılacağı için sigmoid aktivasyon fonksiyonu kullanıldı

model = Model(inputs=base_model.input, outputs=prediction)  # modeli tanımlama

# ----------------------------------------------------------------------------
# 7. CALLBACK TANIMLAMALARI VE MODELİN DEĞERLENDİRİLMESİ (CALLBACKS)
# ----------------------------------------------------------------------------

model.compile(
    optimizer = Adam(learning_rate= 1e-4),  #optimizer
    loss = "binary_crossentropy",    #ikili sınıflandırma kaybı
    metrics = ["accuracy"]
)

callbacks = [
    EarlyStopping( monitor= "val_loss",patience= 3, restore_best_weights= True), #erken durdurma
    ReduceLROnPlateau( monitor= "val_loss",patience= 2, factor=0.2, min_lr= 1e-6), #öğrenme oranını azaltma
    ModelCheckpoint("best_model.keras",monitor= "val_loss",save_best_only= True) # en iyi modeli kaydetme
]

print("Model Summary: ")
print(model.summary()) #model özeti

# ----------------------------------------------------------------------------
# 8. MODELİN EĞİTİLMESİ VE SONUÇLARIN DEĞERLENDİRİLMESİ (MODEL TRAINING & EVALUATION)
# ----------------------------------------------------------------------------
history = model.fit(
    train_genaration,
    validation_data=val_genaration,
    epochs=15,
    callbacks=callbacks,
    verbose=1  # eğitim ilerlemesini göster
)

# Test setindeki tüm görüntüler için modelden olasılık değerlerini al
pred_probs = model.predict(test_genaration, verbose=1)

# Olasılık değerlerini (0.0 - 1.0) ikili etikete (0 veya 1) dönüştür
# Eşik değer 0.5: eğer olasılık > 0.5 ise 1 (Zatürre), değilse 0 (Normal) kabul edilir
# .ravel() fonksiyonu çok boyutlu diziyi tek boyuta (liste haline) getirir
pred_labels = (pred_probs > 0.5).astype(int).ravel()

true_labels = test_genaration.classes  # test verilerinin gerçek etiketlerini (Ground Truth) değişkene al

cm = confusion_matrix(true_labels,
                      pred_labels)  # gerçek etiketler ile tahmin edilen etiketleri karşılaştırarak matrisi hesapla
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)  # matrisi görselleştirmek için

# Görselleştirme ayarlarını yap ve ekrana çizdir
plt.figure(figsize=(8, 8))
disp.plot(cmap="Blues", colorbar=False)
plt.title("Test Seti Confusion Matrix")
plt.show()