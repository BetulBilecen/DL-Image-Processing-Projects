# ----------------------------------------------------------------------------
# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# ----------------------------------------------------------------------------
from tensorflow_datasets import load  # Veri seti yükleme
from tensorflow.data import AUTOTUNE  # Veri pipeline optimizasyonu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,        # Evrişim katmanı (özellik çıkarımı yapar)
    MaxPooling2D,  # Özellik haritalarının boyutunu küçültür, önemli bilgiyi korur
    Flatten,       # Çok boyutlu veriyi 1D vektöre çevirir
    Dense,         # Tam bağlı katman (sınıflandırma/karar verme)
    Dropout        # Overfitting'i azaltmak için rastgele nöronları kapatır
)
from tensorflow.keras.optimizers import Adam  # Optimizasyon algoritması
from tensorflow.keras.callbacks import (
    EarlyStopping,        # Erken durdurma (overfitting önleme)
    ReduceLROnPlateau,   # Öğrenme oranını performansa göre azaltma
    ModelCheckpoint       # En iyi modeli kaydetme
)

import tensorflow as tf
import matplotlib.pyplot as plt  # Görselleştirme

# ----------------------------------------------------------------------------
# 2. GLOBAL PARAMETRELER VE AYARLAR (GLOBAL CONFIGURATION)
# ----------------------------------------------------------------------------
IMG_SIZE = (100, 100)
SHUFFLE_BUFFER = 1000

# ----------------------------------------------------------------------------
# 3. VERİ SETİNİN YÜKLENMESİ (DATA LOADING)
# ----------------------------------------------------------------------------
# Veri seti %80 eğitim ve %20 doğrulama (validation) olacak şekilde ayrılır
(ds_train, ds_validation), ds_info = load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,  # (görüntü, etiket) formatında veri sağlar
    with_info=True       # Veri seti hakkında bilgi döndürür
)

print(ds_info.features)
print("Number of classes: ", ds_info.features['label'].num_classes)

# ----------------------------------------------------------------------------
# 4. VERİ SETİNİN GÖRSEL OLARAK İNCELENMESİ (DATA VISUALIZATION)
# ----------------------------------------------------------------------------
# Eğitim setinden rastgele 3 görüntü ve etiket seçilir
fig = plt.figure(figsize=(10, 5))

for k, (image, label) in enumerate(ds_train.shuffle(SHUFFLE_BUFFER).take(3)):
    ax = fig.add_subplot(1, 3, k + 1)
    ax.imshow(image.numpy().astype("uint8"))
    ax.set_title(f"Etiket: {label.numpy()}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 5. VERİ ÖN İŞLEME VE ARTIRMA (DATA AUGMENTATION & PREPROCESSING)
# ----------------------------------------------------------------------------
# Amaç: Modelin genelleme yeteneğini artırmak ve overfitting'i azaltmak
def Preprocess_Train(image, label):
    """
    Resizing: Tüm görüntüleri sabit boyuta getirir.
    Random Flip: Görüntüyü yatay olarak çevirir.
    Brightness: Parlaklığı rastgele değiştirir.
    Contrast: Kontrastı değiştirerek detayları vurgular.
    Random Crop: Görüntünün rastgele bir kısmını alır.
    Normalization: Piksel değerlerini 0-1 aralığına ölçekler.
    """

    # 1. Boyutlandırma
    image = tf.image.resize(image, IMG_SIZE)

    # 2. Veri artırma işlemleri
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.2)
    image = tf.image.random_crop(image, size=(80, 80, 3))  # Rastgele kırpma

    # 3. Son düzenlemeler
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label

# Doğrulama verisi için preprocessing
def Preprocess_Validation(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label

# ----------------------------------------------------------------------------
# 6. VERİ SETİNİN HAZIRLANMASI (DATA PIPELINE)
# ----------------------------------------------------------------------------
"""
Map: Her görüntüye preprocessing uygular.
Shuffle: Veriyi karıştırarak modelin ezberlemesini engeller.
Batch: Veriyi gruplara ayırarak eğitimi hızlandırır.
Prefetch: Bir sonraki batch'i önceden hazırlayarak performansı artırır.
"""

# Eğitim veri seti pipeline
ds_train = (
    ds_train.map(Preprocess_Train, num_parallel_calls=AUTOTUNE)
    .shuffle(SHUFFLE_BUFFER)
    .batch(32)
    .prefetch(AUTOTUNE)
)

# Doğrulama veri seti pipeline
ds_validation = (
    ds_validation
    .map(Preprocess_Validation, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .prefetch(AUTOTUNE)
)

# ----------------------------------------------------------------------------
# 7. CNN MODEL MİMARİSİNİN OLUŞTURULMASI (MODEL ARCHITECTURE)
# ----------------------------------------------------------------------------
model = Sequential([
    # Feature Extraction Layer
    Conv2D(32,(3,3),activation = "relu", input_shape = (*IMG_SIZE,3)), # 32 filtre, 3x3 kernel, relu aktivasyonu, 3 kanal (RGB)
    MaxPooling2D(2,2), # 2x2 max pooling

    Conv2D(64,(3,3), activation = "relu"), # 64 filtre, 3x3 kernel, relu aktivasyon
    MaxPooling2D(2, 2),  # 2x2 max pooling

    Conv2D(128, (3, 3), activation = "relu"),  # 128 filtre, 3x3 kernel, relu aktivasyon
    MaxPooling2D(2, 2),  # 2x2 max pooling

    # Classification Layers
    Flatten(), # Çok boyutlu veriyi vektöre çevir
    Dense(128, activation = "relu" ),
    Dropout(0.5), # Overfitting'i engellemek için
    Dense(ds_info.features["label"].num_classes, activation = "softmax") # Çıkış katmanı, sınıf sayısı kadar nöron, softmax aktivasyonu ile sonuçlar olasılık değerine çevirilir.

])

# ----------------------------------------------------------------------------
# 8. CALLBACK TANIMLAMALARI (CALLBACKS)
# ----------------------------------------------------------------------------
callbacks = [
    #Eğer va_los 3 epoch boyunca iyileşmezse eğitimi durdur ve en iyi ağırlıkları yükle
    EarlyStopping(
        monitor="val_loss",  # Hangi metriğe bakılacak
        patience=3,  # Kaç epoch sabredilecek
        restore_best_weights=True  # En iyi ağırlıkları geri yükle
    ),

    #val_los e epoch boyunca iyileşmezse learning rate (öğrenme oranı) 0.2 çarpanı ile azalt
    ReduceLROnPlateau(
        monitor="val_loss",
        patience=2,
        factor=0.2,
        verbose=1,  # Konsola bilgi yazdır
        min_lr=1e-9  # Minimum öğrenme oranı
    ),
    # Her epoch sonunda eğer model daha iyi ise kaydet
    ModelCheckpoint(
        "best_model.keras",  # Kaydedilecek dosya adı
        save_best_only=True  # Sadece en iyi model kaydedilecek
    )]


# ----------------------------------------------------------------------------
# 9. MODELİN DERLENMESİ (MODEL COMPILATION)
# ----------------------------------------------------------------------------

# Derleme işlemi
model.compile(
    optimizer = Adam(learning_rate = 0.001),         # Adam optimizer, öğrenme oranını 0.001 olarak ayarla
    loss = "sparse_categorical_crossentropy",         # Kayıp fonksiyonu: etiketler tamsayı oldugu için sparse kullan
    metrics = ["accuracy"]                           # Başarı metriği olarak doğruluk (accuracy) kullan
)

# Model mimarisinin özeti
print(model.summary())  # Her katman, parametre sayısı ve çıkış boyutlarını gösterir

# Model eğitimi
history = model.fit(
    ds_train,                        # Eğitim veri seti
    validation_data = ds_validation, # Doğrulama veri seti
    epochs = 10,                      # Toplam epoch sayısı
    callbacks = callbacks,           # Daha önce tanımlanan callback'ler
    verbose = 1                      # Eğitim sürecini konsolda göster
)

#Model Evulation
plt.figure(figsize=(12,5))

# Doğruluk (Accuracy) grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Eğitim Doğruluğu")
plt.plot(history.history["val_accuracy"], label="Validasyon Doğruluğu")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Doğruluk Grafiği")
plt.legend()

# Kayıp (Loss) grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Eğitim Kaybı")
plt.plot(history.history["val_loss"], label="Validasyon Kaybı")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss Grafiği")
plt.legend()

# Grafikleri düzgün yerleştirme ve gösterme
plt.tight_layout()
plt.show()

