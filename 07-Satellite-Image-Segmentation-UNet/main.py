# ----------------------------------------------------------------------------
# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# ----------------------------------------------------------------------------
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# 2. VERİLERİN HAZIRLANMASI VE YÜKLENMESİ (DATA LOADING & PREPROCESSING)
# ----------------------------------------------------------------------------
def Load_Dataset(root, img_size = (128,128)):
    images, masks = [], []

    #Her bir tile klasörünü sırasıyla dolaşma
    for tile in sorted(os.listdir(root)):
        img_dir = os.path.join(root,tile,"images")      # Görüntülerin olduğu klasörlerin yolu
        masks_dir = os.path.join(root, tile, "masks")    # Maskelerin olduğu klasörlerin yolu

        #Klasör yok ise o kısmı atlama
        if not os.path.isdir(img_dir): continue

        # Görüntü ve maskelerin yollarını belirleme
        # ".jpg" veya ".png" yazmadan önce klasörleri mmanuel olarak açıp uzantısını kontrol edin
        for file in os.listdir(img_dir):

            # Görsel klasörü yoksa atlama
            if not file.lower().endswith(".jpg"): continue

            img_path = os.path.join(img_dir,file)   # Görüntülerin dosya yolu
            mask_path = os.path.join(masks_dir, os.path.splitext(file)[0] + ".png")    # Maskeye karşılık gelenn dosya yolu

            # Maske yok ise atlama
            if not os.path.exists(mask_path): continue

            # Görüntüyü oku, renk uzayını düzelt ve normalize etme
            img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)  # Görüntü renk ayarları
            img = cv2.resize(img,img_size) / 255.0  # Görüntü boyutlandırma ve normalize etme

            # Maskeyi gri tonlamada okuma ve boyutlandırma
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)

            # 127 eşik değeridir. 127'den büyükse 255 yap, değilse 0 yap.
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            mask = np.expand_dims(mask, axis=-1 ) / 255.0  # Boyut arttırma ve normalize etme

            images.append(img)
            masks.append(mask)

    return np.array(images, dtype= "float32"), np.array(masks, dtype= "float32")    # Numpy dizisine çevirildi

X, y = Load_Dataset("aerial_dataset", img_size=(256, 256))
print(f"Toplam veri sayısı (Sum number of data): {len(X)}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.2, random_state= 42)
print(f"Toplam eğitim veri sayısı (Total training samples): {len(X_train)}")
print(f"Toplam doğrulama veri sayısı (Total validation samples): {len(X_val)}")

# ----------------------------------------------------------------------------
# 3. UNET MİMARİSİNİN TANIMLANMASI (MODEL ARCHITECTURE)
# ----------------------------------------------------------------------------
def Unet_Model(input_size = (256, 256, 3) ):

    #Girdi katmanı
    inputs = keras.Input(input_size)

    #Encoder: öznitelik çıkarımı (feature extraction) ve downspling
    c1 = layers.Conv2D(16,3, activation="relu", padding= "same")(inputs)
    c1 = layers.Conv2D(16,3, activation= "relu", padding= "same")(c1)
    p1 = layers.MaxPooling2D()(c1)  # Downsampling 64x64

    c2 = layers.Conv2D(32,3, activation= "relu", padding= "same")(p1)
    c2 = layers.Conv2D(32,3, activation= "relu", padding= "same")(c2)
    p2 = layers.MaxPooling2D()(c2)  # Downsampling 32x32

    c3 = layers.Conv2D(64,3, activation= "relu", padding= "same")(p2)
    c3 = layers.Conv2D(64,3, activation= "relu", padding= "same")(c3)
    p3 = layers.MaxPooling2D()(c3)  # Downsampling 16x16

    c4 = layers.Conv2D(128,3, activation= "relu", padding= "same")(p3)
    c4 = layers.Conv2D(128,3, activation= "relu", padding= "same")(c4)
    p4 = layers.MaxPooling2D()(c4)  # Downsampling 8x8

    c5 = layers.Conv2D(256,3, activation= "relu", padding= "same")(p4)
    c5 = layers.Conv2D(256,3, activation= "relu", padding= "same")(c5)

    # Decoder: up sampling ve skip connection
    u6 = layers.Conv2DTranspose(128,2, strides=(2,2), padding= "same")(c5)  # up sample 8x8 -> 16x16
    u6 = layers.concatenate([u6,c4])  # Skip Connection: küçültmede belirlenen detaylar kenar vb. hatırlanarak büyültme işlemi yapılıyor.

    c6 = layers.Conv2D(128,3, activation= "relu", padding= "same")(u6)
    c6 = layers.Conv2D(128,3, activation= "relu", padding= "same")(c6)


    u7 = layers.Conv2DTranspose(64,2, strides=(2,2), padding= "same")(c6)   # 16x16 -> 32x32
    u7 = layers.concatenate([u7, c3])

    c7 = layers.Conv2D(64,3, activation= "relu", padding= "same")(u7)
    c7 = layers.Conv2D(64,3, activation= "relu", padding= "same")(c7)

    u8 = layers.Conv2DTranspose(32,2, strides=(2,2), padding= "same")(c7)   # 32x32 -> 64x64
    u8 = layers.concatenate([u8, c2])

    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(u8)
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(c8)

    u9 = layers.Conv2DTranspose(16, 2, strides=(2, 2), padding="same")(c8)  # 64x64 -> 128x128
    u9 = layers.concatenate([u9, c1])

    c9 = layers.Conv2D(16, 3, activation="relu", padding="same")(u9)
    c9 = layers.Conv2D(16, 3, activation="relu", padding="same")(c9)

    # Çıkış katmanı
    outputs = layers.Conv2D(1,1, activation= "sigmoid")(c9)

    return keras.Model(inputs,outputs)

# ----------------------------------------------------------------------------
# 4. MODEL EĞİTİMİ (MODEL TRAINING)
# ----------------------------------------------------------------------------

# Eğitim mimarisi
unet_model = Unet_Model()
unet_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint("model_best.keras", save_best_only= True),   #En iyi modeli kaydetme
    keras.callbacks.ReduceLROnPlateau(), # Doğrulama kaybı düşmez ise learning rate'i azalt
    keras.callbacks.EarlyStopping(patience= 10, restore_best_weights= True) # 25 epoch boyunca iyileşmez ise dur
]

history = unet_model.fit(
    X_train, y_train,
    validation_data= (X_val, y_val),    # Doğrulama verisi
    epochs= 50,
    batch_size= 16,
    callbacks= callbacks
)

# ----------------------------------------------------------------------------
# 5. SONUÇLARIN DEĞERLENDİRİLMESİ (EVALUATION & VISUALIZATION)
# ----------------------------------------------------------------------------

# Kayıpların görselleştirilmesi
plt.plot(history.history["loss"], label = "train_loss")
plt.plot(history.history["val_loss"], label = "val_loss")
plt.legend()
plt.show()

def Show_Prediction(idx = 0):
    img = X_val[idx]
    mask_true = y_val[idx].squeeze() #gerçek maske
    pred_raw = unet_model.predict(img[None, ...])[0].squeeze()    # Modelden tahmini al ve kanalı sıkıştır

    #Sonuçların görselleştirilmesi
    plt.figure(figsize = (10,4))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(mask_true, cmap= "gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_raw, cmap="inferno")
    plt.title("Prediction (Heatmap)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

Show_Prediction(5)