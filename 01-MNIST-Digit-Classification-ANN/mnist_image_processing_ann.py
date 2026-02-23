"""
MNIST Veri Seti (MNIST Dataset):
    - Rakamlama (Labeling): 0-9 toplamda 10 sınıf var.
    - 28x28 piksel boyutunda resimler.
    - Grayscale (Gri Tonlama) resimler.
    - 60.000 eğitim, 10.000 test verisi.
    - Amacımız: ANN ile bu resimleri tanımlamak ya da sınıflandırmak.

Görüntü İşleme (Image Processing):
    - Histogram eşitleme (Histogram Equalization): Kontrast iyileştirme.
    - Gaussian blur (Gauss Bulanıklığı): Gürültü azaltma.
    - Canny edge detection (Canny Kenar Tespiti): Kenar tespiti.

Kütüphaneler (Libraries):
    - TensorFlow: Keras ile ANN modeli oluşturma ve eğitim.
    - Matplotlib: Görselleştirme.
    - OpenCV (cv2): Görüntü işleme.
"""

    # 1. Import Libraries (Kütüphanelerin İçe Aktarılması)
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist       # MNIST veri seti
from tensorflow.keras.models import Sequential    # ANN modeli oluşturmak için
from tensorflow.keras.layers import Dense, Dropout # ANN katmanları
from tensorflow.keras.optimizers import Adam      # Eğitim için optimizer

    # 2. Load MNIST Dataset (MNIST Veri Setinin Yüklenmesi)
(x_train, y_train), (x_test, y_test) = mnist.load_data() # 2 adet tuple döner.
print(f'x_train shape: {x_train.shape},\t y_train shape: {y_train.shape}') # Veri boyutlarını kontrol etme

"""
x_train shape: (60000, 28, 28) -> 60.000 adet 28x28 piksellik resim.
y_train shape: (60000,)        -> 60.000 adet etiket (vektör formatında).
"""

    # 3. Image Preprocessing (Görüntü Ön İşleme)
img = x_train[0]  # Eğitim veri setinden ilk resmi aldık.
stages = {"Orijinal": img}

# A. Histogram Equalization (Histogram Eşitleme)
eq = cv2.equalizeHist(img)  # Kontrastı artırmak için histogram eşitleme uygulandı.
stages["Histogram Eşitleme"] = eq

# B. Gaussian Blur (Gauss Bulanıklığı)
blur = cv2.GaussianBlur(eq, (5, 5), 0)  # 5x5'lik bir filtre kullanıldı.
stages["Gaussian Blur"] = blur

# C. Canny Edge Detection (Canny Kenar Tespiti)
edges = cv2.Canny(blur, 50, 150)  # 50 ve 150 alt/üst eşik değerleri ile kenar tespiti.
stages["Canny Kenarları"] = edges

# D. Visualization (Görselleştirme)
fig, axes = plt.subplots(2, 2, figsize=(6, 6))  # 2x2'lik bir ızgara yapısı oluşturuldu.
axes = axes.flat

# For döngüsü ile işlem aşamalarını görselleştirme
for ax, (title, im) in zip(axes, stages.items()):
    ax.imshow(im, cmap="gray")  # Görüntülerin gri tonlamalı gösterilmesi sağlandı.
    ax.set_title(title)         # Belirlenen başlıklar eklendi.
    ax.axis("off")              # Eksen rakamları kapatıldı.

plt.suptitle("MNIST Image Processing Stages", fontsize=16)
plt.tight_layout()
plt.show()  # Ön işleme aşamalarını görmek için aktif edilebilir.

# E. Preprocessing Function (Ön İşleme Fonksiyonu)
def Preprocess_Image(_image):
    """
    Uygulanacak adımlar:
    - Histogram eşitleme (Histogram Equalization)
    - Gaussian blur (Gauss Bulanıklığı)
    - Canny kenar tespiti (Canny Edge Detection)
    - Flattening: 28x28 boyutunu 784 boyutuna çevirme
    - Normalizasyon: 0-255 aralığını 0-1 arasına çekme
    """
    img_eq = cv2.equalizeHist(_image)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    img_edges = cv2.Canny(img_blur, 50, 150)

    # Düzleştirme (flattening) yapıldı ve piksel değerleri normalize edildi:
    features = img_edges.flatten() / 255.0
    return features

# Veri setinin tamamını fonksiyon üzerinden işleme alma
X_train = np.array([Preprocess_Image(img) for img in x_train])
y_train_sub = y_train

X_test = np.array([Preprocess_Image(img) for img in x_test])
y_test_sub = y_test

    # 4. ANN Model Creation (ANN Modelinin Oluşturulması)
model = Sequential([
    Dense(128, activation="relu", input_shape=(784,)),  # İlk gizli katman: 128 nöron.
    Dropout(0.5),  # Aşırı öğrenmeyi (overfitting) engellemek için dropout katmanı.
    Dense(64, activation="relu"),   # İkinci gizli katman: 64 nöron.
    Dense(10, activation="softmax"), # Çıkış katmanı: 10 rakam sınıfı için 10 nöron.
])

    # 5. Compile Model (Modelin Derlenmesi)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",  # Çok sınıflı sınıflandırma için kayıp fonksiyonu.
    metrics=["accuracy"]                     # Başarı ölçütü olarak doğruluk seçildi.
)

print(model.summary())  # Model mimarisini özetler.

    # 6. ANN Model Training (ANN Modelinin Eğitilmesi)
history = model.fit(
    X_train, y_train_sub,
    validation_data=(X_test, y_test_sub),
    epochs=10,      # 50 epoch (eğitim turu).
    batch_size=32,  # Veriler 32'lik paketler halinde işlenir.
    verbose=2       # Her epoch sonunda özet bilgi yazdırır.
)

    # 7. Evaluate Model Performance (Model Performansının Değerlendirilmesi)
test_loss, test_acc = model.evaluate(X_test, y_test_sub)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

# A. Plot Training History (Eğitim Geçmişinin Görselleştirilmesi)
plt.figure(figsize=(12, 5))

# Loss (Kayıp) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Accuracy (Doğruluk) Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()