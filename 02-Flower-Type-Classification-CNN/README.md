# 🌸 02 - Flower Type Classification (CNN)

Bu proje, derin öğrenme teknikleri (Convolutional Neural Networks - CNN) kullanılarak 5 farklı çiçek türünü (Papatya, Karahindiba, Gül, Ayçiçeği, Lale) sınıflandırmak amacıyla geliştirilmiştir.

## 🚀 Proje Hakkında
Bu çalışma kapsamında, `tf_flowers` veri seti üzerinde bir Evrişimli Sinir Ağı modeli eğitilmiştir. Model, görüntüleri analiz ederek hangi çiçek türüne ait olduğunu yüksek doğrulukla tahmin edebilmektedir.

## 🛠️ Kullanılan Teknolojiler
- **Dil:** Python 3.11
- **Kütüphaneler:** - `TensorFlow / Keras` (Model oluşturma ve eğitim)
  - `NumPy` (Veri işleme)
  - `Matplotlib` (Görselleştirme)
  - `TensorFlow Datasets` (Veri seti temini)

## 🏗️ Model Mimarisi
Model şu katmanlardan oluşmaktadır:
- **Rescaling:** Görüntü piksellerini 0-1 arasına normalize eder.
- **Conv2D & MaxPooling2D:** Özellik çıkarımı (Feature Extraction) yapar.
- **Dropout:** Aşırı öğrenmeyi (Overfitting) engellemek için %20 oranında nöronları devre dışı bırakır.
- **Dense:** Sınıflandırma işlemini gerçekleştiren tam bağlı katmanlar.

## 📂 Dosya Yapısı
- `CNN_FlowerTypeClassification.py`: Ana eğitim kodu.
- `requirements.txt`: Gerekli kütüphanelerin listesi.
- `README.md`: Proje dökümantasyonu.

---
*Bu proje Computer Engineering çalışmaları kapsamında geliştirilmiştir.*
