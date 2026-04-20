# 🫁 Pneumonia Detection in Chest X-Rays using Transfer Learning

In this project, I developed a binary classification model to detect pneumonia from chest X-ray images using DenseNet121 as a pre-trained base model. The model was fine-tuned on the Kaggle Chest X-Ray dataset and achieved strong results in identifying pneumonia cases.

---

## 🚀 Project Summary

- **Model Architecture:** DenseNet121 (Transfer Learning) + Custom Classification Head
- **Dataset:** Chest X-Ray Images (Pneumonia) — Kaggle
- **Task:** Binary classification (Normal vs. Pneumonia)
- **Test Accuracy:** ~%84.1
- **Pneumonia Recall:** %97.9 (only 8 out of 390 pneumonia cases missed)

---

## 🖼️ 1. Sample X-Ray Images

![Sample X-Rays](images/sample_xrays.png)

The dataset contains chest X-ray images labeled as either NORMAL or PNEUMONIA. Pneumonia cases often show increased opacity and consolidation patterns in the lungs.

---

## 🧠 2. Model Architecture

The model uses DenseNet121 pre-trained on ImageNet as a feature extractor, with a custom classification head on top:

| Layer | Function |
|---|---|
| DenseNet121 (frozen) | Pre-trained feature extractor, weights frozen during training |
| GlobalAveragePooling2D | Reduces feature maps to a single vector |
| Dense (128, ReLU) | Learns task-specific features |
| Dropout (0.5) | Prevents overfitting |
| Dense (1, Sigmoid) | Outputs probability: 0 = Normal, 1 = Pneumonia |

---

## ⚙️ 3. Data Augmentation & Preprocessing

To improve generalization, the following augmentation techniques were applied to the training set:

- **Rescaling:** Pixel values normalized to 0-1 range
- **Horizontal Flip:** Images randomly flipped horizontally
- **Rotation:** Images randomly rotated ±10 degrees
- **Brightness Adjustment:** Brightness randomly varied between 0.8 and 1.2
- **Validation Split:** 10% of training data reserved for validation

---

## 📈 4. Training Results

The model was trained for 15 epochs with Early Stopping (patience=3) and ReduceLROnPlateau (patience=2).

| Epoch | Training Accuracy | Validation Accuracy | Learning Rate |
|---|---|---|---|
| 1  | %77.96 | %89.06 | 1e-4 |
| 2  | %88.95 | %91.75 | 1e-4 |
| 3  | %90.82 | %92.71 | 1e-4 |
| 4  | %92.25 | %93.47 | 1e-4 |
| 5  | %92.91 | %94.63 | 1e-4 |
| 8  | %93.97 | %94.82 | 1e-4 |
| 11 | %94.46 | %95.78 | 2e-5 |
| 14 | %94.65 | %94.63 | 4e-6 |

ReduceLROnPlateau automatically reduced the learning rate at epochs 11 and 14, helping the model fine-tune further without overshooting.

---

## 📊 5. Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

| | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actual Normal** | 141 ✅ | 93 ❌ |
| **Actual Pneumonia** | 8 ❌ | 382 ✅ |

- **Pneumonia Recall: %97.9** — only 8 pneumonia cases were missed, which is critical in a medical context
- **Normal Precision:** The model occasionally over-predicts pneumonia for normal cases (93 false positives), which is the safer error in clinical settings

---

## 🛠️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/BetulBilecen/DL-Image-Processing-Projects.git

# 2. Install dependencies
pip install tensorflow scikit-learn matplotlib numpy

# 3. Download the dataset from Kaggle and place it in the project folder
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# 4. Run the training script
python transfer_learning.py
```

---

## 📦 Technologies Used

- **Python** — Core programming language
- **TensorFlow & Keras** — Deep learning model
- **DenseNet121** — Pre-trained transfer learning model
- **ImageDataGenerator** — Data augmentation and preprocessing
- **Scikit-Learn** — Confusion matrix evaluation
- **Matplotlib** — Data visualization

---

> **Note:** I developed this project as part of my learning journey on the **BTK Academy** platform. While the documentation is in English for global accessibility, the code comments remain in Turkish as they reflect my original study notes.
