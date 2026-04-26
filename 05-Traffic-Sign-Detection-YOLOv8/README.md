# 🚦 Traffic Sign Detection using YOLOv8

In this project, we focus on one of the fundamental tasks of autonomous driving systems: **traffic sign detection and classification**. Using the Ultralytics YOLOv8 architecture, the model is capable of detecting and classifying 15 different traffic signs and signals with near real-time performance.

---

## 🚀 Project Summary

* **Model Architecture:** YOLOv8n (Nano)
* **Dataset:** Traffic Signs Detection (15 Classes) — Roboflow/Kaggle
* **Task:** Object Detection (Bounding Box + Classification)
* **mAP50 (Overall):** 81.2%
* **mAP50-95 (Overall):** 69.2%
* **Inference Speed:** ~141.6 ms (CPU - 12th Gen Intel Core i7-1255U)

---

## 🧠 1. Model Architecture

The **YOLOv8n** model was selected due to its balance between speed and efficiency. The architecture is optimized for both feature extraction (Backbone) and object detection (Head).

| Feature          | Details        |
| :--------------- | :------------- |
| **Model Layers** | 130 Layers     |
| **Parameters**   | 3,013,773 (3M) |
| **GFLOPs**       | 8.2            |
| **Input Size**   | 640x640        |

---

## 📊 2. Detection Classes (15 Classes)

The model is trained to detect the following 15 classes:

* **Traffic Lights:** Green Light, Red Light
* **Speed Limits:** 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 km/h
* **Others:** Stop Sign

---

## 📈 3. Training Results

The training process was conducted for 10 epochs, and the model's performance improved consistently over time.

| Epoch  | Box Loss   | Class Loss | DFL Loss  | mAP50     | mAP50-95  |
| :----- | :--------- | :--------- | :-------- | :-------- | :-------- |
| 1      | 0.9156     | 4.197      | 1.253     | 0.195     | 0.153     |
| 5      | 0.8170     | 2.184      | 1.104     | 0.513     | 0.421     |
| 8      | 0.7020     | 1.354      | 1.019     | 0.780     | 0.660     |
| **10** | **0.6400** | **1.089**  | **0.980** | **0.812** | **0.692** |

---

## 🖼️ 4. Inference & Validation Results

The model achieved a balanced improvement in both precision and recall. Notable class-wise performance:

* **Stop Sign:** 98.5% mAP50 (Highest performance)
* **Speed Limit 70:** 90.9% mAP50
* **Speed Limit 40:** 90.8% mAP50
* **Speed Limit 20:** 88.3% mAP50

> **Example Inference:** In `Test2.png`, the model successfully detected one **Speed Limit 30** sign and one **Speed Limit 40** sign.

---

## ⚙️ 5. Data Augmentation & Preprocessing

To improve the model’s generalization capability, the following techniques were applied during training:

* **Mosaic Augmentation:** Combining four different images into one to create more complex scenes
* **HSV Adjustments:** Modifying hue, saturation, and brightness
* **Horizontal Flip:** Random horizontal flipping (with 0.5 probability)
* **Scaling:** Resizing images by up to 50%

---

## ⚙️ 6. Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/BetulBilecen/DL-Image-Processing-Projects.git
```

### 2. Navigate to the project directory

```bash
cd 05-Traffic-Sign-Detection-YOLOv8
```

### 3. Install dependencies

```bash
pip install ultralytics opencv-python torch
```

### 4. Run inference

Ensure your `config.json` (or paths inside your script) is correctly set, then run:

```bash
python test.py
```

---

## 🧱 7. Technologies Used

* **Python** — Core programming language
* **Ultralytics YOLOv8** — Object detection framework
* **PyTorch** — Deep learning backend
* **OpenCV** — Image processing and visualization
* **Matplotlib** — Data analysis and plotting

---

## 💡 Notes

* The project is optimized for CPU usage but performs significantly better with GPU.
* For full training, using Google Colab or a CUDA-enabled GPU is recommended.
* A lightweight configuration is provided for demonstration purposes.
