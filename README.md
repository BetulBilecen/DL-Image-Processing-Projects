# 🧠 Deep Learning & Computer Vision Portfolio

A curated collection of deep learning and computer vision applications developed to solve real-world problems. This repository focuses on implementing advanced image processing techniques, neural network architectures, and real-time object tracking systems.

## 📂 Projects Overview

Below is a detailed summary of the projects included in this repository. Click on any project folder for comprehensive documentation, model architectures, and visual results.

| # | Project Folder | Description | Key Technologies |
| :---: | :--- | :--- | :--- |
| **01** | `01-MNIST-Digit-Classification` | Developed a 4-layer ANN to classify digits. Implemented advanced **OpenCV preprocessing** (Gaussian Blur, Canny Edge Detection) to sharpen features, achieving a **96.09% accuracy** over 50 epochs. | `TensorFlow`, `OpenCV`, `ANN` |
| **02** | `02-Flower-Type-Classification` | Built a hierarchical CNN pipeline to classify 5 flower species from RGB images. Utilized dynamic **Data Augmentation** and `ReduceLROnPlateau` for optimal weight updates. | `Keras`, `CNNs`, `Augmentation` |
| **03** | `03-Pneumonia-Detection-in-Chest-X-Rays` | Applied Transfer Learning with a fine-tuned **DenseNet121** model to detect pneumonia from X-ray scans. Achieved a critical **97.9% medical recall rate**, successfully minimizing false negatives. | `Transfer Learning`, `DenseNet121` |
| **04** | `04-Fashion-Product-Generation` | Engineered a Deep Convolutional **GAN** to synthesize realistic fashion items from random noise. Monitored adversarial loss to identify and document "mode collapse" training instabilities. | `GANs`, `Generative AI` |
| **05** | `05-Traffic-Sign-Detection-YOLO` | Trained a YOLOv8 Nano model to detect 15 traffic sign classes (speed limits, lights). Applied Mosaic augmentation, achieving **81.2% mAP50** with an inference speed of ~141ms on CPU. | `YOLOv8`, `PyTorch` |
| **06** | `06-YOLOv8-Vehicle-Tracker-Bytetrack` | Created a robust, real-time video processing pipeline combining YOLOv8 detection with the **ByteTrack algorithm** to assign and maintain persistent multi-object tracking IDs across frames. | `YOLOv8`, `ByteTrack`, `OpenCV` |

## 🛠️ Technologies & Tools Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## 👩‍💻 Author
**Betül Bilecen** | *Computer Engineering Student & Aspiring Data Scientist*
