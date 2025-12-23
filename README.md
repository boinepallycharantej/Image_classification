# Facial Expression Recognition Using Deep Learning (9 Emotions)

This repository contains the implementation and experimental results of my **MATH 895 Research Project** at **San Francisco State University**, focused on building a **Facial Expression Recognition (FER)** system using deep learning.

The project evaluates and compares multiple architectures—including a **Custom CNN**, **VGG16 (Fine-Tuned)**, and **YOLOv8**—for classifying facial images into **nine emotion categories**.

---

## 📌 Project Overview

Facial expression recognition plays a crucial role in applications such as:
- Human–Computer Interaction
- Mental health assessment
- Surveillance and behavioral analysis
- Affective computing

The objective of this project is to identify a model that balances **accuracy, generalization, and inference efficiency** for real-world emotion recognition tasks.

---

## 😊 Emotion Classes

The models classify facial expressions into the following **9 emotions**:

- Angry  
- Contempt  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  
- Other  

---

## 🧠 Models Implemented

### 1. YOLOv8 (Ultralytics)
- Detection + classification pipeline
- Image size: `640 × 640`
- Best overall performance
- Suitable for real-time applications

### 2. Custom CNN (From Scratch)
- 3 convolutional layers (32 → 64 → 128)
- ReLU + MaxPooling + Dropout
- Lightweight baseline model

### 3. VGG16 (Fine-Tuned)
- Pretrained on ImageNet
- Frozen convolutional layers
- Custom dense head with dropout

---

## 📊 Dataset Details

- ~6,000 training images  
- ~1,500 validation images  
- ~1,000 test images  
- Organized into `train / val / test` directories  

### Preprocessing & Augmentation
- Image resizing (128×128 for CNN & VGG16, 640×640 for YOLOv8)
- Pixel normalization
- Random flip, rotation, brightness adjustment
- Label encoding based on folder structure

---

## 📈 Performance Summary

| Model | Val Accuracy | Precision | Recall | F1-Score |
|------|-------------|-----------|--------|---------|
| CNN | 70.5% | 0.72 | 0.68 | 0.70 |
| VGG16 (FT) | 60% | 0.63 | 0.61 | 0.60 |
| YOLOv8 | **84.5%** | **0.80** | **0.77** | **0.79** |

- YOLOv8 achieved the **best generalization**
- CNN showed **overfitting**
- VGG16 benefited from **transfer learning stability**

---

## ⚙️ Setup Instructions

```bash
conda env create -f environment.yml
conda activate image_classification

▶️ Training & Evaluation
# Train CNN
python cnn_train.py

# Train VGG16
python vgg16_train.py

# Evaluate fine-tuned VGG16
python vgg16_evaluate_finetuned.py


YOLOv8 training follows Ultralytics configuration via data.yaml.

🔮 Future Work

Hybrid YOLOv8 + VGG feature extraction

Attention mechanisms (SE blocks, Vision Transformers)

Improved handling of class imbalance

Edge deployment for real-time FER systems
