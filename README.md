# 👻 Ghost Detector — Spooky Image Classifier with Haunted Level + Grad-CAM  
**Day 10 — 30 Days of Data Science**

This project classifies images as **Spooky** vs **Normal**, and outputs a fun metric called **Haunted Level (%)**.  
Additionally, it uses **Grad-CAM** to visualize *what parts of the image* influenced the model's decision.

---

## 🎃 Overview

| Feature | Description |
|--------|-------------|
| **Model** | MobileNetV2 (Transfer Learning, pretrained on ImageNet) |
| **Dataset** | LOL Low-Light Dataset (low-light → spooky, normal-light → normal) |
| **Output** | Prediction + Haunted Level score |
| **Explainability** | Grad-CAM heatmaps |
| **Environment** | Works on Kaggle / Jupyter / GPU or CPU |

---

## 🗂️ Dataset Structure

The notebook automatically discovers LOL dataset folders and assigns labels:

data/
raw/
spooky/     # images from “low” exposure folders
normal/     # images from “high” or “normal” exposure folders
split/
train/      # 70%
val/        # 15%
test/       # 15%

---

## 🧠 Model & Training

- Base model: **MobileNetV2**, last classifier layer replaced.
- Optimizer: **AdamW**
- Loss: **CrossEntropy**
- Scheduler: **ReduceLROnPlateau**
- Early stopping based on validation loss.
- Data augmentation on training set.

```python
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
```
