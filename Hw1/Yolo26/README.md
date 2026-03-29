# Parking Occupancy Detection  
### Based on a Self-Collected Dataset  
### (HOG+SVM, YOLO26, and Single-Class YOLO)

---

## 📌 Project Overview

This project aims to detect parking space occupancy (occupied / empty) using a self-collected dataset, following the requirements of the Artificial Intelligence course assignment.

Three supervised learning approaches are implemented and compared:

1. HOG + SVM (traditional machine learning)
2. YOLO26 (deep learning object detection)
3. Single-Class YOLO (occupied-only detection)

---

## 📊 Dataset Description

### Data Type
- Parking lot images (RGB images)

### Classes
- `Empty`
- `Occupied`

### Data Source
- Self-collected using mobile camera
- Fixed camera angle in parking lot

### Annotation Tool
- Roboflow (YOLO format)

### Dataset Structure

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
├── test/
└── data.yaml
```

### Cropped Dataset (for HOG+SVM)

```
cropped_dataset/
├── occupied/
└── empty/
```

### Dataset Statistics

| Class     | Count |
|----------|------:|
| Occupied | 471   |
| Empty    | 387   |
| Total    | 858   |

---

## 🧠 Methods

### 1. HOG + SVM
- Input: cropped parking space image
- Feature: Histogram of Oriented Gradients (HOG)
- Classifier: Linear SVM

### 2. YOLO26
- Input: full parking lot image
- Output: bounding boxes + class labels
- Model: YOLO26n

### 3. Single-Class YOLO
- Only label: `Occupied`
- Empty spaces inferred from absence of detection

---

## ⚙️ Installation

```
pip install ultralytics
pip install opencv-python scikit-learn joblib
```

---

## 🚀 How to Run

### 1. Train YOLO26

```
python yolo26_train.py
```

### 2. Validate YOLO26

```
python yolo26_val.py
```

### 3. HOG+SVM Training

```
python hog_svm_train.py
```

### 4. HOG+SVM Prediction

```
python hog_svm_predict_all.py
```

---

## 📈 Results

### HOG + SVM

| Metric     | Value |
|-----------|------:|
| Accuracy  | 1.000 |
| Precision | 1.000 |
| Recall    | 1.000 |
| F1-score  | 1.000 |

---

### YOLO26

| Metric        | Value |
|--------------|------:|
| Precision    | 0.988 |
| Recall       | 0.973 |
| mAP@0.5      | 0.994 |
| mAP@0.5:0.95 | 0.911 |

---

## ⚠️ Discussion

### Data Leakage
- Crops from same image may appear in both train/test
- Leads to overly optimistic results

### Limitations
- Single parking lot
- Limited lighting variation
- No night data

---

## 📷 Figures

(Add images here)

- Figure 1: Original parking lot image
- Figure 2: Roboflow annotation
- Figure 3: Cropped samples (occupied vs empty)
- Figure 4: Confusion matrix
- Figure 5: YOLO training curve
- Figure 6: YOLO detection results

---

## 📌 Conclusion

- HOG+SVM works well in fixed environments
- YOLO is more robust and scalable
- Single-class YOLO reduces labeling cost

---

## 🔮 Future Work

- Night-time detection
- Multi-location dataset
- Real-time system
- Handling occlusion and multi-slot parking

---

## 📎 Author

Name: [Your Name]  
Course: Artificial Intelligence  

