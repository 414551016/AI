import os
import cv2
import joblib
import numpy as np
from pathlib import Path
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =========================
# 參數設定
# =========================
DATASET_DIR = "cropped_dataset"     # 資料夾，底下要有 occupied / empty
MODEL_OUTPUT = "hog_svm_model.joblib"

IMG_SIZE = (128, 128)               # 建議固定大小
RANDOM_SEED = 42
N_SPLITS = 5

# HOG 參數
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"


# =========================
# 讀取資料
# =========================
def load_images_from_folder(folder_path, label):
    """
    從指定資料夾讀取所有圖片，回傳 images, labels, paths
    """
    folder = Path(folder_path)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    images = []
    labels = []
    paths = []

    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in image_exts:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 無法讀取圖片: {img_path}")
            continue

        images.append(img)
        labels.append(label)
        paths.append(str(img_path))

    return images, labels, paths


def load_dataset(dataset_dir):
    """
    載入 cropped_dataset/occupied 和 cropped_dataset/empty
    label:
      occupied = 1
      empty    = 0
    """
    dataset_dir = Path(dataset_dir)

    occupied_dir = dataset_dir / "occupied"
    empty_dir = dataset_dir / "empty"

    if not occupied_dir.exists():
        raise FileNotFoundError(f"找不到資料夾: {occupied_dir}")
    if not empty_dir.exists():
        raise FileNotFoundError(f"找不到資料夾: {empty_dir}")

    occ_images, occ_labels, occ_paths = load_images_from_folder(occupied_dir, 1)
    emp_images, emp_labels, emp_paths = load_images_from_folder(empty_dir, 0)

    images = occ_images + emp_images
    labels = occ_labels + emp_labels
    paths = occ_paths + emp_paths

    print(f"[INFO] occupied: {len(occ_images)}")
    print(f"[INFO] empty   : {len(emp_images)}")
    print(f"[INFO] total   : {len(images)}")

    return images, np.array(labels), paths


# =========================
# 特徵處理
# =========================
def preprocess_image(img, img_size=(128, 128)):
    """
    影像前處理：
    1. resize
    2. 轉灰階
    """
    img_resized = cv2.resize(img, img_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return gray


def extract_hog_features(gray_img):
    """
    從灰階圖擷取 HOG 特徵
    """
    features = hog(
        gray_img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        transform_sqrt=True,
        feature_vector=True
    )
    return features


def build_feature_matrix(images, img_size=(128, 128)):
    """
    將所有影像轉成 HOG 特徵矩陣
    """
    feature_list = []

    for img in images:
        gray = preprocess_image(img, img_size)
        feat = extract_hog_features(gray)
        feature_list.append(feat)

    X = np.array(feature_list)
    return X


# =========================
# 交叉驗證
# =========================
def run_cross_validation(X, y, n_splits=5, random_seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    acc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n========== Fold {fold} ==========")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=random_seed
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        acc_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=["empty", "occupied"],
            digits=4,
            zero_division=0
        ))

    print("\n========== Cross-validation Summary ==========")
    print(f"Mean Accuracy : {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Mean Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Mean Recall   : {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"Mean F1-score : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    overall_cm = confusion_matrix(all_y_true, all_y_pred)
    print("\nOverall Confusion Matrix (all folds combined):")
    print(overall_cm)

    print("\nOverall Classification Report:")
    print(classification_report(
        all_y_true,
        all_y_pred,
        target_names=["empty", "occupied"],
        digits=4,
        zero_division=0
    ))

    results = {
        "mean_accuracy": np.mean(acc_scores),
        "std_accuracy": np.std(acc_scores),
        "mean_precision": np.mean(precision_scores),
        "std_precision": np.std(precision_scores),
        "mean_recall": np.mean(recall_scores),
        "std_recall": np.std(recall_scores),
        "mean_f1": np.mean(f1_scores),
        "std_f1": np.std(f1_scores),
        "overall_confusion_matrix": overall_cm,
    }

    return results


# =========================
# 訓練最終模型並存檔
# =========================
def train_final_model(X, y, output_path):
    model = LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=RANDOM_SEED
    )
    model.fit(X, y)

    payload = {
        "model": model,
        "img_size": IMG_SIZE,
        "class_names": {0: "empty", 1: "occupied"},
        "hog_params": {
            "orientations": HOG_ORIENTATIONS,
            "pixels_per_cell": HOG_PIXELS_PER_CELL,
            "cells_per_block": HOG_CELLS_PER_BLOCK,
            "block_norm": HOG_BLOCK_NORM,
        }
    }

    joblib.dump(payload, output_path)
    print(f"\n[INFO] 最終模型已儲存: {output_path}")


# =========================
# 主程式
# =========================
def main():
    print("Loading dataset...")
    images, labels, paths = load_dataset(DATASET_DIR)

    if len(images) == 0:
        raise RuntimeError("資料集沒有任何圖片，請檢查 cropped_dataset/occupied 和 cropped_dataset/empty")

    print("\nExtracting HOG features...")
    X = build_feature_matrix(images, IMG_SIZE)
    y = labels

    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Labels shape        : {y.shape}")

    print("\nRunning 5-fold cross-validation...")
    results = run_cross_validation(X, y, n_splits=N_SPLITS, random_seed=RANDOM_SEED)

    print("\nTraining final model on all data...")
    train_final_model(X, y, MODEL_OUTPUT)

    print("\n[DONE] HOG + SVM training completed.")


if __name__ == "__main__":
    main()