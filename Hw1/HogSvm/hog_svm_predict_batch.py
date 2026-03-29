import cv2
import joblib
from pathlib import Path
from skimage.feature import hog


def preprocess_image(img, img_size):
    """
    將圖片 resize 並轉為灰階
    """
    img_resized = cv2.resize(img, img_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return gray


def extract_hog_features(gray_img, hog_params):
    """
    擷取 HOG 特徵
    """
    feat = hog(
        gray_img,
        orientations=hog_params["orientations"],
        pixels_per_cell=tuple(hog_params["pixels_per_cell"]),
        cells_per_block=tuple(hog_params["cells_per_block"]),
        block_norm=hog_params["block_norm"],
        transform_sqrt=True,
        feature_vector=True
    )
    return feat


def predict_folder(folder_path, model_path="hog_svm_model.joblib"):
    """
    預測資料夾內全部圖片
    """
    payload = joblib.load(model_path)

    model = payload["model"]
    img_size = tuple(payload["img_size"])
    class_names = payload["class_names"]
    hog_params = payload["hog_params"]

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"找不到資料夾: {folder_path}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([p for p in folder.iterdir() if p.suffix.lower() in image_exts])

    if not image_files:
        print(f"[WARN] 資料夾內沒有圖片: {folder_path}")
        return

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 無法讀取圖片: {img_path.name}")
            continue

        gray = preprocess_image(img, img_size)
        feat = extract_hog_features(gray, hog_params).reshape(1, -1)

        pred = model.predict(feat)[0]
        label = class_names[pred]

        print(f"Prediction: {img_path.name} + {label}")


if __name__ == "__main__":
    # 改成你的測試圖片資料夾
    test_folder = "test_images"
    predict_folder(test_folder)