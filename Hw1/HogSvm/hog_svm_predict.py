import cv2
import joblib
from skimage.feature import hog


def preprocess_image(img, img_size):
    img_resized = cv2.resize(img, img_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return gray


def extract_hog_features(gray_img, hog_params):
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


def predict_image(image_path, model_path="hog_svm_model.joblib"):
    payload = joblib.load(model_path)

    model = payload["model"]
    img_size = tuple(payload["img_size"])
    class_names = payload["class_names"]
    hog_params = payload["hog_params"]

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = preprocess_image(img, img_size)
    feat = extract_hog_features(gray, hog_params).reshape(1, -1)

    pred = model.predict(feat)[0]
    print(f"Prediction: {class_names[pred]}")


if __name__ == "__main__":
    #test_image = "IMG20260328134711_Occupied.jpg"   # 改成你的測試圖
    test_image = "IMG20260328140416_Empty.jpg"   # 改成你的測試圖
    predict_image(test_image)