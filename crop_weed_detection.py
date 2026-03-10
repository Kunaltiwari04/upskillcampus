import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Dataset path
DATASET_PATH = "dataset"

# Image size
IMG_SIZE = 128

def load_dataset():
    images = []
    labels = []

    classes = ["crop", "weed"]

    for label, category in enumerate(classes):
        path = os.path.join(DATASET_PATH, category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                images.append(img.flatten())
                labels.append(label)

            except:
                pass

    return np.array(images), np.array(labels)


def train_model():

    print("Loading dataset...")
    X, y = load_dataset()

    print("Dataset loaded:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear"))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    print("Model trained")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)
    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "crop_weed_model.pkl")

    print("\nModel saved as crop_weed_model.pkl")


if __name__ == "__main__":
    train_model()