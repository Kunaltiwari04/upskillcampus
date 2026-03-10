import cv2
import numpy as np
import joblib

IMG_SIZE = 128

model = joblib.load("crop_weed_model.pkl")

def predict_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.flatten().reshape(1, -1)

    prediction = model.predict(img)

    if prediction[0] == 0:
        print("Prediction: Crop")
    else:
        print("Prediction: Weed")


if __name__ == "__main__":

    path = input("Enter image path: ")
    predict_image(path)