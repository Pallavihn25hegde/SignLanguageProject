from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pickle

model = load_model("sign_language_model.h5")

IMG_SIZE = 64

img = cv2.imread("test.jpg")
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = img / 255.0
img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

prediction = model.predict(img)

print("Prediction class:", prediction.argmax())
