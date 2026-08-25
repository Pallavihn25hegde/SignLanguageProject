
import numpy as np
from tensorflow.keras.models import load_model

# CONFIGURATION - Match these to your training script!
IMG_SIZE = 64  # Or 128, whatever you used in training
MODEL_PATH = 'sign_language_model.h5'
IMAGE_PATH = r'E:\SignLanguageProject\test.jpg'
CATEGORIES = ['A', 'B', 'C','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
# Update this to your actual folder names

# 1. Load Model
model = load_model("sign_language_model.h5")

# 2. Load and Preprocess Image
img = cv2.imread("1.jpg")

if img is not None:
    # Resize to match model input
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Scale pixels to 0-1 (Neural networks love this)
    img = img.astype("float32") / 255.0
    
    # CRITICAL STEP: Add the batch dimension
    # Changes shape from (64, 64, 3) to (1, 64, 64, 3)
    img = np.expand_dims(img, axis=0)
    
    # 3. Predict
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    
    print(f"Prediction: {CATEGORIES[class_idx]}")
    print(f"Confidence: {np.max(prediction) * 100:.2f}%")
else:
    print("Failed to load image. Check the path again!")
