import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter # Add this at the top

# 1. Setup
model = load_model("sign_language_model.h5")
IMG_SIZE = 64
CATEGORIES = ["A", "B", "C", "D", "E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
# Ensure this matches your folder count!

# --- SMOOTHING SETUP ---
predictions_history = []
history_length = 15 # Number of frames to average
# -----------------------

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 2. Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))

    # 3. Predict
    prediction = model.predict(reshaped, verbose=0)
    class_idx = np.argmax(prediction)
    
    # 4. Update History
    predictions_history.append(class_idx)
    if len(predictions_history) > history_length:
        predictions_history.pop(0) # Remove oldest prediction

    # 5. Get the most frequent prediction (The "Winner")
    most_common = Counter(predictions_history).most_common(1)[0][0]
    
    # Safety check for the list index
    if most_common < len(CATEGORIES):
        stable_label = CATEGORIES[most_common]
    else:
        stable_label = "Scanning..."

    # 6. Display
    cv2.putText(frame, f"Prediction: {stable_label}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    cv2.imshow("Stable Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
