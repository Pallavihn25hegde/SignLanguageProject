import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1)

DATA_DIR = './dataset'
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

data = []
labels = []

print("Scanning for images...")
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(valid_extensions):
            img_path = os.path.join(root, file)
            # The folder containing the image will be the label (e.g., 'A', 'Hello', etc.)
            label = os.path.basename(root)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                data_aux = []
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(label)

print(f"Total valid samples collected: {len(data)}")

if len(data) == 0:
    print("WARNING: No landmarks found! Please verify where your gesture images are located.")
else:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Saved successfully to data.pickle!")