import os
import pickle
import cv2
import mediapipe as mp

DATA_DIR = './dataset'  # Point to your images directory

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, min_detection_confidence=0.5, max_num_hands=1
)

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
  dir_path = os.path.join(DATA_DIR, dir_)
  if not os.path.isdir(dir_path):
    continue

  print(f'Extracting landmarks for class: {dir_}')
  for img_path in os.listdir(dir_path):
    img = cv2.imread(os.path.join(dir_path, img_path))
    if img is None:
      continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        # Wrist anchor normalization
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y

        coords = []
        for lm in hand_landmarks.landmark:
          coords.extend([lm.x - base_x, lm.y - base_y])

        max_span = max(map(abs, coords)) or 1.0
        normalized_coords = [val / max_span for val in coords]

        if len(normalized_coords) == 42:
          data.append(normalized_coords)
          labels.append(dir_)

hands.close()

with open('data.pickle', 'wb') as f:
  pickle.dump({'data': data, 'labels': labels}, f)

print(f'Extraction complete! Total samples: {len(data)}')