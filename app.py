import os
import pickle
import sys
import threading
import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Suppress TensorFlow/protobuf warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1,
)

# --- 2. LOAD MODEL ---
model = None
MODEL_PATH = './model.p'

if os.path.exists(MODEL_PATH):
  with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']
else:
  print(f"WARNING: '{MODEL_PATH}' not found. Please run train_model.py first.")


# --- 3. LIGHTING ENHANCEMENT FUNCTION ---
def enhance_lighting(image):
  """Applies CLAHE on the L-channel in LAB color space to balance shadows and highlights."""
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  l_channel, a_channel, b_channel = cv2.split(lab)

  # CLAHE equalizes local contrast without blowing out bright spots
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  cl = clahe.apply(l_channel)

  enhanced_lab = cv2.merge((cl, a_channel, b_channel))
  return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# --- 4. SPEECH HELPER ---
def speak_async(text):
  """Speaks text in a background thread to prevent GUI/camera lag."""

  def _speak():
    if text.strip():
      try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
      except Exception:
        pass

  threading.Thread(target=_speak, daemon=True).start()


# --- 5. WEBCAM LOOP ---
current_sentence = ''


def start_webcam():
  global current_sentence, model

  # Reload model in case it was updated while the GUI stayed open
  if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
      model = pickle.load(f)['model']

  if model is None:
    messagebox.showerror(
        'Model Error',
        'model.p not found! Please run train_model.py before starting the'
        ' camera.',
    )
    return

  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    messagebox.showerror('Camera Error', 'Could not access webcam (index 0).')
    return

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    # Apply lighting balance
    processed_frame = enhance_lighting(frame)
    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_prediction = ''

    if results.multi_hand_landmarks:
      hand_landmarks = results.multi_hand_landmarks[0]
      mp_drawing.draw_landmarks(
          frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
      )

      # Wrist-anchored & scale-invariant normalization
      base_x = hand_landmarks.landmark[0].x
      base_y = hand_landmarks.landmark[0].y

      coords = []
      for lm in hand_landmarks.landmark:
        coords.extend([lm.x - base_x, lm.y - base_y])

      max_span = max(map(abs, coords)) or 1.0
      normalized_coords = [val / max_span for val in coords]

      if len(normalized_coords) == 42:
        try:
          probabilities = model.predict_proba([np.asarray(normalized_coords)])[
              0
          ]
          max_idx = np.argmax(probabilities)
          confidence = probabilities[max_idx]

          if confidence >= 0.65:
            pred_raw = str(model.classes_[max_idx]).split('-')[0]
            current_prediction = pred_raw
            text_color = (0, 255, 0)
          else:
            current_prediction = '?'
            text_color = (0, 165, 255)
        except Exception:
          pred = model.predict([np.asarray(normalized_coords)])
          current_prediction = str(pred[0]).split('-')[0]
          text_color = (0, 255, 0)

        # Display letter only (no percentage)
        cv2.putText(
            frame,
            f'{current_prediction}',
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            text_color,
            3,
        )

    # Bottom banner for sentence output
    cv2.rectangle(frame, (0, H - 55), (W, H), (25, 25, 25), -1)
    cv2.putText(
        frame,
        f'Sentence: {current_sentence}',
        (15, H - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow('ASL Translator', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # SPACE: Append recognized letter
      if current_prediction and current_prediction != '?':
        current_sentence += current_prediction
        speak_async(current_prediction)
    elif key == ord('v'):  # V: Speak entire sentence
      speak_async(current_sentence)
    elif key == ord('c'):  # C: Clear sentence
      current_sentence = ''
    elif key == ord('q'):  # Q: Quit camera window
      break

  cap.release()
  cv2.destroyAllWindows()


# --- 6. GUI WINDOW ---
root = tk.Tk()
root.title('Sign Language Recognition')
root.geometry('420x360')
root.resizable(False, False)

tk.Label(
    root, text='SIGN TO SPEECH', font=('Helvetica', 18, 'bold'), fg='#111'
).pack(pady=20)

tk.Button(
    root,
    text='START CAMERA',
    command=start_webcam,
    bg='#2e7d32',
    fg='white',
    font=('Helvetica', 12, 'bold'),
    padx=25,
    pady=12,
    relief='flat',
).pack(pady=25)

tk.Label(
    root,
    text=(
        'Controls:\n[Space] Add Letter | [V] Speak Sentence\n[C] Clear | [Q]'
        ' Exit Camera'
    ),
    font=('Helvetica', 9),
    fg='#555',
    justify='center',
).pack(pady=10)

if __name__ == '__main__':
  root.mainloop()