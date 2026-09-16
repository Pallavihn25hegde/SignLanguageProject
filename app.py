import os
import sys
import threading
import pickle
import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# --- 1. MEDIA PIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,
    max_num_hands=1
)

# --- 2. LOAD MODEL ---
model = None
if os.path.exists('./model.p'):
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
else:
    print("WARNING: './model.p' not found. Please run train_classifier.py first.")

# --- 3. SPEECH HELPER ---
def speak_async(text):
    """Speaks text in a separate thread to prevent webcam freezing."""
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

# --- 4. WEBCAM LOOP ---
current_sentence = ""

def start_webcam():
    global current_sentence

    if model is None:
        messagebox.showerror(
            "Model Error",
            "model.p not found! Please run train_classifier.py before starting the camera."
        )
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not access the webcam (index 0).")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        current_prediction = ""

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data_aux, x_, y_ = [], [], []
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                current_prediction = str(prediction[0]).split('-')[0]

                # Visual Feedback for current gesture
                cv2.putText(
                    frame,
                    current_prediction,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (0, 255, 0),
                    3
                )

        # Bottom banner for accumulated sentence
        cv2.rectangle(frame, (0, H - 50), (W, H), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Sentence: {current_sentence}",
            (10, H - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow('ASL Translator', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE: Add character & speak letter
            if current_prediction:
                current_sentence += current_prediction
                speak_async(current_prediction)
        elif key == ord('v'):  # V: Speak entire sentence
            speak_async(current_sentence)
        elif key == ord('c'):  # C: Clear sentence
            current_sentence = ""
        elif key == ord('q'):  # Q: Quit camera
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 5. GUI WINDOW ---
root = tk.Tk()
root.title("Pallavi - ASL Project")
root.geometry("400x350")
root.resizable(False, False)

tk.Label(root, text="SIGN TO SPEECH", font=("Arial", 20, "bold")).pack(pady=20)
tk.Label(root, text="Developed by Pallavi", font=("Arial", 10)).pack()

tk.Button(
    root,
    text="START CAMERA",
    command=start_webcam,
    bg="green",
    fg="white",
    font=("Arial", 12, "bold"),
    padx=20,
    pady=10
).pack(pady=30)

tk.Label(
    root,
    text="Controls:\n[Space] Add Letter | [V] Speak Sentence\n[C] Clear | [Q] Exit Camera",
    font=("Arial", 9),
    fg="gray"
).pack(pady=10)

root.mainloop()