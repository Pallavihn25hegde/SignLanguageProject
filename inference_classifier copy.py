import os
import sys

# --- STAGE 0: THE ULTIMATE FIXES ---
# 1. Force Protobuf to use the stable version
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Add the official library path (to fix the 'no attribute solutions' error)
# This points Python directly to where 'pip install' puts your libraries
user_site = os.path.join(os.environ['APPDATA'], 'Python', 'Python310', 'site-packages')
sys.path.append(user_site)

import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
import time

# --- INITIALIZE AI MODELS ---
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except:
    print("CRITICAL ERROR: 'model.p' not found in this folder!")

# Force load the solutions if they are being stubborn
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import hands as mp_hands_module
    from mediapipe.python.solutions import drawing_utils as mp_drawing_utils
    mp_hands = mp_hands_module
    mp_drawing = mp_drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1)

current_sentence = ""

def speak_text(text):
    if text:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except: pass

def start_webcam():
    global current_sentence
    cap = cv2.VideoCapture(0) # Try 0 or 1
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        current_prediction = ""

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            data_aux, x_, y_ = [], [], []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x); y_.append(y)
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                current_prediction = str(prediction[0]).split('-')[0]

                # Visual Feedback
                cv2.putText(frame, current_prediction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

        cv2.rectangle(frame, (0, H - 50), (W, H), (0, 0, 0), -1)
        cv2.putText(frame, f"Sentence: {current_sentence}", (10, H - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('ASL Translator', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE to add
            if current_prediction:
                current_sentence += current_prediction
                speak_text(current_prediction)
        elif key == ord('v'):  # V to speak
            speak_text(current_sentence)
        elif key == ord('q'):  # Q to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# --- GUI ---
root = tk.Tk()
root.title("Pallavi - ASL Project")
root.geometry("400x350")

tk.Label(root, text="SIGN TO SPEECH", font=("Arial", 20, "bold")).pack(pady=20)
tk.Label(root, text="Developed by Pallavi,Punya").pack()

tk.Button(root, text="START CAMERA", command=start_webcam, bg="green", fg="white", font=("Arial", 12), padx=20, pady=10).pack(pady=30)

root.mainloop()