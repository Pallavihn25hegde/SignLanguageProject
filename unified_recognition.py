# unified_recognition_complete.py
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque

print("=" * 60)
print("UNIFIED SIGN LANGUAGE RECOGNITION")
print("LETTERS + PHRASES")
print("=" * 60)

# Load model
print("\n📂 Loading unified model...")
try:
    model = joblib.load('unified_model_complete.pkl')
    label_encoder = joblib.load('unified_label_encoder.pkl')
    print(f"✅ Model loaded successfully!")
    print(f"✅ Can recognize {len(label_encoder.classes_)} classes")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nPlease run: python complete_integration_fixed.py")
    exit()

# Initialize MediaPipe
print("\n🖐️ Initializing camera...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Recognition state
mode = 'auto'  # auto, letter, phrase
phrase_buffer = deque(maxlen=30)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("\n✅ Camera ready!")
print("\n🎮 CONTROLS:")
print("   ┌─────────────────────────────────┐")
print("   │  'l' - LETTER mode (single letters)  │")
print("   │  'p' - PHRASE mode (continuous)      │")
print("   │  'a' - AUTO mode (detects both)      │")
print("   │  'q' - QUIT                          │")
print("   └─────────────────────────────────┘")
print("\n📝 Hold signs for 2-3 seconds for phrases")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    display = frame.copy()
    
    # Extract hand landmarks
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
    
    # Extract features
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    
    # Pad to 126 features
    while len(landmarks) < 126:
        landmarks.append(0.0)
    landmarks = landmarks[:126]
    
    # Mode-specific recognition
    if mode == 'letter':
        # Letter mode: single frame
        if np.sum(np.abs(landmarks)) > 0.1:
            features = np.array(landmarks).reshape(1, -1)
            pred = model.predict(features)[0]
            label = label_encoder.inverse_transform([pred])[0]
            
            if label.startswith('LETTER_'):
                letter = label.replace('LETTER_', '')
                # Display result
                cv2.rectangle(display, (50, 100), (350, 180), (0, 0, 0), -1)
                cv2.rectangle(display, (50, 100), (350, 180), (0, 255, 0), 2)
                cv2.putText(display, f"LETTER: {letter}", (60, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    elif mode == 'phrase':
        # Phrase mode: accumulate frames
        if np.sum(np.abs(landmarks)) > 0.1:
            phrase_buffer.append(landmarks)
            
            # Progress bar
            progress = len(phrase_buffer) / 30
            cv2.rectangle(display, (10, 30), (10 + int(progress * 300), 50), (0, 255, 0), -1)
            cv2.putText(display, f"Recording: {len(phrase_buffer)}/30", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Make prediction when buffer is full
            if len(phrase_buffer) == 30:
                sequence = np.array(phrase_buffer).flatten().reshape(1, -1)
                pred = model.predict(sequence)[0]
                label = label_encoder.inverse_transform([pred])[0]
                
                if label.startswith('PHRASE_'):
                    phrase = label.replace('PHRASE_', '')
                    confidence = max(model.predict_proba(sequence)[0])
                    
                    # Display result
                    cv2.rectangle(display, (50, 100), (450, 180), (0, 0, 0), -1)
                    cv2.rectangle(display, (50, 100), (450, 180), (0, 255, 255), 2)
                    cv2.putText(display, f"PHRASE: {phrase.upper()}", (60, 145),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.putText(display, f"Confidence: {confidence:.1%}", (60, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                phrase_buffer.clear()
        else:
            phrase_buffer.clear()
    
    else:  # auto mode
        if np.sum(np.abs(landmarks)) > 0.1:
            phrase_buffer.append(landmarks)
            
            # Progress bar
            progress = len(phrase_buffer) / 30
            cv2.rectangle(display, (10, 30), (10 + int(progress * 300), 50), (0, 255, 0), -1)
            
            if len(phrase_buffer) == 30:
                sequence = np.array(phrase_buffer).flatten().reshape(1, -1)
                pred = model.predict(sequence)[0]
                label = label_encoder.inverse_transform([pred])[0]
                
                if label.startswith('PHRASE_'):
                    phrase = label.replace('PHRASE_', '')
                    confidence = max(model.predict_proba(sequence)[0])
                    
                    cv2.rectangle(display, (50, 100), (450, 180), (0, 0, 0), -1)
                    cv2.rectangle(display, (50, 100), (450, 180), (0, 255, 255), 2)
                    cv2.putText(display, f"PHRASE: {phrase.upper()}", (60, 145),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.putText(display, f"Confidence: {confidence:.1%}", (60, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                phrase_buffer.clear()
        else:
            # Try letter recognition when no hand movement
            if len(phrase_buffer) == 0 and np.sum(np.abs(landmarks)) > 0.1:
                features = np.array(landmarks).reshape(1, -1)
                pred = model.predict(features)[0]
                label = label_encoder.inverse_transform([pred])[0]
                
                if label.startswith('LETTER_'):
                    letter = label.replace('LETTER_', '')
                    cv2.rectangle(display, (50, 100), (350, 180), (0, 0, 0), -1)
                    cv2.rectangle(display, (50, 100), (350, 180), (0, 255, 0), 2)
                    cv2.putText(display, f"LETTER: {letter}", (60, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                phrase_buffer.clear()
    
    # Display current mode
    mode_colors = {'auto': (0, 255, 0), 'letter': (255, 255, 0), 'phrase': (0, 255, 255)}
    cv2.putText(display, f"MODE: {mode.upper()}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_colors.get(mode, (255, 255, 255)), 2)
    
    # Instructions
    cv2.putText(display, "l:Letter | p:Phrase | a:Auto | q:Quit",
               (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Unified Sign Language Recognition', display)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        mode = 'letter'
        phrase_buffer.clear()
        print("\n📝 Switched to LETTER mode")
    elif key == ord('p'):
        mode = 'phrase'
        phrase_buffer.clear()
        print("\n🎬 Switched to PHRASE mode")
    elif key == ord('a'):
        mode = 'auto'
        phrase_buffer.clear()
        print("\n🔄 Switched to AUTO mode")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Recognition stopped")
