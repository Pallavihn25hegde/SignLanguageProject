# unified_recognition_real.py
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque

print("=" * 60)
print("REAL SIGN LANGUAGE RECOGNITION")
print("=" * 60)

# Load real models
models = joblib.load('unified_models_real.pkl')
encoders = joblib.load('unified_encoders_real.pkl')

print(f"\n✅ Loaded models:")
if 'letter' in models:
    print(f"   Letters: {len(encoders['letter'].classes_)} classes")
if 'phrase' in models:
    print(f"   Phrases: {len(encoders['phrase'].classes_)} classes")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,  # Higher confidence
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
phrase_buffer = deque(maxlen=30)
mode = 'auto'

print("\n✅ Camera ready!")
print("\n💡 TIPS FOR BETTER DETECTION:")
print("   1. Good lighting on your hands")
print("   2. Plain background")
print("   3. Keep hands in frame")
print("   4. Hold signs for 2 seconds")
print("\n🎮 Controls: l=Letter | p=Phrase | a=Auto | q=Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    display = frame.copy()
    
    # Process frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Extract features
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    
    while len(landmarks) < 126:
        landmarks.append(0.0)
    landmarks = landmarks[:126]
    
    has_hand = np.sum(np.abs(landmarks)) > 0.1
    
    # Show hand detection status
    color = (0, 255, 0) if has_hand else (0, 0, 255)
    cv2.circle(display, (30, 30), 10, color, -1)
    cv2.putText(display, "Hand Detected" if has_hand else "No Hand", (50, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Recognition logic
    if mode == 'letter' and has_hand and 'letter' in models:
        features = np.array(landmarks).reshape(1, -1)
        pred = models['letter'].predict(features)[0]
        letter = encoders['letter'].inverse_transform([pred])[0]
        confidence = max(models['letter'].predict_proba(features)[0])
        
        if confidence > 0.6:
            cv2.rectangle(display, (50, 100), (300, 170), (0, 0, 0), -1)
            cv2.putText(display, f"LETTER: {letter}", (60, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(display, f"{confidence:.0%}", (60, 165),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    elif mode == 'phrase' and has_hand and 'phrase' in models:
        phrase_buffer.append(landmarks)
        progress = len(phrase_buffer) / 30
        cv2.rectangle(display, (10, 60), (10 + int(progress * 300), 80), (0, 255, 0), -1)
        cv2.putText(display, f"Recording: {len(phrase_buffer)}/30", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(phrase_buffer) == 30:
            flat_seq = np.array(phrase_buffer).flatten().reshape(1, -1)
            pred = models['phrase'].predict(flat_seq)[0]
            phrase = encoders['phrase'].inverse_transform([pred])[0]
            confidence = max(models['phrase'].predict_proba(flat_seq)[0])
            
            if confidence > 0.5:
                cv2.rectangle(display, (50, 100), (450, 170), (0, 0, 0), -1)
                cv2.putText(display, f"PHRASE: {phrase.upper()}", (60, 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"{confidence:.0%}", (60, 165),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            phrase_buffer.clear()
    
    elif mode == 'auto':
        if has_hand:
            phrase_buffer.append(landmarks)
            progress = len(phrase_buffer) / 30
            cv2.rectangle(display, (10, 60), (10 + int(progress * 300), 80), (0, 255, 0), -1)
            
            # Try letter for short sequences
            if len(phrase_buffer) < 10 and 'letter' in models:
                features = np.array(landmarks).reshape(1, -1)
                pred = models['letter'].predict(features)[0]
                letter = encoders['letter'].inverse_transform([pred])[0]
                confidence = max(models['letter'].predict_proba(features)[0])
                
                if confidence > 0.7:
                    cv2.rectangle(display, (50, 100), (300, 170), (0, 0, 0), -1)
                    cv2.putText(display, f"LETTER: {letter}", (60, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Try phrase when buffer is full
            if len(phrase_buffer) == 30 and 'phrase' in models:
                flat_seq = np.array(phrase_buffer).flatten().reshape(1, -1)
                pred = models['phrase'].predict(flat_seq)[0]
                phrase = encoders['phrase'].inverse_transform([pred])[0]
                confidence = max(models['phrase'].predict_proba(flat_seq)[0])
                
                if confidence > 0.5:
                    cv2.rectangle(display, (50, 100), (450, 170), (0, 0, 0), -1)
                    cv2.putText(display, f"PHRASE: {phrase.upper()}", (60, 145),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                phrase_buffer.clear()
        else:
            phrase_buffer.clear()
    
    # Show mode
    cv2.putText(display, f"MODE: {mode.upper()}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Instructions
    cv2.putText(display, "l:Letter | p:Phrase | a:Auto | q:Quit",
               (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Sign Language Recognition', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        mode = 'letter'
        print("\n📝 Letter mode - Show single letters")
    elif key == ord('p'):
        mode = 'phrase'
        phrase_buffer.clear()
        print("\n🎬 Phrase mode - Hold signs for 2 seconds")
    elif key == ord('a'):
        mode = 'auto'
        phrase_buffer.clear()
        print("\n🔄 Auto mode - Detects both")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Done")