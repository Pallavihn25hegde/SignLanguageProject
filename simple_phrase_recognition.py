# simple_phrase_recognition.py
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque

print("=" * 50)
print("REAL-TIME PHRASE RECOGNITION")
print("=" * 50)

# Load model
model = joblib.load('phrase_model_final.pkl')
encoder = joblib.load('phrase_encoder_final.pkl')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Buffer for frames
frame_buffer = deque(maxlen=30)
cap = cv2.VideoCapture(0)

print("\n🎥 Camera started. Press 'q' to quit.")
print("📝 Perform a sign language phrase for 2-3 seconds...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    display = frame.copy()
    
    # Extract landmarks
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Draw hand landmarks
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
    
    # Pad to 126 features
    while len(landmarks) < 126:
        landmarks.append(0.0)
    landmarks = landmarks[:126]
    
    frame_buffer.append(landmarks)
    
    # Make prediction when buffer is full
    if len(frame_buffer) == 30:
        sequence = np.array(frame_buffer)
        sequence_flat = sequence.flatten().reshape(1, -1)
        
        prediction = model.predict(sequence_flat)[0]
        phrase = encoder.classes_[prediction]
        confidence = max(model.predict_proba(sequence_flat)[0])
        
        # Display prediction
        if confidence > 0.3:  # Threshold
            cv2.putText(display, f"Phrase: {phrase}", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(display, f"Confidence: {confidence:.2f}", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Show buffer progress
    progress = len(frame_buffer) / 30
    cv2.rectangle(display, (10, 30), (10 + int(progress * 300), 50), (0, 255, 0), -1)
    cv2.putText(display, f"Buffer: {len(frame_buffer)}/30", (10, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Phrase Recognition', display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✅ Recognition stopped")