# run_recognition.py - Complete working solution
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque
import os

def main():
    print("=" * 60)
    print("SIGN LANGUAGE PHRASE RECOGNITION")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('phrase_model_final.pkl'):
        print("\n❌ Model not found!")
        print("Please train the model first:")
        print("  python train_phrase_final.py")
        return
    
    # Load model
    print("\n📂 Loading model...")
    model = joblib.load('phrase_model_final.pkl')
    encoder = joblib.load('phrase_encoder_final.pkl')
    print(f"✅ Ready to recognize {len(encoder.classes_)} phrases")
    
    # Initialize MediaPipe
    print("🖐️ Starting camera...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    
    # Buffer
    buffer = deque(maxlen=30)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("\n✅ Ready! Press 'q' to quit")
    print("📝 Perform a sign and hold for 2 seconds\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        # Process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    display, hand_lm, mp_hands.HAND_CONNECTIONS)
        
        # Extract features
        features = []
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                for lm in hand_lm.landmark:
                    features.extend([lm.x, lm.y, lm.z])
        
        while len(features) < 126:
            features.append(0.0)
        features = features[:126]
        
        buffer.append(features)
        
        # Predict
        if len(buffer) == 30:
            seq = np.array(buffer).flatten().reshape(1, -1)
            pred = model.predict(seq)[0]
            phrase = encoder.classes_[pred]
            conf = max(model.predict_proba(seq)[0])
            
            if conf > 0.3:
                cv2.rectangle(display, (50, 80), (400, 150), (0, 0, 0), -1)
                cv2.putText(display, phrase.upper(), (60, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.putText(display, f"{conf:.0%}", (60, 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            buffer.clear()
        
        # Progress
        prog = len(buffer) / 30
        cv2.rectangle(display, (10, 30), (10 + int(prog * 300), 50), (0, 255, 0), -1)
        
        cv2.imshow('Sign Language Recognition', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()