# unified_recognition_robust.py
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque
import time
import sys

print("=" * 60)
print("UNIFIED SIGN LANGUAGE RECOGNITION (ROBUST)")
print("=" * 60)

# Load models with error handling
print("\n📂 Loading models...")
try:
    models = joblib.load('unified_models.pkl')
    encoders = joblib.load('unified_encoders.pkl')
    print(f"✅ Models loaded")
    print(f"   Letter model: {'✓' if 'letter' in models else '✗'}")
    print(f"   Phrase model: {'✓' if 'phrase' in models else '✗'}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPlease train models first:")
    print("  python complete_integration_fixed_v2.py")
    sys.exit(1)

# Initialize MediaPipe
print("\n🖐️ Initializing MediaPipe...")
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe ready")
except Exception as e:
    print(f"❌ MediaPipe error: {e}")
    sys.exit(1)

# Initialize camera
print("\n📷 Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    print("Please check if camera is connected and not in use")
    sys.exit(1)
print("✅ Camera ready")

# Recognition state
mode = 'auto'  # auto, letter, phrase
phrase_buffer = deque(maxlen=30)
last_prediction = ""
last_prediction_time = 0
frame_count = 0

print("\n" + "=" * 60)
print("🎮 CONTROLS:")
print("   'l' - LETTER mode (single letters)")
print("   'p' - PHRASE mode (continuous signs)")
print("   'a' - AUTO mode (detects both)")
print("   'q' - QUIT")
print("=" * 60)
print("\n✅ Ready! Start signing...\n")

try:
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame")
            continue
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        frame_count += 1
        
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
        
        # Extract features (126 dimensions)
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        # Pad to 126 features
        while len(landmarks) < 126:
            landmarks.append(0.0)
        landmarks = landmarks[:126]
        
        has_hand = np.sum(np.abs(landmarks)) > 0.1
        
        # Mode-specific recognition
        if mode == 'letter' and 'letter' in models:
            # Letter mode: single frame recognition
            if has_hand:
                features = np.array(landmarks).reshape(1, -1)
                try:
                    pred = models['letter'].predict(features)[0]
                    letter = encoders['letter'].inverse_transform([pred])[0]
                    confidence = max(models['letter'].predict_proba(features)[0])
                    
                    if confidence > 0.5:
                        # Display result
                        cv2.rectangle(display, (50, 100), (350, 180), (0, 0, 0), -1)
                        cv2.rectangle(display, (50, 100), (350, 180), (0, 255, 0), 2)
                        cv2.putText(display, f"LETTER: {letter}", (60, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        cv2.putText(display, f"Confidence: {confidence:.1%}", (60, 175),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                except Exception as e:
                    pass
        
        elif mode == 'phrase' and 'phrase' in models:
            # Phrase mode: accumulate frames
            if has_hand:
                phrase_buffer.append(landmarks)
                
                # Progress bar
                progress = len(phrase_buffer) / 30
                cv2.rectangle(display, (10, 30), (10 + int(progress * 300), 50), (0, 255, 0), -1)
                cv2.putText(display, f"Recording: {len(phrase_buffer)}/30", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Make prediction when buffer is full
                if len(phrase_buffer) == 30:
                    try:
                        flat_sequence = np.array(phrase_buffer).flatten().reshape(1, -1)
                        pred = models['phrase'].predict(flat_sequence)[0]
                        phrase = encoders['phrase'].inverse_transform([pred])[0]
                        confidence = max(models['phrase'].predict_proba(flat_sequence)[0])
                        
                        if confidence > 0.5:
                            # Display result
                            cv2.rectangle(display, (50, 100), (450, 180), (0, 0, 0), -1)
                            cv2.rectangle(display, (50, 100), (450, 180), (0, 255, 255), 2)
                            cv2.putText(display, f"PHRASE: {phrase.upper()}", (60, 145),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                            cv2.putText(display, f"Confidence: {confidence:.1%}", (60, 170),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    except Exception as e:
                        pass
                    
                    phrase_buffer.clear()
            else:
                if len(phrase_buffer) > 0:
                    phrase_buffer.clear()
        
        else:  # auto mode
            if has_hand:
                phrase_buffer.append(landmarks)
                
                # Progress bar
                progress = len(phrase_buffer) / 30
                cv2.rectangle(display, (10, 30), (10 + int(progress * 300), 50), (0, 255, 0), -1)
                
                if len(phrase_buffer) == 30 and 'phrase' in models:
                    # Try phrase recognition
                    try:
                        flat_sequence = np.array(phrase_buffer).flatten().reshape(1, -1)
                        pred = models['phrase'].predict(flat_sequence)[0]
                        phrase = encoders['phrase'].inverse_transform([pred])[0]
                        confidence = max(models['phrase'].predict_proba(flat_sequence)[0])
                        
                        if confidence > 0.5:
                            cv2.rectangle(display, (50, 100), (450, 180), (0, 0, 0), -1)
                            cv2.rectangle(display, (50, 100), (450, 180), (0, 255, 255), 2)
                            cv2.putText(display, f"PHRASE: {phrase.upper()}", (60, 145),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                            cv2.putText(display, f"Confidence: {confidence:.1%}", (60, 170),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    except Exception as e:
                        pass
                    
                    phrase_buffer.clear()
                
                # Also try letter recognition for quick detection
                if len(phrase_buffer) < 10 and 'letter' in models:
                    try:
                        features = np.array(landmarks).reshape(1, -1)
                        pred = models['letter'].predict(features)[0]
                        letter = encoders['letter'].inverse_transform([pred])[0]
                        confidence = max(models['letter'].predict_proba(features)[0])
                        
                        if confidence > 0.7:
                            cv2.rectangle(display, (50, 100), (350, 180), (0, 0, 0), -1)
                            cv2.rectangle(display, (50, 100), (350, 180), (0, 255, 0), 2)
                            cv2.putText(display, f"LETTER: {letter}", (60, 150),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            cv2.putText(display, f"Confidence: {confidence:.1%}", (60, 175),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    except Exception as e:
                        pass
            else:
                if len(phrase_buffer) > 0:
                    phrase_buffer.clear()
        
        # Display current mode
        mode_colors = {'auto': (0, 255, 0), 'letter': (255, 255, 0), 'phrase': (0, 255, 255)}
        cv2.putText(display, f"MODE: {mode.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_colors.get(mode, (255, 255, 255)), 2)
        
        # Show hand status
        if has_hand:
            cv2.circle(display, (display.shape[1] - 30, 40), 10, (0, 255, 0), -1)
        else:
            cv2.circle(display, (display.shape[1] - 30, 40), 10, (0, 0, 255), -1)
        
        # Instructions
        cv2.putText(display, "l:Letter | p:Phrase | a:Auto | q:Quit",
                   (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame count (for debugging)
        cv2.putText(display, f"Frame: {frame_count}", (display.shape[1] - 100, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow('Unified Sign Language Recognition', display)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n👋 Quitting...")
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

except KeyboardInterrupt:
    print("\n\n👋 Interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n✅ Cleanup complete")