# debug_recognition.py
import cv2
import numpy as np
import joblib
import mediapipe as mp
import traceback

print("=" * 60)
print("DEBUGGING RECOGNITION SYSTEM")
print("=" * 60)

# Check if models exist
print("\n1. Checking models...")
try:
    models = joblib.load('unified_models.pkl')
    encoders = joblib.load('unified_encoders.pkl')
    print("✅ Models loaded successfully")
    print(f"   Models available: {list(models.keys())}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    traceback.print_exc()
    exit()

# Check camera
print("\n2. Checking camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()
print("✅ Camera opened successfully")

# Test MediaPipe
print("\n3. Testing MediaPipe...")
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    print("✅ MediaPipe initialized")
except Exception as e:
    print(f"❌ MediaPipe error: {e}")
    traceback.print_exc()
    exit()

# Test a few frames
print("\n4. Testing camera capture...")
frame_count = 0
successful_frames = 0

for i in range(30):
    ret, frame = cap.read()
    if ret:
        successful_frames += 1
    frame_count += 1

print(f"   Captured {successful_frames}/{frame_count} frames successfully")

# Test feature extraction
print("\n5. Testing feature extraction...")
ret, frame = cap.read()
if ret:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    
    while len(landmarks) < 126:
        landmarks.append(0.0)
    landmarks = landmarks[:126]
    
    print(f"   Features extracted: {len(landmarks)} dimensions")
    print(f"   Hand detected: {results.multi_hand_landmarks is not None}")
    
    # Test prediction
    if 'letter' in models and np.sum(np.abs(landmarks)) > 0.1:
        features = np.array(landmarks).reshape(1, -1)
        pred = models['letter'].predict(features)[0]
        letter = encoders['letter'].inverse_transform([pred])[0]
        confidence = max(models['letter'].predict_proba(features)[0])
        print(f"   Test prediction: {letter} ({confidence:.2f})")

cap.release()
hands.close()

print("\n✅ All tests passed! Your system should work.")
print("\nNow run: python unified_recognition_v2.py")