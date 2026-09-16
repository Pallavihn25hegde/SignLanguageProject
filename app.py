import os
import sys
import time
from collections import deque
import cv2
import joblib
import mediapipe as mp
import numpy as np


def load_model_artifacts():
    """Locates and loads unified model and encoder files from models/ or root."""
    search_dirs = [os.path.join(os.path.dirname(__file__), "models"), os.path.dirname(__file__), "."]

    model_path = None
    encoder_path = None

    for directory in search_dirs:
        m_candidate = os.path.join(directory, "unified_models.pkl")
        e_candidate = os.path.join(directory, "unified_encoders.pkl")

        if os.path.exists(m_candidate) and not model_path:
            model_path = m_candidate
        if os.path.exists(e_candidate) and not encoder_path:
            encoder_path = e_candidate

    if not model_path or not encoder_path:
        print("❌ Error: Model or encoder files not found.")
        print("Expected 'unified_models.pkl' and 'unified_encoders.pkl' in root or 'models/' directory.")
        print("\nPlease train models first:")
        print("  python train_model.py")
        sys.exit(1)

    print(f"📂 Loading models from: {model_path}")
    models = joblib.load(model_path)
    encoders = joblib.load(encoder_path)

    print("✅ Models loaded:")
    print(f"   Letter model: {'✓' if 'letter' in models else '✗'}")
    print(f"   Phrase model: {'✓' if 'phrase' in models else '✗'}")

    return models, encoders


def extract_landmarks(results):
    """Extracts up to 2 hands (126 features) and normalizes shape."""
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    # Pad to fixed 126 features (21 landmarks * 3 coordinates * 2 hands)
    while len(landmarks) < 126:
        landmarks.append(0.0)

    return landmarks[:126]


def main():
    print("=" * 60)
    print("  REAL-TIME ASL SIGN LANGUAGE TRANSLATOR")
    print("=" * 60)

    models, encoders = load_model_artifacts()

    # Initialize MediaPipe
    print("\n🖐️ Initializing MediaPipe...")
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        mp_drawing = mp.solutions.drawing_utils
        print("✅ MediaPipe ready")
    except Exception as e:
        print(f"❌ MediaPipe initialization error: {e}")
        sys.exit(1)

    # Initialize Camera
    print("\n📷 Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam. Check connection or app permissions.")
        sys.exit(1)
    print("✅ Camera ready")

    # State variables
    mode = "auto"  # 'auto', 'letter', or 'phrase'
    phrase_buffer = deque(maxlen=30)
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
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame from webcam")
                continue

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            frame_count += 1

            # Landmark extraction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Draw hand skeleton overlay
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                    )

            landmarks = extract_landmarks(results)
            has_hand = np.sum(np.abs(landmarks)) > 0.1

            # ---------------- MODE LOGIC ----------------
            if mode == "letter" and "letter" in models:
                if has_hand:
                    features = np.array(landmarks).reshape(1, -1)
                    try:
                        pred = models["letter"].predict(features)[0]
                        letter = encoders["letter"].inverse_transform([pred])[0]
                        confidence = max(models["letter"].predict_proba(features)[0])

                        if confidence > 0.5:
                            cv2.rectangle(display, (50, 100), (350, 180), (0, 0, 0), -1)
                            cv2.rectangle(display, (50, 100), (350, 180), (0, 255, 0), 2)
                            cv2.putText(display, f"LETTER: {letter}", (60, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            cv2.putText(display, f"Confidence: {confidence:.1%}", (60, 175),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    except Exception:
                        pass

            elif mode == "phrase" and "phrase" in models:
                if has_hand:
                    phrase_buffer.append(landmarks)
                    progress = len(phrase_buffer) / 30
                    cv2.rectangle(display, (10, 30), (10 + int(progress * 300), 50), (0, 255, 0), -1)
                    cv2.putText(display, f"Recording: {len(phrase_buffer)}/30", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if len(phrase_buffer) == 30:
                        try:
                            flat_sequence = np.array(phrase_buffer).flatten().reshape(1, -1)
                            pred = models["phrase"].predict(flat_sequence)[0]
                            phrase = encoders["phrase"].inverse_transform([pred])[0]
                            confidence = max(models["phrase"].predict_proba(flat_sequence)[0])

                            if confidence > 0.5:
                                cv2.rectangle(display, (50, 100), (450, 180), (0, 0, 0), -1)
                                cv2.rectangle(display, (50, 100), (450, 180), (0, 255, 255), 2)
                                cv2.putText(display, f"PHRASE: {phrase.upper()}", (60, 145),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                                cv2.putText(display, f"Confidence: {confidence:.1%}", (60, 170),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        except Exception:
                            pass
                        phrase_buffer.clear()
                else:
                    if len(phrase_buffer) > 0:
                        phrase_buffer.clear()

            else:  # 'auto' mode
                if has_hand:
                    phrase_buffer.append(landmarks)
                    progress = len(phrase_buffer) / 30
                    cv2.rectangle(display, (10, 30), (10 + int(progress * 300), 50), (0, 255, 0), -1)

                    if len(phrase_buffer) == 30 and "phrase" in models:
                        try:
                            flat_sequence = np.array(phrase_buffer).flatten().reshape(1, -1)
                            pred = models["phrase"].predict(flat_sequence)[0]
                            phrase = encoders["phrase"].inverse_transform([pred])[0]
                            confidence = max(models["phrase"].predict_proba(flat_sequence)[0])

                            if confidence > 0.5:
                                cv2.rectangle(display, (50, 100), (450, 180), (0, 0, 0), -1)
                                cv2.rectangle(display, (50, 100), (450, 180), (0, 255, 255), 2)
                                cv2.putText(display, f"PHRASE: {phrase.upper()}", (60, 145),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                                cv2.putText(display, f"Confidence: {confidence:.1%}", (60, 170),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        except Exception:
                            pass
                        phrase_buffer.clear()

                    # Check single letters for quick gestures
                    if len(phrase_buffer) < 10 and "letter" in models:
                        try:
                            features = np.array(landmarks).reshape(1, -1)
                            pred = models["letter"].predict(features)[0]
                            letter = encoders["letter"].inverse_transform([pred])[0]
                            confidence = max(models["letter"].predict_proba(features)[0])

                            if confidence > 0.7:
                                cv2.rectangle(display, (50, 100), (350, 180), (0, 0, 0), -1)
                                cv2.rectangle(display, (50, 100), (350, 180), (0, 255, 0), 2)
                                cv2.putText(display, f"LETTER: {letter}", (60, 150),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                                cv2.putText(display, f"Confidence: {confidence:.1%}", (60, 175),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        except Exception:
                            pass
                else:
                    if len(phrase_buffer) > 0:
                        phrase_buffer.clear()

            # Mode HUD
            mode_colors = {"auto": (0, 255, 0), "letter": (255, 255, 0), "phrase": (0, 255, 255)}
            cv2.putText(display, f"MODE: {mode.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_colors.get(mode, (255, 255, 255)), 2)

            # Hand presence indicator
            cv2.circle(display, (display.shape[1] - 30, 40), 10,
                       (0, 255, 0) if has_hand else (0, 0, 255), -1)

            # Controls HUD
            cv2.putText(display, "l:Letter | p:Phrase | a:Auto | q:Quit",
                        (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Unified Sign Language Recognition", display)

            # Key handler
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n👋 Quitting...")
                break
            elif key == ord("l"):
                mode = "letter"
                phrase_buffer.clear()
                print("\n📝 Switched to LETTER mode")
            elif key == ord("p"):
                mode = "phrase"
                phrase_buffer.clear()
                print("\n🎬 Switched to PHRASE mode")
            elif key == ord("a"):
                mode = "auto"
                phrase_buffer.clear()
                print("\n🔄 Switched to AUTO mode")

    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("✅ Cleanup complete.")


if __name__ == "__main__":
    main()