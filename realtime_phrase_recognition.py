import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import time

class PhraseRecognizer:
    def __init__(self, phrase_model_path='phrase_model.h5', 
                 letter_model_path='model.p'):
        
        # Load phrase model
        self.phrase_model = tf.keras.models.load_model(phrase_model_path)
        
        # Load letter model (your existing model)
        with open(letter_model_path, 'rb') as f:
            self.letter_model = pickle.load(f)
        
        # Load label encoders
        with open('label_encoders.pickle', 'rb') as f:
            self.encoders = pickle.load(f)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Phrase recognition buffer
        self.phrase_buffer = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)
        self.last_phrase = ""
        self.last_phrase_time = 0
        self.cooldown = 2  # seconds between phrase detections
        
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        # Pad to 126 features (2 hands * 21 landmarks * 3)
        while len(landmarks) < 126:
            landmarks.append(0.0)
        landmarks = landmarks[:126]
        
        return np.array(landmarks), results
    
    def recognize_realtime(self):
        """Real-time recognition for both letters and phrases"""
        cap = cv2.VideoCapture(0)
        mode = 'letter'  # 'letter' or 'phrase'
        
        print("=== Real-time Sign Language Recognition ===")
        print("Press 'm' to switch between Letter and Phrase mode")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Extract landmarks
            landmarks, results = self.extract_landmarks(frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            if mode == 'letter':
                # Letter recognition (single frame)
                if np.any(landmarks):  # Only predict if hands detected
                    prediction = self.letter_model.predict([landmarks])[0]
                    confidence = max(self.letter_model.predict_proba([landmarks])[0])
                    
                    if confidence > 0.7:
                        cv2.putText(display_frame, f"Letter: {prediction}", 
                                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        cv2.putText(display_frame, f"Confidence: {confidence:.2%}", 
                                   (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            else:  # phrase mode
                # Add to buffer
                self.phrase_buffer.append(landmarks)
                
                # Show recording progress
                progress = len(self.phrase_buffer) / 30
                cv2.rectangle(display_frame, (10, 30), 
                             (10 + int(progress * 300), 50), (0, 255, 0), -1)
                cv2.putText(display_frame, f"Buffer: {len(self.phrase_buffer)}/30", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Make prediction when buffer is full
                if len(self.phrase_buffer) == 30 and (time.time() - self.last_phrase_time) > self.cooldown:
                    # Prepare sequence for model
                    sequence = np.array(self.phrase_buffer)
                    sequence = sequence.reshape(1, 30, 126)
                    
                    # Predict
                    predictions = self.phrase_model.predict(sequence, verbose=0)[0]
                    predicted_class = np.argmax(predictions)
                    confidence = predictions[predicted_class]
                    
                    if confidence > 0.8:
                        # Get phrase name
                        phrase_name = self.encoders['phrases'].inverse_transform([predicted_class])[0]
                        
                        # Smooth predictions
                        self.prediction_history.append(phrase_name)
                        if len(self.prediction_history) == self.prediction_history.maxlen:
                            most_common = max(set(self.prediction_history), 
                                             key=self.prediction_history.count)
                            if most_common == phrase_name:
                                self.last_phrase = phrase_name
                                self.last_phrase_time = time.time()
                        
                        # Display phrase
                        cv2.putText(display_frame, f"Phrase: {self.last_phrase}", 
                                   (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.putText(display_frame, f"Confidence: {confidence:.2%}", 
                                   (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Reset buffer after prediction
                    self.phrase_buffer.clear()
            
            # Display mode
            mode_color = (0, 255, 0) if mode == 'letter' else (255, 0, 0)
            cv2.putText(display_frame, f"Mode: {mode.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            
            # Instructions
            cv2.putText(display_frame, "Press 'm' to switch mode | 'q' to quit", 
                       (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Sign Language Recognition', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mode = 'phrase' if mode == 'letter' else 'letter'
                self.phrase_buffer.clear()
                print(f"\nSwitched to {mode} mode")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = PhraseRecognizer()
    recognizer.recognize_realtime()