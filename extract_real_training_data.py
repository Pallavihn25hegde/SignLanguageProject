# extract_real_training_data.py
import cv2
import numpy as np
import pickle
import mediapipe as mp
from pathlib import Path
import os

print("=" * 60)
print("EXTRACTING REAL TRAINING DATA")
print("=" * 60)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Process letter images
print("\n📝 Processing LETTER images...")
letters_dir = Path('dataset/letters')

if letters_dir.exists():
    all_letter_landmarks = []
    all_letter_labels = []
    
    for letter_folder in letters_dir.iterdir():
        if letter_folder.is_dir():
            letter = letter_folder.name
            print(f"  Processing letter: {letter}")
            
            images = []
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
                images.extend(letter_folder.glob(ext))
            
            count = 0
            for img_path in images[:50]:  # Process up to 50 images per letter
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                
                if landmarks:
                    while len(landmarks) < 126:
                        landmarks.append(0.0)
                    landmarks = landmarks[:126]
                    all_letter_landmarks.append(landmarks)
                    all_letter_labels.append(letter)
                    count += 1
            
            print(f"    Extracted {count} valid samples")
    
    if all_letter_landmarks:
        letter_data = {
            'data': np.array(all_letter_landmarks),
            'labels': np.array(all_letter_labels)
        }
        with open('real_letter_data.pickle', 'wb') as f:
            pickle.dump(letter_data, f)
        print(f"\n✅ Saved {len(all_letter_landmarks)} letter samples")
else:
    print("  Letter images directory not found")

# Process phrase frames
print("\n🎬 Processing PHRASE frames...")
frames_dir = Path('frames')

if frames_dir.exists():
    all_phrase_sequences = []
    all_phrase_labels = []
    SEQUENCE_LENGTH = 30
    
    for phrase_folder in frames_dir.iterdir():
        if phrase_folder.is_dir() and '_frames' in phrase_folder.name:
            phrase_name = phrase_folder.name.replace('_frames', '')
            print(f"  Processing phrase: {phrase_name}")
            
            # Get all images
            images = []
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
                images.extend(phrase_folder.glob(ext))
            
            images = sorted(images)
            print(f"    Found {len(images)} images")
            
            # Extract landmarks from each image
            landmarks_sequence = []
            for img_path in images[:150]:  # Limit to 150 images
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                
                if landmarks:
                    while len(landmarks) < 126:
                        landmarks.append(0.0)
                    landmarks = landmarks[:126]
                    landmarks_sequence.append(landmarks)
            
            print(f"    Extracted {len(landmarks_sequence)} valid frames")
            
            # Create sequences
            if len(landmarks_sequence) >= SEQUENCE_LENGTH:
                num_sequences = len(landmarks_sequence) // SEQUENCE_LENGTH
                for i in range(num_sequences):
                    start = i * SEQUENCE_LENGTH
                    end = start + SEQUENCE_LENGTH
                    sequence = np.array(landmarks_sequence[start:end])
                    all_phrase_sequences.append(sequence)
                    all_phrase_labels.append(phrase_name)
                
                print(f"    Created {num_sequences} sequences")
    
    if all_phrase_sequences:
        # Save phrase data
        os.makedirs('dataset/phrases_real', exist_ok=True)
        for seq, label in zip(all_phrase_sequences, all_phrase_labels):
            phrase_dir = Path(f'dataset/phrases_real/{label}')
            phrase_dir.mkdir(parents=True, exist_ok=True)
            seq_num = len(list(phrase_dir.glob('*.npy')))
            np.save(phrase_dir / f'seq_{seq_num:03d}.npy', seq)
        
        print(f"\n✅ Saved {len(all_phrase_sequences)} phrase sequences")
else:
    print("  Frames directory not found")

hands.close()

print("\n" + "=" * 60)
print("Now retrain your model with real data:")
print("  python retrain_with_real_data.py")