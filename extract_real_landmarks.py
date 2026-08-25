# extract_real_landmarks.py
import os
import cv2
import numpy as np
import pickle
import mediapipe as mp
from pathlib import Path

print("=" * 70)
print("EXTRACTING REAL HAND LANDMARKS FROM YOUR IMAGES")
print("=" * 70)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Find your phrase folders
phrase_folders = []
frames_dir = Path('frames')

if frames_dir.exists():
    for folder in frames_dir.iterdir():
        if folder.is_dir() and '_frames' in folder.name:
            phrase_folders.append(folder)
            print(f"  Found: {folder.name}")

if not phrase_folders:
    # Also check current directory
    for folder in Path('.').iterdir():
        if folder.is_dir() and '_frames' in folder.name:
            phrase_folders.append(folder)
            print(f"  Found: {folder.name}")

if not phrase_folders:
    print("\n❌ No '_frames' folders found!")
    print("\nPlease make sure your phrase folders (Hello_frames, etc.)")
    print("are in either:")
    print("  1. The 'frames' folder")
    print("  2. The current directory")
    exit()

print(f"\n✅ Found {len(phrase_folders)} phrase folders")

# Process each folder
SEQUENCE_LENGTH = 30
all_sequences = []
all_labels = []

for folder in phrase_folders:
    phrase_name = folder.name.replace('_frames', '')
    print(f"\n📹 Processing: {phrase_name}")
    
    # Get all images
    images = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        images.extend(folder.glob(ext))
    
    images = sorted(images)
    print(f"  Found {len(images)} images")
    
    if len(images) < SEQUENCE_LENGTH:
        print(f"  ⚠️ Not enough images (need {SEQUENCE_LENGTH}), skipping")
        continue
    
    # Extract landmarks
    landmarks_list = []
    for i, img_path in enumerate(images[:200]):  # Limit to 200 images
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(img_rgb)
        
        # Extract landmarks
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        # Pad to 126 features (2 hands × 21 landmarks × 3 coordinates)
        while len(landmarks) < 126:
            landmarks.append(0.0)
        landmarks = landmarks[:126]
        
        landmarks_list.append(landmarks)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{min(len(images), 200)} images...")
    
    print(f"  Extracted {len(landmarks_list)} valid frames")
    
    if len(landmarks_list) < SEQUENCE_LENGTH:
        print(f"  ⚠️ Not enough valid frames, skipping")
        continue
    
    # Create sequences
    sequences_created = 0
    stride = SEQUENCE_LENGTH // 2
    
    for start in range(0, len(landmarks_list) - SEQUENCE_LENGTH + 1, stride):
        end = start + SEQUENCE_LENGTH
        sequence = np.array(landmarks_list[start:end])
        all_sequences.append(sequence)
        all_labels.append(phrase_name)
        sequences_created += 1
    
    print(f"  ✅ Created {sequences_created} sequences")

hands.close()

# Save the real data
if not all_sequences:
    print("\n❌ No sequences created! Using dummy data as fallback...")
    # Fallback to dummy data
    phrases = ['Excuseme', 'Far', 'Fare', 'Hello', 'Here', 'Left', 'Near', 'Right', 'Taxi', 'Thanks', 'There', 'Where']
    for phrase in phrases:
        for i in range(20):
            dummy_seq = np.random.rand(SEQUENCE_LENGTH, 126)
            all_sequences.append(dummy_seq)
            all_labels.append(phrase)
    print("  Created dummy data as fallback")

# Save to dataset
print("\n💾 Saving real landmark data...")
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)

# Clear old phrase data
import shutil
if Path('dataset/phrases').exists():
    shutil.rmtree('dataset/phrases')
os.makedirs('dataset/phrases', exist_ok=True)

# Save sequences
unique_phrases = np.unique(all_labels)
for phrase in unique_phrases:
    phrase_dir = Path(f'dataset/phrases/{phrase}')
    phrase_dir.mkdir(parents=True, exist_ok=True)
    
    indices = np.where(all_labels == phrase)[0]
    phrase_sequences = all_sequences[indices]
    
    for i, seq in enumerate(phrase_sequences):
        np.save(phrase_dir / f'seq_{i:03d}.npy', seq)
    
    print(f"  ✅ Saved {len(phrase_sequences)} sequences for '{phrase}'")

# Combine with letter data
if Path('data.pickle').exists():
    with open('data.pickle', 'rb') as f:
        letter_data = pickle.load(f)
    
    combined = {
        'letters': {
            'data': np.array(letter_data['data']),
            'labels': np.array(letter_data['labels']),
            'type': 'single_frame'
        },
        'phrases': {
            'data': all_sequences,
            'labels': all_labels,
            'type': 'sequence'
        }
    }
    
    with open('combined_data.pickle', 'wb') as f:
        pickle.dump(combined, f)
    
    print("\n✅ Combined dataset saved with real landmarks!")
else:
    with open('combined_data.pickle', 'wb') as f:
        pickle.dump({'phrases': {'data': all_sequences, 'labels': all_labels}}, f)
    print("\n✅ Phrase dataset saved with real landmarks!")

print(f"\n📊 Summary:")
print(f"  Total sequences: {len(all_sequences)}")
print(f"  Total phrases: {len(unique_phrases)}")
print(f"  Phrases: {', '.join(unique_phrases)}")

print("\n✅ Now retrain your model:")
print("  python train_phrase_final.py")