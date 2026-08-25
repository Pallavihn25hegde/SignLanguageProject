# load_phrase_data.py
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import mediapipe as mp

print("=" * 60)
print("LOADING PHRASE DATA")
print("=" * 60)

# Create necessary folders
os.makedirs('frames', exist_ok=True)
os.makedirs('dataset/phrases', exist_ok=True)

# Find all phrase folders
phrase_folders = []
frames_dir = Path('frames')

if not frames_dir.exists():
    print("❌ 'frames' folder not found!")
    exit()

print(f"\nChecking '{frames_dir}' for phrase folders...")

for folder in frames_dir.iterdir():
    if folder.is_dir():
        # Check if it contains images
        images = list(folder.glob('*.jpg')) + list(folder.glob('*.JPG')) + list(folder.glob('*.png'))
        if images:
            phrase_folders.append(folder)
            print(f"  ✅ Found: {folder.name} ({len(images)} images)")

if not phrase_folders:
    print("\n❌ No phrase folders with images found!")
    print("\nPlease make sure:")
    print("  1. You have folders like 'Hello_frames', 'Thanks_frames'")
    print("  2. They are inside the 'frames' folder")
    print("  3. They contain JPG or PNG images")
    print("\nCurrent contents of 'frames':")
    for item in frames_dir.iterdir():
        print(f"  - {item.name}")
    exit()

print(f"\n✅ Found {len(phrase_folders)} phrase folders")

# Initialize MediaPipe
print("\n🖐️ Initializing MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Process each folder
all_sequences = []
all_labels = []
SEQUENCE_LENGTH = 30

for folder in phrase_folders:
    phrase_name = folder.name.replace('_frames', '')
    print(f"\n📹 Processing: {phrase_name}")
    
    # Get all images
    images = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        images.extend(folder.glob(ext))
    
    images = sorted(images)
    print(f"  Found {len(images)} images")
    
    # Extract landmarks
    landmarks_list = []
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        # Pad to 126 features
        while len(landmarks) < 126:
            landmarks.append(0.0)
        landmarks = landmarks[:126]
        
        landmarks_list.append(landmarks)
        
        # Show progress
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(images)} images...")
    
    print(f"  Extracted {len(landmarks_list)} valid frames")
    
    if len(landmarks_list) < SEQUENCE_LENGTH:
        print(f"  ⚠️ Not enough frames! Need {SEQUENCE_LENGTH}, have {len(landmarks_list)}")
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

# Save the data
if not all_sequences:
    print("\n❌ No sequences created!")
    exit()

print(f"\n💾 Saving {len(all_sequences)} sequences...")

all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)

# Save individual sequences
unique_phrases = np.unique(all_labels)
for phrase in unique_phrases:
    phrase_dir = Path(f'dataset/phrases/{phrase}')
    phrase_dir.mkdir(parents=True, exist_ok=True)
    
    indices = np.where(all_labels == phrase)[0]
    phrase_sequences = all_sequences[indices]
    
    for i, seq in enumerate(phrase_sequences):
        np.save(phrase_dir / f'seq_{i:03d}.npy', seq)
    
    print(f"  ✅ Saved {len(phrase_sequences)} sequences for '{phrase}'")

# Save metadata
metadata = {
    'num_sequences': len(all_sequences),
    'num_phrases': len(unique_phrases),
    'phrases': list(unique_phrases),
    'sequence_length': SEQUENCE_LENGTH,
    'feature_size': 126
}

with open('dataset/phrases/metadata.pickle', 'wb') as f:
    pickle.dump(metadata, f)

# Combine with letters if exists
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
    
    print("  ✅ Combined dataset saved")

print("\n" + "=" * 60)
print("✅ SUCCESS!")
print("=" * 60)
print(f"\n📊 SUMMARY:")
print(f"  • Total sequences: {len(all_sequences)}")
print(f"  • Total phrases: {len(unique_phrases)}")
print(f"  • Phrases: {', '.join(unique_phrases)}")

print("\n🎯 Now run:")
print("  python train_simple.py")