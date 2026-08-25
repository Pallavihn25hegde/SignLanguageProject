# final_solution.py
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import mediapipe as mp

print("=" * 70)
print("🎯 SIGN LANGUAGE PROJECT - PHRASE DATASET LOADER")
print("=" * 70)

# Step 1: Find your phrase folders
print("\n📁 STEP 1: Locating phrase folders...")

phrase_folders = []

# Check common locations
locations_to_check = [
    Path('frames'),
    Path('.'),
    Path('E:/SignLanguageProject'),
]

for location in locations_to_check:
    if location.exists():
        print(f"  Checking: {location}")
        for item in location.iterdir():
            if item.is_dir() and ('_frames' in item.name):
                phrase_folders.append(item)
                print(f"    ✅ Found: {item.name}")

if not phrase_folders:
    print("\n❌ No '_frames' folders found!")
    print("\nPlease run these commands:")
    print("  1. mkdir frames")
    print("  2. move *_frames frames\\")
    print("\nThen run this script again.")
    input("\nPress Enter to exit...")
    exit()

print(f"\n✅ Found {len(phrase_folders)} phrase folders")

# Step 2: Initialize MediaPipe
print("\n🖐️ STEP 2: Initializing MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)
print("✅ MediaPipe ready")

# Step 3: Process each folder
print("\n🔄 STEP 3: Processing images...")

all_sequences = []
all_labels = []
SEQUENCE_LENGTH = 30

for folder in phrase_folders:
    # Get phrase name
    phrase_name = folder.name.replace('_frames', '')
    print(f"\n  📹 Processing: {phrase_name}")
    
    # Get images
    images = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        images.extend(folder.glob(ext))
    
    if not images:
        print(f"     ❌ No images found")
        continue
    
    print(f"     Found {len(images)} images")
    
    # Extract landmarks
    landmarks_list = []
    for i, img_path in enumerate(images[:150]):  # Process first 150 images
        # Read and process image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        # Extract landmarks
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
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"     Processed {i+1}/{min(len(images), 150)} images...")
    
    print(f"     Extracted {len(landmarks_list)} valid frames")
    
    if len(landmarks_list) < SEQUENCE_LENGTH:
        print(f"     ⚠️ Not enough frames, need {SEQUENCE_LENGTH}, have {len(landmarks_list)}")
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
    
    print(f"     ✅ Created {sequences_created} sequences")

# Step 4: Save the data
print("\n💾 STEP 4: Saving phrase dataset...")

if not all_sequences:
    print("\n❌ No sequences were created!")
    print("\nPossible issues:")
    print("  1. Images don't contain visible hands")
    print("  2. Need more images per phrase (at least 30)")
    print("  3. Image quality is too low")
    hands.close()
    exit()

# Convert to numpy arrays
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)

# Create output directory
os.makedirs('dataset/phrases', exist_ok=True)

# Save sequences for each phrase
unique_phrases = np.unique(all_labels)
for phrase in unique_phrases:
    phrase_dir = Path(f'dataset/phrases/{phrase}')
    phrase_dir.mkdir(parents=True, exist_ok=True)
    
    # Get indices for this phrase
    indices = np.where(all_labels == phrase)[0]
    phrase_sequences = all_sequences[indices]
    
    # Save each sequence
    for i, seq in enumerate(phrase_sequences):
        save_path = phrase_dir / f'seq_{i:03d}.npy'
        np.save(save_path, seq)
    
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

# Step 5: Combine with letter data
print("\n🔗 STEP 5: Combining with letter dataset...")

if os.path.exists('data.pickle'):
    with open('data.pickle', 'rb') as f:
        letter_data = pickle.load(f)
    
    print(f"  ✅ Letter dataset: {len(letter_data['data'])} samples")
    
    # Create combined dataset
    combined_data = {
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
        pickle.dump(combined_data, f)
    
    print("  ✅ Combined dataset saved")
else:
    print("  ⚠️ No letter dataset found (data.pickle missing)")
    print("  Continue with phrase-only dataset")

# Step 6: Create label encoders
print("\n🏷️ STEP 6: Creating label encoders...")

from sklearn.preprocessing import LabelEncoder

# Phrase encoder
phrase_encoder = LabelEncoder()
phrase_encoder.fit(all_labels)

# Letter encoder (if exists)
letter_encoder = LabelEncoder()
if os.path.exists('data.pickle'):
    with open('data.pickle', 'rb') as f:
        letter_data = pickle.load(f)
    letter_encoder.fit(letter_data['labels'])

encoders = {
    'letters': letter_encoder,
    'phrases': phrase_encoder
}

with open('label_encoders.pickle', 'wb') as f:
    pickle.dump(encoders, f)

print("  ✅ Label encoders saved")

# Final summary
print("\n" + "=" * 70)
print("✅✅✅ SUCCESS! PHRASE DATASET LOADED ✅✅✅")
print("=" * 70)
print("\n📊 DATASET SUMMARY:")
print(f"  • Total sequences: {len(all_sequences)}")
print(f"  • Total phrases: {len(unique_phrases)}")
print(f"  • Phrases: {', '.join(unique_phrases)}")
print(f"  • Sequence length: {SEQUENCE_LENGTH} frames")
print(f"  • Features per frame: 126")

print("\n📁 SAVED FILES:")
print(f"  • dataset/phrases/ - Phrase sequences")
print(f"  • combined_data.pickle - Combined dataset")
print(f"  • label_encoders.pickle - Label encoders")

print("\n🎯 NEXT STEPS:")
print("  1. Fix TensorFlow: pip install protobuf")
print("  2. Then: pip install tensorflow")
print("  3. Then: python train_phrase_model.py")

# Cleanup
hands.close()
print("\n✨ Ready for training!")