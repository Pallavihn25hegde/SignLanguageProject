# complete_solution.py - Copy this entire code
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import mediapipe as mp

print("=" * 70)
print("🎯 SIGN LANGUAGE PROJECT - COMPLETE PHRASE DATASET SOLUTION")
print("=" * 70)

# Step 1: Check and create folders
print("\n📁 STEP 1: Setting up folders...")
os.makedirs('frames', exist_ok=True)
os.makedirs('dataset/phrases', exist_ok=True)
print("✅ Folders ready")

# Step 2: Find phrase folders
print("\n🔍 STEP 2: Looking for phrase folders...")

phrase_folders = []
search_locations = [
    Path('frames'),
    Path('.'),
    Path('E:/SignLanguageProject'),
]

for location in search_locations:
    if location.exists():
        for item in location.iterdir():
            if item.is_dir() and ('_frames' in item.name or 'frames' in item.name.lower()):
                if item.name not in ['frames', 'dataset', 'asl_env']:
                    phrase_folders.append(item)
                    print(f"  ✅ Found: {item.name}")

if not phrase_folders:
    print("\n❌ No phrase folders found!")
    print("\nPlease move your folders (Hello_frames, Thanks_frames, etc.)")
    print("into the 'frames' folder and run this script again.")
    print("\nCurrent folders in 'frames':")
    if Path('frames').exists():
        for f in Path('frames').iterdir():
            print(f"  - {f.name}")
    input("\nPress Enter to exit...")
    exit()

print(f"\n✅ Found {len(phrase_folders)} phrase folders")

# Step 3: Initialize MediaPipe
print("\n🖐️ STEP 3: Initializing hand detection...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)
print("✅ MediaPipe ready")

# Step 4: Process each phrase folder
print("\n🔄 STEP 4: Processing images and extracting landmarks...")

all_sequences = []
all_labels = []
SEQUENCE_LENGTH = 30

for folder in phrase_folders:
    # Get phrase name
    phrase_name = folder.name.replace('_frames', '').replace('_frames', '')
    print(f"\n  📹 Processing: {phrase_name}")
    
    # Get all images
    images = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        images.extend(folder.glob(ext))
    
    if not images:
        print(f"     ❌ No images found")
        continue
    
    print(f"     Found {len(images)} images")
    
    # Extract landmarks from each image
    landmarks_list = []
    for i, img_path in enumerate(images[:200]):  # Limit to 200 images
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Extract landmarks
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
            print(f"     Processed {i+1}/{min(len(images), 200)} images...")
    
    print(f"     Extracted {len(landmarks_list)} valid frames")
    
    if len(landmarks_list) < SEQUENCE_LENGTH:
        print(f"     ⚠️ Not enough frames, skipping")
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

# Step 5: Save processed data
print("\n💾 STEP 5: Saving to dataset...")

if all_sequences:
    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)
    
    # Save each phrase
    unique_phrases = np.unique(all_labels)
    
    for phrase in unique_phrases:
        phrase_dir = Path(f'dataset/phrases/{phrase}')
        phrase_dir.mkdir(parents=True, exist_ok=True)
        
        # Get sequences for this phrase
        indices = np.where(all_labels == phrase)[0]
        phrase_sequences = all_sequences[indices]
        
        # Save sequences
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
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total sequences: {metadata['num_sequences']}")
    print(f"  Total phrases: {metadata['num_phrases']}")
    print(f"  Phrases: {', '.join(metadata['phrases'])}")
    
else:
    print("\n❌ No sequences were created!")
    print("\nPossible issues:")
    print("  1. Images don't contain visible hands")
    print("  2. Image quality is too low")
    print("  3. Need more images per phrase")
    exit()

# Step 6: Combine with letter dataset
print("\n🔗 STEP 6: Combining with letter dataset...")

if os.path.exists('data.pickle'):
    with open('data.pickle', 'rb') as f:
        letter_data = pickle.load(f)
    
    print(f"  ✅ Letter dataset loaded: {len(letter_data['data'])} samples, {len(np.unique(letter_data['labels']))} classes")
    
    # Create combined data structure
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
    
    print(f"  ✅ Combined dataset saved to 'combined_data.pickle'")
else:
    print(f"  ⚠️ No letter dataset found at 'data.pickle'")
    print(f"  Continue with phrase-only dataset")

# Step 7: Create label encoders
print("\n🏷️ STEP 7: Creating label encoders...")

from sklearn.preprocessing import LabelEncoder

letter_encoder = LabelEncoder()
if os.path.exists('data.pickle'):
    with open('data.pickle', 'rb') as f:
        letter_data = pickle.load(f)
    letter_encoder.fit(letter_data['labels'])

phrase_encoder = LabelEncoder()
if all_labels is not None and len(all_labels) > 0:
    phrase_encoder.fit(all_labels)

encoders = {
    'letters': letter_encoder,
    'phrases': phrase_encoder
}

with open('label_encoders.pickle', 'wb') as f:
    pickle.dump(encoders, f)

print("  ✅ Label encoders saved")

# Final summary
print("\n" + "=" * 70)
print("✅✅✅ COMPLETE SUCCESS! ✅✅✅")
print("=" * 70)
print("\n📊 FINAL DATASET SUMMARY:")
print(f"  • Letters: {len(letter_data['data']) if os.path.exists('data.pickle') else 0} samples")
print(f"  • Phrases: {len(all_sequences)} sequences")
print(f"  • Total phrases: {len(unique_phrases)}")
print(f"\n📁 Data saved to:")
print(f"  • dataset/phrases/ - Phrase sequences")
print(f"  • combined_data.pickle - Combined dataset")
print(f"  • label_encoders.pickle - Label encoders")

print("\n🎯 NEXT STEPS:")
print("  1. Install TensorFlow: pip install tensorflow")
print("  2. Train phrase model: python train_phrase_model.py")
print("  3. Run recognition: python realtime_phrase_recognition.py")

# Cleanup
hands.close()
print("\n✨ Setup complete! Run 'python train_phrase_model.py' next")