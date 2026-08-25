# load_phrase_dataset_corrected.py
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import mediapipe as mp

class CorrectPhraseLoader:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks(self, image_path):
        """Extract hand landmarks from image"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
            
            # Pad to 126 features
            while len(landmarks) < 126:
                landmarks.append(0.0)
            landmarks = landmarks[:126]
            
            return np.array(landmarks)
        except Exception as e:
            return None
    
    def find_phrase_folders(self, base_path):
        """Find all _frames folders recursively"""
        base_path = Path(base_path)
        phrase_folders = []
        
        # Look for folders ending with _frames
        for folder in base_path.rglob("*_frames"):
            if folder.is_dir():
                phrase_folders.append(folder)
        
        # Also look for folders containing _frames in name
        for folder in base_path.rglob("*_frames*"):
            if folder.is_dir() and folder not in phrase_folders:
                phrase_folders.append(folder)
        
        return phrase_folders
    
    def process_phrase_folder(self, folder_path, sequence_length=30):
        """Process a single phrase folder"""
        folder_path = Path(folder_path)
        # Extract phrase name (remove _frames suffix)
        phrase_name = folder_path.stem
        if phrase_name.endswith('_frames'):
            phrase_name = phrase_name.replace('_frames', '')
        
        print(f"\n📁 Processing: {phrase_name}")
        
        # Get all images
        images = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            images.extend(folder_path.glob(ext))
        
        images = sorted(images)
        
        if not images:
            print(f"  ❌ No images found in {folder_path}")
            return None, None
        
        print(f"  Found {len(images)} images")
        
        # Extract landmarks (limit to first 200 images for speed)
        if len(images) > 200:
            print(f"  Limiting to first 200 images for processing")
            images = images[:200]
        
        all_landmarks = []
        valid_count = 0
        
        for i, img_path in enumerate(images):
            landmarks = self.extract_landmarks(img_path)
            if landmarks is not None:
                all_landmarks.append(landmarks)
                valid_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(images)} images...")
        
        print(f"  Successfully extracted {valid_count}/{len(images)} frames")
        
        if len(all_landmarks) < sequence_length:
            print(f"  ⚠️ Only {len(all_landmarks)} valid frames (< {sequence_length})")
            return None, None
        
        # Create sequences
        sequences = []
        stride = sequence_length // 2
        
        for start in range(0, len(all_landmarks) - sequence_length + 1, stride):
            end = start + sequence_length
            sequence = np.array(all_landmarks[start:end])
            sequences.append(sequence)
        
        print(f"  Created {len(sequences)} sequences")
        
        return np.array(sequences), phrase_name
    
    def load_all_phrases(self, base_path):
        """Load all phrase folders from base path"""
        base_path = Path(base_path)
        
        if not base_path.exists():
            print(f"❌ Path does not exist: {base_path}")
            return None, None
        
        # Find all phrase folders
        phrase_folders = self.find_phrase_folders(base_path)
        
        if not phrase_folders:
            print(f"No '_frames' folders found in {base_path}")
            
            # Check current directory
            current_folders = [f for f in base_path.iterdir() if f.is_dir()]
            print(f"\nFolders in current directory:")
            for folder in current_folders:
                print(f"  - {folder.name}")
            
            return None, None
        
        print(f"\n✅ Found {len(phrase_folders)} phrase folders:")
        for folder in phrase_folders:
            print(f"  - {folder.name} at {folder.parent}")
        
        all_sequences = []
        all_labels = []
        
        for folder in phrase_folders:
            sequences, phrase_name = self.process_phrase_folder(folder)
            if sequences is not None:
                all_sequences.extend(sequences)
                all_labels.extend([phrase_name] * len(sequences))
        
        if all_sequences:
            return np.array(all_sequences), np.array(all_labels)
        return None, None
    
    def save_dataset(self, sequences, labels):
        """Save to dataset structure"""
        output_dir = 'dataset/phrases'
        os.makedirs(output_dir, exist_ok=True)
        
        unique_phrases = np.unique(labels)
        
        print(f"\n💾 Saving to {output_dir}/")
        for phrase in unique_phrases:
            phrase_dir = os.path.join(output_dir, phrase)
            os.makedirs(phrase_dir, exist_ok=True)
            
            phrase_indices = np.where(labels == phrase)[0]
            phrase_sequences = sequences[phrase_indices]
            
            for i, seq in enumerate(phrase_sequences):
                seq_path = os.path.join(phrase_dir, f'seq_{i:03d}.npy')
                np.save(seq_path, seq)
            
            print(f"  ✓ Saved {len(phrase_sequences)} sequences for '{phrase}'")
        
        # Save metadata
        metadata = {
            'num_sequences': len(sequences),
            'num_phrases': len(unique_phrases),
            'phrases': list(unique_phrases),
            'sequence_length': sequences.shape[1] if len(sequences) > 0 else 0,
            'feature_size': sequences.shape[2] if len(sequences) > 0 else 0
        }
        
        with open('dataset/phrases/metadata.pickle', 'wb') as f:
            pickle.dump(metadata, f)
        
        return metadata

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("PHRASE DATASET LOADER - CORRECTED")
    print("=" * 60)
    
    # Get the current directory
    current_dir = os.getcwd()
    print(f"\nCurrent directory: {current_dir}")
    
    # Look for phrase folders
    loader = CorrectPhraseLoader()
    
    # Try multiple possible locations
    possible_paths = [
        current_dir,
        os.path.join(current_dir, '_frames'),
        os.path.join(current_dir, 'dataset'),
        os.path.join(current_dir, 'dataset', 'phrases'),
        os.path.join(current_dir, 'data'),
    ]
    
    sequences = None
    labels = None
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\n🔍 Checking: {path}")
            sequences, labels = loader.load_all_phrases(path)
            if sequences is not None and len(sequences) > 0:
                print(f"\n✅ Successfully loaded from: {path}")
                break
    
    if sequences is not None and len(sequences) > 0:
        # Save to dataset
        metadata = loader.save_dataset(sequences, labels)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Dataset Summary:")
        print("=" * 60)
        print(f"  Total sequences: {metadata['num_sequences']}")
        print(f"  Total phrases: {metadata['num_phrases']}")
        print(f"  Phrases: {', '.join(metadata['phrases'])}")
        print(f"  Sequence length: {metadata['sequence_length']} frames")
        
        print("\n📁 Data saved to: dataset/phrases/")
        print("\n➡️ Next step: Run 'python prepare_combined_dataset.py'")
        
    else:
        print("\n❌ Could not find any phrase folders!")
        print("\nPlease check the following:")
        print("1. Are your '_frames' folders in the current directory?")
        print("2. Try moving them to: E:\\SignLanguageProject\\_frames\\")
        print("3. Or run: mkdir _frames and move your phrase folders there")
        print("\nCurrent directory contents:")
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}")
                