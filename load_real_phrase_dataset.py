# load_real_phrase_dataset.py
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import mediapipe as mp

class RealPhraseDatasetLoader:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # Use static_image_mode for processing images
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks(self, image_path):
        """Extract hand landmarks from image"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process
            results = self.hands.process(img_rgb)
            
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
            
            return np.array(landmarks)
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def process_phrase_folder(self, folder_path, sequence_length=30):
        """Process a single phrase folder"""
        folder_path = Path(folder_path)
        phrase_name = folder_path.stem.replace('_frames', '')
        
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
        
        # Extract landmarks from each image
        all_landmarks = []
        valid_count = 0
        
        for i, img_path in enumerate(images):
            landmarks = self.extract_landmarks(img_path)
            if landmarks is not None:
                all_landmarks.append(landmarks)
                valid_count += 1
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(images)} images...")
        
        print(f"  Successfully extracted {valid_count}/{len(images)} frames")
        
        if len(all_landmarks) < sequence_length:
            print(f"  ⚠️ Only {len(all_landmarks)} valid frames (< {sequence_length})")
            return None, None
        
        # Create sequences using sliding window
        sequences = []
        stride = sequence_length // 2  # 50% overlap
        
        for start in range(0, len(all_landmarks) - sequence_length + 1, stride):
            end = start + sequence_length
            sequence = np.array(all_landmarks[start:end])
            sequences.append(sequence)
        
        print(f"  Created {len(sequences)} sequences")
        
        return np.array(sequences), phrase_name
    
    def load_all_phrases(self, phrases_base_path, sequence_length=30):
        """Load all phrase folders from a base path"""
        base_path = Path(phrases_base_path)
        
        # Find all _frames folders
        phrase_folders = [f for f in base_path.iterdir() 
                         if f.is_dir() and '_frames' in f.name]
        
        if not phrase_folders:
            print(f"No '_frames' folders found in {phrases_base_path}")
            return None, None
        
        print(f"\nFound {len(phrase_folders)} phrase folders")
        
        all_sequences = []
        all_labels = []
        
        for folder in phrase_folders:
            sequences, phrase_name = self.process_phrase_folder(folder, sequence_length)
            if sequences is not None:
                all_sequences.extend(sequences)
                all_labels.extend([phrase_name] * len(sequences))
        
        if all_sequences:
            return np.array(all_sequences), np.array(all_labels)
        return None, None
    
    def save_dataset(self, sequences, labels):
        """Save to your dataset structure"""
        output_dir = 'dataset/phrases'
        os.makedirs(output_dir, exist_ok=True)
        
        # Remove old data
        for old_file in Path(output_dir).glob('*'):
            if old_file.is_dir():
                import shutil
                shutil.rmtree(old_file)
        
        # Save new data
        unique_phrases = np.unique(labels)
        
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
    print("REAL PHRASE DATASET LOADER")
    print("=" * 60)
    
    # OPTION 1: Enter the path where your _frames folders are
    print("\nWhere are your phrase folders (Hello_frames, Thanks_frames, etc.)?")
    print("Examples:")
    print("  - C:/Users/YourName/Downloads/PhraseDataset")
    print("  - D:/Datasets/SignLanguagePhrases")
    print("  - . (current directory)")
    
    dataset_path = input("\nEnter path: ").strip()
    
    if not dataset_path:
        dataset_path = "."  # Default to current directory
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ Path not found: {dataset_path}")
        print("\nPlease run 'python find_phrase_folders.py' first to locate your dataset")
        exit(1)
    
    # Load the dataset
    loader = RealPhraseDatasetLoader()
    sequences, labels = loader.load_all_phrases(dataset_path, sequence_length=30)
    
    if sequences is not None and len(sequences) > 0:
        # Save to dataset
        metadata = loader.save_dataset(sequences, labels)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Dataset Loaded:")
        print("=" * 60)
        print(f"  Total sequences: {metadata['num_sequences']}")
        print(f"  Total phrases: {metadata['num_phrases']}")
        print(f"  Phrases: {', '.join(metadata['phrases'])}")
        print(f"  Sequence length: {metadata['sequence_length']} frames")
        
        print("\n📁 Data saved to: dataset/phrases/")
        print("\nNext step: Run 'python prepare_combined_dataset.py'")
        
    else:
        print("\n❌ No valid sequences were created!")
        print("\nTroubleshooting:")
        print("1. Make sure the path contains folders ending with '_frames'")
        print("2. Check that those folders contain JPG/PNG images")
        print("3. Try running 'python find_phrase_folders.py' to locate your dataset")