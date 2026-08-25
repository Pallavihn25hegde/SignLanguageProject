import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import mediapipe as mp

class PhraseFrameLoader:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,  # True for processing images
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks_from_image(self, image_path):
        """Extract hand landmarks from a single image"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(image_rgb)
        
        # Extract landmarks
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        # Pad or truncate to 126 features (2 hands * 21 landmarks * 3)
        while len(landmarks) < 126:
            landmarks.append(0.0)
        landmarks = landmarks[:126]
        
        return np.array(landmarks)
    
    def process_phrase_frames(self, frames_folder):
        """Process all frames in a phrase folder"""
        frames_folder = Path(frames_folder)
        phrase_name = frames_folder.stem.replace('_frames', '')  # Remove '_frames' suffix
        
        print(f"\nProcessing: {phrase_name}")
        
        # Get all JPG images
        image_files = sorted(frames_folder.glob('*.jpg')) + sorted(frames_folder.glob('*.JPG'))
        
        if not image_files:
            print(f"  ⚠️ No JPG files found in {frames_folder}")
            return None, None
        
        print(f"  Found {len(image_files)} images")
        
        # Extract landmarks from each image
        sequence_data = []
        valid_frames = 0
        
        for img_path in image_files:
            landmarks = self.extract_landmarks_from_image(img_path)
            if landmarks is not None:
                sequence_data.append(landmarks)
                valid_frames += 1
        
        print(f"  Successfully extracted {valid_frames}/{len(image_files)} frames")
        
        if len(sequence_data) < 10:  # Need at least 10 frames for a sequence
            print(f"  ⚠️ Too few valid frames ({len(sequence_data)}), skipping")
            return None, None
        
        return np.array(sequence_data), phrase_name
    
    def process_all_phrases(self, dataset_path, sequence_length=30):
        """Process all phrase folders in the dataset"""
        dataset_path = Path(dataset_path)
        
        # Find all phrase folders (ending with _frames)
        phrase_folders = [f for f in dataset_path.iterdir() 
                         if f.is_dir() and f.name.endswith('_frames')]
        
        if not phrase_folders:
            print(f"No phrase folders found in {dataset_path}")
            print("Looking for folders ending with '_frames'")
            return None, None
        
        print(f"\nFound {len(phrase_folders)} phrase folders:")
        for folder in phrase_folders:
            print(f"  - {folder.name}")
        
        all_sequences = []
        all_labels = []
        
        for folder in phrase_folders:
            sequence, phrase_name = self.process_phrase_frames(folder)
            
            if sequence is not None:
                # Split long sequences into smaller chunks if needed
                num_frames = len(sequence)
                
                if num_frames >= sequence_length:
                    # Take multiple overlapping sequences
                    num_sequences = num_frames // sequence_length
                    
                    for i in range(num_sequences):
                        start_idx = i * sequence_length
                        end_idx = start_idx + sequence_length
                        seq_chunk = sequence[start_idx:end_idx]
                        all_sequences.append(seq_chunk)
                        all_labels.append(phrase_name)
                    
                    print(f"  Created {num_sequences} sequences of {sequence_length} frames")
                else:
                    # Pad shorter sequence
                    padded_sequence = np.zeros((sequence_length, 126))
                    padded_sequence[:num_frames] = sequence
                    all_sequences.append(padded_sequence)
                    all_labels.append(phrase_name)
                    print(f"  Padded to {sequence_length} frames")
        
        print(f"\n✅ Total: {len(all_sequences)} sequences from {len(np.unique(all_labels))} phrases")
        
        return np.array(all_sequences), np.array(all_labels)
    
    def save_to_dataset(self, sequences, labels):
        """Save processed sequences to your dataset structure"""
        output_dir = 'dataset/phrases'
        os.makedirs(output_dir, exist_ok=True)
        
        unique_phrases = np.unique(labels)
        
        for phrase in unique_phrases:
            phrase_dir = os.path.join(output_dir, phrase)
            os.makedirs(phrase_dir, exist_ok=True)
            
            # Get sequences for this phrase
            phrase_indices = np.where(labels == phrase)[0]
            phrase_sequences = sequences[phrase_indices]
            
            # Save each sequence
            for i, seq in enumerate(phrase_sequences):
                seq_path = os.path.join(phrase_dir, f'seq_{i:03d}.npy')
                np.save(seq_path, seq)
            
            print(f"  Saved {len(phrase_sequences)} sequences for '{phrase}'")
        
        print(f"\n✅ All phrase data saved to {output_dir}/")

# Main execution
if __name__ == "__main__":
    # Set the path to your downloaded dataset
    DATASET_PATH = "."  # Current directory (where your phrase folders are)
    # Or specify full path:
    # DATASET_PATH = "E:/SignLanguageProject"
    
    print("=" * 50)
    print("Loading Phrase Dataset from Frames")
    print("=" * 50)
    
    loader = PhraseFrameLoader()
    
    # Process all phrases
    sequences, labels = loader.process_all_phrases("E:\SignLanguageProject\dataset\phrases", sequence_length=30)
    
    if sequences is not None and len(sequences) > 0:
        # Save to your dataset structure
        loader.save_to_dataset(sequences, labels)
        
        # Also save metadata
        metadata = {
            'num_sequences': len(sequences),
            'num_phrases': len(np.unique(labels)),
            'phrases': list(np.unique(labels)),
            'sequence_length': sequences.shape[1] if len(sequences) > 0 else 0,
            'feature_size': sequences.shape[2] if len(sequences) > 0 else 0
        }
        
        with open('dataset/phrases/metadata.pickle', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("\n📊 Dataset Summary:")
        print(f"  Total sequences: {metadata['num_sequences']}")
        print(f"  Total phrases: {metadata['num_phrases']}")
        print(f"  Phrases: {', '.join(metadata['phrases'])}")
        print(f"  Sequence length: {metadata['sequence_length']} frames")
        print(f"  Features per frame: {metadata['feature_size']}")
        
    else:
        print("\n❌ No data was processed. Please check:")
        print("1. Are the phrase folders in the current directory?")
        print("2. Do the folders contain JPG images?")
        print("3. Do the folder names end with '_frames'?")