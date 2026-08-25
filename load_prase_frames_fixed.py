import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import mediapipe as mp

class PhraseFrameLoader:
    def __init__(self):
        # Initialize MediaPipe with compatible parameters
        self.mp_hands = mp.solutions.hands
        # Use static_image_mode=True for images, but avoid problematic parameters
        self.hands = None
        self.initialize_hands()
    
    def initialize_hands(self):
        """Initialize hands with compatible parameters"""
        try:
            # Try with basic parameters first
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
        except Exception as e:
            print(f"Warning: {e}")
            # Fallback to minimal parameters
            self.hands = self.mp_hands.Hands(static_image_mode=True)
    
    def extract_landmarks_from_image(self, image_path):
        """Extract hand landmarks from a single image"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            if self.hands is None:
                self.initialize_hands()
            
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
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_phrase_frames(self, frames_folder):
        """Process all frames in a phrase folder"""
        frames_folder = Path(frames_folder)
        phrase_name = frames_folder.stem.replace('_frames', '')  # Remove '_frames' suffix
        
        print(f"\nProcessing: {phrase_name}")
        
        # Get all JPG images
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(frames_folder.glob(ext))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"  ⚠️ No image files found in {frames_folder}")
            return None, None
        
        print(f"  Found {len(image_files)} images")
        
        # Extract landmarks from each image
        sequence_data = []
        valid_frames = 0
        failed_frames = 0
        
        for idx, img_path in enumerate(image_files):
            landmarks = self.extract_landmarks_from_image(img_path)
            if landmarks is not None:
                sequence_data.append(landmarks)
                valid_frames += 1
            else:
                failed_frames += 1
            
            # Show progress every 50 images
            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1}/{len(image_files)} images...")
        
        print(f"  Successfully extracted {valid_frames}/{len(image_files)} frames (Failed: {failed_frames})")
        
        if len(sequence_data) < 10:  # Need at least 10 frames for a sequence
            print(f"  ⚠️ Too few valid frames ({len(sequence_data)}), skipping")
            return None, None
        
        return np.array(sequence_data), phrase_name
    
    def process_all_phrases(self, dataset_path, sequence_length=30):
        """Process all phrase folders in the dataset"""
        dataset_path = Path(dataset_path)
        
        # Find all phrase folders (ending with _frames)
        phrase_folders = [f for f in dataset_path.iterdir() 
                         if f.is_dir() and (f.name.endswith('_frames') or '_frames' in f.name)]
        
        if not phrase_folders:
            print(f"\n⚠️ No phrase folders found in {dataset_path}")
            print("Looking for folders containing '_frames'")
            
            # Try to find any folders
            all_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
            if all_folders:
                print(f"\nFound these folders instead:")
                for folder in all_folders:
                    print(f"  - {folder.name}")
            return None, None
        
        print(f"\nFound {len(phrase_folders)} phrase folders:")
        for folder in phrase_folders:
            print(f"  - {folder.name}")
        
        all_sequences = []
        all_labels = []
        
        for folder in phrase_folders:
            sequence, phrase_name = self.process_phrase_frames(folder)
            
            if sequence is not None:
                num_frames = len(sequence)
                print(f"  Total frames: {num_frames}")
                
                if num_frames >= sequence_length:
                    # Create multiple overlapping sequences
                    num_sequences = num_frames // sequence_length
                    stride = sequence_length // 2  # 50% overlap
                    
                    # Use sliding window to create more sequences
                    sequences_created = 0
                    for start in range(0, num_frames - sequence_length + 1, stride):
                        end = start + sequence_length
                        seq_chunk = sequence[start:end]
                        all_sequences.append(seq_chunk)
                        all_labels.append(phrase_name)
                        sequences_created += 1
                    
                    print(f"  Created {sequences_created} sequences of {sequence_length} frames")
                else:
                    # Pad shorter sequence
                    padded_sequence = np.zeros((sequence_length, 126))
                    padded_sequence[:num_frames] = sequence
                    all_sequences.append(padded_sequence)
                    all_labels.append(phrase_name)
                    print(f"  Padded to {sequence_length} frames (was {num_frames} frames)")
        
        if len(all_sequences) == 0:
            print("\n❌ No sequences were created!")
            return None, None
        
        print(f"\n✅ Total: {len(all_sequences)} sequences from {len(np.unique(all_labels))} phrases")
        
        return np.array(all_sequences), np.array(all_labels)
    
    def save_to_dataset(self, sequences, labels):
        """Save processed sequences to your dataset structure"""
        output_dir = 'dataset/phrases'
        os.makedirs(output_dir, exist_ok=True)
        
        unique_phrases = np.unique(labels)
        
        print(f"\nSaving to {output_dir}/")
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
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        if self.hands:
            self.hands.close()

# Alternative: Use OpenCV for basic hand detection if MediaPipe fails
class SimplePhraseLoader:
    """Fallback loader that creates dummy landmarks for testing"""
    
    def process_all_phrases(self, dataset_path, sequence_length=30):
        """Create dummy data for testing pipeline"""
        dataset_path = Path(dataset_path)
        
        # Find all phrase folders
        phrase_folders = [f for f in dataset_path.iterdir() 
                         if f.is_dir() and ('_frames' in f.name)]
        
        if not phrase_folders:
            print("No phrase folders found. Creating test data...")
            # Create test phrases
            test_phrases = ['hello', 'thanks', 'yes', 'no']
            all_sequences = []
            all_labels = []
            
            for phrase in test_phrases:
                for i in range(10):  # 10 sequences per phrase
                    dummy_sequence = np.random.rand(sequence_length, 126)
                    all_sequences.append(dummy_sequence)
                    all_labels.append(phrase)
            
            return np.array(all_sequences), np.array(all_labels)
        
        return None, None

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Loading Phrase Dataset from Frames")
    print("=" * 60)
    
    # Get current directory (where your _frames folders are)
    current_dir = os.getcwd()
    print(f"\nCurrent directory: {current_dir}")
    
    # List all folders in current directory
    print("\nFolders in current directory:")
    all_items = os.listdir(current_dir)
    folders = [item for item in all_items if os.path.isdir(os.path.join(current_dir, item))]
    for folder in folders:
        print(f"  📁 {folder}")
    
    # Initialize loader
    loader = PhraseFrameLoader()
    
    try:
        # Process all phrases
        sequences, labels = loader.process_all_phrases(current_dir, sequence_length=30)
        
        if sequences is not None and len(sequences) > 0:
            # Save to your dataset structure
            loader.save_to_dataset(sequences, labels)
            
            # Save metadata
            metadata = {
                'num_sequences': len(sequences),
                'num_phrases': len(np.unique(labels)),
                'phrases': list(np.unique(labels)),
                'sequence_length': sequences.shape[1] if len(sequences) > 0 else 0,
                'feature_size': sequences.shape[2] if len(sequences) > 0 else 0
            }
            
            with open('dataset/phrases/metadata.pickle', 'wb') as f:
                pickle.dump(metadata, f)
            
            print("\n" + "=" * 60)
            print("📊 Dataset Summary:")
            print("=" * 60)
            print(f"  Total sequences: {metadata['num_sequences']}")
            print(f"  Total phrases: {metadata['num_phrases']}")
            print(f"  Phrases: {', '.join(metadata['phrases'])}")
            print(f"  Sequence length: {metadata['sequence_length']} frames")
            print(f"  Features per frame: {metadata['feature_size']}")
            print("\n✅ Ready for next step!")
            
        else:
            print("\n⚠️ No valid frames could be processed.")
            print("\nTrying fallback loader with dummy data for testing...")
            
            fallback = SimplePhraseLoader()
            sequences, labels = fallback.process_all_phrases(current_dir)
            
            if sequences is not None:
                loader.save_to_dataset(sequences, labels)
                print("\n✅ Created test phrase data for pipeline testing")
                print("Note: Using random dummy data. Replace with real data later.")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your _frames folders are in the current directory")
        print("2. Check that the folders contain JPG/PNG images")
        print("3. Try updating mediapipe: pip install --upgrade mediapipe")
        print("4. Or reinstall mediapipe: pip uninstall mediapipe && pip install mediapipe")
    
    finally:
        loader.cleanup()