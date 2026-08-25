# unified_dataset.py
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class UnifiedDataset:
    """Load and manage both letter and phrase datasets"""
    
    def __init__(self):
        self.letter_data = None
        self.phrase_data = None
        self.letter_encoder = None
        self.phrase_encoder = None
        
    def load_letter_dataset(self):
        """Load your existing letter dataset"""
        if os.path.exists('data.pickle'):
            with open('data.pickle', 'rb') as f:
                data = pickle.load(f)
            self.letter_data = {
                'data': np.array(data['data']),
                'labels': np.array(data['labels'])
            }
            print(f"✅ Letters: {len(self.letter_data['data'])} samples, "
                  f"{len(np.unique(self.letter_data['labels']))} classes")
            return True
        else:
            print("⚠️ No letter dataset found")
            return False
    
    def load_phrase_dataset(self):
        """Load your phrase dataset"""
        phrase_dir = Path('dataset/phrases')
        
        if not phrase_dir.exists():
            print("⚠️ No phrase dataset found")
            return False
        
        sequences = []
        labels = []
        
        for phrase_folder in phrase_dir.iterdir():
            if phrase_folder.is_dir():
                phrase_name = phrase_folder.name
                seq_files = list(phrase_folder.glob('*.npy'))
                
                for seq_file in seq_files:
                    sequence = np.load(seq_file)
                    sequences.append(sequence)
                    labels.append(phrase_name)
        
        if sequences:
            self.phrase_data = {
                'data': np.array(sequences),
                'labels': np.array(labels)
            }
            print(f"✅ Phrases: {len(self.phrase_data['data'])} sequences, "
                  f"{len(np.unique(self.phrase_data['labels']))} classes")
            return True
        return False
    
    def prepare_features(self):
        """Prepare features for training"""
        X_letters = None
        y_letters = None
        X_phrases = None
        y_phrases = None
        
        # Prepare letter data (already flat)
        if self.letter_data:
            X_letters = self.letter_data['data']
            y_letters = self.letter_data['labels']
            
            # Encode letter labels
            self.letter_encoder = LabelEncoder()
            y_letters = self.letter_encoder.fit_transform(y_letters)
        
        # Prepare phrase data (flatten sequences)
        if self.phrase_data:
            # Flatten each sequence: (30, 126) -> (3780,)
            X_phrases = np.array([seq.flatten() for seq in self.phrase_data['data']])
            y_phrases = self.phrase_data['labels']
            
            # Encode phrase labels
            self.phrase_encoder = LabelEncoder()
            y_phrases = self.phrase_encoder.fit_transform(y_phrases)
        
        return {
            'letters': (X_letters, y_letters) if X_letters is not None else None,
            'phrases': (X_phrases, y_phrases) if X_phrases is not None else None
        }
    
    def create_combined_dataset(self):
        """Combine both datasets into one"""
        X_combined = []
        y_combined = []
        types = []  # Track if sample is letter or phrase
        
        # Add letter data
        if self.letter_data:
            for i, (features, label) in enumerate(zip(self.letter_data['data'], 
                                                        self.letter_data['labels'])):
                X_combined.append(features)
                y_combined.append(f"LETTER_{label}")
                types.append('letter')
        
        # Add phrase data (flattened)
        if self.phrase_data:
            for i, (sequence, label) in enumerate(zip(self.phrase_data['data'], 
                                                        self.phrase_data['labels'])):
                # Flatten sequence
                flat_sequence = sequence.flatten()
                X_combined.append(flat_sequence)
                y_combined.append(f"PHRASE_{label}")
                types.append('phrase')
        
        return {
            'data': np.array(X_combined, dtype=object),
            'labels': np.array(y_combined),
            'types': np.array(types),
            'letter_encoder': self.letter_encoder,
            'phrase_encoder': self.phrase_encoder
        }
    
    def save_combined(self, filename='unified_data.pickle'):
        """Save combined dataset"""
        combined = self.create_combined_dataset()
        with open(filename, 'wb') as f:
            pickle.dump(combined, f)
        print(f"\n✅ Saved unified dataset to {filename}")
        print(f"   Total samples: {len(combined['data'])}")
        print(f"   Letters: {sum(combined['types'] == 'letter')}")
        print(f"   Phrases: {sum(combined['types'] == 'phrase')}")
        return combined

# Run this to create unified dataset
if __name__ == "__main__":
    print("=" * 60)
    print("CREATING UNIFIED DATASET")
    print("=" * 60)
    
    loader = UnifiedDataset()
    loader.load_letter_dataset()
    loader.load_phrase_dataset()
    loader.save_combined()