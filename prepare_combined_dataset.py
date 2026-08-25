import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob

def load_letter_data():
    """Load your existing letter dataset from data.pickle"""
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    # Your existing data structure
    X_letters = data['data']  # List of landmark arrays
    y_letters = data['labels']  # List of labels
    
    return np.array(X_letters), np.array(y_letters)

def load_phrase_data():
    """Load phrase sequence data"""
    X_phrases = []
    y_phrases = []
    
    phrase_paths = glob.glob('dataset/phrases/*/')
    
    for phrase_path in phrase_paths:
        phrase_name = os.path.basename(os.path.dirname(phrase_path))
        sequence_files = glob.glob(f'{phrase_path}*.npy')
        
        for seq_file in sequence_files:
            sequence = np.load(seq_file)
            X_phrases.append(sequence)
            y_phrases.append(phrase_name)
    
    return np.array(X_phrases, dtype=object), np.array(y_phrases)

def prepare_features_for_phrase_model(X_phrases):
    """Prepare phrase sequences for LSTM model"""
    # Each phrase is a sequence of frames
    # We'll flatten each frame and keep sequence structure
    sequences = []
    for sequence in X_phrases:
        # sequence shape: (sequence_length, 126)
        sequences.append(sequence)
    
    return np.array(sequences, dtype=object)

def create_combined_dataset():
    """Create combined dataset for both letter and phrase recognition"""
    print("Loading letter dataset...")
    X_letters, y_letters = load_letter_data()
    
    print("Loading phrase dataset...")
    X_phrases, y_phrases = load_phrase_data()
    
    # Prepare phrase data for training
    X_phrases_prepared = prepare_features_for_phrase_model(X_phrases)
    
    # Save separate datasets
    combined_data = {
        'letters': {
            'data': X_letters,
            'labels': y_letters,
            'type': 'single_frame'
        },
        'phrases': {
            'data': X_phrases_prepared,
            'labels': y_phrases,
            'type': 'sequence'
        }
    }
    
    # Save combined dataset
    with open('combined_data.pickle', 'wb') as f:
        pickle.dump(combined_data, f)
    
    print(f"\nDataset Summary:")
    print(f"Letters: {len(X_letters)} samples, {len(np.unique(y_letters))} classes")
    print(f"Phrases: {len(X_phrases)} sequences, {len(np.unique(y_phrases))} classes")
    
    # Save label encoders
    letter_encoder = LabelEncoder()
    letter_encoder.fit(y_letters)
    
    phrase_encoder = LabelEncoder()
    phrase_encoder.fit(y_phrases)
    
    with open('label_encoders.pickle', 'wb') as f:
        pickle.dump({'letters': letter_encoder, 'phrases': phrase_encoder}, f)
    
    return combined_data

if __name__ == "__main__":
    create_combined_dataset()