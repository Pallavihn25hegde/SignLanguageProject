# create_phrase_data.py
import os
import numpy as np
import pickle
from pathlib import Path

print("=" * 70)
print("CREATING PHRASE DATASET FOR TESTING")
print("=" * 70)

# Create directories
os.makedirs('dataset/phrases', exist_ok=True)
print("✅ Created directories")

# Define your phrases (match your folder names)
phrases = [
    'Excuseme', 'Far', 'Fare', 'Hello', 'Here', 
    'Left', 'Near', 'Right', 'Taxi', 'Thanks', 
    'There', 'Where'
]

print(f"\n📝 Creating data for {len(phrases)} phrases:")
for p in phrases:
    print(f"  - {p}")

# Create sequences
SEQUENCE_LENGTH = 30
FEATURES = 126
SEQUENCES_PER_PHRASE = 20  # Number of sequences per phrase

all_sequences = []
all_labels = []

for phrase in phrases:
    phrase_dir = Path(f'dataset/phrases/{phrase}')
    phrase_dir.mkdir(parents=True, exist_ok=True)
    
    phrase_sequences = []
    
    for i in range(SEQUENCES_PER_PHRASE):
        # Create realistic random data (will be replaced with real landmarks later)
        # Each sequence: 30 frames * 126 features
        sequence = np.random.rand(SEQUENCE_LENGTH, FEATURES)
        
        # Add some pattern to make it realistic (different phrases have different patterns)
        # This ensures the model can learn something
        pattern = np.sin(np.linspace(0, np.pi * (phrases.index(phrase) + 1), SEQUENCE_LENGTH))
        for frame_idx in range(SEQUENCE_LENGTH):
            sequence[frame_idx] += pattern[frame_idx] * 0.1
        
        # Save individual sequence
        seq_path = phrase_dir / f'seq_{i:03d}.npy'
        np.save(seq_path, sequence)
        
        phrase_sequences.append(sequence)
        all_sequences.append(sequence)
        all_labels.append(phrase)
    
    print(f"  ✅ Created {len(phrase_sequences)} sequences for '{phrase}'")

# Convert to numpy arrays
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)

# Save metadata
metadata = {
    'num_sequences': len(all_sequences),
    'num_phrases': len(phrases),
    'phrases': phrases,
    'sequence_length': SEQUENCE_LENGTH,
    'feature_size': FEATURES,
    'sequences_per_phrase': SEQUENCES_PER_PHRASE
}

with open('dataset/phrases/metadata.pickle', 'wb') as f:
    pickle.dump(metadata, f)

print(f"\n📊 Total created:")
print(f"  • Total sequences: {len(all_sequences)}")
print(f"  • Total phrases: {len(phrases)}")
print(f"  • Sequence length: {SEQUENCE_LENGTH}")
print(f"  • Features per frame: {FEATURES}")

# Step 2: Check if letter data exists and combine
print("\n🔗 Checking for letter dataset...")

if os.path.exists('data.pickle'):
    with open('data.pickle', 'rb') as f:
        letter_data = pickle.load(f)
    
    print(f"  ✅ Found letter dataset with {len(letter_data['data'])} samples")
    
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
    
    print("  ✅ Combined dataset saved to 'combined_data.pickle'")
else:
    print("  ⚠️ No letter dataset found (data.pickle missing)")
    print("  Saving phrase-only dataset")
    
    phrase_only = {
        'data': all_sequences,
        'labels': all_labels
    }
    
    with open('phrase_only_data.pickle', 'wb') as f:
        pickle.dump(phrase_only, f)
    
    # Also create combined with just phrases
    combined = {
        'phrases': {
            'data': all_sequences,
            'labels': all_labels,
            'type': 'sequence'
        }
    }
    
    with open('combined_data.pickle', 'wb') as f:
        pickle.dump(combined, f)
    
    print("  ✅ Phrase-only dataset saved")

print("\n" + "=" * 70)
print("✅✅✅ DATA CREATED SUCCESSFULLY! ✅✅✅")
print("=" * 70)
print("\nNow run: python train_phrase_final.py")