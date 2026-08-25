# test_model_performance.py
import numpy as np
import joblib
import pickle
from sklearn.metrics import accuracy_score

print("=" * 60)
print("TESTING MODEL PERFORMANCE")
print("=" * 60)

# Load models
models = joblib.load('unified_models.pkl')
encoders = joblib.load('unified_encoders.pkl')

# Load your actual data
print("\n📂 Loading training data...")

# Load letter data
with open('dataset/letters/data.pickle', 'rb') as f:
    letter_data = pickle.load(f)

X_letters = np.array(letter_data['data'])
y_letters = np.array(letter_data['labels'])

print(f"Letter data: {len(X_letters)} samples, {len(np.unique(y_letters))} classes")

# Load phrase data
import os
from pathlib import Path

X_phrases = []
y_phrases = []
phrase_dir = Path('dataset/phrases')

for phrase_folder in phrase_dir.iterdir():
    if phrase_folder.is_dir():
        phrase_name = phrase_folder.name
        seq_files = list(phrase_folder.glob('*.npy'))
        for seq_file in seq_files:
            sequence = np.load(seq_file)
            X_phrases.append(sequence)
            y_phrases.append(phrase_name)

X_phrases = np.array(X_phrases)
y_phrases = np.array(y_phrases)

print(f"Phrase data: {len(X_phrases)} sequences, {len(np.unique(y_phrases))} classes")

# Test letter model on its own training data
print("\n📝 Testing LETTER model...")
if 'letter' in models:
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_letters)
    
    # Test on first 100 samples
    X_test = X_letters[:100]
    y_test = y_encoded[:100]
    
    predictions = models['letter'].predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"  Accuracy on test samples: {accuracy:.2%}")
    
    # Show sample predictions
    print("\n  Sample predictions:")
    for i in range(5):
        pred = encoder.inverse_transform([predictions[i]])[0]
        actual = y_letters[i]
        print(f"    {i+1}. Predicted: {pred}, Actual: {actual}")

# Test phrase model
print("\n🎬 Testing PHRASE model...")
if 'phrase' in models:
    # Flatten sequences
    X_flat = np.array([seq.flatten() for seq in X_phrases])
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_phrases)
    
    # Test on first 20 sequences
    X_test = X_flat[:20]
    y_test = y_encoded[:20]
    
    predictions = models['phrase'].predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"  Accuracy on test samples: {accuracy:.2%}")
    
    # Show sample predictions
    print("\n  Sample predictions:")
    for i in range(5):
        pred = encoder.inverse_transform([predictions[i]])[0]
        actual = y_phrases[i]
        print(f"    {i+1}. Predicted: {pred}, Actual: {actual}")

print("\n" + "=" * 60)
print("If accuracy is low, the model needs better training data.")
print="=" * 60