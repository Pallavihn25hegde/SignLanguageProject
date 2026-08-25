# complete_integration_fixed_v2.py
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("=" * 70)
print("COMPLETE INTEGRATION: LETTERS + PHRASES")
print("=" * 70)

# ============================================
# STEP 1: Load Letter Data
# ============================================
print("\n📂 STEP 1: Loading letter data...")

try:
    with open('dataset/letters/data.pickle', 'rb') as f:
        letter_data_raw = pickle.load(f)
    
    print(f"✅ Found letter data")
    
    # Extract data and labels
    if isinstance(letter_data_raw, dict):
        if 'data' in letter_data_raw and 'labels' in letter_data_raw:
            X_letters = np.array(letter_data_raw['data'])
            y_letters = np.array(letter_data_raw['labels'])
        else:
            keys = list(letter_data_raw.keys())
            X_letters = np.array(letter_data_raw[keys[0]])
            y_letters = np.array(letter_data_raw[keys[1]])
    else:
        X_letters = np.array(letter_data_raw[0])
        y_letters = np.array(letter_data_raw[1])
    
    print(f"  Samples: {len(X_letters)}")
    print(f"  Classes: {len(np.unique(y_letters))}")
    print(f"  Feature shape: {X_letters[0].shape if len(X_letters) > 0 else 'Unknown'}")
    
except Exception as e:
    print(f"❌ Error loading letter data: {e}")
    X_letters = None
    y_letters = None

# ============================================
# STEP 2: Load Phrase Data
# ============================================
print("\n📂 STEP 2: Loading phrase data...")

phrase_dir = Path('dataset/phrases')
X_phrases = []
y_phrases = []

if phrase_dir.exists():
    for phrase_folder in phrase_dir.iterdir():
        if phrase_folder.is_dir():
            phrase_name = phrase_folder.name
            seq_files = list(phrase_folder.glob('*.npy'))
            
            for seq_file in seq_files:
                sequence = np.load(seq_file)
                X_phrases.append(sequence)
                y_phrases.append(phrase_name)
    
    if X_phrases:
        X_phrases = np.array(X_phrases)
        y_phrases = np.array(y_phrases)
        print(f"  Sequences: {len(X_phrases)}")
        print(f"  Classes: {len(np.unique(y_phrases))}")
        print(f"  Sequence shape: {X_phrases[0].shape}")
    else:
        X_phrases = None
        y_phrases = None
else:
    X_phrases = None
    y_phrases = None

# ============================================
# STEP 3: Train Separate Models (Better Approach)
# ============================================
print("\n" + "=" * 70)
print("STEP 3: Training separate models for letters and phrases")
print("=" * 70)

models = {}
encoders = {}

# Train Letter Model
if X_letters is not None and len(X_letters) > 0:
    print("\n📝 Training LETTER model...")
    
    # Encode letter labels
    letter_encoder = LabelEncoder()
    y_letters_encoded = letter_encoder.fit_transform(y_letters)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_letters, y_letters_encoded, test_size=0.2, random_state=42, stratify=y_letters_encoded
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Train Random Forest
    letter_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    letter_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = letter_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  ✅ Letter Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    models['letter'] = letter_model
    encoders['letter'] = letter_encoder

# Train Phrase Model
if X_phrases is not None and len(X_phrases) > 0:
    print("\n🎬 Training PHRASE model...")
    
    # Flatten sequences
    X_phrases_flat = np.array([seq.flatten() for seq in X_phrases])
    print(f"  Flattened shape: {X_phrases_flat.shape}")
    
    # Encode phrase labels
    phrase_encoder = LabelEncoder()
    y_phrases_encoded = phrase_encoder.fit_transform(y_phrases)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_phrases_flat, y_phrases_encoded, test_size=0.2, random_state=42, stratify=y_phrases_encoded
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Train Random Forest
    phrase_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    phrase_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = phrase_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  ✅ Phrase Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    models['phrase'] = phrase_model
    encoders['phrase'] = phrase_encoder

# ============================================
# STEP 4: Create Combined Predictor
# ============================================
print("\n" + "=" * 70)
print("STEP 4: Creating unified predictor")
print("=" * 70)

class UnifiedSignLanguageRecognizer:
    """Combined recognizer for letters and phrases"""
    
    def __init__(self, models, encoders):
        self.models = models
        self.encoders = encoders
        self.phrase_buffer = []
        self.buffer_size = 30
        
    def recognize_letter(self, features):
        """Recognize a single letter"""
        if 'letter' not in self.models:
            return None, 0
        
        features = np.array(features).reshape(1, -1)
        pred = self.models['letter'].predict(features)[0]
        confidence = max(self.models['letter'].predict_proba(features)[0])
        letter = self.encoders['letter'].inverse_transform([pred])[0]
        
        return letter, confidence
    
    def recognize_phrase(self, sequence):
        """Recognize a phrase from sequence"""
        if 'phrase' not in self.models:
            return None, 0
        
        flat_sequence = np.array(sequence).flatten().reshape(1, -1)
        pred = self.models['phrase'].predict(flat_sequence)[0]
        confidence = max(self.models['phrase'].predict_proba(flat_sequence)[0])
        phrase = self.encoders['phrase'].inverse_transform([pred])[0]
        
        return phrase, confidence
    
    def add_frame(self, landmarks):
        """Add frame to phrase buffer"""
        self.phrase_buffer.append(landmarks)
        if len(self.phrase_buffer) > self.buffer_size:
            self.phrase_buffer.pop(0)
    
    def get_phrase(self):
        """Get phrase from current buffer"""
        if len(self.phrase_buffer) == self.buffer_size:
            return self.recognize_phrase(self.phrase_buffer)
        return None, 0
    
    def clear_buffer(self):
        """Clear phrase buffer"""
        self.phrase_buffer = []

# Save the unified recognizer
print("\n💾 Saving unified recognizer...")

# Save models
joblib.dump(models, 'unified_models.pkl')
joblib.dump(encoders, 'unified_encoders.pkl')

# Save metadata
metadata = {
    'letter_samples': len(X_letters) if X_letters is not None else 0,
    'letter_classes': len(encoders['letter'].classes_) if 'letter' in encoders else 0,
    'phrase_sequences': len(X_phrases) if X_phrases is not None else 0,
    'phrase_classes': len(encoders['phrase'].classes_) if 'phrase' in encoders else 0,
}

with open('unified_metadata.pickle', 'wb') as f:
    pickle.dump(metadata, f)

print("✅ Models saved to 'unified_models.pkl'")
print("✅ Encoders saved to 'unified_encoders.pkl'")

# ============================================
# STEP 5: Summary
# ============================================
print("\n" + "=" * 70)
print("✅✅✅ INTEGRATION COMPLETE! ✅✅✅")
print("=" * 70)

print("\n📊 FINAL SUMMARY:")
if 'letter' in models:
    print(f"  📝 LETTER MODEL:")
    print(f"     • Samples: {metadata['letter_samples']}")
    print(f"     • Classes: {metadata['letter_classes']}")
    print(f"     • Features: {X_letters.shape[1] if X_letters is not None else 'N/A'}")

if 'phrase' in models:
    print(f"\n  🎬 PHRASE MODEL:")
    print(f"     • Sequences: {metadata['phrase_sequences']}")
    print(f"     • Classes: {metadata['phrase_classes']}")
    print(f"     • Frames per sequence: 30")
    print(f"     • Features per frame: 126")

print("\n🎯 NOW RUN:")
print("  python unified_recognition_v2.py")