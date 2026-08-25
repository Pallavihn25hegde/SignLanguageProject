# complete_integration_fixed.py
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
    
    # Extract data and labels based on structure
    if isinstance(letter_data_raw, dict):
        # Try different possible key names
        if 'data' in letter_data_raw and 'labels' in letter_data_raw:
            X_letters = np.array(letter_data_raw['data'])
            y_letters = np.array(letter_data_raw['labels'])
        elif 'X' in letter_data_raw and 'y' in letter_data_raw:
            X_letters = np.array(letter_data_raw['X'])
            y_letters = np.array(letter_data_raw['y'])
        else:
            # Try first key for data, second for labels
            keys = list(letter_data_raw.keys())
            X_letters = np.array(letter_data_raw[keys[0]])
            y_letters = np.array(letter_data_raw[keys[1]])
    else:
        # If it's a tuple or list
        X_letters = np.array(letter_data_raw[0])
        y_letters = np.array(letter_data_raw[1])
    
    # Ensure X_letters is 2D
    if len(X_letters.shape) == 1:
        X_letters = X_letters.reshape(-1, 1)
    
    print(f"  Samples: {len(X_letters)}")
    print(f"  Classes: {len(np.unique(y_letters))}")
    print(f"  Features per sample: {X_letters.shape[1] if len(X_letters.shape) > 1 else 1}")
    
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
        print("  No phrase data found")
        X_phrases = None
        y_phrases = None
else:
    print("  Phrase directory not found")
    X_phrases = None
    y_phrases = None

# ============================================
# STEP 3: Create Unified Dataset
# ============================================
print("\n📊 STEP 3: Creating unified dataset...")

X_unified = []
y_unified = []
source_type = []

# Add letter data
if X_letters is not None and len(X_letters) > 0:
    for i, (features, label) in enumerate(zip(X_letters, y_letters)):
        # Ensure features are flat
        if len(features.shape) > 1:
            features = features.flatten()
        X_unified.append(features)
        y_unified.append(f"LETTER_{label}")
        source_type.append('letter')
    print(f"  Added {len(X_letters)} letter samples")

# Add phrase data
if X_phrases is not None and len(X_phrases) > 0:
    for sequence, label in zip(X_phrases, y_phrases):
        # Flatten sequence (30 frames * 126 features = 3780 features)
        flat_sequence = sequence.flatten()
        X_unified.append(flat_sequence)
        y_unified.append(f"PHRASE_{label}")
        source_type.append('phrase')
    print(f"  Added {len(X_phrases)} phrase sequences")

# Convert to numpy arrays
X_unified = np.array(X_unified)
y_unified = np.array(y_unified)
source_type = np.array(source_type)

print(f"\n✅ Unified dataset created:")
print(f"  Total samples: {len(X_unified)}")
print(f"  Letters: {sum(source_type == 'letter')}")
print(f"  Phrases: {sum(source_type == 'phrase')}")
print(f"  Total classes: {len(np.unique(y_unified))}")

# Save unified dataset
with open('unified_data_complete.pickle', 'wb') as f:
    pickle.dump({
        'data': X_unified,
        'labels': y_unified,
        'types': source_type
    }, f)
print("✅ Saved to 'unified_data_complete.pickle'")

# ============================================
# STEP 4: Train Unified Model
# ============================================
print("\n🤖 STEP 4: Training unified model...")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_unified)

# Split data
X_train, X_test, y_train, y_test, types_train, types_test = train_test_split(
    X_unified, y_encoded, source_type, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

# Train Random Forest
print("\n  Training Random Forest (this may take a moment)...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=30,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train, y_train)
print("  ✅ Training complete")

# ============================================
# STEP 5: Evaluate
# ============================================
print("\n📈 STEP 5: Evaluating model...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"✅ OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"{'='*50}")

# Per-type accuracy
letter_mask = types_test == 'letter'
phrase_mask = types_test == 'phrase'

if sum(letter_mask) > 0:
    letter_acc = accuracy_score(y_test[letter_mask], y_pred[letter_mask])
    print(f"\n📝 LETTER ACCURACY: {letter_acc:.4f} ({letter_acc*100:.1f}%)")
    print(f"   (Tested on {sum(letter_mask)} letter samples)")

if sum(phrase_mask) > 0:
    phrase_acc = accuracy_score(y_test[phrase_mask], y_pred[phrase_mask])
    print(f"\n🎬 PHRASE ACCURACY: {phrase_acc:.4f} ({phrase_acc*100:.1f}%)")
    print(f"   (Tested on {sum(phrase_mask)} phrase samples)")

# ============================================
# STEP 6: Save Model
# ============================================
print("\n💾 STEP 6: Saving model...")

joblib.dump(model, 'unified_model_complete.pkl')
joblib.dump(label_encoder, 'unified_label_encoder.pkl')
print("✅ Model saved to 'unified_model_complete.pkl'")
print("✅ Encoder saved to 'unified_label_encoder.pkl'")

# ============================================
# STEP 7: Test Predictions
# ============================================
print("\n🧪 STEP 7: Testing sample predictions...")

# Test on a few samples
for i in range(min(5, len(X_test))):
    sample = X_test[i].reshape(1, -1)
    pred = model.predict(sample)[0]
    actual = y_test[i]
    pred_label = label_encoder.inverse_transform([pred])[0]
    actual_label = label_encoder.inverse_transform([actual])[0]
    
    sample_type = types_test[i]
    confidence = max(model.predict_proba(sample)[0])
    
    print(f"  {i+1}. Type: {sample_type.upper()}")
    print(f"     Predicted: {pred_label}")
    print(f"     Actual: {actual_label}")
    print(f"     Confidence: {confidence:.2f}")
    print()

print("\n" + "=" * 70)
print("✅✅✅ INTEGRATION COMPLETE! ✅✅✅")
print("=" * 70)
print("\n📊 SUMMARY:")
print(f"  • Letter samples: {len(X_letters) if X_letters is not None else 0}")
print(f"  • Phrase sequences: {len(X_phrases) if X_phrases is not None else 0}")
print(f"  • Total classes: {len(label_encoder.classes_)}")
print(f"  • Overall accuracy: {accuracy:.1%}")

print("\n🎯 NOW RUN:")
print("  python unified_recognition_complete.py")