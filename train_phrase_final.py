# train_phrase_final.py
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

print("=" * 70)
print("TRAINING PHRASE RECOGNITION MODEL")
print("=" * 70)

# Load data
print("\n📂 Loading data...")

if Path('combined_data.pickle').exists():
    with open('combined_data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    if 'phrases' in data:
        X = data['phrases']['data']
        y = data['phrases']['labels']
        print("  ✅ Loaded from combined_data.pickle")
    else:
        print("  ❌ Invalid data format")
        exit()
elif Path('phrase_only_data.pickle').exists():
    with open('phrase_only_data.pickle', 'rb') as f:
        data = pickle.load(f)
    X = data['data']
    y = data['labels']
    print("  ✅ Loaded from phrase_only_data.pickle")
else:
    print("  ❌ No data file found!")
    print("  Please run 'python create_phrase_data.py' first")
    exit()

# Check if we have data
print(f"\n📊 Dataset info:")
print(f"  Total sequences: {len(X)}")
print(f"  Unique phrases: {np.unique(y)}")
print(f"  Number of phrase types: {len(np.unique(y))}")

if len(X) == 0:
    print("\n❌ No sequences found!")
    print("Please run 'python create_phrase_data.py' first")
    exit()

# Flatten sequences for Random Forest
print("\n🔄 Preparing features...")
X_flat = np.array([seq.flatten() for seq in X])
print(f"  Feature size per sample: {X_flat.shape[1]:,} features")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

# Train model
print("\n🤖 Training Random Forest classifier...")
print("  (This may take a moment...)")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train, y_train)
print("  ✅ Training complete")

# Evaluate
print("\n📈 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"✅ TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"{'='*50}")

# Detailed report
print("\n📋 Detailed Classification Report:")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Confusion matrix
print("\n📊 Confusion Matrix (first 10x10):")
cm = confusion_matrix(y_test, y_pred)
print("Rows: Actual, Columns: Predicted")
print("-" * 50)
for i in range(min(10, len(encoder.classes_))):
    row = cm[i][:min(10, len(encoder.classes_))]
    print(f"{encoder.classes_[i]:12s} | {row}")

# Save model
print("\n💾 Saving model...")
joblib.dump(model, 'phrase_model_final.pkl')
joblib.dump(encoder, 'phrase_encoder_final.pkl')
print("  ✅ Model saved to 'phrase_model_final.pkl'")
print("  ✅ Encoder saved to 'phrase_encoder_final.pkl'")

# Test prediction on a few samples
print("\n🧪 Testing sample predictions:")
for i in range(min(5, len(X_test))):
    sample = X_test[i].reshape(1, -1)
    pred = model.predict(sample)[0]
    actual = y_test[i]
    confidence = max(model.predict_proba(sample)[0])
    print(f"  Sample {i+1}: Predicted={encoder.classes_[pred]}, Actual={encoder.classes_[actual]}, Confidence={confidence:.2f}")

print("\n" + "=" * 70)
print("✅✅✅ MODEL TRAINING COMPLETE! ✅✅✅")
print("=" * 70)

print("\n🎯 NEXT STEPS:")
print("  1. Your model is ready for phrase recognition")
print("  2. To test: python test_phrase_model.py")
print("  3. For real-time: python realtime_phrase_recognition.py")
