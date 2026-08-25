# train_unified_model.py
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("=" * 60)
print("TRAINING UNIFIED LETTER + PHRASE MODEL")
print("=" * 60)

# Load unified dataset
print("\n📂 Loading unified dataset...")
with open('unified_data.pickle', 'rb') as f:
    data = pickle.load(f)

X = data['data']
y = data['labels']
types = data['types']

print(f"✅ Loaded {len(X)} total samples")
print(f"   Letters: {sum(types == 'letter')}")
print(f"   Phrases: {sum(types == 'phrase')}")
print(f"   Total classes: {len(np.unique(y))}")

# Convert to numpy arrays
X_array = np.array([x for x in X])  # Convert object array to float array
y_array = np.array(y)

# Split data
X_train, X_test, y_train, y_test, types_train, types_test = train_test_split(
    X_array, y_array, types, test_size=0.2, random_state=42, stratify=y_array
)

print(f"\n📊 Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# Train Random Forest
print("\n🤖 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=30,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# Per-type accuracy
letter_mask = types_test == 'letter'
phrase_mask = types_test == 'phrase'

if any(letter_mask):
    letter_acc = accuracy_score(y_test[letter_mask], y_pred[letter_mask])
    print(f"\n📝 Letter Accuracy: {letter_acc:.4f} ({letter_acc*100:.1f}%)")

if any(phrase_mask):
    phrase_acc = accuracy_score(y_test[phrase_mask], y_pred[phrase_mask])
    print(f"🎬 Phrase Accuracy: {phrase_acc:.4f} ({phrase_acc*100:.1f}%)")

# Save model
joblib.dump(model, 'unified_model.pkl')
joblib.dump(data['letter_encoder'], 'letter_encoder.pkl')
joblib.dump(data['phrase_encoder'], 'phrase_encoder.pkl')

print("\n✅ Model saved to 'unified_model.pkl'")

# Classification report
print("\n📋 Classification Report (top 20 classes):")
y_test_decoded = y_test
y_pred_decoded = y_pred
unique_classes = np.unique(y_test)[:20]
print(classification_report(y_test, y_pred, labels=unique_classes))