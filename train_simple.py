# train_simple.py
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading phrase data...")
with open('combined_data.pickle', 'rb') as f:
    data = pickle.load(f)

# For phrases, we need to flatten the sequences
X = data['phrases']['data']
y = data['phrases']['labels']

# Flatten each sequence (30 frames * 126 features = 3780 features)
X_flat = np.array([seq.flatten() for seq in X])

# Encode labels
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_encoded, test_size=0.2, random_state=42
)

print(f"Training: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")
print(f"Features per sample: {X_flat.shape[1]}")

# Train Random Forest (no TensorFlow needed)
print("\nTraining Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")

# Save model
import joblib
joblib.dump(clf, 'phrase_model_rf.pkl')
joblib.dump(encoder, 'phrase_encoder_rf.pkl')
print("✅ Model saved to 'phrase_model_rf.pkl'")