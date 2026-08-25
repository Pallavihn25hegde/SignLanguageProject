# retrain_with_real_data.py
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("=" * 60)
print("RETRAINING WITH REAL DATA")
print("=" * 60)

# Load real letter data
print("\n📂 Loading real letter data...")
try:
    with open('real_letter_data.pickle', 'rb') as f:
        letter_data = pickle.load(f)
    X_letters = letter_data['data']
    y_letters = letter_data['labels']
    print(f"✅ Loaded {len(X_letters)} letter samples")
except:
    print("⚠️ No real letter data, using existing")
    with open('dataset/letters/data.pickle', 'rb') as f:
        letter_data = pickle.load(f)
    X_letters = np.array(letter_data['data'])
    y_letters = np.array(letter_data['labels'])
    print(f"✅ Loaded {len(X_letters)} existing letter samples")

# Load real phrase data
print("\n📂 Loading real phrase data...")
phrase_dir = Path('dataset/phrases_real')
if not phrase_dir.exists():
    phrase_dir = Path('dataset/phrases')

X_phrases = []
y_phrases = []

for phrase_folder in phrase_dir.iterdir():
    if phrase_folder.is_dir():
        phrase_name = phrase_folder.name
        for seq_file in phrase_folder.glob('*.npy'):
            sequence = np.load(seq_file)
            X_phrases.append(sequence)
            y_phrases.append(phrase_name)

if X_phrases:
    X_phrases = np.array(X_phrases)
    y_phrases = np.array(y_phrases)
    print(f"✅ Loaded {len(X_phrases)} phrase sequences")
else:
    print("⚠️ No phrase data found")

# Train letter model
if len(X_letters) > 0:
    print("\n📝 Training LETTER model...")
    letter_encoder = LabelEncoder()
    y_letters_enc = letter_encoder.fit_transform(y_letters)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_letters, y_letters_enc, test_size=0.2, random_state=42
    )
    
    letter_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    letter_model.fit(X_train, y_train)
    
    y_pred = letter_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.2%}")
    
    # Save
    joblib.dump(letter_model, 'letter_model_real.pkl')
    joblib.dump(letter_encoder, 'letter_encoder_real.pkl')

# Train phrase model
if len(X_phrases) > 0:
    print("\n🎬 Training PHRASE model...")
    X_phrases_flat = np.array([seq.flatten() for seq in X_phrases])
    
    phrase_encoder = LabelEncoder()
    y_phrases_enc = phrase_encoder.fit_transform(y_phrases)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_phrases_flat, y_phrases_enc, test_size=0.2, random_state=42
    )
    
    phrase_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    phrase_model.fit(X_train, y_train)
    
    y_pred = phrase_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.2%}")
    
    # Save
    joblib.dump(phrase_model, 'phrase_model_real.pkl')
    joblib.dump(phrase_encoder, 'phrase_encoder_real.pkl')

# Create unified models dictionary
models = {}
encoders = {}

if 'letter_model_real.pkl' in globals() or Path('letter_model_real.pkl').exists():
    models['letter'] = joblib.load('letter_model_real.pkl')
    encoders['letter'] = joblib.load('letter_encoder_real.pkl')
    print("\n✅ Letter model ready")

if 'phrase_model_real.pkl' in globals() or Path('phrase_model_real.pkl').exists():
    models['phrase'] = joblib.load('phrase_model_real.pkl')
    encoders['phrase'] = joblib.load('phrase_encoder_real.pkl')
    print("✅ Phrase model ready")

# Save unified models
joblib.dump(models, 'unified_models_real.pkl')
joblib.dump(encoders, 'unified_encoders_real.pkl')

print("\n" + "=" * 60)
print("✅ RETRAINING COMPLETE!")
print("\nNow run recognition with real models:")
print("  python unified_recognition_real.py")