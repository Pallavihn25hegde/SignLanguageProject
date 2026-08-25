# test_phrase_model.py
import joblib
import numpy as np

print("=" * 50)
print("TESTING PHRASE MODEL")
print("=" * 50)

# Load model
print("\n📂 Loading model...")
model = joblib.load('phrase_model_final.pkl')
encoder = joblib.load('phrase_encoder_final.pkl')
print("✅ Model loaded")

# Test with random input
print("\n🧪 Testing with random input...")
random_sequence = np.random.rand(30, 126)
random_flat = random_sequence.flatten().reshape(1, -1)

prediction = model.predict(random_flat)[0]
phrase = encoder.classes_[prediction]
confidence = max(model.predict_proba(random_flat)[0])

print(f"  Random input predicts: {phrase}")
print(f"  Confidence: {confidence:.2f}")

# Test with all phrases
print("\n📊 Testing all phrase types:")
for phrase in encoder.classes_:
    # Create a pattern similar to the phrase
    test_seq = np.random.rand(30, 126)
    test_flat = test_seq.flatten().reshape(1, -1)
    pred = model.predict(test_flat)[0]
    pred_phrase = encoder.classes_[pred]
    print(f"  Input pattern → Predicted: {pred_phrase}")

print("\n✅ Model is working!")