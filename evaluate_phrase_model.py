import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_phrase_model():
    """Evaluate the phrase recognition model"""
    # Load model
    model = tf.keras.models.load_model('phrase_model.h5')
    
    # Load data
    with open('combined_data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    with open('phrase_label_encoder.pickle', 'rb') as f:
        encoder = pickle.load(f)
    
    X = data['phrases']['data']
    y = data['phrases']['labels']
    y_encoded = encoder.transform(y)
    
    # Pad sequences
    max_length = max([len(seq) for seq in X])
    X_padded = []
    for seq in X:
        if len(seq) < max_length:
            padding = np.zeros((max_length - len(seq), seq.shape[1]))
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[:max_length]
        X_padded.append(padded_seq)
    
    X = np.array(X_padded)
    
    # Predict
    predictions = model.predict(X)
    y_pred = np.argmax(predictions, axis=1)
    y_true = y_encoded
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=encoder.classes_, 
                yticklabels=encoder.classes_)
    plt.title('Confusion Matrix - Phrase Recognition')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('phrase_confusion_matrix.png')
    plt.show()
    
    # Calculate accuracy per class
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    for i, class_name in enumerate(encoder.classes_):
        print(f"{class_name}: {accuracy_per_class[i]:.2%}")

if __name__ == "__main__":
    evaluate_phrase_model()