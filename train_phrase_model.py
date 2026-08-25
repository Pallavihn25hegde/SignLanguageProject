import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_phrase_data():
    """Load phrase data from combined dataset"""
    with open('combined_data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    X = data['phrases']['data']
    y = data['phrases']['labels']
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Save encoder
    with open('phrase_label_encoder.pickle', 'wb') as f:
        pickle.dump(encoder, f)
    
    return X, y_encoded, encoder

def build_lstm_model(input_shape, num_classes):
    """Build LSTM model for phrase recognition"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # LSTM layers for temporal patterns
        layers.LSTM(128, return_sequences=True, dropout=0.3),
        layers.LSTM(64, dropout=0.3),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_phrase_model():
    """Train the phrase recognition model"""
    print("Loading phrase data...")
    X, y, encoder = load_phrase_data()
    
    # Convert to numpy array with padding
    max_sequence_length = max([len(seq) for seq in X])
    print(f"Max sequence length: {max_sequence_length}")
    
    # Pad sequences to same length
    X_padded = []
    for seq in X:
        if len(seq) < max_sequence_length:
            # Pad with zeros
            padding = np.zeros((max_sequence_length - len(seq), seq.shape[1]))
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[:max_sequence_length]
        X_padded.append(padded_seq)
    
    X = np.array(X_padded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Build model
    input_shape = (X.shape[1], X.shape[2])  # (sequence_length, features)
    model = build_lstm_model(input_shape, len(np.unique(y)))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint('best_phrase_model.h5', save_best_only=True)
    ]
    
    # Train model
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('phrase_model.h5')
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    return model, history

if __name__ == "__main__":
    train_phrase_model()