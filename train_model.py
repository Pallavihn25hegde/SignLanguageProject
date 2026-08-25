import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==========================
# Dataset Path
# ==========================

DATASET_PATH = "dataset"   # change to your kaggle dataset folder
IMG_SIZE = 64

data = []
labels = []

# ==========================
# Load Dataset
# ==========================

for label in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    for img_file in os.listdir(folder):

        img_path = os.path.join(folder, img_file)

        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data.append(img)
            labels.append(label)

        except:
            pass

data = np.array(data)
labels = np.array(labels)

# Normalize
data = data / 255.0
data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# ==========================
# Encode Labels
# ==========================

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# ==========================
# CNN Model
# ==========================

model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(len(lb.classes_), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train Model
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==========================
# Evaluate Model
# ==========================

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# ==========================
# Save Model
# ==========================

model.save("sign_language_model.h5")

# ==========================
# Accuracy Graph
# ==========================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"])
plt.show()
