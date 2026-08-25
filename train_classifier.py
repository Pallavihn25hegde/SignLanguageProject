import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Load the landmarks
data_dict = pickle.load(open('./data.pickle', 'rb'))

# 2. Convert to numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# 3. Split into Training (80%) and Testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# 4. Initialize and Train the Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 5. Check Accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100}% of samples were classified correctly!')

# 6. Save the trained "Brain"
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("SUCCESS: model.p created!")
