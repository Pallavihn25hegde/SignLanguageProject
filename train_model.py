import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data_dict = pickle.load(open('data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

model = RandomForestClassifier(
    n_estimators=120, max_depth=20, random_state=42, n_jobs=-1
)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

print(f'Test Accuracy: {accuracy * 100:.2f}%\n')
print(classification_report(y_test, y_predict))

with open('model.p', 'wb') as f:
  pickle.dump({'model': model}, f)

print("Saved model as 'model.p'.")