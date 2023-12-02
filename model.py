import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the data
data = pd.read_csv('iris.csv')

# Split the data into X and y
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open("model.pkl", "wb"))
print("Model Saved!")
