import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Create a sample dataset


df = pd.read_csv('car_insurance.csv')


# Convert categorical variables to numerical variables
df = pd.get_dummies(df, columns=['driving_record'])
print(df)
# Split the data into training and test sets
X = df.drop('covered', axis=1)
y = df['covered']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Load the trained model from a file
with open('auto_insurance_model.pkl', 'wb') as file:
    pickle.dump(clf, file)