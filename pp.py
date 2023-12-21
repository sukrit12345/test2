import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


data = [
    [385, 'Male', 35, 20000, 'No'],
    [681, 'Male', 40, 43500, 'No'],
    
]

columns = ['UserID', 'Gender', 'Age', 'AnnualSalary', 'Purchased']

df = pd.DataFrame(data, columns=columns)


df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Purchased'] = df['Purchased'].map({'No': 0, 'Yes': 1})


X = df[['Gender', 'Age', 'AnnualSalary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')


plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True, precision=2)
plt.show()
