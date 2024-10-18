import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import StackingClassifier
import joblib


df = pd.read_csv("Project_1_Data.csv")

print(df.info())

# Step 2
x = df['X'].values
y = df['Y'].values
z = df['Z'].values
Step = df['Step'].values

plt.plot(Step, x, label= 'X')
plt.plot(Step, y, label= 'Y')
plt.plot(Step, z, label= 'Z')


plt.xlabel('Step')
plt.ylabel('Values')
plt.title('Line Plot')
plt.legend()
plt.show()

# Step 3 Correlation
corr_matrix = df.corr()
sns.heatmap(np.abs(corr_matrix))

# Step 4 Classification
X = np.column_stack((x, y, z))
Y = Step

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

param_grid_log = {'C': [0.01, 0.1, 1, 10, 100]}
param_grid_for = {'n_estimators': [3, 10, 30], 'max_depth': [2, 4, 6, 8]}
param_grid_KN = {'n_neighbors': [3, 5, 7, 9]}

log_model = LogisticRegression()
for_model = RandomForestClassifier()
KN_model = KNeighborsClassifier()
# CV_model = RandomizedSearchCV()

grid_log = GridSearchCV(log_model, param_grid_log, cv=5)
grid_log.fit(X_train, Y_train)
Y_pred_log = grid_log.predict(X_test)

grid_for = GridSearchCV(for_model, param_grid_for, cv=5)
grid_for.fit(X_train, Y_train)
Y_pred_for = grid_for.predict(X_test)

grid_KN = GridSearchCV(KN_model, param_grid_KN, cv=5)
grid_KN.fit(X_train, Y_train)
Y_pred_KN = grid_KN.predict(X_test)

accuracy_log = accuracy_score(Y_test, Y_pred_log)
accuracy_for = accuracy_score(Y_test, Y_pred_for)
accuracy_KN = accuracy_score(Y_test, Y_pred_KN)

print(f"Logistic Regression Accuracy: {accuracy_log}")
print(f"Random Forest Accuracy: {accuracy_for}")
print(f"KNeighbors Accuracy: {accuracy_KN}")

# randomized ssearch cv

"""GridSearchCV"""
param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    }

CV_model = RandomForestClassifier()
grid_search = RandomizedSearchCV(CV_model, param_distributions=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
grid_search.fit(X_train, Y_train)
Y_pred_CV = grid_search.predict(X_test)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_modelCV = grid_search.best_estimator_


accuracy_CV = accuracy_score(Y_test, Y_pred_CV)
print(f"CV Accuracy: {accuracy_CV}")

# Step 5
f1_log = f1_score(Y_test, Y_pred_log, average='weighted')
f1_for = f1_score(Y_test, Y_pred_for, average='weighted')
f1_KN = f1_score(Y_test, Y_pred_KN, average='weighted')
f1_CV = f1_score(Y_test, Y_pred_CV, average='weighted')

print(f"Logistic Regression F1 Score: {f1_log}")
print(f"Random Forest F1 Score: {f1_for}")
print(f"KNeighbors F1 Score: {f1_KN}")
print(f"Search CV Regression F1 Score: {f1_CV}")

# confusion matrix
con_matrix_log = confusion_matrix(Y_test, Y_pred_log)
con_matrix_for = confusion_matrix(Y_test, Y_pred_for)
con_matrix_KN = confusion_matrix(Y_test, Y_pred_KN)
con_matrix_CV = confusion_matrix(Y_test, Y_pred_CV)

def plot_confusion_matrix(matrix,title):
    plt.figure(figsize=(6,4))
    sns.heatmap(matrix, annot=True, fmt="d", cbar=False)
    plt.title(title)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.show()
    

plot_confusion_matrix(con_matrix_log, 'Logistic Regression Confusion Matrix')
plot_confusion_matrix(con_matrix_for, 'Random Forest Confusion Matrix')
plot_confusion_matrix(con_matrix_KN, 'KNeighbors Confusion Matrix')
plot_confusion_matrix(con_matrix_CV, 'Search CV Regression Confusion Matrix')

# Step 6 Stacked Model Performance
tb_RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
tb_kmn_model = KNeighborsClassifier(n_neighbors=5)

tb_RF_model.fit(X_train, Y_train)
tb_kmn_model.fit(X_train, Y_train)

fin_est = LogisticRegression()
stack = StackingClassifier(estimators=[('rf', tb_RF_model), ('kmn', tb_kmn_model)], final_estimator=fin_est)

stack.fit(X_train,Y_train)

Y_pred_stack = stack.predict(X_test)
accuracy_stack = accuracy_score(Y_test, Y_pred_stack)
f1_stack = f1_score(Y_test, Y_pred_stack, average='weighted')

con_matrix_stack = confusion_matrix(Y_test, Y_pred_stack)

print(f"Stacking Classifier Accuracy: {accuracy_stack}")
print(f"Stacking F1 Score: {f1_stack}")

sns.heatmap(con_matrix_stack, annot= True)
plt.title('Stacking Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show

modelfin = stack
joblib.dump(modelfin, 'modelfin.joblib')

modelload = joblib.load('modelfin.joblib')

X_new = np.array([[9.375,3.0625,1.5], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]])

predictions = modelload.predict(X_new)
for input_values, prediction in zip(X_new, predictions):
    print(f"Input: {input_values}, Predicted Maintenance Step: {prediction}")