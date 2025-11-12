import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Load dataset
data = pd.read_csv('data/heart.csv')

# Features and target
X = data.drop(columns='target')
y = data['target']

# Train / test split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,stratify=y, random_state=2
)

# Pipelines (scaling for models that need it)
pipe_log = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=2))])
pipe_svc = Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=2))])
dt = DecisionTreeClassifier(random_state=2)
rf = RandomForestClassifier(n_estimators=200, random_state=2)

voting = VotingClassifier(estimators=[('log', pipe_log), ('svc', pipe_svc), ('rf', rf)], voting='soft')

models = {'logreg': pipe_log, 'dt': dt, 'rf': rf, 'svc': pipe_svc, 'voting': voting}

# Train and save models
for name, model in models.items():
    print(f'Training {name} ...')
    model.fit(X_train, y_train)
    joblib.dump(model, f'results/{name}.joblib')
    print(f'Saved results/{name}.joblib')

# Save test split for evaluation
X_test.to_csv('results/X_test.csv', index=False)
y_test.to_csv('results/y_test.csv', index=False)

print('Training complete. Models and test split saved in results/')
