import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Load data and test split
data = pd.read_csv('data/heart.csv')
X_test = pd.read_csv('results/X_test.csv')
y_test = pd.read_csv('results/y_test.csv').values.ravel()

# EDA outputs
data.describe().to_csv('results/describe.csv')
data['target'].value_counts().to_csv('results/target_counts.csv')

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('figures/corr_heatmap.png', dpi=200)
plt.close()

# Load models from results/
models = {}
for name in ['logreg','dt','rf','svc','voting']:
    path = f'results/{name}.joblib'
    try:
        models[name] = joblib.load(path)
        print(f'Loaded {path}')
    except Exception as e:
        print(f'Could not load {path}: {e}')

summaries = []
for name, model in models.items():
    print(f'Evaluating {name} ...')
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    # Save confusion matrix figure
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.savefig(f'figures/{name}_confusion.png', dpi=200)
    plt.close()
    # Classification report CSV
    cr = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(cr).transpose().to_csv(f'results/{name}_classification_report.csv')
    # ROC if available
    try:
        probs = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
        plt.plot([0,1],[0,1],'--')
        plt.title(f'{name} ROC')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(f'figures/{name}_roc.png', dpi=200)
        plt.close()
    except Exception:
        print(f'No predict_proba for {name}, skipping ROC.')
    summaries.append({'model': name, 'accuracy': acc})

pd.DataFrame(summaries).to_csv('results/model_summaries.csv', index=False)
print('Evaluation complete. Figures in figures/, CSVs in results/')
