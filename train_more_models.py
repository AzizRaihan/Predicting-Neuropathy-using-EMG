import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier

DATA_PATH = '/Users/azizraihan/Desktop/mom499/neuropathy_clean_10k.csv'

df = pd.read_csv(DATA_PATH)
X = df.drop('target', axis=1)
y = df['target']

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=11))
    ]),
    'Extra Trees': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                           learning_rate=0.05, random_state=42))
    ]),
    'CatBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                   random_seed=42, verbose=0))
    ]),
}

results = {}

for name, pipeline in models.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    y_true_cv, y_pred_cv = [], []
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accs.append(acc)
        y_true_cv.extend(y_test)
        y_pred_cv.extend(y_pred)

        print(f"  Fold {fold+1} accuracy: {acc:.4f}")

    print(f"\n  Mean CV Accuracy: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    print("\n  --- CV Test Set Classification Report ---")
    print(classification_report(y_true_cv, y_pred_cv, target_names=['Healthy', 'Neuropathy Risk']))

    pipeline.fit(X, y)
    y_pred_train = pipeline.predict(X)
    print("  --- Training Set Classification Report ---")
    print(classification_report(y, y_pred_train, target_names=['Healthy', 'Neuropathy Risk']))

    results[name] = {
        'mean_cv_acc': np.mean(fold_accs),
        'std_cv_acc': np.std(fold_accs),
        'pipeline': pipeline
    }

# Combined summary with previous results
print(f"\n{'='*60}")
print("  NEW MODELS SUMMARY")
print(f"{'='*60}")
print(f"  {'Model':<25} {'CV Acc':>10} {'Std':>8}")
print(f"  {'-'*45}")
for name, r in sorted(results.items(), key=lambda x: -x[1]['mean_cv_acc']):
    print(f"  {name:<25} {r['mean_cv_acc']:>10.4f} {r['std_cv_acc']:>8.4f}")

print(f"\n{'='*60}")
print("  PREVIOUS MODELS (for reference)")
print(f"{'='*60}")
prev = {
    'LightGBM':             0.8544,
    'XGBoost':              0.8518,
    'Logistic Regression':  0.8508,
    'Random Forest':        0.8485,
}
for name, acc in sorted(prev.items(), key=lambda x: -x[1]):
    print(f"  {name:<25} {acc:>10.4f}")

joblib.dump(results, '/Users/azizraihan/Desktop/mom499/more_model_results.pkl')
print("\nAll new pipelines saved to more_model_results.pkl")
