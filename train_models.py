import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_PATH = '/Users/azizraihan/Desktop/mom499/neuropathy_clean_10k.csv'

df = pd.read_csv(DATA_PATH)
X = df.drop('target', axis=1)
y = df['target']

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=200,
                              random_state=42, eval_metric='logloss', verbosity=0))
    ]),
    'LightGBM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LGBMClassifier(max_depth=4, learning_rate=0.05, n_estimators=200,
                               random_state=42, verbose=-1))
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

    # Train report — refit on full data
    pipeline.fit(X, y)
    y_pred_train = pipeline.predict(X)
    print("  --- Training Set Classification Report ---")
    print(classification_report(y, y_pred_train, target_names=['Healthy', 'Neuropathy Risk']))

    results[name] = {
        'mean_cv_acc': np.mean(fold_accs),
        'std_cv_acc': np.std(fold_accs),
        'pipeline': pipeline
    }

# Summary table
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  {'Model':<25} {'CV Acc':>10} {'Std':>8}")
print(f"  {'-'*45}")
for name, r in sorted(results.items(), key=lambda x: -x[1]['mean_cv_acc']):
    print(f"  {name:<25} {r['mean_cv_acc']:>10.4f} {r['std_cv_acc']:>8.4f}")

# Save all pipelines for later use
joblib.dump(results, '/Users/azizraihan/Desktop/mom499/all_model_results.pkl')
print("\nAll pipelines saved to all_model_results.pkl")
