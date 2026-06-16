import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

DATA_PATH = '/Users/azizraihan/Desktop/mom499/neuropathy_clean_10k.csv'
df = pd.read_csv(DATA_PATH)
X = df.drop('target', axis=1)
y = df['target']

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Base estimators (already scaled inside pipeline so we scale X once outside)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lgbm  = LGBMClassifier(max_depth=4, learning_rate=0.05, n_estimators=200,
                        random_state=42, verbose=-1)
svm   = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
cat   = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                            random_seed=42, verbose=0)
xgb   = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=200,
                       random_state=42, eval_metric='logloss', verbosity=0)
lr    = LogisticRegression(max_iter=1000, random_state=42)


def run_cv(name, clf, X_data, y_data, description):
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"  Method: {description}")
    print(f"{'='*62}")

    y_true_cv, y_pred_cv = [], []
    fold_accs = []

    for fold, (tr, te) in enumerate(kf.split(X_data, y_data)):
        clf.fit(X_data[tr], y_data.iloc[tr])
        y_pred = clf.predict(X_data[te])
        acc = accuracy_score(y_data.iloc[te], y_pred)
        fold_accs.append(acc)
        y_true_cv.extend(y_data.iloc[te])
        y_pred_cv.extend(y_pred)
        print(f"  Fold {fold+1}: {acc:.4f}")

    print(f"\n  Mean CV Accuracy : {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")
    print("\n  --- CV Test Classification Report ---")
    print(classification_report(y_true_cv, y_pred_cv,
                                target_names=['Healthy', 'Neuropathy Risk']))

    clf.fit(X_data, y_data)
    print("  --- Train Classification Report ---")
    print(classification_report(y_data, clf.predict(X_data),
                                target_names=['Healthy', 'Neuropathy Risk']))

    return np.mean(fold_accs), np.std(fold_accs), clf


# ── METHOD 1: Soft Voting
# Each model outputs a probability for each class.
# Final prediction = class with highest AVERAGE probability across all models.
# Better than hard voting because confident models have more influence.
soft_voter = VotingClassifier(
    estimators=[('lgbm', lgbm), ('svm', svm), ('cat', cat)],
    voting='soft'
)
acc1, std1, soft_voter = run_cv(
    'Soft Voting  (LightGBM + SVM + CatBoost)',
    soft_voter, X_scaled, y,
    'Average predicted probabilities → pick class with highest mean prob'
)


# ── METHOD 2: Soft Voting with 4 models (adding XGBoost)
soft_voter_4 = VotingClassifier(
    estimators=[('lgbm', lgbm), ('svm', svm), ('cat', cat), ('xgb', xgb)],
    voting='soft'
)
acc2, std2, soft_voter_4 = run_cv(
    'Soft Voting  (LightGBM + SVM + CatBoost + XGBoost)',
    soft_voter_4, X_scaled, y,
    'Average predicted probabilities across 4 diverse models'
)


# ── METHOD 3: Stacking with Logistic Regression meta-learner
# Base models make predictions on held-out folds (cross_val_predict).
# A meta-learner (Logistic Regression) is trained on those out-of-fold predictions.
# More powerful than voting — the meta-learner learns HOW to combine the base models.
stacker = StackingClassifier(
    estimators=[('lgbm', lgbm), ('svm', svm), ('cat', cat)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    stack_method='predict_proba',
    passthrough=False
)
acc3, std3, stacker = run_cv(
    'Stacking  (LightGBM + SVM + CatBoost → Logistic Regression)',
    stacker, X_scaled, y,
    'Base models → out-of-fold proba → Logistic Regression meta-learner'
)


# ── Final Summary
print(f"\n{'='*62}")
print("  ENSEMBLE SUMMARY vs BEST SINGLE MODEL")
print(f"{'='*62}")
print(f"  {'Model':<45} {'CV Acc':>8}")
print(f"  {'-'*55}")
print(f"  {'LightGBM (best single)':<45} {'0.8544':>8}")
print(f"  {'Soft Voting 3 (LGBM+SVM+Cat)':<45} {acc1:>8.4f}")
print(f"  {'Soft Voting 4 (LGBM+SVM+Cat+XGB)':<45} {acc2:>8.4f}")
print(f"  {'Stacking (LGBM+SVM+Cat → LR)':<45} {acc3:>8.4f}")

# Save the best ensemble
best_acc = max(acc1, acc2, acc3)
if best_acc == acc1:
    best_model, best_name = soft_voter, 'Soft Voting 3'
elif best_acc == acc2:
    best_model, best_name = soft_voter_4, 'Soft Voting 4'
else:
    best_model, best_name = stacker, 'Stacking'

# Save scaler + best model together as a pipeline-like dict
joblib.dump({'scaler': scaler, 'model': best_model, 'name': best_name},
            '/Users/azizraihan/Desktop/mom499/neuropathy_final_model.pkl')
print(f"\n  Best ensemble: {best_name} (CV acc: {best_acc:.4f})")
print("  Saved to neuropathy_final_model.pkl")
