import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# === STEP 1: LOAD DATA ===
file_path = "C:/Users/bngog/Desktop/intern/creditscore/a_Dataset_CreditScoring.xlsx"
dataset = pd.read_excel(file_path)

# Drop ID column
dataset = dataset.drop('ID', axis=1)

# Fill missing values
dataset = dataset.fillna(dataset.mean())

# Split features and target
y = dataset.iloc[:, 0].values  # TARGET
X = dataset.iloc[:, 1:].values  # Features

# === STEP 2: SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# === STEP 3: SCALE DATA ===
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the scaler
joblib.dump(sc, 'C:/Users/bngog/Desktop/intern/creditscore/f2_Normalisation_CreditScoring')

# === STEP 4: APPLY SMOTE ===
sm = SMOTE(random_state=0)
X_train, y_train = sm.fit_resample(X_train, y_train)

# === STEP 5: GRIDSEARCHCV for BEST XGBOOST CLASSIFIER ===
xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2]
}

grid = GridSearchCV(estimator=xgb_base,
                    param_grid=param_grid,
                    cv=3,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)

grid.fit(X_train, y_train)
classifier = grid.best_estimator_

print("\n✅ Best Parameters Found by GridSearchCV:")
print(grid.best_params_)

# Save the best model
joblib.dump(classifier, 'C:/Users/bngog/Desktop/intern/creditscore/f1_Classifier_CreditScoring')

# === STEP 6: EVALUATE MODEL ===
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Accuracy Score ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# === STEP 7: FICO SCORING ===
df_prediction_prob = pd.DataFrame(y_prob, columns=['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(y_pred, columns=['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test, columns=['Actual Outcome'])

dfx = pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)

# FICO score mapping
def prob_to_fico(prob, min_score=300, max_score=850):
    return (1 - prob) * (max_score - min_score) + min_score

dfx['FICO_Score'] = dfx['prob_1'].apply(prob_to_fico).round().astype(int)

# Risk band classification
def classify_fico(score):
    if score >= 800:
        return 'Excellent'
    elif score >= 740:
        return 'Very Good'
    elif score >= 670:
        return 'Good'
    elif score >= 580:
        return 'Fair'
    else:
        return 'Poor'

dfx['Risk_Band'] = dfx['FICO_Score'].apply(classify_fico)

# Show preview
print("\n=== FICO-Style Scoring Output (Preview) ===")
print(dfx[['Actual Outcome', 'prob_1', 'FICO_Score', 'Risk_Band']].head())
