import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
file_path = "C:/Users/bngog/Desktop/intern/creditscore/a_Dataset_CreditScoring.xlsx"
dataset = pd.read_excel(file_path)

# Drop ID column
dataset = dataset.drop('ID', axis=1)

# Fill missing values with column means
dataset = dataset.fillna(dataset.mean())

# Split features and target
y = dataset.iloc[:, 0].values  # TARGET
X = dataset.iloc[:, 1:].values  # All other columns

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the scaler
joblib.dump(sc, 'C:/Users/bngog/Desktop/intern/creditscore/f2_Normalisation_CreditScoring')

# Train Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
classifier.fit(X_train, y_train)

# Save the classifier
joblib.dump(classifier, 'C:/Users/bngog/Desktop/intern/creditscore/f1_Classifier_CreditScoring')

# Predictions
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)

# Evaluation
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Accuracy Score ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Build result DataFrame
df_prediction_prob = pd.DataFrame(y_prob, columns=['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(y_pred, columns=['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test, columns=['Actual Outcome'])

dfx = pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)

# FICO score mapping function
def prob_to_fico(prob, min_score=300, max_score=850):
    return (1 - prob) * (max_score - min_score) + min_score

dfx['FICO_Score'] = dfx['prob_1'].apply(prob_to_fico).round().astype(int)

# Risk Band classification
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

# Output final results to console
print("\n=== FICO-Style Scoring Results ===")
print(dfx[['Actual Outcome', 'prob_1', 'FICO_Score', 'Risk_Band']].head())

# Optional: Show all results
print(dfx[['Actual Outcome', 'prob_1', 'FICO_Score', 'Risk_Band']])
