import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib  # For saving the model and scaler

# Load dataset
dataset = pd.read_excel("C:/Users/bngog/Desktop/intern/creditscore/a_Dataset_CreditScoring.xlsx")

# Drop the 'ID' column
dataset = dataset.drop('ID', axis=1)

# Check and fill missing values with mean
dataset = dataset.fillna(dataset.mean())

# Display missing values (should all be 0 now)
print("\n=== Missing Values After Filling ===")
print(dataset.isna().sum())

# Display target value distribution
print("\n=== Target Value Counts ===")
print(dataset['TARGET'].value_counts())

# Display feature means grouped by TARGET class
print("\n=== Feature Means Grouped by TARGET ===")
print(dataset.groupby('TARGET').mean())

# Split features and target
y = dataset.iloc[:, 0].values  # TARGET column
X = dataset.iloc[:, 1:29].values  # All other columns

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the scaler for future use
joblib.dump(sc, 'C:/Users/bngog/Desktop/intern/creditscore/f2_Normalisation_CreditScoring.xlsx')

# Train Logistic Regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump(classifier, 'C:/Users/bngog/Desktop/intern/creditscore/f1_Classifier_CreditScoring.xlsx')

# Predictions
y_pred = classifier.predict(X_test)

# Evaluation metrics
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Accuracy Score ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Predict probabilities
predictions = classifier.predict_proba(X_test)

# Create result DataFrame
df_prediction_prob = pd.DataFrame(predictions, columns=['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns=['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test, columns=['Actual Outcome'])

dfx = pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)

# Print full output
print("\n=== Model Predictions with Probabilities ===")
print(dfx)

# Optional: Preview first few rows
print("\n=== Preview of Predictions (first 5 rows) ===")
print(dfx.head())

# Function to convert prob_1 (bad loan) into a FICO-style score (lower prob_1 = higher score)
def prob_to_fico(prob, min_score=300, max_score=850):
    return (1 - prob) * (max_score - min_score) + min_score

# Apply FICO scoring
dfx['FICO_Score'] = dfx['prob_1'].apply(prob_to_fico).round().astype(int)

# Classify into risk bands
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

# Print output
print("\n=== FICO-Style Scoring Output ===")
print(dfx[['Actual Outcome', 'prob_1', 'FICO_Score', 'Risk_Band']].head())
