import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from utils.preprocess import preprocess
import dowhy
from dowhy import CausalModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load preprocessed data
df = preprocess()

# Drop NA and encode categorical data
df.dropna(inplace=True)
# Inspect sample responses to define churn logic
print("Sample responses from survey:")
print(df['response'].dropna().astype(str).sample(20))

# Encode categorical features
le_kasse = LabelEncoder()
df['krankenkasse'] = le_kasse.fit_transform(df['krankenkasse'])
le_question = LabelEncoder()
df['question'] = le_question.fit_transform(df['question'])

# Simulate churn column for demo (real data needs true churn labels)
df['response'] = pd.to_numeric(df['response'], errors='coerce')
df['churn'] = df['response'].apply(lambda x: 1 if x < 1.0 else 0)
print("Churn distribution:")
print(df['churn'].value_counts())
print(df[['response', 'churn']].corr())
# Select features
features = ['krankenkasse', 'question', 'response', 'marktanteil_mitglieder', 'risikofaktor',
            'avg_zusatzbeitrag', 'competitor_avg_zusatzbeitrag']
df = df[features + ['churn']]
df['response'] = pd.to_numeric(df['response'], errors='coerce').fillna(0)

X = df.drop(columns=['churn', 'response'])
y = df['churn']
print(y.value_counts())
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print ("Churn label distribution:")
print(y.value_counts())
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title('Feature Importances')
plt.show()
# After your predictive model code and evaluation, add this causal inference section:

# -----------------------------
# Causal Inference with DoWhy
# -----------------------------

# Prepare data for causal analysis
causal_df = df.copy()

# Define variables for causal model
treatment = "avg_zusatzbeitrag"
outcome = "churn"
covariates = [
    "krankenkasse",  # already encoded
    "question",       # already encoded
    "marktanteil_mitglieder",
    "risikofaktor"
]

# Build the causal model
causal_model = CausalModel(
    data=causal_df,
    treatment=treatment,
    outcome=outcome,
    common_causes=covariates
)

# View model graph (optional if running in notebook)
causal_model.view_model()

# Identify causal effect
identified_estimand = causal_model.identify_effect()

# Estimate the causal effect using linear regression
causal_estimate = causal_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print("\nCausal Inference Results:")
print("Estimated causal effect of avg_zusatzbeitrag on churn:", causal_estimate.value)

# Perform a placebo test to check robustness
refutation = causal_model.refute_estimate(
    identified_estimand,
    causal_estimate,
    method_name="placebo_treatment_refuter"
)
print("\nRefutation result:")
print(refutation)
# Now estimate effect of competitors' contribution on churn
treatment = "competitor_avg_zusatzbeitrag"
covariates = ["krankenkasse", "question", "marktanteil_mitglieder", "risikofaktor", "avg_zusatzbeitrag"]

causal_model_comp = CausalModel(
    data=causal_df,
    treatment=treatment,
    outcome=outcome,
    common_causes=covariates
)

causal_model_comp.view_model()
identified_estimand_comp = causal_model_comp.identify_effect()
causal_estimate_comp = causal_model_comp.estimate_effect(
    identified_estimand_comp,
    method_name="backdoor.linear_regression"
)

print("\nCausal Inference Results (Competitor Contribution):")
print("Estimated causal effect of competitor_avg_zusatzbeitrag on churn:", causal_estimate_comp.value)

refutation_comp = causal_model_comp.refute_estimate(
    identified_estimand_comp,
    causal_estimate_comp,
    method_name="placebo_treatment_refuter"
)
print("\nRefutation result (Competitor):")
print(refutation_comp)