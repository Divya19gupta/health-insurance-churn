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

df = preprocess()

df.dropna(inplace=True)
print("Sample responses from survey:")
print(df['response'].dropna().astype(str).sample(20))

le_kasse = LabelEncoder()
df['krankenkasse'] = le_kasse.fit_transform(df['krankenkasse'])
le_question = LabelEncoder()
df['question'] = le_question.fit_transform(df['question'])

df['response'] = pd.to_numeric(df['response'], errors='coerce')
df['churn'] = df['response'].apply(lambda x: 1 if x < 1.0 else 0)
print("Churn distribution:")
print(df['churn'].value_counts())
print(df[['response', 'churn']].corr())

features = ['krankenkasse', 'question', 'response', 'marktanteil_mitglieder', 'risikofaktor',
            'avg_zusatzbeitrag', 'competitor_avg_zusatzbeitrag']
df = df[features + ['churn']]
df['response'] = pd.to_numeric(df['response'], errors='coerce').fillna(0)

X = df.drop(columns=['churn', 'response'])
y = df['churn']
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print ("Churn label distribution:")
print(y.value_counts())

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title('Feature Importances')
plt.show()

causal_df = df.copy()

treatment = "avg_zusatzbeitrag"
outcome = "churn"
covariates = [
    "krankenkasse", 
    "question",      
    "marktanteil_mitglieder",
    "risikofaktor"
]

causal_model = CausalModel(
    data=causal_df,
    treatment=treatment,
    outcome=outcome,
    common_causes=covariates
)

causal_model.view_model()

identified_estimand = causal_model.identify_effect()

causal_estimate = causal_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print("\nCausal Inference Results:")
print("Estimated causal effect on churn:", causal_estimate.value)

refutation = causal_model.refute_estimate(
    identified_estimand,
    causal_estimate,
    method_name="placebo_treatment_refuter"
)
print("\nResult:")
print(refutation)
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

print("\nCausal Inference Results (Competitor):")
print("Estimated causal effect on churn:", causal_estimate_comp.value)

refutation_comp = causal_model_comp.refute_estimate(
    identified_estimand_comp,
    causal_estimate_comp,
    method_name="placebo_treatment_refuter"
)
print("\nResult (Competitor):")
print(refutation_comp)
