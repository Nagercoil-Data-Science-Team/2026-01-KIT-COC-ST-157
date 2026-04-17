# =====================================================
# IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, confusion_matrix,
    classification_report, roc_curve, auc,
    precision_recall_curve, precision_score, recall_score,
    f1_score, average_precision_score, roc_auc_score
)
from sklearn.calibration import calibration_curve

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

# =====================================================
# STEP 1: DATA LOADING
# =====================================================
df = pd.read_csv("adaptive_blended_teaching_dataset.csv")
print("Initial Shape:", df.shape)

# =====================================================
# STEP 2: DATA PREPROCESSING
# =====================================================
df = df.drop_duplicates()

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# =====================================================
# STEP 3: FEATURE EXTRACTION
# =====================================================
df["stat_mean"] = df[num_cols].mean(axis=1)
df["stat_std"] = df[num_cols].std(axis=1)
df["stat_max"] = df[num_cols].max(axis=1)
df["stat_min"] = df[num_cols].min(axis=1)
df["stat_variance"] = df[num_cols].var(axis=1)

df["total_activity"] = df[num_cols].sum(axis=1)
df["avg_activity"] = df[num_cols].mean(axis=1)
df["engagement_score"] = (df["total_activity"] + df["avg_activity"]) / 2

eps = 1e-6
df["activity_ratio"] = df["stat_mean"] / (df["stat_max"] + eps)
df["consistency_ratio"] = df["stat_min"] / (df["stat_mean"] + eps)
df["variation_ratio"] = df["stat_std"] / (df["stat_mean"] + eps)

# =====================================================
# Engagement Level Distribution (BAR CHART)
# =====================================================
plt.figure(figsize=(8,6))
df["Predicted_Performance"].value_counts().plot(
    kind='bar', color='#5B8FF9'
)
plt.xlabel("Engagement Level", fontweight='bold')
plt.ylabel("Number of Students", fontweight='bold')
plt.title("Engagement Level Distribution", fontweight='bold')
plt.savefig("Engagement Level Distribution.png")

plt.show()

# =====================================================
# STEP 4: XGBOOST CLASSIFICATION
# =====================================================
id_cols = [c for c in df.columns if "id" in c.lower() or "student" in c.lower()]
df = df.drop(columns=id_cols)

target_col = "Predicted_Performance"
X = df.drop(columns=[target_col])
y = df[target_col]

for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
classes = target_encoder.classes_
n_classes = len(classes)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2,
    random_state=42, stratify=y_res
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='multi:softprob',
    num_class=n_classes,
    eval_metric=['mlogloss', 'merror'],
    random_state=42
)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# =====================================================
# XGBoost Feature Importance (BAR CHART)
# =====================================================
importances = model.feature_importances_
feature_names = X.columns

feat_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(9,6))
plt.barh(
    feat_imp_df["Feature"][:15][::-1],
    feat_imp_df["Importance"][:15][::-1],
    color='#F6BD16'
)
plt.xlabel("Importance Score (Gain)", fontweight='bold')
plt.ylabel("Feature", fontweight='bold')
plt.title("Top 15 XGBoost Feature Importances", fontweight='bold')
plt.savefig("Top 15 XGBoost Feature Importances.png")
plt.show()

# =====================================================
# STEP 5: MODEL EVALUATION
# =====================================================
accuracy_val = accuracy_score(y_test, y_pred)
loss_val = log_loss(y_test, y_prob)

print("\nModel Accuracy:", accuracy_val)
print("Model Log Loss:", loss_val)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=classes))

# =====================================================
# ROC Curve with AUC (Per Class)
# =====================================================
y_test_bin = label_binarize(y_test, classes=range(n_classes))

plt.figure(figsize=(8,6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{classes[i]} (AUC={roc_auc:.3f})")

plt.plot([0,1], [0,1], "--", color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve with AUC per Class")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# =====================================================
# Precision–Recall Curve with AP (Per Class)
# =====================================================
plt.figure(figsize=(8,6))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(
        y_test_bin[:, i], y_prob[:, i]
    )
    ap = average_precision_score(
        y_test_bin[:, i], y_prob[:, i]
    )
    plt.plot(recall, precision, label=f"{classes[i]} (AP={ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve with AP per Class")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
