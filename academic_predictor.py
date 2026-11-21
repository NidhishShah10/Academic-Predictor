# ACADEMIC PREDICTOR
# Predicting Student Academic Performance Using Machine Learning
# Oakland University - Machine Learning Project
# Made By: Nidhish Shah and Kyle Do

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42

DATA_PATH = "data/StudentPerformanceFactors.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)

print("=" * 65)
print("                  ACADEMIC PREDICTOR")
print("       Predicting Student Academic Performance Using ML")
print("=" * 65)
print("\n[1] Loading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place StudentPerformanceFactors.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())

if "Exam_Score" not in df.columns:
    raise ValueError("Exam_Score target column is missing.")

print("\nExam score statistics:")
print(df["Exam_Score"].describe())
print("\n[2] Validating and cleaning dataset...")

df.columns = df.columns.str.strip()

if "Exam_Score" not in df.columns:
    raise ValueError("Exam_Score column is missing.")

df["Exam_Score"] = pd.to_numeric(
    df["Exam_Score"],
    errors="coerce"
)

if df["Exam_Score"].isnull().any():
    raise ValueError(
        "Exam_Score contains missing or nonnumeric values. "
        "Investigate these records before training."
    )

duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    print("Removing duplicate records:", duplicate_count)
    df = df.drop_duplicates().reset_index(drop=True)

unusual_scores = df[
    (df["Exam_Score"] < 0) |
    (df["Exam_Score"] > 100)
]

print("\nExam scores outside 0-100:")
print(unusual_scores[["Exam_Score"]])

categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns

for col in categorical_columns:
    df[col] = df[col].apply(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )
numeric_ranges = {
    "Hours_Studied": (0, None),
    "Attendance": (0, 100),
    "Sleep_Hours": (0, 24),
    "Previous_Scores": (0, 100),
    "Tutoring_Sessions": (0, None),
    "Physical_Activity": (0, None)
}

for col, (minimum, maximum) in numeric_ranges.items():

    if col not in df.columns:
        continue

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    invalid = df[col] < minimum

    if maximum is not None:
        invalid = invalid | (df[col] > maximum)

    if invalid.any():
        raise ValueError(
            f"Invalid numeric values detected in {col}: "
            f"{invalid.sum()} records."
        )

print("\nDataset validation completed.")
print("Records remaining:", len(df))
print("\n[3] Performing exploratory data analysis...")

df.describe(include="all").to_csv(
    f"{OUTPUT_DIR}/descriptive_statistics.csv"
)

missing_summary = pd.DataFrame({
    "Feature": df.columns,
    "Missing Values": df.isnull().sum().values,
    "Missing Percentage": (
        df.isnull().mean().values * 100
    )
})

missing_summary.to_csv(
    f"{OUTPUT_DIR}/missing_values.csv",
    index=False
)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Exam_Score",
    bins=25,
    kde=True
)

plt.title("Distribution of Student Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/exam_score_distribution.png",
    dpi=300
)

plt.close()
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(11, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Between Numerical Student Factors")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/correlation_heatmap.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Hours_Studied",
    y="Exam_Score",
    alpha=0.5
)

plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/study_hours_vs_score.png",
    dpi=300
)

plt.close()
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Attendance",
    y="Exam_Score",
    alpha=0.5
)

plt.title("Attendance vs Exam Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Exam Score")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/attendance_vs_score.png",
    dpi=300
)

plt.close()
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Motivation_Level",
    y="Exam_Score",
    order=["Low", "Medium", "High"]
)

plt.title("Exam Scores by Motivation Level")
plt.xlabel("Motivation Level")
plt.ylabel("Exam Score")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/motivation_vs_score.png",
    dpi=300
)

plt.close()
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Family_Income",
    y="Exam_Score",
    order=["Low", "Medium", "High"]
)

plt.title("Exam Scores by Family Income")
plt.xlabel("Family Income")
plt.ylabel("Exam Score")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/income_vs_score.png",
    dpi=300
)

plt.close()

print("Exploratory analysis completed.")
print("EDA charts saved in outputs folder.")
print("\n[4] Separating features and target scores...")

X = df.drop(columns=["Exam_Score"])
scores = df["Exam_Score"]

print("Input features:", X.shape[1])

print("\n[5] Splitting dataset...")

X_train, X_temp, scores_train, scores_temp = train_test_split(
    X,
    scores,
    test_size=0.30,
    random_state=RANDOM_STATE
)

X_val, X_test, scores_val, scores_test = train_test_split(
    X_temp,
    scores_temp,
    test_size=0.50,
    random_state=RANDOM_STATE
)

threshold = scores_train.quantile(0.25)

y_train = (scores_train < threshold).astype(int)
y_val = (scores_val < threshold).astype(int)
y_test = (scores_test < threshold).astype(int)

print("\nTraining threshold:", threshold)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

split_info = pd.DataFrame({
    "Dataset": ["Training", "Validation", "Testing"],
    "Records": [
        len(X_train),
        len(X_val),
        len(X_test)
    ],
    "At Risk": [
        int(y_train.sum()),
        int(y_val.sum()),
        int(y_test.sum())
    ],
    "Satisfactory": [
        int((y_train == 0).sum()),
        int((y_val == 0).sum()),
        int((y_test == 0).sum())
    ]
})

split_info.to_csv(
    f"{OUTPUT_DIR}/dataset_split.csv",
    index=False
)

print("\n[6] Identifying feature types...")

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

ordinal_categories = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Parental_Education_Level": [
        "High School",
        "College",
        "Postgraduate"
    ],
    "Distance_from_Home": [
        "Near",
        "Moderate",
        "Far"
    ]
}

ordinal_features = [
    col for col in ordinal_categories
    if col in X_train.columns
]

nominal_features = [
    col for col in X_train.select_dtypes(
        include=["object", "str", "category"]
    ).columns
    if col not in ordinal_features
]

print("\nNumerical features:", numeric_features)
print("\nOrdinal features:", ordinal_features)
print("\nNominal features:", nominal_features)

for col in ordinal_features:

    actual_values = set(
        X_train[col].dropna().unique()
    )

    expected_values = set(
        ordinal_categories[col]
    )

    unexpected = actual_values - expected_values

    if unexpected:
        raise ValueError(
            f"Unexpected categories in {col}: {unexpected}"
        )
assigned_features = set(
    numeric_features + ordinal_features + nominal_features
)

unassigned = set(X_train.columns) - assigned_features

if unassigned:
    raise ValueError(
        f"Unassigned feature columns: {unassigned}"
    )

print("\n[7] Creating preprocessing pipeline...")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

ordinal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[
            ordinal_categories[col]
            for col in ordinal_features
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

nominal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_features),
        ("nominal", nominal_transformer, nominal_features)
    ]
)

print("Preprocessing pipeline created successfully.")

print("\n[8] Creating machine learning models...")

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if positive_count == 0 or negative_count == 0:
    raise ValueError(
        "Training data must contain both target classes."
    )

positive_weight = negative_count / positive_count

print("Positive class weight:", positive_weight)

models = {

    "Baseline": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "Logistic Regression Balanced": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "Decision Tree Balanced": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Random Forest Balanced": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost Balanced": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=positive_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

print("Total model experiments:", len(models))

def evaluate_model(model, X_data, y_data):
    """
    Evaluate a fitted classification pipeline.

    At Risk (1) is the positive class.
    """

    predictions = model.predict(X_data)

    probabilities = model.predict_proba(
        X_data
    )[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_data,
            predictions
        ),
        "Precision": precision_score(
            y_data,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_data,
            predictions,
            zero_division=0
        ),
        "F1-Score": f1_score(
            y_data,
            predictions,
            zero_division=0
        ),
        "ROC-AUC": (
            roc_auc_score(y_data, probabilities)
            if len(np.unique(y_data)) == 2
            else np.nan
        )
    }

    return metrics, predictions, probabilities
print("\n[9] Training and validating models...")

results = []
trained_models = {}

for name, classifier in models.items():

    print(f"\nTraining: {name}")

    model_pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", clone(classifier))
    ])

    model_pipeline.fit(X_train, y_train)
    metrics, predictions, probabilities = evaluate_model(
        model_pipeline,
        X_val,
        y_val
    )

    results.append({
        "Model": name,
        **metrics
    })

    trained_models[name] = model_pipeline

    print(
        f"Accuracy: {metrics['Accuracy']:.4f} | "
        f"Precision: {metrics['Precision']:.4f} | "
        f"Recall: {metrics['Recall']:.4f} | "
        f"F1: {metrics['F1-Score']:.4f} | "
        f"ROC-AUC: {metrics['ROC-AUC']:.4f}"
    )

print("\n[10] Comparing validation performance...")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["F1-Score", "Recall", "ROC-AUC"],
    ascending=False
).reset_index(drop=True)

print("\nVALIDATION RESULTS")
print(results_df.to_string(index=False))

results_df.to_csv(
    f"{OUTPUT_DIR}/model_comparison_validation.csv",
    index=False
)

candidate_results = results_df[
    results_df["Model"] != "Baseline"
].reset_index(drop=True)

best_model_name = candidate_results.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nSELECTED MODEL:", best_model_name)
print("\nBest validation metrics:")
print(candidate_results.iloc[0].to_string())
print("\n[11] Generating model comparison graph...")

results_melted = results_df.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(16, 8))

sns.barplot(
    data=results_melted,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title(
    "Academic Predictor - Validation Model Comparison"
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Score")

plt.xticks(rotation=40, ha="right")
plt.ylim(0, 1.05)

plt.legend(title="Metric")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n[12] Performing final test evaluation...")

test_results = []

for name, model in trained_models.items():

    metrics, _, _ = evaluate_model(
        model,
        X_test,
        y_test
    )

    test_results.append({
        "Model": name,
        **metrics
    })

test_results_df = pd.DataFrame(test_results)
test_results_df["Model"] = pd.Categorical(
    test_results_df["Model"],
    categories=results_df["Model"].tolist(),
    ordered=True
)

test_results_df = test_results_df.sort_values(
    "Model"
).reset_index(drop=True)

print("\nFINAL TEST MODEL COMPARISON")
print(test_results_df.to_string(index=False))

test_results_df.to_csv(
    f"{OUTPUT_DIR}/model_comparison_test.csv",
    index=False
)

test_metrics, test_predictions, test_probabilities = (
    evaluate_model(
        best_model,
        X_test,
        y_test
    )
)

test_accuracy = test_metrics["Accuracy"]
test_precision = test_metrics["Precision"]
test_recall = test_metrics["Recall"]
test_f1 = test_metrics["F1-Score"]
test_auc = test_metrics["ROC-AUC"]

print("\nSELECTED MODEL FINAL TEST RESULTS")
print("Model:", best_model_name)

for metric_name, value in test_metrics.items():
    print(f"{metric_name}: {value:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Satisfactory", "At Risk"],
        zero_division=0
    )
)

final_test_df = pd.DataFrame([{
    "Best Model": best_model_name,
    **test_metrics
}])

final_test_df.to_csv(
    f"{OUTPUT_DIR}/final_test_results.csv",
    index=False
)


print("\n[13] Creating confusion matrix...")

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Satisfactory",
        "Actual At Risk"
    ],
    columns=[
        "Predicted Satisfactory",
        "Predicted At Risk"
    ]
)

cm_df.to_csv(
    f"{OUTPUT_DIR}/test_confusion_matrix.csv"
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Satisfactory", "At Risk"],
    yticklabels=["Satisfactory", "At Risk"]
)

plt.title(f"Test Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Classification")
plt.ylabel("Actual Classification")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/test_confusion_matrix.png",
    dpi=300
)

plt.close()

print("\n[14] Creating ROC curve...")

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    test_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"{best_model_name} (AUC = {test_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Academic Predictor")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/roc_curve.png",
    dpi=300
)

plt.close()

print("\n[15] Calculating permutation feature importance...")

importance = permutation_importance(
    best_model,
    X_val,
    y_val,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring="f1",
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_val.columns,
    "Importance": importance.importances_mean,
    "Standard Deviation": importance.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nFEATURE IMPORTANCE")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    f"{OUTPUT_DIR}/feature_importance.csv",
    index=False
)

top_features = importance_df.head(10).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"],
    xerr=top_features["Standard Deviation"]
)

plt.title(
    f"Top 10 Important Features - {best_model_name}"
)

plt.xlabel("Mean Decrease in Validation F1-Score")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/feature_importance.png",
    dpi=300
)

plt.close()

print("\n[16] Performing subgroup analysis...")

subgroup_results = []

for group_column in ["Gender", "Family_Income"]:

    if group_column not in X_test.columns:
        continue

    group_values = X_test[group_column].fillna(
        "Missing"
    )

    for group_name in sorted(group_values.unique()):

        mask = (group_values == group_name).to_numpy()

        actual = y_test.to_numpy()[mask]
        predicted = test_predictions[mask]

        sample_size = len(actual)
        at_risk_count = int((actual == 1).sum())
        satisfactory_count = int((actual == 0).sum())

        if sample_size == 0:
            continue
        recall = (
            recall_score(
                actual,
                predicted,
                zero_division=0
            )
            if at_risk_count > 0
            else np.nan
        )

        group_f1 = (
            f1_score(
                actual,
                predicted,
                zero_division=0
            )
            if at_risk_count > 0
            else np.nan
        )

        subgroup_results.append({
            "Feature": group_column,
            "Group": group_name,
            "Sample Size": sample_size,
            "Actual At Risk": at_risk_count,
            "Actual Satisfactory": satisfactory_count,
            "Accuracy": accuracy_score(
                actual,
                predicted
            ),
            "Precision": precision_score(
                actual,
                predicted,
                zero_division=0
            ),
            "Recall": recall,
            "F1-Score": group_f1
        })

subgroup_df = pd.DataFrame(subgroup_results)

print("\nSUBGROUP PERFORMANCE")
print(subgroup_df.to_string(index=False))

subgroup_df.to_csv(
    f"{OUTPUT_DIR}/subgroup_performance.csv",
    index=False
)

print("\n[17] Saving selected model...")

joblib.dump(
    best_model,
    f"{OUTPUT_DIR}/academic_predictor_model.pkl"
)

joblib.dump(
    threshold,
    f"{OUTPUT_DIR}/exam_score_threshold.pkl"
)

joblib.dump(
    list(X.columns),
    f"{OUTPUT_DIR}/feature_columns.pkl"
)

experiment_metadata = {
    "Project": "Academic Predictor",
    "Selected Model": best_model_name,
    "Random State": RANDOM_STATE,
    "Training Threshold": float(threshold),
    "Training Records": len(X_train),
    "Validation Records": len(X_val),
    "Testing Records": len(X_test),
    "Date": datetime.now().isoformat()
}

pd.DataFrame(
    [experiment_metadata]
).to_csv(
    f"{OUTPUT_DIR}/experiment_metadata.csv",
    index=False
)

print("Model saved successfully.")
print("Threshold saved successfully.")
print("Feature columns saved successfully.")
print("\n[18] Testing Academic Predictor prototype...")


def predict_student(student_data):
    """
    Predict a student's academic outcome.

    Parameters
    ----------
    student_data : dict
        Student attributes excluding Exam_Score.

    Returns
    -------
    dict
        Predicted classification and At Risk probability.
    """

    saved_model = joblib.load(
        f"{OUTPUT_DIR}/academic_predictor_model.pkl"
    )

    expected_columns = joblib.load(
        f"{OUTPUT_DIR}/feature_columns.pkl"
    )

    missing_fields = [
        col for col in expected_columns
        if col not in student_data
    ]

    if missing_fields:
        raise ValueError(
            f"Missing student attributes: {missing_fields}"
        )

    student_df = pd.DataFrame([student_data])

    student_df = student_df[expected_columns]

    prediction = int(
        saved_model.predict(student_df)[0]
    )

    probability = float(
        saved_model.predict_proba(student_df)[0][1]
    )

    result = (
        "At Risk"
        if prediction == 1
        else "Satisfactory"
    )

    return {
        "Prediction": result,
        "At Risk Probability": probability
    }

sample_student = X_test.iloc[0].to_dict()

sample_result = predict_student(sample_student)

print("\nACADEMIC PREDICTOR SAMPLE RESULT")

print("Predicted Result:", sample_result["Prediction"])

print(
    "Probability of At Risk:",
    f"{sample_result['At Risk Probability'] * 100:.2f}%"
)

print("\n" + "=" * 65)
print("             ACADEMIC PREDICTOR COMPLETED")
print("=" * 65)

print("\nSelected Model:", best_model_name)

print(f"Test Accuracy:  {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1-Score:  {test_f1:.4f}")
print(f"Test ROC-AUC:   {test_auc:.4f}")

print("\nTop Five Predictive Features:")

print(
    importance_df[
        ["Feature", "Importance"]
    ].head(5).to_string(index=False)
)

print("\nAll generated results saved in:", OUTPUT_DIR)

print("\nProject execution completed successfully.")
print("=" * 65)