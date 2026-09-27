
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

TRAIN_PATH = "tourism_project/model_building/train.csv"
TEST_PATH = "tourism_project/model_building/test.csv"
MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


# ---------------------------------------------------------
# Load training and testing data
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

TARGET = "ProdTaken"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# ---------------------------------------------------------
# Identify categorical and numerical features
# ---------------------------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# ---------------------------------------------------------
# Random Forest model
# ---------------------------------------------------------

rf = RandomForestClassifier(
    random_state=42
)


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("randomforestclassifier", rf)
    ]
)


# ---------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------

param_grid = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_depth": [None, 10, 20],
    "randomforestclassifier__min_samples_split": [2, 5]
}


# ---------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


# ---------------------------------------------------------
# MLflow experiment
# ---------------------------------------------------------

mlflow.set_experiment("VisitWithUs_Wellness_Tourism")

with mlflow.start_run():

    print("Starting hyperparameter tuning...")

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)


    # -----------------------------------------------------
    # Evaluate best model on test data
    # -----------------------------------------------------

    y_pred = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)

    print("Test Accuracy:", test_accuracy)


    # -----------------------------------------------------
    # Log parameters and metrics
    # -----------------------------------------------------

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("cv_accuracy", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", test_accuracy)


# ---------------------------------------------------------
# Save trained model
# ---------------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(best_model, MODEL_PATH)

print("Best model saved successfully.")
print("Model path:", MODEL_PATH)
