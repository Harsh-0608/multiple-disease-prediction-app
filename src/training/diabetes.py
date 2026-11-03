
import logging
import yaml

import numpy as np
import pandas as pd

from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score,
)

from src.training.config.settings import Settings


def replace_zeros_with_nan(X):

    X = X.copy()

    cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    for col in cols:

        if col in X.columns:

            X[col] = X[col].replace(0, np.nan)

    return X


def train_diabetes_model():

    try:

        settings = Settings()
        print("Training function is executing!", flush=True)

        DATASET_PATH = settings.diabetes_dataset_path_absolute

        MODEL_PATH = settings.diabetes_model_path_absolute

        LOG_PATH = settings.log_path_absolute
        HYPER_PARAMS_YAML_PATH = (
             settings.hyper_params_yaml_path_absolute
)
      
        MODEL_PATH.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
        
        LOG_PATH.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    LOG_PATH,
                    encoding="utf-8",
                ),
            ],
            force=True,
        )   
        logging.info("Diabetes Training Started")     

        # if not HYPER_PARAMS_YAML_PATH.exists():
        #     raise FileNotFoundError(
        #         f"Hyperparameters YAML file not found: "
        #         f"{HYPER_PARAMS_YAML_PATH}"
        #     )

        with open(
            HYPER_PARAMS_YAML_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            hyperparams = yaml.safe_load(file)

        model_params = hyperparams["diabetes"]["params"]


        TARGET_COL = settings.diabetes_target_col

        TEST_SIZE = settings.test_size

        RANDOM_STATE = settings.RANDOM_STATE
         


        df = pd.read_csv(DATASET_PATH)

        logging.info(
            f"Dataset loaded with shape: {df.shape}"
        )

        X = df.drop(columns=[TARGET_COL])

        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        logging.info(
            f"Train shape: {X_train.shape}, "
            f"Test shape: {X_test.shape}"
        )

        numeric_cols = X_train.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            (
                                "zero_to_nan",
                                FunctionTransformer(
                                    replace_zeros_with_nan,
                                    validate=False,
                                ),
                            ),
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median"
                                ),
                            ),
                            (
                                "scaler",
                                StandardScaler(),
                            ),
                        ]
                    ),
                    numeric_cols,
                ),
            ],
            remainder="drop",
        )

       

        model = SVC(
            random_state=RANDOM_STATE,
            **model_params,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        logging.info("Model training completed")

        y_train_pred = pipeline.predict(X_train)

        y_test_pred = pipeline.predict(X_test)

        logging.info(
            f"Train Acc: {accuracy_score(y_train, y_train_pred):.4f} | "
            f"Recall: {recall_score(y_train, y_train_pred):.4f} | "
            f"F1: {f1_score(y_train, y_train_pred):.4f}"
        )

        logging.info(
            f"Test Acc: {accuracy_score(y_test, y_test_pred):.4f} | "
            f"Recall: {recall_score(y_test, y_test_pred):.4f} | "
            f"F1: {f1_score(y_test, y_test_pred):.4f}"
        )

        logging.info(
            "Train Classification Report:\n"
            + classification_report(
                y_train,
                y_train_pred,
            )
        )

        logging.info(
            "Test Classification Report:\n"
            + classification_report(
                y_test,
                y_test_pred,
            )
        )

        dump(pipeline, MODEL_PATH)

        logging.info(f"Model saved at: {MODEL_PATH}")

        logging.info("Diabetes Training Completed")

    except Exception as e:

        logging.exception(f"Training failed: {e}")

        raise

if __name__ == "__main__":
    print("Starting diabetes training...", flush=True)
    train_diabetes_model()
