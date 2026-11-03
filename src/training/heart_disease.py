

import logging
import yaml

import pandas as pd

from pathlib import Path
from joblib import dump

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score,
)

from src.training.config.settings import settings


def train_model():

    try:

        print("Training function is executing!", flush=True)

        DATA_SET_PATH = Path(
            settings.heart_disease_dataset_path
        )

        MODEL_PATH = Path(
            settings.heart_disease_model_path
        )

        LOG_PATH = Path(settings.LOG_PATH)

        HYPER_PARAMS_YAML_PATH = Path(
            settings.hyper_params_yaml_path
        )

        TARGET_COL = settings.heart_disease_target_col

        RANDOM_STATE = settings.RANDOM_STATE

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
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(
                    LOG_PATH,
                    encoding="utf-8",
                ),
                logging.StreamHandler(),
            ],
            force=True,
        )

        logging.info("Heart Disease Training Started")

        df = pd.read_csv(DATA_SET_PATH)

        logging.info(
            f"Dataset loaded successfully from {DATA_SET_PATH}"
        )

        X = df.drop(columns=[TARGET_COL])

        y = df[TARGET_COL]

        row_signature = pd.util.hash_pandas_object(
            X,
            index=False,
        )

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=settings.test_size,
            random_state=RANDOM_STATE,
        )

        train_idx, test_idx = next(
            gss.split(
                X,
                y,
                groups=row_signature,
            )
        )

        X_train = X.iloc[train_idx]

        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]

        y_test = y.iloc[test_idx]

        logging.info(
            f"Train shape: {X_train.shape}, "
            f"Test shape: {X_test.shape}"
        )

        with open(
            HYPER_PARAMS_YAML_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            hyper_params = yaml.safe_load(file)

        logging.info(
            f"Hyperparameters loaded from "
            f"{HYPER_PARAMS_YAML_PATH}"
        )

        model_params = hyper_params["heart_disease"]["params"]

        best_rf = RandomForestClassifier(
            **model_params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", best_rf),
            ]
        )

        pipeline.fit(X_train, y_train)

        logging.info("Model training completed")

        y_train_pred = pipeline.predict(X_train)

        y_test_pred = pipeline.predict(X_test)

        train_acc = accuracy_score(
            y_train,
            y_train_pred,
        )

        test_acc = accuracy_score(
            y_test,
            y_test_pred,
        )

        train_recall = recall_score(
            y_train,
            y_train_pred,
        )

        test_recall = recall_score(
            y_test,
            y_test_pred,
        )

        train_f1 = f1_score(
            y_train,
            y_train_pred,
        )

        test_f1 = f1_score(
            y_test,
            y_test_pred,
        )

        logging.info(
            f"Train Accuracy: {train_acc:.4f} | "
            f"Recall: {train_recall:.4f} | "
            f"F1: {train_f1:.4f}"
        )

        logging.info(
            f"Test Accuracy: {test_acc:.4f} | "
            f"Recall: {test_recall:.4f} | "
            f"F1: {test_f1:.4f}"
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

        dump(
            pipeline,
            MODEL_PATH,
        )

        logging.info(
            f"Model saved to: {MODEL_PATH}"
        )

        logging.info(
            "Heart Disease Training Completed"
        )

    except Exception as e:

        logging.exception(
            f"Heart Disease Training Failed: {e}"
        )

        raise


if __name__ == "__main__":

    print(
        "Starting heart disease training...",
        flush=True,
    )

    train_model()
