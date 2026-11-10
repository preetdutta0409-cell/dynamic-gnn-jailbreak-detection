"""
baseline_classifier.py

Purpose:
Implement a non-graph machine learning baseline using
Logistic Regression.

The baseline learns from numerical node features without
using graph connectivity.

Primary target:
Proposed-action classification.

Label mapping:
SAFE = 0
MALICIOUS = 1

AMBIGUOUS samples must be removed before training.
"""

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaselineClassifier:
    """
    Logistic Regression baseline for jailbreak detection.

    Input:
        X: Numerical feature matrix.
        y: Binary labels (0 = SAFE, 1 = MALICIOUS).

    Output:
        Predicted labels and risk scores.

    This model does not use graph edges or neighbouring nodes.
    """

    SAFE = 0
    MALICIOUS = 1

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
    ) -> None:

        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight

        # StandardScaler normalises feature scales.
        # Logistic Regression then learns the classification boundary.
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=self.random_state,
                        max_iter=self.max_iter,
                        class_weight=self.class_weight,
                    ),
                ),
            ]
        )

        self.is_fitted = False
        self.input_dimension = None

    @staticmethod
    def _validate_features(X: np.ndarray) -> np.ndarray:
        """
        Validate the input feature matrix.

        Expected shape:
            (number_of_samples, number_of_features)
        """

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional feature matrix."
            )

        if X.shape[0] == 0:
            raise ValueError(
                "X must contain at least one sample."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one feature."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return X

    @staticmethod
    def _validate_labels(y: np.ndarray, sample_count: int) -> np.ndarray:
        """
        Validate binary labels.

        Allowed values:
            0 = SAFE
            1 = MALICIOUS

        Both classes must be present during training.
        """

        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError(
                "y must be a one-dimensional label array."
            )

        if len(y) != sample_count:
            raise ValueError(
                "The number of labels must match the number of samples."
            )

        if not np.isin(y, [0, 1]).all():
            raise ValueError(
                "Labels must contain only 0 (SAFE) and 1 (MALICIOUS). "
                "Remove AMBIGUOUS samples before training."
            )

        y = y.astype(np.int64)

        if len(np.unique(y)) != 2:
            raise ValueError(
                "Training data must contain both SAFE and MALICIOUS samples."
            )

        return y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "BaselineClassifier":
        """
        Train the Logistic Regression baseline.

        X and y must already belong to the training split.
        Do not pass validation or test samples here.
        """

        X = self._validate_features(X)
        y = self._validate_labels(y, sample_count=X.shape[0])

        self.pipeline.fit(X, y)

        self.input_dimension = X.shape[1]
        self.is_fitted = True

        return self

    def _check_fitted(self) -> None:
        """Prevent predictions before model training."""

        if not self.is_fitted:
            raise RuntimeError(
                "The baseline classifier has not been trained. "
                "Call fit() before prediction."
            )

    def _validate_prediction_features(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Validate prediction features and their dimensions."""

        self._check_fitted()

        X = self._validate_features(X)

        if X.shape[1] != self.input_dimension:
            raise ValueError(
                f"Expected {self.input_dimension} features per sample, "
                f"received {X.shape[1]}."
            )

        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Returns:
            0 = SAFE
            1 = MALICIOUS
        """

        X = self._validate_prediction_features(X)

        return self.pipeline.predict(X).astype(np.int64)

    def predict_risk_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return the model's score for the MALICIOUS class.

        These scores are not guaranteed to be calibrated probabilities.
        """

        X = self._validate_prediction_features(X)

        classifier = self.pipeline.named_steps["classifier"]

        malicious_class_index = np.where(
            classifier.classes_ == self.MALICIOUS
        )[0][0]

        scores = self.pipeline.predict_proba(X)[
            :, malicious_class_index
        ]

        return scores.astype(np.float32)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save the trained classifier to disk."""

        self._check_fitted()

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "pipeline": self.pipeline,
            "input_dimension": self.input_dimension,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "class_mapping": {
                "SAFE": self.SAFE,
                "MALICIOUS": self.MALICIOUS,
            },
            "model_name": "logistic_regression_baseline",
            "model_version": "baseline_v1",
        }

        joblib.dump(checkpoint, file_path)

    @classmethod
    def load(
        cls,
        file_path: Union[str, Path],
    ) -> "BaselineClassifier":
        """Load a previously saved classifier."""

        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {file_path}"
            )

        checkpoint = joblib.load(file_path)

        required_keys = {
            "pipeline",
            "input_dimension",
            "random_state",
            "max_iter",
            "class_weight",
            "class_mapping",
            "model_name",
            "model_version",
        }

        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid baseline checkpoint format.")

        if not required_keys.issubset(checkpoint):
            raise ValueError(
                "Baseline checkpoint is missing required fields."
            )

        if checkpoint["class_mapping"] != {
            "SAFE": cls.SAFE,
            "MALICIOUS": cls.MALICIOUS,
        }:
            raise ValueError(
                "Checkpoint label mapping does not match this classifier."
            )

        model = cls(
            random_state=checkpoint["random_state"],
            max_iter=checkpoint["max_iter"],
            class_weight=checkpoint["class_weight"],
        )

        model.pipeline = checkpoint["pipeline"]
        model.input_dimension = checkpoint["input_dimension"]
        model.is_fitted = True

        return model
