import numpy as np
import pytest

from src.models.baseline_classifier import BaselineClassifier


@pytest.fixture
def training_data():
    """
    Small artificial dataset for testing.

    First three samples are SAFE.
    Last three samples are MALICIOUS.
    """

    X = np.array(
        [
            [0.1, 0.2, 0.1],
            [0.2, 0.1, 0.2],
            [0.1, 0.3, 0.2],
            [0.8, 0.9, 0.8],
            [0.9, 0.8, 0.9],
            [0.8, 0.8, 0.9],
        ],
        dtype=np.float32,
    )

    y = np.array(
        [0, 0, 0, 1, 1, 1],
        dtype=np.int64,
    )

    return X, y


@pytest.fixture
def trained_model(training_data):
    X, y = training_data

    model = BaselineClassifier()
    model.fit(X, y)

    return model


def test_model_training(training_data):
    X, y = training_data

    model = BaselineClassifier()
    result = model.fit(X, y)

    assert model.is_fitted is True
    assert result is model


def test_model_prediction_shape(trained_model, training_data):
    X, _ = training_data

    predictions = trained_model.predict(X)

    assert predictions.shape == (len(X),)


def test_model_predicts_only_binary_labels(
    trained_model,
    training_data,
):
    X, _ = training_data

    predictions = trained_model.predict(X)

    assert set(np.unique(predictions)).issubset({0, 1})


def test_model_predicts_training_examples(
    trained_model,
    training_data,
):
    X, y = training_data

    predictions = trained_model.predict(X)

    assert np.array_equal(predictions, y)


def test_risk_scores_shape(trained_model, training_data):
    X, _ = training_data

    scores = trained_model.predict_risk_scores(X)

    assert scores.shape == (len(X),)


def test_risk_scores_are_finite_and_bounded(
    trained_model,
    training_data,
):
    X, _ = training_data

    scores = trained_model.predict_risk_scores(X)

    assert np.isfinite(scores).all()
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_prediction_before_training_is_rejected():
    model = BaselineClassifier()

    X = np.array([[0.1, 0.2]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="has not been trained"):
        model.predict(X)


def test_invalid_feature_dimension_is_rejected(
    trained_model,
):
    X = np.array([[0.1, 0.2]], dtype=np.float32)

    with pytest.raises(ValueError, match="Expected 3 features"):
        trained_model.predict(X)


def test_invalid_label_is_rejected():
    model = BaselineClassifier()

    X = np.array(
        [[0.1, 0.2], [0.8, 0.9]],
        dtype=np.float32,
    )

    y = np.array([0, 2])

    with pytest.raises(ValueError, match="only 0"):
        model.fit(X, y)


def test_ambiguous_label_is_rejected():
    model = BaselineClassifier()

    X = np.array(
        [[0.1, 0.2], [0.8, 0.9]],
        dtype=np.float32,
    )

    # 2 represents an unsupported/ambiguous label.
    y = np.array([0, 2])

    with pytest.raises(ValueError, match="Remove AMBIGUOUS"):
        model.fit(X, y)


def test_single_class_training_data_is_rejected():
    model = BaselineClassifier()

    X = np.array(
        [[0.1, 0.2], [0.2, 0.3]],
        dtype=np.float32,
    )

    y = np.array([0, 0])

    with pytest.raises(ValueError, match="both SAFE and MALICIOUS"):
        model.fit(X, y)


def test_non_finite_features_are_rejected():
    model = BaselineClassifier()

    X = np.array(
        [[0.1, 0.2], [np.nan, 0.9]],
        dtype=np.float32,
    )

    y = np.array([0, 1])

    with pytest.raises(ValueError, match="NaN or infinite"):
        model.fit(X, y)


def test_model_save_and_load(
    trained_model,
    training_data,
    tmp_path,
):
    X, _ = training_data

    checkpoint_path = tmp_path / "baseline.pkl"

    trained_model.save(checkpoint_path)

    loaded_model = BaselineClassifier.load(checkpoint_path)

    original_predictions = trained_model.predict(X)
    loaded_predictions = loaded_model.predict(X)

    assert np.array_equal(
        original_predictions,
        loaded_predictions,
    )


def test_missing_checkpoint_is_rejected(tmp_path):
    checkpoint_path = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError):
        BaselineClassifier.load(checkpoint_path)


def test_empty_feature_matrix_is_rejected():
    model = BaselineClassifier()

    X = np.empty((0, 3), dtype=np.float32)
    y = np.array([], dtype=np.int64)

    with pytest.raises(ValueError, match="at least one sample"):
        model.fit(X, y)
