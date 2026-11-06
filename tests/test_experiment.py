"""Numerical checks for the RBF-EDMD demonstration."""

import numpy as np
import pytest

import experiment as exp


@pytest.fixture
def data():
    # Match the published experiment without changing other tests' RNG state.
    state = np.random.get_state()
    np.random.seed(3)
    try:
        yield exp.generate_data()
    finally:
        np.random.set_state(state)


def test_metrics_known_errors():
    metrics = exp.calculate_metrics(np.array([0.0, 3.0, 4.0]))
    assert metrics['mean'] == pytest.approx(7 / 3)
    assert metrics['rmse'] == pytest.approx(np.sqrt(25 / 3))
    assert metrics['p95'] == pytest.approx(3.9)
    assert metrics['max'] == 4.0


def test_generated_data_matches_simulator(data):
    for X, U, Y, count in (
        (*data[:3], exp.N_train),
        (*data[3:], exp.N_test),
    ):
        assert X.shape == U.shape == Y.shape == (count, 2)
        assert np.all(X >= [-2.5, -1.5])
        assert np.all(X <= [2.5, 1.5])
        assert np.all(np.abs(U) <= exp.umax)
        # Independent, vectorized expression of the specified simulator.
        drift = np.column_stack((
            0.20 * np.sin(1.5 * X[:, 1]),
            -0.15 * np.cos(1.2 * X[:, 0]),
        ))
        np.testing.assert_allclose(Y, X + exp.Ts * (U + drift))


def test_evaluation_with_exact_predictor(data):
    result = exp.evaluate_model(exp.true_F, *data)
    train_errors, test_errors, train_metrics, test_metrics, _, _ = result
    np.testing.assert_allclose(train_errors, 0, atol=1e-14)
    np.testing.assert_allclose(test_errors, 0, atol=1e-14)
    assert train_metrics['rmse'] == pytest.approx(0, abs=1e-14)
    assert test_metrics['rmse'] == pytest.approx(0, abs=1e-14)


def test_trained_model_beats_baselines_on_fixed_dataset(data):
    predict = exp.fit_model(*data[:3])
    X_test, U_test, Y_test = data[3:]
    predictions = np.array([predict(x, u) for x, u in zip(X_test, U_test)])
    assert predictions.shape == Y_test.shape
    assert np.isfinite(predictions).all()
    rmse = np.sqrt(np.mean(np.sum((Y_test - predictions) ** 2, axis=1)))
    persistence_rmse = np.sqrt(np.mean(np.sum((Y_test - X_test) ** 2, axis=1)))
    input_only_rmse = np.sqrt(np.mean(np.sum(
        (Y_test - (X_test + exp.Ts * U_test)) ** 2, axis=1
    )))
    # A broad regression threshold for this fixed, noise-free experiment only.
    assert rmse < 0.001
    assert rmse < input_only_rmse < persistence_rmse
