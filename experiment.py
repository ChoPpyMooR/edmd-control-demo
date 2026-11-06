"""Reproducible RBF-EDMD control experiment. Run this file to generate results."""

import json

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Original experiment parameters.
Ts = 0.08

umax = 1.2

N_train = 300

N_test = 1000

num_centers = 40

sigma = 0.8

lam = 1e-5

def drift(x):
    """
    Unknown nonlinear drift.
    Used only for generating data.

    d(x) = [
        0.20 sin(1.5 x_2),
       -0.15 cos(1.2 x_1)
    ]
    """
    return np.array([
        0.20 * np.sin(1.5 * x[1]),
        -0.15 * np.cos(1.2 * x[0])
    ])

def true_F(x, u):
    """
    True discrete-time nonlinear dynamics:

        x_{k+1} = x_k + Ts * (u_k + d(x_k))
    """
    return x + Ts * (u + drift(x))

def calculate_metrics(errors):

    mean_error = np.mean(errors)

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    percentile_95 = np.percentile(
        errors,
        95
    )

    max_error = np.max(errors)

    return {
        "mean": mean_error,
        "rmse": rmse,
        "p95": percentile_95,
        "max": max_error
    }

def generate_data():
    X_train = np.random.uniform(
        low=np.array([-2.5, -1.5]),
        high=np.array([2.5, 1.5]),
        size=(N_train, 2)
    )

    U_train = np.random.uniform(
        low=-umax,
        high=umax,
        size=(N_train, 2)
    )

    X_next_train = np.array([
        true_F(X_train[i], U_train[i])
        for i in range(N_train)
    ])

    X_test = np.random.uniform(
        low=np.array([-2.5, -1.5]),
        high=np.array([2.5, 1.5]),
        size=(N_test, 2)
    )

    U_test = np.random.uniform(
        low=-umax,
        high=umax,
        size=(N_test, 2)
    )

    X_next_test = np.array([
        true_F(X_test[i], U_test[i])
        for i in range(N_test)
    ])

    return X_train, U_train, X_next_train, X_test, U_test, X_next_test


def fit_model(X_train, U_train, X_next_train):
    center_indices = np.random.choice(
        N_train,
        num_centers,
        replace=False
    )

    centers = X_train[center_indices]

    def lifting(x):
        """
        Finite-dimensional RBF lifting:

            Psi(x) = [x; phi(x)]

        The first two lifted coordinates are the original state.
        """

        x = np.atleast_2d(x)

        diff = x[:, None, :] - centers[None, :, :]

        r2 = np.sum(diff ** 2, axis=2)

        rbf_features = np.exp(
            -r2 / (2.0 * sigma ** 2)
        )

        return np.hstack([
            x,
            rbf_features
        ])

    M = 2 + num_centers

    Phi = lifting(X_train).T

    Phi_next = lifting(X_next_train).T

    U = U_train.T

    Z_linear = np.vstack([
        Phi,
        U
    ])

    reg_linear = lam * np.eye(
        Z_linear.shape[0]
    )

    AB = (
        Phi_next
        @ Z_linear.T
        @ np.linalg.inv(
            Z_linear @ Z_linear.T
            + reg_linear
        )
    )

    A = AB[:, :M]

    B = AB[:, M:]

    def F_hat_linear(x, u):
        """
        Linear Kernel-EDMD surrogate:

            z_{k+1} = A z_k + B u_k

        State prediction is obtained from
        the first two lifted coordinates.
        """

        z = lifting(x).ravel()

        u = np.asarray(u)

        z_next = (
            A @ z
            + B @ u
        )

        return z_next[:2]

    return F_hat_linear


def evaluate_model(predict, X_train, U_train, X_next_train, X_test, U_test, X_next_test):
    F_hat_linear = predict
    X_pred_train = np.array([
        F_hat_linear(
            X_train[i],
            U_train[i]
        )
        for i in range(N_train)
    ])

    train_errors = np.linalg.norm(
        X_next_train - X_pred_train,
        axis=1
    )

    X_pred_test = np.array([
        F_hat_linear(
            X_test[i],
            U_test[i]
        )
        for i in range(N_test)
    ])

    test_errors = np.linalg.norm(
        X_next_test - X_pred_test,
        axis=1
    )

    train_metrics = calculate_metrics(
        train_errors
    )

    test_metrics = calculate_metrics(
        test_errors
    )

    X_pred_persistence = X_test.copy()

    X_pred_input_only = X_test + Ts * U_test

    persistence_metrics = calculate_metrics(
        np.linalg.norm(X_next_test - X_pred_persistence, axis=1)
    )

    input_only_metrics = calculate_metrics(
        np.linalg.norm(X_next_test - X_pred_input_only, axis=1)
    )

    return (train_errors, test_errors, train_metrics, test_metrics, persistence_metrics, input_only_metrics)


def report_results(output_dir, train_metrics, test_metrics, persistence_metrics, input_only_metrics):
    RESULTS_DIR = output_dir
    M = 2 + num_centers
    print("\nTEST RMSE COMPARISON")

    print(f"Persistence       : {persistence_metrics['rmse']:.6e}")

    print(f"Input-only        : {input_only_metrics['rmse']:.6e}")

    print(f"RBF-EDMD          : {test_metrics['rmse']:.6e}")

    results = {
        "config": {
            "seed": 3,
            "sampling_time": Ts,
            "input_limit": umax,
            "n_train": N_train,
            "n_test": N_test,
            "num_centers": num_centers,
            "sigma": sigma,
            "ridge_lambda": lam,
        },
        "metric": "sqrt(mean(sum((x_next - x_pred)**2, axis=1)))",
        "train": {
            "rbf_edmd": train_metrics,
        },
        "test": {
            "persistence": persistence_metrics,
            "input_only": input_only_metrics,
            "rbf_edmd": test_metrics,
        },
    }

    with (RESULTS_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print(f"\nResults directory: {RESULTS_DIR}")

    print()

    print("=" * 65)

    print("RBF-EDMD CONTROL BASELINE")

    print("=" * 65)

    print()

    print("Model dimensions:")

    print(f"State dimension          : 2")

    print(f"Input dimension          : 2")

    print(f"Lifted dimension M       : {M}")

    print(f"Number of RBF centers    : {num_centers}")

    print(f"Training samples         : {N_train}")

    print(f"Test samples             : {N_test}")

    print(f"Ridge lambda             : {lam:.2e}")

    print()

    print("-" * 65)

    print("TRAINING ERROR")

    print("-" * 65)

    print(
        f"Mean error               : "
        f"{train_metrics['mean']:.6e}"
    )

    print(
        f"RMSE                     : "
        f"{train_metrics['rmse']:.6e}"
    )

    print(
        f"95th percentile          : "
        f"{train_metrics['p95']:.6e}"
    )

    print(
        f"Maximum error            : "
        f"{train_metrics['max']:.6e}"
    )

    print()

    print("-" * 65)

    print("INDEPENDENT TEST ERROR")

    print("-" * 65)

    print(
        f"Mean error               : "
        f"{test_metrics['mean']:.6e}"
    )

    print(
        f"RMSE                     : "
        f"{test_metrics['rmse']:.6e}"
    )

    print(
        f"95th percentile          : "
        f"{test_metrics['p95']:.6e}"
    )

    print(
        f"Maximum error            : "
        f"{test_metrics['max']:.6e}"
    )

    generalization_gap = (
        test_metrics["rmse"]
        - train_metrics["rmse"]
    )

    print()

    print("-" * 65)

    print("GENERALIZATION")

    print("-" * 65)

    print(
        f"Test RMSE - Train RMSE   : "
        f"{generalization_gap:.6e}"
    )

    print("=" * 65)


def plot_one_step(output_dir, train_errors, test_errors, test_metrics):
    RESULTS_DIR = output_dir
    plt.figure(figsize=(7, 4.5))

    plt.hist(
        train_errors,
        bins=35,
        alpha=0.6,
        density=True,
        label="Training error"
    )

    plt.hist(
        test_errors,
        bins=35,
        alpha=0.6,
        density=True,
        label="Test error"
    )

    plt.xlabel(
        r"$\|x_{k+1}-\hat{x}_{k+1}\|_2$"
    )

    plt.ylabel("Density")

    plt.title(
        "RBF-EDMD Prediction Error"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "error_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    plt.figure(figsize=(8, 4.5))

    plt.plot(
        np.arange(N_test),
        test_errors,
        linewidth=1.0,
        label="RBF-EDMD"
    )

    plt.axhline(
        test_metrics["p95"],
        linestyle="--",
        label="95th percentile"
    )

    plt.xlabel("Test sample index")

    plt.ylabel(
        r"$\|x_{k+1}-\hat{x}_{k+1}\|_2$"
    )

    plt.title(
        "RBF-EDMD: Independent Test Error"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "test_errors.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


def run_rollout(predict, output_dir):
    F_hat_linear = predict
    RESULTS_DIR = output_dir
    steps = 100

    initial_state = np.array([0.5, -0.3])

    rollout_rng = np.random.default_rng(42)

    inputs = rollout_rng.uniform(
        low=-umax,
        high=umax,
        size=(steps, 2),
    )

    true_states = np.zeros((steps + 1, 2))

    predicted_states = np.zeros((steps + 1, 2))

    true_states[0] = initial_state

    predicted_states[0] = initial_state

    for k in range(steps):
        true_states[k + 1] = true_F(
            true_states[k],
            inputs[k],
        )

        predicted_states[k + 1] = F_hat_linear(
            predicted_states[k],
            inputs[k],
        )

    rollout_errors = np.linalg.norm(
        true_states - predicted_states,
        axis=1,
    )

    rollout_metrics = calculate_metrics(rollout_errors[1:])

    print("\nMULTI-STEP PREDICTION")

    print(f"Steps             : {steps}")

    print(f"Trajectory RMSE   : {rollout_metrics['rmse']:.6e}")

    print(f"Final-step error  : {rollout_errors[-1]:.6e}")

    print(f"Maximum error     : {rollout_metrics['max']:.6e}")

    time = np.arange(steps + 1) * Ts

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    for i in range(2):
        axes[i].plot(
            time, true_states[:, i],
            label="True",
        )
        axes[i].plot(
            time, predicted_states[:, i],
            "--", label="RBF-EDMD",
        )
        axes[i].set_ylabel(f"x{i + 1}")
        axes[i].legend()
        axes[i].grid(True)

    axes[2].plot(time, rollout_errors, color="tab:red")

    axes[2].set_ylabel("State error (L2)")

    axes[2].set_xlabel("Time")

    axes[2].grid(True)

    fig.suptitle("Multi-step prediction with state re-lifting")

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "rollout.png",
        dpi=200,
        bbox_inches="tight",
    )

    with (RESULTS_DIR / "rollout_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "steps": steps,
                "input_seed": 42,
                "initial_state": initial_state.tolist(),
                "metrics": rollout_metrics,
                "final_error": float(rollout_errors[-1]),
            },
            file,
            indent=2,
        )

    plt.show()


def main():
    # Preserve the original random sequence and numerical operations.
    np.random.seed(3)
    RESULTS_DIR.mkdir(exist_ok=True)
    data = generate_data()
    X_train, U_train, X_next_train, X_test, U_test, X_next_test = data
    predict = fit_model(X_train, U_train, X_next_train)
    (train_errors, test_errors, train_metrics, test_metrics,
     persistence_metrics, input_only_metrics) = evaluate_model(predict, *data)
    report_results(
        RESULTS_DIR, train_metrics, test_metrics,
        persistence_metrics, input_only_metrics,
    )
    plot_one_step(RESULTS_DIR, train_errors, test_errors, test_metrics)
    run_rollout(predict, RESULTS_DIR)


if __name__ == "__main__":
    main()
