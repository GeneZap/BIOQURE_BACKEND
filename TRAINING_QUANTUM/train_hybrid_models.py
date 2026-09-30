"""Train and evaluate BioQure classical baselines and a Qiskit VQC honestly.

Default runs report validation performance only. Tune architecture using those
validation results. Run once with ``--evaluate-test`` after configuration is
locked to reveal final held-out test performance and calibrated probabilities.

No score is fabricated when quantum training fails: an exception stops the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_machine_learning.algorithms import VQC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe classical and VQC models."
    )
    parser.add_argument("--data-dir", default="prepared_data")
    parser.add_argument("--output-dir", default="training_outputs")
    parser.add_argument("--feature-map-reps", type=int, default=1)
    parser.add_argument("--ansatz-reps", type=int, default=2)
    parser.add_argument(
        "--entanglement", choices=["linear", "circular", "full"], default="linear"
    )
    parser.add_argument("--optimizer", choices=["cobyla", "spsa"], default="cobyla")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--minimum-sensitivity",
        type=float,
        default=0.80,
        help="Preferred minimum validation sensitivity during threshold selection",
    )
    parser.add_argument(
        "--minimum-specificity",
        type=float,
        default=0.70,
        help="Preferred minimum validation specificity during threshold selection",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Reveal held-out test results only after tuning is finished",
    )
    return parser.parse_args()


def load_array(data_dir: Path, name: str) -> np.ndarray:
    path = data_dir / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared array: {path}")
    return np.load(path)


def positive_probability(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 2:
        return values[:, 1]
    raise ValueError(f"Unexpected predict_proba shape: {values.shape}")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray | None = None,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    metrics: dict[str, float | int] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if score is not None:
        score = np.asarray(score, dtype=float).reshape(-1)
        metrics["roc_auc"] = float(roc_auc_score(y_true, score))
        metrics["average_precision"] = float(average_precision_score(y_true, score))
        if np.all((score >= 0) & (score <= 1)):
            metrics["brier_score"] = float(brier_score_loss(y_true, score))
    return metrics


def print_metrics(model_name: str, split_name: str, metrics: dict) -> None:
    print(f"\n{model_name} — {split_name}")
    print(
        f"  accuracy={metrics['accuracy']:.4f} | "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} | "
        f"sensitivity={metrics['sensitivity_recall']:.4f} | "
        f"specificity={metrics['specificity']:.4f} | "
        f"MCC={metrics['mcc']:.4f}"
    )
    print(
        f"  confusion matrix: TN={metrics['tn']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TP={metrics['tp']}"
    )
    if "roc_auc" in metrics:
        print(
            f"  ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['average_precision']:.4f}"
        )


def search_validation_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    minimum_sensitivity: float,
    minimum_specificity: float,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Select a decision threshold using validation data only.

    The untouched test split is never inspected here. The preferred search
    requires the requested sensitivity and specificity; if no threshold meets
    both constraints, the function falls back to maximum balanced accuracy.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probability = np.asarray(probability, dtype=float).reshape(-1)
    rows: list[dict[str, float | int]] = []

    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        predictions = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, predictions)
                ),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "mcc": float(matthews_corrcoef(y_true, predictions)),
                "youden_j": float(sensitivity + specificity - 1.0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    table = pd.DataFrame(rows)
    eligible = table[
        (table["sensitivity"] >= minimum_sensitivity)
        & (table["specificity"] >= minimum_specificity)
    ]

    if eligible.empty:
        pool = table
        method = (
            "maximum balanced accuracy fallback; no threshold met "
            f"sensitivity >= {minimum_sensitivity:.2f} and "
            f"specificity >= {minimum_specificity:.2f}"
        )
    else:
        pool = eligible
        method = (
            "maximum balanced accuracy subject to "
            f"sensitivity >= {minimum_sensitivity:.2f} and "
            f"specificity >= {minimum_specificity:.2f}"
        )

    best = pool.sort_values(
        ["balanced_accuracy", "mcc", "sensitivity"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    selected: dict[str, float | int | str] = {
        key: (int(best[key]) if key in {"tn", "fp", "fn", "tp"} else float(best[key]))
        for key in table.columns
    }
    selected["selection_method"] = method
    return table, selected


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "preparation_config.json", encoding="utf-8") as handle:
        preparation_config = json.load(handle)
    feature_names = preparation_config["feature_names"]

    X_train_classical = load_array(data_dir, "X_train_classical")
    X_validation_classical = load_array(data_dir, "X_validation_classical")
    X_train_quantum = load_array(data_dir, "X_train_quantum")
    X_validation_quantum = load_array(data_dir, "X_validation_quantum")
    y_train = load_array(data_dir, "y_train")
    y_validation = load_array(data_dir, "y_validation")
    y_train_quantum = load_array(data_dir, "y_train_quantum")

    if X_train_classical.shape[1] != len(feature_names):
        raise ValueError("Prepared feature count does not match preparation_config.json")

    print("=== BioQure Leakage-Safe Training ===")
    print(f"Biomarkers/qubits: {len(feature_names)} — {feature_names}")
    print(
        f"Classical train samples: {len(y_train)} | "
        f"Balanced quantum train samples: {len(y_train_quantum)}"
    )

    results: dict[str, dict] = {
        "configuration": {
            "features": feature_names,
            "feature_map_reps": args.feature_map_reps,
            "ansatz_reps": args.ansatz_reps,
            "entanglement": args.entanglement,
            "optimizer": args.optimizer,
            "maxiter": args.maxiter,
            "shots": args.shots,
            "seed": args.seed,
            "test_evaluated": args.evaluate_test,
            "minimum_sensitivity": args.minimum_sensitivity,
            "minimum_specificity": args.minimum_specificity,
        },
        "validation": {},
    }

    print("\n[Phase A] Classical baselines")
    classical_models = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=5000, random_state=args.seed
        ),
        "rbf_svm": SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=args.seed
        ),
    }

    for name, model in classical_models.items():
        model.fit(X_train_classical, y_train)
        predictions = model.predict(X_validation_classical)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_validation_classical)[:, 1]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X_validation_classical)
        else:
            score = None
        metrics = calculate_metrics(y_validation, predictions, score)
        results["validation"][name] = metrics
        print_metrics(name, "validation", metrics)
        joblib.dump(model, output_dir / f"{name}.joblib")

    print("\n[Phase B] Variational Quantum Classifier")
    num_features = X_train_quantum.shape[1]
    feature_map = zz_feature_map(
        num_features,
        reps=args.feature_map_reps,
        entanglement=args.entanglement,
    )
    ansatz = real_amplitudes(
        num_features,
        reps=args.ansatz_reps,
        entanglement=args.entanglement,
    )

    rng = np.random.default_rng(args.seed)
    initial_point = rng.uniform(-0.1, 0.1, size=ansatz.num_parameters)
    sampler = StatevectorSampler(default_shots=args.shots, seed=args.seed)
    optimizer = (
        COBYLA(maxiter=args.maxiter)
        if args.optimizer == "cobyla"
        else SPSA(maxiter=args.maxiter)
    )

    loss_history: list[float] = []

    def training_callback(_weights, objective_value) -> None:
        loss_history.append(float(objective_value))
        iteration = len(loss_history)
        if iteration == 1 or iteration % 10 == 0:
            print(f"  iteration={iteration:4d} objective={objective_value:.6f}")

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        initial_point=initial_point,
        callback=training_callback,
    )

    start = time.perf_counter()
    # Intentionally no catch-and-invent fallback: a failure must remain visible.
    vqc.fit(X_train_quantum, y_train_quantum)
    training_seconds = time.perf_counter() - start

    validation_probability = positive_probability(vqc.predict_proba(X_validation_quantum))
    validation_predictions = (validation_probability >= 0.50).astype(int)
    vqc_validation = calculate_metrics(
        y_validation, validation_predictions, validation_probability
    )
    vqc_validation["decision_threshold"] = 0.50
    vqc_validation["training_seconds"] = float(training_seconds)
    vqc_validation["optimizer_evaluations"] = int(len(loss_history))
    results["validation"]["vqc"] = vqc_validation
    print_metrics("VQC", "validation at threshold 0.50", vqc_validation)
    print(f"  training time={training_seconds:.2f} seconds")

    threshold_table, selected_threshold = search_validation_threshold(
        y_validation,
        validation_probability,
        minimum_sensitivity=args.minimum_sensitivity,
        minimum_specificity=args.minimum_specificity,
    )
    threshold_table.to_csv(output_dir / "vqc_threshold_search.csv", index=False)

    best_threshold = float(selected_threshold["threshold"])
    tuned_validation_predictions = (
        validation_probability >= best_threshold
    ).astype(int)
    tuned_validation = calculate_metrics(
        y_validation,
        tuned_validation_predictions,
        validation_probability,
    )
    tuned_validation["decision_threshold"] = best_threshold
    tuned_validation["threshold_selection_method"] = selected_threshold[
        "selection_method"
    ]
    results["validation"]["vqc_threshold_tuned"] = tuned_validation
    results["configuration"]["selected_validation_threshold"] = best_threshold

    print("\n=== Validation Threshold Selection ===")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Method: {selected_threshold['selection_method']}")
    print_metrics("VQC threshold-tuned", "validation", tuned_validation)

    pd.DataFrame(
        {
            "y_true": y_validation.astype(int),
            "raw_vqc_probability": validation_probability,
            "prediction_at_0_50": validation_predictions,
            "prediction_at_selected_threshold": tuned_validation_predictions,
        }
    ).to_csv(output_dir / "vqc_validation_predictions.csv", index=False)

    pd.DataFrame(
        {
            "evaluation": np.arange(1, len(loss_history) + 1),
            "objective": loss_history,
        }
    ).to_csv(output_dir / "vqc_loss_history.csv", index=False)
    vqc.to_dill(str(output_dir / "vqc_model.dill"))

    # Do not inspect test performance during architecture/optimizer tuning.
    if args.evaluate_test:
        print("\n[Phase C] One-time held-out test evaluation")
        results["test"] = {}
        X_test_classical = load_array(data_dir, "X_test_classical")
        X_test_quantum = load_array(data_dir, "X_test_quantum")
        y_test = load_array(data_dir, "y_test")

        for name, model in classical_models.items():
            predictions = model.predict(X_test_classical)
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X_test_classical)[:, 1]
            elif hasattr(model, "decision_function"):
                score = model.decision_function(X_test_classical)
            else:
                score = None
            metrics = calculate_metrics(y_test, predictions, score)
            results["test"][name] = metrics
            print_metrics(name, "test", metrics)

        # Calibrate raw VQC tumor probabilities using validation only.
        calibrator = LogisticRegression(random_state=args.seed)
        calibrator.fit(validation_probability.reshape(-1, 1), y_validation)
        raw_test_probability = positive_probability(vqc.predict_proba(X_test_quantum))

        # Apply the threshold selected on validation without inspecting test labels.
        threshold_test_predictions = (
            raw_test_probability >= best_threshold
        ).astype(int)
        threshold_test = calculate_metrics(
            y_test, threshold_test_predictions, raw_test_probability
        )
        threshold_test["decision_threshold"] = best_threshold
        results["test"]["vqc_threshold_tuned"] = threshold_test
        print_metrics("VQC threshold-tuned", "test", threshold_test)

        calibrated_test_probability = calibrator.predict_proba(
            raw_test_probability.reshape(-1, 1)
        )[:, 1]
        calibrated_test_predictions = (calibrated_test_probability >= 0.5).astype(int)
        vqc_test = calculate_metrics(
            y_test, calibrated_test_predictions, calibrated_test_probability
        )
        results["test"]["vqc_calibrated"] = vqc_test
        print_metrics("VQC calibrated", "test", vqc_test)
        print(f"  Brier score={vqc_test['brier_score']:.4f} (lower is better)")
        joblib.dump(calibrator, output_dir / "vqc_probability_calibrator.joblib")

        pd.DataFrame(
            {
                "y_true": y_test,
                "raw_vqc_probability": raw_test_probability,
                "calibrated_vqc_probability": calibrated_test_probability,
                "threshold_tuned_prediction": threshold_test_predictions,
                "calibrated_prediction": calibrated_test_predictions,
            }
        ).to_csv(output_dir / "vqc_test_predictions.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    best_classical = max(
        ("logistic_regression", "rbf_svm"),
        key=lambda name: results["validation"][name]["balanced_accuracy"],
    )
    classical_score = results["validation"][best_classical]["balanced_accuracy"]
    quantum_score = results["validation"]["vqc"]["balanced_accuracy"]
    tuned_quantum_score = results["validation"]["vqc_threshold_tuned"][
        "balanced_accuracy"
    ]

    print("\n=== Validation Decision ===")
    print(
        f"Best classical: {best_classical} "
        f"(balanced accuracy={classical_score:.4f})"
    )
    print(f"VQC balanced accuracy: {quantum_score:.4f}")
    print(
        "Threshold-tuned VQC balanced accuracy: "
        f"{tuned_quantum_score:.4f} "
        f"(threshold={best_threshold:.2f})"
    )
    if tuned_quantum_score >= classical_score:
        print("Result: VQC is competitive on this validation split.")
    else:
        print("Result: classical model remains the preferred validated model.")
    print("This is a model comparison, not evidence of quantum advantage.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
