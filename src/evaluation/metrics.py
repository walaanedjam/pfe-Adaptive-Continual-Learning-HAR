"""
Evaluation metrics for continual HAR.

Standard metrics:
  - Macro F1 score (primary metric, handles class imbalance)
  - Per-class accuracy

Continual learning metrics (Lopez-Paz & Ranzato, 2017):
  - Backward Transfer (BWT): how much performance on old tasks degrades
  - Forward Transfer (FWT): how much pre-learning helps new tasks
  - Forgetting measure:      maximum performance drop per old task
  - Intransigence:           inability to learn new tasks

All metrics tracked in a results matrix R where:
  R[i, j] = test accuracy on task j after training on task i  (i >= j)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
             labels: Optional[List[int]] = None) -> float:
    """Macro-averaged F1 score (unweighted mean over classes)."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=labels,
                          average="macro", zero_division=0))


def per_class_accuracy(y_true: np.ndarray,
                        y_pred: np.ndarray) -> Dict[int, float]:
    """Per-class recall (= accuracy when evaluated per class)."""
    classes = np.unique(y_true)
    return {
        int(c): float((y_pred[y_true == c] == c).mean())
        for c in classes
    }


def confusion_matrix(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      labels: Optional[List[int]] = None) -> np.ndarray:
    from sklearn.metrics import confusion_matrix as _cm
    return _cm(y_true, y_pred, labels=labels)


def classification_report(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            label_names: Optional[Dict[int, str]] = None) -> str:
    from sklearn.metrics import classification_report as _cr
    target_names = None
    if label_names:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        target_names = [label_names.get(l, str(l)) for l in labels]
    return _cr(y_true, y_pred, target_names=target_names, zero_division=0)


# ---------------------------------------------------------------------------
# Continual learning metrics
# ---------------------------------------------------------------------------

class ContinualResultsMatrix:
    """
    Tracks performance across tasks for computing BWT, FWT, and forgetting.

    Usage:
        matrix = ContinualResultsMatrix(n_tasks=5)

        # After training on task i, evaluate on all tasks j <= i:
        for j in range(i + 1):
            acc = evaluate(model, task_j_test)
            matrix.record(task_i=i, task_j=j, accuracy=acc)

        print(matrix.backward_transfer())
        print(matrix.forgetting())
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        # R[i, j] = accuracy on task j after training on task i
        self.R = np.full((n_tasks, n_tasks), np.nan)

    def record(self, task_i: int, task_j: int, accuracy: float):
        """Record R[task_i, task_j] = accuracy."""
        self.R[task_i, task_j] = accuracy

    def backward_transfer(self) -> float:
        """
        BWT = mean of (R[T-1, j] - R[j, j]) for j in 0..T-2

        Negative BWT → forgetting old tasks.
        Positive BWT → learning new tasks improves old ones (rare).
        """
        T   = self.n_tasks
        bwt = []
        for j in range(T - 1):
            if not np.isnan(self.R[T-1, j]) and not np.isnan(self.R[j, j]):
                bwt.append(self.R[T-1, j] - self.R[j, j])
        return float(np.mean(bwt)) if bwt else 0.0

    def forward_transfer(self, random_baseline: Optional[np.ndarray] = None) -> float:
        """
        FWT = mean of (R[i-1, i] - b[i]) for i in 1..T-1

        Where b[i] is the random baseline accuracy for task i.
        Positive FWT → learning earlier tasks helps future tasks.
        """
        T   = self.n_tasks
        fwt = []
        for i in range(1, T):
            if not np.isnan(self.R[i-1, i]):
                baseline = (1.0 / max(i, 1)) if random_baseline is None \
                           else float(random_baseline[i])
                fwt.append(self.R[i-1, i] - baseline)
        return float(np.mean(fwt)) if fwt else 0.0

    def forgetting(self) -> float:
        """
        Forgetting = mean over j of (max_{i<=T-1} R[i,j] - R[T-1,j])

        Maximum accuracy seen on each old task minus its final accuracy.
        """
        T          = self.n_tasks
        forgetting = []
        for j in range(T - 1):
            col = self.R[:T, j]
            valid = col[~np.isnan(col)]
            if len(valid) >= 2:
                forgetting.append(float(np.max(valid) - valid[-1]))
        return float(np.mean(forgetting)) if forgetting else 0.0

    def intransigence(self) -> float:
        """
        Intransigence = mean of (R_joint[j] - R[j, j])

        How much worse the model is at learning new tasks compared to a
        joint-training upper bound. Requires joint_upper_bound to be set.
        Lower is better.

        Note: computed only if joint diagonal was recorded (R[j,j] must exist).
        """
        diag = [self.R[j, j] for j in range(self.n_tasks) if not np.isnan(self.R[j, j])]
        return float(np.mean(diag)) if diag else 0.0

    def final_average_accuracy(self) -> float:
        """Mean accuracy across all tasks after training on the last task."""
        T    = self.n_tasks
        last = self.R[T-1, :]
        valid = last[~np.isnan(last)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=== Continual Learning Results ===",
            f"  Final average accuracy : {self.final_average_accuracy():.4f}",
            f"  Backward Transfer (BWT): {self.backward_transfer():.4f}",
            f"  Forgetting             : {self.forgetting():.4f}",
            f"  Forward Transfer (FWT) : {self.forward_transfer():.4f}",
            "",
            "  Results matrix R[i, j] (row=after task i, col=task j):",
        ]
        header = "         " + "  ".join([f"T{j}" for j in range(self.n_tasks)])
        lines.append(header)
        for i in range(self.n_tasks):
            row = "  ".join([
                f"{self.R[i,j]:.3f}" if not np.isnan(self.R[i,j]) else "  --- "
                for j in range(self.n_tasks)
            ])
            lines.append(f"  T{i}:  {row}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anticipation metrics
# ---------------------------------------------------------------------------

def anticipation_accuracy_by_observation(y_true: np.ndarray,
                                          y_pred_dict: Dict[float, np.ndarray]
                                          ) -> Dict[float, float]:
    """
    Compute prediction accuracy at different observation ratios.

    Args:
        y_true:       (N,) ground-truth next-activity labels
        y_pred_dict:  {obs_ratio: (N,) predicted labels}
                      e.g. {0.25: preds_25, 0.50: preds_50, 0.75: preds_75}

    Returns:
        {obs_ratio: accuracy}
    """
    return {
        ratio: float((y_true == preds).mean())
        for ratio, preds in y_pred_dict.items()
    }
