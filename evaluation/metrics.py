from typing import List, Dict, Any, Tuple
import math


class EvaluationMetrics:
    """
    Mathematical evaluation metric computer for classification accuracy and privacy compliance.
    Implements precision, recall, F1-scores, confusion matrices, and privacy leakage rates.
    """

    def calculate_classification_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        classes = sorted(list(set(y_true + y_pred)))
        total_samples = len(y_true)

        if total_samples == 0:
            return {
                'accuracy': 0.0,
                'macro_precision': 0.0,
                'macro_recall': 0.0,
                'macro_f1': 0.0,
                'per_class': {},
                'confusion_matrix': {}
            }

        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        accuracy = correct / total_samples

        confusion_matrix: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
        for yt, yp in zip(y_true, y_pred):
            confusion_matrix[yt][yp] += 1

        per_class: Dict[str, Dict[str, float]] = {}
        precisions = []
        recalls = []
        f1_scores = []

        for c in classes:
            tp = confusion_matrix[c][c]
            fp = sum(confusion_matrix[other][c] for other in classes if other != c)
            fn = sum(confusion_matrix[c][other] for other in classes if other != c)
            support = sum(confusion_matrix[c].values())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[c] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'support': support
            }

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        macro_precision = sum(precisions) / len(classes) if classes else 0.0
        macro_recall = sum(recalls) / len(classes) if classes else 0.0
        macro_f1 = sum(f1_scores) / len(classes) if classes else 0.0

        return {
            'accuracy': round(accuracy, 4),
            'macro_precision': round(macro_precision, 4),
            'macro_recall': round(macro_recall, 4),
            'macro_f1': round(macro_f1, 4),
            'per_class': per_class,
            'confusion_matrix': confusion_matrix
        }

    def calculate_privacy_metrics(self, true_labels: List[str], retained_flags: List[bool]) -> Dict[str, Any]:
        total = len(true_labels)
        if total == 0:
            return {
                'total_segments': 0,
                'private_segments': 0,
                'private_retained': 0,
                'privacy_leakage_rate': 0.0,
                'professional_retained': 0,
                'professional_retention_rate': 0.0
            }

        private_total = sum(1 for lbl in true_labels if lbl in ('personal', 'casual'))
        private_retained = sum(1 for lbl, ret in zip(true_labels, retained_flags) if lbl in ('personal', 'casual') and ret)

        prof_total = sum(1 for lbl in true_labels if lbl == 'professional')
        prof_retained = sum(1 for lbl, ret in zip(true_labels, retained_flags) if lbl == 'professional' and ret)

        privacy_leakage_rate = (private_retained / private_total) if private_total > 0 else 0.0
        prof_retention_rate = (prof_retained / prof_total) if prof_total > 0 else 0.0

        return {
            'total_segments': total,
            'private_segments': private_total,
            'private_retained': private_retained,
            'privacy_leakage_rate': round(privacy_leakage_rate, 4),
            'professional_segments': prof_total,
            'professional_retained': prof_retained,
            'professional_retention_rate': round(prof_retention_rate, 4)
        }

    def format_metrics_report(self, class_metrics: Dict[str, Any], privacy_metrics: Dict[str, Any]) -> str:
        lines = []
        lines.append("============================================================")
        lines.append("              MEETLYTIC BENCHMARK EVALUATION REPORT         ")
        lines.append("============================================================")
        lines.append("")
        lines.append("[1] CLASSIFICATION ACCURACY & F1 METRICS")
        lines.append(f"  Overall Accuracy        : {class_metrics['accuracy'] * 100:.2f}%")
        lines.append(f"  Macro-Averaged Precision: {class_metrics['macro_precision'] * 100:.2f}%")
        lines.append(f"  Macro-Averaged Recall   : {class_metrics['macro_recall'] * 100:.2f}%")
        lines.append(f"  Macro-Averaged F1-Score : {class_metrics['macro_f1'] * 100:.2f}%")
        lines.append("")
        lines.append("  Per-Class Performance:")
        lines.append("  {:<15} {:<12} {:<12} {:<12} {:<10}".format("Class", "Precision", "Recall", "F1-Score", "Support"))
        lines.append("  " + "-" * 58)
        for c, vals in class_metrics.get('per_class', {}).items():
            lines.append(f"  {c.upper():<15} {vals['precision'] * 100:>8.2f}%    {vals['recall'] * 100:>8.2f}%    {vals['f1_score'] * 100:>8.2f}%    {vals['support']:<10}")
        lines.append("")
        lines.append("[2] PRIVACY & ZERO-RETENTION METRICS")
        lines.append(f"  Total Turns Analyzed    : {privacy_metrics['total_segments']}")
        lines.append(f"  Private/Casual Turns    : {privacy_metrics['private_segments']}")
        lines.append(f"  Private Turns Leaked    : {privacy_metrics['private_retained']}")
        lines.append(f"  Privacy Leakage Rate    : {privacy_metrics['privacy_leakage_rate'] * 100:.2f}% (Target: 0.00%)")
        lines.append(f"  Professional Retained   : {privacy_metrics['professional_retained']}/{privacy_metrics['professional_segments']} ({privacy_metrics['professional_retention_rate'] * 100:.2f}%)")
        lines.append("============================================================")
        return "\n".join(lines)
