import os
import time
from typing import Dict, Any, List, Tuple

from nlp.classifier import ConversationClassifier
from nlp.context import ContextEngine
from privacy.modes import PrivacyConfig, PrivacyMode
from privacy.filter import PrivacyFilter
from evaluation.metrics import EvaluationMetrics
from evaluation.test_data import TestDataGenerator


class Benchmark:
    """
    Comprehensive offline evaluation and benchmarking engine.
    Measures classification accuracy, macro F1, privacy leakage rate, and inference latency.
    """

    def __init__(self):
        self.classifier = ConversationClassifier()
        model_path = "config/trained_model.json"
        if os.path.exists(model_path):
            self.classifier.load_model(model_path)
        else:
            texts, labels = TestDataGenerator.get_full_labeled_corpus()
            self.classifier.train_on_corpus(texts, labels)

        self.context_engine = ContextEngine(window_size=5)
        self.metrics_calc = EvaluationMetrics()

    def run_classifier_benchmark(self) -> Tuple[Dict[str, Any], float]:
        texts, y_true = TestDataGenerator.get_full_labeled_corpus()
        y_pred = []

        start_time = time.perf_counter()
        for text in texts:
            res = self.classifier.classify(text)
            y_pred.append(res['classification'])
        duration = time.perf_counter() - start_time

        avg_latency_ms = (duration / len(texts)) * 1000 if texts else 0.0
        class_metrics = self.metrics_calc.calculate_classification_metrics(y_true, y_pred)
        return class_metrics, avg_latency_ms

    def run_privacy_benchmark(self, mode: str = "BALANCED") -> Dict[str, Any]:
        texts, y_true = TestDataGenerator.get_full_labeled_corpus()
        p_cfg = PrivacyConfig(mode=mode)
        p_filt = PrivacyFilter(p_cfg)

        retained_flags = []
        for text in texts:
            res = self.classifier.classify(text)
            action = p_filt.filter_turn(text, res['classification'], res['confidence'])
            retained_flags.append(action.get('retained', False))

        privacy_metrics = self.metrics_calc.calculate_privacy_metrics(y_true, retained_flags)
        return privacy_metrics

    def run_context_switching_benchmark(self) -> Dict[str, Any]:
        self.context_engine.reset()
        dialogue = TestDataGenerator.get_context_switching_dialogue()
        y_true = []
        y_pred = []

        for speaker, sentence, label in dialogue:
            y_true.append(label)
            tokens = [w.lower() for w in sentence.split()]
            boost = self.context_engine.get_context_boost(sentence, tokens)
            res = self.classifier.classify(sentence, context_boost=boost)
            y_pred.append(res['classification'])
            self.context_engine.update(sentence, res['classification'], res['confidence'], tokens)

        return self.metrics_calc.calculate_classification_metrics(y_true, y_pred)

    def run_full_benchmark(self) -> str:
        class_metrics, avg_latency = self.run_classifier_benchmark()
        privacy_metrics = self.run_privacy_benchmark(mode="BALANCED")
        strict_privacy = self.run_privacy_benchmark(mode="STRICT")
        context_metrics = self.run_context_switching_benchmark()

        lines = []
        lines.append("============================================================")
        lines.append("           MEETLYTIC BENCHMARK & EVALUATION SUITE           ")
        lines.append("============================================================")
        lines.append(f"  Classification Accuracy : {class_metrics['accuracy'] * 100:.2f}%")
        lines.append(f"  Macro-Averaged F1-Score : {class_metrics['macro_f1'] * 100:.2f}%")
        lines.append(f"  Average Turn Latency    : {avg_latency:.2f} ms")
        lines.append(f"  Balanced Privacy Leakage: {privacy_metrics['privacy_leakage_rate'] * 100:.2f}% (Target: 0.00%)")
        lines.append(f"  Strict Privacy Leakage  : {strict_privacy['privacy_leakage_rate'] * 100:.2f}% (Target: 0.00%)")
        lines.append(f"  Context Switching F1    : {context_metrics['macro_f1'] * 100:.2f}%")
        lines.append("")
        lines.append(self.metrics_calc.format_metrics_report(class_metrics, privacy_metrics))
        return "\n".join(lines)
