import json
import sys
import time
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.layer_d.classifier import LayerDClassifier

REPORT_PATH = project_root / "test_results" / "layer_d_eval_latest.json"


def load_trained_thresholds_and_model_dir():
    report = json.loads(REPORT_PATH.read_text())
    t = report.get("thresholds", {})
    m = report.get("model", {})
    low = float(t.get("low", 0.05))
    high = float(t.get("high", 0.95))
    model_dir = m.get("artifact_dir")

    print(f"Loaded trained thresholds: low={low:.4f}, high={high:.4f}")
    print(f"Loading Layer D model from: {model_dir}")

    return low, high, model_dir


def load_test_data(csv_path):
    import os
    df = pd.read_csv(csv_path)
    if not os.getenv("BARRIKADE_TEST_FULL_DATASET"):
        df = df.head(5)
    return df["text"].tolist(), df["label"].tolist()


def evaluate_classifier(classifier, texts, labels):
    rows = []
    for idx, text in enumerate(texts):
        if idx % 500 == 0:
            print(f"Classifying {idx}/{len(texts)}...")
        res = classifier.predict(text)
        rows.append(
            {
                "true_label": int(labels[idx]),
                "verdict": res.verdict,
                "probability": float(res.probability_score),
            }
        )

    df = pd.DataFrame(rows)
    total = len(df)

    safe_allow = int(((df["true_label"] == 0) & (df["verdict"] == "allow")).sum())
    safe_flag = int(((df["true_label"] == 0) & (df["verdict"] == "flag")).sum())
    safe_block = int(((df["true_label"] == 0) & (df["verdict"] == "block")).sum())
    mal_allow = int(((df["true_label"] == 1) & (df["verdict"] == "allow")).sum())
    mal_flag = int(((df["true_label"] == 1) & (df["verdict"] == "flag")).sum())
    mal_block = int(((df["true_label"] == 1) & (df["verdict"] == "block")).sum())

    n_safe = safe_allow + safe_flag + safe_block
    n_mal = mal_allow + mal_flag + mal_block

    print("\n" + "=" * 68)
    print("CONFUSION MATRIX")
    print("=" * 68)
    print(f"{'Ground Truth':<15} | {'Allow':>10} | {'Flag':>10} | {'Block':>10} | {'Total':>10}")
    print("-" * 68)
    print(f"{'SAFE':<15} | {safe_allow:>10} | {safe_flag:>10} | {safe_block:>10} | {n_safe:>10}")
    print(f"{'MALICIOUS':<15} | {mal_allow:>10} | {mal_flag:>10} | {mal_block:>10} | {n_mal:>10}")
    print("-" * 68)

    tot_allow = safe_allow + mal_allow
    tot_flag = safe_flag + mal_flag
    tot_block = safe_block + mal_block
    print(f"{'Total':<15} | {tot_allow:>10} | {tot_flag:>10} | {tot_block:>10} | {total:>10}")
    print("=" * 68)

    tp = mal_block
    fp = safe_block
    fn = mal_allow
    tn = safe_allow
    binary_total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / binary_total if binary_total else 0.0

    malicious_escape_rate = fn / n_mal if n_mal else 0.0
    safe_block_rate = safe_block / n_safe if n_safe else 0.0
    safe_flag_rate = safe_flag / n_safe if n_safe else 0.0
    flag_rate = tot_flag / total if total else 0.0

    print(f"\nAccuracy:  {accuracy:.4f} ")
    print(f"Precision: {precision:.4f}  (of all blocked, fraction truly malicious)")
    print(f"Recall:    {recall:.4f}  (of all malicious allow/block decisions, fraction blocked)")
    print(f"F1:        {f1:.4f}")

    print("\nSecurity rates")
    print(f"Malicious escape rate (allowed malicious / all malicious): {malicious_escape_rate:.4f}  ({fn}/{n_mal})")
    print(f"Safe block rate       (blocked safe / all safe):           {safe_block_rate:.4f}  ({safe_block}/{n_safe})")
    print(f"Safe flag rate        (flagged safe / all safe):           {safe_flag_rate:.4f}  ({safe_flag}/{n_safe})")
    print(f"Overall flag rate     (flagged / total):                   {flag_rate:.4f}  ({tot_flag}/{total})")
    return {"total": total}


import pytest

@pytest.mark.slow
def test_layer_d():
    if not REPORT_PATH.exists():
        pytest.skip(f"Trained thresholds and model dir report not found at {REPORT_PATH}. Skipping Layer D evaluation test.")
    test_texts, true_labels = load_test_data(project_root / "datasets" / "barrikade_test.csv")

    low, high, model_dir = load_trained_thresholds_and_model_dir()
    classifier = LayerDClassifier(model_dir=model_dir, low=low, high=high)

    metrics = evaluate_classifier(classifier, test_texts, true_labels)
    assert metrics["total"] == len(test_texts)


if __name__ == "__main__":
    start_time = time.time()
    test_layer_d()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}s")
