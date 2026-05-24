import sys
from pathlib import Path
import io
from contextlib import redirect_stdout

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.layer_c.classifier import Classifier
from core.layer_a.pipeline import analyze_text
from core.layer_b.signature_engine import SignatureEngine
from core.settings import Settings
import pandas as pd

ARTIFACTS = {
    "model_path": "core/models/layer_c/classifier.joblib",
}


def load_thresholds():
    settings = Settings()
    low = settings.layer_c_low_threshold
    high = settings.layer_c_high_threshold
    print(f"Using manual thresholds from settings: low={low:.4f}, high={high:.4f}")
    return low, high


def load_test_data(csv_path):
    import os
    df = pd.read_csv(csv_path)
    if not os.getenv("BARRIKADE_TEST_FULL_DATASET"):
        df = df.head(5)
    return df["text"].tolist(), df["label"].tolist()


def filter_through_layer_b(texts, labels):
    layer_b = SignatureEngine()
    
    flagged_texts = []
    flagged_labels = []
        
    for idx, (text, label) in enumerate(zip(texts, labels)):
        if idx % 500 == 0:
            print(f"Processing {idx}/{len(texts)}...")
        
        # Layer A: Preprocess text
        with redirect_stdout(io.StringIO()):
            layer_a_result = analyze_text(text)
        
        # Layer B: Signature detection
        layer_b_result = layer_b.detect(layer_a_result.processed_text)
                
        # Only pass "flag" verdicts to classifier
        if layer_b_result.verdict == "flag":
            flagged_texts.append(layer_a_result.processed_text)
            flagged_labels.append(label)
    
    return flagged_texts, flagged_labels


def evaluate_classifier(classifier, texts, labels):
    results = []
    for idx, text in enumerate(texts):
        result = classifier.predict(text)
        results.append({
            'true_label': labels[idx],
            'verdict': result.verdict,
            'probability': result.probability_score,
        })

    df = pd.DataFrame(results)
    total = len(df)

    #Confusion matrix (3-way routing)
    safe_allow = int(((df['true_label'] == 0) & (df['verdict'] == 'allow')).sum())
    safe_flag  = int(((df['true_label'] == 0) & (df['verdict'] == 'flag')).sum())
    safe_block = int(((df['true_label'] == 0) & (df['verdict'] == 'block')).sum())
    mal_allow  = int(((df['true_label'] == 1) & (df['verdict'] == 'allow')).sum())
    mal_flag   = int(((df['true_label'] == 1) & (df['verdict'] == 'flag')).sum())
    mal_block  = int(((df['true_label'] == 1) & (df['verdict'] == 'block')).sum())
    n_safe = safe_allow + safe_flag + safe_block
    n_mal  = mal_allow + mal_flag + mal_block

    print("\n" + "=" * 68)
    print("CONFUSION MATRIX")
    print("=" * 68)
    print(f"{'Ground Truth':<15} | {'Allow':>10} | {'Flag':>10} | {'Block':>10} | {'Total':>10}")
    print("-" * 68)
    print(f"{'SAFE':<15} | {safe_allow:>10} | {safe_flag:>10} | {safe_block:>10} | {n_safe:>10}")
    print(f"{'MALICIOUS':<15} | {mal_allow:>10} | {mal_flag:>10} | {mal_block:>10} | {n_mal:>10}")
    print("-" * 68)
    tot_allow = safe_allow + mal_allow
    tot_flag  = safe_flag + mal_flag
    tot_block = safe_block + mal_block
    print(f"{'Total':<15} | {tot_allow:>10} | {tot_flag:>10} | {tot_block:>10} | {total:>10}")
    print("=" * 68)

    # security metrics
    tp = mal_block                     # malicious correctly blocked
    fp = safe_block                    # safe incorrectly blocked
    fn = mal_allow                     # malicious incorrectly allowed
    tn = safe_allow                    # safe correctly allowed
    binary_total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / binary_total if binary_total else 0.0

    malicious_escape_rate = fn / n_mal if n_mal else 0.0
    safe_block_rate       = safe_block / n_safe if n_safe else 0.0
    safe_flag_rate        = safe_flag / n_safe if n_safe else 0.0
    flag_rate             = tot_flag / total if total else 0.0

    print(f"\nAccuracy:  {accuracy:.4f}  (allow/block only; flag excluded)")
    print(f"Precision: {precision:.4f}  (of all blocked, fraction truly malicious)")
    print(f"Recall:    {recall:.4f}  (of all malicious allow/block decisions, fraction blocked)")
    print(f"F1:        {f1:.4f}")

    print("\nSecurity rates")
    print(f"Malicious escape rate (allowed malicious / all malicious): {malicious_escape_rate:.4f}  ({fn}/{n_mal})")
    print(f"Safe block rate       (blocked safe / all safe):           {safe_block_rate:.4f}  ({safe_block}/{n_safe})")
    print(f"Safe flag rate        (flagged safe / all safe):           {safe_flag_rate:.4f}  ({safe_flag}/{n_safe})")
    print(f"Overall flag rate     (flagged / total):                   {flag_rate:.4f}  ({tot_flag}/{total})")
    return {
        "total": total,
        "n_safe": n_safe,
        "n_mal": n_mal,
        "tot_allow": tot_allow,
        "tot_flag": tot_flag,
        "tot_block": tot_block,
    }


import pytest

@pytest.mark.slow
def test_layer_c():
    test_texts, true_labels = load_test_data(project_root / "datasets" / "barrikade_test.csv")
    
    # Use manual thresholds from settings.
    low, high = load_thresholds()
    
    classifier = Classifier(
        **ARTIFACTS,
        low=low,
        high=high,
    )
    
    # Evaluate with trained thresholds on the full test set
    metrics = evaluate_classifier(classifier, test_texts, true_labels)
    assert metrics["total"] == len(test_texts)
    assert metrics["total"] > 0


if __name__ == "__main__":
    import time
    start_time = time.time()
    test_layer_c()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}s")
