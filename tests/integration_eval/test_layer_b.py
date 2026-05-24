import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import sys
import io
from contextlib import redirect_stdout

from core.layer_a.pipeline import analyze_text
from core.layer_b.signature_engine import SignatureEngine

import os
import pytest

@pytest.mark.slow
def test_layer_b():
    
    df = pd.read_csv(project_root / "datasets" / "barrikade_test.csv")
    if not os.getenv("BARRIKADE_TEST_FULL_DATASET"):
        df = df.head(5)
    
    print(f"Testing Layer B on {len(df)} samples...")
    layer_b = SignatureEngine()
    
    results = []
    
    for idx in range(len(df)):
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(df)}...")
        
        row = df.iloc[idx]
        
        # supress Layer A output
        with redirect_stdout(io.StringIO()):
            layer_a_result = analyze_text(row['text'])
        
        # Run Layer B on preprocessed text
        layer_b_result = layer_b.detect(layer_a_result.processed_text)

        # Three-verdict system: block / flag / allow
        if layer_b_result.verdict == "block":
            predicted_label = 1
        elif layer_b_result.verdict == "allow":
            predicted_label = 0
        else:  # flag → uncertain, keep ground truth
            predicted_label = row['label']
        
        # Correct if predicted matches ground truth
        is_correct = predicted_label == row['label']
        
        results.append({
            'text': row['text'],
            'preprocessed_text': layer_a_result.processed_text,
            'true_label': row['label'],
            'layer_b_matches': len(layer_b_result.matches),
            'layer_b_verdict': layer_b_result.verdict,
            'layer_b_confidence': layer_b_result.confidence_score,
            'layer_b_top_similarity': max((m.confidence for m in layer_b_result.matches), default=0.0),
            'layer_b_attack_similarity': layer_b_result.attack_similarity,
            'layer_b_benign_similarity': layer_b_result.benign_similarity,
            'layer_b_margin': layer_b_result.contrastive_margin,
            'predicted_label': predicted_label,
            'is_correct': is_correct,
            'processing_time_ms': layer_b_result.processing_time_ms,
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"core/layer_b/outputs/layer_b_results_{timestamp}.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Quick stats
    results_df = pd.DataFrame(results)

    # Verdict breakdown
    verdict_counts = (
        results_df["layer_b_verdict"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .value_counts()
    )
    total = verdict_counts.sum()

    
    # Confusion Matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    
    # Calculate confusion matrix values
    safe_allow = ((results_df['true_label'] == 0) & (results_df['layer_b_verdict'] == 'allow')).sum()
    safe_flag = ((results_df['true_label'] == 0) & (results_df['layer_b_verdict'] == 'flag')).sum()
    safe_block = ((results_df['true_label'] == 0) & (results_df['layer_b_verdict'] == 'block')).sum()
    
    malicious_allow = ((results_df['true_label'] == 1) & (results_df['layer_b_verdict'] == 'allow')).sum()
    malicious_flag = ((results_df['true_label'] == 1) & (results_df['layer_b_verdict'] == 'flag')).sum()
    malicious_block = ((results_df['true_label'] == 1) & (results_df['layer_b_verdict'] == 'block')).sum()
    
    # Calculate totals
    safe_total = safe_allow + safe_flag + safe_block
    malicious_total = malicious_allow + malicious_flag + malicious_block
    
    # Print table header
    print(f"{'Ground Truth':<15} | {'Allow':>10} | {'Flag':>10} | {'Block':>10} | {'Total':>10}")
    print("-" * 62)
    
    # Print SAFE row
    print(f"{'SAFE':<15} | {safe_allow:>10} | {safe_flag:>10} | {safe_block:>10} | {safe_total:>10}")
    
    # Print MALICIOUS row
    print(f"{'MALICIOUS':<15} | {malicious_allow:>10} | {malicious_flag:>10} | {malicious_block:>10} | {malicious_total:>10}")
    
    # Print total row
    allow_total = safe_allow + malicious_allow
    flag_total = safe_flag + malicious_flag
    block_total = safe_block + malicious_block
    print("-" * 62)
    print(f"{'Total':<15} | {allow_total:>10} | {flag_total:>10} | {block_total:>10} | {total:>10}")
    print("="*62)
    
    tp = ((results_df['predicted_label'] == 1) & (results_df['true_label'] == 1)).sum()
    fn = ((results_df['predicted_label'] == 0) & (results_df['true_label'] == 1)).sum()

    accuracy = (results_df['predicted_label'] == results_df['true_label']).mean()

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {tp /(tp+fn)}")

    # --- Classification quality metrics ---
    print("\n" + "="*60)
    print("CLASSIFICATION QUALITY")
    print("="*60)
    flag_rate = flag_total / total
    decisive_rate = 1 - flag_rate
    correct_decisive = malicious_block + safe_allow
    incorrect_decisive = safe_block + malicious_allow
    total_decisive = correct_decisive + incorrect_decisive
    decisive_accuracy = correct_decisive / total_decisive if total_decisive > 0 else 0

    print(f"Flag rate:              {flag_rate:.4f} ({flag_total}/{total})")
    print(f"Decisive rate:          {decisive_rate:.4f} ({total_decisive}/{total})")
    print(f"Decisive accuracy:      {decisive_accuracy:.4f} ({correct_decisive}/{total_decisive})")
    print(f"  Correct:   {correct_decisive}  (mal->block: {malicious_block}, safe->allow: {safe_allow})")
    print(f"  Incorrect: {incorrect_decisive}  (safe->block: {safe_block}, mal->allow: {malicious_allow})")

    # Block precision = malicious_block / total_block
    block_precision = malicious_block / block_total if block_total > 0 else 0
    print(f"\nBlock precision:         {block_precision:.4f} ({malicious_block}/{block_total})")

    # False block rate = safe_block / safe_total
    false_block_rate = safe_block / safe_total if safe_total > 0 else 0
    print(f"False block rate:        {false_block_rate:.4f} ({safe_block}/{safe_total})")

    # Similarity score stats
    print("\n" + "="*60)
    print("SIMILARITY SCORE STATS")
    print("="*60)
    print(f"Mean top similarity:  {results_df['layer_b_top_similarity'].mean():.4f}")
    print(f"Median:               {results_df['layer_b_top_similarity'].median():.4f}")
    print(f"Std:                  {results_df['layer_b_top_similarity'].std():.4f}")
    blocked = results_df[results_df['layer_b_verdict'] == 'block']
    flagged = results_df[results_df['layer_b_verdict'] == 'flag']
    allowed = results_df[results_df['layer_b_verdict'] == 'allow']
    if len(blocked) > 0:
        print(f"Blocked avg sim:      {blocked['layer_b_top_similarity'].mean():.4f}")
    if len(flagged) > 0:
        print(f"Flagged avg sim:      {flagged['layer_b_top_similarity'].mean():.4f}")
    if len(allowed) > 0:
        print(f"Allowed avg sim:      {allowed['layer_b_top_similarity'].mean():.4f}")

    print("\n" + "="*60)
    print("CALIBRATION TELEMETRY")
    print("="*60)
    print(f"Attack sim mean:      {results_df['layer_b_attack_similarity'].mean():.4f}")
    print(f"Benign sim mean:      {results_df['layer_b_benign_similarity'].mean():.4f}")
    print(f"Margin mean:          {results_df['layer_b_margin'].mean():.4f}")
    print(f"Margin median:        {results_df['layer_b_margin'].median():.4f}")
    assert len(results) == len(df)
    assert total == len(df)

if __name__ == "__main__":
    import time
    start_time = time.time()
    test_layer_b()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}s")
