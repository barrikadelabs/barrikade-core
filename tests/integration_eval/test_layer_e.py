import sys
import time
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.layer_e.local_judge import Qwen3GuardJudge # noqa: F401
from core.settings import Settings


TEST_PROMPT_LIMIT = 2000


def load_test_data(csv_path):
    import os
    df = pd.read_csv(csv_path)
    if not os.getenv("BARRIKADE_TEST_FULL_DATASET"):
        df = df.head(5)
    elif TEST_PROMPT_LIMIT > 0:
        df = df.head(TEST_PROMPT_LIMIT)
    return df["text"].tolist(), df["label"].tolist()


def build_judge():
    settings = Settings()
    print(f"Using Layer E model dir: {settings.layer_e_model_dir}")
    return Qwen3GuardJudge(
        model_dir=settings.layer_e_model_dir,
        model_name=settings.layer_e_model_dir,
        temperature=settings.layer_e_temperature,
        timeout_s=settings.layer_e_timeout_s,
        max_retries=settings.layer_e_max_retries,
        max_new_tokens=settings.layer_e_max_new_tokens,
        no_think_default=settings.layer_e_no_think_default,
    )


def evaluate_judge(judge, texts, labels):
    rows = []
    fallback_blocks = 0

    for idx, text in enumerate(texts):
        if idx % 25 == 0:
            print(f"Classifying {idx}/{len(texts)}...")

        out = judge.call_judge(text)
        is_fallback = out.rationale.lower().startswith("layer e fallback block")
        fallback_blocks += int(is_fallback)

        rows.append(
            {
                "true_label": int(labels[idx]),
                "decision": out.decision,
                "is_fallback": bool(is_fallback),
                "rationale": out.rationale,
            }
        )

    df = pd.DataFrame(rows)
    total = len(df)

    safe_allow = int(((df["true_label"] == 0) & (df["decision"] == "allow")).sum())
    safe_block = int(((df["true_label"] == 0) & (df["decision"] == "block")).sum())
    mal_allow = int(((df["true_label"] == 1) & (df["decision"] == "allow")).sum())
    mal_block = int(((df["true_label"] == 1) & (df["decision"] == "block")).sum())

    n_safe = safe_allow + safe_block
    n_mal = mal_allow + mal_block

    print("\n" + "=" * 58)
    print("CONFUSION MATRIX")
    print("=" * 58)
    print(f"{'Ground Truth':<15} | {'Allow':>10} | {'Block':>10} | {'Total':>10}")
    print("-" * 58)
    print(f"{'SAFE':<15} | {safe_allow:>10} | {safe_block:>10} | {n_safe:>10}")
    print(f"{'MALICIOUS':<15} | {mal_allow:>10} | {mal_block:>10} | {n_mal:>10}")
    print("-" * 58)
    tot_allow = safe_allow + mal_allow
    tot_block = safe_block + mal_block
    print(f"{'Total':<15} | {tot_allow:>10} | {tot_block:>10} | {total:>10}")
    print("=" * 58)

    tp = mal_block
    fp = safe_block
    fn = mal_allow
    tn = safe_allow

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    malicious_escape_rate = fn / n_mal if n_mal else 0.0
    safe_block_rate = safe_block / n_safe if n_safe else 0.0

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (of all blocked, fraction truly malicious)")
    print(f"Recall:    {recall:.4f}  (of all malicious, fraction blocked)")
    print(f"F1:        {f1:.4f}")
    print("\nSecurity rates")
    print(f"Malicious escape rate (allowed malicious / all malicious): {malicious_escape_rate:.4f}  ({fn}/{n_mal})")
    print(f"Safe block rate       (blocked safe / all safe):           {safe_block_rate:.4f}  ({safe_block}/{n_safe})")
    print(f"Fallback block count: {fallback_blocks}/{total}")
    return {"total": total}


import pytest

@pytest.mark.slow
def test_layer_e():
    test_texts, true_labels = load_test_data(project_root / "datasets" / "barrikade_test.csv")
    judge = build_judge()
    metrics = evaluate_judge(judge, test_texts, true_labels)
    assert metrics["total"] == len(test_texts)


if __name__ == "__main__":
    start_time = time.time()
    test_layer_e()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}s")
