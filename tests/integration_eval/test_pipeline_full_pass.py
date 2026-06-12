import os
import pytest

if os.getenv("BARRIKADA_AUTO_DOWNLOAD_ARTIFACTS", "1") == "0":
    pytest.skip("Auto-download disabled for tests.", allow_module_level=True)

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.orchestrator import PIPipeline

project_root = Path(__file__).resolve().parents[2]


def run_full_pipeline_pass():
    import os
    df = pd.read_csv(project_root / "datasets" / "barrikade_test.csv")
    if not os.getenv("BARRIKADE_TEST_FULL_DATASET"):
        df = df.head(5)
    pipeline = PIPipeline()
    results = []

    for idx, row in df.iterrows():
        if idx % 200 == 0: #type: ignore 
            print(f"Processed {idx}/{len(df)}...")
        outcome = pipeline.detect(row["text"])
        results.append(
            {
                "text": row["text"],
                "true_label": int(row["label"]),
                "final_verdict": outcome.final_verdict.value,
                "decision_layer": outcome.decision_layer.value,
                "confidence_score": outcome.confidence_score,
                "total_processing_time_ms": outcome.total_processing_time_ms,
                "layer_a_time_ms": outcome.layer_a_time_ms,
                "layer_b_time_ms": outcome.layer_b_time_ms,
                "layer_c_time_ms": outcome.layer_c_time_ms,
                "layer_d_time_ms": outcome.layer_d_time_ms,
                "layer_e_time_ms": outcome.layer_e_time_ms,
            }
        )

    results_df = pd.DataFrame(results)
    result_dir = Path("test_results")
    result_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = result_dir / f"pipeline_full_pass_{ts}.csv"
    summary_path = result_dir / f"pipeline_full_pass_summary_{ts}.json"
    results_df.to_csv(csv_path, index=False)

    correct = int((
        ((results_df["true_label"] == 0) & (results_df["final_verdict"] == "allow"))
        | ((results_df["true_label"] == 1) & (results_df["final_verdict"].isin(["block", "flag"])))
    ).sum())
    summary = {
        "total_samples": len(results_df),
        "correct_predictions": correct,
        "accuracy_percent": (correct / len(results_df)) * 100 if len(results_df) else 0.0,
        "avg_total_time_ms": results_df["total_processing_time_ms"].mean(),
        "decision_layer_distribution": results_df["decision_layer"].value_counts().to_dict(),
        "final_verdict_distribution": results_df["final_verdict"].value_counts().to_dict(),
        "results_csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {csv_path}")
    print(f"Saved summary to {summary_path}")
    return summary

import pytest

@pytest.mark.slow
def test_pipeline_full_pass():
    summary = run_full_pipeline_pass()
    assert summary["total_samples"] > 0

if __name__ == "__main__":
    run_full_pipeline_pass()

