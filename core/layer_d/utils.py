import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from transformers import AutoModelForSequenceClassification, Trainer


def route_to_label(scores, low, high):
    verdict = np.full(scores.shape, "allow")
    verdict[(scores >= low) & (scores < high)] = "flag"
    verdict[scores >= high] = "block"
    predicted_label = (verdict != "allow").astype(int)
    return verdict, predicted_label


def binary_report(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=False)


def verdict_breakdown(y_true, verdict):
    y = np.asarray(y_true).astype(int)
    v = np.asarray(verdict)
    out = {
        "allow": {"0": 0, "1": 0},
        "flag": {"0": 0, "1": 0},
        "block": {"0": 0, "1": 0},
    }
    for label in (0, 1):
        for decision in ("allow", "flag", "block"):
            out[decision][str(label)] = int(np.sum((y == label) & (v == decision)))
    return out


def make_compute_metrics(low, high):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        scores = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        labels_arr = np.asarray(labels).astype(int)
        preds = (scores >= 0.5).astype(int)

        tp = int(np.sum((preds == 1) & (labels_arr == 1)))
        fp = int(np.sum((preds == 1) & (labels_arr == 0)))
        fn = int(np.sum((preds == 0) & (labels_arr == 1)))

        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        acc = float(np.mean(preds == labels_arr))

        verdict, routed = route_to_label(scores, low=low, high=high)
        routed_tp = int(np.sum((routed == 1) & (labels_arr == 1)))
        routed_fp = int(np.sum((routed == 1) & (labels_arr == 0)))
        routed_fn = int(np.sum((routed == 0) & (labels_arr == 1)))
        routed_precision = routed_tp / max(1, routed_tp + routed_fp)
        routed_recall = routed_tp / max(1, routed_tp + routed_fn)
        routed_f1 = 2 * routed_precision * routed_recall / max(1e-12, routed_precision + routed_recall)
        safe_fpr = float(np.mean((verdict != "allow") & (labels_arr == 0)))
        mal_allow = float(np.mean((verdict == "allow") & (labels_arr == 1)) / max(1e-12, np.mean(labels_arr == 1)))

        metrics = {
            "accuracy": acc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": float(average_precision_score(labels_arr, scores)),
            "routing_precision": float(routed_precision),
            "routing_recall": float(routed_recall),
            "routing_f1": float(routed_f1),
            "routing_safe_fpr": safe_fpr,
            "routing_malicious_allow_rate": mal_allow,
            "routing_low": float(low),
            "routing_high": float(high),
            "security_score": float(routed_f1 - (0.5 * mal_allow) - (0.25 * safe_fpr)),
        }

        try:
            metrics["roc_auc"] = float(roc_auc_score(labels_arr, scores))
        except ValueError:
            pass
        return metrics

    return compute_metrics


def tokenize_datasets(tokenizer, train_df, val_df, test_df, max_length):
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_ds = train_ds.map(tokenize, batched=True, num_proc=4)
    val_ds = val_ds.map(tokenize, batched=True, num_proc=4)
    test_ds = test_ds.map(tokenize, batched=True, num_proc=4)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_ds, val_ds, test_ds


def predict_scores(trainer, ds):
    pred = trainer.predict(ds)
    logits = pred.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return probs


def pick_hard_negative_indices(y_train, train_scores, low, high, use_routing_band, score_min, score_max, max_samples, ):
    y = np.asarray(y_train).astype(int)
    scores = np.asarray(train_scores, dtype=float)

    mine_min, mine_max = (float(low), float(high)) if use_routing_band else (float(score_min), float(score_max))
    if mine_min > mine_max:
        mine_min, mine_max = mine_max, mine_min

    safe_mask = y == 0
    band_mask = (scores >= mine_min) & (scores < mine_max)
    candidate_idx = np.where(safe_mask & band_mask)[0]

    if candidate_idx.size == 0:
        return np.array([], dtype=int), mine_min, mine_max

    ranked = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
    capped_max = int(max(0, max_samples))
    if capped_max > 0:
        ranked = ranked[:capped_max]

    return ranked.astype(int), mine_min, mine_max


def augment_with_hard_negatives(train_df, hard_idx, multiplier):
    if hard_idx.size == 0 or multiplier <= 0:
        return train_df, 0

    extra_idx = np.tile(hard_idx, int(multiplier))
    augmented = pd.concat([train_df, train_df.iloc[extra_idx]], ignore_index=True)
    return augmented, int(extra_idx.size)


def load_layer_d_model(settings, model_kwargs):
    print("Loading Layer D ModernBERT...")
    return AutoModelForSequenceClassification.from_pretrained(  # nosec B615
        settings.layer_d_model_id,
        attn_implementation="sdpa",
        **model_kwargs,
    )
