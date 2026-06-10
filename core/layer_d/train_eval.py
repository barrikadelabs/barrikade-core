import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

from core.settings import Settings
from core.layer_d.utils import (
    augment_with_hard_negatives,
    binary_report,
    load_layer_d_model,
    make_compute_metrics,
    pick_hard_negative_indices,
    predict_scores,
    route_to_label,
    tokenize_datasets,
    verdict_breakdown,
)

_settings = Settings()
SEED = _settings.layer_c_seed


class LayerDTrainer(Trainer):
    def __init__(self, *args, class_weights= None, focal_gamma= 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        losses = torch.nn.functional.cross_entropy(logits, labels, weight=weight, reduction="none")
        if self.focal_gamma > 0:
            pt = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            losses = losses * torch.pow(1.0 - pt.clamp(1e-6, 1.0), self.focal_gamma)

        loss = losses.mean()
        return (loss, outputs) if return_outputs else loss


def train_eval(X, y, model_out_dir, low=None, high=None):
    s = _settings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    split_train = s.layer_d_split_train
    split_val = s.layer_d_split_val
    split_test = s.layer_d_split_test
    if abs((split_train + split_val + split_test) - 1.0) > 1e-6:
        raise ValueError("Layer D splits must sum to 1.0")

    val_test_ratio = split_val + split_test
    test_ratio_of_temp = split_test / max(1e-12, val_test_ratio)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio_of_temp, stratify=y_temp, random_state=SEED
    )

    train_df = pd.DataFrame({"text": X_train.tolist(), "label": y_train.astype(int).tolist()})
    val_df = pd.DataFrame({"text": X_val.tolist(), "label": y_val.astype(int).tolist()})
    test_df = pd.DataFrame({"text": X_test.tolist(), "label": y_test.astype(int).tolist()})

    print(f"Loading tokenizer '{s.layer_d_model_id}' ...")
    tokenizer = AutoTokenizer.from_pretrained(s.layer_d_model_id)  # nosec B615

    print("Tokenizing train/val/test datasets ...")
    train_ds, val_ds, test_ds = tokenize_datasets(
        tokenizer,
        train_df,
        val_df,
        test_df,
        max_length=s.layer_d_max_length,
    )

    model_kwargs = {
        "num_labels": 2,
        "id2label": {0: "SAFE", 1: "INJECTION"},
        "label2id": {"SAFE": 0, "INJECTION": 1},
    }

    if torch.cuda.is_available() and s.layer_d_use_bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = load_layer_d_model(s, model_kwargs)

    if torch.cuda.is_available() and s.layer_d_use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(model_out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    low_threshold = float(s.layer_d_low_threshold if low is None else low)
    high_threshold = float(s.layer_d_high_threshold if high is None else high)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        #overwrite_output_dir=True,
        num_train_epochs=s.layer_d_num_train_epochs,
        learning_rate=s.layer_d_learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=s.layer_d_warmup_ratio,
        per_device_train_batch_size=s.layer_d_train_batch_size,
        per_device_eval_batch_size=s.layer_d_eval_batch_size,
        gradient_accumulation_steps=s.layer_d_gradient_accumulation_steps,
        gradient_checkpointing=s.layer_d_gradient_checkpointing,
        weight_decay=s.layer_d_weight_decay,
        max_grad_norm=1.0,
        bf16=(torch.cuda.is_available() and s.layer_d_use_bf16),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_security_score",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=100,
        dataloader_num_workers=s.layer_d_dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=SEED,
        report_to=[],
    )

    class_weights = torch.tensor(
        [1.0, float(s.layer_d_malicious_class_weight)],
        dtype=torch.float32,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = LayerDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(low_threshold, high_threshold),
        class_weights=class_weights,
        focal_gamma=s.layer_d_focal_gamma,
    )

    print("Training Layer D ModernBERT (stage 1) ...")
    trainer.train()

    hard_negative_meta = {
        "enabled": bool(s.layer_d_hard_negative_enabled),
        "stage2_ran": False,
        "used_routing_band": bool(s.layer_d_hard_negative_use_routing_band),
        "mine_band_min": None,
        "mine_band_max": None,
        "mined_count": 0,
        "min_samples": int(s.layer_d_hard_negative_min_samples),
        "max_samples": int(s.layer_d_hard_negative_max_samples),
        "augment_multiplier": int(s.layer_d_hard_negative_augment_multiplier),
        "train_rows_before": int(len(train_df)),
        "train_rows_after": int(len(train_df)),
    }

    if s.layer_d_hard_negative_enabled:
        print("Mining Layer D hard negatives from stage-1 train scores ...")
        train_scores = predict_scores(trainer, train_ds)
        hard_idx, mine_min, mine_max = pick_hard_negative_indices(
            y_train=y_train,
            train_scores=train_scores,
            low=low_threshold,
            high=high_threshold,
            use_routing_band=s.layer_d_hard_negative_use_routing_band,
            score_min=s.layer_d_hard_negative_score_min,
            score_max=s.layer_d_hard_negative_score_max,
            max_samples=s.layer_d_hard_negative_max_samples,
        )

        hard_negative_meta["mine_band_min"] = float(mine_min)
        hard_negative_meta["mine_band_max"] = float(mine_max)
        hard_negative_meta["mined_count"] = int(hard_idx.size)

        min_samples = max(0, int(s.layer_d_hard_negative_min_samples))
        if hard_idx.size >= min_samples and int(s.layer_d_hard_negative_augment_multiplier) > 0:
            print(f"Retraining Layer D with {hard_idx.size} hard negatives (stage 2) ...")
            train_df_aug, extra_rows = augment_with_hard_negatives(
                train_df,
                hard_idx,
                multiplier=int(s.layer_d_hard_negative_augment_multiplier),
            )
            hard_negative_meta["train_rows_after"] = int(len(train_df_aug))

            train_ds_aug, _, _ = tokenize_datasets(
                tokenizer,
                train_df_aug,
                val_df,
                test_df,
                max_length=s.layer_d_max_length,
            )

            model = load_layer_d_model(s, model_kwargs)
            trainer = LayerDTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds_aug,
                eval_dataset=val_ds,
                data_collator=data_collator,
                compute_metrics=make_compute_metrics(low_threshold, high_threshold),
                class_weights=class_weights,
                focal_gamma=s.layer_d_focal_gamma,
            )
            trainer.train()
            hard_negative_meta["stage2_ran"] = True
            hard_negative_meta["augmented_rows_added"] = int(extra_rows)
        else:
            print("Skipping stage 2 hard-negative retraining (not enough mined samples or multiplier <= 0).")

    print(f"Saving model artifacts to {output_dir} ...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Scoring validation and test sets ...")
    val_scores = predict_scores(trainer, val_ds)
    test_scores = predict_scores(trainer, test_ds)

    val_pred_05 = val_scores >= 0.5
    test_pred_05 = test_scores >= 0.5

    val_verdict, val_pred_route = route_to_label(val_scores, low=low_threshold, high=high_threshold)
    test_verdict, test_pred_route = route_to_label(test_scores, low=low_threshold, high=high_threshold)

    val_verdict_counts = pd.Series(val_verdict).value_counts().to_dict()
    test_verdict_counts = pd.Series(test_verdict).value_counts().to_dict()

    return {
        "model": {
            "type": "modernbert_sequence_classifier",
            "model_id": s.layer_d_model_id,
            "artifact_dir": str(output_dir),
            "max_length": s.layer_d_max_length,
        },
        "thresholds": {
            "low": low_threshold,
            "high": high_threshold,
        },
        "model_info": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "model_id": s.layer_d_model_id,
            "max_length": s.layer_d_max_length,
        },
        "hard_negative_mining": hard_negative_meta,
        "metrics": {
            "val": {
                "roc_auc": float(roc_auc_score(y_val, val_scores)),
                "report_0.5": binary_report(y_val, val_pred_05),
                "report_routing": binary_report(y_val, val_pred_route),
                "routing_verdict_counts": val_verdict_counts,
                "routing_verdict_by_label": verdict_breakdown(y_val.to_numpy(), val_verdict),
                "routing_f1": float(f1_score(y_val.to_numpy(), val_pred_route, zero_division=0)),
            },
            "test": {
                "roc_auc": float(roc_auc_score(y_test, test_scores)),
                "report_0.5": binary_report(y_test, test_pred_05),
                "report_routing": binary_report(y_test, test_pred_route),
                "routing_verdict_counts": test_verdict_counts,
                "routing_verdict_by_label": verdict_breakdown(y_test.to_numpy(), test_verdict),
                "routing_f1": float(f1_score(y_test.to_numpy(), test_pred_route, zero_division=0)),
            },
        },
    }
