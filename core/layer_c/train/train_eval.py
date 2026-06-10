from pathlib import Path
import hashlib

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from core.layer_c.train.make_model import make_model
from core.layer_c.train.utils import (
    augment_with_hard_negatives,
    build_split_metrics,
    embedding_stats,
    pick_hard_negative_indices,
    top_feature_importance,
)
from core.settings import Settings

_settings = Settings()
SEED = _settings.layer_c_seed

_EMB_CACHE_DIR = Path(__file__).resolve().parent / "outputs" / ".cache" / "embeddings"


def _emb_cache_path(texts, model_name):
    h = hashlib.md5(model_name.encode(), usedforsecurity=False)
    for t in texts:
        h.update(t.encode())
    return _EMB_CACHE_DIR / f"{h.hexdigest()}.npy"


def encode_texts(texts, model, batch_size=None, use_cache=True):
    if batch_size is None:
        batch_size = _settings.layer_c_embedding_batch_size
    texts_list = list(texts)
    cache_path = None

    if use_cache:
        _EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _emb_cache_path(texts_list, _settings.layer_c_embedding_model)
        if cache_path.exists():
            print(f"[emb cache] Loading cached embeddings from {cache_path.name}")
            return np.load(cache_path)

    emb = model.encode(texts_list, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    if use_cache and cache_path is not None:
        np.save(cache_path, emb)
        print(f"[emb cache] Saved embeddings to {cache_path.name}")

    return emb


def train_eval(X, y, low=None, high=None):
    s = _settings
    low = float(s.layer_c_low_threshold if low is None else low)
    high = float(s.layer_c_high_threshold if high is None else high)
    if low >= high:
        raise ValueError("Manual thresholds must satisfy low < high")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=s.layer_c_val_test_size, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=s.layer_c_test_split, stratify=y_temp, random_state=SEED
    )

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer '{s.layer_c_embedding_model}' on {_device} ...")
    encoder = SentenceTransformer(s.layer_c_embedding_model, device=_device)
    emb_dim = encoder.get_sentence_embedding_dimension()

    print("Encoding training texts ...")
    X_train_emb = encode_texts(X_train, encoder)
    print("Encoding validation texts ...")
    X_val_emb = encode_texts(X_val, encoder)
    print("Encoding test texts ...")
    X_test_emb = encode_texts(X_test, encoder)
    print(f"Embedding features: {emb_dim}")

    y_train_np = y_train.to_numpy().astype(int)
    pos = max(1, int(np.sum(y_train_np == 1)))
    neg = max(1, int(np.sum(y_train_np == 0)))
    scale_pos_weight = (neg / pos) * float(s.layer_c_xgb_scale_pos_multiplier)
    print(f"Training class balance: neg={neg}, pos={pos}, scale_pos_weight={scale_pos_weight:.4f}")

    model = make_model(scale_pos_weight=scale_pos_weight)
    print("Training XGBoost ...")
    model.fit(
        X_train_emb,
        y_train,
        eval_set=[(X_val_emb, y_val)],
        verbose=50,
    )
    if model.best_iteration is not None:
        print(f"Early stopping: best iteration = {model.best_iteration}")

    mined_idx, mine_min, mine_max = np.array([], dtype=int), float(low), float(high)
    train_scores_raw = model.predict_proba(X_train_emb)[:, 1]
    val_scores_stage1 = model.predict_proba(X_val_emb)[:, 1]
    train_scores_for_mining = np.asarray(train_scores_raw, dtype=float)
    stage1_calibrator = IsotonicRegression(out_of_bounds="clip")
    stage1_calibrator.fit(np.asarray(val_scores_stage1, dtype=float), y_val.to_numpy().astype(int))
    train_scores_for_mining = np.clip(
        np.asarray(stage1_calibrator.predict(train_scores_for_mining), dtype=float),
        0.0,
        1.0,
    )

    mined_idx, mine_min, mine_max = pick_hard_negative_indices(
        y_train,
        train_scores_for_mining,
        low,
        high,
        s.layer_c_hard_negative_use_routing_band,
        s.layer_c_hard_negative_score_min,
        s.layer_c_hard_negative_score_max,
        s.layer_c_hard_negative_max_samples,
    )
    min_needed = int(max(0, s.layer_c_hard_negative_min_samples))
    if mined_idx.size >= min_needed:
        mult = int(max(0, s.layer_c_hard_negative_augment_multiplier))
        X_train_aug, y_train_aug = augment_with_hard_negatives(X_train, y_train, mined_idx, mult)

        print(
            "Re-training XGBoost with hard-negative augmentation: "
            f"anchors={mined_idx.size}, multiplier={mult}, rows={len(X_train_aug)}"
        )
        X_train_emb_aug = encode_texts(X_train_aug, encoder, use_cache=False)

        y_train_aug_np = y_train_aug.to_numpy().astype(int)
        aug_pos = max(1, int(np.sum(y_train_aug_np == 1)))
        aug_neg = max(1, int(np.sum(y_train_aug_np == 0)))
        aug_scale_pos_weight = (aug_neg / aug_pos) * float(s.layer_c_xgb_scale_pos_multiplier)

        model = make_model(scale_pos_weight=aug_scale_pos_weight)
        model.fit(
            X_train_emb_aug,
            y_train_aug,
            eval_set=[(X_val_emb, y_val)],
            verbose=50,
        )
        scale_pos_weight = aug_scale_pos_weight
        X_train_emb = X_train_emb_aug
        X_train = X_train_aug
        y_train = y_train_aug
    else:
        print(
            "Skipping hard-negative retrain: "
            f"mined={mined_idx.size} < min_required={min_needed}"
        )

    model.set_params(device="cpu")

    val_scores_raw = model.predict_proba(X_val_emb)[:, 1]
    test_scores_raw = model.predict_proba(X_test_emb)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.asarray(val_scores_raw, dtype=float), y_val.to_numpy().astype(int))
    val_scores = np.clip(np.asarray(calibrator.predict(np.asarray(val_scores_raw, dtype=float))), 0.0, 1.0)
    test_scores = np.clip(np.asarray(calibrator.predict(np.asarray(test_scores_raw, dtype=float))), 0.0, 1.0)

    artifact = {
        "model": model,
        "calibrator": calibrator,
        "metadata": {
            "embedding_model": s.layer_c_embedding_model,
            "embedding_dim": emb_dim,
            "calibration_method": "isotonic",
            "scale_pos_weight": float(scale_pos_weight),
            "threshold_source": "manual_settings_or_cli",
            "hard_negative_mining_enabled": True,
            "hard_negative_mined_count": int(mined_idx.size),
            "hard_negative_score_min": float(mine_min),
            "hard_negative_score_max": float(mine_max),
            "hard_negative_use_routing_band": bool(s.layer_c_hard_negative_use_routing_band),
            "hard_negative_augment_multiplier": int(max(0, s.layer_c_hard_negative_augment_multiplier)),
        },
    }

    feature_importance_top = top_feature_importance(model)

    split_metrics_val = build_split_metrics(
        y_true=y_val,
        raw_scores=val_scores_raw,
        calibrated_scores=val_scores,
        low=low,
        high=high,
        cal_bins=s.layer_c_calibration_bins,
    )
    split_metrics_test = build_split_metrics(
        y_true=y_test,
        raw_scores=test_scores_raw,
        calibrated_scores=test_scores,
        low=low,
        high=high,
        cal_bins=s.layer_c_calibration_bins,
    )

    return {
        "artifact": artifact,
        "thresholds": {
            "low": low,
            "high": high,
            "source": "manual",
        },
        "embedding_info": {
            "model": s.layer_c_embedding_model,
            "dim": emb_dim,
            "train": embedding_stats(X_train_emb),
            "val": embedding_stats(X_val_emb),
            "test": embedding_stats(X_test_emb),
        },
        "feature_importance_top": feature_importance_top,
        "metrics": {
            "val": split_metrics_val,
            "test": split_metrics_test,
        },
    }
