from pathlib import Path
import hashlib
import logging
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

# cache pre-processed Layer-A/B filtered data
CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / ".cache"

def would_reach_layer_c(layer_a_result, layer_b_result):
    # Layer B hard-blocks never reach Layer C.
    if getattr(layer_b_result, "verdict", None) == "block":
        return False

    # SAFE allowlisting allows early exit only when Layer A is not suspicious.
    if (not getattr(layer_a_result, "suspicious", False)) and getattr(layer_b_result, "allowlisted", False):
        return False

    return True

def _cache_key(csv_path):
    """Produce a deterministic cache key from the CSV path + file content hash."""
    p = Path(csv_path)
    h = hashlib.md5(usedforsecurity=False)
    h.update(str(p.resolve()).encode())
    # Hash on file size + first/last 8 KB to avoid reading the whole file
    stat = p.stat()
    h.update(str(stat.st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(8192))
        if stat.st_size > 8192:
            f.seek(-8192, 2)
            h.update(f.read(8192))
    return h.hexdigest()


def load_data(csv_path, *, use_cache= True):
    """Load dataset and run Layer A + B filtering.

    Results are cached to disk so subsequent runs skip the expensive
    per-row Layer-A/B inference. The cache is invalidated automatically
    when the source CSV changes.
    """
    from core.layer_a.pipeline import analyze_text
    from core.layer_b.signature_engine import SignatureEngine

    cache_path = None
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(csv_path)
        cache_path = CACHE_DIR / f"filtered_{key}.parquet"
        if cache_path.exists():
            log.info("Loading cached filtered data from %s", cache_path)
            print(f"[cache] Loading pre-filtered data from {cache_path.name}")
            used_df = pd.read_parquet(cache_path)
            X = used_df["processed_text"]
            y = used_df["label"]
            return X, y, used_df

    #Full pass through Layer A + B
    df = pd.read_csv(csv_path)
    y_all = df["label"].astype(int)

    layer_c_results = []
    signature_engine = SignatureEngine()

    allowlisted_allow = 0
    non_allowlisted_allow = 0

    for i in tqdm(range(len(df)), desc="Layer A+B filtering", unit="row"):
        layer_a_result = analyze_text(df["text"].iloc[i])
        layer_b_result = signature_engine.detect(layer_a_result.processed_text)

        if would_reach_layer_c(layer_a_result, layer_b_result):
            layer_c_results.append((layer_a_result.processed_text, y_all[i]))

            if layer_b_result.verdict == "allow" and not getattr(layer_b_result, "allowlisted", False):
                non_allowlisted_allow += 1
            if layer_b_result.verdict == "allow" and getattr(layer_b_result, "allowlisted", False):
                allowlisted_allow += 1

    X = pd.Series([t for (t, _) in layer_c_results], name="processed_text")
    y = pd.Series([lab for (_, lab) in layer_c_results], name="label")
    used_df = pd.DataFrame({"processed_text": X, "label": y})

    log.info(
        "Layer A+B filtering: %d → %d rows (allowlisted_allow=%d, non_allowlisted_allow=%d)",
        len(df), len(used_df), allowlisted_allow, non_allowlisted_allow,
    )
    print(
        f"Filtering complete: {len(df)} → {len(used_df)} rows "
        f"(allowlisted_allow={allowlisted_allow}, non_allowlisted_allow={non_allowlisted_allow})"
    )

    # persist cache
    if use_cache and cache_path is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        used_df.to_parquet(cache_path, index=False)
        print(f"[cache] Saved filtered data to {cache_path.name}")

    return X, y, used_df