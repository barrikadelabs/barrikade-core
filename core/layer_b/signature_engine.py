import time
import hashlib
import json
import logging
import os
import platform
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

from core.settings import Settings
from models.SignatureMatch import SignatureMatch, Severity
from models.LayerBResult import LayerBResult

log = logging.getLogger(__name__)


class SignatureEngine:
    def __init__(self):
        self.settings = Settings()
        self._load_model()
        self._load_signatures()

    #init
    @staticmethod
    def _is_sentence_transformer_dir_ready(model_dir: Path) -> bool:
        required_files = [
            model_dir / "config.json",
            model_dir / "modules.json",
            model_dir / "tokenizer.json",
        ]
        weight_files = [
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
        ]
        return all(path.exists() for path in required_files) and any(
            path.exists() for path in weight_files
        )

    def _resolve_prompt_encoder_model(self) -> str:
        prompt_encoder_path = Path(self.settings.layer_b_signatures_dir) / "prompt_encoder"
        if not prompt_encoder_path.exists():
            log.info("Loading base embedding model: %s", self.settings.layer_b_embedding_model)
            return self.settings.layer_b_embedding_model

        if self._is_sentence_transformer_dir_ready(prompt_encoder_path):
            log.info("Loading trained prompt encoder: %s", prompt_encoder_path)
            return str(prompt_encoder_path)

        missing = []
        for file_name in ["config.json", "modules.json", "tokenizer.json", "model.safetensors|pytorch_model.bin"]:
            if file_name == "model.safetensors|pytorch_model.bin":
                if not any((prompt_encoder_path / name).exists() for name in ["model.safetensors", "pytorch_model.bin"]):
                    missing.append(file_name)
            elif not (prompt_encoder_path / file_name).exists():
                missing.append(file_name)
        log.warning(
            "Prompt encoder directory exists but is incomplete at %s; missing %s. "
            "Falling back to base embedding model %s.",
            prompt_encoder_path,
            ", ".join(missing) if missing else "required files",
            self.settings.layer_b_embedding_model,
        )
        return self.settings.layer_b_embedding_model

    @staticmethod
    def _is_onnx_encoder_dir_ready(model_dir: Path) -> bool:
        required_files = [
            model_dir / "config.json",
            model_dir / "modules.json",
            model_dir / "tokenizer.json",
            model_dir / "onnx" / "model.onnx",
        ]
        return all(path.exists() for path in required_files)

    def _load_model(self):
        # Prefer the ONNX-converted encoder when it's available alongside
        # the PT prompt_encoder. On CPU (production deployment target),
        # SentenceTransformer with backend="onnx" runs ~2.5x faster per
        # single-sample request than the PT path. Produced by
        # core/layer_b/export_layer_b_onnx.py.
        #
        # onnxruntime's execution provider handles device selection
        # internally, so we don't pass device= when using backend="onnx".
        onnx_dir = Path(self.settings.layer_b_signatures_dir) / "prompt_encoder_onnx"
        if onnx_dir.exists() and self._is_onnx_encoder_dir_ready(onnx_dir):
            log.info("Loading ONNX prompt encoder: %s", onnx_dir)
            self.model = SentenceTransformer(
                str(onnx_dir),
                backend="onnx",
                model_kwargs={"providers": ["CPUExecutionProvider"]}, #Inference is CPU only
            )
            return

        # PT backend fallback (fine-tuned local encoder or HF Hub base model)
        model_name = self._resolve_prompt_encoder_model()
        device = self._select_device()
        log.info("Layer B encoder device: %s", device)
        self.model = SentenceTransformer(model_name, device=device)

    def _select_device(self) -> str:
        forced = os.getenv("BARRIKADA_EMBEDDING_DEVICE", "").strip().lower()
        if forced in {"cpu", "cuda", "mps"}:
            return forced

        # # MPS can be unstable under Streamlit reruns on some macOS setups.
        # if platform.system() == "Darwin":
        #     return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _select_index_backend(self) -> str:
        forced = os.getenv("BARRIKADA_LAYER_B_INDEX_BACKEND", "").strip().lower()
        if forced in {"faiss", "sklearn"}:
            return forced

        # # faiss + torch can segfault on some macOS runtimes.
        # if platform.system() == "Darwin":
        #     return "sklearn"

        return "faiss"

    def _load_signatures(self):
        sig = Path(self.settings.layer_b_signatures_dir)
        self.index_backend = self._select_index_backend()

        # Attack artefacts
        attack_idx_path = sig / "faiss_index.bin"
        if self.index_backend == "faiss" and not attack_idx_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {attack_idx_path}. "
                "Run scripts/extract_signature_patterns.py first."
            )
        cpu_attack = None
        if self.index_backend == "faiss":
            import faiss
            self._faiss = faiss
            cpu_attack = faiss.read_index(str(attack_idx_path))
        self.attack_centroids = np.load(str(sig / "centroids.npy"))
        with open(sig / "metadata.json") as f:
            self.metadata = json.load(f)

        # Benign artefacts
        benign_idx_path = sig / "benign_faiss_index.bin"
        if self.index_backend == "faiss" and benign_idx_path.exists():
            cpu_benign = self._faiss.read_index(str(benign_idx_path))
            self.benign_centroids = np.load(str(sig / "benign_centroids.npy"))
        elif self.index_backend == "sklearn" and (sig / "benign_centroids.npy").exists():
            cpu_benign = None
            self.benign_centroids = np.load(str(sig / "benign_centroids.npy"))
        else:
            cpu_benign = None
            self.benign_centroids = None
            log.warning("No benign centroids found — contrastive scoring disabled.")

        # Cluster radii
        radii_path = sig / "cluster_radii.json"
        if radii_path.exists():
            with open(radii_path) as f:
                self.radii = {int(k): v for k, v in json.load(f).items()}
        else:
            self.radii = {}

        if self.index_backend == "faiss":
            self.attack_index = cpu_attack
            self.benign_index = cpu_benign
            log.info("Layer B index backend: faiss")
        else:
            self.attack_index = NearestNeighbors(metric="cosine", algorithm="brute")
            self.attack_index.fit(self.attack_centroids)
            if self.benign_centroids is not None:
                self.benign_index = NearestNeighbors(metric="cosine", algorithm="brute")
                self.benign_index.fit(self.benign_centroids)
            else:
                self.benign_index = None
            log.info("Layer B index backend: sklearn")

        n_attack = self.attack_centroids.shape[0]
        n_benign = self.benign_centroids.shape[0] if self.benign_centroids is not None else 0
        log.info("Loaded %d attack + %d benign centroids (dim=%d)",
                 n_attack, n_benign, self.attack_centroids.shape[1])

    def _search(self, index, query, k):
        if self.index_backend == "faiss":
            return index.search(query, k)

        distances, ids = index.kneighbors(query, n_neighbors=k)
        scores = 1.0 - distances
        return scores.astype(np.float32), ids.astype(np.int64)

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Expose the loaded encoder for reuse by other components."""
        return self.model

    # embedding helper
    def _embed(self, text):
        vec = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    # main detection method
    def detect(self, text):
        start = time.time()
        input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        query = self._embed(text)
        top_k = self.settings.layer_b_top_k

        # Attack similarity (top-k mean) 
        k_attack = min(top_k, self.attack_centroids.shape[0])
        atk_scores, atk_ids = self._search(self.attack_index, query, k_attack)
        atk_scores = atk_scores[0]  # shape (k_attack,)
        atk_ids = atk_ids[0]
        attack_sim = float(np.mean(atk_scores[:k_attack]))

        # Benign similarity (top-k mean) 
        if self.benign_index is not None and self.benign_centroids is not None:
            k_benign = min(top_k, self.benign_centroids.shape[0])
            ben_scores, _ = self._search(self.benign_index, query, k_benign)
            ben_scores = ben_scores[0]
            benign_sim = float(np.mean(ben_scores[:k_benign]))
        else:
            benign_sim = 0.0

        # Contrastive score (attack sim - benign sim) 
        contrastive = attack_sim - benign_sim

        # Build match objects
        matches = []

        cluster_meta = {c["cluster_id"]: c for c in self.metadata.get("clusters", [])}

        for rank, (score, idx) in enumerate(zip(atk_scores, atk_ids)):
            score_f = float(score)

            if score_f < 0.20:
                continue

            cid = int(idx)
            meta = cluster_meta.get(cid, {})
            samples = meta.get("sample_prompts", [])
            desc = samples[0][:100] if samples else f"cluster_{cid}"

            matches.append(SignatureMatch(
                rule_id=f"cluster_{cid}",
                severity=Severity.MALICIOUS,
                pattern="contrastive_embedding",
                matched_text=text[:200],
                start_pos=0,
                end_pos=len(text),
                rule_description=desc,
                tags=[f"cluster_{cid}", f"rank_{rank}",
                      f"atk_sim={attack_sim:.3f}", f"ben_sim={benign_sim:.3f}",
                      f"contrastive={contrastive:.3f}"],
                confidence=score_f,
            ))

        #  Two-threshold decision on mean top-k attack similarity 
        # block_threshold / flag_threshold are compared against attack_sim.
        # Contrastive guard: if benign similarity exceeds attack similarity at
        # the block boundary, demote to FLAG to avoid false positives.
        block_thr = self.settings.layer_b_block_threshold
        flag_thr = self.settings.layer_b_flag_threshold
        block_min_margin = self.settings.layer_b_block_min_margin

        if attack_sim >= block_thr:
            # Hard block only when attack dominates benign by a minimum margin.
            if contrastive >= block_min_margin:
                verdict = "block"
                confidence = self.settings.layer_b_block_confidence
            else:
                verdict = "flag"
                confidence = self.settings.layer_b_flag_confidence
        elif attack_sim >= flag_thr:
            # Conservative safe-recovery path for clear benign dominance in the flag band.
            if (
                self.settings.layer_b_enable_safe_recovery
                and attack_sim <= self.settings.layer_b_safe_recovery_max_attack_sim
                and benign_sim >= self.settings.layer_b_safe_recovery_min_benign_sim
                and contrastive <= self.settings.layer_b_safe_recovery_max_margin
            ):
                verdict = "allow"
                confidence = self.settings.layer_b_safe_confidence
            else:
                verdict = "flag"
                confidence = self.settings.layer_b_flag_confidence
        else:
            verdict = "allow"
            confidence = self.settings.layer_b_safe_confidence

        elapsed = (time.time() - start) * 1000

        return LayerBResult(
            input_hash=input_hash,
            processing_time_ms=elapsed,
            matches=matches,
            verdict=verdict,
            confidence_score=confidence,
            attack_similarity=attack_sim,
            benign_similarity=benign_sim,
            contrastive_margin=contrastive,
            allowlisted=False,
            allowlist_rules=[],
        )
