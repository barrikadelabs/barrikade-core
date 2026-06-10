import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    # Package root works for both editable and wheel installs.
    _package_root = Path(__file__).resolve().parent
    _repo_root = _package_root.parent
    _user_state_root = Path.home() / ".barrikade"

    @staticmethod
    def _env_path(env_var):
        value = os.getenv(env_var)
        if not value:
            return None
        return Path(value).expanduser().resolve()

    def _path_with_override(self, env_var, default_path):
        override = self._env_path(env_var)
        return str(override if override is not None else default_path)

    @staticmethod
    def _ensure_directory(path):
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _output_dir_with_override(self, env_var, default_path):
        path = self._env_path(env_var) or default_path
        return str(self._ensure_directory(path))

    def _output_file_with_override(self, env_var, default_path):
        path = self._env_path(env_var) or default_path
        self._ensure_directory(path.parent)
        return str(path)

    def _existing_path_with_override(
        self,
        env_var: str,
        candidates: list[Path],
        purpose: str,
    ):
        override = self._env_path(env_var)
        if override is not None:
            if override.exists():
                return str(override)
            raise FileNotFoundError(
                f"{purpose} override path from {env_var} does not exist: {override}"
            )

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        searched = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not locate {purpose}. Searched:\n{searched}\n"
            f"Set {env_var} to an explicit path if needed. "
            f"For automatic GCS downloads, see docs/MODEL_HOSTING.md"
        )

    def _default_results_dir(self) -> Path:
        repo_results = self._repo_root / "test_results"
        if repo_results.exists():
            return repo_results
        return self._user_state_root / "test_results"

    def _default_layer_e_output_dir(self) -> Path:
        repo_outputs = self._package_root / "layer_e" / "outputs"
        if repo_outputs.exists() and os.access(repo_outputs, os.W_OK):
            return repo_outputs
        return self._user_state_root / "layer_e" / "outputs"

    @property
    def core_models_dir(self) -> str:
        return self._path_with_override(
            "BARRIKADA_CORE_MODELS_DIR",
            self._package_root / "models",
        )

    @property
    def bundle_root_dir(self) -> str:
        return self._path_with_override(
            "BARRIKADA_BUNDLE_DIR",
            self._user_state_root / "bundle",
        )

    @property
    def artifacts_root_dir(self) -> str:
        default_root = Path(self.bundle_root_dir)
        return self._path_with_override(
            "BARRIKADA_ARTIFACTS_DIR",
            default_root,
        )

    @property
    def bundle_manifest_path(self) -> str:
        default_path = Path(self.bundle_root_dir) / "manifest.json"
        return self._path_with_override(
            "BARRIKADA_BUNDLE_MANIFEST_PATH",
            default_path,
        )

    @property
    def max_download_workers(self) -> int:
        val = os.getenv("BARRIKADA_MAX_DOWNLOAD_WORKERS")
        if val:
            try:
                return int(val)
            except ValueError:
                pass
        return 8

    # Models are centrally managed under `core/models/` (see
    # `BARRIKADA_CORE_MODELS_DIR` override). For hosting details,
    # refer to `docs/MODEL_HOSTING.md`.

    ### Telemetry & Audit Configuration
    telemetry_enabled: bool = True
    telemetry_log_path: str = "test_results/barrikade_telemetry.jsonl"
    telemetry_safe_sample_rate: float = 1.0

    ### Layer B (embedding-based contrastive signature engine)
    layer_b_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Two-threshold decision system (applied to mean top-k attack similarity)
    # Empirically calibrated on barrikade_test.csv (block_prec=0.96, fblk=0.53%)
    layer_b_block_threshold: float = 0.86   # sweep-calibrated conservative block threshold
    layer_b_flag_threshold: float = 0.62    # sweep-calibrated conservative flag threshold
    # Below flag_threshold → SAFE (allow)

    # Contrastive guardrails for safer Layer B calibration
    layer_b_block_min_margin: float = 0.08  # require strong attack dominance for hard block
    layer_b_enable_safe_recovery: bool = True
    layer_b_safe_recovery_max_attack_sim: float = 0.63
    layer_b_safe_recovery_min_benign_sim: float = 0.80
    layer_b_safe_recovery_max_margin: float = -0.10  # allow only when benign clearly dominates

    # Top-k similarity aggregation
    layer_b_top_k: int = 5

    # Cluster building
    layer_b_n_clusters: int = 64
    layer_b_min_cluster_purity: float = 0.70  # drop clusters below this purity
    layer_b_purity_proximity: float = 0.70     # only count benign prompts with sim >= this

    # Confidence values emitted in LayerBResult
    layer_b_block_confidence: float = 0.95
    layer_b_flag_confidence: float = 0.50
    layer_b_safe_confidence: float = 0.10

    # Dual-encoder contrastive training hyperparameters
    layer_b_dual_encoder_temperature: float = 0.05
    layer_b_dual_encoder_epochs: int = 3
    layer_b_dual_encoder_batch_size: int = 8
    layer_b_dual_encoder_lr: float = 2e-5
    layer_b_dual_encoder_hard_negatives: int = 3
    layer_b_dual_encoder_max_samples: int = 50000

    @property
    def layer_b_signatures_dir(self):
        return self._existing_path_with_override(
            "BARRIKADA_LAYER_B_SIGNATURES_DIR",
            self.layer_b_signatures_candidates,
            "Layer B signatures directory",
        )

    @property
    def layer_b_signatures_candidates(self) -> list[Path]:
        return [
            Path(self.core_models_dir) / "layer_b" / "embeddings",
            Path(self.bundle_root_dir) / "layer_b" / "embeddings",
            Path(self.artifacts_root_dir) / "layer_b" / "embeddings",
            self._package_root / "layer_b" / "signatures" / "embeddings",
        ]

    @property
    def layer_b_signatures_dirname(self) -> str:
        return str(Path(self.artifacts_root_dir) / "layer_b" / "embeddings")
    

    ### Layer C
    # Routing thresholds for Layer C classifier:
    layer_c_low_threshold: float = 0.05
    layer_c_high_threshold: float = 0.95

    layer_c_seed: int = 42

    layer_c_val_test_size: float = 0.30
    layer_c_test_split: float = 0.50

    layer_c_embedding_model: str = "all-mpnet-base-v2"
    layer_c_embedding_batch_size: int    = 128

    # Probability calibration (fixed to isotonic in training flow)
    layer_c_calibration_bins: int = 15

    # XGBoost configuration for Layer C
    layer_c_xgb_n_estimators: int = 3000
    layer_c_xgb_max_depth: int = 7
    layer_c_xgb_learning_rate: float = 0.05
    layer_c_xgb_subsample: float = 0.8
    layer_c_xgb_colsample_bytree: float = 0.9
    layer_c_xgb_scale_pos_multiplier: float = 1.5
    layer_c_xgb_early_stopping_rounds: int = 150
    layer_c_xgb_tree_method: str = "hist"
    layer_c_xgb_reg_alpha: float = 0.1
    layer_c_xgb_reg_lambda: float = 1.0
    layer_c_xgb_min_child_weight: int = 5
    layer_c_xgb_gamma: float = 0.1

    # Layer C hard-negative mining (train-split SAFE examples near/inside uncertain band)
    layer_c_hard_negative_use_routing_band: bool = True
    layer_c_hard_negative_score_min: float = 0.20
    layer_c_hard_negative_score_max: float = 0.80
    layer_c_hard_negative_max_samples: int = 5000
    layer_c_hard_negative_min_samples: int = 32
    layer_c_hard_negative_augment_multiplier: int = 1

    # Layer C model version selection
    # Model version selection now maps to `core/models/layer_c/` when
    # available. Older per-layer `outputs/` locations are accepted as
    # fallbacks for compatibility.
    layer_c_model_version: str = "legacy"

    @property
    def dataset_path(self):
        override = self._env_path("BARRIKADA_DATASET_PATH")
        if override is not None:
            return str(override)
        return str(self._repo_root / "datasets" / "barrikade.csv")

    @property
    def layer_c_release_dir(self):
        return self._path_with_override(
            "BARRIKADA_LAYER_C_RELEASE_DIR",
            Path(self.core_models_dir) / "layer_c" / "releases",
        )

    
    @property
    def model_path(self):
        legacy = self._package_root / "layer_c" / "outputs" / "classifier.joblib"
        return self._existing_path_with_override(
            "BARRIKADA_LAYER_C_MODEL_PATH",
            self.layer_c_model_candidates,
            "Layer C classifier model",
        )

    @property
    def layer_c_model_candidates(self) -> list[Path]:
        legacy = self._package_root / "layer_c" / "outputs" / "classifier.joblib"
        return [
            Path(self.core_models_dir) / "layer_c" / "classifier.joblib",
            Path(self.bundle_root_dir) / "layer_c" / "classifier.joblib",
            Path(self.artifacts_root_dir) / "layer_c" / "classifier.joblib",
            legacy,
        ]

    @property
    def layer_c_model_pathname(self) -> str:
        return str(Path(self.artifacts_root_dir) / "layer_c" / "classifier.joblib")

    ### Layer D (ModernBERT classifier)
    layer_d_model_id: str = "answerdotai/ModernBERT-large"
    layer_d_max_length: int = 512
    layer_d_num_train_epochs: int = 3
    layer_d_learning_rate: float = 2e-5
    layer_d_warmup_ratio: float = 0.06
    layer_d_weight_decay: float = 0.01
    layer_d_malicious_class_weight: float = 1.75
    layer_d_focal_gamma: float = 1.5
    layer_d_train_batch_size: int = 8
    layer_d_eval_batch_size: int = 16
    layer_d_gradient_accumulation_steps: int = 4
    layer_d_gradient_checkpointing: bool = True
    layer_d_dataloader_num_workers: int = 4
    layer_d_use_bf16: bool = True
    layer_d_use_tf32: bool = True
    layer_d_split_train: float = 0.80
    layer_d_split_val: float = 0.10
    layer_d_split_test: float = 0.10
    layer_d_low_threshold: float = 0.20
    layer_d_high_threshold: float = 0.80

    # Layer D hard-negative mining (SAFE train-split examples with high malicious scores)
    layer_d_hard_negative_enabled: bool = False
    layer_d_hard_negative_use_routing_band: bool = True
    layer_d_hard_negative_score_min: float = 0.20
    layer_d_hard_negative_score_max: float = 0.80
    layer_d_hard_negative_max_samples: int = 5000
    layer_d_hard_negative_min_samples: int = 32
    layer_d_hard_negative_augment_multiplier: int = 1

    # Layer D model version selection
    # Model version selection maps to `core/models/layer_d/` when present;
    # per-layer `outputs/` paths are kept as backward-compatible fallbacks.
    layer_d_model_version: str = "legacy"

    @property
    def layer_d_release_dir(self):
        return self._path_with_override(
            "BARRIKADA_LAYER_D_RELEASE_DIR",
            Path(self.core_models_dir) / "layer_d" / "releases",
        )

    @property
    def layer_d_output_dir(self):
        legacy = self._package_root / "layer_d" / "outputs" / "model"
        return self._existing_path_with_override(
            "BARRIKADA_LAYER_D_MODEL_DIR",
            self.layer_d_model_candidates,
            "Layer D model directory",
        )

    @property
    def layer_d_model_candidates(self) -> list[Path]:
        legacy = self._package_root / "layer_d" / "outputs" / "model"
        return [
            Path(self.core_models_dir) / "layer_d" / "model",
            Path(self.bundle_root_dir) / "layer_d" / "model",
            Path(self.artifacts_root_dir) / "layer_d" / "model",
            legacy,
        ]

    @property
    def layer_d_model_dirname(self) -> str:
        return str(Path(self.artifacts_root_dir) / "layer_d" / "model")

    @property
    def layer_d_report_path(self):
        return self._output_file_with_override(
            "BARRIKADA_LAYER_D_REPORT_PATH",
            self._default_results_dir() / "layer_d_eval_latest.json",
        )

    ### Layer E (Qwen3Guard local judge)
    layer_e_judge_mode: str = "qwen3guard"
    layer_e_model_hf_id: str = "Qwen/Qwen3Guard-Gen-0.6B"
    layer_e_temperature: float = 0.0
    layer_e_timeout_s: float = 30.0
    layer_e_max_retries: int = 2
    layer_e_max_new_tokens: int = 64
    layer_e_no_think_default: bool = True

    # Teacher SFT defaults (QLoRA)
    layer_e_teacher_hf_model_id: str = "Qwen/Qwen3-4B"
    layer_e_teacher_epochs: int = 3
    layer_e_teacher_lr: float = 2e-4
    layer_e_teacher_warmup_ratio: float = 0.05
    # Memory-safe defaults for 24 GB class GPUs with QLoRA.
    layer_e_teacher_train_batch_size: int = 2
    layer_e_teacher_eval_batch_size: int = 1
    layer_e_teacher_grad_accum_steps: int = 16
    layer_e_teacher_max_seq_length: int = 384
    layer_e_teacher_lora_rank: int = 64
    layer_e_teacher_lora_alpha: int = 128
    layer_e_teacher_lora_dropout: float = 0.05

    @property
    def layer_e_output_dir(self):
        return self._output_dir_with_override(
            "BARRIKADA_LAYER_E_OUTPUT_DIR",
            self._default_layer_e_output_dir(),
        )

    @property
    def layer_e_teacher_output_dir(self):
        return str(self._ensure_directory(Path(self.layer_e_output_dir) / "teacher"))

    @property
    def layer_e_teacher_report_path(self):
        return self._output_file_with_override(
            "BARRIKADA_LAYER_E_TEACHER_REPORT_PATH",
            self._default_results_dir() / "layer_e_teacher_eval_latest.json",
        )

    @property
    def layer_e_model_dir(self):
        return self._existing_path_with_override(
            "BARRIKADA_LAYER_E_MODEL_DIR",
            self.layer_e_model_candidates,
            "Layer E model directory",
        )

    @property
    def layer_e_model_dirname(self) -> str:
        return str(Path(self.artifacts_root_dir) / "layer_e" / "qwen3guard-barrikade")

    @property
    def layer_e_model_candidates(self) -> list[Path]:
        return [
            Path(self.core_models_dir) / "layer_e" / "qwen3guard-barrikade",
            Path(self.bundle_root_dir) / "layer_e" / "qwen3guard-barrikade",
            Path(self.artifacts_root_dir) / "layer_e" / "qwen3guard-barrikade",
        ]
