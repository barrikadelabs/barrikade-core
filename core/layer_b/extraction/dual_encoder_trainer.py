"""Dual-encoder contrastive training for Layer B.

Trains two specialised SentenceTransformer encoders using InfoNCE loss
with hard negative mining:

  prompt_encoder:    encodes incoming prompts at runtime
  signature_encoder: encodes attack/benign centroids at build time

Architecture:

    prompt  ──→  prompt_encoder  ──┐
                                   ├── cosine similarity ──→ detection
    centroid ──→ signature_encoder ┘

Training objective (InfoNCE with in-batch + hard negatives):

    L_i = -log( exp(sim(p_i, s_i⁺) / τ)  /
                ( Σ_j exp(sim(p_i, s_j) / τ)
                + Σ_k exp(sim(p_i, h_ik) / τ) ) )

where:
    p_i   = prompt embedding (from prompt_encoder)
    s_i⁺  = positive signature embedding (from signature_encoder)
    s_j   = in-batch negatives
    h_ik  = hard negatives (closest opposite-class texts)
    τ     = temperature

Hard negative mining:
    For each malicious prompt, find the k closest safe prompts.
    For each safe prompt, find the k closest malicious prompts.
    These are the cases the base model already confuses, forcing the
    fine-tuned encoders to learn precise decision boundaries.

After training, rebuild signatures:
    python scripts/extract_signature_patterns.py

The extraction script will automatically detect and use the trained
signature_encoder.  The runtime SignatureEngine will automatically
detect and use the trained prompt_encoder.
"""

import gc
import logging
import random
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)



# Hard Negative Mining
def mine_hard_negatives(mal_texts, safe_texts, base_model_name, k=5):
    """Find the k closest opposite-class texts for each sample.

    Uses the base (pre-trained) model embeddings and FAISS for fast
    nearest-neighbour retrieval.

    Returns:
        mal_neg_ids:  (n_mal,  k) — indices into safe_texts
        safe_neg_ids: (n_safe, k) — indices into mal_texts
    """
    log.info("Mining hard negatives with %s …", base_model_name)
    # Keep hard-negative mining on CPU so the GPU is fully available for training.
    model = SentenceTransformer(base_model_name, device="cuda")

    log.info("  Encoding %d malicious texts …", len(mal_texts))
    mal_emb = model.encode(
        mal_texts, batch_size=256,
        show_progress_bar=True, normalize_embeddings=True,
    )
    mal_emb = np.asarray(mal_emb, dtype=np.float32)

    log.info("  Encoding %d safe texts …", len(safe_texts))
    safe_emb = model.encode(
        safe_texts, batch_size=256,
        show_progress_bar=True, normalize_embeddings=True,
    )
    safe_emb = np.asarray(safe_emb, dtype=np.float32)

    dim = mal_emb.shape[1]

    # Malicious → closest safe
    safe_idx = faiss.IndexFlatIP(dim)
    safe_idx.add(safe_emb) # type: ignore
    _, mal_neg_ids = safe_idx.search(mal_emb, k) # type: ignore

    # Safe → closest malicious
    mal_idx = faiss.IndexFlatIP(dim)
    mal_idx.add(mal_emb) # type: ignore
    _, safe_neg_ids = mal_idx.search(safe_emb, k) # type: ignore

    del model, mal_emb, safe_emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("Hard negative mining complete.")
    return mal_neg_ids, safe_neg_ids



# Dataset
class ContrastivePairDataset(Dataset):
    """Yields ``(anchor, positive, hard_negatives)`` for dual-encoder training.

    Pair construction:
        Malicious anchors are paired with another random malicious text
        (positive) and the k closest safe texts (hard negatives).

        Safe anchors are paired with another random safe text (positive)
        and the k closest malicious texts (hard negatives).

    Batches contain a mix of both classes.  InfoNCE in-batch negatives
    provide the primary contrastive signal; hard negatives sharpen the
    decision boundary between the classes.
    """

    def __init__(self, mal_texts, safe_texts, mal_neg_ids, safe_neg_ids, n_hard=3, max_pairs=None):
        self.items = []  # list of (anchor, positive, [hard_neg, ...])
        n_mal = len(mal_texts)
        n_safe = len(safe_texts)

        #Malicious positive pairs + safe hard negatives
        indices = list(range(n_mal))
        random.shuffle(indices)
        for i in range(0, n_mal - 1, 2):
            a, b = indices[i], indices[i + 1]
            hards = [safe_texts[int(j)] for j in mal_neg_ids[a][:n_hard]]
            self.items.append((mal_texts[a], mal_texts[b], hards))

        #Safe positive pairs + malicious hard negatives
        n_safe_pairs = min(n_safe // 2, len(self.items))
        indices = list(range(n_safe))
        random.shuffle(indices)
        for i in range(0, n_safe_pairs * 2, 2):
            a, b = indices[i], indices[i + 1]
            hards = [mal_texts[int(j)] for j in safe_neg_ids[a][:n_hard]]
            self.items.append((safe_texts[a], safe_texts[b], hards))

        random.shuffle(self.items)

        if max_pairs and len(self.items) > max_pairs:
            self.items = self.items[:max_pairs]

        log.info("Training dataset: %d pairs (%d hard negatives each)",
                 len(self.items), n_hard)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _collate(batch):
    """Collate into (anchors, positives, flat_hard_negatives)."""
    anchors = [item[0] for item in batch]
    positives = [item[1] for item in batch]
    hard_negs = []
    for item in batch:
        hard_negs.extend(item[2])
    return anchors, positives, hard_negs


# Training
def _to_device(features, device):
    return {k: v.to(device) for k, v in features.items()}


def train_dual_encoder(mal_texts, safe_texts, settings):
    """Train dual encoders and save to the signatures directory.

    Returns (prompt_encoder_path, signature_encoder_path).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training device: %s", device)

    base_model = settings.layer_b_embedding_model
    n_hard = settings.layer_b_dual_encoder_hard_negatives
    temperature = settings.layer_b_dual_encoder_temperature
    batch_size = settings.layer_b_dual_encoder_batch_size
    epochs = settings.layer_b_dual_encoder_epochs
    lr = settings.layer_b_dual_encoder_lr
    max_samples = settings.layer_b_dual_encoder_max_samples

    #Subsample for training speed
    if max_samples and len(mal_texts) > max_samples:
        mal_sample = random.sample(list(mal_texts), max_samples)  # nosec B311
    else:
        mal_sample = list(mal_texts)

    if max_samples and len(safe_texts) > max_samples:
        safe_sample = random.sample(list(safe_texts), max_samples)  # nosec B311
    else:
        safe_sample = list(safe_texts)

    log.info("Training samples: %d malicious, %d safe (max_samples=%s)",
             len(mal_sample), len(safe_sample), max_samples)

    #Mine hard negatives
    mal_neg_ids, safe_neg_ids = mine_hard_negatives(
        mal_sample, safe_sample, base_model, k=n_hard,
    )

    #Build dataset
    dataset = ContrastivePairDataset(
        mal_sample, safe_sample, mal_neg_ids, safe_neg_ids,
        n_hard=n_hard, max_pairs=max_samples,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, drop_last=True,
    )

    #Initialise both encoders from the same base model
    log.info("Initialising dual encoders from %s …", base_model)
    prompt_encoder = SentenceTransformer(base_model)
    signature_encoder = SentenceTransformer(base_model)
    prompt_encoder.to(device)
    signature_encoder.to(device)

    #Optimiser (joint update of both encoders)
    optimizer = AdamW(
        list(prompt_encoder.parameters())
        + list(signature_encoder.parameters()),
        lr=lr,
    )

    #Training loop
    for epoch in range(epochs):
        prompt_encoder.train()
        signature_encoder.train()
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for anchors, positives, hard_negs in dataloader:
            B = len(anchors)

            # Encode anchors with prompt_encoder
            p_feat = prompt_encoder.tokenize(anchors)
            p_feat = _to_device(p_feat, device)
            p_emb = prompt_encoder(p_feat)["sentence_embedding"]
            p_emb = F.normalize(p_emb, p=2, dim=1)

            # Encode positives (B texts)
            pos_feat = signature_encoder.tokenize(positives)
            pos_feat = _to_device(pos_feat, device)
            pos_emb = signature_encoder(pos_feat)["sentence_embedding"]
            pos_emb = F.normalize(pos_emb, p=2, dim=1)

            # Encode hard negatives in small chunks to avoid OOM
            hard_emb_list = []
            chunk_size = max(1, B)
            for i in range(0, len(hard_negs), chunk_size):
                chunk = hard_negs[i:i+chunk_size]
                h_feat = signature_encoder.tokenize(chunk)
                h_feat = _to_device(h_feat, device)
                h_emb = signature_encoder(h_feat)["sentence_embedding"]
                hard_emb_list.append(F.normalize(h_emb, p=2, dim=1))
            hard_emb = torch.cat(hard_emb_list, dim=0).view(B, n_hard, -1)

            # In-batch similarity: (B, B)
            in_batch_sim = torch.mm(p_emb, pos_emb.t()) / temperature

            # Hard negative similarity: (B, n_hard)
            hard_sim = torch.bmm(
                p_emb.unsqueeze(1), hard_emb.transpose(1, 2),
            ).squeeze(1) / temperature

            # Combined logits: (B, B + n_hard)
            logits = torch.cat([in_batch_sim, hard_sim], dim=1)

            # Target: each anchor's positive is at its own index (diagonal)
            labels = torch.arange(B, device=device)

            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_steps += 1

            if n_steps % 100 == 0:
                log.info("  [epoch %d] step %d — loss %.4f",
                         epoch + 1, n_steps, total_loss / n_steps)

        avg_loss = total_loss / max(n_steps, 1)
        log.info("Epoch %d/%d — avg loss: %.4f (%d steps)",
                 epoch + 1, epochs, avg_loss, n_steps)

    #Save encoders
    sig_dir = Path(settings.layer_b_signatures_dir)
    prompt_path = sig_dir / "prompt_encoder"
    sig_path = sig_dir / "signature_encoder"

    prompt_encoder.save(str(prompt_path))
    signature_encoder.save(str(sig_path))

    log.info("Saved prompt encoder    → %s", prompt_path)
    log.info("Saved signature encoder → %s", sig_path)

    del prompt_encoder, signature_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(prompt_path), str(sig_path)
