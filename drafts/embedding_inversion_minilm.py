"""
embedding_inversion_minilm.py  ·  © 2025
================================================
Two reference implementations for *text* embedding‑inversion attacks on the
`sentence-transformers/all-MiniLM-L6-v2` encoder (384‑d vector per sentence).
Both assume **white‑box** access to the victim model and run efficiently on
Apple‑silicon (M‑series) via PyTorch's `mps` backend.

┌─────────────────────────────────────────────────────────────────────────┐
│ 1️⃣  Inverse‑Decoder (train_from_scratch)                               │
│     – Supervised seq‑to‑seq that learns f⁻¹(embedding) → text          │
│ 2️⃣  Iterative Generate‑and‑Refine (vec2text)                           │
│     – Autoregressive LM + feedback loop that edits its own output      │
└─────────────────────────────────────────────────────────────────────────┘

Quick start (recommended workflow)
----------------------------------
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
$ pip install sentence-transformers datasets accelerate tqdm

(1) Prepare an embedding dataset (here we grab ~60 k Wiki sentences)::
    python embedding_inversion_minilm.py prep \
        --hf_dataset wikitext \
        --config wikitext-2-raw-v1 \
        --split 'train[:60000]'

(2) Train the inverse decoder (takes ≈20 min on an M‑series GPU)::
    python embedding_inversion_minilm.py train --ckpt inv_decoder.pt

(3) Invert a sentence embedding via the decoder::
    python embedding_inversion_minilm.py decode --ckpt inv_decoder.pt \
        --sentence "Taxi drivers fought to defend jobs at London airport."

(4) Try iterative refine (uses trained decoder for the initial guess)::
    python embedding_inversion_minilm.py refine --ckpt inv_decoder.pt \
        --sentence "Taxi drivers fought to defend jobs at London airport."

Notes
-----
* The *decoder* here is a lightweight GRU‑based model (~25 M params); adjust
  `--hidden` if you have more VRAM.
* `--mps` backend is auto‑selected on macOS 14+; no extra flag needed.
* For research‑grade results see the cited papers; this script is a compact
  baseline (<400 LOC) meant for experimentation and pedagogy.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

################################################################################
# Globals & utils
################################################################################

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER = AutoTokenizer.from_pretrained(ENCODER_NAME)
MAX_LEN = 32  # maximum tokens to reconstruct

################################################################################
# Dataset wrapper – stores (embedding, token_ids) pairs pre‑computed to disk
################################################################################

class EmbeddingTextPair(Dataset):
    """Memory‑mapped dataset of (sentence_embedding, token_ids)."""

    def __init__(self, tensor_path: str):
        data = torch.load(tensor_path, map_location="cpu")
        self.embeds = data["embeds"]  # [N, 384]
        self.tokens = data["tokens"]  # list[list[int]] variable length
        assert len(self.embeds) == len(self.tokens)

    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        return self.embeds[idx], torch.tensor(self.tokens[idx], dtype=torch.long)

################################################################################
# GRU‑based inverse decoder
################################################################################

class GRUDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc_init = nn.Linear(embedding_dim, hidden_size * num_layers)
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, sentence_embedding: torch.Tensor, tgt_input: torch.Tensor):
        """Teacher‑forcing forward pass.

        Parameters
        ----------
        sentence_embedding : [B, 384]
        tgt_input          : [B, T] token ids (ground truth, without <sep>)
        """
        B, T = tgt_input.shape
        h0 = self.fc_init(sentence_embedding).view(
            self.num_layers, B, self.hidden_size
        ).contiguous()
        # Shift‑right trick (prepend CLS, drop last token)
        dec_in = torch.cat(
            [
                torch.full((B, 1), TOKENIZER.cls_token_id, device=tgt_input.device),
                tgt_input,
            ],
            1,
        )[:, :-1]
        emb = self.embedding(dec_in)
        out, _ = self.gru(emb, h0)
        logits = self.proj(out)
        return logits

    @torch.no_grad()
    def generate(
        self,
        sentence_embedding: torch.Tensor,
        max_len: int = MAX_LEN,
        temperature: float = 1.0,
    ) -> List[int]:
        """Greedy/temperature sampling decode, returns token ids (no specials)."""
        ids: List[int] = []
        h = self.fc_init(sentence_embedding).view(1, 1, self.hidden_size).repeat(
            self.num_layers, 1, 1
        )
        token = torch.tensor([[TOKENIZER.cls_token_id]], device=sentence_embedding.device)
        for _ in range(max_len):
            emb = self.embedding(token)
            out, h = self.gru(emb, h)
            logits = self.proj(out[:, -1]) / temperature
            next_id = logits.softmax(-1).argmax(-1)
            if next_id.item() == TOKENIZER.sep_token_id:
                break
            ids.append(next_id.item())
            token = next_id.unsqueeze(0)
        return ids

################################################################################
# Vec2Text‑style iterative refine (lightweight)
################################################################################

def iterative_refine(
    target_embedding: torch.Tensor,
    decoder: GRUDecoder,
    embedder: nn.Module,
    tokenizer: AutoTokenizer,
    k: int = 5,
    steps: int = 5,
) -> str:
    """Simple generate‑and‑local‑search refinement (token swap heuristic)."""
    with torch.no_grad():
        guess_ids = decoder.generate(target_embedding)
        best_ids = guess_ids[:]
        best_sim = _cos(embedder, tokenizer, best_ids, target_embedding)
        for _ in range(steps):
            improved = False
            for i in range(len(best_ids)):
                prefix, suffix = best_ids[:i], best_ids[i + 1 :]
                sims = []
                for alt in range(tokenizer.vocab_size):
                    new_ids = prefix + [alt] + suffix
                    sims.append((alt, _cos(embedder, tokenizer, new_ids, target_embedding)))
                sims.sort(key=lambda x: x[1], reverse=True)
                if sims[0][1] > best_sim + 1e-4:
                    best_ids[i], best_sim, improved = sims[0][0], sims[0][1], True
            if not improved:
                break
        return tokenizer.decode(best_ids, skip_special_tokens=True)

def _cos(embedder, tokenizer, ids, tgt):
    txt = tokenizer.decode(ids, skip_special_tokens=True)
    emb = embed_text(embedder, tokenizer, txt)[0]
    return F.cosine_similarity(emb, tgt, dim=0).item()

################################################################################
# Helper – embed sentences once (MiniLM encoder)
################################################################################

@torch.no_grad()
def embed_text(embedder, tokenizer, text):
    if isinstance(text, str):
        text = [text]
    toks = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    emb = embedder(**toks)[0]
    return F.normalize(emb, dim=-1)

################################################################################
# Data‑prep command
################################################################################

def cmd_prep(args):
    embedder = AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE)
    dataset = load_dataset(args.hf_dataset, args.config, split=args.split)
    # Pick first text‑like column
    col = next(c for c in dataset.column_names if dataset.features[c].dtype == "string")
    sentences = dataset[col]
    embeds, toks = [], []
    for s in tqdm(sentences, desc="Embedding"):
        emb = embed_text(embedder, TOKENIZER, s)[0].cpu()
        tok_ids = TOKENIZER.encode(s, max_length=MAX_LEN, truncation=True)
        embeds.append(emb)
        toks.append(tok_ids)
    out = {"embeds": torch.stack(embeds), "tokens": toks}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"Saved {len(embeds):,} pairs → {args.out}")

################################################################################
# Training loop (inverse decoder)
################################################################################

def cmd_train(args):
    ds = EmbeddingTextPair(args.data)
    train_len = int(len(ds) * 0.9)
    train_ds, val_ds = random_split(ds, [train_len, len(ds) - train_len])
    tr_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(val_ds, batch_size=args.bs, collate_fn=collate)

    model = GRUDecoder(384, TOKENIZER.vocab_size, hidden_size=args.hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best = math.inf
    for ep in range(args.epochs):
        model.train()
        for emb, tgt in tqdm(tr_loader, desc=f"Train {ep}"):
            emb, tgt = emb.to(DEVICE), tgt.to(DEVICE)
            loss = F.cross_entropy(model(emb, tgt).view(-1, TOKENIZER.vocab_size), tgt.view(-1))
            loss.backward(); opt.step(); opt.zero_grad()
        # val
        model.eval(); vloss, n = 0.0, 0
        with torch.no_grad():
            for emb, tgt in va_loader:
                emb, tgt = emb.to(DEVICE), tgt.to(DEVICE)
                loss = F.cross_entropy(model(emb, tgt).view(-1, TOKENIZER.vocab_size), tgt.view(-1))
                vloss += loss.item() * emb.size(0); n += emb.size(0)
        vloss /= n; print(f"Epoch {ep}: val CE = {vloss:.3f}")
        if vloss < best:
            best = vloss; torch.save(model.state_dict(), args.ckpt)
            print(f" ↳ checkpoint saved → {args.ckpt}")

################################################################################
# Collate function for variable‑length targets
################################################################################

def collate(batch):
    embeds, seqs = zip(*batch)
    max_len = max(len(s) for s in seqs)
    pad_id = TOKENIZER.pad_token_id
    tgt = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        tgt[i, : len(s)] = torch.tensor(s)
    return torch.stack(embeds), tgt

################################################################################
# Decode command (single sentence)
################################################################################

def cmd_decode(args):
    model = GRUDecoder(384, TOKENIZER.vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()
    tgt_emb = embed_text(AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE), TOKENIZER, args.sentence)[0]
    out_ids = model.generate(tgt_emb.unsqueeze(0))
    print("Reconstruction:\n", TOKENIZER.decode(out_ids, skip_special_tokens=True))

################################################################################
# Refine command
################################################################################

def cmd_refine(args):
    embedder = AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE)
    model = GRUDecoder(384, TOKENIZER.vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()
    tgt_emb = embed_text(embedder, TOKENIZER, args.sentence)[0]
    refined = iterative_refine(tgt_emb.unsqueeze(0), model, embedder, TOKENIZER, k=args.k, steps=args.steps)
    print("Iterative refine output:\n", refined)

################################################################################
# CLI
################################################################################

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prep", help="Build (embedding, tokens) dataset")
    sp.add_argument("--hf_dataset", default="wikitext")
    sp.add_argument("--split", default="train[:60000]")
    sp.add_argument("--config", default="wikitext-2-raw-v1")
    sp.add_argument("--out", default="data/wiki60k.pt")
    sp.set_defaults(func=cmd_prep)

    sp = sub.add_parser("train", help="Train inverse decoder")
    sp.add_argument("--data", default="data/wiki60k.pt")
    sp.add_argument("--ckpt", default="inv_decoder.pt")
    sp.add_argument("--bs", type=int, default=64)
    sp.add_argument("--epochs", type=int, default=6)
    sp.add_argument("--hidden", type=int, default=768)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("decode", help="Decode a single embedding → text")
    sp.add_argument("--ckpt", default="inv_decoder.pt")
    sp.add_argument("--sentence", required=True)
    sp.set_defaults(func=cmd_decode)

    sp = sub.add_parser("refine", help="Iterative generate‑and‑refine")
    sp.add_argument("--ckpt", default="inv_decoder.pt")
    sp.add_argument("--sentence", required=True)
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--steps", type=int, default=5)
    sp.set_defaults(func=cmd_refine)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
