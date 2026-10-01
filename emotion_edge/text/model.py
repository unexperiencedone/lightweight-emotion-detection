"""DistilBERT classifier wrapper, vocabulary pruning, and the training loop."""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, CLS, SEP = 0, 100, 101, 102  # bert-base-uncased special ids (re-mapped when vocab is pruned)


class Tok:
    """Tokenizer + optional id remap table (for vocabulary pruning)."""

    def __init__(self, hf_tok, max_len=64, remap: np.ndarray | None = None, specials=None):
        self.t, self.max_len, self.remap = hf_tok, max_len, remap
        self.pad_id = hf_tok.pad_token_id

    def __call__(self, texts, pad=True):
        enc = self.t(list(texts), truncation=True, max_length=self.max_len, padding="longest" if pad else False,
                     return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        if self.remap is not None:
            ids = self.remap[ids]
        return ids, enc["attention_mask"].astype(np.int64)

    def truncation_rate(self, texts):
        n = sum(len(self.t(x, truncation=False)["input_ids"]) > self.max_len for x in texts)
        return n / max(1, len(texts))


def build_remap(hf_tok, texts, vocab_size):
    """Keep special tokens + every wordpiece seen in `texts`; everything else -> [UNK].
    Returns (remap array old_id->new_id, kept old ids in new order)."""
    seen = set(hf_tok.all_special_ids)
    for i in range(0, len(texts), 512):
        for row in hf_tok(list(texts[i:i + 512]), truncation=True, max_length=128)["input_ids"]:
            seen.update(row)
    kept = sorted(seen)
    remap = np.full(vocab_size, kept.index(hf_tok.unk_token_id), dtype=np.int64)
    for new, old in enumerate(kept):
        remap[old] = new
    return remap, kept


def prune_embeddings(model, kept_old_ids):
    emb = model.distilbert.embeddings.word_embeddings
    new = nn.Embedding(len(kept_old_ids), emb.embedding_dim, padding_idx=None)
    new.weight.data = emb.weight.data[torch.tensor(kept_old_ids)].clone()
    model.distilbert.embeddings.word_embeddings = new
    model.config.vocab_size = len(kept_old_ids)
    return model


class ExportWrapper(nn.Module):
    """input_ids, attention_mask -> logits (keeps the ONNX graph free of HF output dataclasses)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def batches(ids_list, y, bs, shuffle, rng):
    idx = np.arange(len(ids_list))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict_logits(model, tok: Tok, texts, bs=128):
    model.eval()
    order = np.argsort([len(t) for t in texts])  # length-bucketed => little padding
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    for i in range(0, len(texts), bs):
        sel = order[i:i + bs]
        ids, mask = tok([texts[j] for j in sel])
        out[sel] = model(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask)).logits.numpy()
    return out


def train(model, tok: Tok, tr, va, *, epochs=3, lr=5e-5, bs=32, wd=0.01, warmup=0.06, label_smoothing=0.0,
          teacher=None, kd_T=2.0, kd_alpha=0.5, seed=0, log=print):
    """Fine-tune; keep the epoch with the best *validation* accuracy. Returns (model, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    texts, y = list(tr.text), tr.label.values
    steps = epochs * math.ceil(len(texts) / bs)
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": wd},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(1, warmup * steps), max(0.0, (steps - s) / max(1, steps * (1 - warmup)))))
    hist, best, best_state, step = [], -1, None, 0
    for ep in range(epochs):
        model.train()
        t0, run = time.time(), 0.0
        # bucket by length inside shuffled mega-batches: fast CPU training w/o hurting randomness
        perm = rng.permutation(len(texts))
        chunks = [perm[i:i + bs * 50] for i in range(0, len(perm), bs * 50)]
        order = np.concatenate([c[np.argsort([len(texts[j]) for j in c])] for c in chunks])
        bl = [order[i:i + bs] for i in range(0, len(order), bs)]
        rng.shuffle(bl)
        for sel in bl:
            ids, mask = tok([texts[j] for j in sel])
            ids_t, mask_t, yt = torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(y[sel])
            logits = model(input_ids=ids_t, attention_mask=mask_t).logits
            loss = F.cross_entropy(logits, yt, label_smoothing=label_smoothing)
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(input_ids=ids_t, attention_mask=mask_t).logits
                kd = F.kl_div(F.log_softmax(logits / kd_T, -1), F.softmax(tl / kd_T, -1), reduction="batchmean") * kd_T ** 2
                loss = (1 - kd_alpha) * loss + kd_alpha * kd
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run += loss.item()
        val_logits = predict_logits(model, tok, list(va.text))
        acc = float((val_logits.argmax(1) == va.label.values).mean())
        hist.append({"epoch": ep + 1, "train_loss": run / len(bl), "val_acc": acc, "sec": time.time() - t0})
        log(f"epoch {ep + 1}/{epochs} loss={run / len(bl):.4f} val_acc={acc:.4f} ({time.time() - t0:.0f}s)")
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


class Int8Embedding(nn.Module):
    """Weight-only per-row int8 embedding. ORT's dynamic quantizer leaves Gather tables in fp32, and the
    word-embedding table is 36% of DistilBERT's parameters, so we quantize it ourselves.
    Lookup = Gather(int8) -> Cast -> Mul(row scale); numerically within ~0.4% of fp32 rows."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        w = emb.weight.data
        scale = (w.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-8)
        self.register_buffer("q", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale)
        self.embedding_dim, self.num_embeddings = emb.embedding_dim, emb.num_embeddings

    def forward(self, ids):
        return self.q[ids].float() * self.scale[ids]


def quantize_embeddings(model):
    e = model.distilbert.embeddings.word_embeddings
    model.distilbert.embeddings.word_embeddings = Int8Embedding(e)
    return model
