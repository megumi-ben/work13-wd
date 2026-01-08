#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =============================================================================================
# 1) Seed / Device
# =============================================================================================

def setup_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[Info] Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("[Info] Using CPU")
    return device


# =============================================================================================
# 2) Optional PG prober (TABLESAMPLE) for prefix% candidate estimation in eval
# =============================================================================================

try:
    import psycopg2
except Exception:
    psycopg2 = None


def like_escape_literal(prefix: str) -> str:
    return prefix.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


@dataclass
class PgProbeConfig:
    table: str
    probe_pct: float
    probe_rounds: int


class PgProber:
    def __init__(self, dsn: str, cfg: PgProbeConfig):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 not available, but --pg_dsn was provided.")
        self.conn = psycopg2.connect(dsn)
        self.cfg = cfg

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def estimate_prefix_candidates(self, prefix: str) -> Optional[int]:
        pct = max(0.0001, min(100.0, float(self.cfg.probe_pct)))
        rounds = max(1, int(self.cfg.probe_rounds))

        pat = like_escape_literal(prefix) + "%"
        sql = f"""
        WITH samp AS (
          SELECT description
          FROM {self.cfg.table} TABLESAMPLE SYSTEM ({pct})
        )
        SELECT count(*)::bigint
        FROM samp
        WHERE description ILIKE %s ESCAPE '\\';
        """

        counts = []
        try:
            with self.conn.cursor() as cur:
                for _ in range(rounds):
                    cur.execute(sql, (pat,))
                    counts.append(int(cur.fetchone()[0]))
        except Exception:
            return None

        if not counts:
            return None

        med = int(statistics.median(counts))
        scale = 100.0 / pct
        return int(med * scale)


# =============================================================================================
# 3) Dataset (char-level) for regex -> prefix
# =============================================================================================

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"


class Regex2PrefixDataset(Dataset):
    """
    Char-level seq2seq dataset.
    Now supports:
      - sample_weight: scalar (e.g., favor smaller cand_est / longer prefixes)
      - token_weight: per-char weights (default all-1 for Route A)
    """

    def __init__(
        self,
        src_list: List[str],
        trg_list: List[str],
        sample_weight: Optional[List[float]] = None,
    ) -> None:
        assert len(src_list) == len(trg_list)
        self.src_list = src_list
        self.trg_list = trg_list
        self.sample_weight = sample_weight if sample_weight is not None else [1.0] * len(src_list)

        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.src_vocab_size: int = 0
        self.trg_vocab_size: int = 0

        self._build_vocab()

    def _build_vocab(self) -> None:
        unique_chars = set()
        for src, trg in zip(self.src_list, self.trg_list):
            unique_chars.update(list(src))
            unique_chars.update(list(trg))

        unique_chars = sorted(unique_chars)

        self.char_to_idx = {SOS_TOKEN: 0, EOS_TOKEN: 1, PAD_TOKEN: 2}
        for idx, ch in enumerate(unique_chars, start=3):
            self.char_to_idx[ch] = idx

        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
        self.src_vocab_size = len(self.char_to_idx)
        self.trg_vocab_size = len(self.char_to_idx)

        print(f"[Dataset] Vocab size: {self.src_vocab_size}")

    def __len__(self) -> int:
        return len(self.src_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src = self.src_list[idx]
        trg = self.trg_list[idx]

        # Route A: no regex-span weighting; keep uniform per-char weights
        # +1 for SOS, +1 for EOS
        trg_weights = [1.0] * (len(trg) + 2)

        src_tokens = [SOS_TOKEN] + list(src) + [EOS_TOKEN]
        trg_tokens = [SOS_TOKEN] + list(trg) + [EOS_TOKEN]

        src_idx = [self.char_to_idx[ch] for ch in src_tokens]
        trg_idx = [self.char_to_idx[ch] for ch in trg_tokens]

        src_tensor = torch.tensor(src_idx, dtype=torch.long)
        trg_tensor = torch.tensor(trg_idx, dtype=torch.long)
        weight_tensor = torch.tensor(trg_weights, dtype=torch.float32)

        sw = float(self.sample_weight[idx])
        sample_weight_tensor = torch.tensor(sw, dtype=torch.float32)
        return src_tensor, trg_tensor, weight_tensor, sample_weight_tensor


def make_collate_fn(pad_idx: int):
    def collate_fn(batch):
        src_batch, trg_batch, weight_batch, sw_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
        weight_padded = pad_sequence(weight_batch, batch_first=True, padding_value=1.0)
        sw_tensor = torch.stack(sw_batch, dim=0)  # [B]
        return src_padded, trg_padded, weight_padded, sw_tensor
    return collate_fn


# =============================================================================================
# 4) Transformer
# =============================================================================================

class ImprovedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            seq_len = self.pe.size(0)
            x = x[:, :seq_len]
        x = x + self.pe[:seq_len, :].permute(1, 0, 2)
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_seq_length: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)

        self.positional_encoding = ImprovedPositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_seq_length
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.output_linear = nn.Linear(d_model, trg_vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in [self.src_embedding, self.trg_embedding]:
            nn.init.normal_(emb.weight, mean=0.0, std=self.d_model ** -0.5)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, 0.0)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        return mask

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        trg_emb = self.trg_embedding(trg) * math.sqrt(self.d_model)

        src_emb = self.positional_encoding(src_emb)
        trg_emb = self.positional_encoding(trg_emb)

        tgt_mask = self.generate_square_subsequent_mask(trg.size(1), device=trg.device)

        out = self.transformer(
            src=src_emb,
            tgt=trg_emb,
            src_mask=None,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.output_linear(out)
        return logits


# =============================================================================================
# 5) Decode
# =============================================================================================

@torch.no_grad()
def greedy_decode_prefix(
    model: nn.Module,
    src_text: str,
    dataset: Regex2PrefixDataset,
    device: torch.device,
    max_len: int = 128,
) -> str:
    model.eval()
    sos_idx = dataset.char_to_idx[SOS_TOKEN]
    eos_idx = dataset.char_to_idx[EOS_TOKEN]
    pad_idx = dataset.char_to_idx[PAD_TOKEN]

    src_tokens = [SOS_TOKEN] + list(src_text) + [EOS_TOKEN]
    src_idx = [dataset.char_to_idx[ch] for ch in src_tokens]
    src = torch.tensor(src_idx, dtype=torch.long, device=device).unsqueeze(0)
    src_pad_mask = src.eq(pad_idx)

    trg = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    for _ in range(max_len):
        trg_pad_mask = trg.eq(pad_idx)
        logits = model(
            src=src,
            trg=trg,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask,
        )
        next_token = logits[:, -1, :].argmax(dim=-1)
        next_id = int(next_token.item())
        trg = torch.cat([trg, next_token.unsqueeze(0)], dim=1)
        if next_id == eos_idx:
            break

    pred_ids = trg[0].tolist()[1:]
    out_chars: List[str] = []
    for idx in pred_ids:
        if idx == eos_idx:
            break
        out_chars.append(dataset.idx_to_char[idx])
    return "".join(out_chars)


@torch.no_grad()
def beam_search_decode_prefixes(
    model: nn.Module,
    prefix_regex: str,
    dataset: Regex2PrefixDataset,
    device: torch.device,
    beam_size: int = 5,
    max_len: int = 128,
) -> List[str]:
    model.eval()
    sos_idx = dataset.char_to_idx[SOS_TOKEN]
    eos_idx = dataset.char_to_idx[EOS_TOKEN]
    pad_idx = dataset.char_to_idx[PAD_TOKEN]

    src_tokens = [SOS_TOKEN] + list(prefix_regex) + [EOS_TOKEN]
    src_idx = [dataset.char_to_idx[ch] for ch in src_tokens]
    src = torch.tensor(src_idx, dtype=torch.long, device=device).unsqueeze(0)  # (1, S)
    src_pad_mask = src.eq(pad_idx)  # (1, S)

    beams: List[Tuple[List[int], float, bool]] = [([sos_idx], 0.0, False)]

    for _ in range(max_len):
        if all(finished for _, _, finished in beams):
            break

        seq_tensors = [torch.tensor(t, dtype=torch.long, device=device) for t, _, _ in beams]
        lengths = torch.tensor([len(t) for t, _, _ in beams], dtype=torch.long, device=device)  # (B,)
        trg = pad_sequence(seq_tensors, batch_first=True, padding_value=pad_idx)  # (B, Tm)
        trg_pad_mask = trg.eq(pad_idx)  # (B, Tm)

        B = trg.size(0)
        src_rep = src.expand(B, -1)                # (B, S)
        src_pad_rep = src_pad_mask.expand(B, -1)  # (B, S)

        logits = model(
            src=src_rep,
            trg=trg,
            src_key_padding_mask=src_pad_rep,
            tgt_key_padding_mask=trg_pad_mask,
        )  # (B, Tm, V)

        last_pos = (lengths - 1).clamp(min=0)  # (B,)
        last_logits = logits[torch.arange(B, device=device), last_pos, :]  # (B, V)
        log_probs = torch.log_softmax(last_logits, dim=-1)  # (B, V)

        all_candidates: List[Tuple[List[int], float, bool]] = []
        for i, (tokens, log_p, finished) in enumerate(beams):
            if finished:
                all_candidates.append((tokens, log_p, True))
                continue

            topk_log_probs, topk_indices = torch.topk(log_probs[i], k=beam_size)
            for k in range(beam_size):
                token_id = int(topk_indices[k].item())
                new_log_p = log_p + float(topk_log_probs[k].item())
                new_tokens = tokens + [token_id]
                all_candidates.append((new_tokens, new_log_p, token_id == eos_idx))

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_size]

    result_strings: List[str] = []
    for tokens, _, _ in beams:
        char_ids = tokens[1:]
        chars: List[str] = []
        for idx in char_ids:
            if idx == eos_idx:
                break
            chars.append(dataset.idx_to_char[idx])
        result_strings.append("".join(chars))

    seen = set()
    uniq: List[str] = []
    for s in result_strings:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:beam_size]


# =============================================================================================
# 6) Data loading for "grouped" format (Route A)
# =============================================================================================

def _pick_prefixes_from_group(
    pref_list: List[Dict],
    mode: str,
    topk: int,
    rng: random.Random,
) -> List[Dict]:
    """
    mode:
      - best: pick best 1
      - topk: pick best topk
      - random: pick random 1
    ranking: (cand_est asc if present else inf, len_chars desc, support_ratio desc, support desc)
    """
    if not pref_list:
        return []

    def rank_key(p: Dict):
        ce = p.get("cand_est", None)
        ce_key = int(ce) if isinstance(ce, int) else 10**18
        lc = int(p.get("len_chars", len(p.get("prefix", ""))))
        sr = float(p.get("support_ratio", 0.0))
        sup = int(p.get("support", 0))
        return (ce_key, -lc, -sr, -sup)

    prefs_sorted = sorted(pref_list, key=rank_key)

    if mode == "random":
        return [rng.choice(pref_list)]
    if mode == "best":
        return [prefs_sorted[0]]
    if mode == "topk":
        return prefs_sorted[:max(1, topk)]
    raise ValueError(f"Unknown prefix_pick_mode: {mode}")


def _prefix_sample_weight(prefix_dict: Dict, alpha: float = 0.5) -> float:
    """
    Weight examples so that prefixes with smaller cand_est are emphasized.
    w = 1 / (cand_est + 1)^alpha  (if cand_est exists), else 1.0
    """
    ce = prefix_dict.get("cand_est", None)
    if isinstance(ce, int) and ce >= 0:
        return float((1.0 / ((ce + 1.0) ** alpha)))
    return 1.0


def load_grouped_workload(
    path: str,
    src_field: str,
    prefix_pick_mode: str,
    prefix_topk: int,
    max_src_len: int,
    max_trg_len: int,
    strip_trailing_space: bool,
    weight_alpha: float,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[float], List[int]]:
    """
    Returns:
      src_list, trg_list, sample_weight_list, group_ids
    group_ids are line indices (group index), used for group-level split (optional).
    """
    src_list: List[str] = []
    trg_list: List[str] = []
    sw_list: List[float] = []
    gid_list: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for gid, line in enumerate(tqdm(f, desc="Loading grouped")):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            src = rec.get(src_field, "") or ""
            if not src:
                continue

            prefs = rec.get("target_prefixes", None)
            if not isinstance(prefs, list) or len(prefs) == 0:
                continue

            picked = _pick_prefixes_from_group(prefs, mode=prefix_pick_mode, topk=prefix_topk, rng=rng)
            for p in picked:
                pref = p.get("prefix", "")
                if not isinstance(pref, str) or not pref.strip():
                    continue
                if strip_trailing_space:
                    pref = pref.rstrip()

                if len(src) > max_src_len or len(pref) > max_trg_len:
                    continue

                src_list.append(src)
                trg_list.append(pref)
                sw_list.append(_prefix_sample_weight(p, alpha=weight_alpha))
                gid_list.append(gid)

    print(f"[Data] Loaded grouped pairs={len(src_list)} from {path}")
    return src_list, trg_list, sw_list, gid_list


# =============================================================================================
# 7) Training
# =============================================================================================

def _build_onecycle_scheduler(
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    accumulation_steps: int,
    epochs: int,
    max_lr: float,
    pct_start: float,
    final_div_factor: float,
) -> optim.lr_scheduler.OneCycleLR:
    steps_per_epoch = math.ceil(len(train_loader) / max(1, accumulation_steps))
    total_steps = steps_per_epoch * epochs
    if total_steps <= 0:
        raise ValueError("total_steps must be positive for OneCycleLR.")
    print(f"[Scheduler] steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")
    return optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        cycle_momentum=False,
        final_div_factor=final_div_factor,
    )


@torch.no_grad()
def evaluate_on_subset(
    model: nn.Module,
    dataset: Regex2PrefixDataset,
    X: List[str],
    Y: List[str],
    device: torch.device,
    beam_size: int,
    max_samples: int,
    pg: Optional[PgProber],
) -> Dict[str, float]:
    model.eval()
    n = min(len(X), max_samples)
    if n <= 0:
        return {}

    greedy_exact = 0.0
    greedy_starts = 0.0
    greedy_len_ratio = 0.0
    beam_exact = 0.0

    pred_cand_list = []
    gt_cand_list = []

    for i in range(n):
        src = X[i]
        gt = Y[i]

        pred = greedy_decode_prefix(model, src, dataset, device, max_len=128)

        greedy_exact += float(pred == gt)
        greedy_starts += float(gt.startswith(pred))
        greedy_len_ratio += (len(pred) / max(1, len(gt)))

        beams = beam_search_decode_prefixes(model, src, dataset, device, beam_size=beam_size, max_len=128)
        beam_exact += float(any(b == gt for b in beams))

        if pg is not None:
            ce_pred = pg.estimate_prefix_candidates(pred) if pred.strip() else None
            ce_gt = pg.estimate_prefix_candidates(gt) if gt.strip() else None
            if ce_pred is not None:
                pred_cand_list.append(ce_pred)
            if ce_gt is not None:
                gt_cand_list.append(ce_gt)

        if i < 5:
            print(f"\n[{i+1}] SRC(regex): {src}")
            print(f"    GT(prefix) : {gt}")
            print(f"    Pred(greedy): {pred}")
            if pg is not None and pred_cand_list and gt_cand_list:
                print(f"    cand_est(pred/gt): {pred_cand_list[-1]} / {gt_cand_list[-1]}")

    out = {
        "greedy_exact": greedy_exact / n,
        "greedy_startswith_gt": greedy_starts / n,
        "greedy_len_ratio": greedy_len_ratio / n,
        f"beam@{beam_size}_exact": beam_exact / n,
    }

    if pg is not None and pred_cand_list and gt_cand_list:
        out["cand_est_pred_median"] = float(statistics.median(pred_cand_list))
        out["cand_est_gt_median"] = float(statistics.median(gt_cand_list))
        out["cand_est_pred_over_gt_median"] = float(
            (out["cand_est_pred_median"] / max(1.0, out["cand_est_gt_median"]))
        )

    print("\n[Eval] Summary:")
    for k, v in out.items():
        print(f"  {k:28s}: {v:.4f}" if isinstance(v, float) else f"  {k:28s}: {v}")

    return out


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    full_dataset: Regex2PrefixDataset,
    src_list: List[str],
    trg_list: List[str],
    eval_indices: List[int],
    device: torch.device,
    args: Any,
) -> None:
    pad_idx = full_dataset.char_to_idx[PAD_TOKEN]

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = _build_onecycle_scheduler(
        optimizer=optimizer,
        train_loader=train_loader,
        accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        max_lr=args.lr,
        pct_start=args.pct_start,
        final_div_factor=args.final_div_factor,
    )

    use_amp = device.type == "cuda"
    scaler = amp.GradScaler(enabled=use_amp)

    save_root = os.path.join("./models", args.saveName)
    os.makedirs(save_root, exist_ok=True)

    best_metric = -1.0
    best_ckpt_path: Optional[str] = None

    global_step = 0
    accum = 0
    optimizer.zero_grad(set_to_none=True)

    pg = None
    if args.pg_dsn:
        pg = PgProber(args.pg_dsn, PgProbeConfig(
            table=args.pg_table,
            probe_pct=args.probe_pct,
            probe_rounds=args.probe_rounds,
        ))

    print(
        f"[Train] FULL-DATA overfit mode. epochs={args.epochs}, lr(max)={args.lr}, "
        f"batch={args.batch_size}, accum={args.accumulation_steps}, "
        f"eval_interval_steps={args.eval_interval_steps}"
    )
    print(f"[Data] train_examples={len(src_list)}  eval_examples={len(eval_indices)} (same pool)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for (src, trg, trg_weight, sample_w) in tqdm(train_loader, desc=f"Epoch {epoch}"):
            src = src.to(device)
            trg = trg.to(device)
            trg_weight = trg_weight.to(device)
            sample_w = sample_w.to(device)  # [B]

            pad_mask_src = src.eq(pad_idx)
            pad_mask_trg = trg.eq(pad_idx)

            with amp.autocast(enabled=use_amp):
                logits = model(
                    src=src,
                    trg=trg[:, :-1],
                    src_key_padding_mask=pad_mask_src,
                    tgt_key_padding_mask=pad_mask_trg[:, :-1],
                )

                B, Tm1, V = logits.size()
                logits_flat = logits.reshape(-1, V)
                target_flat = trg[:, 1:].reshape(-1)

                weight_flat = trg_weight[:, 1:].reshape(-1)

                sw_expand = sample_w.unsqueeze(1).expand(B, Tm1).reshape(-1)
                weight_flat = weight_flat * sw_expand

                loss_token = criterion(logits_flat, target_flat)
                loss_token = loss_token * weight_flat

                valid_mask = (target_flat != pad_idx).float()
                valid_count = valid_mask.sum().clamp_min(1.0)
                loss = (loss_token * valid_mask).sum() / valid_count
                loss = loss / max(1, args.accumulation_steps)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                continue

            scaler.scale(loss).backward()
            accum += 1

            if accum >= args.accumulation_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_step += 1
                accum = 0

                if global_step % args.eval_interval_steps == 0:
                    print(f"\n[Train] step={global_step}, lr={scheduler.get_last_lr()[0]:.6e}  running eval (on full data)...")

                    X_eval = [src_list[i] for i in eval_indices]
                    Y_eval = [trg_list[i] for i in eval_indices]

                    metrics = evaluate_on_subset(
                        model=model,
                        dataset=full_dataset,
                        X=X_eval,
                        Y=Y_eval,
                        device=device,
                        beam_size=args.beam_size,
                        max_samples=args.max_eval_samples,
                        pg=pg,
                    )

                    # checkpoint metric
                    if metrics and ("cand_est_pred_over_gt_median" in metrics):
                        primary = -metrics["cand_est_pred_over_gt_median"]
                    else:
                        primary = metrics.get(f"beam@{args.beam_size}_exact", 0.0) if metrics else 0.0

                    if primary > best_metric:
                        best_metric = primary
                        best_ckpt_path = os.path.join(save_root, f"best_step_{global_step}.pth")
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(f"[Save] New best at step {global_step}. primary={best_metric:.6f}")
                    else:
                        print(f"[Info] No improvement. best_primary={best_metric:.6f}")

            epoch_loss += float(loss.item()) * max(1, args.accumulation_steps)
            n_steps += 1

        if accum > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            accum = 0

        dt = time.time() - t0
        avg_loss = epoch_loss / max(1, n_steps)
        print(f"\n[Epoch {epoch}] avg_loss={avg_loss:.4f}, time={dt:.1f}s, global_step={global_step}")

    if pg is not None:
        pg.close()

    print("\n[Training] Completed.")
    if best_ckpt_path:
        print(f"[Training] Best ckpt: {best_ckpt_path}  best_primary={best_metric:.6f}")


# =============================================================================================
# 9) CLI / main
# =============================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # input
    p.add_argument("--data_path", type=str, required=True, help="Grouped jsonl path (us_acc_1w_grouped.jsonl).")
    p.add_argument("--input_format", choices=["grouped"], default="grouped")

    # grouped loader controls
    p.add_argument("--src_field", choices=["regex_norm", "regex_canon"], default="regex_norm")
    p.add_argument("--prefix_pick_mode", choices=["best", "topk", "random"], default="topk",
                   help="How to choose prefix targets from target_prefixes.")
    p.add_argument("--prefix_topk", type=int, default=2, help="Used when prefix_pick_mode=topk.")
    p.add_argument("--strip_trailing_space", action="store_true", default=False)
    p.add_argument("--max_src_len", type=int, default=500)
    p.add_argument("--max_trg_len", type=int, default=120)
    p.add_argument("--weight_alpha", type=float, default=0.5, help="cand_est weight exponent in sample_weight.")

    # model
    p.add_argument("--HIDDEN_SIZE", type=int, default=256)
    p.add_argument("--LayerNum", type=int, default=4)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=1024)

    # train
    p.add_argument("--GPU", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--saveName", type=str, default="usacc_routeA_overfit_full")

    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--eval_interval_steps", type=int, default=200)
    p.add_argument("--max_eval_samples", type=int, default=256)
    p.add_argument("--beam_size", type=int, default=8)

    # OneCycleLR
    p.add_argument("--pct_start", type=float, default=0.15)
    p.add_argument("--final_div_factor", type=float, default=100.0)

    # dataloader
    p.add_argument("--num_workers", type=int, default=4)

    # optional PG eval
    p.add_argument("--pg_dsn", type=str, default="", help="If set, estimate prefix% candidates in eval.")
    p.add_argument("--pg_table", type=str, default="us_accidents")
    p.add_argument("--probe_pct", type=float, default=0.05)
    p.add_argument("--probe_rounds", type=int, default=3)

    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = setup_device(args.GPU)
    rng = random.Random(args.seed)

    # load grouped
    src_list, trg_list, sw_list, gid_list = load_grouped_workload(
        path=args.data_path,
        src_field=args.src_field,
        prefix_pick_mode=args.prefix_pick_mode,
        prefix_topk=args.prefix_topk,
        max_src_len=args.max_src_len,
        max_trg_len=args.max_trg_len,
        strip_trailing_space=args.strip_trailing_space,
        weight_alpha=args.weight_alpha,
        rng=rng,
    )
    if len(src_list) == 0:
        raise SystemExit("[Error] No data loaded.")

    # ===========================
    # FULL-DATA TRAINING (overfit)
    # ===========================
    # build vocab on full dataset and train on full dataset
    full_dataset = Regex2PrefixDataset(src_list, trg_list, sample_weight=sw_list)
    pad_idx = full_dataset.char_to_idx[PAD_TOKEN]

    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(pad_idx),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # eval indices: also full data (overfit mode)
    eval_indices = list(range(len(src_list)))

    model = TransformerSeq2Seq(
        src_vocab_size=full_dataset.src_vocab_size,
        trg_vocab_size=full_dataset.trg_vocab_size,
        d_model=args.HIDDEN_SIZE,
        nhead=args.nhead,
        num_layers=args.LayerNum,
        dim_feedforward=args.HIDDEN_SIZE * 4,
        max_seq_length=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)

    print(f"[Model] Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_model(
        model=model,
        train_loader=train_loader,
        full_dataset=full_dataset,
        src_list=src_list,
        trg_list=trg_list,
        eval_indices=eval_indices,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
