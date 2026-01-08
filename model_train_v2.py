#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import statistics
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

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
    text_col: str
    probe_pct: float
    probe_rounds: int
    debug: bool = False


class PgProber:
    """
    Estimate candidates for: SELECT count(*) FROM TABLESAMPLE WHERE col ILIKE 'prefix%'.
    Adds simple caching to avoid repeated DB calls for the same prefix.
    """
    def __init__(self, dsn: str, cfg: PgProbeConfig):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 not available, but --pg_dsn was provided.")
        self.conn = psycopg2.connect(dsn)
        self.cfg = cfg
        self._cache: Dict[str, Optional[int]] = {}

        # quick sanity check (fail fast if table/col wrong)
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT {cfg.text_col} FROM {cfg.table} LIMIT 1;")
                _ = cur.fetchone()
        except Exception as e:
            raise RuntimeError(
                f"PG sanity check failed. table={cfg.table}, col={cfg.text_col}. Error={e}"
            )

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def estimate_prefix_candidates(self, prefix: str) -> Optional[int]:
        prefix = prefix.strip()
        if not prefix:
            return None

        if prefix in self._cache:
            return self._cache[prefix]

        pct = max(0.0001, min(100.0, float(self.cfg.probe_pct)))
        rounds = max(1, int(self.cfg.probe_rounds))

        pat = like_escape_literal(prefix) + "%"
        sql = f"""
        WITH samp AS (
          SELECT {self.cfg.text_col} AS t
          FROM {self.cfg.table} TABLESAMPLE SYSTEM ({pct})
        )
        SELECT count(*)::bigint
        FROM samp
        WHERE t ILIKE %s ESCAPE '\\';
        """

        counts = []
        try:
            with self.conn.cursor() as cur:
                for _ in range(rounds):
                    cur.execute(sql, (pat,))
                    counts.append(int(cur.fetchone()[0]))
        except Exception as e:
            if self.cfg.debug:
                print(f"[PG][Warn] estimate failed for prefix={prefix!r}: {e}")
            self._cache[prefix] = None
            return None

        if not counts:
            self._cache[prefix] = None
            return None

        med = int(statistics.median(counts))
        scale = 100.0 / pct
        est = int(med * scale)
        self._cache[prefix] = est
        return est


# =============================================================================================
# 3) Tokenization & prefix extraction (RAW workload)
# =============================================================================================

# consistent with your generator:
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]")

def _is_punct(tok: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]", tok))

def extract_prefix_from_seed_string(
    s: str,
    k_tokens: int,
    include_trailing_punct: bool = True,
    strip_trailing_space: bool = True,
) -> Optional[str]:
    """
    Take the prefix of original string s that covers first k non-punct tokens.
    Return exact substring s[:end_pos], so it is safe for PG ILIKE 'prefix%'.
    """
    if k_tokens <= 0:
        return None

    matches = list(TOKEN_RE.finditer(s))
    if not matches:
        return None

    cnt = 0
    end = None
    idx = -1
    for i, m in enumerate(matches):
        tok = m.group(0)
        if _is_punct(tok):
            continue
        cnt += 1
        if cnt == k_tokens:
            end = m.end()
            idx = i
            break

    if end is None:
        return None

    if include_trailing_punct:
        j = idx + 1
        while j < len(matches):
            tok = matches[j].group(0)
            # include immediate punctuation like "." "," ":" etc.
            if _is_punct(tok):
                end = matches[j].end()
                j += 1
                continue
            break

    pref = s[:end]
    if strip_trailing_space:
        pref = pref.rstrip()
    return pref if pref.strip() else None


# =============================================================================================
# 4) Dataset (char-level) for regex -> prefix
# =============================================================================================

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"


class Regex2PrefixDataset(Dataset):
    """
    Char-level seq2seq dataset.
    Supports:
      - token_weight: per-char weights (we keep all-1 in Route A)
      - sample_weight: scalar per example
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

    def __getitem__(self, idx: int):
        src = self.src_list[idx]
        trg = self.trg_list[idx]

        trg_weights = [1.0] * (len(trg) + 2)  # SOS + chars + EOS

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
# 5) Transformer
# =============================================================================================

class ImprovedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
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
        self.positional_encoding = ImprovedPositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_seq_length)
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
        return self.output_linear(out)


# =============================================================================================
# 6) Decode (greedy + beam)  —— 修复变长 beam 的 pad/last_pos
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
    src = torch.tensor(src_idx, dtype=torch.long, device=device).unsqueeze(0)  # (1,S)
    src_pad_mask = src.eq(pad_idx)

    trg = torch.tensor([[sos_idx]], dtype=torch.long, device=device)  # (1,1)

    for _ in range(max_len):
        trg_pad_mask = trg.eq(pad_idx)
        logits = model(
            src=src,
            trg=trg,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask,
        )
        next_token = logits[:, -1, :].argmax(dim=-1)  # (1,)
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
    src_text: str,
    dataset: Regex2PrefixDataset,
    device: torch.device,
    beam_size: int = 5,
    max_len: int = 128,
) -> List[str]:
    model.eval()
    sos_idx = dataset.char_to_idx[SOS_TOKEN]
    eos_idx = dataset.char_to_idx[EOS_TOKEN]
    pad_idx = dataset.char_to_idx[PAD_TOKEN]

    src_tokens = [SOS_TOKEN] + list(src_text) + [EOS_TOKEN]
    src_idx = [dataset.char_to_idx[ch] for ch in src_tokens]
    src = torch.tensor(src_idx, dtype=torch.long, device=device).unsqueeze(0)  # (1,S)
    src_pad_mask = src.eq(pad_idx)

    beams: List[Tuple[List[int], float, bool]] = [([sos_idx], 0.0, False)]

    for _ in range(max_len):
        if all(f for _, _, f in beams):
            break

        seq_tensors = [torch.tensor(t, dtype=torch.long, device=device) for t, _, _ in beams]
        lengths = torch.tensor([len(t) for t, _, _ in beams], dtype=torch.long, device=device)
        trg = pad_sequence(seq_tensors, batch_first=True, padding_value=pad_idx)  # (B,Tm)
        trg_pad_mask = trg.eq(pad_idx)

        B = trg.size(0)
        src_rep = src.expand(B, -1)
        src_pad_rep = src_pad_mask.expand(B, -1)

        logits = model(
            src=src_rep,
            trg=trg,
            src_key_padding_mask=src_pad_rep,
            tgt_key_padding_mask=trg_pad_mask,
        )  # (B,Tm,V)

        last_pos = (lengths - 1).clamp(min=0)
        last_logits = logits[torch.arange(B, device=device), last_pos, :]  # (B,V)
        log_probs = torch.log_softmax(last_logits, dim=-1)

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

    # unique
    seen = set()
    uniq: List[str] = []
    for s in result_strings:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:beam_size]


# =============================================================================================
# 7) Loaders
# =============================================================================================

def _prefix_sample_weight_from_stats(
    rec: Dict[str, Any],
    tb_beta: float,
    ml_beta: float,
) -> float:
    """
    Optional: weight hard samples slightly more.
    Default: tb_beta=0, ml_beta=0 => all 1.0
    """
    w = 1.0
    stats = rec.get("stats", {})
    if tb_beta > 0:
        tb = stats.get("trgm_budget", None)
        if isinstance(tb, int) and tb >= 0:
            w *= 1.0 / ((tb + 1.0) ** tb_beta)
    if ml_beta > 0:
        ml = stats.get("max_literal_run_len", None)
        if isinstance(ml, int) and ml >= 0:
            w *= 1.0 / ((ml + 1.0) ** ml_beta)
    return float(w)


def load_raw_workload(
    path: str,
    src_field: str,
    seed_field: str,
    prefix_min_tokens: int,
    prefix_max_tokens: int,
    prefix_per_seed: int,
    include_trailing_punct: bool,
    strip_trailing_space: bool,
    max_src_len: int,
    max_trg_len: int,
    tb_beta: float,
    ml_beta: float,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[float], List[str], List[Set[str]]]:
    """
    Returns:
      example_src, example_trg, example_weight,
      group_srcs, group_gt_sets
    Grouping key = src (regex text).

    Modification:
      - If a seed_string has < prefix_min_tokens non-punct tokens, we DO NOT drop it.
        Instead we fallback to using k = len(non_punct) (i.e., take the longest possible prefix).
    """
    src_list: List[str] = []
    trg_list: List[str] = []
    w_list: List[float] = []

    group_map: Dict[str, Set[str]] = {}

    # optional counters for logging
    n_lines = 0
    n_kept = 0
    n_fallback_short = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading RAW"):
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            src = (rec.get(src_field, "") or "").strip()
            seed = (rec.get(seed_field, "") or "").strip()
            if not src or not seed:
                continue
            if len(src) > max_src_len:
                continue

            # determine how many non-punct tokens exist
            toks = [m.group(0) for m in TOKEN_RE.finditer(seed)]
            non_punct = [t for t in toks if not _is_punct(t)]
            if len(non_punct) == 0:
                continue

            # ======= key change: fallback for short sequences =======
            if len(non_punct) < int(prefix_min_tokens):
                # fallback: use the longest possible token-length prefix (full string up to last token)
                k_lo = len(non_punct)
                k_hi = len(non_punct)
                n_fallback_short += 1
            else:
                k_lo = max(1, int(prefix_min_tokens))
                k_hi = min(int(prefix_max_tokens), len(non_punct))

            if k_lo > k_hi:
                continue

            # choose k values
            ks = list(range(k_lo, k_hi + 1))
            if prefix_per_seed > 0 and len(ks) > prefix_per_seed:
                ks = rng.sample(ks, k=prefix_per_seed)

            base_w = _prefix_sample_weight_from_stats(rec, tb_beta=tb_beta, ml_beta=ml_beta)

            for k in ks:
                pref = extract_prefix_from_seed_string(
                    seed,
                    k_tokens=k,
                    include_trailing_punct=include_trailing_punct,
                    strip_trailing_space=strip_trailing_space,
                )
                if pref is None:
                    continue
                if len(pref) > max_trg_len:
                    continue

                src_list.append(src)
                trg_list.append(pref)
                w_list.append(base_w)
                n_kept += 1

                if src not in group_map:
                    group_map[src] = set()
                group_map[src].add(pref)

    group_srcs = list(group_map.keys())
    group_sets = [group_map[s] for s in group_srcs]

    print(
        f"[Data][RAW] lines={n_lines}  examples={len(src_list)}  groups(regex)={len(group_srcs)}  "
        f"fallback_short<{prefix_min_tokens}={n_fallback_short}"
    )
    return src_list, trg_list, w_list, group_srcs, group_sets



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
) -> Tuple[List[str], List[str], List[float], List[str], List[Set[str]]]:
    """
    Grouped format: each line has:
      - src_field (regex_norm / regex_canon / etc)
      - target_prefixes: list of dicts, each dict has "prefix" and optional "cand_est"/"support"/...

    Returns:
      example_src, example_trg, example_weight,
      group_srcs, group_gt_sets  (gt_sets built from ALL target_prefixes, not only picked)
    """
    src_list: List[str] = []
    trg_list: List[str] = []
    sw_list: List[float] = []

    group_srcs: List[str] = []
    group_sets: List[Set[str]] = []

    def rank_key(p: Dict):
        ce = p.get("cand_est", None)
        ce_key = int(ce) if isinstance(ce, int) else 10**18
        lc = int(p.get("len_chars", len(p.get("prefix", ""))))
        sr = float(p.get("support_ratio", 0.0))
        sup = int(p.get("support", 0))
        return (ce_key, -lc, -sr, -sup)

    def sample_weight(p: Dict, alpha: float) -> float:
        ce = p.get("cand_est", None)
        if isinstance(ce, int) and ce >= 0:
            return float(1.0 / ((ce + 1.0) ** alpha))
        return 1.0

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading GROUPED"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            src = (rec.get(src_field, "") or "").strip()
            if not src:
                # fallback
                src = (rec.get("regex_norm", "") or rec.get("regex", "") or "").strip()
            if not src or len(src) > max_src_len:
                continue

            prefs = rec.get("target_prefixes", None)
            if not isinstance(prefs, list) or len(prefs) == 0:
                continue

            # build gt set from ALL prefixes
            gt_set: Set[str] = set()
            for p in prefs:
                if not isinstance(p, dict):
                    continue
                pref = p.get("prefix", "")
                if isinstance(pref, str) and pref.strip():
                    gt_set.add(pref.rstrip() if strip_trailing_space else pref)
            if not gt_set:
                continue

            group_srcs.append(src)
            group_sets.append(gt_set)

            # choose training prefixes
            prefs_sorted = sorted([p for p in prefs if isinstance(p, dict)], key=rank_key)

            picked: List[Dict]
            if prefix_pick_mode == "random":
                picked = [rng.choice(prefs_sorted)]
            elif prefix_pick_mode == "best":
                picked = [prefs_sorted[0]]
            elif prefix_pick_mode == "topk":
                picked = prefs_sorted[:max(1, int(prefix_topk))]
            else:
                raise ValueError(prefix_pick_mode)

            for p in picked:
                pref = p.get("prefix", "")
                if not isinstance(pref, str) or not pref.strip():
                    continue
                if strip_trailing_space:
                    pref = pref.rstrip()
                if len(pref) > max_trg_len:
                    continue

                src_list.append(src)
                trg_list.append(pref)
                sw_list.append(sample_weight(p, alpha=weight_alpha))

    print(f"[Data][GROUPED] examples={len(src_list)}  groups={len(group_srcs)}")
    return src_list, trg_list, sw_list, group_srcs, group_sets


# =============================================================================================
# 8) Eval: group-level hit (matches your “存在可用性”目标)
# =============================================================================================

@torch.no_grad()
def evaluate_on_groups(
    model: nn.Module,
    dataset: Regex2PrefixDataset,
    group_srcs: List[str],
    group_gt_sets: List[Set[str]],
    device: torch.device,
    beam_size: int,
    max_groups: int,
    pg: Optional[PgProber],
    pg_gt_cap: int = 5,
) -> Dict[str, float]:
    model.eval()
    n = min(len(group_srcs), max_groups)
    if n <= 0:
        return {}

    greedy_exact = 0.0
    greedy_starts = 0.0
    greedy_len_ratio = 0.0
    beam_hit = 0.0

    pred_best_cand = []
    gt_best_cand = []

    for i in range(n):
        src = group_srcs[i]
        gt_set = group_gt_sets[i]

        pred = greedy_decode_prefix(model, src, dataset, device, max_len=128)
        greedy_exact += float(pred in gt_set)
        greedy_starts += float(any(g.startswith(pred) for g in gt_set if pred))

        gt_lens = sorted([len(g) for g in gt_set])
        gt_med_len = gt_lens[len(gt_lens) // 2] if gt_lens else 1
        greedy_len_ratio += (len(pred) / max(1, gt_med_len))

        beams = beam_search_decode_prefixes(model, src, dataset, device, beam_size=beam_size, max_len=128)
        beam_hit += float(any(b in gt_set for b in beams))

        # optional PG cand_est comparison: best-of-beam vs best-of-gt-set
        if pg is not None:
            cand_pred = []
            for b in beams:
                ce = pg.estimate_prefix_candidates(b)
                if ce is not None:
                    cand_pred.append(ce)
            if cand_pred:
                pred_best_cand.append(min(cand_pred))

            # gt can be large: probe only a few
            cand_gt = []
            gt_list = list(gt_set)
            if len(gt_list) > pg_gt_cap:
                gt_list = gt_list[:pg_gt_cap]
            for g in gt_list:
                ce = pg.estimate_prefix_candidates(g)
                if ce is not None:
                    cand_gt.append(ce)
            if cand_gt:
                gt_best_cand.append(min(cand_gt))

        if i < 5:
            show_gt = list(sorted(gt_set))[:5]
            print(f"\n[{i+1}] SRC(regex): {src}")
            print(f"    GT(prefix_set) size={len(gt_set)}  examples={show_gt}")
            print(f"    Pred(greedy): {pred}")
            print(f"    Beam@{beam_size}: {beams}")
            if pg is not None and pred_best_cand and gt_best_cand:
                print(f"    best_cand_est(pred/gt): {pred_best_cand[-1]} / {gt_best_cand[-1]}")

    out = {
        "groups": float(n),
        "greedy_exact": greedy_exact / n,
        "greedy_startswith_any_gt": greedy_starts / n,
        "greedy_len_ratio_vs_gt_median": greedy_len_ratio / n,
        f"beam@{beam_size}_hit_any_gt": beam_hit / n,
    }

    if pg is not None and pred_best_cand and gt_best_cand:
        out["cand_est_pred_best_median"] = float(statistics.median(pred_best_cand))
        out["cand_est_gt_best_median"] = float(statistics.median(gt_best_cand))
        out["cand_est_pred_over_gt_best_median"] = float(
            out["cand_est_pred_best_median"] / max(1.0, out["cand_est_gt_best_median"])
        )

    print("\n[Eval] Group Summary:")
    for k, v in out.items():
        print(f"  {k:34s}: {v:.4f}" if isinstance(v, float) else f"  {k:34s}: {v}")
    return out


# =============================================================================================
# 9) Training
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    full_dataset: Regex2PrefixDataset,
    group_srcs: List[str],
    group_gt_sets: List[Set[str]],
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

    pg = None
    if args.pg_dsn:
        pg = PgProber(args.pg_dsn, PgProbeConfig(
            table=args.pg_table,
            text_col=args.pg_col,
            probe_pct=args.probe_pct,
            probe_rounds=args.probe_rounds,
            debug=args.pg_debug,
        ))

    best_metric = -1.0
    best_ckpt_path: Optional[str] = None

    global_step = 0
    accum = 0
    optimizer.zero_grad(set_to_none=True)

    print(
        f"[Train] epochs={args.epochs}, lr(max)={args.lr}, batch={args.batch_size}, "
        f"accum={args.accumulation_steps}, eval_interval_steps={args.eval_interval_steps}"
    )
    print(f"[Data] train_examples={len(full_dataset)}  eval_groups={len(group_srcs)}")

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

                # apply sample weight to every token in the sample
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
                    print(f"\n[Train] step={global_step}, lr={scheduler.get_last_lr()[0]:.6e}  running GROUP eval...")

                    # evaluate on first max_eval_groups groups (already shuffled in main if needed)
                    metrics = evaluate_on_groups(
                        model=model,
                        dataset=full_dataset,
                        group_srcs=group_srcs,
                        group_gt_sets=group_gt_sets,
                        device=device,
                        beam_size=args.beam_size,
                        max_groups=args.max_eval_groups,
                        pg=pg,
                        pg_gt_cap=args.pg_gt_cap,
                    )

                    # primary metric: prefer better cand_est if available, else beam-hit
                    if metrics and ("cand_est_pred_over_gt_best_median" in metrics):
                        primary = -metrics["cand_est_pred_over_gt_best_median"]
                    else:
                        primary = metrics.get(f"beam@{args.beam_size}_hit_any_gt", 0.0) if metrics else 0.0

                    if primary > best_metric:
                        best_metric = primary
                        best_ckpt_path = os.path.join(save_root, f"best_step_{global_step}.pth")
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(f"[Save] New best at step {global_step}. primary={best_metric:.6f}")
                    else:
                        print(f"[Info] No improvement. best_primary={best_metric:.6f}")

            epoch_loss += float(loss.item()) * max(1, args.accumulation_steps)
            n_steps += 1

        # flush leftovers
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
# 10) CLI / main
# =============================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--input_format", choices=["raw", "grouped"], default="raw")

    # raw fields
    p.add_argument("--src_field", type=str, default="regex")
    p.add_argument("--seed_field", type=str, default="seed_string")

    # raw prefix extraction
    p.add_argument("--prefix_min_tokens", type=int, default=6)
    p.add_argument("--prefix_max_tokens", type=int, default=10)
    p.add_argument("--prefix_per_seed", type=int, default=2,
                   help="0=use all k in [min,max]; otherwise sample this many ks per seed line.")
    p.add_argument("--include_trailing_punct", action="store_true", default=True)
    p.add_argument("--strip_trailing_space", action="store_true", default=True)

    # optional weighting from stats
    p.add_argument("--tb_beta", type=float, default=0.0, help="weight by 1/(trgm_budget+1)^tb_beta")
    p.add_argument("--ml_beta", type=float, default=0.0, help="weight by 1/(max_lit_run+1)^ml_beta")

    # grouped loader controls
    p.add_argument("--prefix_pick_mode", choices=["best", "topk", "random"], default="topk")
    p.add_argument("--prefix_topk", type=int, default=2)
    p.add_argument("--weight_alpha", type=float, default=0.5, help="cand_est weight exponent in grouped mode.")

    # length filters
    p.add_argument("--max_src_len", type=int, default=500)
    p.add_argument("--max_trg_len", type=int, default=160)

    # model
    p.add_argument("--HIDDEN_SIZE", type=int, default=512)
    p.add_argument("--LayerNum", type=int, default=4)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=1024)

    # train
    p.add_argument("--GPU", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--saveName", type=str, default="usacc_routeA_v2")

    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--eval_interval_steps", type=int, default=1000)
    p.add_argument("--max_eval_groups", type=int, default=256)
    p.add_argument("--beam_size", type=int, default=8)

    # OneCycleLR
    p.add_argument("--pct_start", type=float, default=0.15)
    p.add_argument("--final_div_factor", type=float, default=100.0)

    # dataloader
    p.add_argument("--num_workers", type=int, default=4)

    # optional PG eval
    p.add_argument("--pg_dsn", type=str, default="")
    p.add_argument("--pg_table", type=str, default="us_accidents")
    p.add_argument("--pg_col", type=str, default="description")
    p.add_argument("--probe_pct", type=float, default=0.05)
    p.add_argument("--probe_rounds", type=int, default=3)
    p.add_argument("--pg_gt_cap", type=int, default=5, help="cap #gt prefixes probed per group")
    p.add_argument("--pg_debug", action="store_true", default=False)

    # eval group order
    p.add_argument("--shuffle_groups_for_eval", action="store_true", default=True)

    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = setup_device(args.GPU)
    rng = random.Random(args.seed)

    if args.input_format == "raw":
        src_list, trg_list, sw_list, group_srcs, group_sets = load_raw_workload(
            path=args.data_path,
            src_field=args.src_field,
            seed_field=args.seed_field,
            prefix_min_tokens=args.prefix_min_tokens,
            prefix_max_tokens=args.prefix_max_tokens,
            prefix_per_seed=args.prefix_per_seed,
            include_trailing_punct=args.include_trailing_punct,
            strip_trailing_space=args.strip_trailing_space,
            max_src_len=args.max_src_len,
            max_trg_len=args.max_trg_len,
            tb_beta=args.tb_beta,
            ml_beta=args.ml_beta,
            rng=rng,
        )
    else:
        src_list, trg_list, sw_list, group_srcs, group_sets = load_grouped_workload(
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

    if len(src_list) == 0 or len(group_srcs) == 0:
        raise SystemExit("[Error] No usable data loaded.")

    # shuffle groups for eval so we don't always evaluate the same head subset
    if args.shuffle_groups_for_eval:
        order = list(range(len(group_srcs)))
        rng.shuffle(order)
        group_srcs = [group_srcs[i] for i in order]
        group_sets = [group_sets[i] for i in order]

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
        group_srcs=group_srcs,
        group_gt_sets=group_sets,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
