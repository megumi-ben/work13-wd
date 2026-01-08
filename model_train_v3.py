#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regex -> Prefix 训练代码（动态数据生成版本）

核心特性：
1. 每次采样同一个 seed_string 时，动态生成不同的 regex
2. 每次采样同一个 seed_string 时，动态生成不同长度的 prefix
3. 分片遍历保证数据覆盖（N条数据，每epoch用M条，N/M个epoch遍历完）

使用示例：
    python train_regex2prefix_dynamic.py \
        --data_path ./data/seeds.jsonl \
        --samples_per_epoch 5000 \
        --n_gram 3 \
        --expose_ratio 0.3 \
        --epochs 500 \
        --batch_size 64 \
        --GPU 0 \
        --save_dir ./models/my_model
"""

import argparse
import json
import math
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =====================================================================================
# 1) Seed / Device
# =====================================================================================

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


# =====================================================================================
# 2) Tokenization
# =====================================================================================

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
ALPHA_RE = re.compile(r"^[A-Za-z]+$")
DIGIT_RE = re.compile(r"\d")
WORDCHAR_RE = re.compile(r"[A-Za-z0-9_]")
WORD_CH = r"[A-Za-z0-9_]"


def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(s or "")


def is_punct(tok: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]", tok or ""))


def wordchar_len(s: str) -> int:
    return len(WORDCHAR_RE.findall(s or ""))


def count_non_punct_tokens(s: str) -> int:
    matches = TOKEN_RE.findall(s or "")
    return sum(1 for tok in matches if not is_punct(tok))


# =====================================================================================
# 3) Prefix 提取
# =====================================================================================

def extract_prefix_from_seed_string(
    s: str,
    k_tokens: int,
    include_trailing_punct: bool = True,
    strip_trailing_space: bool = True,
) -> Optional[str]:
    """从 seed_string 中提取前 k 个非标点 token 构成的前缀"""
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
        if is_punct(tok):
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
            if is_punct(tok):
                end = matches[j].end()
                j += 1
                continue
            break

    pref = s[:end]
    if strip_trailing_space:
        pref = pref.rstrip()
    return pref if pref.strip() else None


# =====================================================================================
# 4) Regex 动态生成
# =====================================================================================

def gap_pattern(N: int) -> str:
    """生成间隔模式"""
    N = max(0, int(N))
    return rf"(?:.|\n){{0,{N}}}?"


def _is_structured_token(tok: str) -> bool:
    """判断是否为结构化 token（包含数字或连字符）"""
    if not tok or is_punct(tok):
        return False
    return (DIGIT_RE.search(tok) is not None) or ("-" in tok)


def _render_structured_token(tok: str) -> str:
    """渲染结构化 token 的正则模式"""
    if not tok:
        return rf"\y{WORD_CH}+\y"
    
    out_parts = []
    i = 0
    while i < len(tok):
        ch = tok[i]
        if ch.isdigit():
            j = i + 1
            while j < len(tok) and tok[j].isdigit():
                j += 1
            k = j - i
            out_parts.append(rf"[0-9]{{{k}}}")
            i = j
        else:
            out_parts.append(re.escape(ch))
            i += 1
    
    body = "".join(out_parts)
    return rf"\y{body}\y"


def _render_ngram_anchor(tok: str, gram: str, start: int, n: int, slack: int = 1) -> str:
    """渲染 n-gram 锚点的正则模式"""
    if not tok or not gram or n <= 0:
        return rf"\y{WORD_CH}+\y"
    
    L = len(tok)
    pre_len = start
    suf_len = L - (start + n)
    
    lo_pre = max(0, pre_len - slack)
    hi_pre = pre_len + slack
    lo_suf = max(0, suf_len - slack)
    hi_suf = suf_len + slack
    
    def render_repeat(lo, hi):
        if hi <= 0:
            return ""
        if lo == hi:
            if lo == 0:
                return ""
            if lo == 1:
                return WORD_CH
            return f"{WORD_CH}{{{lo}}}"
        return f"{WORD_CH}{{{lo},{hi}}}"
    
    pre = render_repeat(lo_pre, hi_pre)
    suf = render_repeat(lo_suf, hi_suf)
    gram_escaped = re.escape(gram)
    
    return rf"\y{pre}{gram_escaped}{suf}\y"


def _extract_ngrams(tok: str, n: int) -> List[Tuple[int, str]]:
    """从 token 中提取所有 n-gram"""
    if not tok or not ALPHA_RE.fullmatch(tok):
        return []
    if len(tok) < n:
        return []
    
    ngrams = []
    for i in range(len(tok) - n + 1):
        gram = tok[i:i + n]
        ngrams.append((i, gram))
    return ngrams


def generate_regex_from_seed(
    seed: str,
    tokens: List[str],
    rng: random.Random,
    n_gram: int = 3,
    expose_ratio: float = 0.3,
    k_max: int = 6,
    gap_base: int = 30,
    gap_per_token: int = 6,
    gap_cap: int = 120,
    anchor_slack: int = 1,
    case_insensitive: bool = True,
) -> Optional[str]:
    """
    从 seed_string 动态生成 regex
    
    随机性来源：
    1. 随机选择哪些位置作为 anchor
    2. 随机选择 n-gram
    3. Score 随机扰动影响排序
    """
    if not tokens:
        return None
    
    W = wordchar_len(seed)
    if W <= 0:
        return None
    
    budget = int(math.ceil(expose_ratio * W))
    budget = max(1, budget)
    
    # 收集候选 anchor
    candidates = []
    
    for i, tok in enumerate(tokens):
        if is_punct(tok):
            continue
        
        # 候选 1: 结构化 token
        if _is_structured_token(tok):
            e = wordchar_len(tok)
            pattern = _render_structured_token(tok)
            score = 1.0 + 0.1 * e + rng.random() * 0.5
            candidates.append((i, e, pattern, score, "struct"))
        
        # 候选 2: n-gram anchor
        if ALPHA_RE.fullmatch(tok):
            ngrams = _extract_ngrams(tok, n_gram)
            if ngrams:
                start, gram = rng.choice(ngrams)  # 随机选择 n-gram
                pattern = _render_ngram_anchor(tok, gram, start, n_gram, anchor_slack)
                score = 1.0 + 0.05 * len(tok) + rng.random() * 0.5
                candidates.append((i, n_gram, pattern, score, "ngram"))
    
    if not candidates:
        return None
    
    # 按 score 排序
    candidates.sort(key=lambda x: x[3], reverse=True)
    
    # 贪心选择 anchor
    chosen = []
    used_positions = set()
    exposure_sum = 0
    
    for pos, expose_len, pattern, score, kind in candidates:
        if pos in used_positions:
            continue
        if len(chosen) >= k_max:
            break
        if exposure_sum >= budget:
            break
        
        chosen.append((pos, pattern))
        used_positions.add(pos)
        exposure_sum += expose_len
    
    if not chosen:
        return None
    
    # 按位置排序
    chosen.sort(key=lambda x: x[0])
    
    # 组装 regex
    parts = []
    if case_insensitive:
        parts.append("(?i)")
    
    parts.append(chosen[0][1])
    prev_pos = chosen[0][0]
    
    for pos, pattern in chosen[1:]:
        dist = pos - prev_pos - 1
        gap_len = min(gap_cap, gap_base + gap_per_token * max(0, dist))
        parts.append(gap_pattern(gap_len))
        parts.append(pattern)
        prev_pos = pos
    
    return "".join(parts)


# =====================================================================================
# 5) 数据加载
# =====================================================================================

def load_seed_strings(
    path: str,
    seed_field: str = "seed_string",
    max_seed_len: int = 2000,
    min_tokens: int = 3,
) -> List[Dict[str, Any]]:
    """
    加载 seed_string 数据
    支持 JSONL 格式和纯文本格式
    """
    records = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading seed strings"):
            line = line.strip()
            if not line:
                continue
            
            # 尝试 JSON 格式
            try:
                obj = json.loads(line)
                seed = (obj.get(seed_field, "") or "").strip()
            except json.JSONDecodeError:
                # 纯文本格式
                seed = line
            
            if not seed or len(seed) > max_seed_len:
                continue
            
            tokens = tokenize(seed)
            n_tokens = sum(1 for t in tokens if not is_punct(t))
            
            if n_tokens < min_tokens:
                continue
            
            records.append({
                "seed_string": seed,
                "tokens": tokens,
                "n_tokens": n_tokens,
            })
    
    print(f"[load_seed_strings] Loaded {len(records)} records")
    return records


# =====================================================================================
# 6) 数据调度器
# =====================================================================================

class DynamicDataScheduler:
    """
    数据调度器：
    1. 分片遍历保证覆盖（N条数据，每epoch用M条，N/M个epoch遍历完）
    2. 每次采样同一个 seed_string 时，动态生成不同的 regex 和 prefix
    """
    
    def __init__(
        self,
        records: List[Dict[str, Any]],
        samples_per_epoch: int,
        # regex 生成参数
        n_gram: int = 3,
        expose_ratio: float = 0.3,
        k_max: int = 6,
        gap_base: int = 30,
        gap_per_token: int = 6,
        gap_cap: int = 120,
        anchor_slack: int = 1,
        case_insensitive: bool = True,
        # prefix 生成参数
        prefix_min_tokens: int = 4,
        prefix_max_tokens: int = 10,
        max_trg_len: int = 160,
        # 其他
        base_seed: int = 42,
    ):
        self.records = records
        self.samples_per_epoch = samples_per_epoch
        
        # regex 参数
        self.n_gram = n_gram
        self.expose_ratio = expose_ratio
        self.k_max = k_max
        self.gap_base = gap_base
        self.gap_per_token = gap_per_token
        self.gap_cap = gap_cap
        self.anchor_slack = anchor_slack
        self.case_insensitive = case_insensitive
        
        # prefix 参数
        self.prefix_min_tokens = prefix_min_tokens
        self.prefix_max_tokens = prefix_max_tokens
        self.max_trg_len = max_trg_len
        
        self.base_seed = base_seed
        self.n_total = len(records)
        self.n_chunks = math.ceil(self.n_total / samples_per_epoch)
        
        self._current_round = -1
        self._shuffled_indices: List[int] = []
        
        print(f"[Scheduler] Total records: {self.n_total}")
        print(f"[Scheduler] Samples per epoch: {samples_per_epoch}")
        print(f"[Scheduler] Epochs per round: {self.n_chunks}")
        print(f"[Scheduler] Regex params: n_gram={n_gram}, expose_ratio={expose_ratio}, k_max={k_max}")
    
    def _shuffle_for_round(self, round_idx: int) -> None:
        """为新一轮 shuffle 数据"""
        if round_idx != self._current_round:
            rng = random.Random(self.base_seed + round_idx * 999983)
            self._shuffled_indices = list(range(self.n_total))
            rng.shuffle(self._shuffled_indices)
            self._current_round = round_idx
    
    def get_epoch_data(self, epoch: int) -> Tuple[List[str], List[str]]:
        """
        获取指定 epoch 的训练数据
        
        Args:
            epoch: 从 1 开始的 epoch 编号
            
        Returns:
            src_list: regex 列表（动态生成）
            trg_list: prefix 列表（动态生成）
        """
        epoch_0 = epoch - 1
        round_idx = epoch_0 // self.n_chunks
        chunk_idx = epoch_0 % self.n_chunks
        
        self._shuffle_for_round(round_idx)
        
        start = chunk_idx * self.samples_per_epoch
        end = min(start + self.samples_per_epoch, self.n_total)
        chunk_indices = self._shuffled_indices[start:end]
        
        # 本 epoch 的随机数生成器
        epoch_rng = random.Random(self.base_seed + epoch * 1000003)
        
        src_list: List[str] = []
        trg_list: List[str] = []
        
        for idx in chunk_indices:
            rec = self.records[idx]
            seed = rec["seed_string"]
            tokens = rec["tokens"]
            n_tokens = rec["n_tokens"]
            
            # 动态生成 regex
            regex = generate_regex_from_seed(
                seed=seed,
                tokens=tokens,
                rng=epoch_rng,
                n_gram=self.n_gram,
                expose_ratio=self.expose_ratio,
                k_max=self.k_max,
                gap_base=self.gap_base,
                gap_per_token=self.gap_per_token,
                gap_cap=self.gap_cap,
                anchor_slack=self.anchor_slack,
                case_insensitive=self.case_insensitive,
            )
            
            if regex is None:
                continue
            
            # 动态生成 prefix
            k_lo = max(1, self.prefix_min_tokens)
            k_hi = min(self.prefix_max_tokens, n_tokens)
            if n_tokens < self.prefix_min_tokens:
                k_lo = n_tokens
                k_hi = n_tokens
            
            if k_lo > k_hi:
                continue
            
            k = epoch_rng.randint(k_lo, k_hi)
            prefix = extract_prefix_from_seed_string(seed, k_tokens=k)
            
            if prefix is None or len(prefix) > self.max_trg_len:
                continue
            
            src_list.append(regex)
            trg_list.append(prefix)
        
        print(f"[Epoch {epoch}] Round {round_idx+1}, Chunk {chunk_idx+1}/{self.n_chunks}, "
              f"Generated {len(src_list)} pairs")
        
        return src_list, trg_list


# =====================================================================================
# 7) 评估数据构建
# =====================================================================================

def build_eval_data(
    records: List[Dict[str, Any]],
    n_gram: int,
    expose_ratio: float,
    k_max: int,
    prefix_min_tokens: int,
    prefix_max_tokens: int,
    max_trg_len: int,
    max_eval_samples: int = 256,
    seed: int = 42,
) -> Tuple[List[str], List[Set[str]]]:
    """构建评估数据：为每个 seed_string 生成固定的 regex 和所有可能的 prefix"""
    rng = random.Random(seed)
    
    indices = list(range(len(records)))
    rng.shuffle(indices)
    indices = indices[:max_eval_samples]
    
    group_srcs = []
    group_gt_sets = []
    
    for idx in indices:
        rec = records[idx]
        seed_str = rec["seed_string"]
        tokens = rec["tokens"]
        n_tokens = rec["n_tokens"]
        
        # 生成固定的 regex
        regex = generate_regex_from_seed(
            seed=seed_str,
            tokens=tokens,
            rng=random.Random(seed + idx),
            n_gram=n_gram,
            expose_ratio=expose_ratio,
            k_max=k_max,
        )
        
        if regex is None:
            continue
        
        # 生成所有可能的 prefix
        gt_set = set()
        k_lo = max(1, min(prefix_min_tokens, n_tokens))
        k_hi = min(prefix_max_tokens, n_tokens)
        
        for k in range(k_lo, k_hi + 1):
            prefix = extract_prefix_from_seed_string(seed_str, k)
            if prefix and len(prefix) <= max_trg_len:
                gt_set.add(prefix)
        
        if gt_set:
            group_srcs.append(regex)
            group_gt_sets.append(gt_set)
    
    avg_size = sum(len(s) for s in group_gt_sets) / max(1, len(group_gt_sets))
    print(f"[Eval] {len(group_srcs)} samples, avg gt_set size: {avg_size:.1f}")
    
    return group_srcs, group_gt_sets


# =====================================================================================
# 8) 词表构建
# =====================================================================================

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"


def build_vocab_from_seeds(records: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """从 seed_string 构建词表"""
    unique_chars = set()
    
    for rec in tqdm(records, desc="Building vocab"):
        unique_chars.update(list(rec["seed_string"]))
    
    # regex 中可能出现的特殊字符
    regex_chars = set(r"()[]{}|?*+.\^$yin0-9A-Za-z_:")
    unique_chars.update(regex_chars)
    
    char_to_idx = {SOS_TOKEN: 0, EOS_TOKEN: 1, PAD_TOKEN: 2}
    for idx, ch in enumerate(sorted(unique_chars), start=3):
        char_to_idx[ch] = idx
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
    
    print(f"[Vocab] Size: {len(char_to_idx)}")
    return char_to_idx, idx_to_char


# =====================================================================================
# 9) Dataset
# =====================================================================================

class Regex2PrefixDataset(Dataset):
    def __init__(self, src_list, trg_list, char_to_idx, idx_to_char):
        self.src_list = src_list
        self.trg_list = trg_list
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        src, trg = self.src_list[idx], self.trg_list[idx]
        pad_idx = self.char_to_idx[PAD_TOKEN]
        
        src_idx = [self.char_to_idx[SOS_TOKEN]] + \
                  [self.char_to_idx.get(c, pad_idx) for c in src] + \
                  [self.char_to_idx[EOS_TOKEN]]
        trg_idx = [self.char_to_idx[SOS_TOKEN]] + \
                  [self.char_to_idx.get(c, pad_idx) for c in trg] + \
                  [self.char_to_idx[EOS_TOKEN]]
        
        return torch.tensor(src_idx, dtype=torch.long), torch.tensor(trg_idx, dtype=torch.long)


def make_collate_fn(pad_idx):
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        return (pad_sequence(src_batch, batch_first=True, padding_value=pad_idx),
                pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx))
    return collate_fn


# =====================================================================================
# 10) Transformer 模型
# =====================================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].permute(1, 0, 2)
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.trg_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_seq_len)
        self.transformer = nn.Transformer(
            d_model, nhead, num_layers, num_layers, dim_feedforward,
            dropout, batch_first=True, norm_first=True
        )
        self.out_linear = nn.Linear(d_model, vocab_size)
        self._init()

    def _init(self):
        for emb in [self.src_emb, self.trg_emb]:
            nn.init.normal_(emb.weight, 0, self.d_model ** -0.5)
        nn.init.xavier_uniform_(self.out_linear.weight)

    def forward(self, src, trg, src_pad_mask=None, trg_pad_mask=None):
        src_emb = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_enc(self.trg_emb(trg) * math.sqrt(self.d_model))
        tgt_mask = torch.triu(torch.ones(trg.size(1), trg.size(1), device=trg.device), 1).bool()
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask, float('-inf'))
        out = self.transformer(src_emb, trg_emb, tgt_mask=tgt_mask,
                              src_key_padding_mask=src_pad_mask,
                              tgt_key_padding_mask=trg_pad_mask,
                              memory_key_padding_mask=src_pad_mask)
        return self.out_linear(out)


# =====================================================================================
# 11) Decode
# =====================================================================================

@torch.no_grad()
def greedy_decode(model, src_text, char_to_idx, idx_to_char, device, max_len=128):
    model.eval()
    sos, eos, pad = char_to_idx[SOS_TOKEN], char_to_idx[EOS_TOKEN], char_to_idx[PAD_TOKEN]
    src_idx = [sos] + [char_to_idx.get(c, pad) for c in src_text] + [eos]
    src = torch.tensor([src_idx], device=device)
    trg = torch.tensor([[sos]], device=device)
    
    for _ in range(max_len):
        logits = model(src, trg, src.eq(pad), trg.eq(pad))
        next_id = logits[:, -1, :].argmax(-1).item()
        trg = torch.cat([trg, torch.tensor([[next_id]], device=device)], 1)
        if next_id == eos:
            break
    return "".join(idx_to_char.get(i, "") for i in trg[0].tolist()[1:] if i != eos)


@torch.no_grad()
def beam_decode(model, src_text, char_to_idx, idx_to_char, device, beam_size=5, max_len=128):
    model.eval()
    sos, eos, pad = char_to_idx[SOS_TOKEN], char_to_idx[EOS_TOKEN], char_to_idx[PAD_TOKEN]
    src_idx = [sos] + [char_to_idx.get(c, pad) for c in src_text] + [eos]
    src = torch.tensor([src_idx], device=device)
    
    beams = [([sos], 0.0, False)]
    for _ in range(max_len):
        if all(f for _, _, f in beams):
            break
        cands = []
        for tokens, score, finished in beams:
            if finished:
                cands.append((tokens, score, True))
                continue
            trg = torch.tensor([tokens], device=device)
            logits = model(src, trg, src.eq(pad), trg.eq(pad))
            log_probs = torch.log_softmax(logits[0, -1], -1)
            topk_lp, topk_idx = torch.topk(log_probs, beam_size)
            for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                cands.append((tokens + [idx], score + lp, idx == eos))
        cands.sort(key=lambda x: x[1], reverse=True)
        beams = cands[:beam_size]
    
    results = []
    for tokens, _, _ in beams:
        s = "".join(idx_to_char.get(i, "") for i in tokens[1:] if i != eos)
        if s not in results:
            results.append(s)
    return results[:beam_size]


# =====================================================================================
# 12) Evaluate
# =====================================================================================

@torch.no_grad()
def evaluate(model, char_to_idx, idx_to_char, group_srcs, group_gt_sets, device, beam_size=8):
    model.eval()
    n = len(group_srcs)
    greedy_hit, beam_hit = 0.0, 0.0
    
    for i in range(n):
        src, gt_set = group_srcs[i], group_gt_sets[i]
        pred = greedy_decode(model, src, char_to_idx, idx_to_char, device)
        greedy_hit += float(pred in gt_set)
        beams = beam_decode(model, src, char_to_idx, idx_to_char, device, beam_size)
        beam_hit += float(any(b in gt_set for b in beams))
        
        if i < 2:
            print(f"\n[Eval {i+1}] Regex: {src[:60]}...")
            print(f"  GT({len(gt_set)}): {list(gt_set)[:2]}")
            print(f"  Greedy: {pred}")
            print(f"  Beam: {beams[:2]}")
    
    metrics = {"greedy_hit": greedy_hit/n, f"beam@{beam_size}_hit": beam_hit/n}
    print(f"\n[Eval] {metrics}")
    return metrics


# =====================================================================================
# 13) Training
# =====================================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, pad_idx, grad_clip, scaler):
    model.train()
    total_loss, n_batch = 0.0, 0
    
    for src, trg in tqdm(loader, desc="Training", leave=False):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        
        with amp.autocast(enabled=scaler is not None):
            logits = model(src, trg[:, :-1], src.eq(pad_idx), trg[:, :-1].eq(pad_idx))
            loss = criterion(logits.reshape(-1, logits.size(-1)), trg[:, 1:].reshape(-1))
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        n_batch += 1
    
    return total_loss / max(1, n_batch)


# =====================================================================================
# 14) Main
# =====================================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train Regex->Prefix model with dynamic data generation")
    
    # 数据
    p.add_argument("--data_path", type=str, required=True, help="Path to seed strings (JSONL or text)")
    p.add_argument("--seed_field", type=str, default="seed_string")
    p.add_argument("--samples_per_epoch", type=int, default=10000,
                   help="Samples per epoch, -1 for all data")
    
    # Regex 生成参数
    p.add_argument("--n_gram", type=int, default=3, choices=[2, 3])
    p.add_argument("--expose_ratio", type=float, default=0.3)
    p.add_argument("--k_max", type=int, default=6)
    p.add_argument("--gap_base", type=int, default=30)
    p.add_argument("--gap_per_token", type=int, default=6)
    p.add_argument("--gap_cap", type=int, default=120)
    p.add_argument("--anchor_slack", type=int, default=1)
    
    # Prefix 生成参数
    p.add_argument("--prefix_min_tokens", type=int, default=4)
    p.add_argument("--prefix_max_tokens", type=int, default=10)
    p.add_argument("--max_trg_len", type=int, default=160)
    
    # 模型
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=1024)
    
    # 训练
    p.add_argument("--GPU", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    
    # 评估与保存
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--max_eval_samples", type=int, default=256)
    p.add_argument("--beam_size", type=int, default=8)
    p.add_argument("--save_dir", type=str, default="./models/regex2prefix")
    p.add_argument("--num_workers", type=int, default=4)
    
    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = setup_device(args.GPU)

    print("\n" + "=" * 70)
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 70 + "\n")

    # 1. 加载 seed strings
    records = load_seed_strings(args.data_path, args.seed_field)
    if not records:
        raise SystemExit("[Error] No data loaded.")

    # 2. 构建词表
    char_to_idx, idx_to_char = build_vocab_from_seeds(records)
    pad_idx = char_to_idx[PAD_TOKEN]

    # 3. 数据调度器
    samples_per_epoch = args.samples_per_epoch if args.samples_per_epoch > 0 else len(records)
    scheduler = DynamicDataScheduler(
        records=records,
        samples_per_epoch=samples_per_epoch,
        n_gram=args.n_gram,
        expose_ratio=args.expose_ratio,
        k_max=args.k_max,
        gap_base=args.gap_base,
        gap_per_token=args.gap_per_token,
        gap_cap=args.gap_cap,
        anchor_slack=args.anchor_slack,
        prefix_min_tokens=args.prefix_min_tokens,
        prefix_max_tokens=args.prefix_max_tokens,
        max_trg_len=args.max_trg_len,
        base_seed=args.seed,
    )

    # 4. 评估数据
    group_srcs, group_gt_sets = build_eval_data(
        records, args.n_gram, args.expose_ratio, args.k_max,
        args.prefix_min_tokens, args.prefix_max_tokens, args.max_trg_len,
        args.max_eval_samples, args.seed
    )

    # 5. 模型
    model = TransformerSeq2Seq(
        len(char_to_idx), args.d_model, args.nhead, args.num_layers,
        args.dim_feedforward, args.max_seq_len, args.dropout
    ).to(device)
    print(f"[Model] {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler() if device.type == "cuda" else None
    os.makedirs(args.save_dir, exist_ok=True)

    best_metric, best_epoch = 0.0, 0

    # 6. 训练循环
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # 动态生成数据
        src_list, trg_list = scheduler.get_epoch_data(epoch)
        
        dataset = Regex2PrefixDataset(src_list, trg_list, char_to_idx, idx_to_char)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=make_collate_fn(pad_idx), num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        
        loss = train_one_epoch(model, loader, optimizer, criterion, device, pad_idx, args.grad_clip, scaler)
        print(f"[Epoch {epoch}] Loss: {loss:.4f}, Time: {time.time()-t0:.1f}s")
        
        # 评估
        if epoch % args.eval_interval == 0:
            metrics = evaluate(model, char_to_idx, idx_to_char, group_srcs, group_gt_sets, device, args.beam_size)
            if metrics[f"beam@{args.beam_size}_hit"] > best_metric:
                best_metric = metrics[f"beam@{args.beam_size}_hit"]
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "vocab": char_to_idx,
                    "args": vars(args),
                    "metrics": metrics,
                }, os.path.join(args.save_dir, "best.pth"))
                print(f"[Save] New best at epoch {epoch}, beam_hit={best_metric:.4f}")
        
        # 定期保存
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "vocab": char_to_idx,
                "args": vars(args),
            }, os.path.join(args.save_dir, f"ckpt_{epoch}.pth"))

    print("\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Best model: epoch {best_epoch}, beam@{args.beam_size}_hit = {best_metric:.4f}")
    print(f"Saved to: {args.save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
