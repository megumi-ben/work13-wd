#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证同一个 seed_string 是否能生成不同的 regex
"""

import random
import re
import math
from typing import List, Optional, Tuple

# =====================================================================================
# Tokenization
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


# =====================================================================================
# Regex 生成相关函数
# =====================================================================================

def gap_pattern(N: int) -> str:
    N = max(0, int(N))
    return rf"(?:.|\n){{0,{N}}}?"


def _is_structured_token(tok: str) -> bool:
    if not tok or is_punct(tok):
        return False
    return (DIGIT_RE.search(tok) is not None) or ("-" in tok)


def _render_structured_token(tok: str) -> str:
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
    """动态生成 regex"""
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
            score = 1.0 + 0.1 * e + rng.random() * 0.5  # ★ 随机扰动
            candidates.append((i, e, pattern, score, "struct", tok, None, None))
        
        # 候选 2: n-gram anchor
        if ALPHA_RE.fullmatch(tok):
            ngrams = _extract_ngrams(tok, n_gram)
            if ngrams:
                start, gram = rng.choice(ngrams)  # ★ 随机选择 n-gram
                pattern = _render_ngram_anchor(tok, gram, start, n_gram, anchor_slack)
                score = 1.0 + 0.05 * len(tok) + rng.random() * 0.5  # ★ 随机扰动
                candidates.append((i, n_gram, pattern, score, "ngram", tok, gram, start))
    
    if not candidates:
        return None
    
    # 按 score 排序
    candidates.sort(key=lambda x: x[3], reverse=True)
    
    # 贪心选择 anchor
    chosen = []
    used_positions = set()
    exposure_sum = 0
    
    for pos, expose_len, pattern, score, kind, tok, gram, start in candidates:
        if pos in used_positions:
            continue
        if len(chosen) >= k_max:
            break
        if exposure_sum >= budget:
            break
        
        chosen.append((pos, pattern, kind, tok, gram, start))
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
    
    for pos, pattern, kind, tok, gram, start in chosen[1:]:
        dist = pos - prev_pos - 1
        gap_len = min(gap_cap, gap_base + gap_per_token * max(0, dist))
        parts.append(gap_pattern(gap_len))
        parts.append(pattern)
        prev_pos = pos
    
    regex = "".join(parts)
    
    # 返回详细信息用于调试
    return regex, chosen


# =====================================================================================
# 测试
# =====================================================================================

def test_single_seed(seed_str: str, n_gram: int = 3, num_trials: int = 10):
    """测试单个 seed_string"""
    tokens = tokenize(seed_str)
    
    print("=" * 80)
    print(f"Seed String: {seed_str}")
    print(f"Tokens: {tokens}")
    print(f"N-gram: {n_gram}")
    print("=" * 80)
    
    regexes = []
    
    for i in range(num_trials):
        rng = random.Random(42 + i * 1000003)  # 每次不同的随机种子
        result = generate_regex_from_seed(
            seed=seed_str,
            tokens=tokens,
            rng=rng,
            n_gram=n_gram,
            expose_ratio=0.3,
            k_max=6,
        )
        
        if result is None:
            print(f"\n[Trial {i+1}] Failed to generate regex")
            continue
        
        regex, chosen = result
        regexes.append(regex)
        
        print(f"\n[Trial {i+1}]")
        print(f"  Regex: {regex}")
        print(f"  Anchors:")
        for pos, pattern, kind, tok, gram, start in chosen:
            if kind == "ngram":
                print(f"    pos={pos}, token='{tok}', ngram='{gram}' at position {start}")
            else:
                print(f"    pos={pos}, token='{tok}', kind={kind}")
    
    # 统计
    unique_regexes = set(regexes)
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Total trials: {num_trials}")
    print(f"  Unique regexes: {len(unique_regexes)}")
    print(f"  Diversity: {len(unique_regexes) / num_trials * 100:.1f}%")
    print("=" * 80)
    
    return unique_regexes


def main():
    # 测试用例
    test_cases = [
        "Richard B. Davies Jr.",
        "Order 12345 shipped on 2024-01-15",
        "Hello world, this is a simple test.",
        "The quick brown fox jumps over the lazy dog.",
        "Invoice INV-2024-00123 for customer ABC-Corp",
    ]
    
    print("\n" + "#" * 80)
    print("# 测试：同一个 seed_string 是否能生成不同的 regex")
    print("#" * 80)
    
    for seed_str in test_cases:
        test_single_seed(seed_str, n_gram=3, num_trials=10)
        print("\n")
    
    # 对比 n_gram=2 和 n_gram=3
    print("\n" + "#" * 80)
    print("# 对比：n_gram=2 vs n_gram=3")
    print("#" * 80)
    
    seed_str = "Richard B. Davies Jr."
    print(f"\nSeed: {seed_str}")
    
    print("\n--- n_gram=2 ---")
    test_single_seed(seed_str, n_gram=2, num_trials=10)
    
    print("\n--- n_gram=3 ---")
    test_single_seed(seed_str, n_gram=3, num_trials=10)


if __name__ == "__main__":
    main()