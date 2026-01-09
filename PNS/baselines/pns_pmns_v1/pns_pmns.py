"""Baseline PNS/PMNS-style verifier wired to regex_span."""
import bisect
import heapq
import logging
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .anchors import parse_structured_regex
from .verify_pg import find_all, find_from, verify_windows
from .literal_search import find_all_literals, occurs_in_window, OccurrenceCache
from .factor_extractor import extract_factors, Factors
from .nfactor_lite import NFactorEngine

Logger = logging.getLogger(__name__)


class PositionProvider:
    """Abstract interface for supplying factor hit positions."""

    def get_LPs(self, txt: str, pattern: str, lmin: int, factors: Optional[Dict]) -> Tuple[List[int], Dict]:
        raise NotImplementedError

    def get_LSs(self, txt: str, pattern: str, lmin: int, factors: Optional[Dict]) -> Tuple[List[int], Dict]:
        raise NotImplementedError

    def get_Nset(self, txt: str, factors: Optional[Dict]) -> List[List[int]]:
        raise NotImplementedError

    def get_Mset(self, txt: str, factors: Optional[Dict]) -> List[List[int]]:
        raise NotImplementedError


class AnchorPositionProvider(PositionProvider):
    """Anchor-derived LP/LS with regex fallback; factors still use regex_find_all."""

    def __init__(self, conn, logger: Optional[logging.Logger] = None):
        self.conn = conn
        self.logger = logger or Logger

    @staticmethod
    def _char_len(txt: str) -> int:
        return len(txt)

    def _full_range(self, txt: str, lmin: int) -> Tuple[List[int], Dict]:
        clen = self._char_len(txt)
        upper = max(clen - lmin + 1, 0)
        if upper <= 0:
            starts = [] if clen == 0 else [1]
        else:
            starts = list(range(1, upper + 1))
        return starts, {"fallback": True, "upper_bound": True}

    def _anchor_hits(self, txt: str, anchor: str, lmin: int) -> List[int]:
        spans = find_all(self.conn, txt, anchor, 1, overlap=True, allow_empty=False)
        clen = self._char_len(txt)
        max_start = max(clen - lmin + 1, 0)
        return [s for s, _ in spans if s <= max_start or max_start == 0]

    def get_LPs(self, txt: str, pattern: str, lmin: int, factors: Optional[Dict]) -> Tuple[List[int], Dict]:
        parsed = parse_structured_regex(pattern)
        if not parsed:
            starts, meta = self._full_range(txt, lmin)
            meta["reason"] = "parse_fail"
            self.logger.debug("LPs fallback to full range (%d candidates)", len(starts))
            return starts, meta

        prefix = parsed["first_island_pattern"]
        leading_gap_max = parsed.get("leading_gap_max", 0)
        anchor_starts = self._anchor_hits(txt, prefix, lmin)
        if not anchor_starts:
            starts, meta = self._full_range(txt, lmin)
            meta["reason"] = "anchor_empty"
            meta["leading_gap_max"] = leading_gap_max
            self.logger.debug("LPs anchor empty; fallback to full range (%d)", len(starts))
            return starts, meta

        expanded = set()
        for a in anchor_starts:
            low = max(1, a - leading_gap_max)
            for s in range(low, a + 1):
                expanded.add(s)
        starts_list = sorted(expanded)
        meta = {
            "fallback": False,
            "anchor": prefix,
            "leading_gap_max": leading_gap_max,
            "anchor_hits": len(anchor_starts),
            "lp_size": len(starts_list),
            "expansion_ratio": len(starts_list) / max(1, len(anchor_starts)),
        }
        return starts_list, meta

    def get_LSs(self, txt: str, pattern: str, lmin: int, factors: Optional[Dict]) -> Tuple[List[int], Dict]:
        parsed = parse_structured_regex(pattern)
        if not parsed:
            starts, meta = self._full_range(txt, lmin)
            meta["reason"] = "parse_fail"
            self.logger.debug("LSs fallback to full range (%d candidates)", len(starts))
            return starts, meta

        suffix = parsed["last_island_pattern"]
        anchor_minlen = parsed.get("last_anchor_minlen", 1)
        trailing_gap_max = parsed.get("trailing_gap_max", 0)
        starts = self._anchor_hits(txt, suffix, lmin)
        if not starts:
            starts, meta = self._full_range(txt, lmin)
            meta["reason"] = "anchor_empty"
            self.logger.debug("LSs anchor empty; fallback to full range (%d)", len(starts))
            return starts, meta

        return sorted(starts), {
            "fallback": False,
            "anchor": suffix,
            "anchor_minlen": anchor_minlen,
            "trailing_gap_max": trailing_gap_max,
        }

    def _collect_hits(self, txt: str, factor_list: Sequence[str]) -> List[List[int]]:
        hits: List[List[int]] = []
        for factor in factor_list:
            spans = find_all(self.conn, txt, factor, 1, overlap=True)
            hits.append([start for start, _ in spans])
        return hits

    def get_Nset(self, txt: str, factors: Optional[Dict]) -> List[List[int]]:
        n_factors = factors.get("N", []) if factors else []
        if n_factors:
            self.logger.debug("N factors via regex_find_all (v1 fallback)")
        return self._collect_hits(txt, n_factors)

    def get_Mset(self, txt: str, factors: Optional[Dict]) -> List[List[int]]:
        m_factors = factors.get("M", []) if factors else []
        if m_factors:
            self.logger.debug("M factors via regex_find_all (v1 fallback)")
        return self._collect_hits(txt, m_factors)


class PNSPMNSVerifier:
    """Baseline skeleton that always validates spans against PostgreSQL regexes."""

    def __init__(self, conn, provider: Optional[PositionProvider] = None, logger: Optional[logging.Logger] = None, mode: str = "pns"):
        self.conn = conn
        self.logger = logger or Logger
        self.provider = provider or AnchorPositionProvider(conn, self.logger)
        self.enable_m_prune = mode in ("pmns-lite", "pmns-lite-n")
        self.enable_n_prune = mode in ("pns-n", "pmns-lite-n")
        self.nfactor_cache: Dict[str, "NFactorEngine"] = {}
        self.last_stats: Optional[Dict] = None

    def matches(self, txt: str, pattern: str, lmin: int = 1, factors: Optional[Dict] = None) -> bool:
        factor_info = extract_factors(pattern)
        factors = factors or {}
        occ_cache = OccurrenceCache(txt)
        windows, stats = self._candidate_windows(txt, pattern, lmin, factors, factor_info, occ_cache)
        necessary_literals: List[str] = []
        if factor_info:
            for isl in factor_info.islands:
                necessary_literals.extend(isl.necessary_literals)
        necessary_literals = sorted({lit for lit in necessary_literals if len(lit) >= 2})
        m_stats = self._apply_M_prune(txt, windows, necessary_literals, occ_cache)
        windows = m_stats["windows_after"]
        stats.update({
            "windows_before_M": m_stats["windows_before"],
            "windows_after_M": len(windows),
            "pruned_by_M": m_stats["pruned_by_M"],
            "M_hit_rate": m_stats["hit_rate"],
            "M_enabled": m_stats["enabled"],
            "M_disabled_reason": m_stats.get("reason"),
            "necessary_literals": necessary_literals,
        })
        min_len_bound = stats.get("lmin", lmin)
        n_stats = self._apply_N_prune(txt, windows, pattern, occ_cache, min_len_bound)
        windows = n_stats["windows_after_list"]
        stats.update({
            "windows_before_N": n_stats.get("windows_before"),
            "windows_after_N": n_stats.get("windows_after"),
            "pruned_by_N": n_stats.get("pruned_by_N"),
            "N_hit_rate": n_stats.get("hit_rate"),
            "N_enabled": n_stats.get("enabled"),
            "N_disabled_reason": n_stats.get("reason"),
            "nf_strings": n_stats.get("nf_strings"),
            "nf_candidates": n_stats.get("nf_candidates"),
            "nf_selected": n_stats.get("nf_selected"),
            "nfa_compile_ms": n_stats.get("nfa_compile_ms"),
            "nf_test_ms": n_stats.get("nf_test_ms"),
            "nf_prune_ms": n_stats.get("nf_prune_ms"),
            "nf_tested": n_stats.get("nf_tested"),
        })
        if not windows:
            hit = find_from(self.conn, txt, pattern, 1, allow_empty=True)
            matched = hit is not None
            self._log_stats(stats, 1, [], matched, batch_ms=0.0, fallback=True)
            return matched

        flat = []
        for w in windows:
            flat.extend([w[0], w[1]])

        import time

        t0 = time.perf_counter()
        hit_idx = verify_windows(self.conn, txt, pattern, flat)
        batch_ms = (time.perf_counter() - t0) * 1000.0
        matched = hit_idx >= 0
        self._log_stats(
            stats,
            verify_calls=1,
            verify_lengths=[],
            matched=matched,
            batch_ms=batch_ms,
            first_hit_window_id=(hit_idx if matched else None),
        )

        if matched:
            return True

        # Safety net only if anchor parsing fell back; otherwise trust verify_windows
        if stats["lp_meta"].get("fallback") or stats["ls_meta"].get("fallback"):
            hit = find_from(self.conn, txt, pattern, 1, allow_empty=True)
            matched = hit is not None
            self._log_stats(stats, 2, [], matched, batch_ms=batch_ms, fallback=True)
            return matched

        return False

    def _candidate_windows(self, txt: str, pattern: str, lmin: int, factors: Dict, factor_info: Optional[Factors], occ_cache: OccurrenceCache) -> Tuple[List[Tuple[int, int]], Dict]:
        lp_hits, lp_meta, lp_source = self._get_LPs_from_factors(txt, factor_info, occ_cache, lmin)
        ls_hits, ls_meta, ls_source = self._get_LSs_from_factors(txt, factor_info, occ_cache, lmin)

        if lp_hits is None or ls_hits is None or not lp_hits or not ls_hits:
            lp_hits, lp_meta = self.provider.get_LPs(txt, pattern, lmin, factors)
            ls_hits, ls_meta = self.provider.get_LSs(txt, pattern, lmin, factors)
            lp_source = lp_source or "legacy_regex"
            ls_source = ls_source or "legacy_regex"
        if not lp_hits or not ls_hits:
            # enumerate cap fallback
            clen = len(txt)
            cap = min(clen, 4096)
            lp_hits = list(range(1, cap + 1))
            ls_hits = list(range(1, cap + 1))
            lp_meta = {"fallback": True}
            ls_meta = {"fallback": True}
            lp_source = "enumerate_cap"
            ls_source = "enumerate_cap"
            if clen > cap:
                # safety net: rely on PG to avoid FN
                return [], {
                    "lp_size": len(lp_hits),
                    "ls_size": len(ls_hits),
                    "windows_lp_ls": 0,
                    "factor_windows": 0,
                    "windows_total": 0,
                    "dedup_windows": 0,
                    "lp_meta": lp_meta,
                    "ls_meta": ls_meta,
                    "char_len": clen,
                    "lmin": lmin,
                    "leading_gap_max": 0,
                    "lp_anchor_hits": 0,
                    "max_len": None,
                    "windows_before_bound": 0,
                    "windows_after_bound": 0,
                    "bound_effect_ratio": 0.0,
                    "LP_source": lp_source,
                    "LS_source": ls_source,
                }

        merged_factors = self._merge_sorted(self.provider.get_Nset(txt, factors) + self.provider.get_Mset(txt, factors))
        min_len_bound = factor_info.min_len if factor_info else lmin
        bound_enabled = factor_info is not None and factor_info.max_len is not None and factor_info.max_len >= min_len_bound
        bound_disabled_reason = None if bound_enabled else "max_len_unknown"
        max_len_bound = factor_info.max_len if bound_enabled else None
        text_len = len(txt)

        windows: List[Tuple[int, int]] = []
        windows_lp_ls = 0
        start_pruned_by_len = 0
        for lp in lp_hits:
            end_lower = lp + min_len_bound - 1
            if end_lower > text_len:
                start_pruned_by_len += 1
                continue
            suffix_start = self._findmin_suffix(ls_hits, lp)
            if suffix_start is not None:
                anchor_minlen = ls_meta.get("anchor_minlen", 1)
                trailing_gap = ls_meta.get("trailing_gap_max", 0)
                end_upper = suffix_start + max(min_len_bound, anchor_minlen) + trailing_gap - 1
            else:
                end_upper = text_len
            if max_len_bound is not None:
                end_upper = min(end_upper, lp + max_len_bound - 1, text_len)
            else:
                end_upper = min(end_upper, text_len)
            if end_lower <= end_upper:
                windows.append((lp, end_upper))
            windows_lp_ls += 1

        factor_windows = [(pos, len(txt)) for pos in merged_factors]
        windows.extend(factor_windows)

        # guard huge explosion
        if len(windows) > 20000:
            return [], {
                "lp_size": len(lp_hits),
                "ls_size": len(ls_hits),
                "windows_lp_ls": windows_lp_ls,
                "factor_windows": len(factor_windows),
                "windows_total": len(windows),
                "dedup_windows": 0,
                "lp_meta": lp_meta,
                "ls_meta": ls_meta,
                "char_len": len(txt),
                "lmin": min_len_bound,
                "leading_gap_max": lp_meta.get("leading_gap_max", 0),
                "lp_anchor_hits": lp_meta.get("anchor_hits", 0),
                "max_len": max_len_bound,
                "windows_before_bound": len(windows),
                "windows_after_bound": 0,
                "bound_effect_ratio": 0.0,
                "bound_enabled": False,
                "bound_disabled_reason": "too_many_windows",
                "start_pruned_by_len": start_pruned_by_len,
                "LP_source": lp_source,
                "LS_source": ls_source,
            }

        deduped = self._dedup_sorted(windows)
        stats = {
            "lp_size": len(lp_hits),
            "ls_size": len(ls_hits),
            "windows_lp_ls": windows_lp_ls,
            "factor_windows": len(factor_windows),
            "windows_total": len(windows),
            "dedup_windows": len(deduped),
            "lp_meta": lp_meta,
            "ls_meta": ls_meta,
            "char_len": len(txt),
            "lmin": min_len_bound,
            "leading_gap_max": lp_meta.get("leading_gap_max", 0),
            "lp_anchor_hits": lp_meta.get("anchor_hits", 0),
            "max_len": max_len_bound,
            "windows_before_bound": len(windows),
            "windows_after_bound": len(deduped),
            "bound_effect_ratio": (len(windows) - len(deduped)) / max(1, len(windows)),
            "bound_enabled": bound_enabled,
            "bound_disabled_reason": bound_disabled_reason,
            "start_pruned_by_len": start_pruned_by_len,
            "LP_source": lp_source,
            "LS_source": ls_source,
        }
        return deduped, stats

    def _get_LPs_from_factors(self, txt: str, factor_info: Optional[Factors], occ_cache: OccurrenceCache, lmin: int) -> Tuple[Optional[List[int]], Dict, Optional[str]]:
        if not factor_info or not factor_info.islands:
            return None, {}, None
        anchor_lits = factor_info.islands[0].anchor_literals
        if not anchor_lits:
            return None, {}, None
        anchor = sorted(anchor_lits, key=lambda x: (-len(x), x))[0]
        hits = occ_cache.find(anchor, overlap=True)
        if not hits:
            return None, {}, None
        expanded = set()
        lgap = factor_info.leading_gap_max
        for a in hits:
            low = max(1, a - lgap)
            for s in range(low, a + 1):
                expanded.add(s)
        starts = sorted(expanded)
        meta = {
            "anchor": anchor,
            "anchor_hits": len(hits),
            "leading_gap_max": lgap,
            "expansion_ratio": len(starts) / max(1, len(hits)),
        }
        return starts, meta, "factor_lit"

    def _get_LSs_from_factors(self, txt: str, factor_info: Optional[Factors], occ_cache: OccurrenceCache, lmin: int) -> Tuple[Optional[List[int]], Dict, Optional[str]]:
        if not factor_info or not factor_info.islands:
            return None, {}, None
        anchor_lits = factor_info.islands[-1].anchor_literals
        if not anchor_lits:
            return None, {}, None
        anchor = sorted(anchor_lits, key=lambda x: (-len(x), x))[0]
        hits = occ_cache.find(anchor, overlap=True)
        if not hits:
            return None, {}, None
        tg = factor_info.trailing_gap_max
        ls_candidates = set()
        clen = len(txt)
        for b in hits:
            high = min(clen, b + tg) if tg > 0 else b
            for pos in range(b, high + 1):
                ls_candidates.add(pos)
        starts = sorted(ls_candidates)
        meta = {
            "anchor": anchor,
            "anchor_hits": len(hits),
            "trailing_gap_max": tg,
            "expansion_ratio": len(starts) / max(1, len(hits)),
        }
        return starts, meta, "factor_lit"

    @staticmethod
    def _findmin_suffix(suffixes: List[int], start: int) -> Optional[int]:
        idx = bisect.bisect_left(suffixes, start)
        if idx >= len(suffixes):
            return None
        return suffixes[idx]

    @staticmethod
    def _merge_sorted(sorted_lists: List[List[int]]) -> List[int]:
        """Merge multiple sorted lists using a min-heap (keeps duplicates collapsed)."""
        heap: List[Tuple[int, int, int]] = []
        result: List[int] = []
        for list_idx, lst in enumerate(sorted_lists):
            if lst:
                heapq.heappush(heap, (lst[0], list_idx, 0))

        prev: Optional[int] = None
        while heap:
            value, list_idx, offset = heapq.heappop(heap)
            if prev is None or value != prev:
                result.append(value)
                prev = value

            next_offset = offset + 1
            lst = sorted_lists[list_idx]
            if next_offset < len(lst):
                heapq.heappush(heap, (lst[next_offset], list_idx, next_offset))
        return result

    @staticmethod
    def _dedup_sorted(items: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        seen = set()
        out: List[Tuple[int, int]] = []
        for item in sorted(items, key=lambda x: (x[0], x[1])):
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

    def _log_stats(
        self,
        stats: Dict,
        verify_calls: int,
        verify_lengths: List[int],
        matched: bool,
        batch_ms: float = 0.0,
        first_hit_window_id: Optional[int] = None,
        fallback: bool = False,
    ) -> None:
        total_positions = max(stats["char_len"] - stats["lmin"] + 1, 0)
        pruned_lp = max(total_positions - stats["lp_size"], 0)
        pruned_ls = max(total_positions - stats["ls_size"], 0)
        pruned_by_pmns = max((stats["windows_lp_ls"] + stats["factor_windows"]) - stats["dedup_windows"], 0)
        avg_verify_len = (sum(verify_lengths) / len(verify_lengths)) if verify_lengths else 0.0
        wb_bound = stats.get("windows_before_bound")
        wa_bound = stats.get("windows_after_bound")
        if wb_bound is None:
            wb_bound = 0
        if wa_bound is None:
            wa_bound = 0

        log_payload = {
            "matched": matched,
            "verify_calls": verify_calls,
            "avg_verify_len": avg_verify_len,
            "lp_size": stats["lp_size"],
            "ls_size": stats["ls_size"],
            "windows_lp_ls": stats["windows_lp_ls"],
            "factor_windows": stats["factor_windows"],
            "windows_total": stats["windows_total"],
            "dedup_windows": stats["dedup_windows"],
            "pruned_candidates_by_pns": pruned_lp + pruned_ls,
            "pruned_by_pmns": pruned_by_pmns,
            "lp_fallback": stats["lp_meta"].get("fallback", False),
            "ls_fallback": stats["ls_meta"].get("fallback", False),
            "fallback_safety": fallback,
            "batch_ms": batch_ms,
            "first_hit_window_id": first_hit_window_id,
            "leading_gap_max": stats.get("leading_gap_max", 0),
            "lp_anchor_hits": stats.get("lp_anchor_hits", 0),
            "max_len": stats.get("max_len", None),
            "windows_before_M": stats.get("windows_before_M"),
            "windows_after_M": stats.get("windows_after_M"),
            "pruned_by_M": stats.get("pruned_by_M"),
            "M_hit_rate": stats.get("M_hit_rate"),
            "M_enabled": stats.get("M_enabled"),
            "M_disabled_reason": stats.get("M_disabled_reason"),
            "windows_before_N": stats.get("windows_before_N"),
            "windows_after_N": stats.get("windows_after_N"),
            "pruned_by_N": stats.get("pruned_by_N"),
            "N_hit_rate": stats.get("N_hit_rate"),
            "N_enabled": stats.get("N_enabled"),
            "N_disabled_reason": stats.get("N_disabled_reason"),
            "nf_strings": stats.get("nf_strings"),
            "nf_candidates": stats.get("nf_candidates"),
            "nf_selected": stats.get("nf_selected"),
            "nf_tested": stats.get("nf_tested"),
            "LP_source": stats.get("LP_source"),
            "LS_source": stats.get("LS_source"),
            "bound_enabled": stats.get("bound_enabled"),
            "bound_disabled_reason": stats.get("bound_disabled_reason"),
            "windows_before_bound": wb_bound,
            "windows_after_bound": wa_bound,
        }
        self.logger.info("pns_pmns_stats=%s", log_payload)
        self.last_stats = log_payload

    def _apply_M_prune(self, txt: str, windows: List[Tuple[int, int]], necessary_literals: List[str], occ_cache: OccurrenceCache) -> Dict:
        before = len(windows)
        if not self.enable_m_prune:
            return {
                "enabled": False,
                "reason": "mode_disabled",
                "windows_before": before,
                "windows_after_count": before,
                "pruned_by_M": 0,
                "hit_rate": None,
                "windows_after": windows,
            }
        if not necessary_literals:
            return {
                "enabled": False,
                "reason": "no_necessary_literals",
                "windows_before": before,
                "windows_after_count": before,
                "pruned_by_M": 0,
                "hit_rate": None,
                "windows_after": windows,
            }
        occ_map = occ_cache.find_many(necessary_literals, overlap=True)
        filtered: List[Tuple[int, int]] = []
        for w in windows:
            s, e = w
            ok = True
            for lit in necessary_literals:
                if not occurs_in_window(occ_map.get(lit, []), s, e, len(lit)):
                    ok = False
                    break
            if ok:
                filtered.append(w)
        after = len(filtered)
        return {
            "enabled": True,
            "windows_before": before,
            "windows_after_count": after,
            "pruned_by_M": before - after,
            "hit_rate": (after / before) if before else None,
            "windows_after": filtered,
        }

    def _nfactor_candidates(self, txt: str, occ_cache: OccurrenceCache, gram: int = 3, stride: int = 8, max_candidates: int = 128) -> List[str]:
        seen = []
        hits = set()
        for i in range(0, max(0, len(txt) - gram + 1), stride):
            w = txt[i : i + gram]
            if len(w) < gram:
                continue
            if w in hits:
                continue
            if occ_cache.find(w, overlap=True):
                hits.add(w)
                seen.append(w)
            if len(seen) >= max_candidates:
                break
        return seen

    def _apply_N_prune(self, txt: str, windows: List[Tuple[int, int]], pattern: str, occ_cache: OccurrenceCache, min_len_bound: int) -> Dict:
        before = len(windows)
        if not self.enable_n_prune:
            return {
                "enabled": False,
                "reason": "mode_disabled",
                "windows_before": before,
                "windows_after": before,
                "pruned_by_N": 0,
                "hit_rate": None,
                "windows_after_list": windows,
                "nf_strings": [],
                "nf_candidates": 0,
                "nf_selected": 0,
                "nf_tested": 0,
            }
        if not windows:
            return {
                "enabled": False,
                "reason": "no_windows",
                "windows_before": before,
                "windows_after": before,
                "pruned_by_N": 0,
                "hit_rate": None,
                "windows_after_list": windows,
                "nf_strings": [],
                "nf_candidates": 0,
                "nf_selected": 0,
                "nf_tested": 0,
            }
        nf_candidates = self._nfactor_candidates(txt, occ_cache)
        if not nf_candidates:
            return {
                "enabled": False,
                "reason": "no_candidates",
                "windows_before": before,
                "windows_after": before,
                "pruned_by_N": 0,
                "hit_rate": None,
                "windows_after_list": windows,
                "nf_strings": [],
                "nf_candidates": 0,
                "nf_selected": 0,
                "nf_tested": 0,
            }
        engine = self.nfactor_cache.get(pattern)
        if engine is None:
            engine = NFactorEngine(pattern)
            self.nfactor_cache[pattern] = engine
        if not engine.ok:
            return {
                "enabled": False,
                "reason": engine.reason,
                "windows_before": before,
                "windows_after": before,
                "pruned_by_N": 0,
                "hit_rate": None,
                "windows_after_list": windows,
                "nf_strings": [],
                "nf_candidates": len(nf_candidates),
                "nf_selected": 0,
                "nf_tested": 0,
            }
        negatives = []
        nf_tested = 0
        nf_budget_ms = 30.0
        nf_max_tested = 64
        t_test0 = time.perf_counter()
        for w in nf_candidates:
            if nf_tested >= nf_max_tested:
                return {
                    "enabled": False,
                    "reason": "nf_max_tested",
                    "windows_before": before,
                    "windows_after": before,
                    "pruned_by_N": 0,
                    "hit_rate": None,
                    "windows_after_list": windows,
                    "nf_strings": [],
                    "nf_candidates": len(nf_candidates),
                    "nf_selected": 0,
                    "nf_tested": nf_tested,
                }
            if (time.perf_counter() - t_test0) * 1000.0 > nf_budget_ms:
                return {
                    "enabled": False,
                    "reason": "nf_budget_ms",
                    "windows_before": before,
                    "windows_after": before,
                    "pruned_by_N": 0,
                    "hit_rate": None,
                    "windows_after_list": windows,
                    "nf_strings": [],
                    "nf_candidates": len(nf_candidates),
                    "nf_selected": 0,
                    "nf_tested": nf_tested,
                }
            nf_tested += 1
            if engine.is_negative(w):
                negatives.append(w)
                if len(negatives) >= 4:
                    break
        test_ms = (time.perf_counter() - t_test0) * 1000.0
        if not negatives:
            return {
                "enabled": False,
                "reason": "no_negative_found",
                "windows_before": before,
                "windows_after": before,
                "pruned_by_N": 0,
                "hit_rate": None,
                "windows_after_list": windows,
                "nf_strings": [],
                "nf_candidates": len(nf_candidates),
                "nf_selected": 0,
                "nf_tested": nf_tested,
                "nf_test_ms": test_ms,
            }
        filtered: List[Tuple[int, int]] = []
        pruned = 0
        t_prune0 = time.perf_counter()
        occ_map = occ_cache.find_many(negatives, overlap=True)
        for s, e in windows:
            bad = False
            must_end = s + min_len_bound - 1
            for w in negatives:
                if occurs_in_window(occ_map.get(w, []), s, must_end, len(w)):
                    bad = True
                    break
            if not bad:
                filtered.append((s, e))
            else:
                pruned += 1
        prune_ms = (time.perf_counter() - t_prune0) * 1000.0
        after = len(filtered)
        return {
            "enabled": True,
            "windows_before": before,
            "windows_after": after,
            "pruned_by_N": pruned,
            "hit_rate": (after / before) if before else None,
            "windows_after_list": filtered,
            "nf_strings": negatives,
            "nf_test_ms": test_ms,
            "nf_prune_ms": prune_ms,
            "nf_candidates": len(nf_candidates),
            "nf_selected": len(negatives),
            "nf_tested": nf_tested,
        }
