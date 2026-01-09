from typing import Dict, Iterable, Iterator, List


def generate_grams_for_doc(text: str, lmax: int) -> Iterator[str]:
    n = len(text)
    lmax = min(lmax, n) if n else 0
    for i in range(n):
        max_len = min(lmax, n - i)
        for k in range(1, max_len + 1):
            yield text[i : i + k]


def grams_by_length(keys: Iterable[str]) -> Dict[int, List[str]]:
    buckets: Dict[int, List[str]] = {}
    for g in keys:
        buckets.setdefault(len(g), []).append(g)
    return buckets
