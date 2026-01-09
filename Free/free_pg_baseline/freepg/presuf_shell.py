from typing import Iterable, List


def presuf_shell(grams: Iterable[str]) -> List[str]:
    """
    Compute suffix-free shell by reversing grams and taking prefix-free subset.
    """
    reversed_sorted = sorted([g[::-1] for g in grams])
    kept: List[str] = []
    prev = None
    for r in reversed_sorted:
        if prev is not None and r.startswith(prev):
            continue
        kept.append(r[::-1])
        prev = r
    return kept
