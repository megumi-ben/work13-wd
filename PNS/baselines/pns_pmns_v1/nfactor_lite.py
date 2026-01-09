"""
N-factor-lite: over-approximate NFA + contains-w DFA emptiness test.

Only supports a safe generator subset; any unsupported construct disables N-prune.
The NFA intentionally over-approximates the regex language so that
  empty(overapprox ∩ contains(w)) => empty(real ∩ contains(w)) (safe to claim negative).
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


ANY = "__ANY__"


@dataclass
class NFA:
    start: int
    accepts: Set[int]
    edges: Dict[int, List[Tuple[Optional[str], int]]]  # label None -> epsilon, ANY or literal char


def _new_state(edges: Dict[int, List[Tuple[Optional[str], int]]]) -> int:
    idx = len(edges)
    edges[idx] = []
    return idx


def _add_edge(edges: Dict[int, List[Tuple[Optional[str], int]]], u: int, v: int, label: Optional[str]) -> None:
    edges[u].append((label, v))


def _tokenize(pattern: str):
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\":
            if i + 1 >= n:
                yield ("fail", None)
                return
            nxt = pattern[i + 1]
            if nxt == "y":
                yield ("epsilon", None)
            else:
                # treat escaped literal as literal; if unknown control, fail
                yield ("lit", nxt)
            i += 2
            continue
        if c in "()|{}*+?,":
            yield (c, c)
            i += 1
            continue
        if c == "[":
            # consume charclass as ANY
            j = i + 1
            depth = 1
            while j < n and depth > 0:
                if pattern[j] == "\\":
                    j += 2
                    continue
                if pattern[j] == "[":
                    depth += 1
                elif pattern[j] == "]":
                    depth -= 1
                j += 1
            if depth != 0:
                yield ("fail", None)
                return
            yield ("any", None)
            i = j
            continue
        if c == ".":
            yield ("any", None)
            i += 1
            continue
        if c == "(" and pattern[i:i + 3] == "(?:":
            yield ("group_start", "(?:")
            i += 3
            continue
        if c == ")":
            yield ("group_end", ")")
            i += 1
            continue
        # literal
        yield ("lit", c)
        i += 1


class AST:
    pass


@dataclass
class Lit(AST):
    ch: str  # literal char or ANY


@dataclass
class Concat(AST):
    parts: List[AST]


@dataclass
class Alt(AST):
    options: List[AST]


@dataclass
class Repeat(AST):
    node: AST
    min_rep: int
    max_rep: int


def _parse(pattern: str) -> Optional[AST]:
    tokens = list(_tokenize(pattern))
    if any(t[0] == "fail" for t in tokens):
        return None
    pos = 0

    def parse_expr() -> Optional[AST]:
        nonlocal pos
        terms: List[AST] = []
        while pos < len(tokens) and tokens[pos][0] not in (")", "group_end"):
            if tokens[pos][0] == "|":
                break
            term = parse_term()
            if term is None:
                return None
            terms.append(term)
        if not terms:
            return Concat([])
        if len(terms) == 1:
            return terms[0]
        return Concat(terms)

    def parse_term() -> Optional[AST]:
        nonlocal pos
        atom = parse_atom()
        if atom is None:
            return None
        # handle quantifier {a,b} and optional trailing '?'
        if pos < len(tokens) and tokens[pos][0] == "{":
            # parse digits until '}'
            j = pos + 1
            num_chars = []
            while j < len(tokens) and tokens[j][0] not in (",", "}"):
                if tokens[j][0] != "lit" or not tokens[j][1].isdigit():
                    return None
                num_chars.append(tokens[j][1])
                j += 1
            if j >= len(tokens) or tokens[j][0] == "}":
                if not num_chars:
                    return None
                min_rep = int("".join(num_chars))
                max_rep = min_rep
                pos = j + 1
            else:
                min_rep = int("".join(num_chars)) if num_chars else 0
                j += 1
                num_chars = []
                while j < len(tokens) and tokens[j][0] != "}":
                    if tokens[j][0] != "lit" or not tokens[j][1].isdigit():
                        return None
                    num_chars.append(tokens[j][1])
                    j += 1
                if j >= len(tokens):
                    return None
                max_rep = int("".join(num_chars)) if num_chars else min_rep
                pos = j + 1
            if min_rep > max_rep or max_rep > 200:
                return None
            # optional non-greedy '?'
            if pos < len(tokens) and tokens[pos][0] == "?":
                pos += 1
            return Repeat(atom, min_rep, max_rep)
        # optional '?' (treat as {0,1})
        if pos < len(tokens) and tokens[pos][0] == "?":
            pos += 1
            return Repeat(atom, 0, 1)
        return atom

    def parse_atom() -> Optional[AST]:
        nonlocal pos
        if pos >= len(tokens):
            return None
        t, val = tokens[pos]
        if t == "lit":
            pos += 1
            return Lit(val)
        if t == "any":
            pos += 1
            return Lit(ANY)
        if t == "epsilon":
            pos += 1
            return Concat([])
        if t in ("(", "group_start"):
            pos += 1
            sub = parse_alts()
            if pos >= len(tokens) or tokens[pos][0] not in (")", "group_end"):
                return None
            pos += 1
            return sub
        return None

    def parse_alts() -> Optional[AST]:
        nonlocal pos
        options: List[AST] = []
        while True:
            expr = parse_expr()
            if expr is None:
                return None
            options.append(expr)
            if pos >= len(tokens) or tokens[pos][0] != "|":
                break
            pos += 1
        if len(options) == 1:
            return options[0]
        return Alt(options)

    ast = parse_alts()
    if ast is None or pos != len(tokens):
        return None
    return ast


def _build_nfa(ast: AST) -> Optional[NFA]:
    edges: Dict[int, List[Tuple[Optional[str], int]]] = defaultdict(list)

    def build(node: AST) -> Optional[Tuple[int, int]]:
        if isinstance(node, Concat):
            if not node.parts:
                s = _new_state(edges)
                t = _new_state(edges)
                _add_edge(edges, s, t, None)
                return s, t
            cur_start, cur_end = None, None
            for part in node.parts:
                sub = build(part)
                if sub is None:
                    return None
                s, t = sub
                if cur_start is None:
                    cur_start, cur_end = s, t
                else:
                    _add_edge(edges, cur_end, s, None)
                    cur_end = t
            return cur_start, cur_end
        if isinstance(node, Alt):
            start = _new_state(edges)
            end = _new_state(edges)
            for opt in node.options:
                sub = build(opt)
                if sub is None:
                    return None
                s, t = sub
                _add_edge(edges, start, s, None)
                _add_edge(edges, t, end, None)
            return start, end
        if isinstance(node, Repeat):
            sub = build(node.node)
            if sub is None:
                return None
            start = _new_state(edges)
            last = start
            end = _new_state(edges)
            _add_edge(edges, start, end, None)
            for _ in range(node.max_rep):
                sub2 = build(node.node)
                if sub2 is None:
                    return None
                s2, t2 = sub2
                _add_edge(edges, last, s2, None)
                _add_edge(edges, t2, end, None)
                last = t2
            return start, end
        if isinstance(node, Lit):
            start = _new_state(edges)
            end = _new_state(edges)
            label = ANY if node.ch == ANY else node.ch
            _add_edge(edges, start, end, label)
            return start, end
        return None

    built = build(ast)
    if built is None:
        return None
    start, end = built
    accepts = {end}
    return NFA(start=start, accepts=accepts, edges=edges)


def compile_overapprox_nfa(pattern: str) -> Optional[NFA]:
    ast = _parse(pattern)
    if ast is None:
        return None
    return _build_nfa(ast)


def build_contains_dfa(w: str):
    if not w:
        return None
    alphabet = set(w)
    alphabet.add("OTHER")
    m = len(w)
    lps = [0] * m
    length = 0
    for i in range(1, m):
        while length > 0 and w[length] != w[i]:
            length = lps[length - 1]
        if w[length] == w[i]:
            length += 1
            lps[i] = length
    trans: List[Dict[str, int]] = []
    for state in range(m + 1):
        row: Dict[str, int] = {}
        for ch in alphabet:
            if state == m:
                row[ch] = m
                continue
            if ch == "OTHER":
                row[ch] = 0
                continue
            k = state
            while k > 0 and w[k] != ch:
                k = lps[k - 1]
            if k < m and w[k] == ch:
                k += 1
            row[ch] = k
        trans.append(row)
    return alphabet, trans, m


def _epsilon_closure(nfa: NFA, states: Set[int]) -> Set[int]:
    stack = list(states)
    seen = set(states)
    while stack:
        u = stack.pop()
        for label, v in nfa.edges.get(u, []):
            if label is None and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def intersects_contains_w(nfa: NFA, w: str) -> bool:
    dfa_info = build_contains_dfa(w)
    if dfa_info is None:
        return False
    alphabet, trans, accept_state = dfa_info
    start_set = _epsilon_closure(nfa, {nfa.start})
    start_prod = (frozenset(start_set), 0)
    q = deque([start_prod])
    visited = {start_prod}
    while q:
        n_states, k_state = q.popleft()
        if k_state == accept_state and n_states & nfa.accepts:
            return True
        for u in n_states:
            for label, v in nfa.edges.get(u, []):
                if label is None:
                    continue
                if label == ANY:
                    syms = alphabet
                else:
                    sym = label if label in alphabet else "OTHER"
                    syms = [sym]
                for sym in syms:
                    next_k = trans[k_state][sym]
                    closure = _epsilon_closure(nfa, {v})
                    next_states = _epsilon_closure(nfa, closure)
                    prod = (frozenset(next_states), next_k)
                    if prod not in visited:
                        visited.add(prod)
                        q.append(prod)
    return False


class NFactorEngine:
    """Per-pattern engine with memoized negative checks."""

    def __init__(self, pattern: str):
        self.nfa = compile_overapprox_nfa(pattern)
        self.ok = self.nfa is not None
        self.reason = None if self.ok else "compile_fail"
        self.cache: Dict[str, bool] = {}

    def is_negative(self, w: str) -> bool:
        if not self.ok or not w or len(w) < 2:
            return False
        if w in self.cache:
            return self.cache[w]
        res = not intersects_contains_w(self.nfa, w)
        self.cache[w] = res
        return res
