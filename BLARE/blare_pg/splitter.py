"""Conservative regex splitter for BLARE-PG."""
import re
from typing import List, Tuple

from .types import SplitInfo


def _consume_char_class(regex: str, start: int) -> Tuple[bool, int]:
    """
    Skip over a character class starting at `[` (position start-1).

    Returns:
        (ok, next_idx) where next_idx is the index after the closing ']'.
    """
    i = start
    escaped = False
    while i < len(regex):
        ch = regex[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == "]":
            return True, i + 1
        i += 1
    return False, len(regex)


def _is_allowed_dot_quantifier(text: str) -> bool:
    """
    Return True if a quantifier string after '.' is of the form {0,k} or {0,}.
    """
    if not text.startswith("{") or not text.endswith("}"):
        return False
    body = text[1:-1]
    if "," not in body:
        return False
    parts = body.split(",", 1)
    try:
        lower = int(parts[0].strip())
    except ValueError:
        return False
    if lower != 0:
        return False
    if parts[1].strip() == "":
        return True
    try:
        upper = int(parts[1].strip())
    except ValueError:
        return False
    return upper >= 0


def _extract_required_literals(regex: str) -> Tuple[bool, List[str], str, str]:
    """
    Extract required literals conservatively.

    Supported shape: literals separated by wildcard segments (., .*, .{0,k}, .{0,}).
    Disallowed constructs lead to splittable=False to avoid false negatives.
    """
    literals: List[str] = []
    buf: List[str] = []
    last_token = "start"  # "literal", "wildcard", "other"

    def flush_buf():
        nonlocal buf
        if buf:
            literals.append("".join(buf))
            buf = []

    i = 0
    while i < len(regex):
        ch = regex[i]

        # Escaped character -> literal unless backreference.
        if ch == "\\":
            if i + 1 >= len(regex):
                return False, [], "UNTERMINATED_ESCAPE", "dangling_escape"
            nxt = regex[i + 1]
            # Backreferences remain unsupported.
            if nxt.isdigit():
                return False, [], "UNSUPPORTED_BACKREFERENCE", "backreference_unsupported"
            # Treat common zero-width/charset escapes as wildcard anchors.
            if nxt in ("y", "b", "B", "A", "Z", "z", "G", "K", "n", "r", "t", "f", "v", "s", "S", "w", "W", "d", "D"):
                flush_buf()
                last_token = "wildcard"
                i += 2
                continue
            literal_escapes = set(".+*?{}[]()|\\-")
            if nxt in literal_escapes:
                buf.append(nxt)
                last_token = "literal"
                i += 2
                continue
            return False, [], "ESCAPE_UNSUPPORTED", "escape_unsupported"

        # Alternation or lookaround/inline flags.
        if ch == "|":
            return False, [], "UNSUPPORTED_ALTERNATION", "alternation_unsupported"
        if ch == "(":
            # Allow non-capturing groups as wildcard regions.
            if i + 1 < len(regex) and regex[i + 1] == "?":
                # Skip until matching ')'
                flush_buf()
                depth = 1
                i += 2
                escaped = False
                while i < len(regex) and depth > 0:
                    c = regex[i]
                    if escaped:
                        escaped = False
                    elif c == "\\":
                        escaped = True
                    elif c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                    i += 1
                last_token = "wildcard"
                continue
            # Simple capturing group treated as wildcard unless proven otherwise.
            flush_buf()
            depth = 1
            i += 1
            escaped = False
            while i < len(regex) and depth > 0:
                c = regex[i]
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                i += 1
            last_token = "wildcard"
            continue
        if ch == ")":
            i += 1
            continue

        # Optional/variable quantifiers on literals are not allowed.
        if ch == "?":
            if last_token == "literal":
                return False, [], "UNSUPPORTED_OPTIONAL", "optional_quantifier_unsupported"
            flush_buf()
            last_token = "wildcard"
            i += 1
            continue
        if ch == "+":
            if last_token == "literal":
                return False, [], "UNSUPPORTED_ONE_OR_MORE", "one_or_more_quantifier_unsupported"
            flush_buf()
            last_token = "wildcard"
            i += 1
            continue

        # Character class -> wildcard; flush current literal.
        if ch == "[":
            flush_buf()
            ok, nxt = _consume_char_class(regex, i + 1)
            if not ok:
                return False, [], "unterminated_char_class"
            last_token = "wildcard"
            i = nxt
            continue

        # Wildcards starting with '.'.
        if ch == ".":
            if buf:
                flush_buf()
            # Check following quantifier.
            if i + 1 < len(regex) and regex[i + 1] == "*":
                i += 2
                last_token = "wildcard"
                continue
            if i + 1 < len(regex) and regex[i + 1] == "{":
                end = regex.find("}", i + 2)
                if end == -1:
                    return False, [], "UNTERMINATED_QUANTIFIER", "unterminated_quantifier"
                quant = "{" + regex[i + 2:end] + "}"
                if not _is_allowed_dot_quantifier(quant):
                    return False, [], "QUANTIFIER_NOT_ZERO_BASED_WILDCARD", "quantifier_not_zero_based_wildcard"
                i = end + 1
                last_token = "wildcard"
                continue
            # Single '.' wildcard.
            i += 1
            last_token = "wildcard"
            continue

        # Quantifier starting at '{' not after '.' -> unsupported.
        if ch == "{":
            if last_token == "literal":
                return False, [], "QUANTIFIER_ON_LITERAL", "quantifier_on_literal_unsupported"
            # Skip quantifier body.
            flush_buf()
            end = regex.find("}", i + 1)
            if end == -1:
                return False, [], "UNTERMINATED_QUANTIFIER", "unterminated_quantifier"
            i = end + 1
            last_token = "wildcard"
            continue
        if ch == "*":
            if last_token == "literal":
                return False, [], "STAR_ON_LITERAL", "star_on_literal_unsupported"
            flush_buf()
            last_token = "wildcard"
            i += 1
            continue

        # Default: literal character.
        buf.append(ch)
        last_token = "literal"
        i += 1

    if buf:
        literals.append("".join(buf))

    literals = [lit for lit in literals if lit]
    if not literals:
        return False, [], "CHARCLASS_ONLY", "no_required_literals"
    return True, literals, "OK", ""


def _detect_case_mode(regex: str) -> str:
    """
    Detect case-sensitivity from inline flags.

    Returns: "sensitive", "insensitive", or "unknown".
    """
    positive_i = False
    negative_i = False
    for m in re.finditer(r"\(\?([A-Za-z\-]*)(?:\)|:)", regex):
        flags = m.group(1)
        if "i" in flags and "-i" not in flags:
            positive_i = True
        if "-i" in flags:
            negative_i = True
    if positive_i and not negative_i:
        return "insensitive"
    if negative_i and not positive_i:
        return "sensitive"
    if positive_i and negative_i:
        return "unknown"
    return "sensitive"


def build_split_info(regex: str) -> SplitInfo:
    """
    Build a conservative SplitInfo.

    Returns splittable=False when unsure about required literals to guarantee zero
    false negatives.
    """
    splittable, required_literals, reason_code, reason_detail = _extract_required_literals(regex)
    case_mode = _detect_case_mode(regex)
    if case_mode == "unknown" and reason_code == "OK":
        reason_code = "UNKNOWN_FLAG_NESTING"
        reason_detail = "conflicting_inline_flags"
    return SplitInfo(
        raw=regex,
        required_literals=required_literals,
        splittable=splittable,
        reason_code=reason_code,
        reason_detail=reason_detail,
        case_mode=case_mode,
    )
