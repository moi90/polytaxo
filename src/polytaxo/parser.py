import re
import shlex
from typing import Iterable, List, Mapping, Tuple


def _compile_re(fsm: Mapping):
    return {
        k: [
            (re.compile(expr).match, action, next_state)
            for (expr, action, next_state) in rules
        ]
        for k, rules in fsm.items()
    }


def _build_tokenize():
    # States
    NEUTRAL = 0
    IN_PATH = 1
    IN_SINGLE_QUOTED_PATH = 2
    IN_DOUBLE_QUOTED_PATH = 3

    tokens: List[str | Tuple[str, ...]] = []
    parts = []
    field = []

    def _append_field(s: str):
        field.append(s)

    def _finalize_field(s: str):
        parts.append("".join(field))
        field.clear()

    def _finalize_part(s: str):
        _finalize_field("")
        tokens.append(tuple(parts))
        parts.clear()

    def _append_token(s: str):
        tokens.append(s)

    def _finalize_part_and_append_token(s: str):
        _finalize_part("")
        tokens.append(s)

    fsm = _compile_re(
        {
            NEUTRAL: [
                (r"\s+", None, NEUTRAL),
                (r"\w+", _append_field, IN_PATH),
                (r":|/", None, IN_PATH),
                (r"[!\-()]", _append_token, NEUTRAL),
                (r"'", None, IN_SINGLE_QUOTED_PATH),
                (r'"', None, IN_DOUBLE_QUOTED_PATH),
                (r"$", None, None),
            ],
            IN_PATH: [
                (r":|/", _finalize_field, IN_PATH),
                (r"\w+", _append_field, IN_PATH),
                (r"(\s)|($)", _finalize_part, NEUTRAL),
                (r"[)]", _finalize_part_and_append_token, NEUTRAL),
            ],
            IN_SINGLE_QUOTED_PATH: [
                (r":|/", _finalize_field, IN_SINGLE_QUOTED_PATH),
                (r'[\w\s"!]+', _append_field, IN_SINGLE_QUOTED_PATH),
                (r"'", _finalize_part, NEUTRAL),
            ],
            IN_DOUBLE_QUOTED_PATH: [
                (r":|/", _finalize_field, IN_DOUBLE_QUOTED_PATH),
                (r"[\w\s']+", _append_field, IN_DOUBLE_QUOTED_PATH),
                (r'"', _finalize_part, NEUTRAL),
            ],
        }
    )

    def tokenize(s: str):
        tokens.clear()
        state = NEUTRAL
        pos = 0
        length = len(s)
        while pos <= length:
            # Find matching rule
            match: re.Match | None
            match, action, next_state = next(
                (
                    (match, action, next_state)
                    for match_expr, action, next_state in fsm[state]
                    if (match := match_expr(s, pos)) is not None
                ),
                (None, None, None),
            )

            if match is None:
                raise ValueError(
                    f"Unexpected character {s[pos:pos+1]!r} at pos {pos} (state={state})\n{s}\n{('-'*pos)+'^'}"
                )

            if action is not None:
                action(match.group(match.lastindex or 0))

            if next_state is None:
                break

            state = next_state
            pos = match.end()

        return tokens

    return tokenize


tokenize = _build_tokenize()

_find_unsafe = re.compile(r"[^\w@%+=:,./-]", re.ASCII).search


def quote(s: str, quote_chars="'\""):
    if s == "":
        return "''"

    if _find_unsafe(s) is None:
        return s

    try:
        quote = next(c for c in quote_chars if c not in s)
    except StopIteration:
        raise ValueError(f"{s} contains all quote chars ({quote_chars!r})")

    return quote + s + quote


def join(seq_of_str: Iterable[str]):
    return " ".join(quote(arg) for arg in seq_of_str)
