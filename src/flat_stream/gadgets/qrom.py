"""Textbook Babbush et al. (2018) unary-iteration QROM helpers (clean-room implementation)."""
from __future__ import annotations

from typing import Callable, Sequence


def xor_word(b, ctrl: int | None, word: int, target: Sequence[int]) -> None:
    rem = int(word)
    idx = 0
    n_t = len(target)
    while rem and idx < n_t:
        if rem & 1:
            if ctrl is None:
                b.x(target[idx])
            else:
                b.cx(ctrl, target[idx])
        rem >>= 1
        idx += 1


def swap_perm(lam: int) -> list[list[int]]:
    log_lam = lam.bit_length() - 1
    out: list[list[int]] = []
    for lv in range(lam):
        arr = list(range(lam))
        for i in range(log_lam):
            if (lv >> i) & 1:
                step = 1 << i
                span = step << 1
                for s in range(0, lam, span):
                    arr[s], arr[s + step] = arr[s + step], arr[s]
        out.append(arr)
    return out


def walk(
    b,
    addr: Sequence[int],
    ctrl: int | None,
    bit: int,
    lo: int,
    hi: int,
    a: int,
    b_lim: int,
    leaf: Callable[[int | None, int], None],
) -> None:
    if hi <= a or lo >= b_lim:
        return
    if bit < 0:
        leaf(ctrl, lo)
        return
    mid = (lo + hi) >> 1
    has_left = lo < b_lim and mid > a
    has_right = mid < b_lim and hi > a
    q = addr[bit]
    if has_left and not has_right:
        if ctrl is None:
            b.x(q)
            walk(b, addr, q, bit - 1, lo, mid, a, b_lim, leaf)
            b.x(q)
        else:
            c0 = b.and_compute(ctrl, q, pol=(True, False))
            walk(b, addr, c0, bit - 1, lo, mid, a, b_lim, leaf)
            b.and_release(c0, ctrl, q, pol=(True, False))
        return
    if has_right and not has_left:
        if ctrl is None:
            walk(b, addr, q, bit - 1, mid, hi, a, b_lim, leaf)
        else:
            c1 = b.and_compute(ctrl, q, pol=(True, True))
            walk(b, addr, c1, bit - 1, mid, hi, a, b_lim, leaf)
            b.and_release(c1, ctrl, q, pol=(True, True))
        return
    if ctrl is None:
        b.x(q)
        walk(b, addr, q, bit - 1, lo, mid, a, b_lim, leaf)
        b.x(q)
        walk(b, addr, q, bit - 1, mid, hi, a, b_lim, leaf)
    else:
        c0 = b.and_compute(ctrl, q, pol=(True, False))
        walk(b, addr, c0, bit - 1, lo, mid, a, b_lim, leaf)
        b.cx(ctrl, c0)
        walk(b, addr, c0, bit - 1, mid, hi, a, b_lim, leaf)
        b.and_release(c0, ctrl, q, pol=(True, True))
