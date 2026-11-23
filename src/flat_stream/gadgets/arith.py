"""Exact Gidney clean-carry ripple-carry adder (quantum-quantum and shared-ctrl constant add)."""
from __future__ import annotations

from typing import Sequence


def add(b, target: Sequence[int], source: Sequence[int | None], shared: int | None = None) -> None:
    """In-place ``target += source (mod 2^len(target))``.

    - When ``shared is None``: quantum-quantum clean-carry Gidney adder (``len(target) - 1`` Toffolis).
    - When ``shared is not None``: ``source[p]`` is either ``shared`` (bit=1) or ``None`` (bit=0),
      never mutating ``shared`` (``len(target) - 1`` Toffolis for odd constant).
    """
    n = len(target)
    if n == 0 or not source:
        return
    m = min(n, len(source))

    if shared is not None:
        if n == 1:
            if source[0] is not None:
                b.cx(shared, target[0])
            return
        # Find lowest non-None bit
        lo = 0
        while lo < m and source[lo] is None:
            lo += 1
        if lo >= m:
            return
        if lo > 0:
            add(b, target[lo:], source[lo:], shared=shared)
            return
        # Now source[0] == shared
        trail = []
        carry = b.and_compute(shared, target[0])
        trail.append(("c0", 0, None, carry))
        for p in range(1, n - 1):
            sp = source[p] if p < m else None
            tp = target[p]
            if sp is None:
                anc = b.and_compute(carry, tp)
                trail.append(("bit0", p, carry, anc))
                carry = anc
            else:
                # sp == shared; by induction carry==1 => shared==1, so
                # maj(carry, shared, tp) = ((carry ^ shared) & (tp ^ shared)) ^ shared!
                b.cx(shared, carry)
                b.cx(shared, tp)
                anc = b.and_compute(carry, tp)
                b.cx(shared, anc)
                trail.append(("bit1", p, carry, anc))
                carry = anc
        # Top bit n - 1
        b.cx(carry, target[n - 1])
        if n - 1 < m and source[n - 1] is not None:
            b.cx(shared, target[n - 1])
        # Uncompute carries in reverse
        for kind, p, prev_c, anc in reversed(trail):
            if kind == "bit1":
                b.cx(shared, anc)
                b.and_release(anc, prev_c, target[p])
                b.cx(shared, prev_c)
                # target[p] still has shared XORed into it; XORing prev_c yields tp ^ shared ^ prev_c!
                b.cx(prev_c, target[p])
            elif kind == "bit0":
                b.and_release(anc, prev_c, target[p])
                b.cx(prev_c, target[p])
            else:
                b.and_release(anc, shared, target[0])
                b.cx(shared, target[0])
        return

    # Quantum-quantum clean-carry Gidney adder
    if n == 1:
        if source[0] is not None:
            b.cx(source[0], target[0])
        return
    trail = []
    if source[0] is not None:
        carry = b.and_compute(source[0], target[0])
        trail.append(("g0", 0, source[0], None, carry))
    else:
        carry = None
    for p in range(1, n - 1):
        sp = source[p] if p < m else None
        tp = target[p]
        if carry is None:
            if sp is not None:
                carry = b.and_compute(sp, tp)
                trail.append(("g0", p, sp, None, carry))
        elif sp is not None:
            b.cx(carry, sp)
            b.cx(carry, tp)
            anc = b.and_compute(sp, tp)
            b.cx(carry, anc)
            trail.append(("g1", p, sp, carry, anc))
            carry = anc
        else:
            anc = b.and_compute(carry, tp)
            trail.append(("g_top", p, None, carry, anc))
            carry = anc
    if carry is not None:
        b.cx(carry, target[n - 1])
    if n - 1 < m and source[n - 1] is not None:
        b.cx(source[n - 1], target[n - 1])
    for kind, p, sp, prev_c, anc in reversed(trail):
        if kind == "g1":
            b.cx(prev_c, anc)
            b.and_release(anc, sp, target[p])
            b.cx(prev_c, sp)
            b.cx(sp, target[p])
        elif kind == "g_top":
            b.and_release(anc, prev_c, target[p])
            b.cx(prev_c, target[p])
        else:
            b.and_release(anc, sp, target[p])
            b.cx(sp, target[p])
