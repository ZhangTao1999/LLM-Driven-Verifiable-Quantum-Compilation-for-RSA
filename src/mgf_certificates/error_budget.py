"""Numerical CRT and state-dependent storage budgets, derived from equations.

This module does not import a circuit producer or any external scorer.
The dyadic masking estimator is Gidney's declared model, not a new
unconditional factoring-success theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from math import isqrt
from typing import Iterable


def as_decimal(value: Fraction, digits: int = 35) -> str:
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def exp_minus_upper(value: Fraction, terms: int = 170) -> Fraction:
    """Exact rational bound exp(-x) <= 1/sum_{j=0}^K x^j/j!, x>=0."""
    if value < 0:
        raise ValueError("The exponential upper bound requires a nonnegative x")
    term = total = Fraction(1)
    for index in range(1, terms + 1):
        term *= value / index
        total += term
    return 1 / total


def sqrt_upper(value: Fraction, denominator: int = 10**60) -> Fraction:
    if value < 0:
        raise ValueError("Negative squared distance")
    scaled = (value.numerator * denominator**2 + value.denominator - 1) // value.denominator
    numerator = isqrt(scaled)
    if numerator * numerator < scaled:
        numerator += 1
    answer = Fraction(numerator, denominator)
    assert answer * answer >= value
    return answer


def centered_certificate(primes: Iterable[int], *, windows: int, bits: int,
                         terms: int = 170, bin_width: int = 1) -> dict:
    """Certify independent uniform exponent windows; allow correlated primes.

    Each dlog term lies in [0,p-2]. The sum of rounded window means is
    within W/2 of the true mean, including a shortened last address.
    Binning is optional and conservative: use the largest prime in each
    actual bin. A bin width of one evaluates every distinct prime.
    """
    periods = tuple(map(int, primes))
    if windows < 1 or bits < 2 or bin_width < 1:
        raise ValueError("Invalid certificate parameters")
    margin = Fraction(2 ** (bits - 1)) - Fraction(windows, 2)
    if margin <= 0:
        raise ValueError("No positive certified margin")
    groups: dict[int, list[int]] = {}
    for prime in periods:
        if prime < 3:
            raise ValueError("The residue primes must be at least three")
        groups.setdefault(prime // bin_width, []).append(prime)
    # Round each rational upward onto a fixed grid before summing. Summing
    # the unreduced prime-specific denominators would create an enormous
    # irrelevant common denominator. The grid is an explicit upper bound.
    grid = 10**100
    bad_numerator = 0
    for group in groups.values():
        upper_range = max(group) - 2
        exponent = 2 * margin * margin / (windows * upper_range**2)
        local_bound = 2 * len(group) * exp_minus_upper(exponent, terms)
        bad_numerator += (local_bound.numerator * grid + local_bound.denominator - 1) // local_bound.denominator
    bad_weight = Fraction(bad_numerator, grid)
    trace = min(Fraction(1), 2 * sqrt_upper(bad_weight))
    return {
        "scope": "specified uniform independent exponent windows, not arbitrary-input diamond norm",
        "primes": len(periods), "windows": windows, "stored_bits": bits,
        "window_center_rounding_bound": str(Fraction(windows, 2)),
        "range_bound": "each window dlog in [0,p-2]",
        "exponential_bound": "positive Taylor polynomial reciprocal; exact rational arithmetic",
        "taylor_terms": terms, "bin_width": bin_width, "groups": len(groups),
        "per_group_upward_rounding_grid": "1e-100",
        "bad_weight_upper": as_decimal(bad_weight),
        "bad_weight_fraction": [str(bad_weight.numerator), str(bad_weight.denominator)],
        "trace_upper_per_shot": as_decimal(trace),
        "trace_fraction": [str(trace.numerator), str(trace.denominator)],
        "trace_square_check": trace * trace >= 4 * bad_weight if trace < 1 else True,
    }


def shifted_unsigned_hoeffding_certificate(
    primes: Iterable[int], *, windows: int, bits: int,
    terms: int = 170, bin_width: int = 1,
) -> dict:
    """Certify offset-unsigned (Du) lanes via exact rational Hoeffding (Appendix C.6, Eq. (C5)).

    For each prime p of bit-length ell, the divisor-aligned half-width is
    H_p = 2^(bits - ell - 1) * (p - 1). Because mu_p = floor(E[S_p] + 1/2) is rounded once
    at the aggregate sum, the worst-case center rounding deviation is at most 1/2,
    giving certified safe margin H_p - 1/2.
    """
    periods = tuple(map(int, primes))
    if windows < 1 or bits < 2 or bin_width < 1:
        raise ValueError("Invalid certificate parameters")
    groups: dict[int, list[int]] = {}
    for prime in periods:
        if prime < 3:
            raise ValueError("The residue primes must be at least three")
        groups.setdefault(prime // bin_width, []).append(prime)
    grid = 10**100
    bad_numerator = 0
    for group in groups.values():
        p_lo = min(group)
        p_hi = max(group)
        ell = p_hi.bit_length()
        if bits < ell + 1:
            raise ValueError("shifted_unsigned requires bits >= ell + 1")
        h_lo = (1 << (bits - ell - 1)) * (p_lo - 1)
        margin = Fraction(h_lo) - Fraction(1, 2)
        if margin <= 0:
            raise ValueError("No positive certified margin")
        upper_range = p_hi - 2
        exponent = 2 * margin * margin / (windows * upper_range**2)
        local_bound = 2 * len(group) * exp_minus_upper(exponent, terms)
        bad_numerator += (local_bound.numerator * grid + local_bound.denominator - 1) // local_bound.denominator
    bad_weight = Fraction(bad_numerator, grid)
    trace = min(Fraction(1), 2 * sqrt_upper(bad_weight))
    return {
        "representation": "shifted_unsigned",
        "scope": "specified uniform independent exponent windows, not arbitrary-input diamond norm",
        "primes": len(periods), "windows": windows, "stored_bits": bits,
        "center_rounding_bound": "1/2",
        "range_bound": "each window dlog in [0,p-2]",
        "exponential_bound": "positive Taylor polynomial reciprocal; exact rational arithmetic",
        "taylor_terms": terms, "bin_width": bin_width, "groups": len(groups),
        "per_group_upward_rounding_grid": "1e-100",
        "bad_weight_upper": as_decimal(bad_weight),
        "bad_weight_fraction": [str(bad_weight.numerator), str(bad_weight.denominator)],
        "trace_upper_per_shot": as_decimal(trace),
        "trace_fraction": [str(trace.numerator), str(trace.denominator)],
        "trace_square_check": trace * trace >= 4 * bad_weight if trace < 1 else True,
    }


@dataclass(frozen=True)
class Budget:
    crt_terms: int
    accumulator_bits: int = 33
    mask_bits: int = 26
    wraparound_gap: int = 33
    shots: int = 9
    trace_per_shot: Fraction = Fraction(0)
    crt_mode: str = "odd"  # "odd" or "dyadic" (Eq. (A5))

    def values(self) -> dict:
        if not 0 <= self.mask_bits < self.accumulator_bits:
            raise ValueError("Invalid mask width")
        if self.crt_terms < 1 or self.shots < 1:
            raise ValueError("Invalid term or shot count")
        if self.crt_mode not in ("odd", "dyadic"):
            raise ValueError(f"Unsupported crt_mode: {self.crt_mode!r}")
        support = Fraction(1, 2 ** (self.accumulator_bits - self.mask_bits))
        num_round = 1 if self.crt_mode == "dyadic" else 2
        epsilon = self.crt_terms * (Fraction(num_round, 2**self.accumulator_bits)
                                    + Fraction(1, 2**self.wraparound_gap))
        keep = 1 - support - epsilon / support
        charged = keep - self.shots * self.trace_per_shot
        if charged <= 0:
            raise ValueError("The declared model has no positive charged keep probability")
        return {"crt_mode": self.crt_mode, "crt_terms": self.crt_terms, "f": self.accumulator_bits,
                "mask_bits": self.mask_bits, "wraparound_gap": self.wraparound_gap,
                "nominal_shots": self.shots,
                "mask_proxy": float(support), "numerical_epsilon": float(epsilon),
                "keep_base_model": float(keep), "keep_charged_model": float(charged),
                "extra_trace_upper_per_shot": float(self.trace_per_shot),
                "extra_trace_debit": float(self.shots * self.trace_per_shot),
                "success_scope": "Gidney dyadic masking/postprocessing estimator; no added 0.99 factor",
                "factor_per_shot": float(Fraction(self.shots) / charged)}

    def factoring_cost(self, toffoli_per_shot: float) -> dict:
        values = self.values()
        values["T_shot"] = toffoli_per_shot
        values["T_fact"] = toffoli_per_shot * values["factor_per_shot"]
        return values


def modular_distance(left: int, right: int, modulus: int) -> int:
    distance = (left - right) % modulus
    return min(distance, modulus - distance)


def check_crt_repartition(*, modulus: int, rns_product: int, ideal_product: int,
                          contributions: Iterable[int], accumulator_bits: int) -> dict:
    """Independent integer check of a declared numerical CRT interface."""
    terms = tuple(map(int, contributions))
    if not 0 <= ideal_product < rns_product:
        raise ValueError("RNS capacity does not cover the ideal unreduced product")
    if not terms or any(not 0 <= term < rns_product for term in terms):
        raise ValueError("Every CRT contribution must be in [0,L)")
    if (sum(terms) - ideal_product) % rns_product:
        raise ValueError("The contributions do not reconstruct the ideal CRT class")
    dropped = modulus.bit_length() - accumulator_bits
    if dropped < 0:
        raise ValueError("Accumulator precision exceeds this numerical interface")
    unit = 1 << dropped
    tau = modulus // unit
    word = sum((term % modulus) // unit for term in terms) % tau
    delta = modular_distance(rns_product, 0, modulus)
    error = modular_distance(unit * word, ideal_product, modulus)
    bound = len(terms) * (delta + unit)
    assert error < bound
    return {"word": word, "modular_error": error, "strict_error_bound": bound,
            "terms": len(terms), "tau": tau}
