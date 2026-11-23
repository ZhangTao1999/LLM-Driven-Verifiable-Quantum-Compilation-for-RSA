"""Integer-only exponential and full-support Chernoff certificate primitives.

No producer, error-budget implementation, numerical exp, or success scorer
is imported. All inequalities use exact positive rational arithmetic.
"""
from __future__ import annotations
from array import array
from fractions import Fraction
from math import isqrt

LAMBDA_NUM = 42
LAMBDA_DEN = 1 << 24


def ceil_div(numerator, denominator):
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil_div requires nonnegative numerator and positive denominator")
    return (numerator + denominator - 1) // denominator


def rate_exp_upper(scale_bits=39, *, lambda_num=None, lambda_den=None):
    """R/2**f >= exp(42/2**24), by a Taylor polynomial + geometric tail."""
    value = Fraction(LAMBDA_NUM if lambda_num is None else lambda_num,
                     LAMBDA_DEN if lambda_den is None else lambda_den)
    if not 0 < value < 4:
        raise ValueError("Exponential upper rate requires 0 < lambda < 4")
    # Ratios after the cubic term are at most value/4.
    upper = 1 + value + value * value / 2 + value**3 / (6 * (1 - value / 4))
    scale = 1 << scale_bits
    return ceil_div(upper.numerator * scale, upper.denominator)


def make_exp_upper_lut(max_value, *, scale_bits=39, lambda_num=None, lambda_den=None):
    """E[x]/F >= exp(lambda*x), with every recurrence rounded upward."""
    if max_value < 0:
        raise ValueError("Negative exponential index")
    scale = 1 << scale_bits
    rate = rate_exp_upper(scale_bits, lambda_num=lambda_num, lambda_den=lambda_den)
    increment = rate - scale
    values = array("Q", [scale])
    for _ in range(max_value):
        previous = values[-1]
        values.append(previous + ceil_div(previous * increment, scale))
    if max(values) > (1 << 64) - 1:
        raise OverflowError("The exponential LUT does not fit uint64")
    return values


def verify_exp_upper_lut(values, *, scale_bits=39, lambda_num=None, lambda_den=None):
    scale = 1 << scale_bits
    rate = rate_exp_upper(scale_bits, lambda_num=lambda_num, lambda_den=lambda_den)
    if not values or values[0] != scale:
        raise ValueError("Incorrect initial exponential value")
    for previous, actual in zip(values, values[1:]):
        if actual != previous + ceil_div(previous * (rate - scale), scale):
            raise ValueError("Exponential LUT violates its outward-rounded recurrence")
    return {"scale_bits": scale_bits, "rate_upper_integer": rate,
            "largest_entry": int(values[-1]), "recurrence_verified": True}


def exp_lower_scaled(numerator, denominator, *, scale_bits=256, terms=512):
    """Integer L/2**g <= exp(numerator/denominator), by downward Taylor terms.

    term_k=floor(term_(k-1)*z/k) <= 2**g*z**k/k!. Summing finitely many
    positive terms is therefore an exponential lower bound even when a
    small term has rounded to zero. No tail estimate or float is needed.
    """
    if numerator < 0 or denominator <= 0 or terms < 0:
        raise ValueError("Invalid positive exponential argument")
    term = total = 1 << scale_bits
    for index in range(1, terms + 1):
        term = term * numerator // (denominator * index)
        total += term
    return total


def probability_numerator(mgf_sums, address_counts, threshold, *, lut_scale_bits=39,
                          probability_bits=192, lower_scale_bits=256, terms=512,
                          lambda_num=None, lambda_den=None):
    """One Chernoff tail, rounded upward to the grid 2**(-probability_bits)."""
    if len(mgf_sums) != len(address_counts) or threshold <= 0:
        raise ValueError("Invalid window count or Markov threshold")
    product = 1
    denominator_bits = lut_scale_bits * len(mgf_sums)
    for value, count in zip(mgf_sums, address_counts, strict=True):
        if value <= 0 or count <= 0 or count & (count - 1):
            raise ValueError("Each complete address domain must be a positive power of two")
        product *= int(value)
        denominator_bits += count.bit_length() - 1
    num = LAMBDA_NUM if lambda_num is None else lambda_num
    den = LAMBDA_DEN if lambda_den is None else lambda_den
    if num <= 0 or den <= 0:
        raise ValueError("Markov parameter must be positive")
    lower = exp_lower_scaled(num * threshold, den,
                              scale_bits=lower_scale_bits, terms=terms)
    numerator = product << (lower_scale_bits + probability_bits)
    denominator = lower << denominator_bits
    return min(1 << probability_bits, ceil_div(numerator, denominator))


def certify_prime(*, prime, center, plus_sums, minus_sums, address_counts,
                  stored_bits=26, lut_scale_bits=39, probability_bits=192,
                  lambda_num=None, lambda_den=None, representation="signed"):
    """Certify per-prime Stage-(i) wraparound tail for signed (Ds) or offset-unsigned (Du) lanes."""
    if prime < 3 or stored_bits < 2:
        raise ValueError("Invalid prime or stored width")
    windows = len(address_counts)
    if representation == "signed":
        half = 1 << (stored_bits - 1)
    elif representation == "shifted_unsigned":
        ell = prime.bit_length()
        if stored_bits < ell + 1:
            raise ValueError("shifted_unsigned requires stored_bits >= prime.bit_length() + 1")
        half = (1 << (stored_bits - ell - 1)) * (prime - 1)
    else:
        raise ValueError(f"Unknown representation: {representation!r}")
    # Safe interval around center is [center - half, center + half - 1].
    # Upper failure: S >= center + half. Lower failure: S <= center - half - 1,
    # i.e. reflected sum W*(p-2) - S >= W*(p-2) - center + half + 1.
    positive_threshold = center + half
    negative_threshold = windows * (prime - 2) - center + half + 1
    plus = probability_numerator(plus_sums, address_counts, positive_threshold,
                                 lut_scale_bits=lut_scale_bits, probability_bits=probability_bits,
                                 lambda_num=lambda_num, lambda_den=lambda_den)
    minus = probability_numerator(minus_sums, address_counts, negative_threshold,
                                  lut_scale_bits=lut_scale_bits, probability_bits=probability_bits,
                                  lambda_num=lambda_num, lambda_den=lambda_den)
    return {"prime": prime, "center": center, "stored_bits": stored_bits,
            "representation": representation, "half_width": half,
            "upper_failure_threshold": positive_threshold,
            "reflected_lower_failure_threshold": negative_threshold,
            "upper_numerator": plus, "lower_numerator": minus,
            "bad_numerator": min(1 << probability_bits, plus + minus),
            "probability_bits": probability_bits}


def certify_prime_shifted_unsigned(*, prime, center, plus_sums, minus_sums, address_counts,
                                   stored_bits=26, lut_scale_bits=39, probability_bits=192,
                                   lambda_num=None, lambda_den=None):
    """Offset-unsigned (Du) Chernoff certification with divisor-aligned half-width H_p = 2^(D-ell-1)*(p-1)."""
    return certify_prime(
        prime=prime, center=center, plus_sums=plus_sums, minus_sums=minus_sums,
        address_counts=address_counts, stored_bits=stored_bits,
        lut_scale_bits=lut_scale_bits, probability_bits=probability_bits,
        lambda_num=lambda_num, lambda_den=lambda_den,
        representation="shifted_unsigned",
    )


def trace_upper_from_numerator(bad_numerator, *, probability_bits=192, trace_bits=120):
    """Return an exact dyadic >= min(1,2*sqrt(p_bad))."""
    if bad_numerator < 0:
        raise ValueError("Negative bad-subspace weight")
    scale = 1 << trace_bits
    squared_scaled = ceil_div(4 * bad_numerator * scale * scale, 1 << probability_bits)
    numerator = isqrt(squared_scaled)
    if numerator * numerator < squared_scaled:
        numerator += 1
    numerator = min(scale, numerator)
    assert numerator == scale or numerator * numerator * (1 << probability_bits) >= 4 * bad_numerator * scale * scale
    return Fraction(numerator, scale)


def exact_center(window_sums, address_counts):
    if len(window_sums) != len(address_counts) or not address_counts:
        raise ValueError("Invalid window statistics")
    denominator = max(address_counts)
    if any(count <= 0 or denominator % count for count in address_counts):
        raise ValueError("Window counts need a common dyadic denominator")
    numerator = sum(int(value) * (denominator // count)
                    for value, count in zip(window_sums, address_counts, strict=True))
    return (numerator + denominator // 2) // denominator, numerator, denominator
