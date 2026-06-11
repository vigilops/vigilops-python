"""Checkout flow — uses orders.cart_total under the hood.

Distractor file (no bug here). Triage agent should NOT focus on this one.
"""

from .orders import cart_total


def quote(cart, coupon_value: float = 0.0) -> dict:
    total = cart_total(cart, discount=coupon_value)
    return {"total": round(total, 2), "currency": "USD"}


def confirm(cart, coupon_value: float = 0.0) -> dict:
    q = quote(cart, coupon_value=coupon_value)
    if q["total"] < 0:
        raise ValueError("total must be non-negative")
    return {"status": "ok", **q}
