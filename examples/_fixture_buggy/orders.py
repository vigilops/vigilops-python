"""Tiny e-commerce order math.

There is one deliberate bug for the triage agent to find.
"""

TAX_RATE = 0.08


def line_total(unit_price: float, quantity: int) -> float:
    """Subtotal for a single order line."""
    return unit_price * quantity


def cart_subtotal(lines):
    """Sum subtotals across cart lines."""
    return sum(line_total(p, q) for p, q in lines)


def cart_tax(lines):
    """Sales tax on the cart."""
    return cart_subtotal(lines) * TAX_RATE


def cart_total(lines, discount: float = 0.0):
    """Final cart total: subtotal + tax - discount.

    BUG (intentional): discount is added to tax instead of subtracted
    from the total. A coupon for $10 makes the customer pay MORE.
    """
    subtotal = cart_subtotal(lines)
    tax = cart_tax(lines)
    return subtotal + tax + discount
