"""Small cart domain used by the deterministic RepoGraph demo."""

from collections.abc import Iterable


def cart_total(items: Iterable[tuple[str, int, int]]) -> int:
    """Return the total price in cents."""

    return sum(price_cents * quantity for _, price_cents, quantity in items)
