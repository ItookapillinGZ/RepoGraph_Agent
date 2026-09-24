from repograph_demo.cart import cart_total
from repograph_demo.report import render_summary


def test_non_positive_quantities_are_ignored() -> None:
    items = [("book", 1200, 2), ("invalid", 999, 0), ("return", 500, -1)]
    assert cart_total(items) == 2400


def test_summary_is_stable() -> None:
    items = [("book", 1200, 2), ("pen", 150, 1), ("invalid", 999, 0)]
    assert render_summary(items) == "items=2 total_cents=2550"
