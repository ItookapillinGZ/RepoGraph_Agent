"""Generate the 20-task H2.3 benchmark with evaluator-only hidden tests."""

from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HardFixture:
    task_id: str
    issue: str
    difficulty: tuple[str, ...]
    files: dict[str, str]
    hidden_test: str
    expected_files: tuple[str, ...] = ()


def _d(value: str) -> str:
    return textwrap.dedent(value).lstrip()


def _fixture(
    task_id: str,
    issue: str,
    difficulty: tuple[str, ...],
    source_files: dict[str, str],
    public_test: str,
    hidden_test: str,
    expected_files: tuple[str, ...] = (),
) -> HardFixture:
    files = {
        "pyproject.toml": _d(
            """
            [tool.pytest.ini_options]
            testpaths = ["tests"]
            addopts = "-q"
            """
        ),
        "README.md": f"# {task_id}\n\nSmall service fixture for RepoGraph evaluation.\n",
        "src/__init__.py": "",
        **{path: _d(value) for path, value in source_files.items()},
        "tests/test_public.py": _d(public_test),
    }
    return HardFixture(
        task_id,
        issue,
        difficulty,
        files,
        _d(hidden_test),
        expected_files,
    )


HARD_FIXTURES = (
    _fixture(
        "unicode-cache-identity",
        "Cache lookups should treat Unicode case variants as one key; ASCII behavior already works.",
        ("multi-file", "edge-case", "exploration-heavy"),
        {
            "src/keys.py": """
                def normalize_key(value: str) -> str:
                    return value.strip().lower()
            """,
            "src/cache.py": """
                from src.keys import normalize_key

                class Cache:
                    def __init__(self):
                        self._values = {}
                    def put(self, key, value):
                        self._values[normalize_key(key)] = value
                    def get(self, key):
                        return self._values.get(normalize_key(key))
            """,
        },
        """
        from src.cache import Cache
        def test_ascii_identity():
            cache = Cache()
            cache.put(" User ", 7)
            assert cache.get("user") == 7
        """,
        """
        from src.cache import Cache
        def test_unicode_case_identity():
            cache = Cache()
            cache.put("Straße", 9)
            assert cache.get("STRASSE") == 9
            assert len(cache._values) == 1
        """,
        ("src/keys.py",),
    ),
    _fixture(
        "timeout-disabled-zero",
        "Timeout zero means disabled, empty input uses the default, and negative values must be rejected.",
        ("multi-file", "configuration", "edge-case"),
        {
            "src/config.py": """
                DEFAULT_TIMEOUT = 30
                def parse_timeout(raw):
                    return int(raw) if raw else DEFAULT_TIMEOUT
            """,
            "src/client.py": """
                from src.config import parse_timeout
                def deadline(now, raw_timeout):
                    return now + parse_timeout(raw_timeout)
            """,
        },
        """
        from src.client import deadline
        def test_default_timeout():
            assert deadline(10, "") == 40
        """,
        """
        import pytest
        from src.client import deadline
        def test_zero_disables_deadline():
            assert deadline(10, "0") is None
        def test_negative_timeout_rejected():
            with pytest.raises(ValueError):
                deadline(10, "-1")
        """,
    ),
    _fixture(
        "session-refresh-persistence",
        "After a 401 the refreshed token works once but is lost; persist it without changing transport APIs.",
        ("multi-file", "stateful", "non-local"),
        {
            "src/tokens.py": """
                class TokenStore:
                    def __init__(self, token):
                        self.token = token
                    def get(self):
                        return self.token
                    def set(self, token):
                        self.token = token
            """,
            "src/session.py": """
                class Session:
                    def __init__(self, store, provider, transport):
                        self.store, self.provider, self.transport = store, provider, transport
                    def request(self):
                        token = self.store.get()
                        response = self.transport(token)
                        if response == 401:
                            token = self.provider.refresh(token)
                            return self.transport(token)
                        return response
            """,
        },
        """
        from src.session import Session
        from src.tokens import TokenStore
        class Provider:
            def refresh(self, token):
                return "new"
        def test_success_does_not_refresh():
            assert Session(TokenStore("ok"), Provider(), lambda token: 200).request() == 200
        """,
        """
        from src.session import Session
        from src.tokens import TokenStore
        class Provider:
            def refresh(self, token):
                return "new"
        def test_refresh_is_persisted():
            seen = []
            def transport(token):
                seen.append(token)
                return 401 if token == "old" else 200
            store = TokenStore("old")
            session = Session(store, Provider(), transport)
            assert session.request() == 200
            assert session.request() == 200
            assert seen == ["old", "new", "new"]
            assert store.get() == "new"
        """,
        ("src/session.py", "src/tokens.py"),
    ),
    _fixture(
        "pagination-zero-cursor",
        "Integer cursor 0 is a valid continuation token, but collection currently stops at that boundary.",
        ("multi-file", "edge-case", "stateful"),
        {
            "src/api.py": """
                class Page:
                    def __init__(self, items, next_cursor):
                        self.items, self.next_cursor = items, next_cursor
            """,
            "src/pager.py": """
                def collect(fetch):
                    items, cursor = [], None
                    while True:
                        page = fetch(cursor)
                        items.extend(page.items)
                        cursor = page.next_cursor
                        if not cursor:
                            return items
            """,
        },
        """
        from src.api import Page
        from src.pager import collect
        def test_single_page():
            assert collect(lambda cursor: Page([1], None)) == [1]
        """,
        """
        from src.api import Page
        from src.pager import collect
        def test_zero_is_valid_cursor():
            pages = {None: Page([1], 0), 0: Page([2], None)}
            assert collect(lambda cursor: pages[cursor]) == [1, 2]
        """,
        ("src/pager.py", "src/api.py"),
    ),
    _fixture(
        "specific-route-precedence",
        "Overlapping admin routes are dispatched to the public handler; preserve routes and prefer specificity.",
        ("multi-file", "configuration", "exploration-heavy"),
        {
            "src/routes.py": """
                ROUTES = [("/api", "public"), ("/api/admin", "admin")]
            """,
            "src/router.py": """
                from src.routes import ROUTES
                def dispatch(path):
                    for prefix, handler in ROUTES:
                        if path.startswith(prefix):
                            return handler
                    return None
            """,
        },
        """
        from src.router import dispatch
        def test_public_route():
            assert dispatch("/api/users") == "public"
        """,
        """
        from src.router import dispatch
        def test_most_specific_route_wins():
            assert dispatch("/api/admin/users") == "admin"
            assert dispatch("/api/users") == "public"
        """,
        ("src/router.py", "src/routes.py"),
    ),
    _fixture(
        "inventory-reservation-atomicity",
        "A failed multi-item reservation must leave all stock unchanged; successful reservations still deduct.",
        ("multi-file", "stateful", "transactional"),
        {
            "src/store.py": """
                class Inventory:
                    def __init__(self, stock):
                        self.stock = dict(stock)
                    def available(self, sku):
                        return self.stock.get(sku, 0)
                    def deduct(self, sku, count):
                        self.stock[sku] = self.available(sku) - count
            """,
            "src/service.py": """
                def reserve(inventory, items):
                    for sku, count in items:
                        if inventory.available(sku) < count:
                            return False
                        inventory.deduct(sku, count)
                    return True
            """,
        },
        """
        from src.service import reserve
        from src.store import Inventory
        def test_success():
            inventory = Inventory({"a": 2})
            assert reserve(inventory, [("a", 1)])
            assert inventory.stock == {"a": 1}
        """,
        """
        from src.service import reserve
        from src.store import Inventory
        def test_failed_batch_is_atomic():
            inventory = Inventory({"a": 2, "b": 0})
            assert reserve(inventory, [("a", 1), ("b", 1)]) is False
            assert inventory.stock == {"a": 2, "b": 0}
        """,
        ("src/service.py", "src/store.py"),
    ),
    _fixture(
        "permission-parent-inheritance",
        "Resources without direct permissions inherit the nearest ancestor; child decisions override parents.",
        ("multi-file", "non-local", "stateful"),
        {
            "src/tree.py": """
                class ResourceTree:
                    def __init__(self, parents):
                        self.parents = dict(parents)
                    def parent(self, node):
                        return self.parents.get(node)
            """,
            "src/policy.py": """
                def allowed(tree, decisions, resource):
                    return decisions.get(resource, False)
            """,
        },
        """
        from src.policy import allowed
        from src.tree import ResourceTree
        def test_direct_decision():
            assert allowed(ResourceTree({}), {"root": True}, "root") is True
        """,
        """
        from src.policy import allowed
        from src.tree import ResourceTree
        def test_nearest_ancestor_and_override():
            tree = ResourceTree({"leaf": "team", "team": "root"})
            assert allowed(tree, {"root": True}, "leaf") is True
            assert allowed(tree, {"root": True, "team": False}, "leaf") is False
        """,
        ("src/policy.py", "src/tree.py"),
    ),
    _fixture(
        "domain-timezone-serialization",
        "Naive domain timestamps use the configured local timezone, not UTC; aware values remain correct.",
        ("multi-file", "configuration", "edge-case"),
        {
            "src/settings.py": """
                from datetime import timedelta, timezone
                DOMAIN_TIMEZONE = timezone(timedelta(hours=8))
            """,
            "src/serializer.py": """
                from datetime import timezone
                def to_wire(value):
                    if value.tzinfo is None:
                        value = value.replace(tzinfo=timezone.utc)
                    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            """,
        },
        """
        from datetime import datetime, timezone
        from src.serializer import to_wire
        def test_aware_utc():
            assert to_wire(datetime(2020, 1, 1, tzinfo=timezone.utc)).endswith("Z")
        """,
        """
        from datetime import datetime
        from src.serializer import to_wire
        def test_naive_uses_domain_timezone():
            assert to_wire(datetime(2020, 1, 1, 12, 0)) == "2020-01-01T04:00:00Z"
        """,
        ("src/serializer.py", "src/settings.py"),
    ),
    _fixture(
        "registry-alias-collision",
        "Plugin aliases differing by whitespace, separators, or Unicode case must be one registration.",
        ("multi-file", "edge-case", "stateful"),
        {
            "src/names.py": """
                def canonical_name(value):
                    return value.lower().replace("-", "_")
            """,
            "src/registry.py": """
                from src.names import canonical_name
                class Registry:
                    def __init__(self):
                        self.items = {}
                    def register(self, name, plugin):
                        key = canonical_name(name)
                        if key in self.items:
                            raise ValueError("duplicate")
                        self.items[key] = plugin
            """,
        },
        """
        from src.registry import Registry
        def test_hyphen_collision():
            registry = Registry()
            registry.register("my-plugin", 1)
            try:
                registry.register("my_plugin", 2)
            except ValueError:
                pass
            else:
                assert False
        """,
        """
        import pytest
        from src.registry import Registry
        def test_unicode_and_whitespace_collision():
            registry = Registry()
            registry.register(" Straße-Plugin ", 1)
            with pytest.raises(ValueError):
                registry.register("STRASSE_plugin", 2)
        """,
        ("src/names.py", "src/registry.py"),
    ),
    _fixture(
        "settings-source-precedence",
        "Configuration precedence is CLI, environment, file, defaults; lower-priority sources now overwrite it.",
        ("multi-file", "configuration", "non-local"),
        {
            "src/sources.py": """
                DEFAULTS = {"mode": "safe", "retries": 1}
                def merge(file_values, env_values, cli_values):
                    return {**DEFAULTS, **env_values, **cli_values, **file_values}
            """,
            "src/settings.py": """
                from src.sources import merge
                def load(file_values=None, env_values=None, cli_values=None):
                    return merge(file_values or {}, env_values or {}, cli_values or {})
            """,
        },
        """
        from src.settings import load
        def test_defaults():
            assert load()["mode"] == "safe"
        """,
        """
        from src.settings import load
        def test_priority_order():
            values = load(
                {"mode": "file", "retries": 2},
                {"mode": "env", "retries": 3},
                {"mode": "cli"},
            )
            assert values == {"mode": "cli", "retries": 3}
        """,
        ("src/sources.py", "src/settings.py"),
    ),
    _fixture(
        "retry-idempotency-key",
        "A timed-out write may have committed; retries must reuse one idempotency key for the logical operation.",
        ("multi-file", "stateful", "non-local"),
        {
            "src/keys.py": """
                import uuid
                def new_key():
                    return uuid.uuid4().hex
            """,
            "src/client.py": """
                from src.keys import new_key
                def create(transport, payload, attempts=2):
                    for attempt in range(attempts):
                        try:
                            return transport(payload, new_key())
                        except TimeoutError:
                            if attempt + 1 == attempts:
                                raise
            """,
        },
        """
        from src.client import create
        def test_success():
            assert create(lambda payload, key: payload["value"], {"value": 3}) == 3
        """,
        """
        from src.client import create
        def test_retry_reuses_key():
            keys = []
            def transport(payload, key):
                keys.append(key)
                if len(keys) == 1:
                    raise TimeoutError
                return "ok"
            assert create(transport, {"value": 1}) == "ok"
            assert keys[0] == keys[1]
        """,
        ("src/client.py", "src/keys.py"),
    ),
    _fixture(
        "batch-rollback-state",
        "If one record in a batch raises, earlier writes from that batch must roll back via store hooks.",
        ("multi-file", "stateful", "transactional"),
        {
            "src/store.py": """
                class Store:
                    def __init__(self):
                        self.values, self.snapshot = [], None
                    def begin(self):
                        self.snapshot = list(self.values)
                    def add(self, value):
                        self.values.append(value)
                    def rollback(self):
                        pass
                    def commit(self):
                        self.snapshot = None
            """,
            "src/processor.py": """
                def process(store, records, transform):
                    store.begin()
                    try:
                        for record in records:
                            store.add(transform(record))
                    except Exception:
                        store.rollback()
                        raise
                    store.commit()
            """,
        },
        """
        from src.processor import process
        from src.store import Store
        def test_success():
            store = Store()
            process(store, [1, 2], lambda value: value * 2)
            assert store.values == [2, 4]
        """,
        """
        import pytest
        from src.processor import process
        from src.store import Store
        def test_failure_rolls_back():
            store = Store()
            store.values = ["existing"]
            def transform(value):
                if value == 2:
                    raise ValueError
                return value
            with pytest.raises(ValueError):
                process(store, [1, 2], transform)
            assert store.values == ["existing"]
        """,
        ("src/store.py", "src/processor.py"),
    ),
    _fixture(
        "workspace-path-containment",
        "Containment checks must support Windows separators and casing without accepting sibling prefix paths.",
        ("multi-file", "edge-case", "platform"),
        {
            "src/paths.py": """
                def normalize(value):
                    return value.replace(chr(92), "/")
            """,
            "src/loader.py": """
                from src.paths import normalize
                def contains(root, candidate):
                    return normalize(candidate).startswith(normalize(root))
            """,
        },
        """
        from src.loader import contains
        def test_child_path():
            assert contains("/work/app", "/work/app/src/a.py")
        """,
        """
        from src.loader import contains
        def test_windows_case_and_sibling_boundary():
            assert contains(r"C:\\Work\\App", r"c:\\work\\app\\src\\a.py")
            assert not contains(r"C:\\Work\\App", r"C:\\Work\\App2\\a.py")
        """,
        ("src/loader.py", "src/paths.py"),
    ),
    _fixture(
        "record-parser-contract",
        "Parsed records use named fields, but end-to-end rendering still follows the legacy tuple contract.",
        ("multi-file", "refactor", "api-consistency"),
        {
            "src/parser.py": """
                def parse_record(line):
                    name, raw_count = line.split(",", 1)
                    return {"name": name.strip(), "count": int(raw_count)}
            """,
            "src/formatter.py": """
                def render(record):
                    name, count = record
                    return f"{name}:{count}"
            """,
            "src/service.py": """
                from src.formatter import render
                from src.parser import parse_record
                def convert(line):
                    return render(parse_record(line))
            """,
        },
        """
        from src.parser import parse_record
        def test_parser():
            assert parse_record("alpha,2") == {"name": "alpha", "count": 2}
        """,
        """
        from src.service import convert
        def test_end_to_end_contract():
            assert convert(" alpha ,2") == "alpha:2"
        """,
        ("src/parser.py", "src/formatter.py", "src/service.py"),
    ),
    _fixture(
        "nested-secret-redaction",
        "Audit records redact top-level secrets but leak nested dictionary and list values; preserve structure.",
        ("multi-file", "security", "recursive"),
        {
            "src/redact.py": """
                SECRET_KEYS = {"token", "password", "api_key"}
                def redact(payload):
                    return {
                        key: ("[REDACTED]" if key in SECRET_KEYS else value)
                        for key, value in payload.items()
                    }
            """,
            "src/audit.py": """
                from src.redact import redact
                def record(event, payload):
                    return {"event": event, "payload": redact(payload)}
            """,
        },
        """
        from src.audit import record
        def test_top_level_redaction():
            assert record("login", {"password": "x"})["payload"]["password"] == "[REDACTED]"
        """,
        """
        from src.audit import record
        def test_nested_redaction():
            value = record("call", {"meta": {"token": "x"}, "items": [{"api_key": "y"}]})
            assert value["payload"] == {
                "meta": {"token": "[REDACTED]"},
                "items": [{"api_key": "[REDACTED]"}],
            }
        """,
        ("src/redact.py", "src/audit.py"),
    ),
    _fixture(
        "dependency-load-order",
        "Components must load after dependencies, independent order stays deterministic, and cycles are rejected.",
        ("multi-file", "exploration-heavy", "graph"),
        {
            "src/graph.py": """
                class DependencyGraph:
                    def __init__(self, dependencies):
                        self.dependencies = dependencies
            """,
            "src/loader.py": """
                def load_order(graph):
                    order, visiting, visited = [], set(), set()
                    def visit(node):
                        if node in visiting:
                            raise ValueError("cycle")
                        if node in visited:
                            return
                        visiting.add(node)
                        order.append(node)
                        for dependency in graph.dependencies.get(node, []):
                            visit(dependency)
                        visiting.remove(node)
                        visited.add(node)
                    for node in sorted(graph.dependencies):
                        visit(node)
                    return order
            """,
        },
        """
        from src.graph import DependencyGraph
        from src.loader import load_order
        def test_single():
            assert load_order(DependencyGraph({"a": []})) == ["a"]
        """,
        """
        import pytest
        from src.graph import DependencyGraph
        from src.loader import load_order
        def test_dependencies_precede_consumers():
            order = load_order(
                DependencyGraph({"api": ["zdb"], "zdb": ["zcore"], "zcore": []})
            )
            assert order.index("zcore") < order.index("zdb") < order.index("api")
        def test_cycle_rejected():
            with pytest.raises(ValueError):
                load_order(DependencyGraph({"a": ["b"], "b": ["a"]}))
        """,
        ("src/loader.py", "src/graph.py"),
    ),
    _fixture(
        "invoice-rounding-policy",
        "Invoice totals must follow the configured half-even currency policy, not a hard-coded tie rule.",
        ("multi-file", "configuration", "edge-case"),
        {
            "src/settings.py": """
                from decimal import ROUND_HALF_EVEN
                CURRENCY_ROUNDING = ROUND_HALF_EVEN
            """,
            "src/money.py": """
                from decimal import Decimal, ROUND_HALF_UP
                def cents(value):
                    return Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            """,
            "src/invoice.py": """
                from src.money import cents
                def total(values):
                    return cents(sum(values))
            """,
        },
        """
        from decimal import Decimal
        from src.invoice import total
        def test_regular_rounding():
            assert total([Decimal("1.111"), Decimal("1.112")]) == Decimal("2.22")
        """,
        """
        from decimal import Decimal
        from src.invoice import total
        def test_half_even_tie():
            assert total([Decimal("2.345")]) == Decimal("2.34")
        """,
        ("src/money.py", "src/settings.py", "src/invoice.py"),
    ),
    _fixture(
        "event-retry-deduplication",
        "A failed handler must be retryable; marking an event seen before success currently loses it.",
        ("multi-file", "stateful", "failure-recovery"),
        {
            "src/events.py": """
                class SeenEvents:
                    def __init__(self):
                        self.ids = set()
                    def contains(self, event_id):
                        return event_id in self.ids
                    def add(self, event_id):
                        self.ids.add(event_id)
            """,
            "src/projector.py": """
                def project(seen, event_id, handler):
                    if seen.contains(event_id):
                        return False
                    seen.add(event_id)
                    handler()
                    return True
            """,
        },
        """
        from src.events import SeenEvents
        from src.projector import project
        def test_duplicate_ignored_after_success():
            seen = SeenEvents()
            assert project(seen, "a", lambda: None)
            assert not project(seen, "a", lambda: None)
        """,
        """
        import pytest
        from src.events import SeenEvents
        from src.projector import project
        def test_failure_can_retry():
            seen = SeenEvents()
            with pytest.raises(RuntimeError):
                project(seen, "a", lambda: (_ for _ in ()).throw(RuntimeError()))
            calls = []
            assert project(seen, "a", lambda: calls.append("ok"))
            assert calls == ["ok"]
        """,
        ("src/projector.py", "src/events.py"),
    ),
    _fixture(
        "email-domain-policy",
        "Domain allow-list checks are case-insensitive and must reject suffix spoofing of an allowed domain.",
        ("multi-file", "security", "edge-case"),
        {
            "src/address.py": """
                def domain(email):
                    return email.rsplit("@", 1)[-1]
            """,
            "src/policy.py": """
                from src.address import domain
                def allowed(email, domains):
                    value = domain(email)
                    return any(value.endswith(item) for item in domains)
            """,
        },
        """
        from src.policy import allowed
        def test_exact_domain():
            assert allowed("a@example.com", ["example.com"])
        """,
        """
        from src.policy import allowed
        def test_case_and_suffix_boundary():
            assert allowed("a@EXAMPLE.COM", ["example.com"])
            assert not allowed("a@attacker-example.com", ["example.com"])
        """,
        ("src/address.py", "src/policy.py"),
    ),
    _fixture(
        "subscription-listener-removal",
        "Removing one callback during emit must not skip the next callback, and later emits stay correct.",
        ("multi-file", "stateful", "edge-case"),
        {
            "src/subscriptions.py": """
                class Subscriptions:
                    def __init__(self):
                        self.listeners = []
                    def add(self, callback):
                        self.listeners.append(callback)
                    def remove(self, callback):
                        self.listeners.remove(callback)
            """,
            "src/emitter.py": """
                def emit(subscriptions, value):
                    for listener in subscriptions.listeners:
                        listener(value)
            """,
        },
        """
        from src.emitter import emit
        from src.subscriptions import Subscriptions
        def test_one_listener():
            values = []
            subscriptions = Subscriptions()
            subscriptions.add(values.append)
            emit(subscriptions, 1)
            assert values == [1]
        """,
        """
        from src.emitter import emit
        from src.subscriptions import Subscriptions
        def test_removal_during_emit_does_not_skip_next():
            calls = []
            subscriptions = Subscriptions()
            def first(value):
                calls.append("first")
                subscriptions.remove(first)
            def second(value):
                calls.append("second")
            subscriptions.add(first)
            subscriptions.add(second)
            emit(subscriptions, 1)
            assert calls == ["first", "second"]
            emit(subscriptions, 2)
            assert calls == ["first", "second", "second"]
        """,
        ("src/emitter.py", "src/subscriptions.py"),
    ),
)


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(  # nosec B603 B607
        ["git", *args],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    return completed.stdout.strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_h2_3_benchmark(destination: str | Path) -> Path:
    """Create 20 committed repositories and hidden evaluator-only suites."""

    root = Path(destination).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Fixture destination must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    sources = root / "sources"
    hidden_root = root / "evaluator-only" / "hidden-tests"
    sources.mkdir()
    hidden_root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for definition in HARD_FIXTURES:
        repository = sources / definition.task_id
        repository.mkdir()
        for relative, value in definition.files.items():
            target = repository / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value, encoding="utf-8")
        task_hidden_root = hidden_root / definition.task_id
        task_hidden_root.mkdir()
        hidden_test_path = task_hidden_root / "test_hidden.py"
        hidden_test_path.write_text(definition.hidden_test, encoding="utf-8")
        _git(repository, "init", "-b", "main")
        _git(repository, "config", "user.name", "RepoGraph H2.3 Fixtures")
        _git(repository, "config", "user.email", "evaluation@example.invalid")
        _git(repository, "config", "core.autocrlf", "false")
        _git(repository, "add", ".")
        _git(repository, "commit", "-m", "h2.3 benchmark base")
        source_file_count = sum(
            path.is_file() for path in repository.rglob("*") if ".git" not in path.parts
        )
        records.append(
            {
                "id": definition.task_id,
                "dataset": "repograph-local-h2-3-v1",
                "repository": str(repository),
                "base_commit": _git(repository, "rev-parse", "HEAD"),
                "task": definition.issue,
                "test_command": [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests",
                    str(task_hidden_root),
                    "-q",
                ],
                "expected_files": list(definition.expected_files),
                "metadata": {
                    "fixture_version": 1,
                    "difficulty": list(definition.difficulty),
                    "requires_multi_file_exploration": (
                        "multi-file" in definition.difficulty
                    ),
                    "repository_file_count": source_file_count,
                    "hidden_test_sha256": _sha256(definition.hidden_test),
                },
            }
        )
    dataset = root / "h2-3-local-v1.jsonl"
    dataset.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    return dataset
