"""Deterministic artifact DAG inspection."""

from __future__ import annotations

from collections import defaultdict

from observability.models import ArtifactEdge, ArtifactRecord


def render_lineage(
    artifacts: list[ArtifactRecord], edges: list[ArtifactEdge]
) -> str:
    by_id = {item.artifact_id: item for item in artifacts}
    children: dict[str, list[ArtifactEdge]] = defaultdict(list)
    child_ids: set[str] = set()
    for edge in edges:
        children[edge.parent_artifact_id].append(edge)
        child_ids.add(edge.child_artifact_id)
    roots = sorted(
        (item for item in artifacts if item.artifact_id not in child_ids),
        key=lambda item: (item.created_at, item.kind, item.artifact_id),
    )
    lines: list[str] = []

    def visit(artifact: ArtifactRecord, prefix: str, seen: set[str]) -> None:
        label = f"{artifact.kind} [{artifact.artifact_id[:8]}] sha256:{artifact.sha256[:12]}"
        lines.append(f"{prefix}{label}")
        if artifact.artifact_id in seen:
            lines.append(f"{prefix}  <cycle rejected by storage>")
            return
        outgoing = sorted(
            children.get(artifact.artifact_id, []),
            key=lambda edge: (edge.relation, edge.child_artifact_id),
        )
        for index, edge in enumerate(outgoing):
            child = by_id.get(edge.child_artifact_id)
            branch = "`-" if index == len(outgoing) - 1 else "|-"
            lines.append(f"{prefix}{branch} {edge.relation}")
            if child is not None:
                continuation = "   " if index == len(outgoing) - 1 else "|  "
                visit(child, prefix + continuation, seen | {artifact.artifact_id})

    for root in roots:
        visit(root, "", set())
    if not lines and artifacts:
        for item in sorted(artifacts, key=lambda value: value.artifact_id):
            lines.append(f"{item.kind} [{item.artifact_id[:8]}]")
    return "\n".join(lines)


def span_tree(spans: list[object]) -> str:
    by_parent: dict[str | None, list[object]] = defaultdict(list)
    for span in spans:
        by_parent[span.parent_span_id].append(span)
    for values in by_parent.values():
        values.sort(key=lambda item: item.sequence)
    lines: list[str] = []

    def visit(span: object, prefix: str, connector: str = "") -> None:
        duration = span.duration_ms
        suffix = f" {duration:.3f}ms" if duration is not None else " incomplete"
        lines.append(
            f"{prefix}{connector}{span.name} [{span.kind}] "
            f"{span.status}{suffix}"
        )
        children = by_parent.get(span.span_id, [])
        nested_prefix = prefix
        if connector == "|- ":
            nested_prefix += "|  "
        elif connector == "`- ":
            nested_prefix += "   "
        for index, child in enumerate(children):
            last = index == len(children) - 1
            branch = "`- " if last else "|- "
            visit(child, nested_prefix, branch)

    roots = by_parent.get(None, [])
    for root in roots:
        visit(root, "")
    return "\n".join(lines)
