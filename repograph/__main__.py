"""Unified, thin developer launcher for RepoGraph."""

from __future__ import annotations

import argparse

from repograph.demo import add_demo_arguments, run_demo
from repograph.doctor import run_doctor
from repograph.hygiene import main as run_hygiene
from repograph.release_check import run_release_check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m repograph",
        description="RepoGraph developer launcher (no implicit LLM or mutation).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check prerequisites without exposing secrets.")
    demo = commands.add_parser("demo", help="Run the deterministic preview demo.")
    add_demo_arguments(demo)
    release = commands.add_parser(
        "release-check", help="Run deterministic portfolio release checks."
    )
    release.add_argument("--skip-studio", action="store_true")
    release.add_argument("--skip-docker", action="store_true")
    commands.add_parser("hygiene", help="Scan public files for secrets and personal paths.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return run_doctor()
    if args.command == "demo":
        return run_demo(args)
    if args.command == "release-check":
        return run_release_check(
            skip_studio=args.skip_studio, skip_docker=args.skip_docker
        )
    if args.command == "hygiene":
        return run_hygiene()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
