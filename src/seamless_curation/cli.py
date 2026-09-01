"""Command-line interface for the measurement harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .harness import load_config, read_manifest, run, select_manifest_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML harness configuration")
    parser.add_argument("--limit", type=int, help="process only the first N manifest rows")
    parser.add_argument("--index", type=int, help="process one zero-based selected manifest row")
    parser.add_argument("--force", action="store_true", help="recompute matching completed files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        rows = select_manifest_rows(read_manifest(config), args.limit, args.index)
        results = run(config, rows, force=args.force)
    except Exception as exc:
        print(f"configuration/selection error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for result in results:
        print(
            json.dumps(
                {
                    "file_id": result.file_id,
                    "status": result.status,
                    "rows": result.rows,
                    "output": str(result.output_path) if result.output_path else None,
                    "message": result.message,
                },
                sort_keys=True,
            )
        )
    return 1 if any(result.status == "error" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

