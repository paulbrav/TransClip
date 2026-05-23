from __future__ import annotations

from .dispatch import handle_command
from .parser import build_parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return handle_command(args, parser)


__all__ = ["handle_command", "main"]
