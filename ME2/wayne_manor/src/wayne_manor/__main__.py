from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Wayne Manor simulator")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--ui-mode",
        choices=("display", "controls"),
        default="display",
        help="display hides device controls; controls exposes the manual testing panel",
    )
    return parser.parse_args()


def main() -> None:
    from .app import create_app

    args = parse_args()
    config = load_config(args.config)
    uvicorn.run(
        create_app(config, ui_mode=args.ui_mode),
        host=args.host or config.server.host,
        port=args.port or config.server.port,
    )


if __name__ == "__main__":
    main()
