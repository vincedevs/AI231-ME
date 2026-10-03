from __future__ import annotations

import argparse
import logging
import time

from .audio import playback_output_device
from .config import load_config, load_environment_file
from .playback import FeedbackPlayer
from .tts import build_tts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a scheduled Alfred timer or alarm")
    parser.add_argument("--kind", choices=("timer", "alarm"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    load_environment_file(config.root / ".env")
    logging.basicConfig(
        level=getattr(logging, str(config.document["runtime"]["log_level"]).upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    values = config.document["actions"]["scheduler"]
    output_device = playback_output_device(config.document["audio"])
    player = FeedbackPlayer(
        config,
        output_device,
        build_tts(config, output_device),
    )
    message = "Your timer is finished." if args.kind == "timer" else "Your alarm is ringing."
    for index in range(int(values["alert_repeat_count"])):
        player.play("action_success")
        player.respond(message)
        if index + 1 < int(values["alert_repeat_count"]):
            time.sleep(float(values["alert_interval_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
