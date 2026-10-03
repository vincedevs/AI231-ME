"""Read-only Textual browser for the packaged VCM training dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

try:
    from rich.text import Text
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal
    from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Select, Static
except ModuleNotFoundError as error:
    if error.name in {"textual", "rich"}:
        raise SystemExit(
            "DJ requires Textual. Install it with: "
            "python -m pip install -r dataset_reviewer/requirements.txt"
        ) from error
    raise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "training"
ALL = "__ALL__"
REAL = "real"
SYNTHETIC = "synthetic"
SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
REQUIRED_FIELDS = {
    "sample_id",
    "audio_path",
    "source_dataset",
    "source_id",
    "transcript",
    "duration_seconds",
}


@dataclass(frozen=True)
class DatasetRecord:
    """One canonical manifest row plus its reviewer-only provenance."""

    data: dict[str, Any]
    manifest: str
    supported: bool

    @property
    def sample_id(self) -> str:
        return str(self.data["sample_id"])

    @property
    def split(self) -> str:
        return str(self.data.get("split") or self.data.get("source_split") or "train")

    @property
    def source(self) -> str:
        return str(self.data["source_dataset"])

    @property
    def intent(self) -> str:
        if not self.supported:
            return "UNSUPPORTED"
        return str(self.data.get("canonical_intent") or "[missing]")

    @property
    def transcript(self) -> str:
        return str(self.data.get("transcript") or "")

    @property
    def speaker(self) -> str:
        value = self.data.get("speaker_id")
        return "—" if value is None or value == "" else str(value)

    @property
    def duration(self) -> float:
        return float(self.data["duration_seconds"])

    @property
    def synthetic(self) -> bool:
        return bool(self.data.get("is_synthetic"))

    @property
    def origin(self) -> str:
        return SYNTHETIC if self.synthetic else REAL

    @property
    def scope(self) -> str:
        return SUPPORTED if self.supported else UNSUPPORTED

    @property
    def searchable_text(self) -> str:
        fields: Iterable[Any] = (
            self.sample_id,
            self.transcript,
            self.speaker,
            self.source,
            self.data.get("source_id", ""),
            self.data.get("source_intent", ""),
            self.data.get("mapping_rule", ""),
            self.data.get("ood_type", ""),
            self.data.get("unsupported_category", ""),
            json.dumps(self.data.get("slots", {}), ensure_ascii=False),
        )
        return " ".join(str(value) for value in fields).casefold()


class DatasetIndex:
    """In-memory, read-only index of one benchmark protocol and auxiliaries."""

    def __init__(
        self,
        dataset_root: Path,
        protocol: str,
        include_synthetic: bool,
        include_unsupported: bool,
    ) -> None:
        self.root = dataset_root.expanduser().resolve()
        self.protocol = protocol
        if not (self.root / "release.json").is_file():
            raise FileNotFoundError(f"Packaged dataset release not found: {self.root / 'release.json'}")

        manifests = [
            (self.root / "manifests" / protocol / f"{split}.jsonl", True)
            for split in SPLIT_ORDER
        ]
        if include_unsupported:
            manifests.extend(
                (self.root / "manifests" / "unsupported" / f"{split}.jsonl", False)
                for split in SPLIT_ORDER
            )
        if include_synthetic:
            manifests.append((self.root / "synthetic" / "train.jsonl", True))

        self.records: list[DatasetRecord] = []
        seen_ids: set[str] = set()
        for manifest, supported in manifests:
            self.records.extend(self._load_manifest(manifest, supported, seen_ids, self.root))
        if not self.records:
            raise ValueError("No records were loaded from the selected manifests")
        self.records.sort(
            key=lambda record: (
                SPLIT_ORDER.get(record.split, 99),
                record.scope,
                record.intent,
                record.sample_id,
            )
        )

    @staticmethod
    def _load_manifest(
        manifest: Path,
        supported: bool,
        seen_ids: set[str],
        dataset_root: Path,
    ) -> list[DatasetRecord]:
        if not manifest.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
        records = []
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {manifest}:{line_number}: {error}") from error
                missing = REQUIRED_FIELDS - set(row)
                if supported and "canonical_intent" not in row:
                    missing.add("canonical_intent")
                if missing:
                    raise ValueError(
                        f"Missing fields in {manifest}:{line_number}: {sorted(missing)}"
                    )
                sample_id = str(row["sample_id"])
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate sample ID across selected manifests: {sample_id}")
                seen_ids.add(sample_id)
                records.append(
                    DatasetRecord(
                        data=row,
                        manifest=str(manifest.relative_to(dataset_root)),
                        supported=supported,
                    )
                )
        return records

    @property
    def sources(self) -> list[str]:
        return sorted({record.source for record in self.records}, key=str.casefold)

    @property
    def intents(self) -> list[str]:
        return sorted({record.intent for record in self.records})

    @property
    def splits(self) -> list[str]:
        present = {record.split for record in self.records}
        return [split for split in SPLIT_ORDER if split in present]

    def filter(
        self,
        split: str = ALL,
        source: str = ALL,
        intent: str = ALL,
        origin: str = ALL,
        scope: str = ALL,
        query: str = "",
    ) -> list[int]:
        needle = query.strip().casefold()
        return [
            index
            for index, record in enumerate(self.records)
            if (split == ALL or record.split == split)
            and (source == ALL or record.source == source)
            and (intent == ALL or record.intent == intent)
            and (origin == ALL or record.origin == origin)
            and (scope == ALL or record.scope == scope)
            and (not needle or needle in record.searchable_text)
        ]

    def audio_path(self, record: DatasetRecord) -> Path:
        value = Path(str(record.data["audio_path"]))
        if value.is_absolute():
            candidates = [value.resolve()]
        else:
            candidates = [(PROJECT_ROOT / value).resolve(), (self.root / value).resolve()]
            if "audio" in value.parts:
                candidates.append((self.root / Path(*value.parts[value.parts.index("audio") :])).resolve())
        for candidate in candidates:
            try:
                candidate.relative_to(self.root)
            except ValueError:
                continue
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Audio file not found inside dataset root: {value}")


class AudioPlayer:
    """Subprocess wrapper around a native audio player."""

    BACKENDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "afplay": ("afplay",),
        "ffplay": ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
        "mpv": ("mpv", "--no-video", "--really-quiet"),
        "paplay": ("paplay",),
        "aplay": ("aplay", "-q"),
    }

    def __init__(self, preferred: str = "auto") -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.backend = self._discover(preferred)

    @classmethod
    def _discover(cls, preferred: str) -> str | None:
        candidates = cls.BACKENDS if preferred == "auto" else {preferred: cls.BACKENDS[preferred]}
        return next((name for name in candidates if shutil.which(name)), None)

    @property
    def name(self) -> str:
        return self.backend or "unavailable"

    def play(self, path: Path) -> None:
        if self.backend is None:
            raise RuntimeError("No audio player found. Install ffplay or mpv; macOS includes afplay.")
        self.stop()
        self.process = subprocess.Popen(
            [*self.BACKENDS[self.backend], str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> bool:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return False
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
        return True


class DJ(App[None]):
    """Terminal application for browsing and listening to VCM records."""

    TITLE = "DJ"
    SUB_TITLE = "VCM dataset reviewer"
    CSS = """
    Screen { layout: vertical; }
    #filters { height: 6; padding: 0 1; }
    #filters Select { width: 1fr; margin-right: 1; }
    #search { width: 2fr; }
    #content { height: 1fr; }
    #records { width: 2fr; height: 1fr; border: round $primary; }
    #details { width: 1fr; height: 1fr; border: round $secondary; padding: 1 2; overflow-y: auto; }
    #pager { height: 3; padding: 0 1; align: center middle; }
    #pager Button { min-width: 10; margin-right: 1; }
    #page-status { width: 1fr; content-align: center middle; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("space", "play", "Play"),
        Binding("s", "stop", "Stop"),
        Binding("left_square_bracket", "previous_page", "Previous page"),
        Binding("right_square_bracket", "next_page", "Next page"),
        Binding("r", "random_record", "Random"),
        Binding("c", "clear_filters", "Clear filters"),
        Binding("slash", "focus_search", "Search"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, index: DatasetIndex, page_size: int = 250, player: str = "auto") -> None:
        super().__init__()
        self.index = index
        self.page_size = page_size
        self.player = AudioPlayer(player)
        self.filtered_indices = list(range(len(index.records)))
        self.page = 0
        self.selected_index: int | None = None
        self._suppress_filter_events = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="filters"):
            yield Select([("All splits", ALL), *[(x, x) for x in self.index.splits]], value=ALL, allow_blank=False, id="split-filter")
            yield Select([("All sources", ALL), *[(x, x) for x in self.index.sources]], value=ALL, allow_blank=False, id="source-filter")
            yield Select([("All intents", ALL), *[(x, x) for x in self.index.intents]], value=ALL, allow_blank=False, id="intent-filter")
            yield Select([("Real + synthetic", ALL), ("Real", REAL), ("Synthetic", SYNTHETIC)], value=ALL, allow_blank=False, id="origin-filter")
            yield Select([("Supported + unsupported", ALL), ("Supported", SUPPORTED), ("Unsupported", UNSUPPORTED)], value=ALL, allow_blank=False, id="scope-filter")
            yield Input(placeholder="Search transcript, speaker, ID, source, OOD type, or slot…", id="search")
        with Horizontal(id="content"):
            yield DataTable(id="records", cursor_type="row", zebra_stripes=True)
            yield Static(id="details")
        with Horizontal(id="pager"):
            yield Button("Previous", id="previous-page")
            yield Button("Next", id="next-page")
            yield Button("Random", id="random-record")
            yield Button("Play", id="play-record", variant="primary")
            yield Button("Stop", id="stop-audio", variant="warning")
            yield Button("Clear", id="clear-filters")
            yield Label(id="page-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#records", DataTable)
        for label, width in (("#", 7), ("Split", 10), ("Intent", 19), ("Source", 10), ("Scope", 12), ("Origin", 10), ("Speaker", 16), ("Sec", 7), ("Transcript", 64)):
            table.add_column(label, width=width)
        self.refresh_table()
        table.focus()
        self.notify(f"Loaded {len(self.index.records):,} records · player: {self.player.name}", timeout=4)

    @property
    def page_count(self) -> int:
        return max(1, (len(self.filtered_indices) + self.page_size - 1) // self.page_size)

    def current_page_indices(self) -> list[int]:
        start = self.page * self.page_size
        return self.filtered_indices[start : start + self.page_size]

    def refresh_table(self, select_row: int = 0) -> None:
        table = self.query_one("#records", DataTable)
        table.clear(columns=False)
        page_indices = self.current_page_indices()
        start = self.page * self.page_size
        for offset, index in enumerate(page_indices, start=1):
            record = self.index.records[index]
            table.add_row(
                f"{start + offset:,}", record.split, record.intent, record.source,
                record.scope.title(), record.origin.title(), record.speaker,
                f"{record.duration:.2f}", record.transcript or "[non-speech audio]", key=str(index),
            )
        if page_indices:
            row = min(max(0, select_row), len(page_indices) - 1)
            table.move_cursor(row=row, column=0, animate=False)
            self.selected_index = page_indices[row]
            self.update_details(self.index.records[self.selected_index])
        else:
            self.selected_index = None
            self.query_one("#details", Static).update(Text("No matching records.", style="bold yellow"))
        self.update_page_status()

    def update_page_status(self) -> None:
        total = len(self.filtered_indices)
        if total:
            first = self.page * self.page_size + 1
            last = min((self.page + 1) * self.page_size, total)
            text = f"{first:,}–{last:,} of {total:,} matching"
        else:
            text = "0 matching"
        self.query_one("#page-status", Label).update(
            f"{text} · page {self.page + 1}/{self.page_count} · {len(self.index.records):,} total"
        )

    def update_details(self, record: DatasetRecord) -> None:
        row = record.data
        details = Text()
        fields = (
            ("Sample ID", record.sample_id), ("Manifest", record.manifest),
            ("Split", record.split), ("Intent", record.intent), ("Scope", record.scope),
            ("Source", record.source), ("Source ID", str(row.get("source_id") or "—")),
            ("Source intent", str(row.get("source_intent") or "—")),
            ("Speaker", record.speaker), ("Duration", f"{record.duration:.3f} seconds"),
            ("Sample rate", str(row.get("sample_rate") or "—")),
            ("Transcript", record.transcript or "[none: non-speech audio]"),
            ("Slots", json.dumps(row.get("slots", {}), ensure_ascii=False)),
            ("OOD type", str(row.get("ood_type") or "—")),
            ("Audio", str(row["audio_path"])),
            ("SHA-256", str(row.get("audio_sha256") or "—")),
        )
        for label, value in fields:
            details.append(f"{label}\n", style="bold cyan")
            details.append(f"{value}\n\n")
        details.append("Quality flags\n", style="bold cyan")
        details.append(json.dumps(row.get("quality_flags", []), ensure_ascii=False))
        details.append("\n\nMetadata\n", style="bold cyan")
        details.append(json.dumps(row.get("metadata", {}), indent=2, ensure_ascii=False))
        self.query_one("#details", Static).update(details)

    def apply_filters(self) -> None:
        if self._suppress_filter_events:
            return
        value = lambda selector: str(self.query_one(selector, Select).value)
        self.filtered_indices = self.index.filter(
            value("#split-filter"), value("#source-filter"), value("#intent-filter"),
            value("#origin-filter"), value("#scope-filter"), self.query_one("#search", Input).value,
        )
        self.page = 0
        self.refresh_table()

    @on(Select.Changed)
    def on_select_changed(self, _event: Select.Changed) -> None:
        self.apply_filters()

    @on(Input.Changed, "#search")
    def on_search_changed(self, _event: Input.Changed) -> None:
        self.apply_filters()

    @on(DataTable.RowHighlighted, "#records")
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            index = int(event.row_key.value)
        except (TypeError, ValueError):
            return
        self.selected_index = index
        self.update_details(self.index.records[index])

    @on(DataTable.RowSelected, "#records")
    def on_row_selected(self, _event: DataTable.RowSelected) -> None:
        self.action_play()

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "previous-page": self.action_previous_page, "next-page": self.action_next_page,
            "random-record": self.action_random_record, "play-record": self.action_play,
            "stop-audio": self.action_stop, "clear-filters": self.action_clear_filters,
        }
        action = actions.get(event.button.id or "")
        if action is not None:
            action()

    def action_play(self) -> None:
        if self.selected_index is None:
            self.notify("Select a record first.", severity="warning")
            return
        record = self.index.records[self.selected_index]
        try:
            self.player.play(self.index.audio_path(record))
        except (OSError, RuntimeError, ValueError) as error:
            self.notify(str(error), severity="error", timeout=8)
            return
        self.notify(f"Playing {record.sample_id} with {self.player.name}", timeout=3)

    def action_stop(self) -> None:
        if self.player.stop():
            self.notify("Playback stopped.", timeout=2)

    def action_previous_page(self) -> None:
        if self.page > 0:
            self.page -= 1
            self.refresh_table()

    def action_next_page(self) -> None:
        if self.page + 1 < self.page_count:
            self.page += 1
            self.refresh_table()

    def action_random_record(self) -> None:
        if not self.filtered_indices:
            self.notify("No matching records.", severity="warning")
            return
        position = random.randrange(len(self.filtered_indices))
        self.page, row = divmod(position, self.page_size)
        self.refresh_table(select_row=row)
        self.query_one("#records", DataTable).focus()

    def action_clear_filters(self) -> None:
        self._suppress_filter_events = True
        for selector in ("#split-filter", "#source-filter", "#intent-filter", "#origin-filter", "#scope-filter"):
            self.query_one(selector, Select).value = ALL
        self.query_one("#search", Input).value = ""
        self._suppress_filter_events = False
        self.apply_filters()

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    def action_quit(self) -> None:
        self.player.stop()
        self.exit()

    def on_unmount(self) -> None:
        self.player.stop()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse and play the packaged VCM dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--protocol", choices=("speaker_generalization", "phrase_generalization"),
        default="speaker_generalization",
    )
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--no-unsupported", action="store_true")
    parser.add_argument("--page-size", type=positive_int, default=250)
    parser.add_argument("--player", choices=("auto", *AudioPlayer.BACKENDS), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        index = DatasetIndex(
            args.dataset, args.protocol,
            include_synthetic=not args.no_synthetic,
            include_unsupported=not args.no_unsupported,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    DJ(index, page_size=args.page_size, player=args.player).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
