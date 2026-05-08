from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
from pathlib import Path
from typing import List, Union, Optional

import mido
import yaml


# =========================
# 1. Config
# =========================

def _default_config_path() -> Path:
    """
    默认假设本脚本放在：
        D:/organ-amt-generalization/scripts/clean_midi_channel.py

    配置文件放在：
        D:/organ-amt-generalization/configs/clean_midi_channel.yaml
    """
    return Path(__file__).resolve().parents[1] / "configs" / "clean_midi_channel.yaml"


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    if config_path is None:
        config_path = _default_config_path()

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg or {}


# =========================
# 2. Utilities
# =========================

def collect_midi_files(
    input_midi_dir: Union[str, Path],
    recursive: bool = False,
) -> List[Path]:
    input_midi_dir = Path(input_midi_dir)

    if not input_midi_dir.exists():
        raise FileNotFoundError(f"Input MIDI folder not found: {input_midi_dir}")

    if not input_midi_dir.is_dir():
        raise NotADirectoryError(f"Expected MIDI folder, got: {input_midi_dir}")

    if recursive:
        midi_files = list(input_midi_dir.rglob("*.mid")) + list(input_midi_dir.rglob("*.midi"))
    else:
        midi_files = list(input_midi_dir.glob("*.mid")) + list(input_midi_dir.glob("*.midi"))

    midi_files = sorted(midi_files)

    if len(midi_files) == 0:
        raise FileNotFoundError(f"No .mid or .midi files found in: {input_midi_dir}")

    return midi_files


def make_output_path(
    midi_path: Path,
    input_midi_dir: Path,
    output_midi_dir: Path,
    preserve_subfolders: bool = True,
) -> Path:
    if preserve_subfolders:
        rel_path = midi_path.relative_to(input_midi_dir)
        return output_midi_dir / rel_path

    return output_midi_dir / midi_path.name


def append_metadata(metadata_csv: Path, row: dict) -> None:
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    file_exists = metadata_csv.exists()

    fieldnames = [
        "time",
        "input_midi_path",
        "output_midi_path",
        "target_channel",
        "num_tracks",
        "num_channel_messages_changed",
        "num_note_messages",
        "status",
        "error_message",
    ]

    with open(metadata_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def make_metadata_row(
    input_midi_path: Path,
    output_midi_path: Path,
    target_channel: int,
    num_tracks: int = 0,
    num_channel_messages_changed: int = 0,
    num_note_messages: int = 0,
    status: str = "started",
    error_message: str = "",
) -> dict:
    return {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "input_midi_path": str(input_midi_path),
        "output_midi_path": str(output_midi_path),
        "target_channel": target_channel,
        "num_tracks": num_tracks,
        "num_channel_messages_changed": num_channel_messages_changed,
        "num_note_messages": num_note_messages,
        "status": status,
        "error_message": error_message,
    }


# =========================
# 3. Core cleaning
# =========================

def clean_one_midi_to_channel(
    input_midi_path: Union[str, Path],
    output_midi_path: Union[str, Path],
    target_channel: int = 1,
    overwrite: bool = False,
) -> dict:
    """
    复制原始 MIDI 的内容到新 MIDI，只修改所有带 channel 属性的 MIDI 消息。

    注意：
    - 不会修改原始 MIDI 文件；
    - target_channel 使用人类习惯的 1-16；
    - mido 内部 channel 使用 0-15，所以 Channel 1 对应 0；
    - 本脚本不删除 CC / Program Change / Pitch Bend，只把它们的 channel 也统一到 target_channel。
    """
    input_midi_path = Path(input_midi_path)
    output_midi_path = Path(output_midi_path)

    if not input_midi_path.exists():
        raise FileNotFoundError(f"Input MIDI file not found: {input_midi_path}")

    if not (1 <= int(target_channel) <= 16):
        raise ValueError(f"target_channel must be in 1..16, got: {target_channel}")

    target_channel_zero_based = int(target_channel) - 1

    if output_midi_path.exists() and not overwrite:
        return make_metadata_row(
            input_midi_path=input_midi_path,
            output_midi_path=output_midi_path,
            target_channel=int(target_channel),
            status="skipped_exists",
        )

    mid = mido.MidiFile(str(input_midi_path))

    # 深拷贝，确保不改原始 mid 对象，也不会改原始文件。
    new_mid = copy.deepcopy(mid)

    num_channel_messages_changed = 0
    num_note_messages = 0

    for track in new_mid.tracks:
        for msg in track:
            if msg.type in {"note_on", "note_off"}:
                num_note_messages += 1

            if hasattr(msg, "channel"):
                msg.channel = target_channel_zero_based
                num_channel_messages_changed += 1

    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    new_mid.save(str(output_midi_path))

    return make_metadata_row(
        input_midi_path=input_midi_path,
        output_midi_path=output_midi_path,
        target_channel=int(target_channel),
        num_tracks=len(new_mid.tracks),
        num_channel_messages_changed=num_channel_messages_changed,
        num_note_messages=num_note_messages,
        status="ok",
    )


def clean_midi_folder_to_channel(
    input_midi_dir: Union[str, Path],
    output_midi_dir: Union[str, Path],
    target_channel: int = 1,
    recursive: bool = False,
    preserve_subfolders: bool = True,
    overwrite: bool = False,
    write_metadata: bool = True,
    metadata_filename: str = "clean_midi_channel_metadata.csv",
) -> List[Path]:
    input_midi_dir = Path(input_midi_dir)
    output_midi_dir = Path(output_midi_dir)

    input_resolved = input_midi_dir.resolve()
    output_resolved = output_midi_dir.resolve()

    if input_resolved == output_resolved:
        raise ValueError(
            "Input folder and output folder must not be the same. "
            "This script is designed to avoid modifying original MIDI files."
        )

    midi_files = collect_midi_files(input_midi_dir, recursive=recursive)
    output_midi_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = output_midi_dir / metadata_filename

    print(f"[INFO] Input MIDI dir: {input_midi_dir}")
    print(f"[INFO] Output MIDI dir: {output_midi_dir}")
    print(f"[INFO] Found MIDI files: {len(midi_files)}")
    print(f"[INFO] Target MIDI channel: {target_channel}")
    print(f"[INFO] Recursive: {recursive}")
    print(f"[INFO] Preserve subfolders: {preserve_subfolders}")
    print(f"[INFO] Overwrite: {overwrite}")

    output_paths: List[Path] = []

    for i, midi_path in enumerate(midi_files, start=1):
        out_path = make_output_path(
            midi_path=midi_path,
            input_midi_dir=input_midi_dir,
            output_midi_dir=output_midi_dir,
            preserve_subfolders=preserve_subfolders,
        )

        print(f"\n[{i}/{len(midi_files)}] {midi_path.name}")

        try:
            row = clean_one_midi_to_channel(
                input_midi_path=midi_path,
                output_midi_path=out_path,
                target_channel=target_channel,
                overwrite=overwrite,
            )

            if row["status"] == "ok":
                print(f"[OK] Saved: {out_path}")
                print(
                    f"[INFO] changed_channel_messages={row['num_channel_messages_changed']}, "
                    f"note_messages={row['num_note_messages']}, "
                    f"tracks={row['num_tracks']}"
                )
                output_paths.append(out_path)
            elif row["status"] == "skipped_exists":
                print(f"[SKIP] Exists: {out_path}")
                output_paths.append(out_path)

        except Exception as e:
            row = make_metadata_row(
                input_midi_path=midi_path,
                output_midi_path=out_path,
                target_channel=target_channel,
                status="error",
                error_message=repr(e),
            )
            print(f"[ERROR] Failed: {midi_path}")
            print(repr(e))

        if write_metadata:
            append_metadata(metadata_csv, row)

    print(f"\n[DONE] Processed: {len(midi_files)}")
    print(f"[DONE] Output files recorded: {len(output_paths)}")
    print(f"[DONE] Output dir: {output_midi_dir}")

    return output_paths


# =========================
# 4. CLI
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy MIDI files to a new folder and force all MIDI channel messages to Channel 1 by default."
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML path. Default: project configs/clean_midi_channel.yaml",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    input_midi_dir = cfg["input_midi_dir"]
    output_midi_dir = cfg["output_midi_dir"]

    target_channel = int(cfg.get("target_channel", 1))
    recursive = bool(cfg.get("recursive", False))
    preserve_subfolders = bool(cfg.get("preserve_subfolders", True))
    overwrite = bool(cfg.get("overwrite", False))

    write_metadata = bool(cfg.get("write_metadata", True))
    metadata_filename = str(cfg.get("metadata_filename", "clean_midi_channel_metadata.csv"))

    clean_midi_folder_to_channel(
        input_midi_dir=input_midi_dir,
        output_midi_dir=output_midi_dir,
        target_channel=target_channel,
        recursive=recursive,
        preserve_subfolders=preserve_subfolders,
        overwrite=overwrite,
        write_metadata=write_metadata,
        metadata_filename=metadata_filename,
    )


if __name__ == "__main__":
    main()
