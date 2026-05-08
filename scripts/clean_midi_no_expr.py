from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

import mido
import yaml


# =========================
# 1. Config
# =========================

def _default_config_path() -> Path:
    """
    默认假设本脚本放在：
        D:/organ-amt-generalization/scripts/clean_midi_no_expr.py

    配置文件放在：
        D:/organ-amt-generalization/configs/clean_midi_no_expr.yaml
    """
    return Path(__file__).resolve().parents[1] / "configs" / "clean_midi_no_expr.yaml"


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
        "num_note_messages_kept",
        "num_meta_messages_kept",
        "num_messages_removed",
        "removed_event_counts_json",
        "overlap_note_on_count",
        "orphan_note_off_count",
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
    num_note_messages_kept: int = 0,
    num_meta_messages_kept: int = 0,
    num_messages_removed: int = 0,
    removed_event_counts: Optional[Dict[str, int]] = None,
    overlap_note_on_count: int = 0,
    orphan_note_off_count: int = 0,
    status: str = "started",
    error_message: str = "",
) -> dict:
    return {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "input_midi_path": str(input_midi_path),
        "output_midi_path": str(output_midi_path),
        "target_channel": target_channel,
        "num_tracks": num_tracks,
        "num_note_messages_kept": num_note_messages_kept,
        "num_meta_messages_kept": num_meta_messages_kept,
        "num_messages_removed": num_messages_removed,
        "removed_event_counts_json": json.dumps(removed_event_counts or {}, ensure_ascii=False),
        "overlap_note_on_count": overlap_note_on_count,
        "orphan_note_off_count": orphan_note_off_count,
        "status": status,
        "error_message": error_message,
    }


def count_overlap_after_merge(midi_path: Union[str, Path]) -> tuple[int, int]:
    """
    统计清理后同一 channel + note 上的潜在重叠 note_on，以及孤立 note_off。

    这个函数不修改 MIDI，只用于诊断：
    - overlap_note_on_count > 0 表示同一音高在前一个 note_off 前又 note_on；
    - orphan_note_off_count > 0 表示遇到了没有对应活动 note_on 的 note_off。

    注意：这不是严格的音乐错误判断，只是用来提示 channel 合并后可能存在 note_off 冲突。
    """
    midi_path = Path(midi_path)
    mid = mido.MidiFile(str(midi_path))
    merged = mido.merge_tracks(mid.tracks)

    active = Counter()
    overlap_note_on_count = 0
    orphan_note_off_count = 0

    for msg in merged:
        if msg.is_meta:
            continue

        if msg.type == "note_on":
            key = (getattr(msg, "channel", 0), msg.note)

            if msg.velocity == 0:
                if active[key] <= 0:
                    orphan_note_off_count += 1
                else:
                    active[key] -= 1
                continue

            if active[key] > 0:
                overlap_note_on_count += 1

            active[key] += 1

        elif msg.type == "note_off":
            key = (getattr(msg, "channel", 0), msg.note)

            if active[key] <= 0:
                orphan_note_off_count += 1
            else:
                active[key] -= 1

    return overlap_note_on_count, orphan_note_off_count


# =========================
# 3. Core cleaning
# =========================

def clean_one_midi_no_expression(
    input_midi_path: Union[str, Path],
    output_midi_path: Union[str, Path],
    target_channel: int = 1,
    overwrite: bool = False,
    keep_meta_events: bool = True,
    remove_sysex: bool = True,
    remove_unknown_channel_events: bool = True,
) -> dict:
    """
    复制 MIDI 到新文件，并做清理：

    保留：
    - note_on
    - note_off
    - meta events，默认保留，包括 tempo / time_signature / key_signature / track_name / end_of_track

    修改：
    - 所有 note_on / note_off 的 channel 改成 target_channel

    删除：
    - control_change
    - program_change
    - pitchwheel
    - aftertouch
    - polytouch
    - sysex，默认删除
    - 其他非 note 的 channel 事件，默认删除

    重要：
    删除事件时，本脚本会把被删除事件的 delta time 累加到下一个保留事件上。
    因此删除 CC / Program Change 不会压缩 MIDI 时间轴。
    """
    input_midi_path = Path(input_midi_path)
    output_midi_path = Path(output_midi_path)

    if not input_midi_path.exists():
        raise FileNotFoundError(f"Input MIDI file not found: {input_midi_path}")

    if not (1 <= int(target_channel) <= 16):
        raise ValueError(f"target_channel must be in 1..16, got: {target_channel}")

    if output_midi_path.exists() and not overwrite:
        return make_metadata_row(
            input_midi_path=input_midi_path,
            output_midi_path=output_midi_path,
            target_channel=int(target_channel),
            status="skipped_exists",
        )

    target_channel_zero_based = int(target_channel) - 1

    mid = mido.MidiFile(str(input_midi_path))
    new_mid = mido.MidiFile(
        type=mid.type,
        ticks_per_beat=mid.ticks_per_beat,
        charset=getattr(mid, "charset", "latin1"),
        debug=False,
        clip=False,
    )

    remove_types = {
        "control_change",
        "program_change",
        "pitchwheel",
        "aftertouch",
        "polytouch",
    }

    removed_event_counts = Counter()
    num_note_messages_kept = 0
    num_meta_messages_kept = 0
    num_messages_removed = 0

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)

        pending_time = 0
        kept_end_of_track = False

        for msg in track:
            # 如果前面删除过事件，pending_time 用来保留时间轴。
            new_time = msg.time + pending_time

            if msg.is_meta:
                if keep_meta_events:
                    new_track.append(msg.copy(time=new_time))
                    num_meta_messages_kept += 1
                    pending_time = 0

                    if msg.type == "end_of_track":
                        kept_end_of_track = True
                else:
                    pending_time = new_time
                    removed_event_counts[msg.type] += 1
                    num_messages_removed += 1

                continue

            if msg.type in {"note_on", "note_off"}:
                new_track.append(
                    msg.copy(
                        time=new_time,
                        channel=target_channel_zero_based,
                    )
                )
                num_note_messages_kept += 1
                pending_time = 0
                continue

            should_remove = False

            if msg.type in remove_types:
                should_remove = True

            if msg.type == "sysex" and remove_sysex:
                should_remove = True

            if hasattr(msg, "channel") and remove_unknown_channel_events:
                should_remove = True

            if should_remove:
                pending_time = new_time
                removed_event_counts[msg.type] += 1
                num_messages_removed += 1
                continue

            # 理论上很少走到这里。保留未知非 channel 事件，并保留时间。
            new_track.append(msg.copy(time=new_time))
            pending_time = 0

        # 如果原轨没有 end_of_track，补一个。若有 pending_time，也保留下来。
        if not kept_end_of_track:
            new_track.append(mido.MetaMessage("end_of_track", time=pending_time))

    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    new_mid.save(str(output_midi_path))

    overlap_count, orphan_count = count_overlap_after_merge(output_midi_path)

    return make_metadata_row(
        input_midi_path=input_midi_path,
        output_midi_path=output_midi_path,
        target_channel=int(target_channel),
        num_tracks=len(new_mid.tracks),
        num_note_messages_kept=num_note_messages_kept,
        num_meta_messages_kept=num_meta_messages_kept,
        num_messages_removed=num_messages_removed,
        removed_event_counts=dict(removed_event_counts),
        overlap_note_on_count=overlap_count,
        orphan_note_off_count=orphan_count,
        status="ok",
    )


def clean_midi_folder_no_expression(
    input_midi_dir: Union[str, Path],
    output_midi_dir: Union[str, Path],
    target_channel: int = 1,
    recursive: bool = False,
    preserve_subfolders: bool = True,
    overwrite: bool = False,
    keep_meta_events: bool = True,
    remove_sysex: bool = True,
    remove_unknown_channel_events: bool = True,
    write_metadata: bool = True,
    metadata_filename: str = "clean_midi_no_expr_metadata.csv",
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
    print("[INFO] Cleaning rule: keep notes + meta, remove CC / program_change / pitchwheel / aftertouch / sysex.")

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
            row = clean_one_midi_no_expression(
                input_midi_path=midi_path,
                output_midi_path=out_path,
                target_channel=target_channel,
                overwrite=overwrite,
                keep_meta_events=keep_meta_events,
                remove_sysex=remove_sysex,
                remove_unknown_channel_events=remove_unknown_channel_events,
            )

            if row["status"] == "ok":
                print(f"[OK] Saved: {out_path}")
                print(
                    f"[INFO] notes={row['num_note_messages_kept']}, "
                    f"removed={row['num_messages_removed']}, "
                    f"overlap_note_on={row['overlap_note_on_count']}, "
                    f"orphan_note_off={row['orphan_note_off_count']}"
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
        description=(
            "Copy MIDI files to a new folder, force note_on/note_off to Channel 1, "
            "and remove expression/control events."
        )
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML path. Default: project configs/clean_midi_no_expr.yaml",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    input_midi_dir = cfg["input_midi_dir"]
    output_midi_dir = cfg["output_midi_dir"]

    target_channel = int(cfg.get("target_channel", 1))
    recursive = bool(cfg.get("recursive", False))
    preserve_subfolders = bool(cfg.get("preserve_subfolders", True))
    overwrite = bool(cfg.get("overwrite", False))

    keep_meta_events = bool(cfg.get("keep_meta_events", True))
    remove_sysex = bool(cfg.get("remove_sysex", True))
    remove_unknown_channel_events = bool(cfg.get("remove_unknown_channel_events", True))

    write_metadata = bool(cfg.get("write_metadata", True))
    metadata_filename = str(cfg.get("metadata_filename", "clean_midi_no_expr_metadata.csv"))

    clean_midi_folder_no_expression(
        input_midi_dir=input_midi_dir,
        output_midi_dir=output_midi_dir,
        target_channel=target_channel,
        recursive=recursive,
        preserve_subfolders=preserve_subfolders,
        overwrite=overwrite,
        keep_meta_events=keep_meta_events,
        remove_sysex=remove_sysex,
        remove_unknown_channel_events=remove_unknown_channel_events,
        write_metadata=write_metadata,
        metadata_filename=metadata_filename,
    )


if __name__ == "__main__":
    main()
