from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import soundfile as sf
import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "render_sine.yaml"


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    if config_path is None:
        config_path = _default_config_path()

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg or {}


def _get_midi_duration_sec(midi_path: Path, default_render_sec: float) -> float:
    try:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        duration = float(pm.get_end_time())

        if duration <= 0:
            return float(default_render_sec)

        return duration

    except Exception:
        return float(default_render_sec)


def _normalize_audio(audio: np.ndarray, target_peak: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0

    if peak <= 0:
        return audio

    return audio / peak * float(target_peak)


def _append_metadata(metadata_csv: Path, row: dict) -> None:
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    file_exists = metadata_csv.exists()

    fieldnames = [
        "time",
        "midi_path",
        "output_wav_path",
        "vst_path",
        "sample_rate",
        "buffer_size",
        "render_sec",
        "tail_sec",
        "peak_abs",
        "status",
        "error_message",
    ]

    with open(metadata_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def _collect_midi_files(midi_dir: Union[str, Path], recursive: bool = False) -> List[Path]:
    midi_dir = Path(midi_dir)

    if not midi_dir.exists():
        raise FileNotFoundError(f"MIDI folder not found: {midi_dir}")

    if not midi_dir.is_dir():
        raise NotADirectoryError(f"Expected a MIDI folder, got: {midi_dir}")

    if recursive:
        midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    else:
        midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))

    midi_files = sorted(midi_files)

    if len(midi_files) == 0:
        raise FileNotFoundError(f"No .mid or .midi files found in: {midi_dir}")

    return midi_files


def render_one_midi_to_wav(
    midi_path: Union[str, Path],
    out_wav_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    open_editor: Optional[bool] = None,
    overwrite: Optional[bool] = None,
) -> Path:
    import dawdreamer as daw

    cfg = load_config(config_path)

    midi_path = Path(midi_path)
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    vst_path = Path(cfg["vst_path"])

    sample_rate = int(cfg.get("sample_rate", 44100))
    buffer_size = int(cfg.get("buffer_size", 512))

    tail_sec = float(cfg.get("tail_sec", 3.0))
    default_render_sec = float(cfg.get("default_render_sec", 120.0))

    beats = bool(cfg.get("beats", False))
    all_events = bool(cfg.get("all_events", True))

    normalize_peak = bool(cfg.get("normalize_peak", False))
    target_peak = float(cfg.get("target_peak", 0.98))

    write_metadata = bool(cfg.get("write_metadata", True))
    metadata_filename = str(cfg.get("metadata_filename", "render_metadata.csv"))

    if open_editor is None:
        open_editor = bool(cfg.get("open_editor", False))

    if overwrite is None:
        overwrite = bool(cfg.get("overwrite", False))

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    if not vst_path.exists():
        raise FileNotFoundError(f"VST plugin not found: {vst_path}")

    if out_wav_path.exists() and not overwrite:
        print(f"[SKIP] WAV already exists: {out_wav_path}")
        return out_wav_path

    render_sec = _get_midi_duration_sec(
        midi_path=midi_path,
        default_render_sec=default_render_sec,
    )
    render_sec = render_sec + tail_sec

    metadata_csv = out_wav_path.parent / metadata_filename

    row = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "midi_path": str(midi_path),
        "output_wav_path": str(out_wav_path),
        "vst_path": str(vst_path),
        "sample_rate": sample_rate,
        "buffer_size": buffer_size,
        "render_sec": render_sec,
        "tail_sec": tail_sec,
        "peak_abs": "",
        "status": "started",
        "error_message": "",
    }

    try:
        engine = daw.RenderEngine(sample_rate, buffer_size)
        synth = engine.make_plugin_processor("sine_player", str(vst_path))

        if open_editor:
            print("[INFO] Opening SINE Player editor.")
            print("[INFO] Load Crucible / organ preset, then close the plugin window.")
            synth.open_editor()

        synth.load_midi(
            str(midi_path),
            clear_previous=True,
            beats=beats,
            all_events=all_events,
        )

        engine.load_graph([
            (synth, []),
        ])

        engine.render(render_sec)

        audio = engine.get_audio()
        audio = audio.T.astype(np.float32)

        peak_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0

        if normalize_peak:
            audio = _normalize_audio(audio, target_peak=target_peak)
            peak_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0

        sf.write(str(out_wav_path), audio, sample_rate)

        row["peak_abs"] = peak_abs
        row["status"] = "ok"

        if peak_abs == 0.0:
            print(f"[WARNING] Silent render: {out_wav_path}")

        print(f"[OK] Rendered: {out_wav_path}")

        return out_wav_path

    except Exception as e:
        row["status"] = "error"
        row["error_message"] = repr(e)
        print(f"[ERROR] Failed: {midi_path}")
        print(repr(e))
        raise

    finally:
        if write_metadata:
            _append_metadata(metadata_csv, row)


def render_midi_folder_to_wavs(
    midi_dir: Union[str, Path],
    save_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    recursive: bool = False,
    open_editor_first: bool = False,
    overwrite: Optional[bool] = None,
) -> List[Path]:
    """
    Batch render all MIDI files in a folder.

    Parameters
    ----------
    midi_dir:
        Folder containing .mid / .midi files.

    save_dir:
        Folder where rendered .wav files will be saved.

    config_path:
        YAML config path.

    recursive:
        If True, search MIDI files recursively.

    open_editor_first:
        If True, open SINE Player GUI only for the first MIDI.
        Use this only when you need to load/check the preset.

    overwrite:
        Whether to overwrite existing WAV files.

    Returns
    -------
    List[Path]
        Rendered WAV paths.
    """
    midi_dir = Path(midi_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    midi_files = _collect_midi_files(midi_dir, recursive=recursive)

    print(f"[INFO] Found MIDI files: {len(midi_files)}")
    print(f"[INFO] Save dir: {save_dir}")

    out_wavs = []

    for i, midi_path in enumerate(midi_files, start=1):
        out_wav_path = save_dir / f"{midi_path.stem}__sine_player.wav"

        print(f"\n[{i}/{len(midi_files)}] {midi_path.name}")

        out_wav = render_one_midi_to_wav(
            midi_path=midi_path,
            out_wav_path=out_wav_path,
            config_path=config_path,
            open_editor=(open_editor_first and i == 1),
            overwrite=overwrite,
        )

        out_wavs.append(out_wav)

    print(f"\n[DONE] Rendered files: {len(out_wavs)}")

    return out_wavs


def render_midi_to_wav(
    midi_path: Union[str, Path],
    save_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    recursive: bool = False,
    open_editor_first: bool = False,
    overwrite: Optional[bool] = None,
):
    """
    Unified entry point.

    If midi_path is a file:
        save_path can be an exact .wav path or a folder.

    If midi_path is a folder:
        save_path must be an output folder.
    """
    midi_path = Path(midi_path)
    save_path = Path(save_path)

    if midi_path.is_dir():
        return render_midi_folder_to_wavs(
            midi_dir=midi_path,
            save_dir=save_path,
            config_path=config_path,
            recursive=recursive,
            open_editor_first=open_editor_first,
            overwrite=overwrite,
        )

    if midi_path.is_file():
        if save_path.suffix.lower() == ".wav":
            out_wav_path = save_path
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            out_wav_path = save_path / f"{midi_path.stem}__sine_player.wav"

        return render_one_midi_to_wav(
            midi_path=midi_path,
            out_wav_path=out_wav_path,
            config_path=config_path,
            open_editor=open_editor_first,
            overwrite=overwrite,
        )

    raise FileNotFoundError(f"MIDI path not found: {midi_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render MIDI file or MIDI folder to WAV using SINE Player VST via DAWdreamer."
    )

    parser.add_argument(
        "--midi",
        required=True,
        help="Input MIDI file path or MIDI folder path.",
    )

    parser.add_argument(
        "--save",
        required=True,
        help="Output WAV path or output folder. If --midi is a folder, this must be a folder.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML path. Default: project configs/render_sine.yaml",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search MIDI files if --midi is a folder.",
    )

    parser.add_argument(
        "--open-editor-first",
        action="store_true",
        help="Open SINE Player GUI before rendering the first MIDI file.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output WAV if it already exists.",
    )

    args = parser.parse_args()

    render_midi_to_wav(
        midi_path=args.midi,
        save_path=args.save,
        config_path=args.config,
        recursive=args.recursive,
        open_editor_first=args.open_editor_first,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()