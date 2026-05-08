from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import soundfile as sf
import yaml


# =========================
# 1. Config
# =========================

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


# =========================
# 2. Utilities
# =========================

def _collect_midi_files(
    midi_dir: Union[str, Path],
    recursive: bool = False,
) -> List[Path]:
    midi_dir = Path(midi_dir)

    if not midi_dir.exists():
        raise FileNotFoundError(f"MIDI folder not found: {midi_dir}")

    if not midi_dir.is_dir():
        raise NotADirectoryError(f"Expected MIDI folder, got: {midi_dir}")

    if recursive:
        midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    else:
        midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))

    midi_files = sorted(midi_files)

    if len(midi_files) == 0:
        raise FileNotFoundError(f"No .mid or .midi files found in: {midi_dir}")

    return midi_files


def _get_midi_duration_sec(
    midi_path: Union[str, Path],
    default_render_sec: float,
) -> float:
    midi_path = Path(midi_path)

    try:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        duration = float(pm.get_end_time())

        if duration <= 0:
            return float(default_render_sec)

        return duration

    except Exception:
        return float(default_render_sec)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio

    if audio.ndim == 2:
        if audio.shape[1] == 1:
            return audio[:, 0]
        return audio.mean(axis=1)

    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def _sanitize_audio(audio: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _normalize_if_needed(
    audio: np.ndarray,
    normalize_peak: bool,
    target_peak: float,
) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0

    if peak <= 0:
        return audio

    if normalize_peak:
        return audio / peak * float(target_peak)

    if peak > 1.0:
        return audio / peak * 0.95

    return audio


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
        "output_channels",
        "wav_subtype",
        "peak_abs",
        "rms",
        "status",
        "error_message",
    ]

    with open(metadata_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def _make_metadata_row(
    midi_path: Path,
    out_wav_path: Path,
    vst_path: Path,
    sample_rate: int,
    buffer_size: int,
    render_sec,
    tail_sec: float,
    output_channels: int,
    wav_subtype: str,
    peak_abs="",
    rms="",
    status: str = "started",
    error_message: str = "",
) -> dict:
    return {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "midi_path": str(midi_path),
        "output_wav_path": str(out_wav_path),
        "vst_path": str(vst_path),
        "sample_rate": sample_rate,
        "buffer_size": buffer_size,
        "render_sec": render_sec,
        "tail_sec": tail_sec,
        "output_channels": output_channels,
        "wav_subtype": wav_subtype,
        "peak_abs": peak_abs,
        "rms": rms,
        "status": status,
        "error_message": error_message,
    }


# =========================
# 3. Core render
# =========================

def _render_one_loaded_midi(
    engine,
    synth,
    midi_path: Path,
    out_wav_path: Path,
    cfg: dict,
    vst_path: Path,
) -> dict:
    sample_rate = int(cfg.get("sample_rate", 44100))
    buffer_size = int(cfg.get("buffer_size", 512))

    tail_sec = float(cfg.get("tail_sec", 3.0))
    default_render_sec = float(cfg.get("default_render_sec", 120.0))

    beats = bool(cfg.get("beats", False))
    all_events = bool(cfg.get("all_events", True))

    normalize_peak = bool(cfg.get("normalize_peak", False))
    target_peak = float(cfg.get("target_peak", 0.98))

    wav_subtype = str(cfg.get("wav_subtype", "PCM_16"))

    render_sec = _get_midi_duration_sec(
        midi_path=midi_path,
        default_render_sec=default_render_sec,
    )
    render_sec = float(render_sec + tail_sec)

    row = _make_metadata_row(
        midi_path=midi_path,
        out_wav_path=out_wav_path,
        vst_path=vst_path,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        render_sec=render_sec,
        tail_sec=tail_sec,
        output_channels=1,
        wav_subtype=wav_subtype,
        status="started",
    )

    try:
        synth.load_midi(
            str(midi_path),
            clear_previous=True,
            beats=beats,
            all_events=all_events,
        )

        engine.render(render_sec)

        audio = engine.get_audio()

        # DAWdreamer returns [channels, samples].
        # Convert to [samples, channels].
        audio = audio.T.astype(np.float32)

        # Force mono.
        audio = _to_mono(audio)

        # Clean numerical issues.
        audio = _sanitize_audio(audio)

        # Normalize only if configured, otherwise protect PCM_16.
        audio = _normalize_if_needed(
            audio=audio,
            normalize_peak=normalize_peak,
            target_peak=target_peak,
        )

        peak_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size > 0 else 0.0

        out_wav_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(
            str(out_wav_path),
            audio,
            sample_rate,
            subtype=wav_subtype,
        )

        row["peak_abs"] = peak_abs
        row["rms"] = rms
        row["status"] = "ok"

        if peak_abs < 1e-8:
            print(f"[WARNING] Silent render: {out_wav_path}")
        else:
            print(f"[OK] Rendered: {out_wav_path}")
            print(f"[INFO] peak_abs={peak_abs:.6f}, rms={rms:.6f}, channels=1")

    except Exception as e:
        row["status"] = "error"
        row["error_message"] = repr(e)
        print(f"[ERROR] Failed: {midi_path}")
        print(repr(e))

    return row


# =========================
# 4. Public functions
# =========================

def render_midi_folder_to_wavs(
    midi_dir: Union[str, Path],
    save_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    recursive: bool = False,
    open_editor_first: bool = False,
    overwrite: Optional[bool] = None,
) -> List[Path]:
    """
    连续渲染版本：

    1. 只创建一次 RenderEngine
    2. 只创建一次 SINE Player
    3. 只打开一次插件窗口
    4. 循环 load_midi + render
    5. 输出 mono / PCM_16 WAV
    """
    import dawdreamer as daw

    cfg = load_config(config_path)

    midi_dir = Path(midi_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    midi_files = _collect_midi_files(
        midi_dir=midi_dir,
        recursive=recursive,
    )

    vst_path = Path(cfg["vst_path"])

    sample_rate = int(cfg.get("sample_rate", 44100))
    buffer_size = int(cfg.get("buffer_size", 512))

    tail_sec = float(cfg.get("tail_sec", 3.0))
    wav_subtype = str(cfg.get("wav_subtype", "PCM_16"))

    write_metadata = bool(cfg.get("write_metadata", True))
    metadata_filename = str(cfg.get("metadata_filename", "render_metadata.csv"))

    if overwrite is None:
        overwrite = bool(cfg.get("overwrite", False))

    if not vst_path.exists():
        raise FileNotFoundError(f"VST plugin not found: {vst_path}")

    metadata_csv = save_dir / metadata_filename

    print(f"[INFO] Found MIDI files: {len(midi_files)}")
    print(f"[INFO] MIDI dir: {midi_dir}")
    print(f"[INFO] Save dir: {save_dir}")
    print(f"[INFO] VST path: {vst_path}")
    print(f"[INFO] sample_rate = {sample_rate}")
    print(f"[INFO] buffer_size = {buffer_size}")
    print(f"[INFO] output channels = 1")
    print(f"[INFO] wav subtype = {wav_subtype}")
    print("[INFO] Continuous rendering mode.")
    print("[INFO] Creating RenderEngine once.")
    print("[INFO] Creating SINE Player once.")

    engine = daw.RenderEngine(sample_rate, buffer_size)
    synth = engine.make_plugin_processor("sine_player", str(vst_path))

    engine.load_graph([
        (synth, []),
    ])

    if open_editor_first:
        print("[INFO] Opening SINE Player editor once.")
        print("[INFO] Load Crucible -> 01 Organ, then close the plugin window.")
        synth.open_editor()

    out_wavs: List[Path] = []

    for i, midi_path in enumerate(midi_files, start=1):
        out_wav_path = save_dir / f"{midi_path.stem}__sine_player.wav"

        print(f"\n[{i}/{len(midi_files)}] {midi_path.name}")

        if out_wav_path.exists() and not overwrite:
            print(f"[SKIP] WAV already exists: {out_wav_path}")

            row = _make_metadata_row(
                midi_path=midi_path,
                out_wav_path=out_wav_path,
                vst_path=vst_path,
                sample_rate=sample_rate,
                buffer_size=buffer_size,
                render_sec="",
                tail_sec=tail_sec,
                output_channels=1,
                wav_subtype=wav_subtype,
                status="skipped_exists",
            )

            if write_metadata:
                _append_metadata(metadata_csv, row)

            out_wavs.append(out_wav_path)
            continue

        row = _render_one_loaded_midi(
            engine=engine,
            synth=synth,
            midi_path=midi_path,
            out_wav_path=out_wav_path,
            cfg=cfg,
            vst_path=vst_path,
        )

        if write_metadata:
            _append_metadata(metadata_csv, row)

        if row["status"] == "ok":
            out_wavs.append(out_wav_path)

    print(f"\n[DONE] Processed MIDI files: {len(midi_files)}")
    print(f"[DONE] WAV files recorded: {len(out_wavs)}")
    print(f"[DONE] Save dir: {save_dir}")

    return out_wavs


def render_one_midi_to_wav(
    midi_path: Union[str, Path],
    save_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    open_editor: bool = False,
    overwrite: Optional[bool] = None,
) -> Path:
    """
    单文件渲染。
    """
    import dawdreamer as daw

    cfg = load_config(config_path)

    midi_path = Path(midi_path)
    save_path = Path(save_path)

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    if not midi_path.is_file():
        raise ValueError(f"Expected MIDI file, got: {midi_path}")

    if save_path.suffix.lower() == ".wav":
        out_wav_path = save_path
    else:
        save_path.mkdir(parents=True, exist_ok=True)
        out_wav_path = save_path / f"{midi_path.stem}__sine_player.wav"

    vst_path = Path(cfg["vst_path"])

    sample_rate = int(cfg.get("sample_rate", 44100))
    buffer_size = int(cfg.get("buffer_size", 512))

    write_metadata = bool(cfg.get("write_metadata", True))
    metadata_filename = str(cfg.get("metadata_filename", "render_metadata.csv"))

    if overwrite is None:
        overwrite = bool(cfg.get("overwrite", False))

    if not vst_path.exists():
        raise FileNotFoundError(f"VST plugin not found: {vst_path}")

    if out_wav_path.exists() and not overwrite:
        print(f"[SKIP] WAV already exists: {out_wav_path}")
        return out_wav_path

    metadata_csv = out_wav_path.parent / metadata_filename

    print(f"[INFO] MIDI: {midi_path}")
    print(f"[INFO] Output: {out_wav_path}")
    print(f"[INFO] VST path: {vst_path}")
    print("[INFO] Creating RenderEngine once.")
    print("[INFO] Creating SINE Player once.")

    engine = daw.RenderEngine(sample_rate, buffer_size)
    synth = engine.make_plugin_processor("sine_player", str(vst_path))

    engine.load_graph([
        (synth, []),
    ])

    if open_editor:
        print("[INFO] Opening SINE Player editor.")
        print("[INFO] Load Crucible -> 01 Organ, then close the plugin window.")
        synth.open_editor()

    row = _render_one_loaded_midi(
        engine=engine,
        synth=synth,
        midi_path=midi_path,
        out_wav_path=out_wav_path,
        cfg=cfg,
        vst_path=vst_path,
    )

    if write_metadata:
        _append_metadata(metadata_csv, row)

    return out_wav_path


def render_midi_to_wav(
    midi_path: Union[str, Path],
    save_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    recursive: bool = False,
    open_editor_first: bool = False,
    overwrite: Optional[bool] = None,
):
    midi_path = Path(midi_path)
    save_path = Path(save_path)

    if midi_path.is_dir():
        if save_path.suffix.lower() == ".wav":
            raise ValueError(
                "When midi_path is a folder, save_path must be an output folder, not a .wav file."
            )

        return render_midi_folder_to_wavs(
            midi_dir=midi_path,
            save_dir=save_path,
            config_path=config_path,
            recursive=recursive,
            open_editor_first=open_editor_first,
            overwrite=overwrite,
        )

    if midi_path.is_file():
        return render_one_midi_to_wav(
            midi_path=midi_path,
            save_path=save_path,
            config_path=config_path,
            open_editor=open_editor_first,
            overwrite=overwrite,
        )

    raise FileNotFoundError(f"MIDI path not found: {midi_path}")


# =========================
# 5. CLI
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render MIDI file or folder to mono PCM_16 WAV using SINE Player VST via DAWdreamer."
    )

    parser.add_argument(
        "--midi",
        required=True,
        help="Input MIDI file path or MIDI folder path.",
    )

    parser.add_argument(
        "--save",
        required=True,
        help="Output WAV path or output folder.",
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
        help="Open SINE Player GUI once before batch rendering.",
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