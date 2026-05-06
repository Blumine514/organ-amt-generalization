from pathlib import Path
import argparse
import subprocess
import yaml
import pandas as pd
import soundfile as sf


def resolve_path(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/pre_organ_synth.yaml",
        help="Path to synthesis config yaml."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(project_root, args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]["name"]
    domain = cfg["dataset"]["domain"]

    dataset_root = resolve_path(project_root, cfg["paths"]["dataset_root"])
    midi_dir = resolve_path(project_root, cfg["paths"]["midi_dir"])
    audio_dir = resolve_path(project_root, cfg["paths"]["audio_dir"])
    metadata_csv = resolve_path(project_root, cfg["paths"]["metadata_csv"])
    soundfont = resolve_path(project_root, cfg["paths"]["soundfont"])

    fluidsynth_exe = cfg["synthesis"].get("fluidsynth_exe", "fluidsynth")
    sample_rate = int(cfg["synthesis"].get("sample_rate", 44100))
    overwrite = bool(cfg["synthesis"].get("overwrite", True))

    audio_dir.mkdir(parents=True, exist_ok=True)

    if not midi_dir.exists():
        raise FileNotFoundError(f"MIDI directory not found: {midi_dir}")

    if not soundfont.exists():
        raise FileNotFoundError(f"SoundFont not found: {soundfont}")

    midi_files = sorted(
        list(midi_dir.glob("*.mid")) +
        list(midi_dir.glob("*.midi"))
    )

    if len(midi_files) == 0:
        raise FileNotFoundError(f"No .mid or .midi files found in: {midi_dir}")

    rows = []

    for i, midi_path in enumerate(midi_files, start=1):
        stem = midi_path.stem
        wav_path = audio_dir / f"{stem}.wav"

        print(f"[{i}/{len(midi_files)}] {midi_path.name} -> {wav_path.name}")

        if wav_path.exists() and not overwrite:
            print(f"  Skip existing file: {wav_path.name}")
        else:
            cmd = [
                fluidsynth_exe,
                "-ni",
                "-a",
                "file",
                "-F",
                str(wav_path),
                "-T",
                "wav",
                "-r",
                str(sample_rate),
                str(soundfont),
                str(midi_path),
                    ]

            subprocess.run(cmd, check=True)

        try:
            audio_info = sf.info(str(wav_path))
            duration = float(audio_info.duration)
            actual_sample_rate = int(audio_info.samplerate)
            channels = int(audio_info.channels)
        except Exception:
            duration = None
            actual_sample_rate = sample_rate
            channels = None

        rows.append({
            "id": stem,
            "audio_path": str(wav_path.relative_to(project_root)).replace("\\", "/"),
            "midi_path": str(midi_path.relative_to(project_root)).replace("\\", "/"),
            "dataset": dataset_name,
            "domain": domain,
            "soundfont": str(soundfont.relative_to(project_root)).replace("\\", "/"),
            "synthesizer": "fluidsynth",
            "sample_rate": actual_sample_rate,
            "channels": channels,
            "duration": duration,
        })

    metadata = pd.DataFrame(rows)
    metadata.to_csv(metadata_csv, index=False, encoding="utf-8-sig")

    print()
    print("Done.")
    print(f"Dataset root: {dataset_root}")
    print(f"Audio dir:    {audio_dir}")
    print(f"Metadata:     {metadata_csv}")
    print(f"Samples:      {len(metadata)}")


if __name__ == "__main__":
    main()