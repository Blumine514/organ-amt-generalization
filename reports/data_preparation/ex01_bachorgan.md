# Data Preparation: ex01_bachorgan

## 1. 目的

本流程用于将 Bach 管风琴 MIDI 渲染为 WAV 音频，构造第一个管风琴目标域数据集。

该数据集用于后续实验：

- Experiment 01: Piano AMT Baseline Zero-shot Transfer to Organ

## 2. 数据集位置

Dataset root:

```text
data/raw/organ/ex01_bachorgan/
```

目录结构：

```text
data/raw/organ/ex01_bachorgan/
├── midi/
├── audio/
└── metadata.csv
```

## 3. 输入数据

### 3.1 MIDI

MIDI 文件位置：

```text
data/raw/organ/ex01_bachorgan/midi/
```

MIDI 来源：

```text
填写下载网站，例如 Suzumidi / jsbach.net
```

文件数量：

```text
15
```

### 3.2 SoundFont

使用的 SoundFont：

```text
data/raw/soundfonts/Pipe Organ Samples v1.1.sf2
```

SoundFont 名称：

```text
Pipe Organ Samples v1.1
```

## 4. 合成工具

使用工具：

```text
FluidSynth
```

可执行文件路径：

```text
D:/fluidsynth-v2.5.4-win10-x64-glib/bin/fluidsynth.exe
```

Python 脚本：

```text
scripts/organ_bach_synth.py
```

配置文件：

```text
configs/data/pre_organ_synth.yaml
```

## 5. 合成参数

```yaml
sample_rate: 44100
output_format: wav
overwrite: true
force_program: null
```

说明：

```text
force_program 设置为 null，因为 Pipe Organ Samples v1.1 不是标准 GM SoundFont，不强制改 MIDI program。
```

## 6. 输出文件命名规则

每个 MIDI 文件生成同名 WAV 文件：

```text
midi/faf_gm11.mid
audio/faf_gm11.wav
```

即：

```text
MIDI stem 相同，只改变后缀。
```

## 7. metadata.csv

metadata 文件位置：

```text
data/raw/organ/ex01_bachorgan/metadata.csv
```

metadata.csv 用于记录每个样本的 MIDI 文件、合成 WAV 文件、数据集名称、domain、SoundFont、采样率和时长等信息。

## 8. 已知 warning

运行时可能出现：

```text
No preset found on channel 9 [bank=128 prog=0]
Instrument not found on channel ...
```

含义：

```text
某些 MIDI channel 请求的 preset 在当前 SoundFont 中不存在，FluidSynth 会自动替换。
```

当前处理方式：

```text
先保留 warning，确认 WAV 能正常生成和播放。
如果后续发现明显缺声部，再单独处理 MIDI channel 和 program。
```