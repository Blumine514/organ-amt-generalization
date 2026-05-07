# Organ AMT Generalization

研究目标：测试钢琴自动转录模型在管风琴音色上的跨域泛化能力。

# Organ AMT Generalization

## 1. Project Overview

本项目研究自动音乐转录模型在不同乐器音色之间的跨域泛化能力。

当前重点问题是：

> 使用钢琴数据训练的自动音乐转录模型，能否泛化到管风琴音色？  
> 如果性能下降，下降主要来自 onset 检测、frame 检测、音高识别，还是长持续音建模？

自动音乐转录，Automatic Music Transcription, AMT，是将音频信号转换为符号音乐表示的任务，例如 MIDI、note events 或 piano roll。

本项目目前以钢琴到管风琴的跨音色泛化为主要研究场景。

---

## 2. Research Question

核心研究问题：

1. 钢琴 AMT 模型在钢琴测试集上表现如何？
2. 同一个模型直接测试到管风琴音频上时，性能下降多少？
3. 性能下降主要体现在哪些错误类型？
4. 数据增强、少量目标域微调、合成管风琴音色是否能够改善跨域泛化？
5. 模型学到的是音高结构，还是对钢琴音色存在过拟合？

---
## 3. 任务定义 Task Definition

本项目属于自动音乐转录，Automatic Music Transcription, AMT，任务目标是：

> 输入一段音乐音频，输出对应的符号音乐表示，例如 piano roll、note events 或 MIDI。

### 输入 Input

模型可以使用以下几类输入表示：

- Audio waveform：原始音频波形；
- Spectrogram：频谱图；
- Log-mel spectrogram：对数梅尔频谱图；
- CQT representation：Constant-Q Transform 表示。

其中，原始音频不能直接反映音高结构，因此通常会先转换为二维时频表示，再输入神经网络。

### 输出 Output

模型的输出可以包括：

- Frame-level piano roll：每一帧上哪些音高处于激活状态；
- Onset prediction：预测每个音符开始的位置；
- Offset prediction：预测每个音符结束的位置；
- Note events：完整音符事件，通常包含 pitch、onset time、offset time；
- MIDI file：最终可播放、可分析的符号音乐文件。

### 源域与目标域 Domains

本项目关注跨乐器音色泛化，因此需要区分 source domain 和 target domain。

| Domain | Description |
|---|---|
| Source domain | 钢琴音频 Piano audio |
| Target domain | 管风琴音频 Organ audio |

在基础实验中，模型先在钢琴数据上训练，然后直接测试到管风琴数据上。

核心观察点是：

> 模型在 source domain 上表现正常时，迁移到 target domain 后性能会下降多少，以及下降主要发生在哪些预测目标上。

---

## 4. 项目流程 Project Workflow

本项目的基本流程如下：

```text
原始音频 + MIDI 标注
        ↓
数据清洗
        ↓
划分 train / validation / test
        ↓
提取音频特征
        ↓
构造 frame / onset / offset 标签
        ↓
训练 AMT 模型
        ↓
生成预测结果
        ↓
转换为 MIDI
        ↓
计算评价指标
        ↓
错误分析
```
---
## 5.依赖库

### Python

| Name | Version |
|---|---:|
| Python | 3.10.20|
| pip | latest |
| conda | latest |

### Deep Learning

| Package | Version |
|---|---:|
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| torchaudio | 2.11.0 |
| torchmetrics | 1.9.0 |

### Scientific Computing

| Package | Version |
|---|---:|
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.9 |
| tqdm | latest |

### Audio / Music / MIDI

| Package | Version |
|---|---:|
| librosa | 0.11.0 |
| soundfile | 0.13.1 |
| audioread | 3.1.0 |
| pretty_midi | 0.2.11 |
| mido | 1.3.3 |
| music21 | 9.9.1 |
| mir_eval | 0.8.2 |
|ffmpeg|8.11|
### Notebook / Logging

| Package | Version |
|---|---:|
| jupyter | 1.1.1 |
| ipykernel | 7.2.0 |
| tensorboard | 2.20.0 |

### requirements.txt

```txt
torch==2.11.0
torchvision==0.26.0
torchaudio==2.11.0
torchmetrics==1.9.0

numpy==2.4.4
pandas==3.0.2
scipy==1.17.1
scikit-learn==1.8.0
matplotlib==3.10.9
tqdm

librosa==0.11.0
soundfile==0.13.1
audioread==3.1.0
pretty_midi==0.2.11
mido==1.3.3
music21==9.9.1
mir_eval==0.8.2

jupyter==1.1.1
ipykernel==7.2.0
tensorboard==2.20.0
