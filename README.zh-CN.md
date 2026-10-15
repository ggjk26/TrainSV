# TrainSV：Synthesizer V AI 声库再训练工具

[简体中文](README.zh-CN.md) | [English](README.md)

TrainSV 是用于微调 Synthesizer V AI 声库模型的 Python 工具。它从已有声库（`voice.nofs` 或 `info.<版本>.nofs`）出发，使用录音与时间对齐的音素标注训练声学、时长、音高等模型。

> **请先了解**
> - 并非所有 `.nofs` 都受支持。底模必须包含 TrainSV 当前实现的模型架构；如果报架构不支持，请保留完整错误日志和 `trainsv inspect` 输出。
> - 请只使用你有权修改的底模、录音、标注、头像和背景素材。

## 功能

- WebUI：在浏览器中选择底模、准备数据、训练和对比结果。
- 自动生成 CSV：导入 WAV + LAB 时，自动创建 `transcriptions.csv`。
- 数据输入：同目录 WAV/LAB、分开的 WAV 与 LAB 目录、TrainV6 `manifest.json`。
- 训练模型：声学（acoustic）、时长（duration）、音高（pitch）和可选声码器（vocoder）。
- 训练恢复、日志、验证指标和原始/训练后声库试听对比。

## 系统要求

- Python 3.10+（推荐 3.11）、PyTorch 2.2+。
- Windows、macOS 或 Linux。
- NVIDIA CUDA 或 Apple Silicon MPS 可加速；否则可使用 CPU，但会慢很多。
- 一个 TrainSV 当前支持的 Synthesizer V AI 底模。
- 已经完成时间对齐的录音数据。只有 WAV、没有 LAB 或等价的时间标签，不能直接训练。

## 安装

在项目根目录执行。

### 使用 uv（推荐）

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[test]"
```

以上是通用安装方式，不会根据电脑是否有 NVIDIA 显卡自动选择 CUDA 版 PyTorch。

#### NVIDIA 显卡（CUDA）

如果 `torch.__version__` 显示 `+cpu`，这个 PyTorch 版本无法使用 CUDA。在 Windows 上先完成上面的安装，然后在同一个虚拟环境里替换 PyTorch：

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```

Linux 请将解释器路径改为 `.venv/bin/python`。示例使用 PyTorch 2.14.1 官方 CUDA 13.0 安装源；若你的 NVIDIA 驱动不支持该版本，请在[官方安装配置页面](https://pytorch.org/get-started/locally/)选择兼容的 CUDA 版本。通常不需要另外安装完整 CUDA Toolkit，但必须安装兼容的 NVIDIA 驱动。

验证安装：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:',torch.__version__); print('CUDA build:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

版本带有 `+cu...` 且 `torch.version.cuda` 不是 `None`，说明安装的是 CUDA 版；最终还需要 `torch.cuda.is_available()` 返回 `True`，TrainSV 才能用 GPU 训练。

Windows PowerShell 也可以使用虚拟环境内的 Python 安装：

```powershell
uv venv --python 3.11 .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

### 使用 Python venv

```bash
python -m venv .venv
```

Windows：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

macOS / Linux：

```bash
.venv/bin/python -m pip install -e ".[test]"
```

## 启动 WebUI

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe trainsv_gui.py
```

macOS / Linux：

```bash
.venv/bin/python trainsv_gui.py
```

程序会在本机启动 Web 服务，并尝试打开浏览器，默认端口为 `8765`。如果端口被占用，可运行 `python trainsv_gui.py --port 8888`；不需要自动打开浏览器时使用 `--no-browser`。不要把本地 Web 服务暴露到不可信网络。

## 推荐流程：从录音到训练结果

### 1. 选择底模

1. 在 WebUI 选择 `.nofs` 文件。
2. 选择录音对应的语言：日语、普通话、英语、粤语或西班牙语。
3. 点击检查声库，查看模型、参数量和音素表。
4. 若提示模型架构不受支持，请先停止，不要强行训练；记录完整错误日志和底模信息。

普通 `voice.nofs` 与共享模型声库 `info.<版本>.nofs` 的输出结构不同，工具会按底模类型使用相应的保存方式。

### 2. 准备 WAV 与 LAB

推荐目录结构：

```text
my-corpus/
├── song_001.wav
├── song_001.lab
├── song_002.wav
└── song_002.lab
```

也支持 `wav/` 与 `lab/` 两个子目录。每个 WAV 必须有同名 LAB。

如果 WAV 文件名本身包含空格分隔的音素，也可以让 TrainSV **按文件名生成近似 LAB**，不必先手工制作标签。在 WebUI 的“录音”页面选择“Generate .lab from phonemes in WAV filenames”，指定 WAV 目录；可选指定 `phone_table.json`，或选择一个包含对应音素的声库作为备用音素表。生成的标签会保存在输出数据集的 `lab/` 子目录中。

```powershell
trainsv import-lab --from-filenames --wavs C:\\my-wavs --voice C:\\path\\to\\voice.nofs --output C:\\my-dataset --language mandarin
```

此模式按文件名从左到右排列音素，只删除末尾的 `_0` / `_1` 重录编号，保留音素内部下划线；首尾固定添加 `sil`，不会合并文件名中原有的静音。首尾静音各占音频时长的 10%，中间音素平均分配剩余 80%。**这是按规则均分的近似标注，不是声学强制对齐**；实际音素边界可能不同，建议先检查生成的 LAB。

如果 `--language mandarin`，`--source-phoneme-system` 默认是 `auto`，会将常见 VOCALOID 中文音素转成目标 SynthV 普通话音素（兼容 SynthV 简化音素名与旧式 X-SAMPA 音素名），并优先匹配目标声库实际包含的符号。通常只需指定 `--voice` 来读取目标音素表，不必准备 `phone_table.json`。对于不能转换的文件，程序会跳过该 WAV 并将文件名和错误原因写入输出目录的 `unmatched_phones.csv`；其余可处理的样本会继续生成标签。可用 `--source-phoneme-system none` 关闭转换，或显式指定 `vocaloid-mandarin`。

X-SAMPA 的 `@\``、`3\`` 表示卷舌化元音：若目标音素表中有完整符号则原样保留；若没有，程序会尝试用对应元音加音素表中已有的卷舌音近似拆分（优先 `r\\`，其次 `ɹ`、`ɻ`、`r` 等）。若表内既没有完整符号，也没有可用近似音，会报出更具体的错误，不会直接丢掉卷舌特征。

LAB 使用 HTS / vLabeler 对齐格式，每行包含开始时间、结束时间、音素，时间单位为 **100 纳秒**：

```text
0 1200000 sil
1200000 1850000 k
1850000 7400000 a
7400000 9000000 sil
```

- 区间必须按时间排序，不能有重叠或间隙。
- `pau` / `sil` 会转换为静音 `SP`，`br` 会转换为呼吸 `AP`。
- 音素名必须与底模音素表匹配；不匹配时需要音素映射。
- 普通 LAB 通常只有音素与时长信息，可用于声学模型训练；时长和音高模型还需要音符、词边界等额外标注。

### 3. 导入数据，自动生成 CSV

在 WebUI 的数据页面：

1. 选择“我有 WAV 和 LAB”。
2. 选择同目录 WAV/LAB、分开的目录，或 TrainV6 的 `manifest.json`。
3. 指定数据集输出目录并开始导入。
4. 导入完成后，程序会复制 WAV，并自动生成 `transcriptions.csv`。

生成的目录类似：

```text
my-dataset/
├── wavs/
│   ├── song_001.wav
│   └── song_002.wav
└── transcriptions.csv
```

因此，大多数情况下不需要手动制作 CSV。已有符合格式的数据集可以直接选择并跳过导入。

CSV 必需列：
- `name`：WAV 名称，不含扩展名。
- `ph_seq`：空格分隔的音素序列。
- `ph_dur`：空格分隔的音素时长，单位秒。

时长/音高训练还需要有效的 `ph_num`、`note_seq`、`note_dur` 等音符和词边界信息。

### 4. 训练

初次使用建议：

1. 先只选择 **Sound / Acoustic（声学模型）**。
2. 先运行快速测试，确认环境、底模和数据正常。
3. 快速测试通过后，再运行标准训练。
4. 初次使用保留默认学习率、验证比例和自动设备选择。
5. 只有数据包含音符与词边界标注时，才选择 Timing / Pitch。
6. Vocoder 属于可选高级项目，初次训练建议关闭。

| 参数 | 作用 | 建议 |
| --- | --- | --- |
| Steps | 训练步数 | 先快速测试，再逐步增加 |
| Learning rate | 学习率 | 初次使用留空，采用默认值 |
| Batch | 每步累计的短语数 | 显存不足时保持较小值 |
| Validation | 留作验证的数据比例或数量 | 默认 0.1 约为 10% |
| Device | 训练设备 | 优先 Automatic |
| Save best | 验证结果改善时额外保存 | 建议开启 |

训练步数更多不代表效果一定更好。请结合验证结果和试听判断，不能只看训练损失。

### 5. 评估并试听

训练完成后，进入结果页面，选择刚生成的声库并点击“比较声库”。工具会在留出的验证短语上比较原始声库和新声库，并可输出试听文件与 `metrics.json`。

检查辅音清晰度、音色一致性、音高和时长。如果结果变差，先检查录音、对齐、音素映射和标注质量。

### 6. 保存结果

- 普通底模输出为更新后的 `voice.nofs` 文件。
- `info.<版本>.nofs` 共享模型底模输出为数据库目录和相关模型文件。
- 保留原始底模、数据集和训练运行目录，避免覆盖唯一副本。

## 命令行使用

### 检查底模

```bash
trainsv inspect --voice /path/to/voice.nofs
```

### 导入并生成 CSV

```bash
trainsv import-lab \
  --corpus /path/to/my-corpus \
  --output /path/to/my-dataset \
  --voice /path/to/voice.nofs
```

WAV 与 LAB 分开放置时：

```bash
trainsv import-lab \
  --wavs /path/to/my-corpus/wav \
  --labs /path/to/my-corpus/lab \
  --output /path/to/my-dataset \
  --voice /path/to/voice.nofs
```

### 训练声学模型

```bash
trainsv finetune \
  --voice /path/to/voice.nofs \
  --data /path/to/my-dataset \
  --language ja \
  --models acoustic \
  --steps 2000 \
  --output out/voice.nofs \
  --name "我的声库" \
  --run runs/my-voice
```

支持的语言名/代码：`japanese/ja`、`mandarin/zh`、`english/en`、`cantonese/yue`、`spanish/es`。

### 评估训练结果

```bash
trainsv evaluate \
  --voice /path/to/voice.nofs \
  --voice out/voice.nofs \
  --data /path/to/my-dataset \
  --language ja \
  --out out/evaluation
```

### JSON 配置文件

```json
{
  "steps": 6000,
  "lr": 0.00002,
  "warmup": 100,
  "lr-end": 0.000002,
  "models": "acoustic",
  "validation": 0.1
}
```

运行：`trainsv finetune --config config.json --voice ... --data ...`。

## 数据集格式

```text
my-dataset/
├── wavs/<名称>.wav
└── transcriptions.csv
```

| 列名 | 说明 |
| --- | --- |
| `name` | WAV 文件名，不含扩展名 |
| `ph_seq` | 音素序列 |
| `ph_dur` | 音素时长（秒） |
| `ph_num` | 词边界/音素分组 |
| `note_seq` | 音符序列 |
| `note_dur` | 音符时长 |
| `note_slur` | 连音标记 |
| `style` | Vocal Mode 名称 |

## 其他工具

- `trainsv synthesize-dataset --voice <声库> --output <数据集>`：生成练习数据，检查训练流程。
- `trainsv convert-dict`：在源、目标字典之间转换音素拼写。
- `--resume`：从运行目录的 `state.pt` 恢复；模型和参数范围必须与之前一致。
- `--history`：将验证指标保存为 JSON。
- `--save-best`：验证结果改善时额外保存声库。

## WebUI 声库打包（SVPK 实验版）

在 WebUI 的 Results 页面完成“Compare voices”后，会弹出格式选择窗口。选择“使用 SVPK 格式打包”，在弹窗中填写声库名称、供应商/作者、整数版本，选择 `profile.png` 头像与背景 PNG，然后点击“开始打包”。不需要手动创建或导入 JSON。当前 SVPK 打包仅支持普通单文件 `voice.nofs`，共享模型 `info.*.nofs` 暂不支持该流程。

SFPK 按钮当前是占位提示。如果所选结果是普通 `voice.nofs`，程序会说明它缺少共享模型结构所需的 `info.nofs`、`config.json` 和相应的 `.dnni` 模型引用；不会生成一个未经验证的伪包。

命令行也可使用 `trainsv package-svpk`：

```powershell
trainsv package-svpk `
  --voice "out\voice.nofs" `
  --profile "assets\profile.png" `
  --background "assets\background.png" `
  --output "out\MySinger.svpk" `
  --name "My Singer" `
  --vendor "作者或工作室" `
  --version-string 101 `
  --version-number 101
```

**注意：这是实验性封装。** 已观察到的 SVPK 样本由 8 字节格式头、256 字节不透明字段和 ZIP 数据组成；当前工具将这 256 字节置零，因为它的正式含义尚未确认。生成后必须在 Sharp 中实测，不能将其称为 Dreamtonics 官方签名的安装包。命令行的 `--license` 参数可选；若未提供，打包器会生成明确标注“尚未指定许可条款”的 `license.txt` 占位文本，并不会替你声明授权。实际发布前应填写真实许可条款。

包内文件名与顺序按已检查的 `KASANE_TETO_AI_104.svpk` 样例设置：`voice.nofs`、`license.txt`、`profile.png`、`background.png`、`package-config`。

该命令只生成 SVPK。SFPK 是 Flat Manager 使用的另一种格式，不能仅通过更改扩展名生成。

## 常见问题

**导入后没有生成 CSV？** 检查 LAB 是否存在、是否与 WAV 同名、标注区间是否连续，以及输出目录是否可写。

**为什么 Timing / Pitch 无法训练？** 它们需要音符时长与词边界等标注。只有音素对齐的 LAB 通常不足以训练它们。

**为什么提示底模架构不支持？** TrainSV 只支持代码明确实现的模型架构。请提供 `trainsv inspect` 输出和完整错误日志，以便针对具体底模扩展。

**训练慢或显存不足？** 减小 Batch、先只训练声学模型、缩短测试步数，或使用 GPU。CPU 可用但速度可能很慢。

**运行测试：**

```bash
cmake -S parity -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target OpenSVParity
.venv/bin/python -m pytest
```

部分测试需要可用的声库样本，可通过 `OPENSV_TEST_VOICES` 指定。

## 项目结构

- `opensv_train/`：CLI、数据集、模型、训练和导出
- `parity/`：OpenSV 引擎及对比测试
- `tests/`：训练和引擎测试
- `trainsv_gui.py`：WebUI
- `pyproject.toml`：Python 项目配置

## 许可

项目采用 GPL-3.0。感谢 [OpenSV](https://github.com/cubeww/OpenSV) 项目提供的参考实现。请通过当前仓库的 [Issues](https://github.com/ggjk26/TrainSV/issues) 提交复现步骤、底模类型和完整日志。
现在支持输入Vocaloid音频样本的X-SAMPA音素，不会因为不匹配Synthesizer V的而报错退出。目前F0预测器的清浊音部分处理有问题（如“Sil k_01.wav”会识别损坏的wav从而终止训练）
