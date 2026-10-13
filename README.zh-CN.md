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
# 先诊断架构限制；不执行训练
trainsv diagnose-nofs --path /path/to/voice.nofs --json-output /path/to/architecture-report.json
```

### 导出 SVPK 安装包

训练完成后，将输出的模型文件和资源打成 `.svpk`。当前命令依据你提供的样例结构，生成根目录中的 `voice.nofs`、可选的 `background.png` / `license.txt` / `profile.png`，以及无扩展名的 `package-profile` JSON 文件。

```bash
trainsv package-svpk \
  --voice ./output/voice.nofs \
  --output ./dist/kasane-teto-ai-100.svpk \
  --product-name "Kasane Teto" \
  --vendor-name "AHS Co., Ltd." \
  --version-string "104" \
  --version-number 104 \
  --depending-version-number 0 \
  --profile ./assets/profile.png \
  --background ./assets/background.png \
  --license-file ./assets/license.txt
```

头像、背景和许可证是可选参数；图片必须是 PNG。默认背景条目名为 `background.png`；如果要逐字复现你提供的样本拼写 `backgrond.png`，可加 `--background-entry backgrond.png`。生成器会检查 ZIP CRC，但这不等同于在 Synthesizer V Studio 中验证安装成功。

### 打包共享声库 SFPK

```bash
trainsv package-sfpk \
  --info ./shared/info.nofs \
  --config-json ./shared/config.json \
  --output ./dist/hanakuma-chifuyu.sfpk \
  --profile ./shared/profile.png \
  --background ./shared/background.png
```

SFPK 打包器会保留你提供的 `info.nofs` 和 `config.json` 内容，并按样例根目录名添加资源。配置中的 `profile` / `background` 布尔值必须与是否提供对应图片一致；`models`、`model_types`、`phonesets` 必须为数组。请不要为了让命令通过而随意将这些数组填空：模型引用及共享声库的实际语义仍需从完整样例核实。

**重要限制：** 这条命令只负责封装已经准备好的 `info.nofs` 和配置文件，不会自动把普通 `voice.nofs` 转换成 Flat 所需的共享模型描述，也不会凭空生成 `.dnni` 模型文件。生成后必须在 Synthesizer V Flat 中验证加载和渲染结果。
### 检查 Flat 声库包或解包目录（只读）

```bash
trainsv inspect-sfpk --path /path/to/sample.sfpk --json-output /path/to/sfpk-report.json
# 同时检查本机的 DNNI 模型目录（Windows 示例）
trainsv inspect-sfpk --path /path/to/sample.sfpk --models-root "C:\\Users\\admin\\Documents\\OPSV\\Dreamtonics\\Synthesizer V Studio\\models" --json-output /path/to/sfpk-report.json
# 或者直接检查已解包的目录
trainsv inspect-sfpk --path /path/to/extracted-sfpk
```

`diagnose-nofs` 会分别报告 NOFS 容器解析、各角色 DNNI 图解析和当前架构标识限制；这是初步诊断，不等于完整训练兼容保证。`inspect-sfpk` 只读检查容器或解包目录，不会提取或修改输入；如果发现 JSON 格式的 `info.nofs`，还会列出音源元数据、风格名称、`acoustic` / `vocoder` / `duration` / `f0` 模型 ID，并按 TrainSV 共享声库的路径约定检查容器内是否存在对应 `.dnni` 文件；如果发现 `config.json`，也会检查 `profile` / `background` 标志与根目录 `profile.png` / `background.png` 是否对应。使用 `--models-root` 可以额外扫描本机 `models` 目录：按 `acoustic` / `vocoder` / `duration` / `f0` 目录列出 `.dnni` 文件，并对大小不超过 128 MiB 的候选输出 DNNI 版本、根节点类型、节点数和节点类型计数。检查器会核对精确模型 ID 是否存在，但不会因为某个候选文件“也是 DNNI”或节点类型相似，就擅自替换模型引用。结构摘要不是语义、张量形状或宿主兼容性保证。上述检查仅描述当前容器与本机文件清单，不等于正式宿主兼容性验证。ZIP 可读不代表它一定能被目标宿主加载。当前打包命令依据已观察到的文件名和配置字段实现，但仍需在 Synthesizer V Studio / Flat 中做实际加载验证。WebUI 的 “More tools” 页面也提供结构检查功能。

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

## 常见问题

**导入后没有生成 CSV？** 检查 LAB 是否存在、是否与 WAV 同名、标注区间是否连续，以及输出目录是否可写。

**为什么 Timing / Pitch 无法训练？** 它们需要音符时长与词边界等标注。只有音素对齐的 LAB 通常不足以训练它们。

**为什么提示底模架构不支持？** 能解析 `.nofs` 容器并不代表它的所有 DNNI 网络架构都可训练。TrainSV 目前通过固定的 duration、pitch、acoustic、vocoder 模型实现加载网络；遇到未实现的层或张量布局时不能假定兼容。请提供 `trainsv inspect` 输出、完整错误日志和底模版本，以便先识别具体限制再开发适配。

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
