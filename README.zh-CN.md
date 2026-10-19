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

导入器还会在数据集输出目录生成 `phone_mapping_report.csv`，逐条记录源文件音素、映射后的目标音素序列和映射规则。训练普通话 VOCALOID 数据前应先检查此报告：目标音素存在于声库音素表，只说明标签可被读取，并不能证明源标记体系判断正确、发音含义一定匹配。仅凭文件名无法推断一段音频内部的真实音素边界。

`--source-phoneme-system` 默认值 `auto` 会保留现有 VOCALOID 中文映射。如果要专门为 Synthesizer V Studio 1.11.2 生成标签，请选择 `sv1112-chinese`（普通话）或 `sv1112-cantonese`（粤语）。这两个显式模式根据 Dreamtonics 官方音素变更表，将指定的 X-SAMPA 音素及明确列出的 VOCALOID 普通话别名映射到 `Old` 列，也就是 2.3 版本之前的音素标签；不会输出新版简化罗马字标签。所有映射结果都必须存在于所选目标声库的音素表中，不支持的符号会被报告，不会猜测拆分或擅自加入目标音素。原有 `xsampa`、`synthv-xsampa` 和 `vocaloid-mandarin` 模式继续保留；`none` 关闭转换。参考：[Dreamtonics 官方音素变更表（Chinese / Mandarin）](https://resource.dreamtonics.com/sv230b1_phoneme_changes.html#mandarin) 与 [Cantonese](https://resource.dreamtonics.com/sv230b1_phoneme_changes.html#cantonese)。通常指定 `--voice` 读取目标音素表即可，不必准备 `phone_table.json`。无法映射的音素或无法解码的 WAV 会记录在输出目录的 `unmatched_phones.csv`，其他文件仍会继续处理。

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

#### 音色微调预设、明亮度与鼻腔感诊断

Training 页面新增 Timbre fine-tuning preset（音色微调预设），提供以下起始配置：

- **Preserve base timbre（保留底模音色）**：只训练声库身份／风格向量，尽量避免修改底模的声学网络。
- **Gentle target adaptation（温和适配）**：较低的全权重学习率并加强 Anchor 约束。
- **Balanced VOCALOID fit（平衡拟合）**：使用默认的全权重学习率，适合作为对照基线。
- **Brightness / clarity experiment（明亮度／通透度实验）**：启用明亮度平衡损失，权重 0.05。
- **Higher detail fit（更强细节拟合，实验）**：明亮度权重 0.10，对目标频谱的额外约束更强，也更需要 A/B 试听。

这些数值是实验起点，不保证对每个底模都有效。明亮度平衡损失默认关闭（--brightness-weight 0）。设置为 0.05 或 0.10 时，训练会比较生成音频与目标 WAV 的 3–10 kHz 和 1–3 kHz 能量比，帮助模型保留目标音源实际具有的频谱平衡；它不是输出后的 EQ，不会无条件提高高频，也不能保证消除鼻腔共鸣。预设填写参数后仍可手动修改，建议从原始底模分别开新运行做比较，不要连续叠加不同预设。

“比较声库”并指定输出目录后，评估会额外生成 timbre_diagnostics.json，报告原始 WAV 和各声库渲染的频谱质心、低频／中频比、高频／中频比，以及与原 WAV 的差值。除整组短语平均值外，也会列出每条已评估短语的差值，以免某一句的局部问题被平均值掩盖。它们是定位整体发闷或高频能量不足的客观线索，不是可靠的自动鼻音评分；鼻腔感也与具体元音、音素边界和底模特征有关，最终仍需试听验证。

#### F0 分析与清理预设

Training 页面将 F0 Analysis & Calibration 与 Pitch Model & Training 分成两个独立面板。F0 预设只控制从输入 WAV 检测和清理基频目标，不修改 Pitch 网络的训练损失。

- **Standard**：65–1100 Hz、浊音阈值 0.60。
- **VOCALOID F0 cleanup**：45–1400 Hz、浊音阈值 0.45；额外启用有声片段内 25 ms 中值平滑与保守的孤立八度跳点修正。
- **Sensitive**：35–1800 Hz、浊音阈值 0.30，并使用 15 ms 轻度平滑，适合尝试找回较弱的有声帧。
- **Low / High / Wide**：选择不同检测音域，不默认打开额外清理。

F0 的音域、阈值、中值平滑、孤立八度修正、参考音分校正均可单独调整。有效 WAV 即使没有可测 F0（例如静音或单独的清辅音 /k/）也会保留用于声学模型训练；其音高掩码标记为无声，不会提供有效 Pitch 监督。敏感预设可以帮助跟踪较弱有声片段，但不能为真正静音或清辅音凭空生成基频。处理作用于训练/评估使用的 F0 目标曲线，并非对最终 WAV 做后期滤镜，也不能保证单独消除源音频、声学模型或声码器产生的金属音色。

#### Pitch 模型训练

Pitch 面板单独管理底模 Pitch 初始化或兼容 .dnni 导入，以及旋律平滑截止频率、旋律轮廓损失权重、快速音高运动/颤音损失权重。**切换 F0 预设不会改变 Pitch 损失参数**。有效的 Pitch 训练需要音符与词边界标注；改善整体音色还应训练并验证 Sound / Acoustic。
### 5. 评估并试听

训练完成后，进入结果页面，选择刚生成的声库并点击“比较声库”。工具会在留出的验证短语上比较原始声库和新声库，并可输出试听文件与 `metrics.json`。

检查辅音清晰度、音色一致性、音高和时长。如果结果变差，先检查录音、对齐、音素映射和标注质量。

### 6. 保存结果

- 普通底模输出为更新后的 `voice.nofs` 文件。
- `info.<版本>.nofs` 共享模型底模输出为数据库目录和相关模型文件。
- 保留原始底模、数据集和训练运行目录，避免覆盖唯一副本。

#### Flat 声库包含 DNNI v3 依赖时

DNNI v3 文件可以是随 SV2 兼容 Flat 声库分发的正常依赖；仅凭 v3 头部不能判断文件损坏。开发时检查的一个完整 SFPK 样本包含 4 个 DNNI 依赖，头部版本均为 v3。TrainSV 现在可以在读取 info 声库或旧式 NOFS 容器时，对任意模型角色的不透明 v3 文件进行逐字节保留，并在保存时原样写回，不尝试改写其内部数据。

**这只表示容器保留兼容，并不代表 DNNI v3 已可运行，也不代表已支持直接导入 SFPK。** 这次提供的 SFPK 包含 `config.json`、根目录 `voice.nofs`、`models/*.dnni`、图片和音素表；它不包含当前共享模型加载器需要的 `info.<版本>.nofs`。包内的 `voice.nofs` 长度字段也为 0，无法通过 TrainSV 严格的 NOFS v10 加载校验。TrainSV 目前还不能把 SFPK 中列出的模型 ID 映射到时长、Pitch、声学、声码器角色，也不能直接导入整个压缩包。不透明 v3 保留仅适用于模型依赖已经由 TrainSV 支持读取的声库容器引用到的情况。TrainSV 仍无法解码 v3 网络图或张量，也不能对不透明模型执行推理或训练。要实现完整 v3 支持，仍需要可合法访问的模型表示或运行时，并与 Synthesizer V Flat 做端到端验证。

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
  --f0-profile vocaloid \
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
  --f0-profile vocaloid \
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

### 训练前检查音素与过渡覆盖率

训练前可以先检查缺失 WAV、重复录音名、音素/时长标注格式、过短的非静音音素，以及只出现在少量录音中的相邻音素组合：

```powershell
trainsv audit-data --data my-dataset --check-audio --json-out runs/dataset-audit.json
```

默认会提示少于 3 份不同录音包含的音素，以及少于 2 份不同录音包含的相邻非静音音素组合。`--short-phone-ms` 可调整过短音素阈值（默认 20 毫秒）；`--min-phone-recordings` 和 `--min-transition-recordings` 可调整覆盖率阈值。使用 `--strict` 时，若发现缺失 WAV、无效标注行、重复名称或无法读取的 WAV 文件头，将以状态码 2 退出。该命令只读数据集，不会伪造样本或改写对齐。罕见过渡提示是需要进一步检查的线索，并不等于模型一定会发音错误。 WebUI 数据集页面也提供“检查缺失样本与音素过渡”按钮，可查看缺失 WAV、无效行、稀有音素/相邻音素组合及过短音素示例。训练高级选项新增“音素过渡强调”和“过渡范围”：`--transition-weight` 默认 `1.0`（关闭），`--transition-frames` 以 5 毫秒帧设置边界两侧范围。建议先完成数据审计，再谨慎尝试权重 `1.25`、范围 `2`。这只调整训练损失中非静音音素交界附近的权重，不会对生成音频做模糊平滑；验证评分仍保持不加权。

## 其他工具

- `trainsv synthesize-dataset --voice <声库> --output <数据集>`：生成练习数据，检查训练流程。
- `trainsv convert-dict`：在源、目标字典之间转换音素拼写。
- `--resume`：从运行目录的 `state.pt` 恢复；模型和参数范围必须与之前一致。
- `--history`：将验证指标保存为 JSON。
- `--save-best`：验证结果改善时额外保存声库。

## WebUI 声库打包（SVPK 实验版）

在 WebUI 的 Results 页面完成“Compare voices”后，会弹出格式选择窗口。选择“使用 SVPK 格式打包”，在弹窗中填写声库名称、供应商/作者、整数版本，选择 `profile.png` 头像与背景 PNG，然后点击“开始打包”。不需要手动创建或导入 JSON。当前 SVPK 打包仅支持普通单文件 `voice.nofs`，共享模型 `info.*.nofs` 暂不支持该流程。弹窗还提供四种 **仅用于诊断** 的 256 字节字段预设：`zero`（默认）、`random`、`sha256-chain`、`aes256-cbc-test`。命令行也可通过 `--header-preset zero|random|sha256-chain|aes256-cbc-test` 选择（默认 `zero`）。AES 选项使用合成、可公开推导的测试密钥/IV，并非 Synthesizer V 的密钥或官方格式。

SFPK 现在提供**实验性本机依赖版导出**。选择普通二进制 `voice.nofs` 时，`trainsv package-sfpk` 会把四类 DNNI 模型写入你指定的 Synthesizer V `models` 目录，以模型内容 MD5 作为 ID，并从原声库提取 32 个 float32 的声学风格/F0 向量，以及 128 个 float32 的 `duration_style` 向量，生成 `info.nofs` JSON 描述与 `config.json`，最后将这两个文件按样本结构打入 SFPK。SFPK **不包含模型二进制**，因此目标安装必须已有对应的模型依赖。Flat 中的导入、歌手选择和实际渲染尚未验证；通过手动测试之前，请勿把生成文件视为可用声库。

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

**已知限制：当前输出不可安装。** 已实测：当前工具将 256 字节外层字段置零后，Sharp 导入没有反应；将 Hanakuma Chifuyu 的 256 字节字段复制到自制包后，Sharp 能识别该包，但提示数据损坏。这强烈暗示字段可能参与包专属完整性/认证校验。用户提供的 STARDUST_INFINITY 样本也在偏移量 264 处开始 ZIP；其 256 字节字段包含 162 种不同字节、没有零字节，与 Teto 字段按位置仅有 1 个字节相同，观察到的字节熵约为 7.17 bit/byte。这符合高熵数据的表现，却不能识别具体算法。仅凭 256 字节长度，无法区分 RSA-2048 签名、任意数据或 16 个 AES-CBC 密文块；算法、签名覆盖范围和官方校验流程仍未确认。当前输出只能用于研究内部 ZIP 和元数据，不能当作可安装声库或发布；不能直接复制其他官方包的字段来解决。`random`、`sha256-chain` 和 `aes256-cbc-test` 只用于比较应用对不同合成字节的反应，不会重现官方认证数据，也不代表能够生成有效安装包。命令行的 `--license` 参数可选；若未提供，打包器会生成包含 `!language en-us` 与 `!language zh-cn` 区块的 `license.txt` 占位文本（按样本中观察到的语言标记格式）。它只说明尚未指定正式许可条款，并不会替你声明授权；实际发布前应填写你有权发布的真实许可条款。

包内文件名与顺序按已检查的 `KASANE_TETO_AI_104.svpk` 样例设置：`voice.nofs`、`license.txt`、`profile.png`、`background.png`、`package-config`。

### 实验性 SFPK 导出

```powershell
trainsv package-sfpk `
  --voice "out\voice.nofs" `
  --models-root "C:\path\to\Synthesizer V Studio\models" `
  --output "out\我的声库.sfpk" `
  --name "我的声库" `
  --vendor "作者或工作室" `
  --version 101 `
  --profile "assets\profile.png" `
  --background "assets\background.png"
```

也可以将从 Flat 包中提取的 `info.nofs` JSON 传给 `--voice`；其引用的所有模型 ID 必须已存在于所选 models 目录。普通 `voice.nofs` 则会导出 acoustic、vocoder、duration、F0 四类模型到相应子目录，并以模型内容的 MD5 作为 ID。SFPK 基本包含 `config.json` 和 `info.nofs`；指定 `--profile` 和/或 `--background` 后，会把 PNG 以 `profile.png`、`background.png` 的名称额外放入包根目录。两个图片参数都是可选的。不像模型依赖一样，图片会随 SFPK 一起打包；四类模型二进制仍在本机 `models` 目录，不包含在包内，因此它不是可独立迁移的完整包。请选择 Synthesizer V Studio Flat 实际使用的 `models` 目录，并在测试前备份。当前实现参照一个已检查的 SFPK 样本，仍需在 Flat 中导入、选择歌手并实际渲染验证。

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


## WebUI 三语切换

页面最上方新增 **简体中文 / English / 日本語** 语言选择器。选择会保存到本机 WebUI 设置，并在下次启动时恢复。导航、控件、参数帮助、动态状态提示、文件选择器和打包弹窗会随界面语言切换；录音本身使用的训练语言仍是独立设置。

## 项目结构

- `opensv_train/`：CLI、数据集、模型、训练和导出
- `parity/`：OpenSV 引擎及对比测试
- `tests/`：训练和引擎测试
- `trainsv_gui.py`：WebUI
- `pyproject.toml`：Python 项目配置

## 许可

项目采用 GPL-3.0。感谢 [OpenSV](https://github.com/cubeww/OpenSV) 项目提供的参考实现。请通过当前仓库的 [Issues](https://github.com/ggjk26/TrainSV/issues) 提交复现步骤、底模类型和完整日志。
文件名转 LAB 现支持按源标记体系选择明确映射：`auto`、`vocaloid-mandarin`、`xsampa` 与 `synthv-xsampa`。一个 VOCALOID 源音素映射成多个目标音素时，会尽量保留该源音素对应的整体时长，并缩短 `@_n`、`@n` 等组合中的鼻尾时长；一些 VOCALOID 音源的 `sta/@n_0.wav` 和 `art/@n @n_0.wav` 这类命名现已作为同一类源音素组合解析；这仍是启发式估算，不是真正的声学强制对齐，正式训练前建议抽查 `phone_mapping_report.csv` 和关键样本的 `.lab` 边界。VOCALOID 的 `u`（如 wu）与 `uo`（如 wo）是不同韵母，不能互相替代。F0 无有声帧不等同于 WAV 损坏；像 `Sil Sil k Sil_0.wav` 这样的有效静音/清辅音样本应保留，映射错误或真正无法解码的音频会单独记录到 `unmatched_phones.csv`。

## F0 分析与校准

Training 页面的 F0 Analysis & Calibration 面板专门处理输入 WAV 的基频分析和训练目标清理，与 Pitch 网络的训练设置分开。

- **Standard**：65–1100 Hz、浊音阈值 0.60。
- **VOCALOID F0 cleanup**：45–1400 Hz、浊音阈值 0.45；额外启用有声片段内的 25 ms 中值平滑和保守的孤立八度跳点修正。
- **Sensitive**：35–1800 Hz、浊音阈值 0.30，并启用 15 ms 轻度平滑。
- **Low / High / Wide**：分别覆盖低音、高音或宽音域，默认不启用额外清理。
- 可以单独覆盖 F0 最低/最高频率、浊音阈值、中值平滑、孤立八度修正和参考音分校正。

Sensitive 预设使用 35–1800 Hz 和 0.30 阈值，并以 15 ms 轻度平滑尝试找回较弱的有声帧。有效 WAV 即使没有测得 F0 也会保留用于声学训练，但没有有声帧的样本不会提供有效 Pitch 监督。若导入后样本数明显减少，应先检查导入摘要与 `unmatched_phones.csv`：数据加载逻辑不会仅因 F0 全零就剔除有效 WAV，音素映射失败或无法解码的音频才可能在生成数据集时被跳过。这项处理作用于训练/评估的目标曲线，而不是对导出 WAV 做后期平滑；它不能保证单靠 F0 就去除全部金属感。F0 选项可在 CLI 的 finetune 和 evaluate 中使用：--f0-profile、--f0-floor、--f0-ceiling、--f0-voicing-threshold、--f0-smoothing、--f0-octave-correction / --no-f0-octave-correction 与 --f0-reference-cents。

## Pitch 模型训练

独立的 Pitch Model & Training 面板用于管理 Pitch 网络本身：

- **模型初始化**：使用当前底模的 Pitch 网络，或导入结构/音素表兼容的预训练 .dnni。
- **Melody smoothing cutoff**：区分较慢的旋律轮廓和更快的音高变化；不会直接平滑最终音频。
- **Smooth melody loss weight**：控制整体旋律轮廓在 Pitch 训练损失中的权重。
- **Vibrato / fast-motion loss weight**：控制快速变化与颤音细节的损失权重。
- 这些设置与 F0 预设互相独立；切换 F0 档位不会更改 Pitch 损失值。

CLI 的 finetune 和 evaluate 支持 --pitch-smoothing-hz、--pitch-melody-weight 和 --pitch-motion-weight，默认分别为 3 Hz、1.0 和 1.0。Pitch 训练需要有效的 note_seq / note_dur 与词边界标注；如果希望改善整体音色，还需训练并验证 Sound / Acoustic。

TrainSV 可导入外部预训练 Pitch 权重（.dnni），CLI 对应 --pitch-base-model <path-to-pitch.dnni>。程序会严格检查网络架构、音素表、归一化缓冲区、稀疏结构与参数形状；仅在兼容时复制非 speaker 参数，保留目标声库自己的 speaker 向量。仓库**尚未随包附带经过验证的通用神经 Pitch 权重**，不同架构或音素表也不会因此自动兼容。
