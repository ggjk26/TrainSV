# TrainSV - Synthesizer V 声库训练工具包

[简体中文](README.zh-CN.md) | [English](README.md)

一个完整的 Python 工具包，用于在录制的语料库上微调 SynthV AI 声音的四个神经模型。简而言之，这是一个 **SynthV 开发工具包**。

## ✨ 主要特性

### 🎨 图形界面
- **Web GUI** - 内置图形界面，开箱即用
- 点击式操作，无需记忆命令行参数
- 实时显示训练日志和进度
- 自动检测 GPU/CPU 设备

### 🎯 四个可训练模型

| 模型 | DNNI 架构 | 训练目标 |
|------|-----------|---------|
| **duration** | `gen2a` | 每个音素在其音符中的时长分配 |
| **pitch** | `f0model-dds` | 录音的 F0 音高曲线（有声段） |
| **acoustic** | `dds` | 通过可微分声码器训练声学模型 |
| **vocoder** | `nhv` | 相同的音频损失（可选） |

### 🌍 支持的语言
- 🇯🇵 日语 (japanese / ja)
- 🇨🇳 中文 (mandarin / zh)
- 🇬🇧 英语 (english / en)
- 🇭🇰 粤语 (cantonese / yue)
- 🇪🇸 西班牙语 (spanish / es)

## 🚀 快速开始

### 安装

**前置要求**: Python 3.10+

```bash
# 创建虚拟环境
uv venv --python 3.11 .venv

# 安装依赖
uv pip install --python .venv/bin/python -e '.[test]'
```

这会安装 `trainsv` 命令（等同于 `python -m opensv_train`）。

### 启动 GUI

```bash
# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate.bat  # Windows

# 启动 Web GUI
python trainsv_gui.py
```

GUI 会在浏览器中自动打开（默认端口 8765）。

## 📖 完整工作流程

### 使用 GUI（推荐）

1. 启动 `python trainsv_gui.py`
2. 在浏览器中选择声库文件 (`.nofs`)
3. 导入你的语料库（WAV + LAB 文件）
4. 点击"Finetune"开始训练
5. 评估并导出训练后的声库

### 使用命令行

```bash
# 1. 导入数据集
trainsv import-lab --corpus <语料库目录> --output <数据集目录> --voice <声库文件>

# 2. 评估基础声库
trainsv evaluate --voice <声库文件> --data <数据集目录> --language ja

# 3. 微调训练
trainsv finetune \
  --voice <声库文件> \
  --data <数据集目录> \
  --language ja \
  --models acoustic,duration,pitch \
  --output out/voice.nofs \
  --name "我的歌手" \
  --run runs/my-singer

# 4. 评估训练结果
trainsv evaluate \
  --voice <声库文件> \
  --voice out/voice.nofs \
  --data <数据集目录> \
  --language ja \
  --out <评估目录>
```

## 📦 语料库格式

### 目录结构

支持两种格式：

**格式 1: 简单目录**
```
corpus/
  song_001.wav
  song_001.lab
  song_002.wav
  song_002.lab
```

或使用子目录：
```
corpus/
  wav/
    song_001.wav
    song_002.wav
  lab/
    song_001.lab
    song_002.lab
```

**格式 2: TrainV6 manifest.json**
```json
{
  "recordings": [
    {
      "id": "song_000",
      "wav": "/path/song_000.wav",
      "split": "train",
      "labels": [
        {"start_100ns": 0, "end_100ns": 5000000, "phoneme": "sil"}
      ]
    }
  ]
}
```

### 标注格式

每个 `.lab` 文件为 HTS/vLabeler 格式（每行 3 列）：

```
0 1200000 sil
1200000 1850000 k
1850000 7400000 a
7400000 9000000 sil
```

- 时间单位：100 纳秒
- 格式：`开始时间 结束时间 音素`
- `pau` 和 `sil` 表示静音，`br` 表示呼吸

## 🎓 训练选项

### 基础选项

```bash
--models acoustic,duration,pitch    # 选择要训练的模型
--steps 6000                        # 训练步数（默认 2000）
--lr 2e-5                           # 学习率（默认全参数 2e-5，嵌入向量 3e-3）
--batch 4                           # 批大小
--validation 0.1                    # 验证集比例
--device cuda                       # 设备：cuda/mps/cpu
```

### 高级选项

```bash
--params all                        # 训练所有参数（默认）
--params embeddings                 # 仅训练嵌入向量
--warmup 100                        # 学习率预热步数
--lr-end 2e-6                       # 余弦衰减的最终学习率
--consonant-weight 4                # 辅音帧权重
--vocal-modes Solid,High            # 声乐模式（用于 info 声库）
--resume                            # 从检查点恢复训练
```

### 配置文件

可以使用 JSON 配置文件（支持多个文件）：

```json
{
  "steps": 6000,
  "lr": 2e-5,
  "warmup": 100,
  "lr-end": 2e-6,
  "models": "acoustic,duration,pitch",
  "consonant-weight": 4
}
```

使用方式：
```bash
trainsv finetune --config config.json --voice ... --data ...
```

## 🔍 其他命令

### 检查声库信息
```bash
trainsv inspect --voice <声库文件>
```

列出声库的模型、参数数量、声乐模式和音素集。

### 合成数据集
```bash
trainsv synthesize-dataset --voice <声库文件> --output <数据集目录>
```

用任何声库渲染标注数据集（可用于端到端测试）。

### 转换字典
```bash
trainsv convert-dict \
  --data <源数据集> \
  --output <目标数据集> \
  --source-dict <源字典> \
  --target-dict <目标字典> \
  --voice <声库文件> \
  --language zh
```

## 📊 数据集格式

训练使用 DiffSinger 风格的数据集目录：

```
my-dataset/
  wavs/<名称>.wav              # 任意采样率，会重采样到 48 kHz
  transcriptions.csv           # 元数据文件
```

### transcriptions.csv 格式

必需列：
- `name` - 文件名（不含扩展名）
- `ph_seq` - 音素序列（用空格分隔）
- `ph_dur` - 音素时长（秒，用空格分隔）

可选列（用于时长/音高训练）：
- `ph_num` - 音素分组（DiffSinger 词边界）
- `note_seq` - 音符序列（MIDI 音高）
- `note_dur` - 音符时长
- `note_slur` - 连音标记
- `style` - 声乐风格

示例：
```csv
name,ph_seq,ph_dur,ph_num,note_seq,note_dur,note_slur
song_001,SP k a SP,0.1 0.05 0.6 0.1,0 1 1 2,rest C4 rest,0.1 0.65 0.1,0 0 0
```

## 🎯 训练建议

1. **从声学模型开始** - 先只训练 `--models acoustic`，验证效果后再添加其他模型
2. **使用验证集** - 设置 `--validation 0.1` 预留 10% 数据用于验证
3. **监控指标** - 关注 `log-mel` 距离和 `voicing` 准确率
4. **声乐模式训练** - 先用 `--params embeddings` 训练模式向量，再用 `--params all` 训练全网络
5. **保持录音水平** - 使用 `--level-reference` 指定基础声库，防止音量漂移

## 🧪 验证

### 运行测试

```bash
# 构建原生引擎
cmake -S parity -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target OpenSVParity

# 运行测试
.venv/bin/python -m pytest
```

测试需要已安装的声库（通过 `OPENSV_TEST_VOICES` 环境变量指定）。

## 📁 项目结构

```
TrainSV_1.0/
├── opensv_train/          # CLI、数据集、模型、训练和导出代码
├── parity/                # 内置的 OpenSV 引擎和测试工具
├── tests/                 # 对等性和训练测试
├── trainsv_gui.py         # Web GUI（新增！）
├── pyproject.toml         # 项目配置
└── README.md              # 英文文档
```

## 🔧 系统要求

- **Python**: 3.10 或更高
- **PyTorch**: 2.2 或更高
- **CMake**: 3.20 或更高（用于测试）
- **GPU**: 支持 CUDA/MPS（可选，CPU 也能训练）

## 🐛 常见问题

### 端口被占用
```bash
python trainsv_gui.py --port 8888  # 使用其他端口
```

### CUDA 不可用
训练会自动回退到 MPS（Apple Silicon）或 CPU。

### 内存不足
- 减少 `--batch` 批大小
- 使用更短的音频片段
- 训练单个模型而不是多个

## 📝 许可证

GPL-3.0

## 🙏 致谢

感谢 [OpenSV](https://github.com/cubeww/OpenSV) 项目提供的参考实现。

## 📮 联系方式

- **GitHub Issues**: [报告问题](https://github.com/Quadimension-CL/TrainSV/issues)
- **Email**: amd-miku@foxmail.com

---

**开始训练你的 SynthV AI 声库！🎤**
