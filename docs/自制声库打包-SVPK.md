# 自制声库打包：SVPK（Sharp 测试版）

本流程用于把 TrainSV 训练得到的 `voice.nofs`、自备的头像/背景、许可证文本和元数据封装成供 Synthesizer V Sharp 测试的 `.svpk`。

> **实验性功能：** 样本对比确认，官方 SVPK 的前 8 字节是 `SVPK 00 01 00 00`，后面有 256 字节不透明字段，ZIP 数据从偏移 264 开始。本实现将这 256 字节写为零；它没有生成官方签名，是否被 Sharp 接受仍须在目标软件里实测。不要把它描述成 Dreamtonics 官方签名安装包。

## 命令行用法

训练完成后运行：

```powershell
trainsv package-svpk `
  --voice "D:\\TrainSV\\out\\voice.nofs" `
  --license "D:\\Voicebank\\license.txt" `
  --profile "D:\\Voicebank\\profile.png" `
  --background "D:\\Voicebank\\background.png" `
  --output "D:\\Voicebank\\Luo_Tianyi.svpk" `
  --name "Luo Tianyi" `
  --vendor "Your name or studio" `
  --version-string 101 `
  --version-number 101
```

也可以直接运行 `python -m opensv_train package-svpk --help` 查看参数。图片必须是 PNG；`license.txt` 应当使用你有权随自制声库分发的许可文本，而不是默认复制商业声库的授权文件。

## 包含的内容

- `voice.nofs`：训练后的模型
- `license.txt`：你提供的许可文本
- `profile.png`：头像
- `background.png`：背景
- `package-config`：`productName`、`vendorName`、`versionString`、`versionNumber`、`dependingVersionNumber`

输出完成后，程序会检查 ZIP 成员、CRC 和配置 JSON，但这不能代替在 Sharp 中实际安装并选择歌手的测试。

## SFPK

`.sfpk` 是 Synthesizer V Studio Flat / Flat Manager 的另一种格式，不是把 `.svpk` 改名即可。Flat 的声库结构还涉及其 NOFS-JSON、模型 ID 与依赖模型等内容；在确认目标 SFPK 的真实结构及 TrainSV 输出与其模型架构兼容前，本命令不会伪造 SFPK。
