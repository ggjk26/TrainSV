# 自制声库打包：SVPK（Sharp 测试版）

在 WebUI 的 Results 阶段，完成 Compare voices 后选择“使用 SVPK 格式打包”，填写声库名称、供应商/作者、整数版本，并选择头像与背景 PNG，即可自动启动打包。不需要手动编辑 JSON。命令行也可使用 `trainsv package-svpk`。`license.txt` 参数是可选项；未提供时会生成带有 `!language en-us` 与 `!language zh-cn` 语言区块的占位文本。它只提示尚未指定正式许可条款，并不会擅自声明授权；实际发布前应填入你有权发布的真实条款。

> **实验性功能：** 样本对比确认，官方 SVPK 的前 8 字节是 `SVPK 00 01 00 00`，后面有 256 字节不透明字段，ZIP 数据从偏移 264 开始。本实现将这 256 字节写为零；它没有生成官方签名，是否被 Sharp 接受仍须在目标软件里实测。不要把它描述成 Dreamtonics 官方签名安装包。

## 命令行用法

训练完成后运行：

```powershell
trainsv package-svpk `
  --voice "D:\TrainSV\out\voice.nofs" `
  --profile "D:\Voicebank\profile.png" `
  --background "D:\Voicebank\background.png" `
  --output "D:\Voicebank\Luo_Tianyi.svpk" `
  --name "Luo Tianyi" `
  --vendor "Your name or studio" `
  --version-string 101 `
  --version-number 101
```

也可以直接运行 `python -m opensv_train package-svpk --help` 查看参数。图片必须是 PNG。如果提供 `--license`，应当使用你有权随自制声库分发的许可文本，而不是复制商业声库的授权文件。

## 包含的内容

- `voice.nofs`：训练后的模型
- `license.txt`：如果提供 `--license` 就使用该文件；否则自动生成包含 `!language en-us` 和 `!language zh-cn` 区块的占位文本。它会明确提示尚未指定正式许可条款；发布前应换成你有权分发的真实许可条款。
- `profile.png`：头像
- `background.png`：背景图片
- `package-config`：`productName`、`vendorName`、`versionString`、`versionNumber`、`dependingVersionNumber`

条目名称和顺序与已检查的 `KASANE_TETO_AI_104.svpk` 样例一致：`voice.nofs`、`license.txt`、`profile.png`、`background.png`、`package-config`。

输出完成后，程序会检查 SVPK 头、ZIP 成员、CRC 和 `package-config` JSON。但 256 字节不透明字段当前置零，其含义尚未验证；必须在 Sharp 中实际安装并选择歌手测试。

## SFPK

`.sfpk` 是 Synthesizer V Studio Flat / Flat Manager 的另一种格式，不是把 `.svpk` 改名即可。Flat 的声库结构还涉及其 NOFS-JSON、模型 ID 与依赖模型等内容；在确认目标 SFPK 的真实结构及 TrainSV 输出与其模型架构兼容前，本命令不会伪造 SFPK。
