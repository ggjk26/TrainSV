# 自制声库打包：SVPK（Sharp 测试版）

在 WebUI 的 Results 阶段，完成 Compare voices 后选择“使用 SVPK 格式打包”，填写声库名称、供应商/作者、整数版本，并选择头像与背景 PNG，即可自动启动打包。不需要手动编辑 JSON。命令行也可使用 `trainsv package-svpk`。`license.txt` 参数是可选项；未提供时会生成带有 `!language en-us` 与 `!language zh-cn` 语言区块的占位文本。它只提示尚未指定正式许可条款，并不会擅自声明授权；实际发布前应填入你有权发布的真实条款。

> **已知不可安装：** 用户已实测，当前实现将 256 字节外层字段置零后，Sharp 导入没有反应；将 Hanakuma Chifuyu 的 256 字节字段复制到自制包后，Sharp 能识别到包，但报告“数据损坏”。结合 Teto 与 Chifuyu 原版字段内容不同，这强烈暗示该区域参与包专属完整性/认证校验。256 字节长度与 RSA-2048 签名相符，但具体算法、签名输入数据和官方签署流程都尚未确认。当前命令只生成用于研究 ZIP/元数据结构的诊断包，不能当作可安装声库或发布。不能把其他官方包的字段直接复制到修改过的包里；必须找到受支持的打包/签署流程，或选择有明确自定义声库支持的其他格式。

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

输出完成后，程序会检查 ZIP 成员、CRC 和 `package-config` JSON；这只验证内部归档结构，不代表 SVPK 外层有效。全零字段版本在 Sharp 中无响应。外层 256 字节字段目前仍未有已验证的公开格式说明，因此当前输出仅供诊断，不可安装；只有通过受支持的打包/签署方式生成并完成 Sharp 安装测试，才能称为可用声库包。

## SFPK

`.sfpk` 是 Synthesizer V Studio Flat / Flat Manager 的另一种格式，不是把 `.svpk` 改名即可。Flat 的声库结构还涉及其 NOFS-JSON、模型 ID 与依赖模型等内容；在确认目标 SFPK 的真实结构及 TrainSV 输出与其模型架构兼容前，本命令不会伪造 SFPK。
