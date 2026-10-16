# 自制声库打包：SVPK（Sharp 测试版）

在 WebUI 的 Results 阶段，完成 Compare voices 后选择“使用 SVPK 格式打包”，填写声库名称、供应商/作者、整数版本，选择外层 256 字节字段诊断预设，再选择头像与背景 PNG，即可启动打包。不需要手动编辑 JSON。命令行也可使用 `trainsv package-svpk`。预设包括 `zero`（默认）、`random`、`sha256-chain` 和 `aes256-cbc-test`；CLI 参数为 `--header-preset`。后三者均为合成诊断字节，其中 AES 使用公开可推导的测试密钥/IV，不是官方算法或密钥。`license.txt` 参数是可选项；未提供时会生成带有 `!language en-us` 与 `!language zh-cn` 语言区块的占位文本。它只提示尚未指定正式许可条款，并不会擅自声明授权；实际发布前应填入你有权发布的真实条款。

> **已知不可安装：** 用户已实测，当前实现将 256 字节外层字段置零后，Sharp 导入没有反应；将 Hanakuma Chifuyu 的 256 字节字段复制到自制包后，Sharp 能识别到包，但报告“数据损坏”。结合 Teto、Chifuyu 与 STARDUST_INFINITY 样本的字段内容不同，这强烈暗示该区域可能参与包专属完整性/认证校验。STARDUST_INFINITY 的 ZIP 也从偏移量 264 开始；它的字段有 162 种不同字节、没有零字节，与 Teto 字段按位置只有 1 个字节相同，观察到的字节熵约 7.17 bit/byte。该统计只能说明数据具有高熵特征。256 字节长度既可能对应 RSA-2048 签名，也相当于 16 个 AES-CBC 密文块；仅凭长度无法确认算法、签名输入或官方签署流程。当前命令只生成用于研究 ZIP/元数据结构的诊断包，不能当作可安装声库或发布。四种预设都只是诊断输入，不能伪装成已验证的官方签名，也不能据此断言 AES-CBC 或 RSA 是真实实现。不能把其他官方包的字段直接复制到修改过的包里；必须找到受支持的打包/签署流程，或选择有明确自定义声库支持的其他格式。

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

## SFPK（实验性本机依赖版）

`.sfpk` 是 Synthesizer V Studio Flat / Flat Manager 使用的另一种格式，不能把 `.svpk` 改名后得到。我们检查的 Chifuyu 样本包含 ZIP 归档中的 `config.json` 和 `info.nofs`，而 `info.nofs` 通过模型 ID 引用 acoustic、vocoder、duration 与 F0 模型；样本中的声学风格和 F0 向量各为 32 个 float32，`duration_style` 则为 128 个 float32（1024 个十六进制字符）。当前 TrainSV 的实验实现根据这个样本格式生成本机依赖版包。

```powershell
trainsv package-sfpk `
  --voice "out\voice.nofs" `
  --models-root "C:\path\to\Synthesizer V Studio\models" `
  --output "out\MySinger.sfpk" `
  --name "My Singer" `
  --vendor "Your name or studio" `
  --version 101
```

- `--voice` 可指定训练后普通二进制 `voice.nofs`，或已有 `info.nofs` JSON 文件。
- `--models-root` 必须是目标 Synthesizer V Studio Flat 使用的实际 `models` 目录。普通 `voice.nofs` 的模型会以其序列化字节内容的 MD5 命名，写入 `acoustic/`、`vocoder/`、`duration/` 和 `f0/` 子目录；如果目标 ID 文件已存在但内容不同，程序会拒绝覆盖。
- `--name`、`--vendor` 和 `--version` 可覆盖原元数据。
- 输出 SFPK 当前仅包含 `config.json` 和 `info.nofs`，不会把模型二进制嵌入包内，因此它不是自包含/便携包。目标安装必须能够访问这些同 ID 的模型依赖。

这版参考一个实际样本的文件结构，但尚未通过 Synthesizer V Studio Flat 的运行时验证。生成后需手动导入/安装、确认歌手出现在列表、选择声库并渲染音频；如果失败，请保留生成的 SFPK、所选 models 目录结构和完整报错。不要把通过 ZIP/JSON 检查当成 Flat 兼容性的证明。