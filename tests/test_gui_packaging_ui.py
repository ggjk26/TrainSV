"""Regression checks for the Results-stage package UI."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from trainsv_gui import PAGE


def test_results_page_exposes_both_package_choices():
    assert 'id="openPackageFormat"' in PAGE
    assert 'id="chooseSvpk"' in PAGE
    assert "Build SVPK Diagnostic Package (Not Installable)" in PAGE
    assert 'id="chooseSfpk"' in PAGE
    assert "Export SFPK Package (Experimental)" in PAGE


def test_svpk_dialog_collects_required_metadata_and_assets():
    for field in (
        "svpk_name",
        "svpk_vendor",
        "svpk_version",
        "svpk_header_preset",
        "svpk_profile",
        "svpk_background",
        "svpk_output",
    ):
        assert f'data-id="{field}"' in PAGE
    assert "JSON" in PAGE
    assert "全零字段版本在 Sharp 中无反应" in PAGE
    assert "Hanakuma Chifuyu" in PAGE
    assert "复制 Chifuyu 字段会触发“数据损坏”" in PAGE
    assert "完整性/认证校验" in PAGE
    assert "Build Diagnostic Package (Not Installable)" in PAGE
    assert "startJob(\"svpk\")" in PAGE
    assert '"--version-string",version' in PAGE
    assert '"--version-number",String(Number(version))' in PAGE
    assert '"--header-preset",val("svpk_header_preset")||"zero"' in PAGE
    assert "aes256-cbc-test|Synthetic AES-256-CBC test (unofficial)" in PAGE
    assert "不是官方签名或密钥" in PAGE


def test_sfpk_dialog_collects_local_model_dependencies_and_starts_job():
    for field in (
        "sfpk_name",
        "sfpk_vendor",
        "sfpk_version",
        "sfpk_profile",
        "sfpk_background",
        "sfpk_models_root",
        "sfpk_output",
    ):
        assert f'data-id="{field}"' in PAGE
    assert 'id="sfpkDialog"' in PAGE
    assert "function openSfpkDialog()" in PAGE
    assert "info(?:\\..*)?" in PAGE
    assert 'cmd:"package-sfpk"' in PAGE
    assert '"--models-root",val("sfpk_models_root")' in PAGE
    assert 'data-ext="png" data-label="Avatar image profile.png (optional)"' in PAGE
    assert 'data-ext="png" data-label="Background image background.png (optional)"' in PAGE
    assert 'if(val("sfpk_profile"))a.push("--profile",val("sfpk_profile"))' in PAGE
    assert 'if(val("sfpk_background"))a.push("--background",val("sfpk_background"))' in PAGE
    assert 'startJob("sfpk")' in PAGE
    assert "SFPK 打包尚未实现" not in PAGE


def test_embedded_webui_javascript_has_valid_syntax():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; CI provides it for this check.")
    match = re.search(r"<script>\s*(.*?)\s*</script>", PAGE, flags=re.DOTALL)
    assert match is not None
    with tempfile.TemporaryDirectory() as temp_dir:
        script = Path(temp_dir) / "trainsv-gui.js"
        script.write_text(match.group(1), encoding="utf-8")
        result = subprocess.run(
            [node, "--check", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout



def test_svpk_output_suggestion_preserves_unicode_name_and_manual_paths():
    assert "function packageFileStem(name)" in PAGE
    assert "function suggestSvpkOutput()" in PAGE
    assert 'packageFileStem(val("svpk_name"))+".svpk"' in PAGE
    assert 'if(t.id==="svpk_name")suggestSvpkOutput()' in PAGE
    assert 'if(t.id==="svpk_output"&&SVPK_AUTO_OUTPUT' in PAGE



def test_successful_comparison_opens_the_package_format_chooser():
    assert 'if(JOB.kind==="evaluate"){go("results");showPackageFormatDialog()}' in PAGE
    assert 'if(target.isfile&&!confirm("The output file already exists:\\n"+val("svpk_output")+"\\n\\nOverwrite it?"))return;' in PAGE


def test_training_and_evaluation_expose_f0_tracking_profiles():
    assert 'data-id="t_f0_profile"' in PAGE
    assert 'data-options="standard|Standard (65–1100 Hz, threshold 0.60);vocaloid|VOCALOID F0 cleanup (45–1400 Hz, threshold 0.45);low|Low register (45–650 Hz, threshold 0.60);high|High register (100–1400 Hz, threshold 0.60);wide|Wide range (40–1600 Hz, threshold 0.60);sensitive|Sensitive (35–1800 Hz, threshold 0.30)"' in PAGE
    assert 'a.push("--f0-profile",val("t_f0_profile")||"standard")' in PAGE
    assert "F0 预设只控制输入 WAV 的基频检测范围与目标曲线清理" in PAGE
    assert 'id="f0CalibrationPanel"' in PAGE



def test_webui_language_switch_supports_chinese_english_and_japanese():
    header = PAGE.split('<header class="top">', 1)[1].split("</header>", 1)[0]
    assert header.index('id="uiLanguage"') < header.index('id="quitBtn"')
    for locale in ('value="zh-CN"', 'value="en"', 'value="ja"'):
        assert locale in header
    assert "function setUiLanguage(language)" in PAGE
    assert "function watchTranslatedContent()" in PAGE
    assert 'if(t.id==="uiLanguage"){setUiLanguage(val("uiLanguage"));persist();return;}' in PAGE
    assert 'setUiLanguage(val("uiLanguage")||"en");' in PAGE
    assert "I18N_INDEX = {en:0,\"zh-CN\":1,ja:2}" in PAGE
    assert "F0 分析与校准" in PAGE
    assert "F0 解析と調整" in PAGE
    assert "F0 Analysis & Calibration" in PAGE


def test_f0_and_pitch_have_distinct_workspaces():
    assert 'id="f0CalibrationPanel"' in PAGE
    assert 'id="pitchTrainingPanel"' in PAGE
    f0_panel = PAGE.split('<fieldset id="f0CalibrationPanel">', 1)[1].split("</fieldset>", 1)[0]
    pitch_panel = PAGE.split('<fieldset id="pitchTrainingPanel">', 1)[1].split("</fieldset>", 1)[0]
    for field in (
        "t_f0_profile",
        "t_f0_floor",
        "t_f0_ceiling",
        "t_f0_threshold",
        "t_f0_smoothing",
        "t_f0_correction",
    ):
        assert f'data-id="{field}"' in f0_panel
        assert f'data-id="{field}"' not in pitch_panel
    assert 'id="t_f0_octave"' in f0_panel
    for field in (
        "t_pitch_source",
        "t_pitch_model",
        "t_pitch_smoothing",
        "t_pitch_melody_weight",
        "t_pitch_motion_weight",
    ):
        assert f'data-id="{field}"' in pitch_panel
        assert f'data-id="{field}"' not in f0_panel
    assert 'a.push("--pitch-base-model",val("t_pitch_model"))' in PAGE
    assert 'pitch(){return this.finetune(true)}' in PAGE
    assert "F0 options control only the analysis and cleanup" in f0_panel
    assert "independent of F0 detection" in pitch_panel


def test_vocaloid_f0_profile_does_not_override_pitch_loss_settings():
    assert 'vocaloid:{floor:"",ceiling:"",threshold:"",smoothing:"5",octave:true}' in PAGE
    assert 'if(t.id==="t_f0_profile")applyF0ProfileControls(val("t_f0_profile"))' in PAGE
    assert '"--no-f0-octave-correction"' in PAGE
    assert "VOCALOID F0 cleanup" in PAGE
    fn = PAGE.split("function applyF0ProfileControls(profile,clearOverrides=true){", 1)[1].split("\nconst B={", 1)[0]
    assert "t_pitch_smoothing" not in fn
    assert "t_pitch_melody_weight" not in fn
    assert "t_pitch_motion_weight" not in fn



def test_dynamic_webui_messages_translate_in_all_locales():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; CI provides it for this check.")
    rows_match = re.search(r"const I18N_ROWS = (\[.*?\]);\nconst I18N_MAP", PAGE, flags=re.DOTALL)
    function_match = re.search(
        r"function translateString\(source, language=UI_LANG\) \{.*?\n\}", PAGE, flags=re.DOTALL
    )
    assert rows_match is not None
    assert function_match is not None
    cases = [
        [
            "✓ 4 recordings, about 2.5 min of singing. 3 have notes. Warning: 1 listed recordings have no .wav file.",
            "zh-CN",
            "✓ 4 条录音，歌唱时长约 2.5 分钟。 3 条录音包含音符。 警告：有 1 条已列出的录音缺少 .wav 文件。",
        ],
        [
            "Found 2 .wav and 1 .lab files. Each recording needs both.",
            "ja",
            ".wav 2 個、.lab 1 個が見つかりました。 各録音には両方のファイルが必要です。",
        ],
        [
            "SFPK test package created: D:\\voice.sfpk. Model dependencies were written to the local directory: D:\\models. Import and test rendering in Synthesizer V Studio Flat; runtime compatibility has not yet been verified.",
            "zh-CN",
            "已生成 SFPK 测试包：D:\\voice.sfpk。模型依赖已写入本机目录：D:\\models。请在 Synthesizer V Studio Flat 中导入并测试渲染；运行时兼容性尚未验证。",
        ],
        [
            "Missing Python packages: numpy, torch. In the TrainSV folder run: pip install -e .   (use the same Python that started this page: C:\\Python\\python.exe)",
            "ja",
            "不足している Python パッケージ：numpy, torch。TrainSV フォルダーで pip install -e . を実行してください（このページの起動に使用した Python：C:\\Python\\python.exe）。",
        ],
        [
            "C:\\Dataset (No permission to open this folder.)",
            "zh-CN",
            "C:\\Dataset（无权限打开此文件夹）",
        ],
        [
            "“steps” must be a number (you typed “oops”).",
            "zh-CN",
            "“steps”必须是数字（你输入的是“oops”）。",
        ],
        [
            "Finished. Your new voice is saved at D:\\voice.nofs (1m 30s)",
            "ja",
            "学習が完了しました。新しい音声の保存先：D:\\voice.nofs（所要時間：1m 30s）",
        ],
        [
            "The output file already exists:\nD:\\voice.svpk\n\nOverwrite it?",
            "zh-CN",
            "输出文件已存在：\nD:\\voice.svpk\n\n是否覆盖？",
        ],
        ["Request failed (404)", "ja", "リクエストに失敗しました（HTTP 404）。"],
        [
            "Could not read transcriptions.csv: invalid encoding",
            "zh-CN",
            "无法读取 transcriptions.csv：invalid encoding",
        ],
        ["Choose a voice (step 1).", "ja", "音声ライブラリを選択してください（手順 1）。"],
        ["Help: Steps", "zh-CN", "帮助：步骤"],
        ["Help: Steps", "ja", "ヘルプ：手順"],
        ["step 3/8", "zh-CN", "第 3 / 8 步"],
        ["step 3/8", "ja", "ステップ 3 / 8"],
        ["Session expired. Reload the page.", "zh-CN", "会话已过期，请重新加载页面。"],
        ["Something is already running. Stop it first.", "ja", "既にタスクが実行中です。先に停止してください。"],
        ["Could not start Python: Permission denied", "zh-CN", "无法启动 Python：Permission denied"],
        ["unknown command: probe", "ja", "不明なコマンド：probe"],
        ["Timing (duration)", "ja", "Timing（長さ）"],
        ["No permission to open this folder.", "zh-CN", "无权打开此文件夹。"],
        [
            "This setting affects the current operation. Choose based on your training goal.",
            "zh-CN",
            "此设置会影响当前操作，请根据训练目标选择。",
        ],
        [
            "This setting affects the current operation. Choose based on your training goal.",
            "ja",
            "この設定は現在の操作に影響します。学習の目的に合わせて選択してください。",
        ],
        ["Sound (acoustic)", "zh-CN", "Sound（声学）"],
        ["Pitch (melody and vibrato)", "zh-CN", "Pitch（旋律与颤音）"],
        ["Preset default, e.g. 45", "zh-CN", "预设默认值，例如 45"],
        ["Preset default, e.g. 1400", "ja", "プリセットの既定値（例：1400）"],
        ["0 (no correction)", "zh-CN", "0（不校正）"],
        ["2e-5 (3e-3 for identity only)", "zh-CN", "2e-5（仅训练身份向量时使用 3e-3）"],
        ["e.g. Artist or studio", "ja", "例：アーティスト名またはスタジオ名"],
        [
            "4473 utterances, 4473 phrases, 37.1 min. Getting ready: loading the voice and your data. This can take a minute the first time.",
            "zh-CN",
            "4473 条发声片段，4473 个短语，约 37.1 分钟。正在准备：加载声库和数据。首次运行可能需要一点时间。",
        ],
    ]
    script = "\n".join(
        [
            "const I18N_ROWS = " + rows_match.group(1) + ";",
            "const I18N_MAP = new Map(); for (const row of I18N_ROWS) for (const source of row) if (source && !I18N_MAP.has(source)) I18N_MAP.set(source, row);",
            'const I18N_INDEX = {en:0,"zh-CN":1,ja:2}; const UI_LANG = "en";',
            function_match.group(0),
            "const cases = " + json.dumps(cases, ensure_ascii=False) + ";",
            "for (const [source, language, expected] of cases) { const actual = translateString(source, language); if (actual !== expected) { console.error(JSON.stringify({source, language, expected, actual})); process.exit(1); } }",
        ]
    )
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert r"source.match(/^✓ (\d+) recordings, about ([\d.]+) min" in PAGE
    assert r"source.match(/^SFPK test package created: (.+?)\. Model dependencies" in PAGE
    assert 'id="uiLanguage"' in PAGE
    assert "characterData:true" in PAGE


def test_filename_import_exposes_explicit_phone_conventions():
    assert 'data-id="i_source_system"' in PAGE
    assert "Standard X-SAMPA" in PAGE
    assert "SynthV legacy X-SAMPA" in PAGE
    assert '"--source-phoneme-system",val("i_source_system")||"auto"' in PAGE


def test_preserve_non_acoustic_models_ui_and_cli_are_wired():
    assert 'id="t_preserve_non_acoustic_models"' in PAGE
    assert "function syncPreserveModelsUI()" in PAGE
    assert 'const acoustic=$("#m_acoustic"),vocoder=$("#m_vocoder")' in PAGE
    assert "Preserving non-acoustic models requires an existing info.nofs" in PAGE
    assert 'a.push("--preserve-original-models")' in PAGE
    assert 'a.push("--custom-vocoder-dnni",replacement)' in PAGE
    assert "The target Vocoder is not loaded" in PAGE
    assert "Replacement Vocoder model (.dnni)" in PAGE
    assert "Train Acoustic from an alternate base" in PAGE
    assert 'data-id="t_custom_acoustic_dnni"' in PAGE
    assert "the target Acoustic is never loaded" in PAGE
    assert 'a.push("--custom-acoustic-dnni",acousticBase)' in PAGE
    root = Path(__file__).resolve().parents[1]
    cli = (root / "opensv_train" / "__main__.py").read_text("utf-8")
    voice = (root / "opensv_train" / "voice.py").read_text("utf-8")
    calibration = (root / "opensv_train" / "calibration.py").read_text("utf-8")
    assert '"--preserve-non-acoustic-models"' in cli
    assert "preserve_non_acoustic_models=arguments.preserve_non_acoustic_models" in cli
    assert 'set(models) & {"acoustic", "vocoder"}' in cli
    assert 'training_models=models' in cli
    assert 'if not arguments.preserve_non_acoustic_models and hasattr(voice, "pitch")' in cli
    assert "training_models: tuple[str, ...]" in voice
    assert "role not in voice.training_models" in voice
    assert 'if "duration" not in voice.training_models' in voice
    assert 'if "pitch" not in voice.training_models' in voice
    assert "Timing and Pitch are not" not in voice
    assert "--preserve-original-models requires --custom-acoustic-dnni" in cli
    assert "--preserve-original-models requires --custom-vocoder-dnni" in cli
    model_tools = (root / "opensv_train" / "model_tools.py").read_text("utf-8")
    assert "_resolve_output_info_path" in model_tools
    assert "refusing to create a models directory under the Windows system/user root" in model_tools
    assert "output package folder (recommended" in cli
    assert "_resolve_output_info_path" in model_tools
    assert "output info.nofs path is too shallow" in model_tools
    assert "output package folder (recommended" in cli
    assert 'commands.add_parser("replace-model"' in cli
    assert 'command.add_argument("--models-root"' in cli
    assert 'command.add_argument("--duration-model"' in cli
    assert 'command.add_argument("--pitch-model-file"' in cli
    assert 'command.add_argument("--vocoder-model"' in cli
    assert "replace_info_models(" in cli
    assert "replace_info_model" in cli
    assert "Acoustic is intentionally excluded" in model_tools
    assert "OpaqueDnniFile.from_bytes(data)" in model_tools
    writer = voice[voice.index("def write_info_voice("):voice.index("\n\nclass Voice(", voice.index("def write_info_voice("))]
    assert "voice.preserve_non_acoustic_models" in writer
    assert "voice.files.preserve_non_acoustic_models" not in writer
    assert 'document[INFO_KINDS[role]] = reference' in voice
    assert "original_bytes = info.models[role].read_bytes()" in voice
    assert "target.write_bytes(original_bytes)" in voice
    assert 'if voice.files.preserve_non_acoustic_models and role not in voice.files.training_models' in calibration
    assert 'if hasattr(voice, "acoustic")' in calibration
    assert 'if hasattr(voice, "pitch")' in calibration


def test_custom_non_acoustic_dnn_is_available_in_training_and_more_tools():
    assert 'data-id="t_custom_duration_dnni"' in PAGE
    assert 'data-id="t_custom_pitch_dnni"' in PAGE
    assert 'data-id="t_custom_vocoder_dnni"' in PAGE
    assert 'data-run="replaceModel"' in PAGE
    assert 'data-id="rm_source"' in PAGE
    assert 'data-id="rm_models_root"' in PAGE
    assert 'data-id="rm_duration_model"' in PAGE
    assert 'data-id="rm_pitch_model"' in PAGE
    assert 'data-id="rm_vocoder_model"' in PAGE
    assert 'data-run="replaceModel"' in PAGE
    assert 'data-id="rm_output"' in PAGE
    assert 'data-id="rm_output" data-kind="path" data-mode="dir"' in PAGE
    assert 'data-label="Output package folder"' in PAGE
    root = Path(__file__).resolve().parents[1]
    cli = (root / "opensv_train" / "__main__.py").read_text("utf-8")
    voice = (root / "opensv_train" / "voice.py").read_text("utf-8")
    model_tools = (root / "opensv_train" / "model_tools.py").read_text("utf-8")
    assert 'commands.add_parser("replace-model"' in cli
    assert 'command.add_argument("--custom-acoustic-dnni"' in cli
    assert 'command.add_argument("--custom-duration-dnni"' in cli
    assert 'command.add_argument("--custom-pitch-dnni"' in cli
    assert 'command.add_argument("--custom-vocoder-dnni"' in cli
    assert "model_overrides=model_overrides" in cli
    assert "custom DNNI overrides cannot be combined with --resume" in cli
    assert "model_overrides: dict[str, str | Path] | None = None" in voice
    assert "preservation mode accepts only an alternate Acoustic base model and a replacement Vocoder" in voice
    assert "output_model_overrides" in voice
    assert "replacement Vocoder" in voice
    assert "replace_info_model" in cli
    assert "Acoustic is intentionally excluded" in model_tools



def test_alternate_acoustic_base_is_wired_as_trained_model():
    root = Path(__file__).resolve().parents[1]
    cli = (root / "opensv_train" / "__main__.py").read_text("utf-8")
    voice = (root / "opensv_train" / "voice.py").read_text("utf-8")
    acoustic = (root / "opensv_train" / "acoustic.py").read_text("utf-8")
    assert '("acoustic", arguments.custom_acoustic_dnni)' in cli
    assert 'command.add_argument("--custom-acoustic-dnni"' in cli
    assert 'if role in model_overrides:' in voice
    assert "Target Acoustic/Vocoder are deliberately never read" in voice
    assert "choose a structurally compatible Acoustic .dnni" in acoustic


def test_i18n_rows_keep_simplified_chinese_and_japanese_columns_separate():
    start = PAGE.index("const I18N_ROWS = [")
    end = PAGE.index("\nconst I18N_MAP", start)
    block = PAGE[start:end]
    rows = re.findall(r'\["((?:\\.|[^"\\])*)","((?:\\.|[^"\\])*)","((?:\\.|[^"\\])*)"\]', block)
    assert len(rows) > 500

    kana = re.compile(r"[\u3040-\u30ff]")
    cjk = re.compile(r"[\u3400-\u9fff]")
    bad_chinese_targets = [zh for _, zh, _ in rows if kana.search(zh)]
    reversed_locale_rows = [
        (en, zh) for en, zh, _ in rows
        if cjk.search(en) and not cjk.search(zh) and re.search(r"[A-Za-z]{3,}", zh)
    ]
    assert not bad_chinese_targets, bad_chinese_targets[:3]
    assert not reversed_locale_rows, reversed_locale_rows[:3]
    assert any(en == "Choose the WAV folder. The filename mode creates LAB labels from space-separated phones; only a final _0/_1 retake suffix is removed." and cjk.search(zh) for en, zh, _ in rows)
