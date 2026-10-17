"""Regression checks for the Results-stage package UI."""

from __future__ import annotations

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
    assert "生成 SVPK 诊断包（当前不可安装）" in PAGE
    assert 'id="chooseSfpk"' in PAGE
    assert "使用 SFPK 格式打包（实验）" in PAGE


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
    assert "复制 Chifuyu 字段后提示数据损坏" in PAGE
    assert "完整性/认证校验" in PAGE
    assert "生成诊断包（当前不可安装）" in PAGE
    assert "startJob(\"svpk\")" in PAGE
    assert '"--version-string",version' in PAGE
    assert '"--version-number",String(Number(version))' in PAGE
    assert '"--header-preset",val("svpk_header_preset")||"zero"' in PAGE
    assert "aes256-cbc-test|AES-256-CBC 合成测试（非官方）" in PAGE
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
    assert 'data-ext="png" data-label="头像图片 profile.png（可留空）"' in PAGE
    assert 'data-ext="png" data-label="背景图片 background.png（可留空）"' in PAGE
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
    assert 'if(target.isfile&&!confirm("输出文件已存在：\\n"+val("svpk_output")+"\\n\\n是否覆盖它？"))return;' in PAGE


def test_training_and_evaluation_expose_f0_tracking_profiles():
    assert 'data-id="t_f0_profile"' in PAGE
    assert 'data-options="standard|Standard (65–1100 Hz, threshold 0.60);vocaloid|VOCALOID / synthetic singing (45–1400 Hz, threshold 0.45)"' in PAGE
    assert 'a.push("--f0-profile",val("t_f0_profile")||"standard")' in PAGE
    assert "F0 追踪预设只影响训练/评估时从 WAV 提取基频的方式" in PAGE
    assert "没有 note_seq/音符标注，Pitch 模型无法训练" in PAGE


def test_training_pitch_workspace_exposes_full_f0_and_pretrained_controls():
    assert 'id="pitchTrainingPanel"' in PAGE
    for field in (
        "t_pitch_source",
        "t_pitch_model",
        "t_f0_profile",
        "t_f0_floor",
        "t_f0_ceiling",
        "t_f0_threshold",
        "t_f0_smoothing",
        "t_f0_correction",
        "t_pitch_smoothing",
        "t_pitch_melody_weight",
        "t_pitch_motion_weight",
    ):
        assert f'data-id="{field}"' in PAGE
    assert 'value="low|低音（45–650 Hz，阈值 0.60）"' in PAGE or "low|低音（45–650 Hz，阈值 0.60）" in PAGE
    assert "high|高音（100–1400 Hz，阈值 0.60）" in PAGE
    assert "wide|宽音域（40–1600 Hz，阈值 0.60）" in PAGE
    assert 'a.push("--pitch-base-model",val("t_pitch_model"))' in PAGE
    assert '["--f0-reference-cents","t_f0_correction"]' in PAGE
    assert '["--pitch-smoothing-hz","t_pitch_smoothing"]' in PAGE
    assert '["--pitch-melody-weight","t_pitch_melody_weight"]' in PAGE
    assert '["--pitch-motion-weight","t_pitch_motion_weight"]' in PAGE
    assert 'pitch(){return this.finetune(true)}' in PAGE
