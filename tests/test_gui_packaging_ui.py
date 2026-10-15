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
    assert "使用 SVPK 格式打包" in PAGE
    assert 'id="chooseSfpk"' in PAGE
    assert "使用 SFPK 格式打包（占位）" in PAGE


def test_svpk_dialog_collects_required_metadata_and_assets():
    for field in (
        "svpk_name",
        "svpk_vendor",
        "svpk_version",
        "svpk_profile",
        "svpk_background",
        "svpk_output",
    ):
        assert f'data-id="{field}"' in PAGE
    assert "JSON" in PAGE
    assert "全零外层字段的包在 Sharp 中无响应" in PAGE
    assert "Teto 与 Chifuyu" in PAGE
    assert "可能是包专属签名" in PAGE
    assert "生成诊断包（当前不可安装）" in PAGE
    assert "startJob(\"svpk\")" in PAGE
    assert '"--version-string",version' in PAGE
    assert '"--version-number",String(Number(version))' in PAGE


def test_sfpkg_placeholder_does_not_launch_a_packaging_command():
    helper = PAGE.split("function showSfpkPlaceholder()", 1)[1].split("async function loadResults()", 1)[0]
    assert "info.nofs" in helper
    assert "config.json" in helper
    assert ".dnni" in helper
    assert "SFPK 打包尚未实现" in helper
    assert "startJob(" not in helper


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
