import json
import zipfile

from opensv_train.sfpk import format_report, inspect_sfpk
from opensv_train.diagnostics import diagnose_voice


def test_inspects_zip_without_extracting(tmp_path):
    sample = tmp_path / "sample.sfpk"
    with zipfile.ZipFile(sample, "w") as archive:
        archive.writestr("voice.nofs", b"model")
        archive.writestr("avatar.png", b"image")
        archive.writestr("voice.json", json.dumps({"name": "Example", "version": "100"}))
    report = inspect_sfpk(sample)
    assert report["container_type"] == "zip-container"
    assert {item["path"] for item in report["model_candidates"]} == {"voice.nofs"}
    assert {item["path"] for item in report["image_candidates"]} == {"avatar.png"}
    config = report["configuration_candidates"][0]
    assert config["parsed"] is True
    assert config["scalar_fields"]["name"] == "Example"
    assert report["packaging_supported"] is False
    assert "does not prove" in " ".join(report["warnings"])
    assert "Container: zip-container" in format_report(report)


def test_inspects_extracted_directory(tmp_path):
    root = tmp_path / "extracted"
    root.mkdir()
    (root / "info.json").write_text('{"name":"Sample","nested":{"ignored":true}}', encoding="utf-8")
    (root / "avatar.jpg").write_bytes(b"image")
    report = inspect_sfpk(root)
    assert report["container_type"] == "extracted-directory"
    assert report["entry_count"] == 2
    assert report["configuration_candidates"][0]["top_level_keys"] == ["name", "nested"]


def test_missing_path_is_reported(tmp_path):
    report = inspect_sfpk(tmp_path / "missing.sfpk")
    assert report["exists"] is False
    assert report["warnings"] == ["Path does not exist."]


def test_architecture_diagnostic_reports_missing_input(tmp_path):
    report = diagnose_voice(tmp_path / "missing.nofs")
    assert report["exists"] is False
    assert report["container_parsed"] is False
    assert report["trainability"] == "unknown"
    assert report["warnings"] == ["Input path is not a file."]


def test_inspects_json_info_nofs_model_references_without_assuming_payloads(tmp_path):
    sample = tmp_path / "sample.sfpk"
    acoustic_id = "a" * 32
    vocoder_id = "b" * 32
    duration_id = "c" * 32
    f0_id = "d" * 32
    info = {
        "name": "Sample AI",
        "version": "1.0",
        "vendor": "Example Vendor",
        "language": "japanese",
        "phoneset": "romaji",
        "support_languages": ["japanese"],
        "acoustic": acoustic_id,
        "vocoder": vocoder_id,
        "duration": duration_id,
        "f0": f0_id,
        "styles": [{"name": "(base)", "data": "00"}],
        "f0_style": "00",
        "duration_style": "00",
    }
    config = {
        "name": "Sample AI",
        "version": "1.0",
        "profile": True,
        "background": True,
        "models": [],
        "model_types": [],
        "phonesets": [],
        "translations": "{}",
    }
    with zipfile.ZipFile(sample, "w") as archive:
        archive.writestr("info.nofs", json.dumps(info))
        archive.writestr("config.json", json.dumps(config))
        archive.writestr(f"models/acoustic/{acoustic_id}.dnni", b"model")
    report = inspect_sfpk(sample)
    summary = report["info_nofs_candidates"][0]
    assert summary["voice_metadata"]["name"] == "Sample AI"
    assert summary["style_names"] == ["(base)"]
    config_check = report["config_asset_checks"][0]
    assert config_check["asset_checks"]["profile"]["present_in_container"] is False
    assert config_check["asset_checks"]["background"]["present_in_container"] is False
    assert any("profile=true" in warning for warning in report["warnings"])
    assert any("background=true" in warning for warning in report["warnings"])
    assert summary["model_references"]["acoustic"] == acoustic_id
    assert summary["model_payload_found_in_container"]["acoustic"] is True
    assert summary["model_payload_found_in_container"]["vocoder"] is False
    assert set(summary["model_payloads_not_found_in_container"]) == {"vocoder", "duration", "f0"}
    assert any("model payload files were not found" in item for item in report["warnings"])
    formatted = format_report(report)
    assert "info.nofs" in formatted
    assert f"model acoustic: {acoustic_id}" in formatted


def test_inspects_local_model_store_and_does_not_substitute_models(tmp_path):
    archive = tmp_path / "sample.sfpk"
    models_root = tmp_path / "models"
    (models_root / "acoustic").mkdir(parents=True)
    (models_root / "acoustic" / ("e" * 32 + ".dnni")).write_bytes(b"not dnni")
    (models_root / "vocoder").mkdir()
    (models_root / "vocoder" / "candidate.dnni").write_bytes(b"not dnni")
    info = {
        "name": "Sample AI", "version": "1", "vendor": "Example",
        "language": "japanese", "phoneset": "romaji",
        "support_languages": ["japanese"],
        "acoustic": "a" * 32, "vocoder": "b" * 32,
        "duration": "c" * 32, "f0": "d" * 32,
        "styles": [{"name": "(base)", "data": "00"}],
        "f0_style": "00", "duration_style": "00",
    }
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("info.nofs", json.dumps(info))
    report = inspect_sfpk(archive, models_root=models_root)
    inventory = report["model_store_inventory"]
    assert inventory["exists"] is True
    assert inventory["dnni_file_count"] == 2
    assert len(inventory["files_by_role"]["acoustic"]) == 1
    refs = report["model_store_reference_checks"][0]["models"]
    assert refs["acoustic"]["expected_path_found"] is False
    assert refs["acoustic"]["same_id_found_elsewhere"] == []
    assert refs["acoustic"]["role_candidate_count"] == 1
    assert "must not be used for automatic substitution" in inventory["note"]
    formatted = format_report(report)
    assert "Installed model store:" in formatted
    assert "available same-role models are listed as candidates only" in formatted
