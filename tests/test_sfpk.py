import json
import zipfile

from opensv_train.sfpk import format_report, inspect_sfpk


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
