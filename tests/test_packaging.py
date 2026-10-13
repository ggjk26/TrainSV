import json
import zipfile

import pytest

from opensv_train.packaging import package_sfpk, package_svpk


def _file(path, data=b"sample"):
    path.write_bytes(data)
    return path


def test_package_svpk_uses_sample_layout(tmp_path):
    voice = _file(tmp_path / "trained-output.nofs", b"voice-model")
    avatar = _file(tmp_path / "avatar.png", b"png-data")
    background = _file(tmp_path / "backdrop.png", b"png-data")
    license_file = _file(tmp_path / "license.txt", b"License")
    output = tmp_path / "voice.svpk"

    package_svpk(
        voice=voice, output=output, product_name="Kasane Teto",
        vendor_name="AHS Co., Ltd.", version_string="104",
        version_number=104, depending_version_number=0,
        profile=avatar, background=background, license_file=license_file,
    )

    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "voice.nofs", "background.png", "license.txt", "profile.png", "package-profile"
        }
        assert archive.read("voice.nofs") == b"voice-model"
        profile = json.loads(archive.read("package-profile"))
        assert profile == {
            "productName": "Kasane Teto",
            "vendorName": "AHS Co., Ltd.",
            "versionString": "104",
            "versionNumber": 104,
            "dependingVersionNumber": 0,
        }
        assert archive.testzip() is None


def test_package_sfpk_preserves_supplied_config_and_entries(tmp_path):
    info = _file(tmp_path / "info.nofs", b'{"example":"info"}')
    config_obj = {
        "name": "Hanakuma Chifuyu AI",
        "version": "106b1",
        "profile": True,
        "background": True,
        "models": [],
        "model_types": [],
        "phonesets": [],
        "translations": json.dumps({"English": '"Hanakuma Chifuyu AI"', "简体中文": '"花隈千冬 AI"'}, ensure_ascii=False),
    }
    config = _file(tmp_path / "config.json", json.dumps(config_obj, ensure_ascii=False).encode("utf-8"))
    avatar = _file(tmp_path / "avatar.png", b"png-data")
    background = _file(tmp_path / "background-input.png", b"png-data")
    output = tmp_path / "voice.sfpk"

    package_sfpk(info=info, config=config, output=output, profile=avatar, background=background)

    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"info.nofs", "config.json", "profile.png", "background.png"}
        assert json.loads(archive.read("config.json")) == config_obj
        assert archive.read("info.nofs") == b'{"example":"info"}'
        assert archive.testzip() is None


def test_package_sfpk_requires_flags_to_match_assets(tmp_path):
    info = _file(tmp_path / "info.nofs")
    config_obj = {
        "name": "Test", "version": "1", "profile": True, "background": False,
        "models": [], "model_types": [], "phonesets": [], "translations": "{}",
    }
    config = _file(tmp_path / "config.json", json.dumps(config_obj).encode())
    with pytest.raises(ValueError, match="profile flag"):
        package_sfpk(info=info, config=config, output=tmp_path / "bad.sfpk")


def test_package_outputs_require_correct_extension(tmp_path):
    voice = _file(tmp_path / "voice.nofs")
    with pytest.raises(ValueError, match="must end in .svpk"):
        package_svpk(
            voice=voice, output=tmp_path / "voice.zip", product_name="Test",
            vendor_name="Vendor", version_string="1", version_number=1,
        )


def test_package_svpk_can_preserve_observed_background_typo(tmp_path):
    voice = _file(tmp_path / "voice.nofs", b"voice-model")
    background = _file(tmp_path / "background.png", b"png-data")
    output = tmp_path / "voice.svpk"
    package_svpk(
        voice=voice, output=output, product_name="Sample", vendor_name="Vendor",
        version_string="1", version_number=1, background=background,
        background_entry="backgrond.png",
    )
    with zipfile.ZipFile(output) as archive:
        assert "backgrond.png" in archive.namelist()
        assert "background.png" not in archive.namelist()
        assert archive.testzip() is None
