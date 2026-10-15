import json
import zipfile

import pytest

from opensv_train.packaging import SVPK_HEADER_SIZE, SVPK_PREFIX, build_svpk


def _png(path):
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test image payload")
    return path


def _assets(tmp_path):
    voice = tmp_path / "trained-voice.nofs"
    license_file = tmp_path / "license.txt"
    voice.write_bytes(b"self-trained voice model data")
    license_file.write_text("Custom voicebank license", encoding="utf-8")
    profile = _png(tmp_path / "profile.png")
    background = _png(tmp_path / "background.png")
    return voice, license_file, profile, background


def test_build_svpk_has_expected_header_and_archive(tmp_path):
    voice, license_file, profile, background = _assets(tmp_path)
    output = tmp_path / "voice.svpk"

    result = build_svpk(
        voice=voice,
        license_file=license_file,
        profile=profile,
        background=background,
        output=output,
        product_name="Test Voice",
        vendor_name="TrainSV test",
        version_string="101",
        version_number=101,
    )

    raw = result.read_bytes()
    assert raw[:8] == SVPK_PREFIX
    assert raw[8:SVPK_HEADER_SIZE] == bytes(256)
    assert raw.find(b"PK\x03\x04") == SVPK_HEADER_SIZE

    with zipfile.ZipFile(result) as archive:
        assert archive.namelist() == [
            "voice.nofs", "backgrond.png", "license.txt", "profile.png", "package-profile"
        ]
        assert archive.testzip() is None
        assert archive.read("voice.nofs") == voice.read_bytes()
        assert archive.read("license.txt") == license_file.read_bytes()
        assert json.loads(archive.read("package-profile")) == {
            "productName": "Test Voice",
            "vendorName": "TrainSV test",
            "versionString": "101",
            "versionNumber": 101,
            "dependingVersionNumber": 0,
        }


def test_build_svpk_requires_png_assets(tmp_path):
    voice, license_file, profile, background = _assets(tmp_path)
    background.write_bytes(b"not a png")
    with pytest.raises(ValueError, match="background.png must be a PNG"):
        build_svpk(
            voice=voice,
            license_file=license_file,
            profile=profile,
            background=background,
            output=tmp_path / "voice.svpk",
            product_name="Test Voice",
            vendor_name="TrainSV test",
        )


def test_build_svpk_requires_svpk_extension(tmp_path):
    voice, license_file, profile, background = _assets(tmp_path)
    with pytest.raises(ValueError, match=r"end in .svpk"):
        build_svpk(
            voice=voice,
            license_file=license_file,
            profile=profile,
            background=background,
            output=tmp_path / "voice.zip",
            product_name="Test Voice",
            vendor_name="TrainSV test",
        )



def test_build_svpk_allows_omitting_license_text(tmp_path):
    voice, _license_file, profile, background = _assets(tmp_path)
    output = tmp_path / "voice.svpk"

    build_svpk(
        voice=voice,
        profile=profile,
        background=background,
        output=output,
        product_name="Test Voice",
        vendor_name="TrainSV test",
        version_string="101",
        version_number=101,
    )

    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [
            "voice.nofs", "backgrond.png", "license.txt", "profile.png", "package-profile"
        ]
        assert archive.testzip() is None
        license_text = archive.read("license.txt").decode("utf-8")
        assert "License terms have not been specified" in license_text
        assert "Test Voice" in license_text


@pytest.mark.parametrize("invalid_version", ["101", 1.5, True, -1])
def test_build_svpk_rejects_invalid_version_number(tmp_path, invalid_version):
    voice, license_file, profile, background = _assets(tmp_path)
    with pytest.raises(ValueError, match="version"):
        build_svpk(
            voice=voice,
            license_file=license_file,
            profile=profile,
            background=background,
            output=tmp_path / "voice.svpk",
            product_name="Test Voice",
            vendor_name="TrainSV test",
            version_string="101",
            version_number=invalid_version,
        )


def test_build_svpk_rejects_blank_version_string(tmp_path):
    voice, license_file, profile, background = _assets(tmp_path)
    with pytest.raises(ValueError, match="version string"):
        build_svpk(
            voice=voice,
            license_file=license_file,
            profile=profile,
            background=background,
            output=tmp_path / "voice.svpk",
            product_name="Test Voice",
            vendor_name="TrainSV test",
            version_string=" ",
            version_number=101,
        )
