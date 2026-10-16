import json
import zipfile

import pytest

from opensv_train.packaging import (SVPK_HEADER_SIZE, SVPK_HEADER_PRESETS, SVPK_PREFIX, _make_diagnostic_header_field, build_svpk)


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
            "voice.nofs", "license.txt", "profile.png", "background.png", "package-config"
        ]
        assert archive.testzip() is None
        assert archive.read("voice.nofs") == voice.read_bytes()
        assert archive.read("license.txt") == license_file.read_bytes()
        assert json.loads(archive.read("package-config")) == {
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
            "voice.nofs", "license.txt", "profile.png", "background.png", "package-config"
        ]
        assert archive.testzip() is None
        license_text = archive.read("license.txt").decode("utf-8")
        assert license_text.startswith("!language en-us\n")
        assert "!language zh-cn\n" in license_text
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


@pytest.mark.parametrize("preset", ["random", "sha256-chain", "aes256-cbc-test"])
def test_nonzero_header_presets_keep_zip_at_offset_264(tmp_path, preset):
    voice, license_file, profile, background = _assets(tmp_path)
    output = tmp_path / f"voice-{preset}.svpk"

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
        header_preset=preset,
    )

    raw = result.read_bytes()
    assert raw[:8] == SVPK_PREFIX
    assert len(raw[8:SVPK_HEADER_SIZE]) == 256
    assert raw[8:SVPK_HEADER_SIZE] != bytes(256)
    assert raw.find(b"PK\x03\x04") == SVPK_HEADER_SIZE
    with zipfile.ZipFile(result) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [
            "voice.nofs", "license.txt", "profile.png", "background.png", "package-config"
        ]


def test_header_preset_helper_has_fixed_field_size_and_validates_names():
    digest = bytes(range(32))
    assert set(SVPK_HEADER_PRESETS) == {"zero", "random", "sha256-chain", "aes256-cbc-test"}

    for preset in SVPK_HEADER_PRESETS:
        field = _make_diagnostic_header_field(preset, digest)
        assert len(field) == 256
        if preset == "zero":
            assert field == bytes(256)
        else:
            assert field != bytes(256)

    sha_a = _make_diagnostic_header_field("sha256-chain", digest)
    sha_b = _make_diagnostic_header_field("sha256-chain", digest)
    aes_a = _make_diagnostic_header_field("aes256-cbc-test", digest)
    aes_b = _make_diagnostic_header_field("aes256-cbc-test", digest)
    assert sha_a == sha_b
    assert aes_a == aes_b


def test_build_svpk_rejects_unknown_header_preset(tmp_path):
    voice, license_file, profile, background = _assets(tmp_path)
    with pytest.raises(ValueError, match="unknown SVPK header preset"):
        build_svpk(
            voice=voice,
            license_file=license_file,
            profile=profile,
            background=background,
            output=tmp_path / "voice.svpk",
            product_name="Test Voice",
            vendor_name="TrainSV test",
            header_preset="official-signature",
        )


def test_header_presets_change_only_the_outer_opaque_field(tmp_path):
    voice, _license_file, profile, background = _assets(tmp_path)
    outputs = {}
    for preset in ("zero", "random", "sha256-chain", "aes256-cbc-test"):
        outputs[preset] = build_svpk(
            voice=voice,
            profile=profile,
            background=background,
            output=tmp_path / f"same-body-{preset}.svpk",
            product_name="Same Body",
            vendor_name="TrainSV test",
            version_string="101",
            version_number=101,
            header_preset=preset,
        ).read_bytes()

    bodies = {package[SVPK_HEADER_SIZE:] for package in outputs.values()}
    assert len(bodies) == 1
    assert all(package[:8] == SVPK_PREFIX for package in outputs.values())
    assert all(package[SVPK_HEADER_SIZE:SVPK_HEADER_SIZE + 4] == b"PK\x03\x04" for package in outputs.values())
