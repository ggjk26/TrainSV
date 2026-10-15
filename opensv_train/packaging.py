"""Build experimental SVPK v1 packages for TrainSV-finetuned voice banks.

The SVPK samples inspected so far have an eight-byte magic/version prefix,
a 256-byte opaque field, then a ZIP archive. The field's official semantics
are unknown; for the custom-bank/Sharp test workflow we write zeroes there.
This does not produce a Dreamtonics-signed official installer.
"""
from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

SVPK_PREFIX = b"SVPK\x00\x01\x00\x00"
SVPK_OPAQUE_FIELD_SIZE = 256
SVPK_HEADER_SIZE = len(SVPK_PREFIX) + SVPK_OPAQUE_FIELD_SIZE
_ARCHIVE_MEMBERS = ("voice.nofs", "license.txt", "profile.png", "background.png", "package-config")


def build_svpk(
    *,
    voice: str | Path,
    license_file: str | Path | None = None,
    profile: str | Path,
    background: str | Path,
    output: str | Path,
    product_name: str,
    vendor_name: str,
    version_string: str = "101",
    version_number: int = 101,
    depending_version_number: int = 0,
) -> Path:
    """Build a custom-bank SVPK container for testing in Synthesizer V Sharp.

    This writes the observed SVPK v1 outer layout, but zero-fills the
    256-byte opaque header field. Runtime acceptance must be tested in Sharp.
    """
    voice = Path(voice).expanduser()
    license_file = Path(license_file).expanduser() if license_file is not None else None
    profile = Path(profile).expanduser()
    background = Path(background).expanduser()
    output = Path(output).expanduser()

    for label, path in (
        ("voice.nofs", voice),
        ("profile.png", profile),
        ("background.png", background),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if license_file is not None and not license_file.is_file():
        raise FileNotFoundError(f"license.txt file not found: {license_file}")

    for label, path in (("profile.png", profile), ("background.png", background)):
        with path.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"{label} must be a PNG file: {path}")

    if not product_name.strip():
        raise ValueError("product name must not be empty")
    if not vendor_name.strip():
        raise ValueError("vendor name must not be empty")
    if not str(version_string).strip():
        raise ValueError("version string must not be empty")
    if (
        not isinstance(version_number, int)
        or isinstance(version_number, bool)
        or not isinstance(depending_version_number, int)
        or isinstance(depending_version_number, bool)
    ):
        raise ValueError("version numbers must be integers")
    if version_number < 0 or depending_version_number < 0:
        raise ValueError("version numbers must be non-negative integers")
    if output.suffix.lower() != ".svpk":
        raise ValueError("output filename must end in .svpk")

    config = {
        "productName": product_name,
        "vendorName": vendor_name,
        "versionString": str(version_string),
        "versionNumber": int(version_number),
        "dependingVersionNumber": int(depending_version_number),
    }
    # Observed SVPK licenses start with a !language directive. Keep generated
    # text explicitly marked as a placeholder, while emitting English and
    # Simplified Chinese sections in the same directive-based format.
    generated_license = (
        "!language en-us\n"
        f"Voicebank: {product_name}\n"
        f"Vendor: {vendor_name}\n"
        f"Version: {version_string}\n\n"
        "License terms have not been specified in this package.\n"
        "Please contact the rights holder before redistribution or commercial use.\n\n"
        "!language zh-cn\n"
        f"声库名称：{product_name}\n"
        f"作者/供应商：{vendor_name}\n"
        f"版本：{version_string}\n\n"
        "此实验声库尚未指定正式许可条款。\n"
        "重新分发或商业使用前，请先联系权利人确认授权。\n"
    ).encode("utf-8")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="trainsv-svpk-") as temp_dir:
        temp_dir = Path(temp_dir)
        inner_zip = temp_dir / "package.zip"
        temp_output = output.with_name(output.name + ".tmp")

        with zipfile.ZipFile(
            inner_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            # Match the entry order of the official sample supplied for comparison.
            entries = [("voice.nofs", voice)]
            if license_file is not None:
                entries.append(("license.txt", license_file))
            else:
                entries.append(("license.txt", generated_license))
            entries.extend((("profile.png", profile), ("background.png", background)))
            for archive_name, source in entries:
                if isinstance(source, Path):
                    archive.write(source, arcname=archive_name)
                else:
                    archive.writestr(archive_name, source)
            archive.writestr(
                "package-config",
                json.dumps(config, ensure_ascii=False, indent=2).replace("\n", "\r\n"),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )

        try:
            with temp_output.open("wb") as result, inner_zip.open("rb") as source:
                result.write(SVPK_PREFIX)
                result.write(bytes(SVPK_OPAQUE_FIELD_SIZE))
                while chunk := source.read(1024 * 1024):
                    result.write(chunk)

            # Validate the inner ZIP and metadata round-trip before publishing.
            with zipfile.ZipFile(temp_output) as archive:
                expected_members = list(_ARCHIVE_MEMBERS)
                if archive.namelist() != expected_members:
                    raise ValueError(f"unexpected SVPK members: {archive.namelist()}")
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ValueError(f"corrupt ZIP member: {bad_member}")
                if json.loads(archive.read("package-config")) != config:
                    raise ValueError("package-config failed validation")

            os.replace(temp_output, output)
        finally:
            temp_output.unlink(missing_ok=True)

    return output
