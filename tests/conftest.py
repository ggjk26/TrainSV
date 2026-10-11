import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_VOICES = [
    "SET_ME_voice.nofs",
    "SET_ME_info.nofs",
]


def voice_paths():
    configured = os.environ.get("OPENSV_TEST_VOICES")
    paths = configured.split(os.pathsep) if configured else DEFAULT_VOICES
    return [str(Path(path).expanduser()) for path in paths if Path(path).expanduser().exists()]


def pytest_generate_tests(metafunc):
    if "voice_path" in metafunc.fixturenames:
        paths = voice_paths()
        if not paths:
            pytest.skip("no test voices found; set OPENSV_TEST_VOICES")
        metafunc.parametrize("voice_path", paths, ids=[Path(path).parent.name for path in paths])
