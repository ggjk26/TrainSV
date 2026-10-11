import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HARNESS = Path(os.environ.get("OPENSV_PARITY", REPO / "build/OpenSVParity_artefacts/Release/opensv-parity"))


def run(request: dict, arrays: dict[str, np.ndarray] | None = None) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for name, array in (arrays or {}).items():
            np.save(directory / f"{name}.npy", np.ascontiguousarray(array, dtype="<f4"))
            request = {**request, name: f"{name}.npy"}
        (directory / "request.json").write_text(json.dumps(request))
        subprocess.run([str(HARNESS), str(directory / "request.json"), str(directory / "out")], check=True, capture_output=True, text=True)
        output = {}
        for path in (directory / "out").iterdir():
            output[path.stem] = np.load(path) if path.suffix == ".npy" else json.loads(path.read_text())
        return output
