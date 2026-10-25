#!/usr/bin/env python3
"""TrainSV GUI - a point-and-click front end for the `trainsv` command line.

Run it from the TrainSV folder, with the same Python you installed TrainSV into:

    python trainsv_gui.py

It starts a small web page on your own computer (nothing is uploaded anywhere)
and opens it in your browser. Every button runs the real `trainsv` command and
shows you the exact command it used, so you can also copy it and run it by hand.

Options:  --port 8765   --no-browser   --project /path/to/TrainSV_1.0
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import heapq
import json
import os
import platform
import secrets
import string
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from opensv_train.audit import audit_dataset

ROOT = Path(__file__).resolve().parent
TOKEN = secrets.token_urlsafe(16)
COMMANDS = {"import-lab", "convert-dict", "evaluate", "finetune", "synthesize-dataset", "inspect", "package-svpk", "package-sfpk", "replace-model", "audio-to-midi"}
MAX_LOG_LINES = 40000


# --------------------------------------------------------------------------- jobs
class Runner:
    """Runs one `trainsv` command at a time and keeps its output."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.reset()

    def reset(self):
        self.lines, self.base = [], 0
        self.status, self.label, self.command = "idle", "", ""
        self.started = self.ended = 0.0
        self.code = None
        self.stopping = False

    def start(self, command: str, args: list, label: str):
        if command not in COMMANDS:
            raise ValueError(f"unknown command: {command}")
        with self.lock:
            if self.status == "running":
                raise RuntimeError("Something is already running. Stop it first.")
            self.reset()
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            full = [sys.executable, "-u", "-m", "opensv_train", command, *[str(a) for a in args]]
            self.command, self.label, self.status, self.started = " ".join(full[4:]), label, "running", time.time()
            try:
                self.proc = subprocess.Popen(full, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            except OSError as error:
                self.status, self.ended = "failed", time.time()
                self.lines.append(f"Could not start Python: {error}")
                return
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        proc = self.proc
        for line in proc.stdout:
            with self.lock:
                self.lines.append(line.rstrip("\r\n"))
                if len(self.lines) > MAX_LOG_LINES:
                    drop = MAX_LOG_LINES // 4
                    del self.lines[:drop]
                    self.base += drop
        code = proc.wait()
        with self.lock:
            self.code, self.ended = code, time.time()
            self.status = "stopped" if self.stopping else "done" if code == 0 else "failed"

    def stop(self):
        with self.lock:
            proc = self.proc
            if self.status != "running" or proc is None:
                return
            self.stopping = True
        proc.terminate()
        threading.Timer(8.0, lambda: proc.poll() is None and proc.kill()).start()

    def snapshot(self, since: int):
        with self.lock:
            start = max(since, self.base)
            end = time.time() if self.status == "running" else self.ended
            return {"status": self.status, "label": self.label, "command": self.command, "code": self.code,
                    "elapsed": max(0.0, end - self.started) if self.started else 0.0,
                    "lines": self.lines[start - self.base:], "next": self.base + len(self.lines)}


RUNNER = Runner()


# --------------------------------------------------------------------------- helpers
def settings_path() -> Path:
    return ROOT / ".trainsv_gui.json"


def read_settings() -> dict:
    try:
        return json.loads(settings_path().read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def write_settings(data: dict) -> None:
    try:
        settings_path().write_text(json.dumps(data, indent=1), "utf-8")
    except OSError:
        pass


def env_info() -> dict:
    modules = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "torch", "soundfile", "parselmouth")}
    return {"root": str(ROOT), "home": str(Path.home()), "python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.system(), "sep": os.sep, "modules": modules,
            "package": (ROOT / "opensv_train" / "__main__.py").is_file()}


_DEVICE = None


def device_info() -> dict:
    global _DEVICE
    if _DEVICE is None:
        code = ("import torch\n"
                "if torch.cuda.is_available(): print('cuda|' + torch.cuda.get_device_name(0))\n"
                "elif torch.backends.mps.is_available(): print('mps|Apple GPU')\n"
                "else: print('cpu|CPU only')\n")
        try:
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
            if out.returncode == 0 and "|" in out.stdout:
                kind, name = out.stdout.strip().splitlines()[-1].split("|", 1)
                _DEVICE = {"ok": True, "kind": kind, "name": name}
            else:
                _DEVICE = {"ok": False, "error": (out.stderr.strip().splitlines() or ["PyTorch could not be loaded"])[-1]}
        except (OSError, subprocess.SubprocessError) as error:
            _DEVICE = {"ok": False, "error": str(error)}
    return _DEVICE


def browse(path: str, want_files: bool, exts: set) -> dict:
    p = Path(path).expanduser() if path else Path.home()
    while not p.exists() and p != p.parent:
        p = p.parent
    if p.is_file():
        p = p.parent
    entries, error = [], None
    try:
        # Keep directory listing memory bounded for folders with very large
        # recording collections while preserving the 3000-entry UI limit.
        def visible_entries():
            for entry in p.iterdir():
                if entry.name.startswith("."):
                    continue
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    continue
                matches_file = want_files and (not exts or entry.suffix.lower().lstrip(".") in exts)
                if is_dir or matches_file:
                    yield entry, is_dir

        selected = heapq.nsmallest(
            3000, visible_entries(), key=lambda item: (not item[1], item[0].name.lower())
        )
        entries = [{"name": entry.name, "dir": is_dir} for entry, is_dir in selected]
    except PermissionError:
        error = "No permission to open this folder."
    roots = [f"{c}:\\" for c in string.ascii_uppercase if Path(f"{c}:\\").exists()] if os.name == "nt" else ["/"]
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "entries": entries, "roots": roots,
            "home": str(Path.home()), "project": str(ROOT), "sep": os.sep, "error": error}


def check_dataset(path: str) -> dict:
    d = Path(path).expanduser()
    out = {"exists": d.is_dir()}
    if not out["exists"]:
        return out
    csv_path, wavs = d / "transcriptions.csv", d / "wavs"
    out.update(has_csv=csv_path.is_file(), has_wavs=wavs.is_dir())
    if csv_path.is_file():
        rows = notes = missing = 0
        seconds = 0.0
        styles = set()
        # Scan the WAV directory once rather than issuing a filesystem query
        # for every CSV row; normcase preserves the host OS's filename rules.
        try:
            wav_names = (
                {os.path.normcase(item.stem) for item in wavs.glob("*.wav") if item.exists()}
                if out["has_wavs"] else set()
            )
        except OSError as error:
            out["error"] = f"Could not inspect WAV files: {error}"
            return out
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                for row in reader:
                    rows += 1
                    if (row.get("note_seq") or "").strip():
                        notes += 1
                    try:
                        seconds += sum(float(x) for x in (row.get("ph_dur") or "").split())
                    except ValueError:
                        pass
                    name = row.get("name") or ""
                    if out["has_wavs"] and os.path.normcase(name) not in wav_names:
                        missing += 1
                    if (row.get("style") or "").strip():
                        styles.add(row["style"].strip())
            out.update(rows=rows, notes=notes, seconds=seconds, missing=missing, styles=sorted(styles), columns=columns)
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            out["error"] = f"Could not read transcriptions.csv: {error}"
    return out


def folder_info(path: str) -> dict:
    d = Path(path).expanduser()
    if not d.is_dir():
        return {"exists": False}

    def count(ext, subs):
        # Count lazily to avoid allocating every path in large corpora.
        total = sum(1 for _ in d.glob(f"*.{ext}"))
        for sub in subs:
            child = d / sub
            if child.is_dir():
                total += sum(1 for _ in child.glob(f"*.{ext}"))
        return total

    return {"exists": True, "wav": count("wav", ("wav", "wavs")), "lab": count("lab", ("lab", "labs"))}


def resolve_voice(path: str) -> dict:
    p = Path(path).expanduser()
    if p.is_file():
        return {"path": str(p)}
    if p.is_dir():
        found = sorted(p.glob("info.*.nofs"), key=lambda x: x.stat().st_mtime)
        if found:
            return {"path": str(found[-1])}
        if (p / "voice.nofs").is_file():
            return {"path": str(p / "voice.nofs")}
    return {"error": "No voice file found there yet. Train first, or pick the file yourself."}


def list_results(path: str) -> dict:
    d = Path(path).expanduser()
    if not d.is_dir():
        return {"exists": False}
    metrics = {}
    if (d / "metrics.json").is_file():
        try:
            metrics = json.loads((d / "metrics.json").read_text("utf-8"))
        except (OSError, ValueError):
            pass
    timbre_diagnostics = {}
    if (d / "timbre_diagnostics.json").is_file():
        try:
            timbre_diagnostics = json.loads((d / "timbre_diagnostics.json").read_text("utf-8"))
        except (OSError, ValueError):
            pass
    phrases = []
    for sub in sorted(x for x in d.iterdir() if x.is_dir()):
        wavs = sorted(sub.glob("*.wav"), key=lambda w: (w.stem != "recording", w.name))
        if wavs:
            phrases.append({"name": sub.name, "files": [{"label": w.stem, "path": str(w)} for w in wavs]})
    return {"exists": True, "metrics": metrics, "phrases": phrases, "timbre_diagnostics": timbre_diagnostics}


# --------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "TrainSVGUI"

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _host_ok(self):
        return self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]") in ("127.0.0.1", "localhost")

    def _authorized(self, query):
        supplied = self.headers.get("X-Token") or query.get("token", [""])[0]
        return self._host_ok() and secrets.compare_digest(supplied, TOKEN)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            if not self._host_ok():
                return self._send(403, "Forbidden", "text/plain")
            return self._send(200, PAGE.replace("__TOKEN__", TOKEN), "text/html; charset=utf-8")
        if not url.path.startswith("/api/") or not self._authorized(query):
            return self._send(403, {"error": "Session expired. Reload the page."})
        one = lambda key, default="": query.get(key, [default])[0]
        try:
            route = url.path[5:]
            if route == "env":
                return self._send(200, env_info())
            if route == "device":
                return self._send(200, device_info())
            if route == "browse":
                exts = {e for e in one("ext").lower().split(",") if e}
                return self._send(200, browse(one("path"), one("files") == "1", exts))
            if route == "check-dataset":
                return self._send(200, check_dataset(one("path")))
            if route == "audit-dataset":
                return self._send(200, audit_dataset(one("path")))
            if route == "folder-info":
                return self._send(200, folder_info(one("path")))
            if route == "resolve-voice":
                return self._send(200, resolve_voice(one("path")))
            if route == "stat":
                p = Path(one("path")).expanduser()
                return self._send(200, {"exists": p.exists(), "isdir": p.is_dir(), "isfile": p.is_file()})
            if route == "job":
                return self._send(200, RUNNER.snapshot(int(one("since", "0") or 0)))
            if route == "results":
                return self._send(200, list_results(one("path")))
            if route == "settings":
                return self._send(200, read_settings())
            if route == "media":
                p = Path(one("path")).expanduser()
                if p.suffix.lower() != ".wav" or not p.is_file():
                    return self._send(404, {"error": "Not found"})
                return self._send(200, p.read_bytes(), "audio/wav")
            return self._send(404, {"error": "Unknown request"})
        except Exception as error:  # noqa: BLE001 - report anything to the page
            return self._send(500, {"error": str(error)})

    def do_POST(self):
        url = urlparse(self.path)
        if not url.path.startswith("/api/") or not self._authorized({}):
            return self._send(403, {"error": "Session expired. Reload the page."})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            route = url.path[5:]
            if route == "run":
                RUNNER.start(str(body.get("command", "")), list(body.get("args", [])), str(body.get("label", "")))
                return self._send(200, {"ok": True})
            if route == "stop":
                RUNNER.stop()
                return self._send(200, {"ok": True})
            if route == "mkdir":
                Path(body["path"]).expanduser().mkdir(parents=True, exist_ok=True)
                return self._send(200, {"ok": True})
            if route == "settings":
                write_settings(body)
                return self._send(200, {"ok": True})
            if route == "quit":
                RUNNER.stop()
                self._send(200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            return self._send(404, {"error": "Unknown request"})
        except Exception as error:  # noqa: BLE001
            return self._send(400, {"error": str(error)})


def main():
    global ROOT
    parser = argparse.ArgumentParser(description="TrainSV GUI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--project", help="the TrainSV folder (default: the folder this file is in)")
    args = parser.parse_args()
    if args.project:
        ROOT = Path(args.project).expanduser().resolve()
    server = None
    for port in range(args.port, args.port + 25):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("Could not find a free port. Try --port 9000.")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"TrainSV GUI is running at {url}\nPress Ctrl+C here (or the Quit button in the page) to stop it.", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RUNNER.stop()
        server.server_close()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrainSV</title>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--ink:#18212b;--mute:#5b6773;--line:#d8dee5;--accent:#0d6f7a;--accent-ink:#fff;--accent-soft:#e1f1f2;--ok:#2b7a3d;--warn:#8f6200;--bad:#b3261e;--code:#eef1f4}
@media(prefers-color-scheme:dark){:root{--bg:#12171c;--panel:#1a2128;--ink:#e6ebf0;--mute:#9aa7b3;--line:#2c3640;--accent:#4cb8c4;--accent-ink:#06262a;--accent-soft:#16343a;--ok:#6cc183;--warn:#e0b34d;--bad:#ff8a80;--code:#222b33}}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;display:flex;flex-direction:column}
code,pre,.mono{font-family:ui-monospace,"Cascadia Mono",Menlo,Consolas,monospace;font-size:.88em}
code{background:var(--code);padding:1px 5px;border-radius:4px}
button,input,select,textarea{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.top{display:flex;align-items:center;gap:14px;padding:10px 20px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
.top h1{margin:0;font-size:1.15rem;letter-spacing:.2px}
.top .sub{color:var(--mute)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto}
.lang-switch{display:flex;align-items:center;gap:6px;color:var(--mute);font-size:.86rem}.lang-switch label{white-space:nowrap}#uiLanguage{width:auto;min-width:118px;padding:4px 8px;background:var(--panel)}
.chip{display:inline-flex;align-items:center;gap:6px;padding:2px 10px;border-radius:99px;border:1px solid var(--line);font-size:.85rem;color:var(--mute);background:var(--panel)}
.chip.ok{color:var(--ok);border-color:currentColor}.chip.bad{color:var(--bad);border-color:currentColor}.chip.warn{color:var(--warn);border-color:currentColor}
.banner{margin:0;padding:10px 20px;background:var(--accent-soft);border-bottom:1px solid var(--line)}
.banner.bad{color:var(--bad)}
.shell{flex:1;min-height:0;display:grid;grid-template-columns:230px 1fr}
.rail{padding:22px 14px 22px 20px;border-right:1px solid var(--line);overflow:auto}
.rail ol{list-style:none;margin:0 0 20px 14px;padding:0;border-left:2px solid var(--line)}
.rail li{position:relative}
.rail button.nav{all:unset;box-sizing:border-box;display:block;width:100%;cursor:pointer;padding:9px 8px 9px 22px;border-radius:6px;color:var(--mute)}
.rail button.nav:hover{background:var(--accent-soft);color:var(--ink)}
.rail button.nav[aria-current=page]{color:var(--ink);font-weight:600}
.rail .dot{position:absolute;left:-13px;top:8px;width:24px;height:24px;border-radius:50%;background:var(--panel);border:2px solid var(--line);display:grid;place-items:center;font-size:.78rem;font-weight:600;color:var(--mute)}
.rail li.current .dot{border-color:var(--accent);color:var(--accent)}
.rail li.done .dot{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.rail small{display:block;font-weight:400;color:var(--mute);font-size:.8rem}
.rail .more{margin-left:0;border:0}.rail .more button.nav{padding-left:8px}
main{display:flex;flex-direction:column;min-width:0;min-height:0}
.pages{flex:1;overflow:auto;padding:26px 32px 40px}
.page{max-width:760px;display:none}.page.show{display:block}
h2{margin:0 0 4px;font-size:1.5rem}
h3{margin:28px 0 8px;font-size:1.05rem}
.lead{margin:0 0 22px;color:var(--mute);max-width:62ch}
.field{margin:0 0 16px}
.field label,legend{display:block;font-weight:600;margin-bottom:4px;padding:0}
.field-label-row{display:flex;align-items:center;gap:5px;margin-bottom:4px}
.field-label-row>label{display:block;font-weight:600;margin:0;padding:0}
.help-wrap{position:relative;display:inline-flex;vertical-align:middle;flex:0 0 auto;margin-left:3px}
button.help-btn{display:inline-grid;place-items:center;width:18px;height:18px;min-width:18px;padding:0;border:1px solid var(--mute);border-radius:50%;background:transparent;color:var(--accent);font-size:12px;font-weight:700;line-height:1}
button.help-btn:hover{background:var(--accent-soft);border-color:var(--accent)}
.help-pop{position:absolute;z-index:50;left:0;top:calc(100% + 7px);display:block;width:min(320px,76vw);padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--panel);box-shadow:0 8px 24px rgba(0,0,0,.18);color:var(--ink);font-size:.86rem;font-weight:400;line-height:1.5;white-space:normal}
.help-pop[hidden]{display:none!important}
label.choice .help-wrap{margin-left:5px;vertical-align:baseline}
@media(max-width:540px){.help-pop{width:min(280px,78vw)}}
.field small,.hint{display:block;color:var(--mute);margin-top:3px;font-size:.86rem;max-width:66ch}
input[type=text],select,textarea{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel)}
textarea{min-height:60px;resize:vertical}
.path{display:flex;gap:8px}.path input{flex:1;min-width:0}
.status{margin-top:4px;font-size:.88rem}.status.ok{color:var(--ok)}.status.warn{color:var(--warn)}.status.bad{color:var(--bad)}
button{cursor:pointer;border-radius:6px;border:1px solid var(--line);background:var(--panel);padding:8px 14px}
button:hover:not(:disabled){border-color:var(--accent)}
button:disabled{opacity:.5;cursor:not-allowed}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:600;padding:10px 18px}
button.danger{color:var(--bad);border-color:var(--bad)}
button.small{padding:3px 10px;font-size:.85rem}
button.ghost{background:transparent}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0 6px;align-items:center}
.needs{color:var(--warn);font-size:.9rem;margin:6px 0}.needs ul{margin:2px 0 0;padding-left:20px}
details.cmd{margin:6px 0 0;color:var(--mute);font-size:.9rem}
details.cmd pre{white-space:pre-wrap;word-break:break-all;background:var(--code);padding:10px;border-radius:6px;margin:6px 0;color:var(--ink)}
fieldset{border:1px solid var(--line);border-radius:8px;margin:0 0 18px;padding:12px 16px 6px;background:var(--panel)}
fieldset legend{padding:0 6px}
label.choice{display:flex;gap:10px;align-items:flex-start;padding:7px 0;cursor:pointer}
label.choice input{margin-top:5px}
label.choice b{font-weight:600}
label.choice small{display:block;color:var(--mute);font-size:.86rem}
.seg{display:flex;gap:10px;margin-bottom:18px;flex-wrap:wrap}
.seg label{flex:1;min-width:230px;border:1px solid var(--line);border-radius:8px;padding:10px 14px;cursor:pointer;background:var(--panel);display:flex;gap:10px}
.seg label:has(input:checked){border-color:var(--accent);background:var(--accent-soft)}
.seg small{display:block;color:var(--mute);font-size:.86rem}
details.adv{margin:8px 0 0;border-top:1px solid var(--line);padding-top:10px}
details.adv>summary{cursor:pointer;font-weight:600;padding:4px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:0 18px}
.note{background:var(--accent-soft);border-radius:6px;padding:8px 12px;margin:0 0 14px}
table.t{border-collapse:collapse;width:100%;margin:8px 0;background:var(--panel)}
table.t th,table.t td{border:1px solid var(--line);padding:6px 10px;text-align:left}
table.t td.n{font-variant-numeric:tabular-nums}
.phrase{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:10px 0;background:var(--panel)}
.phrase h4{margin:0 0 6px;font-size:.95rem}
.players{display:flex;flex-wrap:wrap;gap:14px}
.players figure{margin:0}.players figcaption{font-size:.85rem;color:var(--mute)}
.players audio{height:34px;max-width:260px}
/* activity dock */
#activity{border-top:1px solid var(--line);background:var(--panel);max-height:48vh;display:flex;flex-direction:column}
#activity[hidden]{display:none}
.act-head{display:flex;gap:12px;align-items:center;padding:10px 20px;flex-wrap:wrap}
.act-head .grow{flex:1;min-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--mute)}
.act-body{padding:0 20px 14px;overflow:auto}
#activity.collapsed .act-body{display:none}
.bar{height:8px;background:var(--code);border-radius:99px;overflow:hidden;margin:2px 0 10px}
.bar i{display:block;height:100%;width:0;background:var(--accent);border-radius:99px;transition:width .4s}
.bar.indet i{width:35%;animation:slide 1.4s ease-in-out infinite alternate}
@keyframes slide{from{margin-left:0}to{margin-left:65%}}
@media(prefers-reduced-motion:reduce){.bar.indet i{animation:none;margin-left:20%}.bar i{transition:none}}
#actNow{margin:0 0 8px}
.errbox{border:1px solid var(--bad);color:var(--bad);border-radius:6px;padding:8px 12px;margin:0 0 10px;white-space:pre-wrap;word-break:break-word}
.errbox span{display:block;color:var(--ink);margin-top:4px}
#logPre{background:var(--code);padding:10px;border-radius:6px;max-height:240px;overflow:auto;white-space:pre-wrap;word-break:break-word;margin:6px 0 0}
dialog{border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--ink);padding:0;width:min(640px,94vw)}
dialog::backdrop{background:rgba(0,0,0,.45)}
.pk-head{padding:14px 16px 8px}.pk-head b{display:block;margin-bottom:6px}
#pkPath{word-break:break-all;color:var(--mute);font-size:.88rem}
#pkList{list-style:none;margin:0;padding:0 8px;height:300px;overflow:auto;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
#pkList button{all:unset;display:block;width:100%;box-sizing:border-box;padding:6px 10px;cursor:pointer;border-radius:5px}
#pkList button:hover,#pkList button:focus-visible{background:var(--accent-soft)}
.pk-foot{padding:12px 16px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.pk-foot input{flex:1;min-width:140px;width:auto}
.pkg-section{margin-top:26px;padding-top:12px;border-top:1px solid var(--line)}
.pkg-choice-body{padding:6px 16px 14px;display:grid;gap:14px}
.pkg-choice-body button{width:100%;text-align:left}
.pkg-form{padding:0 16px 8px;max-height:62vh;overflow:auto}
#svpkDialog{max-height:92vh;overflow:auto}
#packageStatus:empty{display:none}
#packageFormError:empty{display:none}
@media(max-width:860px){#activity{max-height:40vh}.shell{grid-template-columns:1fr;grid-template-rows:auto 1fr}.rail{border-right:0;border-bottom:1px solid var(--line);padding:10px 14px}.rail ol{display:flex;flex-wrap:wrap;border-left:0;margin:0 0 6px;gap:4px}.rail .dot{display:none}.rail button.nav{padding:6px 10px;border:1px solid var(--line)}.rail small{display:none}.pages{padding:18px 16px 30px}.grid2{grid-template-columns:1fr}}
</style>
</head>
<body>
<header class="top">
  <h1>TrainSV</h1><span class="sub">Teach a Synthesizer V voice your recordings</span>
  <div class="chips" id="chips"></div>
  <div class="lang-switch"><label for="uiLanguage">UI language</label><select id="uiLanguage" data-s aria-label="WebUI language"><option value="zh-CN">简体中文</option><option value="en" selected>English</option><option value="ja">日本語</option></select></div>
  <button class="ghost small" id="quitBtn" title="Stop the GUI program">Quit</button>
</header>
<div id="banner" class="banner bad" hidden></div>
<div class="shell">
<nav class="rail" aria-label="Steps">
  <ol id="steps">
    <li data-step="voice"><span class="dot">1</span><button class="nav" data-go="voice">Voice<small>Pick where to start</small></button></li>
    <li data-step="data"><span class="dot">2</span><button class="nav" data-go="data">Recordings<small>Prepare your data</small></button></li>
    <li data-step="train"><span class="dot">3</span><button class="nav" data-go="train">Train<small>Teach the voice</small></button></li>
    <li data-step="results"><span class="dot">4</span><button class="nav" data-go="results">Results<small>Listen and compare</small></button></li>
  </ol>
  <ol class="more"><li data-step="tools"><button class="nav" data-go="tools">More tools<small>Practice data, dictionaries</small></button></li></ol>
</nav>
<main>
<div class="pages">

<section class="page" id="page-voice">
  <h2>Choose a voice</h2>
  <p class="lead">Start from a Synthesizer V voice. TrainSV makes a new copy and teaches it your recordings, so your original voice file is left alone.</p>
  <div class="field" data-id="voice" data-kind="path" data-mode="file" data-ext="nofs" data-label="Voice file" data-ph="C:\path\to\voice.nofs" data-hint="A file named <code>voice.nofs</code> or <code>info.&lt;version&gt;.nofs</code>."></div>
  <div class="status" id="voice_st"></div>
  <div class="field" style="margin-top:16px">
    <label for="language">Language your recordings are sung in</label>
    <select id="language" data-s>
      <option value="ja">Japanese</option><option value="zh">Mandarin</option><option value="en">English</option><option value="yue">Cantonese</option><option value="es">Spanish</option>
    </select>
    <small>This tells TrainSV which set of phonemes (sounds) your labels use.</small>
  </div>
  <div class="actions">
    <button class="ghost" data-run="inspect">Look inside this voice</button>
    <button class="primary" data-go="data">Next: your recordings</button>
  </div>
</section>

<section class="page" id="page-data">
  <h2>Prepare your recordings</h2>
  <p class="lead">TrainSV learns from a <b>dataset</b>: your recordings plus a text file saying which sounds are sung when. Pick the option that matches what you have.</p>
  <div class="seg">
    <label><input type="radio" name="d_mode" id="d_mode_have" data-s checked><span><b>I already have a dataset</b><small>A folder with <code>wavs</code> and <code>transcriptions.csv</code>.</small></span></label>
    <label><input type="radio" name="d_mode" id="d_mode_lab" data-s><span><b>I have recordings and .lab files</b><small>Convert them into a dataset first.</small></span></label>
  </div>

  <div id="pane-have">
    <div class="field" data-id="dataset" data-kind="path" data-mode="dir" data-label="Dataset folder" data-ph="C:\path\to\my-dataset"></div>
    <div class="status" id="dataset_st"></div>
    <div class="actions">
      <button type="button" class="secondary" data-audit="dataset">Audit missing samples and phone transitions</button>
      <button class="ghost" data-run="score">Score the original voice on this data (optional)</button>
    </div>
    <pre id="datasetAudit" hidden style="white-space:pre-wrap;word-break:break-word;background:var(--code);padding:10px;border-radius:6px;max-height:320px;overflow:auto"></pre>
    <p class="hint">Optional: shows how close the starting voice already is, so you can compare after training.</p>
  </div>

  <div id="pane-lab" hidden>
    <div class="field"><label for="i_src">Where are your files?</label>
      <select id="i_src" data-s>
        <option value="folder">One folder with both .wav and .lab files</option>
        <option value="split">One folder of .wav files and another of .lab files</option>
        <option value="filenames">Generate .lab from phonemes in WAV filenames</option>
        <option value="manifest">A TrainV6 manifest.json</option>
      </select></div>
    <div data-src="folder"><div class="field" data-id="i_corpus" data-kind="path" data-mode="dir" data-label="Recordings folder" data-hint="Each <code>song.wav</code> needs a matching <code>song.lab</code>."></div></div>
    <div data-src="split" hidden>
      <div class="field" data-id="i_wavs" data-kind="path" data-mode="dir" data-label="Folder of .wav files"></div>
      <div class="field" data-id="i_labs" data-kind="path" data-mode="dir" data-label="Folder of .lab files"></div>
    </div>
    <div data-src="filenames" hidden>
      <div class="field" data-id="i_wavs_names" data-kind="path" data-mode="dir" data-label="Folder of WAV files" data-hint="Use spaces between phones, e.g. <code>@_n 7_0.wav</code>. Only the final _0/_1 retake suffix is removed. Choose the correct source phone convention below so symbols are mapped explicitly."></div>
      <div class="field" data-id="i_source_system" data-kind="select" data-options="auto|Auto (language default);vocaloid-mandarin|VOCALOID Mandarin (legacy);xsampa|Standard X-SAMPA;synthv-xsampa|SynthV legacy X-SAMPA;sv1112-chinese|SynthV 1.11.2 Chinese (Old labels);sv1112-cantonese|SynthV 1.11.2 Cantonese (Old labels);none|No conversion" data-value="auto" data-label="Source phone convention" data-hint="Choose the exact notation used in filenames. Standard X-SAMPA and legacy SynthV Mandarin spellings use separate explicit maps; Auto retains the VOCALOID Mandarin mapping. After import, inspect phone_mapping_report.csv because an existing target symbol does not prove that a source token has the right Mandarin phonetic meaning. In explicit modes, unknown tokens are reported rather than heuristically split."></div>
      <div class="field" data-id="i_phone_table" data-kind="path" data-mode="file" data-ext="json" data-label="phone_table.json (optional)" data-hint="Usually not needed when a voice is selected above. If omitted, keep “Mark word boundaries using my voice” enabled so TrainSV can read the selected voice phone set."></div>
      <p class="note">Approximation: the first and last <code>sil</code> each take 10% of the audio; all middle phones share the remaining 80% equally. This creates labels from filename order, not acoustic forced alignment.</p>
      <p class="note">After import, review phone_mapping_report.csv in the dataset folder. It lists each source token, the target voice phone(s), and the mapping rule. A technically valid mapping can still be phonetically wrong for your source convention.</p>
    </div>
    <div data-src="manifest" hidden><div class="field" data-id="i_manifest" data-kind="path" data-mode="file" data-ext="json" data-label="manifest.json"></div></div>
    <div class="status" id="i_src_st"></div>
    <div class="field" data-id="i_output" data-kind="path" data-mode="dir" data-label="Save the new dataset to" data-hint="When it finishes, this becomes your dataset automatically."></div>
    <label class="choice"><input type="checkbox" id="i_words" data-s checked><span><b>Mark word boundaries using my voice</b><small>Needed later if you add notes (for example with GAME) to train timing and pitch. Uses the voice and language from step 1.</small></span></label>
    <details class="adv"><summary>More options</summary>
      <div class="field" data-id="i_songs" data-label="Only these songs" data-ph="song_001,song_002" data-hint="Comma-separated names. Leave empty to use everything."></div>
      <div class="field"><label for="i_map">Rename phonemes</label><textarea id="i_map" data-s placeholder="vf=cl"></textarea><small>One <code>old=new</code> pair per line, for labels that use a different name than the voice.</small></div>
    </details>
    <div class="actions"><button class="primary" data-run="import">Convert to a dataset</button></div>
  </div>
</section>

<section class="page" id="page-train">
  <h2>Train your voice</h2>
  <p class="lead">Training adjusts a copy of the voice so it sounds more like your recordings. Choose what to teach it and for how long, then press Train.</p>
  <div class="field" data-id="t_timbre_preset" data-kind="select" data-options="manual|Manual (no changes);preserve|Preserve base timbre;gentle|Gentle target adaptation;balanced|Balanced VOCALOID fit;natural|Natural transitions (conservative);soft-consonants|Softer consonants (experimental);clarity-lite|Light source-faithful clarity (experimental);clarity|Brightness / clarity experiment;detail|Higher detail fit (experimental)" data-value="manual" data-label="Timbre fine-tuning preset" data-hint="Choose a starting profile for voice preservation, target adaptation, or an experimental brightness-balance loss. The preset fills parameter fields; you can still edit them afterward."></div>
  <p id="timbrePresetHelp" class="hint" aria-live="polite">Manual mode leaves current parameters unchanged.</p>
  <p class="hint">The brightness option learns the high/mid spectral balance present in your reference WAV. It does not apply an EQ filter or guarantee that nasal resonance disappears.</p>
  <fieldset>
    <legend>What should it learn?</legend>
    <label class="choice"><input type="checkbox" id="m_acoustic" data-s checked><span><b>Sound</b> (acoustic)<small>How the voice sounds. Works with any dataset. Start here.</small></span></label>
    <label class="choice"><input type="checkbox" id="m_duration" data-s><span><b>Timing</b> (duration)<small>How long each sound lasts. Needs notes in the dataset.</small></span></label>
    <label class="choice"><input type="checkbox" id="m_pitch" data-s><span><b>Pitch</b> (melody and vibrato)<small>How the singer moves between notes. Needs notes in the dataset.</small></span></label>
    <label class="choice"><input type="checkbox" id="m_vocoder" data-s><span><b>Vocoder</b> (final audio stage)<small>Optional and slower. Most people leave this off.</small></span></label>
    <label class="choice"><input type="checkbox" id="t_preserve_non_acoustic_models" data-s disabled><span><b>Preserve original voicebank models (except acoustic)</b><small>Available only for info.nofs. Train Sound only; keep the original Timing, Pitch/F0 and Vocoder model references and skip loading/checking those networks. A regular voice.nofs cannot use this option.</small></span></label>
  </fieldset>
  <fieldset id="f0CalibrationPanel">
    <legend>F0 Analysis &amp; Calibration</legend>
    <p class="hint">F0 options control only the analysis and cleanup of the recorded WAV's fundamental-frequency targets. They do not change Pitch-network loss settings.</p>
    <div class="grid2">
      <div class="field" data-id="t_f0_profile" data-kind="select" data-options="standard|Standard (65–1100 Hz, threshold 0.60);vocaloid|VOCALOID F0 cleanup (45–1400 Hz, threshold 0.45);low|Low register (45–650 Hz, threshold 0.60);high|High register (100–1400 Hz, threshold 0.60);wide|Wide range (40–1600 Hz, threshold 0.60);sensitive|Sensitive (35–1800 Hz, threshold 0.30)" data-value="standard" data-label="F0 analysis preset" data-hint="此预设只控制输入 WAV 的 F0 检测范围与目标曲线清理，不会更改 Pitch 模型的损失权重。VOCALOID 预设会启用中值平滑与保守的孤立八度跳点修正。该操作不是最终 WAV 的后期滤镜。"></div>
      <div class="field" data-id="t_f0_floor" data-label="F0 floor (Hz; overrides preset)" data-ph="Preset default, e.g. 45" data-hint="低于此频率的音高不在检测范围内；应根据歌手最低音设置。"></div>
      <div class="field" data-id="t_f0_ceiling" data-label="F0 ceiling (Hz; overrides preset)" data-ph="Preset default, e.g. 1400" data-hint="高于此频率的音高不在检测范围内；过高也可能增加谐波误检。"></div>
      <div class="field" data-id="t_f0_threshold" data-label="Voicing threshold (0–1; overrides preset)" data-ph="Preset default, e.g. 0.45" data-hint="降低可能减少漏检，也可能把噪声误判为有声；留空时使用所选预设值。"></div>
      <div class="field" data-id="t_f0_smoothing" data-kind="select" data-options="0|Off;3|Light (15 ms);5|Medium (25 ms);7|Strong (35 ms)" data-value="0" data-label="F0 median smoothing" data-hint="在连续有声片段内对 log2(F0) 中值滤波；VOCALOID F0 cleanup 预设默认启用 25 ms 平滑来压制短促跳点，强度过高也可能压低真实颤音细节。"></div>
      <div class="field" data-id="t_f0_correction" data-label="F0 reference correction (cents)" data-value="0" data-ph="0 (no correction)" data-hint="在缓存后的 F0 参考曲线上整体校正音分。若检测结果测得高约 25 音分，可先试 -25。不能修复错误八度、错误音符或对齐问题。范围 -1200 到 +1200。"></div>
    </div>
    <label class="choice"><input type="checkbox" id="t_f0_octave" data-s><span><b>Correct isolated octave jumps</b><small>仅修正前后有声帧一致、且中间帧疑似高/低 1–2 个八度的离群点；不会改写持续的八度变化。</small></span></label>
  </fieldset>
  <fieldset id="pitchTrainingPanel">
    <legend>Pitch Model &amp; Training</legend>
    <p class="hint">Pitch options control how the Pitch network is initialized and trained. They are independent of F0 detection, smoothing and reference calibration.</p>
    <div class="grid2">
      <div class="field" data-id="t_pitch_source" data-kind="select" data-options="base|Use Pitch model from selected base voice;pretrained|Import compatible pretrained Pitch DNNI" data-value="base" data-label="Pitch model initialization" data-hint="默认使用当前底模的 Pitch 模型；导入外部模型前会检查模型结构、音素表和说话人向量维度。"></div>
      <div class="field" data-id="t_pitch_model" data-kind="path" data-mode="file" data-ext="dnni" data-label="External pretrained Pitch model (.dnni)" data-hint="仅在选择导入预训练模型时需要。必须是 TrainSV 当前支持的 Pitch DNNI，并且音素表与底模匹配。"></div>
      <div class="field" data-id="t_pitch_smoothing" data-label="Melody smoothing cutoff (Hz)" data-value="3" data-hint="用于 Pitch 训练损失中区分慢旋律轮廓和快速音高运动的低通截止频率；不会直接对导出的 WAV 做后期平滑。"></div>
      <div class="field" data-id="t_pitch_melody_weight" data-label="Smooth melody loss weight" data-value="1" data-hint="控制 Pitch 模型学习整体音符走向的权重。数值越大，越重视较平滑的旋律误差；设为 0 可关闭该损失分量。"></div>
      <div class="field" data-id="t_pitch_motion_weight" data-label="Vibrato / fast-motion loss weight" data-value="1" data-hint="控制 Pitch 模型学习颤音和快速音高运动的权重。太高可能使模型把原始合成器的抖动也学进去。至少一个 Pitch loss 权重必须大于 0。"></div>
    </div>
    <div class="actions">
      <button type="button" class="secondary" data-run="pitch">Train Pitch Model Only</button>
    </div>
    <p class="hint">当前仓库没有可验证的通用神经 Pitch 权重，因此不会把现有底模伪装成“通用预训练模型”。你可以使用底模 Pitch 或导入结构/音素表兼容的预训练 .dnni；内置 F0 平滑与校准属于通用信号处理，不是神经 Pitch 模型。</p>
  </fieldset>
  <div class="note" id="trainWarn" hidden></div>
  <fieldset>
    <legend>How long?</legend>
    <label class="choice"><input type="radio" name="preset" id="p_quick" data-s><span><b>Quick test</b> (200 steps)<small>Just checks that everything works. Do this first.</small></span></label>
    <label class="choice"><input type="radio" name="preset" id="p_std" data-s checked><span><b>Standard</b> (2,000 steps)<small>A good starting point.</small></span></label>
    <label class="choice"><input type="radio" name="preset" id="p_long" data-s><span><b>Thorough</b> (6,000 steps)<small>Slower. Gradually lowers the learning rate for a finer result.</small></span></label>
    <div class="field" data-id="t_steps" data-label="Steps" data-value="2000" data-hint="One step looks at one short phrase from your recordings. More steps take longer."></div>
  </fieldset>
  <div class="field" data-id="v_name" data-label="Name of your new voice" data-value="My Singer"></div>
  <div class="field" data-id="t_output" data-kind="path" data-mode="dynamic" data-ext="nofs" data-label="Save the new voice to" data-hint="Suggested automatically. Voices that start with <code>info.</code> are saved into a folder; others into a single file."></div>
  <div class="field" data-id="t_acoustic_dnni_version" data-kind="select" data-options="2|DNNI v2 (default);1|DNNI v1 (legacy compatibility)" data-value="2" data-label="Acoustic DNNI output version" data-hint="只转换输出声学模型的 DNNI 容器格式，不改变网络架构或训练权重。新格式仍需要在目标 Synthesizer V 中实际打开验证；旧版不兼容时可选 v1。"></div>
  <label class="choice"><input type="checkbox" id="t_savebest" data-s checked><span><b>Also save whenever it improves</b><small>So you still have a voice if you stop early.</small></span></label>

  <details class="adv"><summary>Advanced settings</summary>
    <p class="hint">Leave these empty to use the defaults shown in grey.</p>
    <div class="grid2">
      <div class="field"><label for="t_params">What to adjust</label><select id="t_params" data-s><option value="all">Everything (default)</option><option value="embeddings">Only the voice's identity vectors</option></select><small>The second option is quicker and gentler; used first when adding vocal modes.</small></div>
      <div class="field"><label for="t_device">Computer part to use</label><select id="t_device" data-s><option value="">Automatic (default)</option><option value="cuda">GPU (NVIDIA)</option><option value="mps">GPU (Apple)</option><option value="cpu">CPU (slow)</option></select></div>
      <div class="field" data-id="t_lr" data-label="Learning rate" data-ph="2e-5 (3e-3 for identity only)" data-hint="How big each adjustment is."></div>
      <div class="field" data-id="t_warmup" data-label="Warm-up steps" data-ph="0" data-hint="Start gently over this many steps."></div>
      <div class="field" data-id="t_lrend" data-label="Final learning rate" data-ph="constant" data-hint="Slowly lower the learning rate to this."></div>
      <div class="field" data-id="t_anchor" data-label="Stay close to the original" data-ph="1.0" data-hint="Higher keeps more of the original voice."></div>
      <div class="field" data-id="t_batch" data-label="Phrases per step" data-ph="1"></div>
      <div class="field" data-id="t_cons" data-label="Consonant emphasis" data-ph="1.0" data-hint="Above 1 makes it work harder on consonants."></div>
      <div class="field" data-id="t_transition_weight" data-label="Phone transition emphasis" data-value="1" data-hint="额外提高非静音音素交界附近的训练权重，不会对输出音频做模糊平滑。默认 1 表示关闭；可从 1.25 开始试验。"></div>
      <div class="field" data-id="t_transition_frames" data-label="Transition radius (5 ms frames)" data-value="2" data-hint="音素边界两侧的加权范围；2 帧约为边界两侧各 10 ms。仅当过渡权重大于 1 时生效。"></div>
      <div class="field" data-id="t_transition_sampling" data-label="Transition sampling probability" data-value="0.5" data-hint="以此概率将训练窗口放在有效的非静音音素边界附近；其余保持均匀随机采样。设为 0 可关闭。"></div>
      <div class="field" data-id="t_transition_local_weight" data-label="Transition-local spectral loss weight" data-value="0.25" data-hint="对边界中心约 80 ms 的生成音频与参考 WAV 额外计算多分辨率频谱损失；设为 0 可关闭，不会对最终输出做平滑滤镜。"></div>
      <div class="field" data-id="t_brightness_weight" data-label="Brightness-balance loss weight" data-value="0" data-hint="0 表示关闭；0.05 是轻度实验，0.10 更强。比较输出与目标 WAV 的高频/中频能量平衡，不会对生成的 WAV 做后期 EQ，也不保证消除鼻音。"></div>
      <div class="field" data-id="t_val" data-label="Recordings held back for testing" data-ph="0.1" data-hint="A fraction (0.1 = 10%) or a count (5)."></div>
      <div class="field" data-id="t_seed" data-label="Random seed" data-ph="0" data-hint="Use the same value when comparing results."></div>
      <div class="field" data-id="t_evalevery" data-label="Test every (steps)" data-ph="250"></div>
      <div class="field" data-id="t_logevery" data-label="Report every (steps)" data-ph="25"></div>
    </div>
    <h4>Custom base DNNI models (info.nofs only)</h4>
    <p class="hint">Replace the complete non-acoustic model with a .dnni from another voicebank. DNNI v1/v2 must pass role-specific TrainSV structure checks. DNNI v3 can only be preserved opaquely and cannot be trained or runtime-validated by TrainSV. Do not combine these overrides with “Preserve original voicebank models”.</p>
    <div class="grid2">
      <div class="field" data-id="t_custom_duration_dnni" data-kind="path" data-mode="file" data-ext="dnni" data-label="Custom Duration model (.dnni)"></div>
      <div class="field" data-id="t_custom_pitch_dnni" data-kind="path" data-mode="file" data-ext="dnni" data-label="Custom Pitch/F0 model (.dnni)" data-hint="Replaces the base Pitch model. This differs from “External pretrained Pitch model”, which initializes weights in the current target model."></div>
      <div class="field" data-id="t_custom_vocoder_dnni" data-kind="path" data-mode="file" data-ext="dnni" data-label="Custom Vocoder model (.dnni)"></div>
    </div>
    <div class="field" data-id="t_pmap" data-kind="path" data-mode="file" data-ext="json" data-label="Phoneme map (JSON)" data-hint="Renames your dataset's phonemes to the voice's phonemes."></div>
    <div class="field" data-id="t_modes" data-label="Vocal modes" data-ph="Solid,High" data-hint="Info voices only. Comma-separated."></div>
    <div class="grid2">
      <div class="field" data-id="t_version" data-label="Version name" data-hint="Info voices only."></div>
      <div class="field" data-id="t_final" data-label="Also save the last step as version" data-hint="Info voices only."></div>
    </div>
    <label class="choice"><input type="checkbox" id="t_keep" data-s><span><b>Keep recordings at their original loudness</b><small>By default they are scaled to the starting voice's level.</small></span></label>
    <div class="grid2">
      <div class="field" data-id="t_levelref" data-kind="path" data-mode="file" data-ext="nofs" data-label="Loudness reference voice" data-hint="Use the original voice when continuing from an earlier finetune."></div>
      <div class="field" data-id="t_gain" data-label="Fixed loudness factor" data-ph="measured"></div>
    </div>
    <div class="field" data-id="t_run" data-kind="path" data-mode="dir" data-label="Run folder" data-hint="Keeps the log and lets you continue later."></div>
    <label class="choice"><input type="checkbox" id="t_resume" data-s><span><b>Continue the previous run</b><small id="resumeHint">Raise the number of steps above what was already done to train further.</small></span></label>
    <div class="field" data-id="t_history" data-kind="path" data-mode="save" data-ext="json" data-label="Save test scores to a file (JSON)"></div>
  </details>

  <div class="actions"><button class="primary" data-run="finetune">Train</button></div>
</section>

<section class="page" id="page-results">
  <h2>Check the result</h2>
  <p class="lead">Compare the original voice with your new one on recordings that were held back from training. You can listen to each next to the real recording. Lower scores are closer to the recording.</p>
  <div class="field" data-id="e_new" data-kind="path" data-mode="file" data-ext="nofs" data-label="Your new voice"></div>
  <button class="ghost small" id="useTrained">Use the voice I just trained</button>
  <p class="hint" id="e_models" style="margin-top:12px"></p>
  <div class="field" style="margin-top:14px" data-id="e_out" data-kind="path" data-mode="dir" data-label="Save the comparison to"></div>
  <div class="field" data-id="e_max" data-label="Recordings to render" data-value="4" data-hint="Each one is rendered by both voices so you can listen."></div>
  <div class="actions">
    <button class="primary" data-run="evaluate">Compare voices</button>
    <button class="ghost" id="showResults">Show saved results</button>
  </div>
  <div id="results"></div>
  <div class="pkg-section">
    <h3>Export Trained Voicebank</h3>
    <p class="hint">SVPK 当前仅为格式研究诊断包。全零字段版本在 Sharp 中无反应；把 Hanakuma Chifuyu 的 256 字节字段复制到自制包后，Sharp 能识别包但提示数据损坏。这强烈暗示该字段参与包专属完整性/认证校验（RSA-2048 签名只是未证实的假设）。当前生成结果不可安装或发布。SFPK 为实验性 Flat 格式导出：会将模型依赖写入所选本机 models 目录，包内仅含 config.json 与 info.nofs，暂非自包含包，必须在 Flat 中实际验证。</p>
    <div class="actions"><button class="primary" type="button" id="openPackageFormat">Export Voicebank / Choose Package Format</button></div>
    <p class="status" id="packageStatus" role="status"></p>
  </div>
</section>

<section class="page" id="page-tools">
  <h2>More tools</h2>
  <p class="lead">Handy extras. You do not need these for a normal first run.</p>
  <h3>Make practice data</h3>
  <p class="hint">Renders a small labelled dataset using your chosen voice. Useful to test the whole process without any recordings. When it finishes, it becomes your dataset.</p>
  <div class="field" data-id="s_output" data-kind="path" data-mode="dir" data-label="Save practice data to"></div>
  <div class="grid2">
    <div class="field" data-id="s_phrases" data-label="Phrases" data-value="40"></div>
    <div class="field" data-id="s_notes" data-label="Notes per phrase" data-value="8"></div>
  </div>
  <div class="actions"><button data-run="synth">Make practice data</button></div>
  <h3>Convert a dictionary</h3>
  <p class="hint">Rewrites a DiffSinger dataset's spelling to Synthesizer V's dictionary for its language. The original is untouched. Uses the voice and language from step 1.</p>
  <div class="field" data-id="c_data" data-kind="path" data-mode="dir" data-label="Source dataset"></div>
  <div class="field" data-id="c_output" data-kind="path" data-mode="dir" data-label="Save converted dataset to"></div>
  <div class="field" data-id="c_src" data-kind="path" data-mode="file" data-ext="txt" data-label="Source dictionary" data-hint="For example opencpop-extension.txt"></div>
  <div class="field" data-id="c_tgt" data-kind="path" data-mode="file" data-ext="txt" data-label="Synthesizer V dictionary" data-hint="For example mandarin-xsampa-dict.txt"></div>
  <div class="field" data-id="c_coda" data-label="Share of a syllable given to its ending" data-ph="0.3"></div>
  <div class="actions"><button data-run="convert">Convert dictionary</button></div>

  <h3>Convert audio to MIDI for pitch calibration</h3>
  <p class="hint">Extracts a monophonic melody from audio using continuous F0 smoothing, conservative repair of short isolated octave errors, and more stable note boundaries. Import the MIDI into Synthesizer V to compare pitch. This is not polyphonic transcription, and semitone MIDI notes cannot preserve the source timbre or its full continuous pitch/vibrato curve.</p>
  <div class="field" data-id="am_input" data-kind="path" data-mode="file" data-ext="wav,flac,aiff,aif" data-label="Source audio (WAV / FLAC / AIFF)" data-hint="Use a clear monophonic vocal recording. Polyphonic music can confuse the pitch tracker."></div>
  <div class="field" data-id="am_output" data-kind="path" data-mode="save" data-ext="mid" data-label="Save MIDI file to" data-hint="Choose a .mid output path. Import this MIDI into Synthesizer V Studio to compare the voicebank's rendered notes with the original recording."></div>
  <div class="grid2">
    <div class="field" data-id="am_floor" data-label="Pitch floor (Hz)" data-value="65" data-hint="Lowest fundamental frequency to track. Raise this to avoid low-frequency rumble being detected as notes."></div>
    <div class="field" data-id="am_ceiling" data-label="Pitch ceiling (Hz)" data-value="1100" data-hint="Highest fundamental frequency to track. Lower this if octave jumps appear in the extracted melody."></div>
    <div class="field" data-id="am_voicing" data-label="Voicing threshold (0.05–1.0)" data-value="0.45" data-hint="Higher values reject uncertain/unvoiced frames more aggressively; lower values may recover weak notes but can add false notes."></div>
    <div class="field" data-id="am_min_note" data-label="Minimum note duration (ms)" data-value="70" data-hint="Short pitch fragments are merged into a neighbouring note when possible. Reduce this for fast passages; increase it to suppress brief tracking glitches."></div>
    <div class="field" data-id="am_median" data-label="Pitch smoothing window (1–21, odd)" data-value="5" data-hint="Median smoothing is applied within voiced spans before semitone quantization. Use 1 to disable smoothing; larger windows suppress pitch flicker but can blur fast runs."></div>
    <div class="field" data-id="am_a4" data-label="A4 reference (Hz)" data-value="440" data-hint="Reference pitch for MIDI note 69. Keep 440 Hz for standard concert tuning, or adjust for a non-standard reference."></div>
    <div class="field" data-id="am_transpose" data-label="Transpose (semitones, −24 to +24)" data-value="0" data-hint="Shift all extracted notes by a fixed number of semitones for testing a different vocal range."></div>
    <div class="field" data-id="am_tempo" data-label="MIDI tempo metadata (BPM)" data-value="120" data-hint="Sets MIDI tempo metadata. Extracted note timing remains aligned to the source audio duration."></div>
  </div>
  <div class="actions"><button data-run="audioMidi">Convert audio to MIDI</button></div>

  <h3>Batch replace non-acoustic DNNI models</h3>
  <p class="hint">Replace any combination of Duration, Pitch/F0 and Vocoder in one operation. Acoustic is never changed. Choose the Synthesizer V models folder containing duration/, f0/, acoustic/ and vocoder/; this fixes missing-reference errors when the info.nofs does not sit next to its model library. Unselected models and their IDs stay unchanged. The original info.nofs and DNNI files are never overwritten.</p>
  <div class="field" data-id="rm_source" data-kind="path" data-mode="file" data-ext="nofs" data-label="Source info.nofs" data-hint="Choose the existing JSON info.nofs file, not a regular binary voice.nofs."></div>
  <div class="field" data-id="rm_models_root" data-kind="path" data-mode="dir" data-label="Source models folder" data-hint="Choose the models directory whose direct subfolders are duration, f0, acoustic and vocoder. For example, the models folder inside your Synthesizer V Studio installation."></div>
  <div class="field" data-id="rm_duration_model" data-kind="path" data-mode="file" data-ext="dnni" data-label="Replacement Duration (.dnni, optional)"></div>
  <div class="field" data-id="rm_pitch_model" data-kind="path" data-mode="file" data-ext="dnni" data-label="Replacement Pitch/F0 (.dnni, optional)"></div>
  <div class="field" data-id="rm_vocoder_model" data-kind="path" data-mode="file" data-ext="dnni" data-label="Replacement Vocoder (.dnni, optional)"></div>
  <div class="field" data-id="rm_output" data-kind="path" data-mode="dir" data-label="Output package folder" data-hint="Choose a separate writable output folder, for example Documents/TrainSV-output. TrainSV creates database/voices/<name>/info.<version>.nofs and database/models/ inside it. Do not select the installed Synthesizer V models folder."></div>
  <div class="actions"><button data-run="replaceModel">Batch replace models</button></div>
</section>

</div>
<aside id="activity" hidden aria-live="polite">
  <div class="act-head">
    <span class="chip" id="actChip"></span><strong id="actTitle"></strong><span class="grow" id="actSummary"></span>
    <button class="danger small" id="stopBtn" hidden>Stop</button>
    <button class="ghost small" id="actToggle">Hide details</button>
  </div>
  <div class="act-body">
    <div class="bar" id="bar"><i id="barFill"></i></div>
    <p id="actNow"></p>
    <div class="errbox" id="actErr" hidden></div>
    <table class="t" id="actMetrics" hidden></table>
    <button class="primary" id="actNext" hidden></button>
    <details id="actLog"><summary>Full log</summary><button class="ghost small" id="copyLog">Copy log</button><pre id="logPre" class="mono"></pre></details>
  </div>
</aside>
</main>
</div>

<dialog id="packageFormatDialog">
  <div class="pk-head"><b>Choose Voicebank Package Format</b><div class="hint">SVPK 目前只生成诊断包，不能安装。SFPK 提供实验性导出：生成 config.json + info.nofs，并将四类 .dnni 依赖写入所选本机 models 目录；不会把模型二进制嵌入 SFPK，迁移到其他电脑时需另行安装对应模型。</div></div>
  <div class="pkg-choice-body">
    <div><button class="primary" type="button" id="chooseSvpk">Build SVPK Diagnostic Package (Not Installable)</button><p class="hint">仅用于检查内部 ZIP 与元数据结构，不能导入或发布为可用声库。</p></div>
    <div><button type="button" id="chooseSfpk">Export SFPK Package (Experimental)</button><p class="hint">可输入训练后的 voice.nofs 或已有 info.*.nofs；需选择 Synthesizer V 的 models 目录。</p></div>
  </div>
  <div class="pk-foot"><button type="button" class="ghost" id="closePackageFormat">Cancel</button></div>
</dialog>

<dialog id="svpkDialog">
  <div class="pk-head"><b>SVPK Diagnostic Package (Not Installable)</b><div class="hint">无需手动编辑 JSON。图片请选择 PNG 文件。当前仅生成诊断包：全零字段会被 Sharp 忽略；复制 Chifuyu 字段会触发“数据损坏”。这表明字段可能参与包专属完整性/认证校验，算法和签署流程尚未确认，因此此功能不会生成可安装包。</div></div>
  <div class="pkg-form">
    <div class="field" data-id="svpk_name" data-label="Voicebank name" data-ph="e.g. Luo Tianyi"></div>
    <div class="field" data-id="svpk_vendor" data-label="Vendor / author" data-ph="e.g. Artist or studio"></div>
    <div class="field" data-id="svpk_version" data-label="Voicebank version (integer)" data-value="101" data-ph="101" data-hint="该数值会同时写入 package-config 的 versionString 和 versionNumber；当前版本使用整数版本号，例如 101。"></div>
    <div class="field" data-id="svpk_header_preset" data-kind="select" data-options="zero|All zero (baseline);random|Random byte test;sha256-chain|Synthetic SHA-256 chain;aes256-cbc-test|Synthetic AES-256-CBC test (unofficial)" data-value="zero" data-label="256-byte field diagnostic preset" data-hint="这些选项仅用于比较 Sharp 对不同字段字节的反应，不是官方签名或密钥，也不能生成可安装包。AES 选项使用从包内容派生的公开测试密钥和 IV。"></div>
    <div class="field" data-id="svpk_profile" data-kind="path" data-mode="file" data-ext="png" data-label="Avatar image (profile.png)" data-hint="选择声库头像 PNG 文件；打包后将按样例名称存放在 profile.png。"></div>
    <div class="field" data-id="svpk_background" data-kind="path" data-mode="file" data-ext="png" data-label="Background image (background.png)" data-hint="选择声库背景 PNG 文件；当前会按样例将文件条目命名为 background.png。"></div>
    <div class="field" data-id="svpk_output" data-kind="path" data-mode="save" data-ext="svpk" data-label="SVPK output path" data-hint="输出的 .svpk 文件路径。默认与训练后的 voice.nofs 放在同一目录。"></div>
    <p class="status bad" id="packageFormError" role="alert"></p>
  </div>
  <div class="pk-foot"><button type="button" class="ghost" id="cancelSvpk">Cancel</button><button type="button" class="primary" id="buildSvpk">Build Diagnostic Package (Not Installable)</button></div>
</dialog>


<dialog id="sfpkDialog">
  <div class="pk-head"><b>Export SFPK Package (Experimental)</b><div class="hint">普通 voice.nofs 将转换为 Flat 使用的 info.nofs 描述，并把 acoustic / vocoder / duration / f0 的 .dnni 文件写入你选定的本机 models 目录。SFPK 内包含 config.json 与 info.nofs；选填的 profile.png / background.png 会一起放入包内，但模型二进制仍在本机 models 目录。当前尚未验证 Flat 安装和渲染。</div></div>
  <div class="pkg-form">
    <div class="field" data-id="sfpk_name" data-label="Voicebank name (optional)" data-ph="e.g. Luo Tianyi" data-hint="留空时尽量保留来源声库名称；填写后覆盖。"></div>
    <div class="field" data-id="sfpk_vendor" data-label="Author / vendor (optional)" data-ph="e.g. Artist or studio" data-hint="留空时保留来源声库的作者信息；普通 voice.nofs 未提供时会使用 TrainSV。"></div>
    <div class="field" data-id="sfpk_version" data-label="Voicebank version (optional)" data-ph="101" data-hint="留空时保留来源声库已有版本；输入后会覆盖来源版本，可使用 101、106 或 106b1 等标识。"></div>
    <div class="field" data-id="sfpk_profile" data-kind="path" data-mode="file" data-ext="png" data-label="Avatar image profile.png (optional)" data-hint="选择 PNG 头像后，会作为 profile.png 放入 SFPK 根目录。不选择则不额外打包头像。"></div>
    <div class="field" data-id="sfpk_background" data-kind="path" data-mode="file" data-ext="png" data-label="Background image background.png (optional)" data-hint="选择 PNG 背景后，会作为 background.png 放入 SFPK 根目录。不选择则不额外打包背景。"></div>
    <div class="field" data-id="sfpk_models_root" data-kind="path" data-mode="dir" data-label="Synthesizer V models directory" data-hint="选择包含 acoustic、vocoder、duration、f0 等子目录的现有 models 文件夹。新模型会按内容 MD5 命名写入相应子目录，已有同名但内容不同的文件不会被覆盖。"></div>
    <div class="field" data-id="sfpk_output" data-kind="path" data-mode="save" data-ext="sfpk" data-label="SFPK output path" data-hint="生成的 SFPK 包含 config.json 与 info.nofs，并可包含已选择的头像/背景 PNG；所引用的模型仍在上方选定的本机 models 目录。"></div>
    <p class="status bad" id="sfpkFormError" role="alert"></p>
  </div>
  <div class="pk-foot"><button type="button" class="ghost" id="cancelSfpk">Cancel</button><button type="button" class="primary" id="buildSfpk">Build SFPK Test Package</button></div>
</dialog>

<dialog id="picker">
  <div class="pk-head"><b id="pkTitle">Choose</b><div id="pkPath"></div></div>
  <ul id="pkList"></ul>
  <div class="pk-foot" id="pkFoot"></div>
</dialog>

<script>
const I18N_ROWS = [["Teach a Synthesizer V voice your recordings","用你的录音训练 Synthesizer V 声库","録音を使って Synthesizer V 音声を学習"],["Phone transition emphasis","音素过渡强调","音素遷移の強調"],["Transition radius (5 ms frames)","过渡范围（5 毫秒帧）","遷移範囲（5 ms フレーム）"],["Timbre fine-tuning preset","音色微调预设","音色微調プリセット"],["Manual (no changes)","手动（不修改参数）","手動（設定を変更しない）"],["Preserve base timbre","保留底模音色","ベース音色を維持"],["Gentle target adaptation","温和适配目标音源","ターゲット音源へ穏やかに適応"],["Balanced VOCALOID fit","平衡型 VOCALOID 拟合","バランス型 VOCALOID 適合"],["Brightness / clarity experiment","明亮度／通透度实验","明るさ／明瞭度の実験"],["Higher detail fit (experimental)","更强细节拟合（实验）","より詳細な適合（実験）"],["Brightness-balance loss weight","明亮度平衡损失权重","明るさバランス損失の重み"],["The brightness option learns the high/mid spectral balance present in your reference WAV. It does not apply an EQ filter or guarantee that nasal resonance disappears.","明亮度选项会学习参考 WAV 中实际存在的高频／中频能量平衡，不会对输出音频套用 EQ，也不保证能消除鼻腔共鸣。","明るさ設定は参照 WAV に実際に含まれる高域／中域のスペクトルバランスを学習します。EQ をかけたり、鼻声が必ず消えたりするものではありません。"],["Choose a starting profile for voice preservation, target adaptation, or an experimental brightness-balance loss. The preset fills parameter fields; you can still edit them afterward.","选择保留底模、温和适配目标音源或实验性明亮度损失的起始配置。预设会填入参数，但之后仍可手动修改。","音声の保持、ターゲットへの適応、実験的な明るさバランス損失の開始設定を選択します。プリセット適用後も手動編集できます。"],["Brightness & nasality clues (heuristic)","明亮度与鼻腔感线索（启发式）","明るさと鼻声の手がかり（ヒューリスティック）"],["The spectral measures below compare each render to its source WAV. They are clues, not a reliable automatic nasality score.","下列频谱指标用于比较渲染结果和对应的原始 WAV，只是排查线索，并非可靠的自动鼻音评分。","以下のスペクトル指標はレンダーと元 WAV を比較する手がかりであり、信頼できる自動鼻声スコアではありません。"],["Spectral centroid (Hz)","频谱质心（Hz）","スペクトル重心（Hz）"],["Low/mid ratio (dB)","低频／中频比（dB）","低域／中域比（dB）"],["High/mid ratio (dB)","高频／中频比（dB）","高域／中域比（dB）"],["Reference WAV → render; Δ is render minus reference.","参考 WAV → 渲染；Δ 为渲染值减去参考值。","参照 WAV → レンダー；Δ はレンダーから参照を引いた値です。"],["Phrase-level diagnostic deltas","逐短语诊断差值","フレーズ別診断の差分"],["The averages above can hide a local issue; this table shows the difference for each evaluated phrase.","上面的平均值可能掩盖局部问题；此表会逐条显示本次评估短语的差异。","上の平均値では局所的な問題が隠れる場合があります。この表は評価した各フレーズの差を表示します。"],["UI language","界面语言","UI 言語"],["Quit","退出","終了"],["Stop the GUI program","停止 GUI 程序","GUI プログラムを終了"],["Steps","步骤","手順"],["Voice","声库","音声ライブラリ"],["Pick where to start","选择起点","開始地点を選択"],["Recordings","录音","録音データ"],["Prepare your data","准备数据","データを準備"],["Train","训练","学習"],["Teach the voice","训练声库","音声を学習"],["Results","结果","結果"],["Listen and compare","试听和比较","試聴して比較"],["More tools","更多工具","その他のツール"],["Practice data, dictionaries","练习数据、字典","練習データ、辞書"],["Choose a voice","选择声库","音声ライブラリを選択"],["Start from a Synthesizer V voice. TrainSV makes a new copy and teaches it your recordings, so your original voice file is left alone.","从一个 Synthesizer V 声库开始。TrainSV 会复制声库并使用你的录音进行训练，不会修改原始文件。","Synthesizer V の音声ライブラリを元に開始します。TrainSV はコピーを作成して録音を学習するため、元のファイルは変更されません。"],["Voice file","声库文件","音声ファイル"],["Language your recordings are sung in","录音演唱语言","録音の歌唱言語"],["Japanese","日语","日本語"],["Mandarin","普通话","中国語（普通話）"],["English","英语","英語"],["Cantonese","粤语","広東語"],["Spanish","西班牙语","スペイン語"],["This tells TrainSV which set of phonemes (sounds) your labels use.","此选项用于指定标注所使用的音素（发音）体系。","ラベルで使用している音素セットを指定します。"],["Look inside this voice","检查声库","音声ライブラリを検査"],["Next: your recordings","下一步：录音","次へ：録音データ"],["Prepare your recordings","准备录音数据","録音データを準備"],["TrainSV learns from a","TrainSV 会从以下数据中学习：","TrainSV は次のデータから学習します："],["dataset","数据集","データセット"],[": your recordings plus a text file saying which sounds are sung when. Pick the option that matches what you have.","：录音以及记录每个发音时间的文本文件。请选择与你现有文件相符的选项。","：録音と、各音素の発音タイミングを記したテキストファイルです。手元のデータに合う項目を選択してください。"],["I already have a dataset","我已经有数据集","データセットを用意済み"],["A folder with","包含以下内容的文件夹：","次のファイルを含むフォルダー："],["wavs","wavs","wavs"],["and","和","および"],["transcriptions.csv","transcriptions.csv","transcriptions.csv"],["I have recordings and .lab files","我有录音和 .lab 文件","録音と .lab ファイルがある"],["Convert them into a dataset first.","需要先将它们转换为数据集。","まずデータセットに変換します。"],["Dataset folder","数据集文件夹","データセットフォルダー"],["Score the original voice on this data (optional)","在此数据上评估原始声库（可选）","このデータで元の音声を評価（任意）"],["Optional: shows how close the starting voice already is, so you can compare after training.","可选：查看底模与录音的初始接近程度，方便训练后比较。","任意：元の音声が録音にどの程度近いかを確認し、学習後と比較できます。"],["Where are your files?","文件位于哪里？","ファイルはどこにありますか？"],["One folder with both .wav and .lab files","一个文件夹中同时包含 .wav 和 .lab",".wav と .lab が同じフォルダーにある"],["One folder of .wav files and another of .lab files",".wav 和 .lab 分别位于不同文件夹",".wav と .lab が別々のフォルダーにある"],["Generate .lab from phonemes in WAV filenames","根据 WAV 文件名中的音素生成 .lab","WAV ファイル名の音素から .lab を生成"],["Source phone convention","源音素标记体系","音素表記の規則"],["Auto (language default)","自动（使用语言默认规则）","自動（言語の既定規則）"],["VOCALOID Mandarin (legacy)","VOCALOID 中文（旧版标记）","VOCALOID 中国語（旧表記）"],["Standard X-SAMPA","标准 X-SAMPA","標準 X-SAMPA"],["SynthV legacy X-SAMPA","Synthesizer V 旧版 X-SAMPA","Synthesizer V 旧 X-SAMPA"],["SynthV 1.11.2 Chinese (Old labels)","Synthesizer V 1.11.2 中文（旧版音素标签）","Synthesizer V 1.11.2 中国語（旧音素ラベル）"],["SynthV 1.11.2 Cantonese (Old labels)","Synthesizer V 1.11.2 粤语（旧版音素标签）","Synthesizer V 1.11.2 広東語（旧音素ラベル）"],["No conversion","不转换","変換しない"],["Choose the source notation used in filenames. Standard X-SAMPA and legacy SynthV Mandarin spellings use their explicit maps. The SV 1.11.2 Chinese/Cantonese options convert through Dreamtonics Old-label tables. Auto retains the VOCALOID Mandarin mapping. Unknown phones in explicit modes are reported rather than guessed.","请按文件名实际使用的标记选择。标准 X-SAMPA、Synthesizer V 旧版中文标记分别使用明确映射；SV 1.11.2 Chinese/Cantonese 选项按 Dreamtonics 的 Old 旧标签表转换。自动模式保留 VOCALOID 中文映射。显式模式遇到未知音素会报告错误，不会猜测拆分。","ファイル名の表記を選択してください。標準 X-SAMPA と旧 SynthV 中国語表記には明示的な変換表を使用します。SV 1.11.2 Chinese/Cantonese は Dreamtonics の Old ラベル表で変換します。Auto は VOCALOID 中国語の変換を維持し、未知音素は推測で分割しません。"],["Sensitive (35–1800 Hz, threshold 0.30)","灵敏检测（35–1800 Hz，阈值 0.30）","高感度（35～1800 Hz、閾値 0.30）"],["选择 WAV 文件夹。生成模式会按文件名空格分隔的音素顺序生成 LAB；末尾 _0/_1 是重录编号。","Choose the WAV folder. The filename mode creates LAB labels from space-separated phones; only a final _0/_1 retake suffix is removed.","WAV フォルダーを選択します。ファイル名モードは空白区切りの音素から LAB を作成し、末尾の _0/_1 だけをテイク番号として削除します。"],["根据文件名实际使用的音素标记选择源体系。auto 保留 VOCALOID 中文默认转换；xsampa 用标准 X-SAMPA 映射；synthv-xsampa 用 Synthesizer V 旧版中文标记映射。显式模式遇到未知音素会报告错误，不会猜测拆分。","Choose the notation actually used in the filenames. Auto keeps the VOCALOID Mandarin mapping; xsampa uses the standard X-SAMPA map; synthv-xsampa maps legacy Synthesizer V Mandarin spellings. Unknown phones in explicit modes are reported instead of guessed splits.","ファイル名で使われている表記を選択します。Auto は VOCALOID 中国語の変換を使い、xsampa は標準 X-SAMPA、synthv-xsampa は Synthesizer V 旧中国語表記を変換します。明示モードの未知音素は推測分割せず報告します。"],["F0 预设只控制输入 WAV 的基频检测范围与目标曲线清理，不会更改 Pitch 模型的损失函数参数。Sensitive 使用更低的浊音阈值和 15 ms 轻度平滑，可尝试找回较弱的有声帧；静音或清辅音本身没有可测 F0，不能靠降低阈值创造音高。建议检查训练日志中的有声帧比例和中位 F0；范围放宽也可能增加错误音高候选。","F0 presets only control WAV pitch tracking and target cleanup, not Pitch-network loss settings. Sensitive lowers the voicing threshold and uses light 15 ms smoothing to recover weakly voiced frames; silence and voiceless consonants have no measurable F0, so lowering the threshold cannot create pitch. Check voiced-frame percentage and median F0 in the log; wider ranges can also add false candidates.","F0 プリセットは WAV のピッチ追跡と目標補正のみを制御し、Pitch ネットワークの損失設定は変えません。Sensitive は有声音のしきい値を下げ、15 ms の軽い平滑化で弱い有声音を検出しやすくします。無音や無声子音には測定可能な F0 がないため、閾値を下げてもピッチは作れません。ログの有声音フレーム率と F0 中央値を確認してください。範囲を広げると誤検出が増えることもあります。"],["A TrainV6 manifest.json","TrainV6 的 manifest.json","TrainV6 の manifest.json"],["Recordings folder","录音文件夹","録音フォルダー"],["Folder of .wav files","存放 .wav 的文件夹",".wav ファイルのフォルダー"],["Folder of .lab files","存放 .lab 的文件夹",".lab ファイルのフォルダー"],["Folder of WAV files","存放 WAV 的文件夹","WAV ファイルのフォルダー"],["phone_table.json (optional)","phone_table.json（可选）","phone_table.json（任意）"],["Approximation: the first and last","近似标注：开头和结尾的","近似ラベル：先頭と末尾の"],["each take 10% of the audio; all middle phones share the remaining 80% equally. This creates labels from filename order, not acoustic forced alignment.","各占音频时长的 10%；中间音素平均分配剩余的 80%。这些标签仅按文件名顺序生成，并非声学强制对齐。","それぞれ音声の 10% を割り当て、中央の音素は残り 80% を均等に分けます。ファイル名の順序から作る近似ラベルであり、音響強制アラインメントではありません。"],["Where are your files?","文件位于哪里？","ファイルはどこにありますか？"],["Save the new dataset to","将新数据集保存到","新しいデータセットの保存先"],["Mark word boundaries using my voice","使用所选声库标记词边界","選択した音声で単語境界を付与"],["Needed later if you add notes (for example with GAME) to train timing and pitch. Uses the voice and language from step 1.","如果之后添加音符（例如使用 GAME），训练 Timing 和 Pitch 时可能需要这些信息。将使用第 1 步选择的声库和语言。","後で音符（GAME などで追加）を使って Timing や Pitch を学習する際に必要です。手順 1 の音声と言語を使用します。"],["More options","更多选项","その他のオプション"],["Only these songs","仅处理这些曲目","これらの曲のみ"],["Rename phonemes","重命名音素","音素名を変更"],["One","一组","1 組の"],["pair per line, for labels that use a different name than the voice.","，用于标注与声库使用不同音素名称的情况。","を各行に記入します。ラベルの音素名が音声ライブラリと異なる場合に使用します。"],["Convert to a dataset","转换为数据集","データセットに変換"],["Train your voice","训练你的声库","音声ライブラリを学習"],["Training adjusts a copy of the voice so it sounds more like your recordings. Choose what to teach it and for how long, then press Train.","训练会调整声库副本，使声音更接近你的录音。选择要训练的模型和训练时长，然后点击“训练”。","学習では音声ライブラリのコピーを調整し、録音に近づけます。学習するモデルとステップ数を選び、「学習」を押してください。"],["What should it learn?","要训练哪些模型？","どのモデルを学習しますか？"],["Sound","声学模型","音響モデル"],["(acoustic)","（声学）","（音響）"],["How the voice sounds. Works with any dataset. Start here.","决定声音音色与音频细节。适用于各种数据集，建议从这里开始。","声質や音声の細部を担当します。どのデータセットでも使えます。まずはこちらから。"],["Timing","时长模型","長さモデル"],["(duration)","（时长）","（長さ）"],["How long each sound lasts. Needs notes in the dataset.","学习每个发音的时长分配。数据集需要包含音符。","各音素の長さを学習します。データセットに音符が必要です。"],["Pitch","音高模型","ピッチモデル"],["(melody and vibrato)","（旋律与颤音）","（メロディーとビブラート）"],["How the singer moves between notes. Needs notes in the dataset.","学习音符之间的音高变化。数据集需要包含音符。","音符間の音高変化を学習します。データセットに音符が必要です。"],["Vocoder","声码器","ボコーダー"],["(final audio stage)","（最终音频阶段）","（最終音声段階）"],["Optional and slower. Most people leave this off.","可选，训练速度较慢。通常建议先关闭。","任意項目です。学習に時間がかかるため、通常はオフのままにします。"],["F0 Analysis & Calibration","F0 分析与校准","F0 解析と調整"],["F0 options control only the analysis and cleanup of the recorded WAV's fundamental-frequency targets. They do not change Pitch-network loss settings.","F0 选项只控制录音 WAV 的基频分析与清理，不会更改 Pitch 网络的损失设置。","F0 の設定は録音 WAV の基本周波数の解析と補正のみを制御し、Pitch ネットワークの損失設定は変更しません。"],["Standard (65–1100 Hz, threshold 0.60)","标准（65–1100 Hz，阈值 0.60）","標準（65–1100 Hz、しきい値 0.60）"],["VOCALOID F0 cleanup (45–1400 Hz, threshold 0.45)","VOCALOID F0 清理（45–1400 Hz，阈值 0.45）","VOCALOID F0 補正（45–1400 Hz、しきい値 0.45）"],["Low register (45–650 Hz, threshold 0.60)","低音域（45–650 Hz，阈值 0.60）","低音域（45–650 Hz、しきい値 0.60）"],["High register (100–1400 Hz, threshold 0.60)","高音域（100–1400 Hz，阈值 0.60）","高音域（100–1400 Hz、しきい値 0.60）"],["Wide range (40–1600 Hz, threshold 0.60)","宽音域（40–1600 Hz，阈值 0.60）","広音域（40–1600 Hz、しきい値 0.60）"],["F0 analysis preset","F0 分析预设","F0 解析プリセット"],["F0 floor (Hz; overrides preset)","F0 最低频率（Hz；覆盖预设）","F0 下限（Hz、プリセットを上書き）"],["F0 ceiling (Hz; overrides preset)","F0 最高频率（Hz；覆盖预设）","F0 上限（Hz、プリセットを上書き）"],["Voicing threshold (0–1; overrides preset)","浊音阈值（0–1；覆盖预设）","有声音しきい値（0–1、プリセットを上書き）"],["F0 median smoothing","F0 中值平滑","F0 中央値平滑"],["Off","关闭","オフ"],["Light (15 ms)","轻度（15 ms）","弱（15 ms）"],["Medium (25 ms)","中度（25 ms）","中（25 ms）"],["Strong (35 ms)","强度（35 ms）","強（35 ms）"],["F0 reference correction (cents)","F0 参考校正（音分）","F0 基準音高補正（セント）"],["Correct isolated octave jumps","修正孤立八度跳点","孤立したオクターブ飛びを補正"],["仅修正前后有声帧一致、且中间帧疑似高/低 1–2 个八度的离群点；不会改写持续的八度变化。","Only repairs isolated frames that jump one or two octaves while neighboring voiced frames agree; sustained octave changes are preserved.","前後の有声音フレームが一致し、中間フレームだけが 1～2 オクターブ飛んでいる場合に補正します。継続的なオクターブ変化は保持します。"],["Pitch Model & Training","Pitch 模型与训练","Pitch モデルと学習"],["Pitch options control how the Pitch network is initialized and trained. They are independent of F0 detection, smoothing and reference calibration.","Pitch 选项控制 Pitch 网络的初始化和训练，与 F0 检测、平滑和参考校准相互独立。","Pitch の設定は Pitch ネットワークの初期化と学習を制御します。F0 検出、平滑化、基準補正とは独立しています。"],["Pitch model initialization","Pitch 模型初始化","Pitch モデルの初期化"],["Use Pitch model from selected base voice","使用所选底模中的 Pitch 模型","選択したベース音声の Pitch モデルを使用"],["Import compatible pretrained Pitch DNNI","导入兼容的预训练 Pitch DNNI","互換性のある事前学習 Pitch DNNI を読み込む"],["External pretrained Pitch model (.dnni)","外部预训练 Pitch 模型（.dnni）","外部の事前学習 Pitch モデル（.dnni）"],["Melody smoothing cutoff (Hz)","旋律平滑截止频率（Hz）","メロディー平滑化のカットオフ（Hz）"],["Smooth melody loss weight","平滑旋律损失权重","滑らかなメロディーの損失重み"],["Vibrato / fast-motion loss weight","颤音 / 快速变化损失权重","ビブラート／高速変化の損失重み"],["Train Pitch Model Only","仅训练 Pitch 模型","Pitch モデルのみ学習"],["当前仓库没有可验证的通用神经 Pitch 权重，因此不会把现有底模伪装成“通用预训练模型”。你可以使用底模 Pitch 或导入结构/音素表兼容的预训练 .dnni；内置 F0 平滑与校准属于通用信号处理，不是神经 Pitch 模型。","The repository does not include tested general-purpose neural Pitch weights. Use the selected base voice's Pitch model or import a compatible pretrained .dnni. Built-in F0 smoothing/calibration is general signal processing, not a neural Pitch model.","検証済みの汎用ニューラル Pitch 重みは同梱されていません。ベース音声の Pitch モデルを使うか、互換性のある事前学習済み .dnni を読み込んでください。内蔵の F0 平滑化・補正は一般的な信号処理であり、ニューラル Pitch モデルではありません。"],["How long?","训练时长","学習時間"],["Quick test","快速测试","クイックテスト"],["(200 steps)","（200 步）","（200 ステップ）"],["Just checks that everything works. Do this first.","仅检查整体流程是否正常，建议先运行。","動作確認用です。最初に実行してください。"],["Standard","标准","標準"],["(2,000 steps)","（2,000 步）","（2,000 ステップ）"],["A good starting point.","适合作为初始设置。","最初の設定として適しています。"],["Thorough","充分训练","じっくり学習"],["(6,000 steps)","（6,000 步）","（6,000 ステップ）"],["Slower. Gradually lowers the learning rate for a finer result.","耗时更长，并逐渐降低学习率以进行更细致的调整。","時間はかかりますが、学習率を徐々に下げて細かく調整します。"],["Name of your new voice","新声库名称","新しい音声の名前"],["Save the new voice to","新声库保存位置","新しい音声の保存先"],["Suggested automatically. Voices that start with info. are saved into a folder; others into a single file.","会自动建议路径。以 info. 开头的共享模型声库将保存为文件夹，其他声库保存为单个文件。","保存先は自動提案されます。info. で始まる共有モデル音声はフォルダーに、それ以外は単一ファイルとして保存されます。"],["Also save whenever it improves","验证效果改善时也保存","検証結果が改善するたびに保存"],["So you still have a voice if you stop early.","即使提前停止，也能保留较好的阶段结果。","途中で停止しても、良い状態の音声を残せます。"],["Advanced settings","高级设置","詳細設定"],["Leave these empty to use the defaults shown in grey.","留空则使用灰色提示中显示的默认值。","空欄の場合はグレー表示の既定値を使用します。"],["What to adjust","调整范围","調整対象"],["Everything (default)","全部参数（默认）","すべて（既定）"],["Only the voice's identity vectors","仅声库身份向量","音声の話者埋め込みのみ"],["The second option is quicker and gentler; used first when adding vocal modes.","第二种方式更快、更保守；新增 Vocal Mode 时可先用它。","2 番目の方法は速く、変更も控えめです。Vocal Mode を追加するときは先にこちらを使います。"],["Computer part to use","使用的计算设备","使用する計算デバイス"],["Automatic (default)","自动（默认）","自動（既定）"],["GPU (NVIDIA)","GPU（NVIDIA）","GPU（NVIDIA）"],["GPU (Apple)","GPU（Apple）","GPU（Apple）"],["CPU (slow)","CPU（较慢）","CPU（低速）"],["Keep recordings at their original loudness","保留录音原始响度","録音の元の音量を維持"],["By default they are scaled to the starting voice's level.","默认会将录音响度缩放到起始声库的水平。","既定では元の音声の音量レベルに合わせて調整します。"],["Continue the previous run","继续上次的训练","前回の学習を再開"],["Raise the number of steps above what was already done to train further.","将步数设置为高于已完成的步数，以继续训练。","学習を続けるには、ステップ数を完了済みの値より大きくしてください。"],["Learning rate","学习率","学習率"],["Warm-up steps","预热步数","ウォームアップのステップ数"],["Final learning rate","最终学习率","最終学習率"],["Stay close to the original","保持接近原始声库","元の音声に近づける"],["Higher keeps more of the original voice.","数值越高，越能保留原声库特征。","値を大きくすると元の音声の特徴が多く保持されます。"],["Phrases per step","每步短语数","1 ステップあたりのフレーズ数"],["Consonant emphasis","辅音权重","子音の重み"],["Above 1 makes it work harder on consonants.","大于 1 时会更强调辅音。","1 より大きい値で子音をより重視します。"],["Recordings held back for testing","留作测试的录音比例/数量","テスト用に保留する録音"],["A fraction (0.1 = 10%) or a count (5).","可以填写比例（0.1 = 10%）或数量（5）。","割合（0.1 = 10%）または件数（5）を指定します。"],["Random seed","随机种子","乱数シード"],["Use the same value when comparing results.","比较结果时请使用相同的值。","結果を比較するときは同じ値を使用してください。"],["Test every (steps)","每隔多少步评估","評価間隔（ステップ）"],["Report every (steps)","每隔多少步报告","ログ出力間隔（ステップ）"],["Phoneme map (JSON)","音素映射（JSON）","音素マップ（JSON）"],["Renames your dataset's phonemes to the voice's phonemes.","将数据集中的音素名转换为声库使用的名称。","データセットの音素名を音声ライブラリの音素名に変換します。"],["Vocal modes","Vocal Mode","Vocal Mode"],["Info voices only. Comma-separated.","仅适用于 info 声库，多个名称用逗号分隔。","info 形式の音声のみ。複数はカンマ区切りです。"],["Version name","版本名称","バージョン名"],["Info voices only.","仅适用于 info 声库。","info 形式の音声のみ。"],["Also save the last step as version","将最后一步另存为版本","最終ステップも別バージョンとして保存"],["Loudness reference voice","响度参考声库","音量基準の音声"],["Use the original voice when continuing from an earlier finetune.","从之前的微调继续训练时，使用最初的声库作为响度参考。","以前の微調整から続ける場合、元の音声を音量基準として使用します。"],["Fixed loudness factor","固定响度系数","固定音量係数"],["Run folder","运行目录","実行フォルダー"],["Keeps the log and lets you continue later.","保存日志与训练状态，方便之后继续。","ログと学習状態を保存し、後で再開できるようにします。"],["Save test scores to a file (JSON)","将测试评分保存为文件（JSON）","テストスコアをファイルに保存（JSON）"],["Check the result","检查结果","結果を確認"],["Compare the original voice with your new one on recordings that were held back from training. You can listen to each next to the real recording. Lower scores are closer to the recording.","在未参与训练的录音上比较原始声库和新声库。你可以将生成音频与真实录音并排试听。评分越低，通常代表越接近录音。","学習に使っていない録音で元の音声と新しい音声を比較します。生成音声を実際の録音と並べて試聴できます。スコアが低いほど録音に近いことを示します。"],["Your new voice","你的新声库","新しい音声"],["Use the voice I just trained","使用刚训练完成的声库","今学習した音声を使用"],["Save the comparison to","将比较结果保存到","比較結果の保存先"],["Recordings to render","要渲染的录音数","レンダリングする録音数"],["Each one is rendered by both voices so you can listen.","每条录音都会由两个声库分别渲染，供你试听比较。","それぞれの録音を両方の音声でレンダリングし、比較試聴できます。"],["Compare voices","比较声库","音声を比較"],["Show saved results","显示已保存的结果","保存済みの結果を表示"],["Export Trained Voicebank","导出训练后的声库","学習済み音声ライブラリをエクスポート"],["SVPK 当前仅为格式研究诊断包。全零字段版本在 Sharp 中无反应；把 Hanakuma Chifuyu 的 256 字节字段复制到自制包后，Sharp 能识别包但提示数据损坏。这强烈暗示该字段参与包专属完整性/认证校验（RSA-2048 签名只是未证实的假设）。当前生成结果不可安装或发布。SFPK 为实验性 Flat 格式导出：会将模型依赖写入所选本机 models 目录，包内仅含 config.json 与 info.nofs，暂非自包含包，必须在 Flat 中实际验证。","SVPK is currently a format-research diagnostic package only. The all-zero field was ignored by Sharp; copying Hanakuma Chifuyu's 256-byte field made the custom package recognizable but produced a corruption warning. The field may participate in package-specific integrity/authentication checks, but the algorithm and signature process are unverified. Generated SVPK files are not installable or suitable for release. SFPK export is experimental: model dependencies are written to a selected local models directory, while the package contains config.json and info.nofs. Verify it directly in Flat.","SVPK は現在、形式調査用の診断パッケージのみです。全ゼロのフィールドは Sharp で無視され、花隈千冬の 256 バイトフィールドを複製したパッケージは認識されましたが、破損警告が表示されました。フィールドはパッケージ固有の整合性／認証チェックに関与する可能性がありますが、方式や署名手順は未確認です。生成した SVPK はインストールや公開に使用できません。SFPK 出力も実験段階で、モデル依存ファイルはローカルの models フォルダーに置かれ、パッケージ内には config.json と info.nofs が含まれます。Flat で直接検証してください。"],["Export Voicebank / Choose Package Format","导出声库 / 选择打包格式","音声ライブラリをエクスポート／形式を選択"],["Handy extras. You do not need these for a normal first run.","一些实用的附加工具，首次训练通常不需要。","便利な補助ツールです。初回の学習では通常必要ありません。"],["Make practice data","生成练习数据","練習データを生成"],["Renders a small labelled dataset using your chosen voice. Useful to test the whole process without any recordings. When it finishes, it becomes your dataset.","使用所选声库渲染一个小型带标签数据集。无需录音即可测试完整流程；生成后会自动设为当前数据集。","選択した音声で小規模なラベル付きデータセットをレンダリングします。録音がなくても一連の処理を試せ、完了すると現在のデータセットになります。"],["Save practice data to","练习数据保存位置","練習データの保存先"],["Phrases","短语数","フレーズ数"],["Notes per phrase","每条短语的音符数","各フレーズの音符数"],["Convert a dictionary","转换音素字典","辞書を変換"],["Rewrites a DiffSinger dataset's spelling to Synthesizer V's dictionary for its language. The original is untouched. Uses the voice and language from step 1.","将 DiffSinger 数据集中的音素拼写转换为对应语言的 Synthesizer V 字典，不会修改原始数据。使用第 1 步选择的声库和语言。","DiffSinger データセットの音素表記を、その言語の Synthesizer V 辞書に合わせて変換します。元データは変更せず、手順 1 の音声と言語を使用します。"],["Source dataset","源数据集","変換元データセット"],["Save converted dataset to","转换后数据集保存位置","変換後データセットの保存先"],["Source dictionary","源字典","変換元の辞書"],["Synthesizer V dictionary","Synthesizer V 字典","Synthesizer V の辞書"],["Share of a syllable given to its ending","音节韵尾时长占比","音節末尾に割り当てる割合"],["Convert dictionary","转换字典","辞書を変換"],["Stop","停止","停止"],["Hide details","隐藏详情","詳細を隠す"],["Show details","显示详情","詳細を表示"],["Full log","完整日志","ログ全体"],["Copy log","复制日志","ログをコピー"],["Choose Voicebank Package Format","选择声库打包格式","音声ライブラリのパッケージ形式を選択"],["SVPK 目前只生成诊断包，不能安装。SFPK 提供实验性导出：生成 config.json + info.nofs，并将四类 .dnni 依赖写入所选本机 models 目录；不会把模型二进制嵌入 SFPK，迁移到其他电脑时需另行安装对应模型。","SVPK currently produces a diagnostic package only and cannot be installed. SFPK export is experimental: it creates config.json and info.nofs, and writes four types of .dnni dependencies to the selected local models directory. Model binaries are not embedded in SFPK, so corresponding models must be installed separately on another computer.","SVPK は診断パッケージのみ生成でき、インストールできません。SFPK 出力は実験段階で、config.json と info.nofs を作成し、4 種類の .dnni 依存ファイルを指定したローカルの models フォルダーに書き込みます。モデル本体は SFPK に埋め込まれないため、別の PC では対応モデルを別途インストールする必要があります。"],["Build SVPK Diagnostic Package (Not Installable)","生成 SVPK 诊断包（不可安装）","SVPK 診断パッケージを作成（インストール不可）"],["仅用于检查内部 ZIP 与元数据结构，不能导入或发布为可用声库。","Only checks the internal ZIP and metadata structure. It cannot be imported or released as a usable voicebank.","内部 ZIP とメタデータ構造の確認専用です。使用可能な音声としてインポート／公開することはできません。"],["Export SFPK Package (Experimental)","导出 SFPK 包（实验性）","SFPK パッケージをエクスポート（実験段階）"],["可输入训练后的 voice.nofs 或已有 info.*.nofs；需选择 Synthesizer V 的 models 目录。","可选择训练后的 voice.nofs 或已有的 info.*.nofs；需要指定 Synthesizer V 的 models 目录。","学習済み voice.nofs または既存の info.*.nofs を指定できます。Synthesizer V の models フォルダーが必要です。"],["Cancel","取消","キャンセル"],["SVPK Diagnostic Package (Not Installable)","SVPK 诊断包（不可安装）","SVPK 診断パッケージ（インストール不可）"],["无需手动编辑 JSON。图片请选择 PNG 文件。当前仅生成诊断包：全零字段会被 Sharp 忽略；复制 Chifuyu 字段会触发“数据损坏”。这表明字段可能参与包专属完整性/认证校验，算法和签署流程尚未确认，因此此功能不会生成可安装包。","No need to edit JSON manually. Select PNG images. This tool creates a diagnostic package only: Sharp ignored the all-zero field, while copying the Chifuyu field triggered a corruption warning. The field may be used for package-specific integrity/authentication checks, but the algorithm and signing process are unconfirmed, so this does not produce an installable package.","JSON を手動編集する必要はありません。画像は PNG を選択してください。この機能は診断パッケージのみを生成します。全ゼロのフィールドは Sharp に無視され、千冬のフィールドを複製すると破損警告が出ました。専用の整合性／認証チェックに使われている可能性がありますが、方式と署名手順は未確認のため、インストール可能なパッケージは生成しません。"],["Build Diagnostic Package (Not Installable)","生成诊断包（不可安装）","診断パッケージを作成（インストール不可）"],["Build SFPK Test Package","生成 SFPK 测试包","SFPK テストパッケージを作成"],["普通 voice.nofs 将转换为 Flat 使用的 info.nofs 描述，并把 acoustic / vocoder / duration / f0 的 .dnni 文件写入你选定的本机 models 目录。SFPK 内包含 config.json 与 info.nofs；选填的 profile.png / background.png 会一起放入包内，但模型二进制仍在本机 models 目录。当前尚未验证 Flat 安装和渲染。","A regular voice.nofs is converted to an info.nofs descriptor for Flat. The acoustic, vocoder, duration, and f0 .dnni files are written to your selected local models directory. SFPK contains config.json and info.nofs, and optionally profile.png/background.png; model binaries remain in the local models directory. Flat installation and rendering are not yet verified.","通常の voice.nofs は Flat 用の info.nofs 記述に変換されます。acoustic／vocoder／duration／f0 の .dnni ファイルは選択したローカルの models フォルダーに書き込まれます。SFPK には config.json と info.nofs が含まれ、任意で profile.png／background.png も格納されますが、モデル本体はローカルに残ります。Flat でのインストールとレンダリングは未検証です。"],["Voicebank name","声库名称","音声ライブラリ名"],["Vendor / author","供应商 / 作者","ベンダー／作者"],["Voicebank version (integer)","声库版本（整数）","音声バージョン（整数）"],["256-byte field diagnostic preset","256 字节字段诊断预设","256 バイトフィールドの診断プリセット"],["Avatar image (profile.png)","头像图片（profile.png）","アバター画像（profile.png）"],["Background image (background.png)","背景图片（background.png）","背景画像（background.png）"],["SVPK output path","SVPK 输出路径","SVPK の出力先"],["Voicebank name (optional)","声库名称（可选）","音声ライブラリ名（任意）"],["Author / vendor (optional)","作者 / 供应商（可选）","作者／ベンダー（任意）"],["Voicebank version (optional)","声库版本（可选）","音声バージョン（任意）"],["Avatar image profile.png (optional)","头像图片 profile.png（可选）","アバター画像 profile.png（任意）"],["Background image background.png (optional)","背景图片 background.png（可选）","背景画像 background.png（任意）"],["Synthesizer V models directory","Synthesizer V 模型目录","Synthesizer V のモデルフォルダー"],["SFPK output path","SFPK 输出路径","SFPK の出力先"],["A file named voice.nofs or info.<version>.nofs.","文件名为 voice.nofs 或 info.<version>.nofs。","voice.nofs または info.<version>.nofs という名前のファイルです。"],["Each song.wav needs a matching song.lab file.","每个 song.wav 都需要对应的 song.lab 文件。","各 song.wav には対応する song.lab が必要です。"],["Use spaces between phones, e.g. @_n 7_0.wav. For Mandarin VOCALOID labels, TrainSV converts source phones to the selected voice inventory automatically. Only the final _0/_1 retake suffix is removed.","音素之间用空格分隔，例如 @_n 7_0.wav。对于普通话 VOCALOID 标注，TrainSV 会自动转换为所选声库的音素表。只会删除末尾的 _0/_1 重录后缀。","音素はスペースで区切ります（例：@_n 7_0.wav）。中国語 VOCALOID ラベルは選択した音声の音素一覧へ自動変換します。末尾の _0／_1 のリテイク番号のみ削除します。"],["Usually not needed when a voice is selected above. If omitted, keep “Mark word boundaries using my voice” enabled so TrainSV can read the selected voice phone set.","通常选择上方声库后无需填写。若留空，请启用“使用所选声库标记词边界”，使 TrainSV 能读取该声库的音素表。","上で音声を選択していれば通常不要です。空欄にする場合は「選択した音声で単語境界を付与」を有効にし、音素一覧を読み込めるようにしてください。"],["When it finishes, this becomes your dataset automatically.","完成后会自动将其设为当前数据集。","完了すると自動的に現在のデータセットになります。"],["Comma-separated names. Leave empty to use everything.","多个名称以逗号分隔；留空表示全部导入。","複数の名前はカンマで区切ります。空欄の場合はすべてを使用します。"],["For example opencpop-extension.txt","例如 opencpop-extension.txt","例：opencpop-extension.txt"],["For example mandarin-xsampa-dict.txt","例如 mandarin-xsampa-dict.txt","例：mandarin-xsampa-dict.txt"],["Output packages are research/experimental only.","输出包仅用于研究或实验。","出力パッケージは研究／実験用途のみです。"],["此预设只控制输入 WAV 的 F0 检测范围与目标曲线清理，不会更改 Pitch 模型的损失函数参数。预设只是起点，建议检查训练日志中的有声帧比例和中位 F0；范围放宽也可能增加错误音高候选。","This preset only controls F0 detection range and target-curve cleanup for the input WAV; it does not change Pitch model loss settings. Treat it as a starting point, check voiced-frame ratio and median F0 in the log, and note that wider ranges can create false pitch candidates.","このプリセットは入力 WAV の F0 検出範囲と目標曲線の補正のみを制御し、Pitch モデルの損失設定は変更しません。開始点として使い、ログの有声音フレーム率と F0 中央値を確認してください。範囲を広げると誤検出が増える可能性があります。"],["低于此频率的音高不在检测范围内；应根据歌手最低音设置。","Pitches below this frequency are outside the detection range. Set it based on the singer's lowest note.","この周波数を下回る音高は検出範囲外です。歌手の最低音に合わせて設定してください。"],["高于此频率的音高不在检测范围内；过高也可能增加谐波误检。","Pitches above this frequency are outside the detection range; setting it too high may increase harmonic errors.","この周波数を超える音高は検出範囲外です。高くしすぎると倍音の誤検出が増えることがあります。"],["降低可能减少漏检，也可能把噪声误判为有声；留空时使用所选预设值。","A lower threshold may reduce missed pitches but can classify noise as voiced. Leave blank to use the selected preset.","しきい値を下げると取りこぼしが減る場合がありますが、ノイズを有声音と誤認する可能性があります。空欄なら選択したプリセットを使います。"],["在连续有声片段内对 log2(F0) 中值滤波；VOCALOID F0 cleanup 预设默认启用 25 ms 平滑来压制短促跳点，强度过高也可能压低真实颤音细节。","Applies a median filter to log2(F0) within continuous voiced regions. VOCALOID F0 cleanup uses 25 ms by default to reduce brief jumps; stronger smoothing may suppress real vibrato detail.","連続した有声音区間内で log2(F0) に中央値フィルターを適用します。VOCALOID F0 補正では短い跳ねを抑えるため、既定で 25 ms を使用します。強すぎる平滑化は実際のビブラートも弱める可能性があります。"],["在缓存后的 F0 参考曲线上整体校正音分。若检测结果测得高约 25 音分，可先试 -25。不能修复错误八度、错误音符或对齐问题。范围 -1200 到 +1200。","Applies an overall cents offset to the cached F0 reference. If the measured pitch is about 25 cents sharp, try -25. It cannot fix octave errors, wrong notes or alignment problems. Range: -1200 to +1200.","キャッシュした F0 基準に全体のセント補正を適用します。測定音高が約 25 セント高い場合は -25 を試してください。オクターブ誤り、音符の誤り、位置合わせの問題は修正できません。範囲：-1200～+1200。"],["默认使用当前底模的 Pitch 模型；导入外部模型前会检查模型结构、音素表和说话人向量维度。","Uses the Pitch model from the selected base voice by default. External models are checked for architecture, phoneme inventory and speaker-vector dimensions.","既定では選択したベース音声の Pitch モデルを使用します。外部モデルはアーキテクチャ、音素一覧、話者ベクトルの次元を確認します。"],["仅在选择导入预训练模型时需要。必须是 TrainSV 当前支持的 Pitch DNNI，并且音素表与底模匹配。","Required only when importing a pretrained model. The Pitch DNNI must be supported by TrainSV and its phoneme inventory must match the base voice.","事前学習モデルを読み込む場合のみ必要です。TrainSV が対応する Pitch DNNI であり、音素一覧がベース音声と一致している必要があります。"],["用于 Pitch 训练损失中区分慢旋律轮廓和快速音高运动的低通截止频率；不会直接对导出的 WAV 做后期平滑。","Low-pass cutoff that separates the smooth melody contour from fast pitch motion in the Pitch training loss. It does not post-process the exported WAV.","Pitch の学習損失で滑らかなメロディー輪郭と高速な音高変化を分けるローパスカットオフです。出力 WAV を直接平滑化するものではありません。"],["控制 Pitch 模型学习整体音符走向的权重。数值越大，越重视较平滑的旋律误差；设为 0 可关闭该损失分量。","Controls how strongly the Pitch model learns the overall melody contour. Higher values emphasize smoother melody error; 0 disables this loss component.","Pitch モデルが全体のメロディー輪郭を学習する重みです。値が大きいほど滑らかなメロディーの誤差を重視し、0 でこの損失項を無効にします。"],["控制 Pitch 模型学习颤音和快速音高运动的权重。太高可能使模型把原始合成器的抖动也学进去。至少一个 Pitch loss 权重必须大于 0。","Controls how strongly Pitch learns vibrato and fast pitch motion. Too high a value may teach it unwanted synthesizer jitter. At least one Pitch loss weight must be above 0.","Pitch がビブラートや高速な音高変化を学ぶ重みです。高すぎると、元のシンセサイザーの揺れまで学習する可能性があります。Pitch の損失重みは少なくとも一つを 0 より大きくする必要があります。"],["F0 中值平滑只在连续有声片段内部工作，不会把平滑结果跨过静音或辅音。强度越大越可能压低原有颤音细节。","F0 median smoothing only works inside continuous voiced regions and never crosses silence or consonants. Stronger smoothing may reduce real vibrato detail.","F0 の中央値平滑化は連続した有声音区間内だけで動作し、無音や子音をまたぎません。強くすると実際のビブラートの細部が弱まることがあります。"],["默认使用当前底模的 Pitch 网络。选择外部预训练时，必须提供兼容的 .dnni 文件；TrainSV 会严格检查网络树、音素表、归一化和稀疏掩码，不兼容时会拒绝。当前仓库没有随包附带经过验证的通用神经 Pitch 权重。","Uses the base voice's Pitch network by default. Imported .dnni files are checked for network structure, phoneme inventory, normalization and sparse masks; incompatible files are rejected. No verified general-purpose neural Pitch weights are bundled.","既定ではベース音声の Pitch ネットワークを使用します。読み込んだ .dnni はネットワーク構造、音素一覧、正規化、スパースマスクを検査し、互換性がなければ拒否します。検証済みの汎用ニューラル Pitch 重みは同梱されていません。"],["训练总步数。数值越大，训练通常越久；更多步数不保证效果更好。可先用快速测试确认环境和数据正常，再逐步增加。","Total training steps. More steps usually take longer and do not guarantee better results. Run a quick test first, then increase gradually.","学習ステップ数です。増やすと通常は時間がかかり、品質が必ず向上するわけではありません。まずクイックテストを行い、徐々に増やしてください。"],["为输出声库设置显示名称。建议使用容易识别的名称；该名称也会用于自动生成部分输出路径。","Sets the display name of the output voicebank and is also used to generate some output paths.","出力音声ライブラリの表示名を設定します。一部の保存先パスにも使用されます。"],["指定训练后声库的保存位置。普通 voice.nofs 声库保存为文件；info.*.nofs 类型的共享模型声库会按工具规则保存到目录中。","Output path for the trained voicebank. Regular voice.nofs outputs are files; shared-model info.*.nofs outputs are saved to a folder.","学習後の音声ライブラリの保存先です。通常の voice.nofs はファイルとして、共有モデル形式の info.*.nofs はフォルダーとして保存されます。"],["学习率，控制每次更新模型参数时的调整幅度。默认值取决于“调整内容”：全部权重通常为 2e-5，仅训练身份向量时通常为 3e-3。过高可能不稳定，过低可能进展缓慢。","Learning rate controls the size of each parameter update. Defaults depend on the parameter scope: usually 2e-5 for all weights and 3e-3 for identity vectors only. Too high may destabilize training; too low may slow it down.","学習率は各パラメータ更新の大きさを制御します。既定値は対象によって異なり、全重みでは通常 2e-5、話者ベクトルのみでは通常 3e-3 です。高すぎると不安定になり、低すぎると進行が遅くなります。"],["F0 预设只控制输入 WAV 的基频检测范围与目标曲线清理，不会更改 Pitch 模型的损失函数参数。预设只是起点，建议检查训练日志中的有声帧比例和中位 F0；范围放宽也可能增加错误音高候选。","The F0 preset controls only input-WAV tracking range and target cleanup; it does not change Pitch loss settings. Check the voiced-frame ratio and median F0 in the training log. A wider range can increase false candidates.","F0 プリセットは入力 WAV の検出範囲と目標曲線の補正のみを制御し、Pitch の損失設定は変更しません。学習ログで有声音フレーム率と F0 中央値を確認してください。範囲を広げると誤検出が増えることがあります。"],["对从 WAV 检测并经过基本清理的 F0 参考曲线整体做音分校正。负值降低参考音高，正值提高。建议先测量实际偏差并小幅调整；不应用于修正八度跳变、错误音符或时间对齐。","Applies a cents offset to the cleaned F0 reference curve. Negative values lower the pitch and positive values raise it. Measure the bias first and adjust gradually; this does not fix octave jumps, wrong notes or alignment.","補正済みの F0 基準曲線にセント単位のオフセットを適用します。負の値で音高を下げ、正の値で上げます。まず偏差を測り、少しずつ調整してください。オクターブ飛び、音符誤り、位置合わせは修正できません。"],["低通截止频率，用于 Pitch 训练损失里分离平滑旋律轮廓和快速变化。较低会更强调旋律大轮廓，较高会把更多快速变化纳入旋律项。这不是直接平滑最终音频；它独立于 F0 检测预设。","Low-pass cutoff used by Pitch loss to separate smooth melody contour from fast motion. Lower values emphasize the broad melody; higher values include more rapid variation. It does not smooth the final audio and is independent of F0 presets.","Pitch 損失で滑らかなメロディー輪郭と高速変化を分けるローパスカットオフです。低い値は大きなメロディー輪郭を、高い値はより速い変化も重視します。出力音声を平滑化するものではなく、F0 プリセットとは独立しています。"],["平滑旋律轮廓损失权重，表示 Pitch 训练更重视整体旋律曲线还是其他音高细节。请结合验证集的 pitch_semitones/pitch_melody 和试听设置。","Weight of the smooth melody contour loss. Use validation metrics pitch_semitones/pitch_melody and listening tests to decide how much to emphasize overall melody versus pitch detail.","滑らかなメロディー輪郭の損失重みです。全体のメロディーと音高の細部のどちらを重視するか、検証指標 pitch_semitones／pitch_melody と試聴で判断してください。"],["快速变化/颤音损失权重。提高它可能保留更多细节，也可能学到不需要的抖动；初始建议保留 1.0。","Weight of fast motion and vibrato loss. Higher values may preserve detail but can also learn unwanted jitter; start with 1.0.","高速変化／ビブラートの損失重みです。大きくすると細部を保持する一方、不要な揺れまで学習する可能性があります。まずは 1.0 を推奨します。"],["预热步数。大于 0 时，学习率会在指定步数内逐渐升高；留空或为 0 时不启用预热。","Warm-up steps gradually increase the learning rate during this many steps. Leave blank or set 0 to disable warm-up.","ウォームアップのステップ数です。0 より大きい場合、このステップ数の間に学習率を徐々に上げます。空欄または 0 なら無効です。"],["最终学习率。填写后，训练会按余弦衰减逐步降至该值；留空时学习率保持不变（除非设置了预热）。","Final learning rate. When set, the rate decays toward this value using cosine scheduling; leave blank to keep it constant (unless warm-up is set).","最終学習率です。指定するとコサインスケジュールでこの値まで下げます。空欄なら（ウォームアップ以外は）一定です。"],["控制微调权重向原始声库权重靠拢的强度。值越大，约束通常越强、越能保留原声库特征；具体效果仍取决于数据和训练设置。","Controls how strongly fine-tuned weights are pulled toward the original voice. Higher values usually preserve more original characteristics; results depend on data and settings.","微調整した重みを元の音声に引き寄せる強さを制御します。値が大きいほど元の特徴を保ちやすくなりますが、結果はデータと設定によります。"],["每一步累计计算的短语数量；音频训练使用对应音频窗口。增加批次通常会占用更多显存，显存不足时可尝试减小，常从 1 开始。","Number of phrases accumulated per step (audio models use audio windows). Larger batches usually need more GPU memory. Reduce this value if memory is insufficient; start at 1.","各ステップでまとめて計算するフレーズ数です（音声モデルは音声ウィンドウを使用）。バッチを増やすと通常は VRAM を多く使います。不足する場合は減らし、まず 1 から始めてください。"],["辅音帧在训练音频损失中的权重。大于 1 会更强调辅音；留空使用程序默认值。验证集评分不应用此额外权重。","Weight of consonant frames in the training audio loss. Values above 1 emphasize consonants. Leave blank for the default; validation scores do not use this extra weight.","学習時の音声損失における子音フレームの重みです。1 より大きいと子音を重視します。空欄なら既定値で、検証スコアにはこの追加重みを使いません。"],["留作验证、不会用于训练的短语比例或数量。小于 1 的数值表示比例（例如 0.1 约为 10%）；大于或等于 1 时按数量解释。设为 0 可不留验证数据。","Fraction or count of phrases held out for validation, not training. Values below 1 are fractions (0.1 ≈ 10%); values ≥1 are counts. Use 0 for no validation set.","検証用に保留し、学習には使わないフレーズの割合または件数です。1 未満は割合（0.1 は約 10%）、1 以上は件数です。0 で検証データを使いません。"],["随机种子，用于控制随机数据划分等步骤。比较不同设置时尽量使用相同的种子；更改种子可能会改变验证集。","Random seed for operations such as splitting data. Use the same seed when comparing settings; changing it may change the validation set.","データ分割などの乱数シードです。設定を比較するときは同じ値を使ってください。変更すると検証セットが変わる場合があります。"],["每隔多少训练步数评估一次验证集。间隔较小能更频繁观察效果，但会增加评估开销。","How often to evaluate the validation set in training steps. Smaller intervals give faster feedback but add evaluation overhead.","何ステップごとに検証セットを評価するかを指定します。間隔を短くすると早く結果が分かりますが、評価の負荷が増えます。"],["每隔多少训练步数输出一次训练日志。它主要影响日志的详细程度，不直接改变训练目标。","How often to print training logs. This mainly changes log detail and does not directly change the training objective.","何ステップごとに学習ログを出力するかを指定します。主にログの詳細度が変わり、学習目標自体は変わりません。"],["可选的 JSON 音素映射文件，用于把数据集中的音素名称转换成当前声库使用的名称。映射不匹配时可能导致训练数据无法正确识别。","Optional JSON phoneme map to rename dataset phonemes to match the current voice. Incorrect mappings can prevent the data from being recognized properly.","データセットの音素名を現在の音声に合わせて変更する JSON マップです。対応が間違っているとデータを正しく認識できない場合があります。"],["可选：设置要训练的 Vocal Mode 名称，多个名称用逗号分隔。此功能主要适用于 info 声库；新增模式时通常先用“仅身份向量”训练，再进行完整微调。","Optional Vocal Mode names to train, comma-separated. Mainly applies to info voicebanks; when adding a mode, consider training identity vectors only before full fine-tuning.","学習する Vocal Mode 名をカンマ区切りで指定します。主に info 形式の音声向けです。モード追加時は「話者埋め込みのみ」で学習してから全体を微調整する方法があります。"],["可选：指定 info 声库输出使用的版本名称。仅适用于 info.*.nofs 类型声库。","Optional output version name for info voicebanks only.","info 形式の音声にのみ適用される出力バージョン名です（任意）。"],["可选：为最后一步的权重另存一个版本。填写要使用的版本标识；它与验证过程中保存的最佳权重版本不同。仅适用于 info 声库。","Optionally save the final-step weights as a named version. This differs from the best-validation checkpoint and applies only to info voicebanks.","最終ステップの重みを別バージョンとして保存します。検証で保存された最良の重みとは異なり、info 形式の音声のみで使用できます。"],["可选的响度参考声库。连续微调时可指定最初的基础声库，避免每次训练都以微调后的声音重新定标。","Optional loudness-reference voice. When fine-tuning repeatedly, use the original base voice to avoid recalibrating loudness from each intermediate result.","任意の音量基準音声です。繰り返し微調整する場合、元のベース音声を指定すると中間結果ごとに音量が再調整されるのを防げます。"],["可选：手动固定录音响度缩放系数。留空时由程序测量；只有在明确知道需要的缩放幅度时再填写。","Optional fixed recording gain. Leave blank for automatic measurement; fill it only when you know the required scaling factor.","録音の音量倍率を固定する場合に指定します。空欄なら自動測定されるため、必要な倍率が分かっているときだけ入力してください。"],["训练运行目录，用来保存日志和可恢复的训练状态。启用后，之后可以尝试从该目录继续训练。","Run folder for logs and resumable training state. You can later try to continue training from this folder.","ログと再開用の状態を保存する実行フォルダーです。後でこのフォルダーから学習を再開できます。"],["可选：将验证评分历史写入 JSON 文件，便于比较多次训练结果或进一步分析。","Optional JSON file for validation-score history, useful for comparing runs and further analysis.","検証スコアの履歴を JSON ファイルに保存します（任意）。複数回の学習結果の比較や分析に役立ちます。"],["选择训练后生成的新声库文件。对比时会将它与第 1 步选择的原始声库进行评估。","Select the newly trained voicebank file. Evaluation compares it with the original selected in step 1.","学習後の新しい音声ファイルを選択します。手順 1 で選んだ元の音声と比較評価します。"],["指定评估结果的保存目录。结果包含 metrics.json，以及用于试听对比的渲染音频（如果成功生成）。","Folder for evaluation results, including metrics.json and rendered audio for listening comparisons when available.","評価結果の保存先です。metrics.json と、生成に成功した場合は比較試聴用の音声が含まれます。"],["最多渲染多少条留出的短语用于试听比较。数量越多，评估与渲染可能越耗时。","Maximum number of held-out phrases to render for listening comparisons. More phrases can take longer to evaluate and render.","比較試聴用にレンダリングする保留フレーズの最大数です。多いほど評価とレンダリングに時間がかかります。"],["指定合成练习数据的保存目录。生成完成后，该数据集会自动设为当前数据集。","Folder for generated practice data. It becomes the current dataset when generation finishes.","生成した練習データの保存先です。作成後は自動的に現在のデータセットになります。"],["要合成的短语数量。数量越多，生成测试数据所需时间和磁盘空间通常越多。","Number of phrases to synthesize. More phrases usually need more time and disk space.","合成するフレーズ数です。増やすほど通常は時間とディスク容量が必要になります。"],["每条练习短语包含的音符数量。它影响合成数据的长度与复杂程度。","Number of notes in each practice phrase. Affects the length and complexity of the generated data.","各練習フレーズに含める音符数です。生成データの長さと複雑さに影響します。"],["选择需要转换音素拼写的数据集目录。原始数据集不会被直接修改。","Dataset folder whose phoneme spelling should be converted. The original dataset is not modified.","音素表記を変換するデータセットのフォルダーを選択します。元のデータセットは変更されません。"],["指定转换后的数据集保存目录。建议使用新目录，以便保留原始数据并比较结果。","Output folder for the converted dataset. Use a new folder to preserve the original and compare results.","変換後のデータセットの保存先です。元データを残して比較できるよう、新しいフォルダーを推奨します。"],["选择源数据集使用的字典文本文件，用于解释原有音素拼写。请确保它与源数据集的标注体系相匹配。","Source dictionary text file defining the dataset's current phoneme spelling. It must match the source annotation system.","元データセットの音素表記を定義する変換元辞書です。元のラベル体系に一致している必要があります。"],["选择目标 Synthesizer V 声库对应的字典文本文件。目标字典需要与所选声库及语言相匹配。","Target dictionary file for Synthesizer V. It must match the selected voicebank and language.","Synthesizer V 用の変換先辞書です。選択した音声ライブラリと言語に合っている必要があります。"],["控制一个音节中分配给韵尾的时长比例。只在特定字典转换场景中需要调整；不确定时建议先使用默认值。","Share of syllable duration assigned to the coda. Usually only needs adjustment in specific dictionary conversions; use the default if unsure.","音節の長さのうち末尾子音に割り当てる割合です。特定の辞書変換でのみ調整し、不明な場合は既定値を使ってください。"],["声库对外显示的名称，会写入 SVPK 的 package-config 元数据。","Display name written to the SVPK package-config metadata.","SVPK の package-config メタデータに書き込む表示名です。"],["声库供应商、作者或工作室名称，会写入 package-config 的 vendorName。","Vendor, author or studio name written to vendorName in package-config.","package-config の vendorName に記録するベンダー／作者／スタジオ名です。"],["SVPK 元数据要求 versionString（文本）和 versionNumber（整数）。这个输入使用整数版本，例如 101，并将同一版本写入两个字段。","SVPK metadata requires versionString (text) and versionNumber (integer). Enter an integer such as 101; it will be written to both fields.","SVPK メタデータには versionString（文字列）と versionNumber（整数）が必要です。101 などの整数を入力すると両方のフィールドに記録されます。"],["选择要写入外层 256 字节字段的诊断数据：全零、随机字节、SHA-256 链式扩展或合成 AES-256-CBC。非零模式只是测试数据，不是 Synthesizer V 官方签名/密钥；所有模式生成的文件仍不可安装。","Select diagnostic data for the outer 256-byte field: zeros, random bytes, a synthetic SHA-256 chain, or synthetic AES-256-CBC. Non-zero modes are test data, not official Synthesizer V signatures or keys; none creates an installable file.","外側の 256 バイトフィールドに書き込む診断データを選択します（ゼロ、ランダム、合成 SHA-256 チェーン、合成 AES-256-CBC）。ゼロ以外はテストデータであり、Synthesizer V 公式の署名や鍵ではありません。どの方式でもインストール可能なファイルは作成されません。"],["选择声库头像 PNG。打包器会将其作为根目录 profile.png 写入包内。","Choose the voicebank avatar PNG. The packager stores it as profile.png in the package root.","音声ライブラリのアバター PNG を選択します。パッケージのルートに profile.png として格納されます。"],["选择声库背景 PNG。为匹配目前检查过的样例，包内文件条目使用其样例拼写 background.png。","Choose the voicebank background PNG. The package entry is named background.png to match the inspected sample.","音声ライブラリの背景 PNG を選択します。確認済みサンプルに合わせて background.png として格納します。"],["生成的 SVPK 文件完整路径。推荐保存在训练结果附近，并使用 .svpk 扩展名。","Full path for the generated SVPK. Saving it near the training output with a .svpk extension is recommended.","生成する SVPK のフルパスです。学習結果の近くに .svpk 拡張子で保存することを推奨します。"],["选择声库头像 PNG 文件；打包后将按样例名称存放在 profile.png。","Select the voicebank avatar PNG; it will be stored as profile.png in the package.","音声ライブラリのアバター PNG を選択します。パッケージ内では profile.png として保存されます。"],["选择声库背景 PNG 文件；当前会按样例将文件条目命名为 background.png。","Select the background PNG; the package entry will be named background.png to match the sample.","背景 PNG を選択します。サンプルに合わせて background.png として保存されます。"],["输出的 .svpk 文件路径。默认与训练后的 voice.nofs 放在同一目录。","Path for the output .svpk file. By default it is saved next to the trained voice.nofs.","出力する .svpk ファイルのパスです。既定では学習済み voice.nofs と同じフォルダーに保存します。"],["留空时尽量保留来源声库名称；填写后覆盖。","When blank, keeps the source voicebank name where possible; filling it overrides the name.","空欄なら可能な限り元の音声名を保持し、入力すると上書きします。"],["留空时保留来源声库的作者信息；普通 voice.nofs 未提供时会使用 TrainSV。","When blank, keeps the source author info. For a regular voice.nofs without author metadata, TrainSV is used.","空欄なら元の作者情報を保持します。通常の voice.nofs に情報がない場合は TrainSV を使用します。"],["留空时保留来源声库已有版本；输入后会覆盖来源版本，可使用 101、106 或 106b1 等标识。","When blank, keeps the source version. A supplied value overrides it; identifiers such as 101, 106, or 106b1 are accepted.","空欄なら元のバージョンを保持します。入力すると上書きされ、101、106、106b1 などを指定できます。"],["选择 PNG 头像后，会作为 profile.png 放入 SFPK 根目录。不选择则不额外打包头像。","If selected, the PNG avatar is placed in the SFPK root as profile.png. Leave blank to omit it.","選択した PNG アバターは SFPK のルートに profile.png として格納されます。空欄なら含めません。"],["选择 PNG 背景后，会作为 background.png 放入 SFPK 根目录。不选择则不额外打包背景。","If selected, the PNG background is placed in the SFPK root as background.png. Leave blank to omit it.","選択した PNG 背景は SFPK のルートに background.png として格納されます。空欄なら含めません。"],["选择包含 acoustic、vocoder、duration、f0 等子目录的现有 models 文件夹。新模型会按内容 MD5 命名写入相应子目录，已有同名但内容不同的文件不会被覆盖。","Select the existing models folder with acoustic, vocoder, duration, f0, and other subfolders. New models are written to the matching subfolder using a content MD5 name; same-named files with different contents are not overwritten.","acoustic、vocoder、duration、f0 などのサブフォルダーを含む models フォルダーを選択します。新しいモデルは内容の MD5 に基づく名前で保存し、同名でも内容が異なる既存ファイルは上書きしません。"],["生成的 SFPK 包含 config.json 与 info.nofs，并可包含已选择的头像/背景 PNG；所引用的模型仍在上方选定的本机 models 目录。","The SFPK contains config.json and info.nofs, plus optional avatar/background PNGs. Referenced models remain in the selected local models directory.","SFPK には config.json と info.nofs、および任意のアバター／背景 PNG が含まれます。参照するモデルは選択したローカルの models フォルダーに残ります。"],["选择录音所使用的语言/音素集。它必须与数据集标签采用的音素体系一致，否则可能出现声库不认识某些音素的问题。","Select the language/phoneme set used by the recordings. It must match the dataset labels or the voice may not recognize some phonemes.","録音で使われている言語／音素セットを選択します。データセットのラベル体系と一致していないと、音声が一部の音素を認識できないことがあります。"],["选择素材组织方式：现有 WAV/LAB、分开放置的 WAV 与 LAB、从 WAV 文件名生成 LAB，或 TrainV6 manifest.json。","Choose how the recordings are organized: WAV/LAB together, separate WAV and LAB folders, LAB generation from WAV filenames, or a TrainV6 manifest.json.","録音の構成を選択します。WAV と LAB が同じ場所、別々のフォルダー、WAV ファイル名から LAB を生成、または TrainV6 manifest.json に対応します。"],["开启后，导入时会借助第 1 步选择的声库为词边界写入额外信息。若之后要训练 Timing 或 Pitch，这些信息可能是必要条件；只训练 Sound 时可以不勾选。","When enabled, import adds word-boundary information using the voice selected in step 1. This may be required for Timing or Pitch training; it can be disabled if training Sound only.","有効にすると、手順 1 で選択した音声を使ってインポート時に単語境界情報を追加します。Timing や Pitch の学習に必要な場合があります。Sound のみを学習する場合はオフでも構いません。"],["可选：给导入过程中的音素改名。每行写一组 old=new；适用于标注里的音素名与声库使用的名称不一致的情况。","Optional phoneme renaming during import. Enter one old=new pair per line when dataset phoneme names differ from the voice inventory.","インポート時に音素名を変更する場合に指定します（任意）。データセットの音素名が音声の一覧と異なる場合、各行に old=new を入力します。"],["Sound（声学模型）决定合成声音的音色与音频细节。它可使用只有音素与时长的数据集，也是建议优先尝试的模型。","Sound (acoustic) controls the synthesized voice's timbre and audio detail. It works with datasets containing only phonemes and durations, so it is the recommended model to try first.","Sound（音響モデル）は合成音声の声質や細部を担当します。音素と長さだけのデータセットでも使えるため、最初に試すモデルとして推奨します。"],["Timing（时长模型）学习每个音素在音符中的时长分配。数据集需要包含音符信息和词边界；缺少这些信息时不建议选择。","Timing (duration) learns how phoneme durations are distributed within notes. The dataset needs note information and word boundaries; do not select it without them.","Timing（長さモデル）は音符内での音素長の配分を学習します。音符情報と単語境界が必要なため、それらがない場合は選択しないでください。"],["Pitch（音高模型）学习音高曲线及相关表达。需要包含音符与词边界的标注；不带音符的数据集无法有效训练该模型。","Pitch learns the pitch contour and related expression. It needs notes and word-boundary labels; a dataset without notes cannot train it effectively.","Pitch は音高曲線と表現を学習します。音符と単語境界のラベルが必要で、音符がないデータセットでは効果的に学習できません。"],["Vocoder 是最终音频生成阶段的模型。它属于可选项目，训练通常更慢；初次使用建议先保持关闭。","Vocoder is the final audio-generation model. It is optional and usually slower to train, so leave it off for initial runs.","Vocoder は最終的な音声生成モデルです。任意項目で学習に時間がかかるため、初回はオフのままにしてください。"],["快速测试：运行少量步数，用来确认声库、数据集和运行环境可以正常工作。它不是完整训练。","Quick test: runs a small number of steps to confirm the voice, dataset and environment work. It is not full training.","クイックテスト：少ないステップ数で音声、データセット、環境が動作するか確認します。完全な学習ではありません。"],["标准训练：使用适中的初始步数，适合作为第一轮正式尝试。之后可根据验证结果调整。","Standard training uses a moderate initial step count and is a good first full run. Adjust later based on validation results.","標準学習は中程度のステップ数で、最初の本格的な学習に適しています。その後、検証結果に合わせて調整してください。"],["较长训练：步数更多、耗时更长，并会缓慢降低学习率。只有在短训练正常且验证结果仍有改善空间时再尝试。","Longer training uses more steps, takes longer, and gradually lowers the learning rate. Try it only if shorter training works and validation can still improve.","長時間学習はステップ数が多く、時間がかかり、学習率も徐々に下げます。短い学習が正常に動作し、検証結果に改善の余地がある場合に試してください。"],["每当验证指标改善时额外保存一次权重版本，方便保留训练中间表现较好的结果，即使之后提前停止也有机会使用。","Saves an additional checkpoint whenever validation improves, so a better intermediate result remains available if you stop early.","検証指標が改善するたびに追加チェックポイントを保存します。途中で停止しても、良い中間結果を使えます。"],["选择要更新的参数范围。“Everything” 会训练所选模型的权重，并通过 Anchor 约束接近原权重；“Only the voice's identity vectors” 只训练声库的说话人、风格和 Vocal Mode 向量，通常更轻量。","Choose which parameters to update. “Everything” trains selected model weights while Anchor keeps them near the originals. “Only the voice's identity vectors” trains only speaker, style and Vocal Mode vectors and is usually lighter.","更新するパラメータ範囲を選びます。「すべて」は Anchor で元の重みに近づけながらモデル重みを学習します。「話者埋め込みのみ」は話者、スタイル、Vocal Mode のベクトルのみを学習し、通常は軽量です。"],["选择训练设备。Automatic 会按环境选择可用设备；NVIDIA 使用 CUDA，Apple 芯片可使用 MPS，CPU 通常较慢但可作为兼容性选项。","Select the training device. Automatic chooses based on the environment; NVIDIA uses CUDA, Apple chips can use MPS, and CPU is slower but broadly compatible.","学習デバイスを選択します。自動では環境に応じて選び、NVIDIA は CUDA、Apple チップは MPS を使用できます。CPU は遅いものの互換性のある選択肢です。"],["勾选后保留录音的原始响度；不勾选时，程序通常会将录音缩放到起始声库的输出响度。","Keeps recordings at their original loudness when enabled. Otherwise, recordings are usually scaled to match the starting voice's loudness.","有効にすると録音の元の音量を維持します。無効の場合、通常は開始時の音声の音量に合わせて調整します。"],["尝试从运行目录中已有的 state.pt 继续训练。应保持此前使用的模型与参数范围一致，并将 Steps 设为高于已完成步数的值。","Attempts to resume from state.pt in the run folder. Keep the same models and parameter scope as before, and set Steps above the completed count.","実行フォルダーの state.pt から再開します。以前と同じモデルとパラメータ範囲を使い、Steps は完了済みの数より大きく設定してください。"],["选择此项表示你已经有 TrainSV 数据集：目录内应包含 wavs 子文件夹和 transcriptions.csv。若尚未生成数据集，请选择另一项并导入 .wav/.lab 文件。","Choose this if you already have a TrainSV dataset containing a wavs subfolder and transcriptions.csv. Otherwise, choose the other option and import .wav/.lab files.","wavs サブフォルダーと transcriptions.csv を含む TrainSV データセットをすでに持っている場合に選択します。まだない場合はもう一方を選んで .wav／.lab をインポートしてください。"],["选择此项表示你手头是录音和 .lab 音素标注，需要先将它们转换为 TrainSV 数据集，然后才能进行训练。","Choose this if you have recordings and .lab phoneme labels that must first be converted into a TrainSV dataset.","録音と .lab 音素ラベルがあり、学習前に TrainSV データセットへ変換する必要がある場合に選択します。"],["Choose the Synthesizer V voice file to use as the fine-tuning starting point. Supports voice.nofs or info.<version>.nofs. Training creates a new voice and does not overwrite this source file.","选择作为微调起点的 Synthesizer V 声库文件。支持 voice.nofs 或 info.<版本>.nofs。训练会生成新声库，不会直接覆盖这个源文件。","微調の開始点にする Synthesizer V 音声ファイルを選択します。voice.nofs または info.<version>.nofs に対応します。元ファイルを上書きせず、新しい音声を作成します。"],["Choose a dataset folder with a wavs subfolder and transcriptions.csv. The CSV needs phoneme sequences and durations; Timing or Pitch training also needs note and word-boundary information.","选择包含 wavs 子文件夹和 transcriptions.csv 的数据集目录。CSV 至少需要音素序列和时长；要训练 Timing 或 Pitch，还需要相应的音符与词边界信息。","wavs サブフォルダーと transcriptions.csv を含むデータセットを選択します。CSV には音素列と長さが必要で、Timing や Pitch の学習には音符と単語境界の情報も必要です。"],["Choose a folder containing both recordings and labels. Each song.wav needs a song.lab file with the same name.","选择同时存放录音与标注的文件夹。每个 song.wav 都需要同名的 song.lab 文件。","録音とラベルを含むフォルダーを選択します。各 song.wav には同名の song.lab が必要です。"],["Choose the folder containing .wav recordings. Each recording should have a matching .lab label file.","选择存放录音 .wav 文件的文件夹。每份录音都应有同名的 .lab 标注文件。","録音 .wav が入ったフォルダーを選択します。各録音には同名の .lab ラベルファイルが必要です。"],["Choose the WAV folder. Generation creates LAB labels from the space-separated phone order in each filename; a final _0/_1 is treated as a retake suffix.","选择 WAV 文件夹。生成模式会按文件名空格分隔的音素顺序生成 LAB；末尾 _0/_1 是重录编号。","WAV フォルダーを選択します。ファイル名のスペース区切りの音素順から LAB を作成し、末尾の _0／_1 はリテイク番号として扱います。"],["Choose phone_table.json to check custom phones and split compound phones. If omitted, TrainSV tries to use the selected voicebank's phone inventory.","选择 phone_table.json，用于判断自定义音素是否存在以及复合音素应如何拆分。若留空，TrainSV 会尝试使用所选声库的音素表。","カスタム音素の有無や複合音素の分割方法を確認する phone_table.json を選択します。省略すると、選択した音声の音素一覧を使用します。"],["Choose the folder containing .lab phoneme labels. Filenames must match the corresponding .wav recordings.","选择存放 .lab 音素标注文件的文件夹。文件名需要与对应的 .wav 录音一致。",".lab 音素ラベルのフォルダーを選択します。ファイル名は対応する .wav と一致する必要があります。"],["Choose a TrainV6 manifest.json file. It should list recording paths and time-aligned phoneme labels.","选择 TrainV6 的 manifest.json 文件。文件中应列出录音路径及其时间对齐的音素标签。","TrainV6 の manifest.json を選択します。録音パスと時間に対応した音素ラベルが記載されている必要があります。"],["Choose where to save the converted dataset. When conversion succeeds, TrainSV automatically selects that folder as the current dataset.","指定转换后的数据集保存目录。转换成功后，TrainSV 会自动将该目录设为当前数据集。","変換後のデータセットの保存先です。変換に成功すると、そのフォルダーが現在のデータセットとして自動選択されます。"],["Optional: import only named recordings. Enter names without extensions, separated by commas. Leave blank to import everything.","可选：只导入指定录音。填写不带扩展名的名称，多个名称用逗号分隔；留空表示导入全部录音。","指定した録音のみをインポートできます（任意）。拡張子なしの名前をカンマ区切りで入力し、空欄ならすべてを使用します。"],["Sound match","声音匹配度","音声の近さ"],["lower is closer","越低越接近","低いほど近い"],["Voiced/unvoiced match","清浊音匹配度","有声音／無声音の一致"],["Vibrato and motion","颤音与音高变化","ビブラートと音高変化"],["1.0 matches the singer","1.0 表示与歌手一致","1.0 は歌手に一致"],["Choose","选择","選択"],["Choose a folder","选择文件夹","フォルダーを選択"],["Choose a folder and file name","选择文件夹和文件名","フォルダーとファイル名を選択"],["Choose a file","选择文件","ファイルを選択"],["Up one folder","上一级文件夹","1 つ上のフォルダー"],["Home folder","主文件夹","ホームフォルダー"],["TrainSV folder","TrainSV 文件夹","TrainSV フォルダー"],["File name","文件名","ファイル名"],["New folder name","新文件夹名称","新しいフォルダー名"],["Save here","保存到此处","ここに保存"],["Use this folder","使用此文件夹","このフォルダーを使用"],["Create","创建","作成"],["Browse…","浏览…","参照…"],["Show the command this runs","显示此操作执行的命令","実行するコマンドを表示"],["Running","运行中","実行中"],["Done","完成","完了"],["Failed","失败","失敗"],["Stopped","已停止","停止しました"],["Next: train your voice","下一步：训练声库","次へ：音声を学習"],["Use it: go to Train","使用它：前往训练","使用する：学習へ進む"],["Next: check the result","下一步：检查结果","次へ：結果を確認"],["See the comparison","查看比较结果","比較を表示"],["A required Python package is missing. In the TrainSV folder run: pip install -e .","缺少必要的 Python 包。请在 TrainSV 文件夹中运行：pip install -e .","必要な Python パッケージがありません。TrainSV フォルダーで pip install -e . を実行してください。"],["The graphics card ran out of memory. Try “Phrases per step” = 1, or choose CPU in Advanced settings.","显卡显存不足。可尝试将“每步短语数”设为 1，或在高级设置中选择 CPU。","GPU メモリが不足しています。「1 ステップあたりのフレーズ数」を 1 にするか、詳細設定で CPU を選んでください。"],["Some sounds in your dataset do not exist in this voice. Check the language in step 1, or use a phoneme map (Advanced settings).","数据集中的部分音素不在此声库中。请检查第 1 步的语言设置，或使用音素映射（高级设置）。","データセットにこの音声で使えない音素があります。手順 1 の言語を確認するか、詳細設定で音素マップを使用してください。"],["Your dataset uses a style the voice does not have. Add it under “Vocal modes” in Advanced settings.","数据集使用了此声库没有的风格。请在高级设置的“Vocal Mode”中添加该风格。","データセットに音声が持たないスタイルがあります。詳細設定の「Vocal Mode」に追加してください。"],["Training became unstable. Try a lower learning rate in Advanced settings.","训练不稳定。请尝试在高级设置中降低学习率。","学習が不安定になりました。詳細設定で学習率を下げてみてください。"],["Untick “Continue the previous run”, or keep the same models and settings as the earlier run.","请取消勾选“继续上次的训练”，或确保使用与上次相同的模型和设置。","「前回の学習を再開」のチェックを外すか、前回と同じモデルと設定を使用してください。"],["Getting ready: loading the voice and your data. This can take a minute the first time.","正在准备：加载声库和数据。首次运行可能需要一点时间。","準備中：音声とデータを読み込んでいます。初回は少し時間がかかる場合があります。"],["This package is not installable.","此包不可安装。","このパッケージはインストールできません。"],["Its model dependencies remain in the local models directory and must be verified in Flat.","模型依赖仍保留在本机 models 目录中，需在 Flat 中验证。","モデル依存ファイルはローカルの models フォルダーに残ります。Flat で検証してください。"],["The command stopped unexpectedly.","命令意外终止。","コマンドが予期せず終了しました。"],["Something went wrong:","发生错误：","エラーが発生しました："],["No results saved there yet. Press “Compare voices”.","该目录中尚无保存的结果。请点击“比较声库”。","そのフォルダーにはまだ結果がありません。「音声を比較」を押してください。"],["Scores","评分","スコア"],["Listen","试听","試聴"],["Measure","指标","測定項目"],["original","原始","元"],["Test score","测试评分","テストスコア"],["At start","开始时","開始時"],["Latest","最新","最新"],["Real recording","真实录音","実際の録音"],["Choose a voice file.","请选择声库文件。","音声ファイルを選択してください。"],["Looking inside the voice","正在检查声库","音声を検査中"],["Choose a dataset first.","请先选择数据集。","先にデータセットを選択してください。"],["Scoring the original voice","正在评估原始声库","元の音声を評価中"],["Choose the recordings folder.","请选择录音文件夹。","録音フォルダーを選択してください。"],["Choose the manifest.json file.","请选择 manifest.json 文件。","manifest.json を選択してください。"],["Choose the folder of WAV files.","请选择 WAV 文件夹。","WAV フォルダーを選択してください。"],["Choose the folder of .wav files.","请选择 .wav 文件夹。",".wav フォルダーを選択してください。"],["Choose the folder of .lab files.","请选择 .lab 文件夹。",".lab フォルダーを選択してください。"],["Choose where to save the dataset.","请选择数据集保存位置。","データセットの保存先を選択してください。"],["Choose a voice (step 1), or untick “Mark word boundaries”.","请选择声库（第 1 步），或取消勾选“使用所选声库标记词边界”。","音声（手順 1）を選ぶか、「選択した音声で単語境界を付与」をオフにしてください。"],["Converting recordings to a dataset","正在将录音转换为数据集","録音をデータセットに変換中"],["Name your new voice.","请为新声库命名。","新しい音声に名前を付けてください。"],["Choose where to save the new voice.","请选择新声库的保存位置。","新しい音声の保存先を選択してください。"],["Select a .dnni file when using an external pretrained Pitch model.","使用外部预训练 Pitch 模型时，请选择 .dnni 文件。","外部の事前学習 Pitch モデルを使う場合は .dnni ファイルを選択してください。"],["Select Pitch in the model list above to import pretrained weights.","请在上方模型列表中选择 Pitch，才能导入预训练权重。","事前学習済み重みを読み込むには、上のモデル一覧で Pitch を選択してください。"],["Pitch model training","Pitch 模型训练","Pitch モデルの学習"],["Choose the original voice (step 1).","请选择原始声库（第 1 步）。","元の音声（手順 1）を選択してください。"],["Choose your new voice.","请选择新声库。","新しい音声を選択してください。"],["Choose where to save the comparison.","请选择比较结果的保存位置。","比較結果の保存先を選択してください。"],["Comparing voices","正在比较声库","音声を比較中"],["Select the trained voicebank file first.","请先选择训练后的声库文件。","先に学習済み音声ファイルを選択してください。"],["This is an info.*.nofs shared-model voicebank; the current SVPK diagnostic exporter supports only a regular single-file voice.nofs.","这是 info.*.nofs 共享模型声库；当前 SVPK 诊断导出器仅支持普通的单文件 voice.nofs。","これは info.*.nofs の共有モデル音声です。現在の SVPK 診断エクスポーターは単一ファイルの voice.nofs のみ対応します。"],["The selected path is not a .nofs file. Choose the trained voice.nofs from Results.","所选路径不是 .nofs 文件。请在“结果”中选择训练后的 voice.nofs。","選択したパスは .nofs ファイルではありません。「結果」から学習済み voice.nofs を選択してください。"],["Choose the trained .nofs file or an existing info.*.nofs file.","请选择训练后的 .nofs 文件或已有的 info.*.nofs 文件。","学習済み .nofs または既存の info.*.nofs ファイルを選択してください。"],["Enter a voicebank name.","请输入声库名称。","音声ライブラリ名を入力してください。"],["Version must be a non-negative safe integer, e.g. 101.","版本必须是非负安全整数，例如 101。","バージョンは 101 などの 0 以上の安全な整数である必要があります。"],["Choose a profile.png avatar image.","请选择 profile.png 头像图片。","profile.png のアバター画像を選択してください。"],["Choose a background PNG image.","请选择背景 PNG 图片。","背景 PNG 画像を選択してください。"],["Choose an SVPK output path.","请选择 SVPK 输出路径。","SVPK の出力先を選択してください。"],["Build SVPK diagnostic package","正在生成 SVPK 诊断包","SVPK 診断パッケージを作成中"],["Choose a .nofs file, not a directory or another file type.","请选择 .nofs 文件，而不是文件夹或其他文件类型。","フォルダーや別の形式ではなく .nofs ファイルを選択してください。"],["Choose the Synthesizer V models directory.","请选择 Synthesizer V 模型目录。","Synthesizer V の models フォルダーを選択してください。"],["Choose an SFPK output path.","请选择 SFPK 输出路径。","SFPK の出力先を選択してください。"],["Build SFPK test package","正在生成 SFPK 测试包","SFPK テストパッケージを作成中"],["Choose where to save the data.","请选择数据保存位置。","データの保存先を選択してください。"],["Making practice data","正在生成练习数据","練習データを作成中"],["Choose the source dataset.","请选择源数据集。","変換元データセットを選択してください。"],["Choose where to save the converted dataset.","请选择转换后数据集的保存位置。","変換後データセットの保存先を選択してください。"],["Choose the source dictionary.","请选择源字典。","変換元の辞書を選択してください。"],["Choose the Synthesizer V dictionary.","请选择 Synthesizer V 字典。","Synthesizer V の辞書を選択してください。"],["Converting the dictionary","正在转换字典","辞書を変換中"],["The dataset folder is not ready (see the message in step 2).","数据集文件夹尚未就绪（请查看第 2 步中的提示）。","データセットフォルダーの準備ができていません（手順 2 のメッセージを確認してください）。"],["Your dataset has no notes yet, so Timing and Pitch cannot learn anything. Add notes first (for example with GAME), or untick them to train Sound only.","数据集尚无音符，因此 Timing 和 Pitch 无法学习。请先添加音符（例如使用 GAME），或取消选择它们，仅训练 Sound。","データセットに音符がないため Timing と Pitch は学習できません。GAME などで音符を追加するか、両方をオフにして Sound のみを学習してください。"],["Timing and Pitch need a dataset with notes. Check your dataset in step 2.","Timing 和 Pitch 需要包含音符的数据集。请在第 2 步检查数据集。","Timing と Pitch には音符を含むデータセットが必要です。手順 2 のデータセットを確認してください。"],["✓ Found. This is a single-file voice.","✓ 已找到，这是单文件声库。","✓ 単一ファイルの音声が見つかりました。"],["That is a folder. Choose the .nofs file inside it.","这是一个文件夹。请选择其中的 .nofs 文件。","これはフォルダーです。中にある .nofs ファイルを選択してください。"],["File not found.","未找到文件。","ファイルが見つかりません。"],["Folder not found.","未找到文件夹。","フォルダーが見つかりません。"],["This folder needs a “wavs” folder and a “transcriptions.csv” file.","此文件夹需要包含 “wavs” 文件夹和 “transcriptions.csv” 文件。","このフォルダーには “wavs” フォルダーと “transcriptions.csv” ファイルが必要です。"],[" One of them is missing.","其中一个缺失。","いずれかが不足しています。"],[" If you have .lab files, choose “I have recordings and .lab files” above.","如果你有 .lab 文件，请选择上方的“我有录音和 .lab 文件”。",".lab ファイルがある場合は、上の「録音と .lab ファイルがある」を選択してください。"],["Notes included, so Sound, Timing and Pitch can all be trained.","已包含音符，因此可以训练 Sound、Timing 和 Pitch。","音符が含まれているため、Sound、Timing、Pitch をすべて学習できます。"],["No notes yet, so only Sound can be trained.","尚无音符，因此只能训练 Sound。","まだ音符がないため、Sound のみ学習できます。"],[" Each recording needs both.","每条录音都需要两者。","各録音に両方が必要です。"],["SVPK diagnostic package generation failed. See the job log for details.","SVPK 诊断包生成失败。详情请查看任务日志。","SVPK 診断パッケージの生成に失敗しました。詳細はジョブログを確認してください。"],["SFPK generation failed. See the job log for details.","SFPK 生成失败。详情请查看任务日志。","SFPK の生成に失敗しました。詳細はジョブログを確認してください。"],["SFPK export is experimental and depends on local model files. Verify the package in Flat after generation.","SFPK 导出仍属实验性功能，依赖本机模型文件。生成后请在 Flat 中验证包。","SFPK 出力は実験段階で、ローカルのモデルファイルに依存します。生成後、Flat で検証してください。"],["SVPK diagnostic package created: ","已生成 SVPK 诊断包：","SVPK 診断パッケージを作成しました："],["SFPK test package created: ","已生成 SFPK 测试包：","SFPK テストパッケージを作成しました："],["Please check the output path before continuing.","继续前请检查输出路径。","続行する前に出力先を確認してください。"],["The output file already exists:","输出文件已存在：","出力ファイルは既に存在します："],["Overwrite it?","是否覆盖？","上書きしますか？"],["A saved run was found here. Set “Steps” higher than what it already did to train further.","此处找到已保存的训练记录。将“步骤”设置为高于已完成数量的值以继续训练。","保存済みの実行が見つかりました。続けるには「ステップ数」を完了済みの値より大きく設定してください。"],["No saved run here yet, so this would start fresh.","此处尚无已保存的训练记录，将从头开始。","保存済みの実行はありません。新規に開始します。"],["Training steps. Higher values usually take longer and do not guarantee better results. Test the environment and data with a quick run first, then increase gradually.","训练总步数。数值越大，训练通常越久；更多步数不保证效果更好。可先用快速测试确认数据和环境正常，再逐步增加。","学習ステップ数です。多くすると通常は時間がかかりますが、品質が必ず向上するわけではありません。まずクイックテストでデータと環境を確認し、徐々に増やしてください。"],["Stop the TrainSV GUI? Anything running will be stopped.","要停止 TrainSV 图形界面吗？正在运行的任务也会停止。","TrainSV GUI を終了しますか？実行中のタスクも停止します。"],["Checking graphics card…","正在检测显卡…","GPU を確認中…"],["CPU only (training will be slow)","仅使用 CPU（训练会较慢）","CPU のみ（学習に時間がかかります）"],["PyTorch not working","PyTorch 无法正常运行","PyTorch が動作していません"],["Device unknown","未知设备","デバイス不明"],["✓ Found. This is an info voice (shared models); the result is saved into a folder.","✓ 已找到。这是 info 共享模型声库；结果将保存到文件夹。","✓ info 形式の共有モデル音声が見つかりました。結果はフォルダーに保存されます。"],["Warning: ","警告：","警告："],["Choose a voice (step 1).","请先选择声库（步骤 1）。","音声ライブラリを選択してください（手順 1）。"],["Choose a dataset (step 2).","请先选择数据集（步骤 2）。","データセットを選択してください（手順 2）。"],["Tick at least one thing to learn.","请至少选择一个要训练的模型。","学習するモデルを少なくとも 1 つ選択してください。"],["Choose the trained voice.nofs file first.","请先选择训练生成的 voice.nofs 文件。","学習済みの voice.nofs ファイルを先に選択してください。"],["This is an info.*.nofs shared-model voicebank; the SVPK diagnostic exporter supports only a regular single-file voice.nofs.","这是 info.*.nofs 共享模型声库；当前 SVPK 诊断导出器仅支持常规的单文件 voice.nofs。","これは info.*.nofs 共有モデル音声です。現在の SVPK 診断エクスポーターは通常の単一 voice.nofs ファイルのみ対応しています。"],["Choose the trained .nofs file, not a directory or another file type.","请选择训练生成的 .nofs 文件，而不是文件夹或其他类型的文件。","学習済みの .nofs ファイルを選択してください。フォルダーや別の種類のファイルは選択できません。"],["Enter a vendor / author.","请输入供应商／作者。","ベンダー／作者を入力してください。"],["Choose the trained voicebank file first.","请先选择已训练的声库文件。","学習済みの音声ファイルを先に選択してください。"],["Before you can continue:","继续前需要先完成以下事项：","続行する前に、次の項目を完了してください："],["Finish training and select the resulting voice.nofs file first.","请先完成训练，并选择生成的 voice.nofs 文件。","先に学習を完了し、生成された voice.nofs ファイルを選択してください。"],["This is an info.*.nofs shared-model voicebank. The current SVPK diagnostic exporter supports only a regular single-file voice.nofs.","这是 info.*.nofs 共享模型声库。当前 SVPK 诊断导出器仅支持常规的单文件 voice.nofs。","info.*.nofs の共有モデル音声です。現行の SVPK 診断エクスポーターで扱えるのは、通常形式の単一 voice.nofs ファイルのみです。"],["No voice file found there yet. Train first, or pick the file yourself.","尚未在此位置找到声库文件。请先训练，或手动选择文件。","音声ファイルがまだ見つかりません。先に学習するか、ファイルを手動で選択してください。"],["The TrainSV GUI has stopped. You can close this tab.","TrainSV GUI 已停止。你可以关闭此标签页。","TrainSV GUI は停止しました。このタブを閉じてもかまいません。"],["This file needs to sit inside the TrainSV folder (the one containing opensv_train). Move it there, or start it with: python trainsv_gui.py --project <the TrainSV folder>","此文件需要放在包含 opensv_train 的 TrainSV 文件夹中。请将其移到该文件夹，或使用以下命令启动：python trainsv_gui.py --project <the TrainSV folder>","このファイルは opensv_train を含む TrainSV フォルダー内に置く必要があります。移動するか、次のコマンドで起動してください：python trainsv_gui.py --project <the TrainSV folder>"],["⬑ Up one folder","⬑ 返回上级目录","⬑ 上のフォルダーへ"],["🏠 Home folder","🏠 用户主目录","🏠 ユーザーのホーム"],["📁 TrainSV folder","📁 TrainSV 项目文件夹","📁 TrainSV プロジェクトフォルダー"],["Timing (duration)","Timing（时长）","Timing（長さ）"],["Vocoder (final audio stage)","Vocoder（最终音频阶段）","Vocoder（最終音声段階）"],["Help","帮助","ヘルプ"],["Session expired. Reload the page.","会话已过期，请重新加载页面。","セッションの有効期限が切れました。ページを再読み込みしてください。"],["Something is already running. Stop it first.","已有任务正在运行，请先停止它。","既にタスクが実行中です。先に停止してください。"],["No permission to open this folder.","无权打开此文件夹。","このフォルダーを開く権限がありません。"],["Unknown request","未知请求","不明なリクエストです"],["Not found","未找到","見つかりません"],["PyTorch could not be loaded","无法加载 PyTorch","PyTorch を読み込めませんでした"],["This setting affects the current operation. Choose based on your training goal.","此设置会影响当前操作，请根据训练目标选择。","この設定は現在の操作に影響します。学習の目的に合わせて選択してください。"],["Sound (acoustic)","Sound（声学）","Sound（音響）"],["Pitch (melody and vibrato)","Pitch（旋律与颤音）","Pitch（メロディーとビブラート）"],["Preset default, e.g. 45","预设默认值，例如 45","プリセットの既定値（例：45）"],["Preset default, e.g. 1400","预设默认值，例如 1400","プリセットの既定値（例：1400）"],["Preset default, e.g. 0.45","预设默认值，例如 0.45","プリセットの既定値（例：0.45）"],["0 (no correction)","0（不校正）","0（補正なし）"],["2e-5 (3e-3 for identity only)","2e-5（仅训练身份向量时使用 3e-3）","2e-5（話者埋め込みのみの場合は 3e-3）"],["constant","恒定","一定"],["e.g. Luo Tianyi","例如：洛天依","例：洛天依"],["e.g. Artist or studio","例如：作者或工作室","例：アーティスト名またはスタジオ名"],["One step looks at one short phrase from your recordings. More steps take longer.","每个训练步会处理录音中的一个短语。步数越多，耗时越长。","1 ステップで録音の短いフレーズを 1 つ処理します。ステップ数が増えるほど時間がかかります。"],["How big each adjustment is.","控制每次参数调整的幅度。","パラメーターを調整する幅を指定します。"],["Start gently over this many steps.","在指定的步数内逐渐提高学习率。","指定したステップ数の間に学習率を徐々に上げます。"],["Slowly lower the learning rate to this.","逐渐降低学习率，直到此数值。","学習率をこの値まで徐々に下げます。"],["After import, review phone_mapping_report.csv in the dataset folder. It lists each source token, the target voice phone(s), and the mapping rule. A technically valid mapping can still be phonetically wrong for your source convention.","导入后请检查数据集目录中的 phone_mapping_report.csv。该报告列出源音素、映射后的目标音素和所用规则。映射结果即使存在于目标声库音素表中，也仍可能与源标记体系的实际发音含义不一致。","インポート後、データセット内の phone_mapping_report.csv を確認してください。各ソース音素、変換先音素、変換規則が記録されます。変換先の音素が存在していても、元の表記体系の発音と一致するとは限りません。"],["Transition sampling probability","过渡采样概率","遷移サンプリング確率"],["Transition-local spectral loss weight","过渡局部频谱损失权重","遷移局所スペクトル損失の重み"],["Acoustic DNNI output version","声学模型 DNNI 输出版本","音響モデルの DNNI 出力バージョン"],["DNNI v2 (default)","DNNI v2（默认）","DNNI v2（既定）"],["DNNI v1 (legacy compatibility)","DNNI v1（旧版兼容）","DNNI v1（旧形式互換）"],["Preserve original voicebank models (except acoustic)","保留原声库模型（声学模型除外）","元の音声ライブラリのモデルを保持（音響モデルを除く）"],["Available only for info.nofs. Train Sound only; keep the original Timing, Pitch/F0 and Vocoder model references and skip loading/checking those networks. A regular voice.nofs cannot use this option.","仅适用于 info.nofs。只训练声学模型，保留原始 Timing、Pitch/F0、Vocoder 模型引用，并跳过加载/检查这些网络。普通 voice.nofs 不可使用此选项。","info.nofs 専用です。音響モデルのみ学習し、元の Timing、Pitch/F0、Vocoder モデル参照を保持して、それらのネットワークの読み込み・検証を省略します。通常の voice.nofs では使用できません。"],["Replace a non-acoustic DNNI model","替换非声学 DNNI 模型","非音響 DNNI モデルを置換"],["Use a DNNI file from another voicebank to create a new info.nofs variant. Duration, Pitch/F0 and Vocoder can be replaced; Acoustic is intentionally excluded. The original info.nofs and model files are never overwritten. Save the new info.nofs at the same directory depth as an installed info voice so its models folder resolves correctly.","使用其他声库的 DNNI 文件生成新的 info.nofs 版本。可替换 Duration、Pitch/F0 和 Vocoder；不允许替换 Acoustic。不会覆盖原 info.nofs 和模型文件。请将新 info.nofs 保存到与已安装 info 声库相同的目录层级，以确保能正确解析 models 文件夹。","別の音声ライブラリの DNNI ファイルを使って、新しい info.nofs を作成します。Duration、Pitch/F0、Vocoder を置換できます。Acoustic は対象外です。元の info.nofs とモデルファイルは上書きしません。新しい info.nofs は、models フォルダーを正しく解決できるよう、インストール済み info 音声と同じ階層に保存してください。"],["Source info.nofs","源 info.nofs","元の info.nofs"],["Choose the existing JSON info.nofs file, not a regular binary voice.nofs.","选择现有的 JSON 格式 info.nofs，不要选择普通二进制 voice.nofs。","通常のバイナリ voice.nofs ではなく、JSON 形式の info.nofs を選択してください。"],["Model role to replace","要替换的模型类型","置換するモデルの役割"],["Acoustic is excluded. For DNNI v1/v2 TrainSV checks the selected role's supported structure. DNNI v3 can only be copied opaquely; runtime compatibility cannot be verified.","Acoustic 不可替换。对于 DNNI v1/v2，TrainSV 会检查所选模型类型的结构；DNNI v3 只能作为不透明文件原样复制，无法验证运行兼容性。","Acoustic は対象外です。DNNI v1/v2 は選択した役割の構造を TrainSV で検証します。DNNI v3 は不透明なファイルとしてコピーするだけで、実行互換性は検証できません。"],["Replacement .dnni file","替换用 .dnni 文件","置換する .dnni ファイル"],["Choose the role's model from another voicebank. Different voicebanks may have incompatible model structure, phoneme sets or embeddings.","选择其他声库中对应类型的模型。不同声库之间的模型结构、音素表或嵌入向量可能不兼容。","別の音声ライブラリから同じ役割のモデルを選択してください。音声ライブラリ間でモデル構造、音素セット、埋め込みが互換とは限りません。"],["Save updated info.nofs as","另存为新的 info.nofs","更新した info.nofs の保存先"],["Choose a new filename, for example info.101.nofs. Do not select the source file.","选择新的文件名，例如 info.101.nofs。不要选择源文件。","info.101.nofs など新しいファイル名を選択してください。元ファイルは選択しないでください。"],["Replace model","替换模型","モデルを置換"],["Choose source info.nofs.","请选择源 info.nofs。","元の info.nofs を選択してください。"],["Only info.nofs JSON voicebanks are supported.","仅支持 JSON 格式的 info.nofs 声库。","JSON 形式の info.nofs 音声のみ対応しています。"],["Select Duration, Pitch/F0, or Vocoder; Acoustic cannot be replaced.","请选择 Duration、Pitch/F0 或 Vocoder；不能替换 Acoustic。","Duration、Pitch/F0、Vocoder のいずれかを選択してください。Acoustic は置換できません。"],["Choose a replacement .dnni file.","请选择替换用的 .dnni 文件。","置換する .dnni ファイルを選択してください。"],["Replacement model must be a .dnni file.","替换模型必须是 .dnni 文件。","置換モデルは .dnni ファイルである必要があります。"],["Choose a new output info.nofs path.","请选择新的 info.nofs 输出路径。","新しい info.nofs の保存先を選択してください。"],["Output path must end with .nofs.","输出路径必须以 .nofs 结尾。","出力パスは .nofs で終わる必要があります。"],["Output must not overwrite the source info.nofs.","输出路径不能覆盖源 info.nofs。","出力先に元の info.nofs を指定することはできません。"],["Custom base DNNI models (info.nofs only)","自定义底层 DNNI 模型（仅 info.nofs）","カスタムベース DNNI モデル（info.nofs のみ）"],["Replace the complete non-acoustic model with a .dnni from another voicebank. DNNI v1/v2 must pass role-specific TrainSV structure checks. DNNI v3 can only be preserved opaquely and cannot be trained or runtime-validated by TrainSV. Do not combine these overrides with “Preserve original voicebank models”.","使用其他声库的 .dnni 替换完整的非声学模型。DNNI v1/v2 必须通过 TrainSV 针对模型类型的结构检查。DNNI v3 只能作为不透明文件保留，TrainSV 无法训练或验证其运行兼容性。不要与“保留原声库模型”同时使用。","別の音声ライブラリの .dnni で非音響モデル全体を置き換えます。DNNI v1/v2 は TrainSV の役割別構造検証を通過する必要があります。DNNI v3 は不透明なまま保持することしかできず、TrainSV で学習や実行互換性の検証はできません。「元のモデルを保持」と併用しないでください。"],["Custom Duration model (.dnni)","自定义 Duration 模型（.dnni）","カスタム Duration モデル（.dnni）"],["Custom Pitch/F0 model (.dnni)","自定义 Pitch/F0 模型（.dnni）","カスタム Pitch/F0 モデル（.dnni）"],["Replaces the base Pitch model. This differs from “External pretrained Pitch model”, which initializes weights in the current target model.","替换底模中的完整 Pitch 模型。它不同于“外部预训练 Pitch 模型”：后者只会把权重初始化到当前目标模型。","ベースの Pitch モデル全体を置き換えます。「外部の事前学習 Pitch モデル」は現在のモデルに重みを初期化する機能であり、こちらとは異なります。"],["Custom Vocoder model (.dnni)","自定义 Vocoder 模型（.dnni）","カスタム Vocoder モデル（.dnni）"],["Custom DNNI overrides cannot be combined with preservation mode.","自定义 DNNI 替换不能与保留原非声学模型模式同时使用。","カスタム DNNI の置換は、元の非音響モデル保持モードとは併用できません。"],["Choose either a custom Pitch/F0 DNNI or pretrained Pitch initialization, not both.","自定义 Pitch/F0 DNNI 与预训练 Pitch 权重初始化只能二选一。","カスタム Pitch/F0 DNNI と事前学習済み Pitch の重み初期化は、どちらか一方のみ選択してください。"],["Custom DNNI overrides cannot be combined with resume; use a separate run directory.","自定义 DNNI 替换不能与继续上次训练同时使用；请使用新的运行目录。","カスタム DNNI の置換は学習再開と併用できません。別の実行フォルダーを使用してください。"],["Batch replace non-acoustic DNNI models","批量替换非声学 DNNI 模型","非音響 DNNI モデルを一括置換"],["Replace any combination of Duration, Pitch/F0 and Vocoder in one operation. Acoustic is never changed. Choose the Synthesizer V models folder containing duration/, f0/, acoustic/ and vocoder/; this fixes missing-reference errors when the info.nofs does not sit next to its model library. Unselected models and their IDs stay unchanged. The original info.nofs and DNNI files are never overwritten.","一次操作可替换 Duration、Pitch/F0、Vocoder 中的任意组合。Acoustic 始终不变。请选择直接包含 duration、f0、acoustic、vocoder 子目录的 Synthesizer V models 文件夹；当 info.nofs 不在模型库旁边时，这可以解决模型引用路径错误。未选择的模型及其 ID 保持不变。原 info.nofs 和 DNNI 文件不会被覆盖。","Duration、Pitch/F0、Vocoder の任意の組み合わせを一度に置換できます。Acoustic は変更されません。duration、f0、acoustic、vocoder の各サブフォルダーを直接含む Synthesizer V の models フォルダーを選択してください。モデルライブラリと info.nofs が別の場所にある場合の参照パスエラーを解消できます。未選択のモデルと ID は変更されず、元のファイルも上書きされません。"],["Source models folder","源模型文件夹","元のモデルフォルダー"],["Choose the models directory whose direct subfolders are duration, f0, acoustic and vocoder. For example, the models folder inside your Synthesizer V Studio installation.","选择直接包含 duration、f0、acoustic、vocoder 子目录的 models 文件夹。例如 Synthesizer V Studio 安装目录中的 models 文件夹。","duration、f0、acoustic、vocoder の各サブフォルダーを直接含む models フォルダーを選択してください。Synthesizer V Studio のインストール先にある models フォルダーなどです。"],["Replacement Duration (.dnni, optional)","替换的 Duration（.dnni，可选）","置換する Duration（.dnni、任意）"],["Replacement Pitch/F0 (.dnni, optional)","替换的 Pitch/F0（.dnni，可选）","置換する Pitch/F0（.dnni、任意）"],["Replacement Vocoder (.dnni, optional)","替换的 Vocoder（.dnni，可选）","置換する Vocoder（.dnni、任意）"],["The output must use the same directory layout as an installed info voice.","输出路径的目录层级必须与已安装的 info 声库一致。","出力先はインストール済み info 音声と同じディレクトリ構成にしてください。"],["Batch replace models","批量替换模型","モデルを一括置換"],["Choose the Synthesizer V models folder.","请选择 Synthesizer V 的 models 文件夹。","Synthesizer V の models フォルダーを選択してください。"],["Choose at least one model to replace.","请至少选择一个需要替换的模型。","置換するモデルを少なくとも 1 つ選択してください。"],["Output package folder","输出声库包文件夹","出力パッケージフォルダー"],["Choose a separate writable output folder, for example Documents/TrainSV-output. TrainSV creates database/voices/<name>/info.<version>.nofs and database/models/ inside it. Do not select the installed Synthesizer V models folder.","请选择一个独立且可写的输出文件夹，例如 Documents/TrainSV-output。TrainSV 会在其中创建 database/voices/<name>/info.<version>.nofs 和 database/models/。不要选择 Synthesizer V 已安装的 models 文件夹作为输出。","Documents/TrainSV-output など、書き込み可能な別の出力フォルダーを選択してください。その中に database/voices/<name>/info.<version>.nofs と database/models/ を作成します。インストール済み Synthesizer V の models フォルダーを出力先に選択しないでください。"],["Choose an output package folder, not an info.nofs filename; TrainSV creates the directory layout for you.","info.nofs ファイル名ではなく、出力パッケージ用のフォルダーを選択してください。TrainSV が必要なディレクトリ構成を作成します。","info.nofs ファイル名ではなく、出力パッケージ用のフォルダーを選択してください。TrainSV が必要なディレクトリ構成を作成します。"],["Natural transitions (conservative)","自然衔接（保守）","自然な音素接続（保守）"],["Softer consonants (experimental)","柔化辅音（实验）","子音を柔らかく（実験）"],["Light source-faithful clarity (experimental)","轻度清晰度适配（实验）","軽い明瞭度適合（実験）"],["Manual mode leaves current parameters unchanged.","手动模式不会修改当前参数。","手動モードでは現在のパラメーターを変更しません。"],["Best when you want minimal change: trains only per-voice identity/style embeddings. It does not retrain the acoustic network.","适合尽量保留底模：只训练声库身份／风格向量，不重新训练声学网络。","変更を最小限に抑えたい場合に適しています。声庫の話者／スタイル埋め込みのみを学習し、音響ネットワークは再学習しません。"],["Use when your target data is limited or you want to stay close to the base. This conservative all-weight fit is not a universal fix for pronunciation errors.","适合目标数据较少或希望保留底模特征时使用。此档以保守方式微调整体权重，但不能保证修复发音错误。","対象データが少ない場合や元の音色を保ちたい場合に適しています。全重みを保守的に調整しますが、発音ミスを必ず直すものではありません。"],["Use as a baseline on a clean, well-covered dataset. It adds no special transition emphasis or brightness loss.","适合作为标注干净、音素覆盖充分的数据集基线。此档不会额外强调过渡，也不会启用明亮度损失。","ラベルが正確で音素が十分に網羅されたデータセットの基準設定です。遷移の追加強調や明るさ損失は使用しません。"],["Use when consonant attacks are too sharp: slightly lower consonant-frame emphasis while keeping transition and brightness extras off. It may soften articulation or reduce intelligibility, so compare plosives and fricatives on held-out phrases.","当辅音起音过尖时可尝试：轻微降低辅音帧权重，同时关闭额外过渡强调与明亮度损失。它可能让辅音变软或降低可懂度，请在留出短语中重点试听塞音和擦音。","子音の立ち上がりが鋭すぎる場合の実験です。子音フレームの重みを少し下げ、遷移強調と明るさ損失は無効にします。調音がぼやけたり明瞭度が下がったりする可能性があるため、検証フレーズの破裂音・摩擦音を比較してください。"],["A light experiment for renders that are darker than the source WAV: brightness-balance weight 0.025. It matches the source spectrum, not a generic treble boost, and cannot guarantee to remove nasality.","当渲染结果比源 WAV 更闷时可尝试此轻度实验：明亮度平衡权重为 0.025。它拟合源音频频谱，而非通用高频增强，也不能保证消除鼻腔感。","レンダーが元の WAV より暗い場合の軽い実験です。明るさバランスの重みは 0.025 です。一般的な高域ブーストではなく元音源のスペクトルに合わせるもので、鼻声の解消を保証しません。"],["This stronger experiment uses brightness-balance weight 0.05. Try only when held-out comparisons show the render lacks upper-band energy relative to the source; it may also change timbre.","此档使用较强的 0.05 明亮度平衡权重。仅当留出集对比显示渲染结果相对源音频缺少高频能量时再尝试；它也可能改变音色。","この強めの実験では明るさバランスの重みを 0.05 にします。検証用データで元音源より高域エネルギーが不足している場合にのみ試してください。音色自体も変わる可能性があります。"],["Most aggressive brightness experiment (0.10) with a weaker anchor. Use last, compare held-out phrases, and stop if consonants become sharper or the tone starts to drift.","最激进的明亮度实验（0.10），且 Anchor 更弱。应最后尝试并比较留出短语；若辅音变尖锐或音色开始漂移，应停止使用。","最も強い明るさ実験（0.10）で、Anchor も弱めです。最後に試し、検証フレーズを比較してください。子音が鋭くなる、または音色が変化しすぎる場合は使用を中止してください。"],["For VOCALOID/UTAU-style sources with abrupt phone handoffs: lower learning rate, stronger anchor, boundary-window sampling at 0.5, mild non-silence boundary emphasis (weight 1.15, radius 2), and local multi-resolution spectral loss at 0.25; brightness loss stays off. This is not waveform smoothing. Audit labels first and A/B test against the stock base.","适合 VOCALOID／UTAU 声源衔接突兀时先尝试：降低学习率、加强 Anchor，并以 0.5 概率围绕有效的非静音音素边界采样，采用 1.15 的边界权重、2 帧范围和 0.25 的局部多分辨率频谱损失；明亮度损失保持关闭。它不是波形平滑。请先检查标注与过渡覆盖率，并与原底模 A/B 对比。","VOCALOID／UTAU 系ソースの音素接続が不自然な場合の開始設定です。学習率を下げ、Anchor を強め、境界窓を確率 0.5 でサンプリングし、非無音境界の重み 1.15、範囲 2 フレーム、局所マルチ解像度スペクトル損失 0.25 を使用します。明るさ損失は無効です。波形平滑化ではありません。先にラベルと遷移の網羅性を確認し、元のモデルと A/B 比較してください。"]  ,["Convert audio to MIDI for pitch calibration","将音频转换为 MIDI 以校准音准","音程校正用に音声を MIDI に変換"]
  ,["Extracts a monophonic melody from audio using continuous F0 smoothing, conservative repair of short isolated octave errors, and more stable note boundaries. Import the MIDI into Synthesizer V to compare pitch. This is not polyphonic transcription, and semitone MIDI notes cannot preserve the source timbre or its full continuous pitch/vibrato curve.","从音频中提取单声部旋律，并通过连续 F0 平滑、谨慎修复短暂孤立的八度错误和更稳定的音符边界来改善 MIDI。可将 MIDI 导入 Synthesizer V 比较音准。这不是复调转录；半音 MIDI 音符无法保留原音色或完整连续音高/颤音曲线。","連続 F0 の平滑化、短く孤立したオクターブ誤検出の保守的な修正、より安定したノート境界によって単旋律を抽出します。MIDI を Synthesizer V に読み込んで音程を比較できます。ポリフォニックな採譜ではなく、半音単位の MIDI では元の音色や連続的なピッチ／ビブラート全体は保持できません。"]
  ,["Source audio (WAV / FLAC / AIFF)","源音频（WAV / FLAC / AIFF）","元音声（WAV / FLAC / AIFF）"]
  ,["Save MIDI file to","MIDI 文件保存位置","MIDI ファイルの保存先"]
  ,["Pitch floor (Hz)","音高下限（Hz）","音高の下限（Hz）"]
  ,["Pitch ceiling (Hz)","音高上限（Hz）","音高の上限（Hz）"]
  ,["Voicing threshold (0.05–1.0)","有声音阈值（0.05–1.0）","有声音のしきい値（0.05–1.0）"]
  ,["Minimum note duration (ms)","最短音符时长（毫秒）","最小音符長（ms）"]
  ,["Pitch smoothing window (1–21, odd)","音高平滑窗口（1–21，奇数）","音高平滑ウィンドウ（1–21、奇数）"]
  ,["A4 reference (Hz)","A4 参考频率（Hz）","A4 基準周波数（Hz）"]
  ,["Transpose (semitones, −24 to +24)","移调（半音，−24 至 +24）","移調（半音、−24～+24）"]
  ,["MIDI tempo metadata (BPM)","MIDI 速度元数据（BPM）","MIDI テンポ情報（BPM）"]
  ,["Convert audio to MIDI","将音频转换为 MIDI","音声を MIDI に変換"]
  ,["Converting audio to MIDI","正在将音频转换为 MIDI","音声を MIDI に変換中"]
  ,["Use a clear monophonic vocal recording. Polyphonic music can confuse the pitch tracker.","请使用清晰的单声部人声录音。复调音乐可能会干扰音高跟踪。","明瞭な単旋律の歌声を使用してください。ポリフォニックな音楽では音高追跡が乱れる場合があります。"]
  ,["Choose a .mid output path. Import this MIDI into Synthesizer V Studio to compare the voicebank's rendered notes with the original recording.","选择 .mid 输出路径。将 MIDI 导入 Synthesizer V Studio，即可把声库渲染的音符与原始录音进行比较。","出力先の .mid ファイルを選択します。MIDI を Synthesizer V Studio に読み込み、音声ライブラリのレンダリングと元の録音を比較します。"]
  ,["Lowest fundamental frequency to track. Raise this to avoid low-frequency rumble being detected as notes.","要跟踪的最低基频。提高此值可以避免将低频轰鸣误识别为音符。","追跡する基本周波数の下限です。値を上げると低周波ノイズを音符と誤認しにくくなります。"]
  ,["Highest fundamental frequency to track. Lower this if octave jumps appear in the extracted melody.","要跟踪的最高基频。如果提取旋律出现八度跳变，可以降低此值。","追跡する基本周波数の上限です。抽出した旋律にオクターブ誤りがある場合は下げてください。"]
  ,["Higher values reject uncertain/unvoiced frames more aggressively; lower values may recover weak notes but can add false notes.","较高的值会更积极地排除不确定或无声帧；较低的值可能恢复较弱的音符，但也可能增加误检。","値を上げると不確かなフレームや無声音をより強く除外します。下げると弱い音を拾いやすくなりますが、誤検出も増える場合があります。"]
  ,["Short pitch fragments are merged into a neighbouring note when possible. Reduce this for fast passages; increase it to suppress brief tracking glitches.","较短的音高片段会尽可能并入相邻音符。快速乐句可降低此值；若想抑制短暂的跟踪错误，可提高此值。","短い音高区間は可能であれば隣接ノートに統合されます。速いフレーズでは下げ、短い追跡ノイズを抑えるには上げてください。"]
  ,["Median smoothing is applied within voiced spans before semitone quantization. Use 1 to disable smoothing; larger windows suppress pitch flicker but can blur fast runs.","在半音量化之前，仅对有声音段进行中值平滑。设为 1 可关闭；较大的窗口可抑制音高抖动，但可能模糊快速音阶。","半音量子化の前に、有声音区間内だけ中央値平滑化を行います。1 で無効になり、大きな値はピッチの揺れを抑えますが、速い音符の動きがぼやける場合があります。"]
  ,["Reference pitch for MIDI note 69. Keep 440 Hz for standard concert tuning, or adjust for a non-standard reference.","MIDI 音符 69 的参考音高。标准音高使用 440 Hz；非标准调音时可调整。","MIDI ノート 69 の基準音高です。標準チューニングでは 440 Hz を使い、異なる基準では調整してください。"]
  ,["Shift all extracted notes by a fixed number of semitones for testing a different vocal range.","将所有提取音符统一移调指定的半音数，用于测试不同音域。","抽出した全音符を指定した半音数だけ移調し、別の声域をテストします。"]
  ,["Sets MIDI tempo metadata. Extracted note timing remains aligned to the source audio duration.","设置 MIDI 速度元数据。提取出的音符时间仍与源音频时长对齐。","MIDI テンポ情報を設定します。抽出された音符のタイミングは元音声の長さに合わせて維持されます。"]
  ,["Choose a source audio file.","请选择源音频文件。","元音声ファイルを選択してください。"]
  ,["Choose a WAV, FLAC or AIFF audio file.","请选择 WAV、FLAC 或 AIFF 音频文件。","WAV、FLAC、AIFF の音声ファイルを選択してください。"]
  ,["Choose where to save the MIDI file.","请选择 MIDI 文件的保存位置。","MIDI ファイルの保存先を選択してください。"]
  ,["The output filename must end in .mid or .midi.","输出文件名必须以 .mid 或 .midi 结尾。","出力ファイル名は .mid または .midi で終わる必要があります。"]
  ,["MIDI output must not overwrite the source audio.","MIDI 输出不能覆盖源音频。","MIDI 出力で元音声を上書きすることはできません。"]
  ,["Pitch range must satisfy 20 ≤ floor < ceiling ≤ 5000 Hz.","音高范围必须满足 20 ≤ 下限 < 上限 ≤ 5000 Hz。","音高範囲は 20 ≤ 下限 < 上限 ≤ 5000 Hz を満たす必要があります。"]
  ,["Voicing threshold must be between 0.05 and 1.0.","有声音阈值必须介于 0.05 和 1.0 之间。","有声音のしきい値は 0.05～1.0 の範囲にしてください。"]
  ,["Minimum note duration must be between 20 and 1000 ms.","最短音符时长必须介于 20 到 1000 毫秒之间。","最小音符長は 20～1000 ms の範囲にしてください。"]
  ,["Smoothing window must be an odd integer from 1 to 21.","平滑窗口必须是 1 到 21 之间的奇数整数。","平滑ウィンドウは 1～21 の奇数にしてください。"]
  ,["A4 reference must be between 400 and 480 Hz.","A4 参考频率必须介于 400 和 480 Hz 之间。","A4 基準周波数は 400～480 Hz の範囲にしてください。"]
  ,["Transpose must be an integer from −24 to +24 semitones.","移调值必须是 −24 到 +24 之间的整数半音。","移調は −24～+24 の整数半音にしてください。"]
  ,["MIDI tempo must be between 20 and 300 BPM.","MIDI 速度必须介于 20 到 300 BPM 之间。","MIDI テンポは 20～300 BPM の範囲にしてください。"]
];
const I18N_MAP = new Map();
for (const row of I18N_ROWS) for (const source of row) if (source && !I18N_MAP.has(source)) I18N_MAP.set(source, row);
const I18N_INDEX = {en:0,"zh-CN":1,ja:2};
let UI_LANG = "en";
let I18N_OBSERVER = null;
function translateString(source, language=UI_LANG) {
  const row=I18N_MAP.get(source);
  if(row) return row[I18N_INDEX[language]??0] || source;
  let m;
  if((m=source.match(/^Step (\d+) of (\d+) \((\d+)%\)\. About (.+) left\. Elapsed (.+)\.$/))){
    if(language==="zh-CN")return `第 ${m[1]}/${m[2]} 步（${m[3]}%）。预计还需 ${m[4]}。已用时 ${m[5]}。`;
    if(language==="ja")return `ステップ ${m[1]}/${m[2]}（${m[3]}%）。残り約 ${m[4]}。経過時間 ${m[5]}。`;
  }
  if((m=source.match(/^(\d+) of (\d+) recordings have notes\. The others will train Sound only\. Pitch correction needs complete note and word-boundary labels; the F0 preset alone cannot repair the base Pitch network\.$/))){
    if(language==="zh-CN")return `${m[1]}/${m[2]} 条录音有音符标注。其余录音只能训练 Sound。Pitch 校正需要完整音符和词边界标签；F0 预设本身无法修复底模的 Pitch 网络。`;
    if(language==="ja")return `${m[1]}/${m[2]} 件の録音に音符があります。残りは Sound のみ学習されます。Pitch の補正には完全な音符と単語境界ラベルが必要で、F0 プリセットだけでは元の Pitch ネットワークを修正できません。`;
  }
  if((m=source.match(/^Working… (.+) so far\.$/))){
    if(language==="zh-CN")return `正在处理……已用时 ${m[1]}。`;
    if(language==="ja")return `処理中…経過時間 ${m[1]}。`;
  }
  if((m=source.match(/^Finished\. Your new voice is saved at (.+?)(?: \((.+)\))?$/))){
    if(language==="zh-CN")return "训练完成。新声库已保存到："+m[1]+(m[2]?"（用时 "+m[2]+"）":"");
    if(language==="ja")return "学習が完了しました。新しい音声の保存先："+m[1]+(m[2]?"（所要時間："+m[2]+"）":"");
  }
  if((m=source.match(/^Finished\. Your new voice is saved at (.+)$/))){
    if(language==="zh-CN")return `训练完成。新声库已保存到：${m[1]}`;
    if(language==="ja")return `学習が完了しました。新しい音声の保存先：${m[1]}`;
  }
  if((m=source.match(/^Could not check the output path: (.+)$/))){
    if(language==="zh-CN")return `无法检查输出路径：${m[1]}`;
    if(language==="ja")return `出力先を確認できません：${m[1]}`;
  }
  if((m=source.match(/^SVPK diagnostic package created: (.+)\. This package is not installable or suitable for release\.$/))){
    if(language==="zh-CN")return `已生成 SVPK 诊断包：${m[1]}。此包不可安装，也不适合发布。`;
    if(language==="ja")return `SVPK 診断パッケージを作成しました：${m[1]}。インストールや公開には使用できません。`;
  }
  if((m=source.match(/^Found (\d+) \.wav and (\d+) \.lab files\.$/))){
    if(language==="zh-CN")return `找到 ${m[1]} 个 .wav 和 ${m[2]} 个 .lab 文件。`;
    if(language==="ja")return `.wav ${m[1]} 個、.lab ${m[2]} 個が見つかりました。`;
  }
  if((m=source.match(/^Found (\d+) WAV files\. Labels will be estimated from filenames and equal-duration timing\.$/))){
    if(language==="zh-CN")return `找到 ${m[1]} 个 WAV 文件。将根据文件名生成标签并均分时长。`;
    if(language==="ja")return `WAV ファイルが ${m[1]} 個見つかりました。ファイル名と均等な時間配分からラベルを推定します。`;
  }
  if((m=source.match(/^SFPK test package created: (.+?)\. Model dependencies were written to the local directory: (.+?)\. Import and test rendering in Synthesizer V Studio Flat; runtime compatibility has not yet been verified\.$/))){
    if(language==="zh-CN")return `已生成 SFPK 测试包：${m[1]}。模型依赖已写入本机目录：${m[2]}。请在 Synthesizer V Studio Flat 中导入并测试渲染；运行时兼容性尚未验证。`;
    if(language==="ja")return `SFPK テストパッケージを作成しました：${m[1]}。モデル依存ファイルはローカルに保存されました：${m[2]}。Synthesizer V Studio Flat で読み込んでレンダリングを確認してください。実行時の互換性は未検証です。`;
  }
  if((m=source.match(/^SFPK test package created: (.+?)\. Its model dependencies remain in the local models directory and must be verified in Flat\.$/))){
    if(language==="zh-CN")return `已生成 SFPK 测试包：${m[1]}。模型依赖仍在本机 models 目录中，需在 Flat 中验证。`;
    if(language==="ja")return `SFPK テストパッケージを作成しました：${m[1]}。モデル依存ファイルはローカルの models フォルダーに残るため、Flat で検証してください。`;
  }
  if((m=source.match(/^✓ (\d+) recordings, about ([\d.]+) min of singing\. (.*?)(?: Warning: (\d+) listed recordings have no \.wav file\.)?$/))){
    const note=m[3].match(/^(\d+) have notes\.$/);
    const detail=note?(language==="zh-CN"?note[1]+" 条录音包含音符。":language==="ja"?note[1]+" 件の録音に音符があります。":m[3]):translateString(m[3],language);
    if(language==="zh-CN")return "✓ "+m[1]+" 条录音，歌唱时长约 "+m[2]+" 分钟。 "+detail+(m[4]?" 警告：有 "+m[4]+" 条已列出的录音缺少 .wav 文件。":"");
    if(language==="ja")return "✓ 録音 "+m[1]+" 件、歌唱時間約 "+m[2]+" 分。 "+detail+(m[4]?" 警告：一覧にある録音 "+m[4]+" 件に .wav ファイルがありません。":"");
  }
  if((m=source.match(/^(\d+) have notes\.$/))){
    if(language==="zh-CN")return `${m[1]} 条录音包含音符。`;
    if(language==="ja")return `${m[1]} 件の録音に音符があります。`;
  }
  if((m=source.match(/^The output file already exists:\n([\s\S]+)\n\nOverwrite it\?$/))){
    if(language==="zh-CN")return `输出文件已存在：\n${m[1]}\n\n是否覆盖？`;
    if(language==="ja")return `出力ファイルは既に存在します：\n${m[1]}\n\n上書きしますか？`;
  }
  if((m=source.match(/^Getting ready: loading the voice and your data\. This can take a minute the first time\.$/))){
    if(language==="zh-CN")return "正在准备：加载声库和数据。首次运行可能需要一点时间。";
    if(language==="ja")return "準備中：音声とデータを読み込んでいます。初回は少し時間がかかる場合があります。";
  }
  if((m=source.match(/^Found (\d+) \.wav and (\d+) \.lab files\.(?: Each recording needs both\.)?$/))){
    if(language==="zh-CN")return "找到 "+m[1]+" 个 .wav 和 "+m[2]+" 个 .lab 文件。"+(source.endsWith("Each recording needs both.")?" 每条录音都需要同时具备这两种文件。":"");
    if(language==="ja")return ".wav "+m[1]+" 個、.lab "+m[2]+" 個が見つかりました。"+(source.endsWith("Each recording needs both.")?" 各録音には両方のファイルが必要です。":"");
  }
  if((m=source.match(/^This folder needs a “wavs” folder and a “transcriptions\.csv” file\.(.*)$/))){
    let suffix="";
    if(m[1].includes("One of them is missing"))suffix=language==="zh-CN"?"其中一个缺失。":"そのうち 1 つがありません。";
    else if(m[1].includes("If you have .lab files"))suffix=language==="zh-CN"?"如果你有 .lab 文件，请选择上方的“我有录音和 .lab 文件”。":" .lab ファイルがある場合は、上の「録音と .lab ファイルがある」を選択してください。";
    if(language==="zh-CN")return "此文件夹需要包含 wavs 文件夹和 transcriptions.csv 文件。"+(suffix?" "+suffix:"");
    if(language==="ja")return "このフォルダーには wavs フォルダーと transcriptions.csv ファイルが必要です。"+(suffix?" "+suffix.replace(/^ /,""):"");
  }
  if((m=source.match(/^(\d+) of (\d+) recordings have notes\. The others will train Sound only\.$/))){
    if(language==="zh-CN")return m[1]+" / "+m[2]+" 条录音包含音符。其余录音将仅用于训练 Sound。";
    if(language==="ja")return m[1]+" / "+m[2]+" 件の録音に音符があります。残りは Sound の学習のみに使用されます。";
  }
  if((m=source.match(/^It will check: (.+) \(the same as in Train\), using the same held-out recordings as training\.$/))){
    if(language==="zh-CN")return "将检查："+m[1]+"（与“训练”步骤相同），并使用与训练相同的留出录音。";
    if(language==="ja")return "チェック対象："+m[1]+"（「学習」と同じ設定）。学習時と同じ検証用録音を使用します。";
  }
  if((m=source.match(/^You stopped this\.( Training progress is kept in the run folder\. Tick “Continue the previous run” in Advanced settings to carry on\.)?$/))){
    if(language==="zh-CN")return "你已停止任务。"+(m[1]?"训练进度保存在运行目录中。若要继续，请在“高级设置”中勾选“继续上次的训练”。":"");
    if(language==="ja")return "タスクを停止しました。"+(m[1]?"学習の進行状況は実行フォルダーに保存されています。続けるには詳細設定で「前回の学習を再開」をオンにしてください。":"");
  }
  if((m=source.match(/^SVPK diagnostic package created: (.+?)\. This package is not installable\.$/))){
    if(language==="zh-CN")return "已生成 SVPK 诊断包："+m[1]+"。此包不可安装。";
    if(language==="ja")return "SVPK 診断パッケージを作成しました："+m[1]+"。このパッケージはインストールできません。";
  }
  if((m=source.match(/^Missing Python packages: (.+?)\. In the TrainSV folder run: pip install -e \.\s+\(use the same Python that started this page: (.+)\)$/))){
    if(language==="zh-CN")return "缺少 Python 软件包："+m[1]+"。请在 TrainSV 文件夹中执行 pip install -e .（使用启动此页面的 Python："+m[2]+"）";
    if(language==="ja")return "不足している Python パッケージ："+m[1]+"。TrainSV フォルダーで pip install -e . を実行してください（このページの起動に使用した Python："+m[2]+"）。";
  }
  if((m=source.match(/^Could not read transcriptions\.csv: (.*)$/))){
    if(language==="zh-CN")return "无法读取 transcriptions.csv："+m[1];
    if(language==="ja")return "transcriptions.csv を読み込めません："+m[1];
  }
  if((m=source.match(/^(.+?) \(No permission to open this folder\.\)$/))){
    if(language==="zh-CN")return m[1]+"（无权限打开此文件夹）";
    if(language==="ja")return m[1]+"（このフォルダーを開く権限がありません）";
  }
  if((m=source.match(/^Request failed \((\d{3})\)$/))){
    if(language==="zh-CN")return "请求失败（HTTP "+m[1]+"）。";
    if(language==="ja")return "リクエストに失敗しました（HTTP "+m[1]+"）。";
  }
  if(source==="Failed to fetch"||source==="Load failed"||source==="NetworkError when attempting to fetch resource."){
    if(language==="zh-CN")return "无法连接到 TrainSV 本地服务。请确认 GUI 仍在运行，然后刷新页面。";
    if(language==="ja")return "TrainSV のローカルサービスに接続できません。GUI が起動していることを確認してページを再読み込みしてください。";
  }
  if((m=source.match(/^(?:\[Errno (\d+)\] )?(No such file or directory|Permission denied|Is a directory|Not a directory|File exists|Invalid argument)(?:: (.*))?$/i))){
    const messages={
      "no such file or directory":["找不到文件或目录","ファイルまたはディレクトリが見つかりません"],
      "permission denied":["权限被拒绝","アクセスが拒否されました"],
      "is a directory":["该路径是一个文件夹","指定されたパスはディレクトリです"],
      "not a directory":["该路径不是文件夹","指定されたパスはディレクトリではありません"],
      "file exists":["文件已存在","ファイルは既に存在します"],
      "invalid argument":["参数无效","引数が無効です"]
    };
    const pair=messages[m[2].toLowerCase()];
    if(pair&&language==="zh-CN")return pair[0]+(m[3]? "："+m[3]:"");
    if(pair&&language==="ja")return pair[1]+(m[3]? "："+m[3]:"");
  }
  if((m=source.match(/^Enter a number for “(.+?)”\.$/))){
    if(language==="zh-CN")return "“"+m[1]+"”需要填写数字。";
    if(language==="ja")return "「"+m[1]+"」には数値を入力してください。";
  }
  if((m=source.match(/^“(.+?)” must be a number \(you typed “([\s\S]*?)”\)\.$/))){
    if(language==="zh-CN")return "“"+m[1]+"”必须是数字（你输入的是“"+m[2]+"”）。";
    if(language==="ja")return "「"+m[1]+"」には数値を入力してください（入力値：「"+m[2]+"」）。";
  }
  if((m=source.match(/^Help: (.+)$/))){
    const label=translateString(m[1],language);
    if(language==="zh-CN")return "帮助："+label;
    if(language==="ja")return "ヘルプ："+label;
  }
  if((m=source.match(/^step (\d+)\/(\d+)$/i))){
    if(language==="zh-CN")return "第 "+m[1]+" / "+m[2]+" 步";
    if(language==="ja")return "ステップ "+m[1]+" / "+m[2];
  }
  if((m=source.match(/^Could not start Python: (.*)$/))){
    if(language==="zh-CN")return "无法启动 Python："+m[1];
    if(language==="ja")return "Python を起動できません："+m[1];
  }
  if((m=source.match(/^unknown command: (.*)$/i))){
    if(language==="zh-CN")return "未知命令："+m[1];
    if(language==="ja")return "不明なコマンド："+m[1];
  }
  if((m=source.match(/^(\d+) utterances, (\d+) phrases, ([\d.]+) min(?: of singing)?\. Getting ready: loading the voice and your data\. This can take a minute the first time\.$/))){
    if(language==="zh-CN")return m[1]+" 条发声片段，"+m[2]+" 个短语，约 "+m[3]+" 分钟。正在准备：加载声库和数据。首次运行可能需要一点时间。";
    if(language==="ja")return "発話 "+m[1]+" 件、フレーズ "+m[2]+" 件、約 "+m[3]+" 分。準備中：音声とデータを読み込んでいます。初回は少し時間がかかる場合があります。";
  }
  if((m=source.match(/^(.+?)\. Getting ready: loading the voice and your data\. This can take a minute the first time\.$/))){
    const intro=translateString(m[1],language);
    if(language==="zh-CN"&&intro!==m[1])return intro+" 正在准备：加载声库和数据。首次运行可能需要一点时间。";
    if(language==="ja"&&intro!==m[1])return intro+" 準備中：音声とデータを読み込んでいます。初回は少し時間がかかる場合があります。";
  }
  return source;
}
function localizeTree(root=document.body) {
  if(!root)return;
  const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
  let node;
  while((node=walker.nextNode())){
    if(node.parentElement?.closest("script,style,pre,#logPre"))continue;
    const current=node.nodeValue||"";
    if(!current.trim())continue;
    let meta=node.__i18nText;
    if(!meta || current!==meta.rendered){
      const match=current.match(/^(\s*)([\s\S]*?)(\s*)$/);
      meta={source:match[2],leading:match[1],trailing:match[3],rendered:current};
      node.__i18nText=meta;
    }
    const translated=meta.leading+translateString(meta.source)+meta.trailing;
    if(current!==translated)node.nodeValue=translated;
    meta.rendered=translated;
  }
  const elements=root.nodeType===Node.ELEMENT_NODE?[root,...root.querySelectorAll("*")]:[...root.querySelectorAll("*")];
  for(const el of elements){
    for(const name of ["title","placeholder","aria-label"]){
      if(!el.hasAttribute(name))continue;
      const current=el.getAttribute(name);
      el.__i18nAttrs=el.__i18nAttrs||{};
      let meta=el.__i18nAttrs[name];
      if(!meta||current!==meta.rendered)meta={source:current,rendered:current};
      const translated=translateString(meta.source);
      if(current!==translated)el.setAttribute(name,translated);
      meta.rendered=translated;
      el.__i18nAttrs[name]=meta;
    }
  }
}
function setUiLanguage(language){
  UI_LANG=Object.prototype.hasOwnProperty.call(I18N_INDEX,language)?language:"en";
  document.documentElement.lang=UI_LANG==="zh-CN"?"zh-CN":UI_LANG;
  localizeTree(document.body);
}
function watchTranslatedContent(){
  if(I18N_OBSERVER)return;
  I18N_OBSERVER=new MutationObserver(records=>{
    for(const record of records){
      if(record.target?.nodeType===Node.TEXT_NODE && record.target.parentElement?.closest("script,style,pre,#logPre"))continue;
    }
    localizeTree(document.body);
  });
  I18N_OBSERVER.observe(document.body,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:["title","placeholder","aria-label"]});
}
const nativeAlert=window.alert.bind(window),nativeConfirm=window.confirm.bind(window);
window.alert=message=>nativeAlert(translateString(String(message)));
window.confirm=message=>nativeConfirm(translateString(String(message)));

const TOKEN="__TOKEN__";
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const MODELS=["acoustic","duration","pitch","vocoder"];
const MODEL_NAME={acoustic:"Sound",duration:"Timing",pitch:"Pitch",vocoder:"Vocoder"};
const AUTO=["t_output","t_run","e_out","i_output","s_output","c_output"];
const METRIC={mel:["Sound match","lower is closer"],voicing:["Voiced/unvoiced match","lower is closer"],pitch_motion:["Vibrato and motion","1.0 matches the singer"],brightness:["Brightness-balance error","lower is closer"]};
const PRESETS={p_quick:{t_steps:"200",t_evalevery:"100",t_logevery:"10",t_warmup:"",t_lrend:""},
 p_std:{t_steps:"2000",t_evalevery:"",t_logevery:"",t_warmup:"",t_lrend:""},
 p_long:{t_steps:"6000",t_evalevery:"",t_logevery:"",t_warmup:"100",t_lrend:"2e-6"}};
let ENV={root:"",sep:"/"},EXTRA={manual:{},trained:false,evaluated:false},DS=null,VOICE_OK=false,JOB=null,polling=false,SVPK_AUTO_OUTPUT="",SFPK_AUTO_OUTPUT="";

const val=id=>{const e=document.getElementById(id);return e?String(e.value||"").trim():""};
const chk=id=>!!document.getElementById(id)?.checked;
const enc=encodeURIComponent;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const quote=s=>/^[\w@%+=:,.\/\\-]+$/.test(s)?s:'"'+s.replace(/"/g,'\\"')+'"';
function sepOf(p){return p.includes("\\")&&!p.includes("/")?"\\":"/"}
function trimEnd(p){return p.replace(/[\\\/]+$/,"")}
function join(a,...b){return [trimEnd(a),...b].join(sepOf(a))}
function dirname(p){p=trimEnd(p);const i=Math.max(p.lastIndexOf("/"),p.lastIndexOf("\\"));return i>0?p.slice(0,i):p}
function basename(p){return trimEnd(p).split(/[\\\/]/).pop()}
function slug(s){return (s||"").trim().toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/^-|-$/g,"")||"my-voice"}
function isInfo(p){return /^info(?:\..*)?\.nofs$/i.test(basename(p||""))}
function fmtTime(s){s=Math.max(0,Math.round(s));if(s<60)return s+" s";if(s<3600)return Math.round(s/60)+" min";return Math.floor(s/3600)+" h "+Math.round(s%3600/60)+" min"}
function fmtNum(x){return Number.isFinite(x)?x.toFixed(4):"n/a"}

async function api(path,body){
  const o={headers:{"X-Token":TOKEN}};
  if(body!==undefined){o.method="POST";o.headers["Content-Type"]="application/json";o.body=JSON.stringify(body)}
  const r=await fetch("/api/"+path,o);
  let j={};try{j=await r.json()}catch(e){}
  if(!r.ok)throw new Error(j.error||("Request failed ("+r.status+")"));
  return j;
}

const FIELD_HELP = {
  "voice": "选择作为微调起点的 Synthesizer V 声库文件。支持 voice.nofs 或 info.<版本>.nofs。训练会生成新声库，不会直接覆盖这个源文件。",
  "dataset": "选择包含 wavs 子文件夹和 transcriptions.csv 的数据集目录。CSV 至少需要音素序列和时长；要训练 Timing 或 Pitch，还需要相应的音符与词边界信息。",
  "i_corpus": "选择同时存放录音与标注的文件夹。每个 song.wav 都需要同名的 song.lab 文件。",
  "i_wavs": "选择存放录音 .wav 文件的文件夹。每份录音都应有同名的 .lab 标注文件。",
  "i_wavs_names": "选择 WAV 文件夹。生成模式会按文件名空格分隔的音素顺序生成 LAB；末尾 _0/_1 是重录编号。",
  "i_source_system": "按文件名实际使用的标记选择源体系。auto 保留 VOCALOID 中文默认转换；xsampa 使用标准 X-SAMPA；synthv-xsampa 使用 Synthesizer V 旧版中文标记；sv1112-chinese/cantonese 按 Dreamtonics 的 Old 标签表转换到 Synthesizer V Studio 1.11.2。显式模式遇到未知音素会报告错误，不会猜测拆分。",
  "i_phone_table": "选择 phone_table.json，用于判断自定义音素是否存在以及复合音素应如何拆分。若留空，TrainSV 会尝试使用所选声库的音素表。",
  "i_labs": "选择存放 .lab 音素标注文件的文件夹。文件名需要与对应的 .wav 录音一致。",
  "i_manifest": "选择 TrainV6 的 manifest.json 文件。文件中应列出录音路径及其时间对齐的音素标签。",
  "i_output": "指定转换后的数据集保存目录。转换成功后，TrainSV 会自动将该目录设为当前数据集。",
  "i_songs": "可选：只导入指定录音。填写不带扩展名的名称，多个名称用逗号分隔；留空表示导入全部录音。",
  "t_steps": "训练总步数。数值越大，训练通常越久；更多步数不保证效果更好。可先用快速测试确认数据和环境正常，再逐步增加。",
  "v_name": "为输出声库设置显示名称。建议使用容易识别的名称；该名称也会用于自动生成部分输出路径。",
  "t_output": "指定训练后声库的保存位置。普通 voice.nofs 声库保存为文件；info.*.nofs 类型的共享模型声库会按工具规则保存到目录中。",
  "t_lr": "学习率，控制每次更新模型参数时的调整幅度。默认值取决于“调整内容”：全部权重通常为 2e-5，仅训练身份向量时通常为 3e-3。过高可能不稳定，过低可能进展缓慢。",
  "t_f0_profile": "F0 预设只控制输入 WAV 的基频检测范围与目标曲线清理，不会更改 Pitch 模型的损失函数参数。Sensitive 使用更低的浊音阈值和 15 ms 轻度平滑，可尝试找回较弱的有声帧；静音或清辅音本身没有可测 F0，不能靠降低阈值创造音高。建议检查训练日志中的有声帧比例和中位 F0；范围放宽也可能增加错误音高候选。",
  "t_f0_correction": "对从 WAV 检测并经过基本清理的 F0 参考曲线整体做音分校正。负值降低参考音高，正值提高。建议先测量实际偏差并小幅调整；不应用于修正八度跳变、错误音符或时间对齐。",
  "t_pitch_smoothing": "低通截止频率，用于 Pitch 训练损失里分离平滑旋律轮廓和快速变化。较低会更强调旋律大轮廓，较高会把更多快速变化纳入旋律项。这不是直接平滑最终音频；它独立于 F0 检测预设。",
  "t_pitch_melody_weight": "平滑旋律轮廓损失权重，表示 Pitch 训练更重视整体旋律曲线还是其他音高细节。请结合验证集的 pitch_semitones/pitch_melody 和试听设置。",
  "t_pitch_motion_weight": "快速变化/颤音损失权重。提高它可能保留更多细节，也可能学到不需要的抖动；初始建议保留 1.0。",
  "t_f0_smoothing": "F0 中值平滑只在连续有声片段内部工作，不会把平滑结果跨过静音或辅音。强度越大越可能压低原有颤音细节。",
  "t_pitch_source": "默认使用当前底模的 Pitch 网络。选择外部预训练时，必须提供兼容的 .dnni 文件；TrainSV 会严格检查网络树、音素表、归一化和稀疏掩码，不兼容时会拒绝。当前仓库没有随包附带经过验证的通用神经 Pitch 权重。",
  "t_warmup": "预热步数。大于 0 时，学习率会在指定步数内逐渐升高；留空或为 0 时不启用预热。",
  "t_lrend": "最终学习率。填写后，训练会按余弦衰减逐步降至该值；留空时学习率保持不变（除非设置了预热）。",
  "t_anchor": "控制微调权重向原始声库权重靠拢的强度。值越大，约束通常越强、越能保留原声库特征；具体效果仍取决于数据和训练设置。",
  "t_timbre_preset": "音色微调预设只填入训练参数，适用于保留底模、温和适配、平衡拟合或实验性明亮度目标。各预设都只是起点，不保证适用于每个底模或数据集。",
  "t_brightness_weight": "明亮度平衡损失比较生成音频与目标 WAV 的 3–10 kHz / 1–3 kHz 能量比。0 关闭；0.05 轻度实验；0.10 更强。它会学习目标自身的频谱平衡，而不是无条件提升高频，鼻腔感仍需听音判断。",
  "t_batch": "每一步累计计算的短语数量；音频训练使用对应音频窗口。增加批次通常会占用更多显存，显存不足时可尝试减小，常从 1 开始。",
  "t_cons": "辅音帧在训练音频损失中的权重。大于 1 会更强调辅音；留空使用程序默认值。验证集评分不应用此额外权重。",
  "t_val": "留作验证、不会用于训练的短语比例或数量。小于 1 的数值表示比例（例如 0.1 约为 10%）；大于或等于 1 时按数量解释。设为 0 可不留验证数据。",
  "t_seed": "随机种子，用于控制随机数据划分等步骤。比较不同设置时尽量使用相同的种子；更改种子可能会改变验证集。",
  "t_evalevery": "每隔多少训练步数评估一次验证集。间隔较小能更频繁观察效果，但会增加评估开销。",
  "t_logevery": "每隔多少训练步数输出一次训练日志。它主要影响日志的详细程度，不直接改变训练目标。",
  "t_pmap": "可选的 JSON 音素映射文件，用于把数据集中的音素名称转换成当前声库使用的名称。映射不匹配时可能导致训练数据无法正确识别。",
  "t_modes": "可选：设置要训练的 Vocal Mode 名称，多个名称用逗号分隔。此功能主要适用于 info 声库；新增模式时通常先用“仅身份向量”训练，再进行完整微调。",
  "t_version": "可选：指定 info 声库输出使用的版本名称。仅适用于 info.*.nofs 类型声库。",
  "t_final": "可选：为最后一步的权重另存一个版本。填写要使用的版本标识；它与验证过程中保存的最佳权重版本不同。仅适用于 info 声库。",
  "t_levelref": "可选的响度参考声库。连续微调时可指定最初的基础声库，避免每次训练都以微调后的声音重新定标。",
  "t_gain": "可选：手动固定录音响度缩放系数。留空时由程序测量；只有在明确知道需要的缩放幅度时再填写。",
  "t_run": "训练运行目录，用来保存日志和可恢复的训练状态。启用后，之后可以尝试从该目录继续训练。",
  "t_history": "可选：将验证评分历史写入 JSON 文件，便于比较多次训练结果或进一步分析。",
  "e_new": "选择训练后生成的新声库文件。对比时会将它与第 1 步选择的原始声库进行评估。",
  "e_out": "指定评估结果的保存目录。结果包含 metrics.json，以及用于试听对比的渲染音频（如果成功生成）。",
  "e_max": "最多渲染多少条留出的短语用于试听比较。数量越多，评估与渲染可能越耗时。",
  "s_output": "指定合成练习数据的保存目录。生成完成后，该数据集会自动设为当前数据集。",
  "s_phrases": "要合成的短语数量。数量越多，生成测试数据所需时间和磁盘空间通常越多。",
  "s_notes": "每条练习短语包含的音符数量。它影响合成数据的长度与复杂程度。",
  "c_data": "选择需要转换音素拼写的数据集目录。原始数据集不会被直接修改。",
  "c_output": "指定转换后的数据集保存目录。建议使用新目录，以便保留原始数据并比较转换结果。",
  "c_src": "选择源数据集使用的字典文本文件，用于解释原有音素拼写。请确保它与源数据集的标注体系相匹配。",
  "c_tgt": "选择目标 Synthesizer V 声库对应的字典文本文件。目标字典需要与所选声库及语言相匹配。",
  "c_coda": "控制一个音节中分配给韵尾的时长比例。只在特定字典转换场景中需要调整；不确定时建议先使用默认值。",
  "svpk_name": "声库对外显示的名称，会写入 SVPK 的 package-config 元数据。",
  "svpk_vendor": "声库供应商、作者或工作室名称，会写入 package-config 的 vendorName。",
  "svpk_version": "SVPK 元数据要求 versionString（文本）和 versionNumber（整数）。这个输入使用整数版本，例如 101，并将同一版本写入两个字段。",
  "svpk_header_preset": "选择要写入外层 256 字节字段的诊断数据：全零、随机字节、SHA-256 链式扩展或合成 AES-256-CBC。非零模式只是测试数据，不是 Synthesizer V 官方签名/密钥；所有模式生成的文件仍不可安装。",
  "svpk_profile": "选择声库头像 PNG。打包器会将其作为根目录 profile.png 写入包内。",
  "svpk_background": "选择声库背景 PNG。为匹配目前检查过的样例，包内文件条目使用其样例拼写 background.png。",
  "svpk_output": "生成的 SVPK 文件完整路径。推荐保存在训练结果附近，并使用 .svpk 扩展名。"
};
const STATIC_HELP = {
  "language": "选择录音所使用的语言/音素集。它必须与数据集标签采用的音素体系一致，否则可能出现声库不认识某些音素的问题。",
  "i_src": "选择素材组织方式：现有 WAV/LAB、分开放置的 WAV 与 LAB、从 WAV 文件名生成 LAB，或 TrainV6 manifest.json。",
  "i_words": "开启后，导入时会借助第 1 步选择的声库为词边界写入额外信息。若之后要训练 Timing 或 Pitch，这些信息可能是必要条件；只训练 Sound 时可以不勾选。",
  "i_map": "可选：给导入过程中的音素改名。每行写一组 old=new；适用于标注里的音素名与声库使用的名称不一致的情况。",
  "m_acoustic": "Sound（声学模型）决定合成声音的音色与音频细节。它可使用只有音素与时长的数据集，也是建议优先尝试的模型。",
  "m_duration": "Timing（时长模型）学习每个音素在音符中的时长分配。数据集需要包含音符信息和词边界；缺少这些信息时不建议选择。",
  "m_pitch": "Pitch（音高模型）学习音高曲线及相关表达。需要包含音符与词边界的标注；不带音符的数据集无法有效训练该模型。",
  "m_vocoder": "Vocoder 是最终音频生成阶段的模型。它属于可选项目，训练通常更慢；初次使用建议先保持关闭。",
  "t_preserve_non_acoustic_models": "仅用于 info.nofs 共享模型声库。启用后只训练声学模型；原始 Timing、Pitch/F0 和 Vocoder 的模型 ID 与 DNNI 字节保持不变。Timing/Pitch 不会加载或校验；Vocoder 必须作为冻结解码器参与声学音频损失，但不会训练或进行输入范围校准。依赖声码器的自动响度校准会跳过，训练 WAV 的 F0 提取仍会保留。普通 voice.nofs 不支持。",
  "p_quick": "快速测试：运行少量步数，用来确认声库、数据集和运行环境可以正常工作。它不是完整训练。",
  "p_std": "标准训练：使用适中的初始步数，适合作为第一轮正式尝试。之后可根据验证结果调整。",
  "p_long": "较长训练：步数更多、耗时更长，并会缓慢降低学习率。只有在短训练正常且验证结果仍有改善空间时再尝试。",
  "t_savebest": "每当验证指标改善时额外保存一次权重版本，方便保留训练中间表现较好的结果，即使之后提前停止也有机会使用。",
  "t_params": "选择要更新的参数范围。“Everything” 会训练所选模型的权重，并通过 Anchor 约束接近原权重；“Only the voice's identity vectors” 只训练声库的说话人、风格和 Vocal Mode 向量，通常更轻量。",
  "t_device": "选择训练设备。Automatic 会按环境选择可用设备；NVIDIA 使用 CUDA，Apple 芯片可使用 MPS，CPU 通常较慢但可作为兼容性选项。",
  "t_keep": "勾选后保留录音的原始响度；不勾选时，程序通常会将录音缩放到起始声库的输出响度。",
  "t_resume": "尝试从运行目录中已有的 state.pt 继续训练。应保持此前使用的模型与参数范围一致，并将 Steps 设为高于已完成步数的值。",
  "d_mode_have": "选择此项表示你已经有 TrainSV 数据集：目录内应包含 wavs 子文件夹和 transcriptions.csv。若尚未生成数据集，请选择另一项并导入 .wav/.lab 文件。",
  "d_mode_lab": "选择此项表示你手头是录音和 .lab 音素标注，需要先将它们转换为 TrainSV 数据集，然后才能进行训练。"
};
const controlLabels = {
  "language": "Language your recordings are sung in",
  "i_src": "Where are your files?",
  "i_words": "Mark word boundaries using my voice",
  "i_map": "Rename phonemes",
  "m_acoustic": "Sound (acoustic)",
  "m_duration": "Timing (duration)",
  "m_pitch": "Pitch (melody and vibrato)",
  "m_vocoder": "Vocoder (final audio stage)",
  "t_preserve_non_acoustic_models": "Preserve original voicebank models (except acoustic)",
  "p_quick": "Quick test",
  "p_std": "Standard",
  "p_long": "Thorough",
  "t_savebest": "Also save whenever it improves",
  "t_params": "What to adjust",
  "t_device": "Computer part to use",
  "t_keep": "Keep recordings at their original loudness",
  "t_resume": "Continue the previous run",
  "d_mode_have": "I already have a dataset",
  "d_mode_lab": "I have recordings and .lab files"
};


function helpMarkup(label, text) {
  return '<span class="help-wrap"><button class="help-btn" type="button" data-help-toggle aria-label="Help: ' + esc(label) + '" aria-expanded="false" title="Help">?</button><span class="help-pop" role="tooltip" hidden>' + esc(text) + '</span></span>';
}
function addStaticHelp() {
  Object.entries(STATIC_HELP).forEach(([id, text]) => {
    const control = document.getElementById(id);
    if (!control) return;
    const choice = control.closest("label.choice, .seg label");
    const template = document.createElement("template");
    template.innerHTML = helpMarkup(controlLabels[id] || id, text);
    const node = template.content.firstElementChild;
    if (choice) {
      const title = choice.querySelector("b");
      if (title) title.insertAdjacentElement("afterend", node);
      else choice.append(node);
      return;
    }
    const label = document.querySelector('label[for="' + id + '"]');
    if (!label) return;
    const row = document.createElement("div");
    row.className = "field-label-row";
    label.parentNode.insertBefore(row, label);
    row.append(label, node);
  });
}
function closeHelp(except) {
  $$(".help-wrap.open").forEach(wrap => {
    if (wrap === except) return;
    wrap.classList.remove("open");
    const button = $("[data-help-toggle]", wrap);
    const tip = $(".help-pop", wrap);
    if (button) button.setAttribute("aria-expanded", "false");
    if (tip) tip.hidden = true;
  });
}
function wireHelpInteractions() {
  document.addEventListener("click", event => {
    const button = event.target.closest("[data-help-toggle]");
    if (button) {
      event.preventDefault();
      event.stopPropagation();
      const wrap = button.closest(".help-wrap");
      const shouldOpen = !wrap.classList.contains("open");
      closeHelp(wrap);
      wrap.classList.toggle("open", shouldOpen);
      button.setAttribute("aria-expanded", String(shouldOpen));
      const tip = $(".help-pop", wrap);
      if (tip) tip.hidden = !shouldOpen;
      return;
    }
    if (event.target.closest(".help-wrap")) return;
    closeHelp();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeHelp();
  });
}

/* ---------- build the repeated field markup ---------- */
function buildFields(){
  $$(".field[data-id]").forEach(f=>{
    const d=f.dataset,id=d.id;
    const description = FIELD_HELP[id] || (d.hint ? d.hint.replace(/<[^>]*>/g, "") : "This setting affects the current operation. Choose based on your training goal.");
    let h='<div class="field-label-row"><label for="' + id + '">' + esc(d.label) + '</label>' + helpMarkup(d.label, description) + '</div>';
    if(d.kind==="path")h+=`<div class="path"><input type="text" id="${id}" data-s placeholder="${esc(d.ph||"")}" spellcheck="false" autocomplete="off"><button type="button" class="ghost" data-browse="${id}" data-mode="${d.mode}" data-ext="${d.ext||""}">Browse…</button></div>`;
    else if(d.kind==="select"){
      const options=(d.options||"").split(";").filter(Boolean).map(item=>{const pair=item.split("|");return `<option value="${esc(pair[0])}">${esc(pair.slice(1).join("|"))}</option>`;}).join("");
      h+=`<select id="${id}" data-s>${options}</select>`;
    }
    else h+=`<input type="text" id="${id}" data-s placeholder="${esc(d.ph||"")}" spellcheck="false" autocomplete="off">`;
    f.innerHTML=h;
    if(d.value)document.getElementById(id).value=d.value;
  });
  $$("[data-run]").forEach(b=>{
    const k=b.dataset.run,box=b.closest(".actions");
    const x=document.createElement("div");
    x.innerHTML=`<div class="needs" id="needs-${k}"></div><details class="cmd"><summary>Show the command this runs</summary><pre id="cmd-${k}" class="mono"></pre></details>`;
    box.after(x);
  });
}

/* ---------- settings ---------- */
let saveTimer;
function persist(){
  clearTimeout(saveTimer);
  saveTimer=setTimeout(()=>{
    const s={};
    $$("[data-s]").forEach(e=>{if(e.id)s[e.id]=(e.type==="checkbox"||e.type==="radio")?e.checked:e.value});
    s._extra=EXTRA;
    api("settings",s).catch(()=>{});
  },300);
}
function restore(s){
  Object.entries(s||{}).forEach(([k,v])=>{
    if(k==="_extra"){EXTRA=Object.assign(EXTRA,v||{});EXTRA.manual=EXTRA.manual||{};return}
    const e=document.getElementById(k);if(!e)return;
    if(e.type==="checkbox"||e.type==="radio")e.checked=!!v;else e.value=v;
  });
}
function setVal(id,v){const e=document.getElementById(id);if(e){e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}}

/* ---------- suggestions for output paths ---------- */
function setAuto(id,v){if(!EXTRA.manual[id]){const e=document.getElementById(id);if(e)e.value=v}}
function suggest(){
  const name=val("v_name"),sl=slug(name),v=val("voice");
  const safe=(name||sl).replace(/[<>:"\/\\|?*]/g,"").trim()||sl;
  if(ENV.root){
    setAuto("t_output",v&&isInfo(v)?join(dirname(dirname(v)),safe):join(ENV.root,"out",sl,"voice.nofs"));
    setAuto("t_run",join(ENV.root,"runs",sl));
    setAuto("e_out",join(ENV.root,"runs",sl,"eval"));
    const src=val("i_src");
    let from=src==="manifest"?dirname(val("i_manifest")||""):src==="split"?val("i_wavs"):src==="filenames"?val("i_wavs_names"):val("i_corpus");
    if(/^(wav|wavs)$/i.test(basename(from||"")))from=dirname(from);
    setAuto("i_output",join(ENV.root,"data",from?slug(basename(from)):"dataset"));
    setAuto("s_output",join(ENV.root,"data","practice"));
    setAuto("c_output",join(ENV.root,"data","converted"));
  }
}

/* ---------- command builders ---------- */
function lang(){return val("language")||"ja"}
function trainedModels(){const m=MODELS.filter(x=>chk("m_"+x));return m.length?m:["acoustic"]}
function applyF0ProfileControls(profile,clearOverrides=true){
  const preset={
    standard:{floor:"",ceiling:"",threshold:"",smoothing:"0",octave:false},
    vocaloid:{floor:"",ceiling:"",threshold:"",smoothing:"5",octave:true},
    low:{floor:"",ceiling:"",threshold:"",smoothing:"0",octave:false},
    high:{floor:"",ceiling:"",threshold:"",smoothing:"0",octave:false},
    wide:{floor:"",ceiling:"",threshold:"",smoothing:"0",octave:false},
    sensitive:{floor:"",ceiling:"",threshold:"",smoothing:"3",octave:false}
  }[profile]||null;
  if(!preset)return;
  const controls=[["t_f0_smoothing",preset.smoothing]];
  if(clearOverrides)controls.unshift(["t_f0_floor",preset.floor],["t_f0_ceiling",preset.ceiling],["t_f0_threshold",preset.threshold]);
  controls.forEach(([id,value])=>{const el=document.getElementById(id);if(el)el.value=value});
  const octave=document.getElementById("t_f0_octave");if(octave)octave.checked=preset.octave;
}
function applyTimbrePreset(name){
  const notes={
    manual:"Manual mode leaves current parameters unchanged.",
    preserve:"Best when you want minimal change: trains only per-voice identity/style embeddings. It does not retrain the acoustic network.",
    gentle:"Use when your target data is limited or you want to stay close to the base. This conservative all-weight fit is not a universal fix for pronunciation errors.",
    balanced:"Use as a baseline on a clean, well-covered dataset. It adds no special transition emphasis or brightness loss.",
    natural:"For VOCALOID/UTAU-style sources with abrupt phone handoffs: lower learning rate, stronger anchor, boundary-window sampling at 0.5, mild non-silence boundary emphasis (weight 1.15, radius 2), and local multi-resolution spectral loss at 0.25; brightness loss stays off. This is not waveform smoothing. Audit labels first and A/B test against the stock base.",
    "soft-consonants":"Use when consonant attacks are too sharp: slightly lower consonant-frame emphasis while keeping transition and brightness extras off. It may soften articulation or reduce intelligibility, so compare plosives and fricatives on held-out phrases.",
    "clarity-lite":"A light experiment for renders that are darker than the source WAV: brightness-balance weight 0.025. It matches the source spectrum, not a generic treble boost, and cannot guarantee to remove nasality.",
    clarity:"This stronger experiment uses brightness-balance weight 0.05. Try only when held-out comparisons show the render lacks upper-band energy relative to the source; it may also change timbre.",
    detail:"Most aggressive brightness experiment (0.10) with a weaker anchor. Use last, compare held-out phrases, and stop if consonants become sharper or the tone starts to drift."
  };
  const help=document.getElementById("timbrePresetHelp");
  if(help)help.textContent=notes[name]||notes.manual;
  const presets={
    preserve:{t_params:"embeddings",t_steps:"800",t_lr:"",t_warmup:"50",t_lrend:"3e-4",t_anchor:"1.5",t_batch:"2",t_cons:"1.0",t_transition_weight:"1",t_transition_frames:"2",t_transition_sampling:"0",t_transition_local_weight:"0",t_acoustic_dnni_version:"2",t_brightness_weight:"0"},
    gentle:{t_params:"all",t_steps:"1500",t_lr:"1e-5",t_warmup:"100",t_lrend:"1e-6",t_anchor:"1.5",t_batch:"2",t_cons:"1.0",t_transition_weight:"1",t_transition_frames:"2",t_transition_sampling:"0",t_transition_local_weight:"0",t_acoustic_dnni_version:"2",t_brightness_weight:"0"},
    balanced:{t_params:"all",t_steps:"3000",t_lr:"2e-5",t_warmup:"100",t_lrend:"2e-6",t_anchor:"1.0",t_batch:"2",t_cons:"1.0",t_transition_weight:"1",t_transition_frames:"2",t_transition_sampling:"0",t_transition_local_weight:"0",t_acoustic_dnni_version:"2",t_brightness_weight:"0"},
    natural:{t_params:"all",t_steps:"2500",t_lr:"1e-5",t_warmup:"100",t_lrend:"1e-6",t_anchor:"1.5",t_batch:"2",t_cons:"1.0",t_transition_weight:"1.15",t_transition_frames:"2",t_transition_sampling:"0.5",t_transition_local_weight:"0.25",t_acoustic_dnni_version:"2",t_brightness_weight:"0"},
    "soft-consonants":{t_params:"all",t_steps:"2000",t_lr:"1e-5",t_warmup:"100",t_lrend:"1e-6",t_anchor:"1.5",t_batch:"2",t_cons:"0.85",t_transition_weight:"1",t_transition_frames:"2",t_transition_sampling:"0",t_transition_local_weight:"0",t_acoustic_dnni_version:"2",t_brightness_weight:"0"},
    "clarity-lite":{t_params:"all",t_steps:"2500",t_lr:"1.5e-5",t_warmup:"100",t_lrend:"1e-6",t_anchor:"1.25",t_batch:"2",t_cons:"1.0",t_transition_weight:"1",t_transition_frames:"2",t_transition_sampling:"0",t_transition_local_weight:"0",t_acoustic_dnni_version:"2",t_brightness_weight:"0.025"},
    clarity:{t_params:"all",t_steps:"3000",t_lr:"2e-5",t_warmup:"100",t_lrend:"2e-6",t_anchor:"1.25",t_batch:"2",t_cons:"1.0",t_transition_weight:"1",t_transition_frames:"2",t_transition_sampling:"0",t_transition_local_weight:"0",t_acoustic_dnni_version:"2",t_brightness_weight:"0.05"},
    detail:{t_params:"all",t_steps:"4000",t_lr:"2e-5",t_warmup:"100",t_lrend:"2e-6",t_anchor:"0.8",t_batch:"2",t_cons:"1.0",t_transition_weight:"1",t_transition_frames:"2",t_transition_sampling:"0",t_transition_local_weight:"0",t_acoustic_dnni_version:"2",t_brightness_weight:"0.10"}
  };
  const preset=presets[name];if(!preset)return;
  Object.entries(preset).forEach(([id,value])=>{const el=document.getElementById(id);if(el)el.value=value});
}
const B={
  inspect(){const p=[];if(!val("voice"))p.push("Choose a voice file.");
    return{cmd:"inspect",label:"Looking inside the voice",args:["--voice",val("voice")],p,log:true}},
  score(){const p=[];if(!val("voice"))p.push("Choose a voice (step 1).");dsNeed(p,"Choose a dataset first.");
    const a=["--voice",val("voice"),"--data",val("dataset"),"--language",lang(),"--models",trainedModels().join(",")];
    addOpt(a,p,"--validation","t_val");addOpt(a,p,"--seed","t_seed");
    return{cmd:"evaluate",label:"Scoring the original voice",args:a,p,log:true}},
  import(){const p=[],a=[],src=val("i_src");
    if(src==="folder"){if(!val("i_corpus"))p.push("Choose the recordings folder.");a.push("--corpus",val("i_corpus"))}
    else if(src==="manifest"){if(!val("i_manifest"))p.push("Choose the manifest.json file.");a.push("--corpus",val("i_manifest"))}
    else if(src==="filenames"){if(!val("i_wavs_names"))p.push("Choose the folder of WAV files.");a.push("--from-filenames","--wavs",val("i_wavs_names"),"--source-phoneme-system",val("i_source_system")||"auto");if(val("i_phone_table"))a.push("--phone-table",val("i_phone_table"))}
    else{if(!val("i_wavs"))p.push("Choose the folder of .wav files.");if(!val("i_labs"))p.push("Choose the folder of .lab files.");a.push("--labs",val("i_labs"),"--wavs",val("i_wavs"))}
    if(!val("i_output"))p.push("Choose where to save the dataset.");a.push("--output",val("i_output"));
    if(val("i_songs"))a.push("--songs",val("i_songs"));
    if(chk("i_words")){if(!val("voice"))p.push("Choose a voice (step 1), or untick “Mark word boundaries”.");a.push("--voice",val("voice"),"--language",lang())}
    const pairs=val("i_map").split(/[\s,]+/).filter(x=>x.includes("="));if(pairs.length)a.push("--map",...pairs);
    return{cmd:"import-lab",label:"Converting recordings to a dataset",args:a,p}},
  finetune(pitchOnly=false){const p=[],a=[],preserve=chk("t_preserve_non_acoustic_models");
    if(!val("voice"))p.push("Choose a voice (step 1).");dsNeed(p,"Choose a dataset (step 2).");
    if(!val("v_name"))p.push("Name your new voice.");if(!val("t_output"))p.push("Choose where to save the new voice.");
    if(preserve&&(!isInfo(val("voice"))||!VOICE_OK))p.push("Preserving non-acoustic models requires an existing info.nofs shared-model voicebank.");
    if(preserve&&pitchOnly)p.push("Pitch-only training is unavailable while preserving non-acoustic models; this mode trains Sound only.");
    const customIds=["t_custom_duration_dnni","t_custom_pitch_dnni","t_custom_vocoder_dnni"];
    const hasCustom=customIds.some(id=>!!val(id));
    if(preserve&&hasCustom)p.push("Custom DNNI overrides cannot be combined with preservation mode.");
    if(chk("t_resume")&&hasCustom)p.push("Custom DNNI overrides cannot be combined with resume; use a separate run directory.");
    if(val("t_custom_pitch_dnni")&&val("t_pitch_source")==="pretrained")p.push("Choose either a custom Pitch/F0 DNNI or pretrained Pitch initialization, not both.");
    const models=pitchOnly?["pitch"]:(preserve?["acoustic"]:MODELS.filter(x=>chk("m_"+x)));if(!models.length)p.push("Tick at least one thing to learn.");
    a.push("--voice",val("voice"),"--data",val("dataset"),"--output",val("t_output"),"--language",lang(),"--models",models.join(","),"--name",val("v_name"));
    if(preserve)a.push("--preserve-non-acoustic-models");
    if(!preserve)[["--custom-duration-dnni","t_custom_duration_dnni"],["--custom-pitch-dnni","t_custom_pitch_dnni"],["--custom-vocoder-dnni","t_custom_vocoder_dnni"]].forEach(([flag,id])=>{
      const path=val(id);if(!path)return;
      if(!/\.dnni$/i.test(path))p.push("Replacement model must be a .dnni file.");else a.push(flag,path);
    });
    a.push("--f0-profile",val("t_f0_profile")||"standard");
    [
      ["--f0-floor","t_f0_floor"],["--f0-ceiling","t_f0_ceiling"],["--f0-voicing-threshold","t_f0_threshold"],
      ["--f0-smoothing","t_f0_smoothing"],["--f0-reference-cents","t_f0_correction"]
    ].forEach(([flag,id])=>addOpt(a,p,flag,id));
    if(!preserve)[
      ["--pitch-smoothing-hz","t_pitch_smoothing"],["--pitch-melody-weight","t_pitch_melody_weight"],
      ["--pitch-motion-weight","t_pitch_motion_weight"]
    ].forEach(([flag,id])=>addOpt(a,p,flag,id));
    a.push(chk("t_f0_octave")?"--f0-octave-correction":"--no-f0-octave-correction");
    if(!preserve&&val("t_pitch_source")==="pretrained"){
      if(!val("t_pitch_model"))p.push("Select a .dnni file when using an external pretrained Pitch model.");
      else if(!models.includes("pitch"))p.push("Select Pitch in the model list above to import pretrained weights.");
      else a.push("--pitch-base-model",val("t_pitch_model"));
    }
    addOpt(a,p,"--steps","t_steps",true);
    if(val("t_params")&&val("t_params")!=="all")a.push("--params",val("t_params"));
    [["--lr","t_lr"],["--warmup","t_warmup"],["--lr-end","t_lrend"],["--anchor","t_anchor"],["--batch","t_batch"],["--consonant-weight","t_cons"],["--transition-weight","t_transition_weight"],["--transition-frames","t_transition_frames"],["--transition-sampling-probability","t_transition_sampling"],["--transition-local-weight","t_transition_local_weight"],["--brightness-weight","t_brightness_weight"],["--validation","t_val"],["--evaluate-every","t_evalevery"],["--log-every","t_logevery"],["--seed","t_seed"]].forEach(([f,i])=>addOpt(a,p,f,i));
    if(!preserve)addOpt(a,p,"--gain","t_gain");
    [["--device","t_device"],["--phoneme-map","t_pmap"],["--vocal-modes","t_modes"],["--version","t_version"],["--final-version","t_final"],["--acoustic-dnni-version","t_acoustic_dnni_version"],["--history","t_history"],["--run","t_run"]].forEach(([f,i])=>{if(val(i))a.push(f,val(i))});
    if(!preserve&&val("t_levelref"))a.push("--level-reference",val("t_levelref"));
    if(chk("t_keep"))a.push("--keep-level");if(chk("t_savebest"))a.push("--save-best");if(chk("t_resume"))a.push("--resume");
    return{cmd:"finetune",label:pitchOnly?"Pitch model training":"Training",args:a,p}},
  pitch(){return this.finetune(true)},
  evaluate(){const p=[],a=[];
    if(!val("voice"))p.push("Choose the original voice (step 1).");if(!val("e_new"))p.push("Choose your new voice.");
    dsNeed(p,"Choose a dataset (step 2).");if(!val("e_out"))p.push("Choose where to save the comparison.");
    a.push("--voice",val("voice"),"--voice",val("e_new"),"--data",val("dataset"),"--language",lang(),"--models",trainedModels().join(","));
    a.push("--f0-profile",val("t_f0_profile")||"standard");
    [
      ["--f0-floor","t_f0_floor"],["--f0-ceiling","t_f0_ceiling"],["--f0-voicing-threshold","t_f0_threshold"],
      ["--f0-smoothing","t_f0_smoothing"],["--f0-reference-cents","t_f0_correction"],
      ["--pitch-smoothing-hz","t_pitch_smoothing"],["--pitch-melody-weight","t_pitch_melody_weight"],
      ["--pitch-motion-weight","t_pitch_motion_weight"],["--brightness-weight","t_brightness_weight"]
    ].forEach(([flag,id])=>addOpt(a,p,flag,id));
    a.push(chk("t_f0_octave")?"--f0-octave-correction":"--no-f0-octave-correction");
    addOpt(a,p,"--validation","t_val");addOpt(a,p,"--seed","t_seed");
    if(val("t_pmap"))a.push("--phoneme-map",val("t_pmap"));
    a.push("--out",val("e_out"));addOpt(a,p,"--max-phrases","e_max");
    return{cmd:"evaluate",label:"Comparing voices",args:a,p}},
  svpk(){const p=[],a=[],voicePath=val("e_new")||val("t_output"),version=val("svpk_version");
    if(!voicePath)p.push("Choose the trained voice.nofs file first.");
    else if(isInfo(voicePath))p.push("This is an info.*.nofs shared-model voicebank; the SVPK diagnostic exporter supports only a regular single-file voice.nofs.");
    else if(!/\.nofs$/i.test(voicePath))p.push("Choose the trained .nofs file, not a directory or another file type.");
    if(!val("svpk_name"))p.push("Enter a voicebank name.");
    if(!val("svpk_vendor"))p.push("Enter a vendor / author.");
    if(!/^\d+$/.test(version)||!Number.isSafeInteger(Number(version)))p.push("Version must be a non-negative safe integer, e.g. 101.");
    if(!val("svpk_profile"))p.push("Choose a profile.png avatar image.");
    if(!val("svpk_background"))p.push("Choose a background PNG image.");
    let output=val("svpk_output");
    if(!output)p.push("Choose an SVPK output path.");
    else if(!/\.svpk$/i.test(output)){output += ".svpk";const el=document.getElementById("svpk_output");if(el)el.value=output;}
    a.push("--voice",voicePath,"--profile",val("svpk_profile"),"--background",val("svpk_background"),
      "--output",output,"--name",val("svpk_name"),"--vendor",val("svpk_vendor"),
      "--version-string",version,"--version-number",String(Number(version)),
      "--header-preset",val("svpk_header_preset")||"zero");
    return{cmd:"package-svpk",label:"Build SVPK diagnostic package",args:a,p}
  },
  sfpk(){const p=[],a=[],voicePath=val("e_new")||val("t_output"),version=val("sfpk_version");
    if(!voicePath)p.push("Choose the trained voicebank file first.");
    else if(!/\.nofs$/i.test(voicePath))p.push("Choose a .nofs file, not a directory or another file type.");
    
    if(!val("sfpk_models_root"))p.push("Choose the Synthesizer V models directory.");
    let output=val("sfpk_output");
    if(!output)p.push("Choose an SFPK output path.");
    else if(!/\.sfpk$/i.test(output)){output += ".sfpk";const el=document.getElementById("sfpk_output");if(el)el.value=output;}
    a.push("--voice",voicePath,"--models-root",val("sfpk_models_root"),"--output",output);
    if(val("sfpk_name"))a.push("--name",val("sfpk_name"));
    if(val("sfpk_vendor"))a.push("--vendor",val("sfpk_vendor"));
    if(version)a.push("--version",version);
    if(val("sfpk_profile"))a.push("--profile",val("sfpk_profile"));
    if(val("sfpk_background"))a.push("--background",val("sfpk_background"));
    return{cmd:"package-sfpk",label:"Build SFPK test package",args:a,p}
  },
  synth(){const p=[];if(!val("voice"))p.push("Choose a voice (step 1).");if(!val("s_output"))p.push("Choose where to save the data.");
    const a=["--voice",val("voice"),"--output",val("s_output"),"--language",lang()];
    addOpt(a,p,"--phrases","s_phrases");addOpt(a,p,"--notes","s_notes");
    return{cmd:"synthesize-dataset",label:"Making practice data",args:a,p}},
  convert(){const p=[];
    [["c_data","Choose the source dataset."],["c_output","Choose where to save the converted dataset."],["c_src","Choose the source dictionary."],["c_tgt","Choose the Synthesizer V dictionary."],["voice","Choose a voice (step 1)."]].forEach(([i,m])=>{if(!val(i))p.push(m)});
    const a=["--data",val("c_data"),"--output",val("c_output"),"--source-dict",val("c_src"),"--target-dict",val("c_tgt"),"--voice",val("voice"),"--language",lang()];
    addOpt(a,p,"--coda-share","c_coda");
    return{cmd:"convert-dict",label:"Converting the dictionary",args:a,p}},
  audioMidi(){const p=[],source=val("am_input"),output=val("am_output");
    if(!source)p.push("Choose a source audio file.");
    else if(!/\\.(wav|flac|aiff|aif)$/i.test(source))p.push("Choose a WAV, FLAC or AIFF audio file.");
    if(!output)p.push("Choose where to save the MIDI file.");
    else if(!/\\.(mid|midi)$/i.test(output))p.push("The output filename must end in .mid or .midi.");
    if(source&&output&&source.replace(/\\\\/g,"/").toLowerCase()===output.replace(/\\\\/g,"/").toLowerCase())p.push("MIDI output must not overwrite the source audio.");
    const a=["--input",source,"--output",output];
    [["--pitch-floor","am_floor"],["--pitch-ceiling","am_ceiling"],["--voicing-threshold","am_voicing"],["--min-note-ms","am_min_note"],["--median-window","am_median"],["--a4-hz","am_a4"],["--transpose-semitones","am_transpose"],["--tempo-bpm","am_tempo"]].forEach(([flag,id])=>addOpt(a,p,flag,id));
    const floor=Number(val("am_floor")),ceiling=Number(val("am_ceiling")),threshold=Number(val("am_voicing")),minimum=Number(val("am_min_note")),window=Number(val("am_median")),a4=Number(val("am_a4")),transpose=Number(val("am_transpose")),tempo=Number(val("am_tempo"));
    if(!(floor>=20&&floor<ceiling&&ceiling<=5000))p.push("Pitch range must satisfy 20 ≤ floor < ceiling ≤ 5000 Hz.");
    if(!(threshold>=0.05&&threshold<=1))p.push("Voicing threshold must be between 0.05 and 1.0.");
    if(!(minimum>=20&&minimum<=1000))p.push("Minimum note duration must be between 20 and 1000 ms.");
    if(!(Number.isInteger(window)&&window>=1&&window<=21&&window%2===1))p.push("Smoothing window must be an odd integer from 1 to 21.");
    if(!(a4>=400&&a4<=480))p.push("A4 reference must be between 400 and 480 Hz.");
    if(!(Number.isInteger(transpose)&&transpose>=-24&&transpose<=24))p.push("Transpose must be an integer from −24 to +24 semitones.");
    if(!(tempo>=20&&tempo<=300))p.push("MIDI tempo must be between 20 and 300 BPM.");
    return{cmd:"audio-to-midi",label:"Converting audio to MIDI",args:a,p,log:true}},
  replaceModel(){const p=[],source=val("rm_source"),modelsRoot=val("rm_models_root"),output=val("rm_output");
    const replacements=[
      ["duration","rm_duration_model","--duration-model"],
      ["pitch","rm_pitch_model","--pitch-model-file"],
      ["vocoder","rm_vocoder_model","--vocoder-model"]
    ].filter(([,id])=>!!val(id)).map(([role,id,flag])=>({role,path:val(id),flag}));
    if(!source)p.push("Choose source info.nofs.");
    else if(!isInfo(source))p.push("Only info.nofs JSON voicebanks are supported.");
    if(!modelsRoot)p.push("Choose the Synthesizer V models folder.");
    if(!replacements.length)p.push("Choose at least one model to replace.");
    replacements.forEach(item=>{if(!/\.dnni$/i.test(item.path))p.push("Replacement model must be a .dnni file.")});
    if(!output)p.push("Choose a new output info.nofs path.");
    else if(/\.nofs$/i.test(output))p.push("Choose an output package folder, not an info.nofs filename; TrainSV creates the directory layout for you.");
    const norm=path=>path.replace(/\\/g,"/").replace(/\/+$/,"").toLowerCase();
    if(source&&output&&norm(source)===norm(output))p.push("Output must not overwrite the source info.nofs.");
    const a=["--voice",source,"--models-root",modelsRoot,"--output",output];
    replacements.forEach(item=>a.push(item.flag,item.path));
    return{cmd:"replace-model",label:"Batch replace non-acoustic DNNI models",args:a,p,log:true}}
};
function dsNeed(p,msg){if(!val("dataset"))p.push(msg);else if(!DS)p.push("The dataset folder is not ready (see the message in step 2).")}
function addOpt(a,p,flag,id,required){
  const v=val(id);
  if(v===""){if(required)p.push("Enter a number for “"+flag.replace("--","")+"”.");return}
  if(!isFinite(Number(v))){p.push("“"+flag.replace("--","")+"” must be a number (you typed “"+v+"”).");return}
  a.push(flag,v);
}

/* ---------- refresh run buttons, previews, page hints ---------- */
function refresh(){
  syncPreserveModelsUI();
  const running=JOB&&JOB.status==="running";
  $$("[data-run]").forEach(b=>{
    const k=b.dataset.run,r=B[k]();
    b.disabled=r.p.length>0||running;
    $("#needs-"+k).innerHTML=r.p.length?"Before you can continue:<ul>"+r.p.map(x=>"<li>"+esc(x)+"</li>").join("")+"</ul>":"";
    $("#cmd-"+k).textContent="trainsv "+r.cmd+" "+r.args.map(quote).join(" ");
  });
  const warn=$("#trainWarn"),m=MODELS.filter(x=>chk("m_"+x));
  const wantsNotes=m.includes("duration")||m.includes("pitch");
  if(wantsNotes&&DS&&DS.rows&&DS.notes<DS.rows){
    warn.hidden=false;
    if(DS.notes){
      const noteWarning={
        en:`${DS.notes} of ${DS.rows} recordings have notes. The others will train Sound only. Pitch correction needs complete note and word-boundary labels; the F0 preset alone cannot repair the base Pitch network.`,
        "zh-CN":`${DS.notes}/${DS.rows} 条录音有音符标注。其余录音只能训练 Sound。Pitch 校正需要完整音符和词边界标签；F0 预设本身无法修复底模的 Pitch 网络。`,
        ja:`${DS.notes}/${DS.rows} 件の録音に音符があります。残りは Sound のみ学習されます。Pitch の補正には完全な音符と単語境界ラベルが必要で、F0 プリセットだけでは元の Pitch ネットワークを修正できません。`
      };
      warn.textContent=noteWarning[UI_LANG]||noteWarning.en;
    } else {
      const noNotesWarning={
        en:"Your dataset has no note labels yet, so Timing and Pitch cannot learn anything. Add note_seq/note_dur plus ph_num labels (for example from a suitable GAME workflow), or untick them to train Sound only. F0 analysis presets only measure and clean the reference WAV; they do not train the score-to-F0 Pitch network.",
        "zh-CN":"数据集尚无音符标注，因此 Timing 和 Pitch 无法学习。请添加 note_seq/note_dur 及 ph_num 标注（例如通过合适的 GAME 工作流），或取消勾选，仅训练 Sound。F0 分析预设只检测和清理参考 WAV 的基频，不会训练从乐谱预测 F0 的 Pitch 网络。",
        ja:"データセットに音符ラベルがないため、Timing と Pitch は学習できません。note_seq/note_dur と ph_num ラベル（適切な GAME ワークフローなど）を追加するか、チェックを外して Sound のみを学習してください。F0 プリセットは参照 WAV の F0 を解析・補正するだけで、楽譜から F0 を予測する Pitch ネットワークは学習しません。"
      };
      warn.textContent=noNotesWarning[UI_LANG]||noNotesWarning.en;
    }
  }
  else if(wantsNotes&&!DS){warn.hidden=false;warn.textContent="Timing and Pitch need a dataset with notes. Check your dataset in step 2."}
  else if(!wantsNotes&&DS&&DS.rows&&DS.notes===0){
    warn.hidden=false;
    const acousticOnlyWarning={
      en:"This dataset has no note labels, so this run can adapt Acoustic/Sound only. It can improve phoneme realization and timbre, but it cannot correct the base voice's note-to-F0 Pitch prediction. F0 presets only analyze the reference WAV; inspect phone_mapping_report.csv for Mandarin source-to-target phoneme mappings before training.",
      "zh-CN":"此数据集没有音符标注，因此本次只能微调 Acoustic/Sound。它可以改善音素实现和音色，但无法修正底模从音符预测 F0 的 Pitch。F0 预设只分析参考 WAV；普通话训练前还应检查 phone_mapping_report.csv 中的源音素到目标音素映射。",
      ja:"このデータセットには音符ラベルがないため、今回は Acoustic/Sound のみを調整できます。音素の実現や音色は改善できますが、元の音声モデルの音符から F0 への Pitch 予測は修正できません。F0 プリセットは参照 WAV のみを解析します。中国語の学習前に phone_mapping_report.csv の音素変換を確認してください。"
    };
    warn.textContent=acousticOnlyWarning[UI_LANG]||acousticOnlyWarning.en;
  }
  else warn.hidden=true;
  $("#e_models").textContent="It will check: "+trainedModels().map(x=>MODEL_NAME[x]).join(", ")+" (the same as in Train), using the same held-out recordings as training.";
  $$("#steps li").forEach(li=>{
    const s=li.dataset.step;
    const done=s==="voice"?VOICE_OK&&!!val("language"):s==="data"?!!(DS&&DS.rows>0&&DS.has_wavs):s==="train"?EXTRA.trained:EXTRA.evaluated;
    li.classList.toggle("done",!!done);
    const dot=$(".dot",li);if(dot)dot.textContent=done?"✓":({voice:1,data:2,train:3,results:4})[s];
  });
}

/* ---------- navigation ---------- */
function go(name){
  $$(".page").forEach(p=>p.classList.toggle("show",p.id==="page-"+name));
  $$("#steps li,.more li").forEach(li=>{li.classList.toggle("current",li.dataset.step===name);$("button.nav",li).toggleAttribute("aria-current",li.dataset.step===name);if(li.dataset.step===name)$("button.nav",li).setAttribute("aria-current","page")});
  location.hash=name;$(".pages").scrollTop=0;
}
function setMode(){const lab=chk("d_mode_lab");$("#pane-have").hidden=lab;$("#pane-lab").hidden=!lab;$$("[data-src]").forEach(d=>d.hidden=d.dataset.src!==val("i_src"))}

function syncPreserveModelsUI(){
  const option=$("#t_preserve_non_acoustic_models");
  if(!option)return;
  const infoPath=isInfo(val("voice"));
  option.disabled=!(infoPath&&VOICE_OK);
  if(!infoPath)option.checked=false;
  const active=infoPath&&VOICE_OK&&option.checked;
  ["m_duration","m_pitch","m_vocoder"].forEach(id=>{
    const control=$("#"+id);if(!control)return;
    if(active)control.checked=false;
    control.disabled=active;
  });
  if(active){const sound=$("#m_acoustic");if(sound)sound.checked=true;}
  const pitchPanel=$("#pitchTrainingPanel");if(pitchPanel)pitchPanel.disabled=active;
  ["t_levelref","t_gain","t_custom_duration_dnni","t_custom_pitch_dnni","t_custom_vocoder_dnni"].forEach(id=>{const control=$("#"+id);if(control)control.disabled=active;});
}

/* ---------- checks: voice, dataset, recordings ---------- */
function say(id,cls,text){const e=document.getElementById(id);if(e){e.className="status "+cls;e.textContent=text}}
let tVoice,tData,tSrc;
function checkVoice(){
  clearTimeout(tVoice);tVoice=setTimeout(async()=>{
    const v=val("voice");VOICE_OK=false;
    if(!v){say("voice_st","","");syncPreserveModelsUI();refresh();return}
    try{const s=await api("stat?path="+enc(v));
      if(s.isfile){VOICE_OK=true;say("voice_st","ok",isInfo(v)?"✓ Found. This is an info voice (shared models); the result is saved into a folder.":"✓ Found. This is a single-file voice.")}
      else {$("#t_preserve_non_acoustic_models").checked=false;say("voice_st","bad",s.isdir?"That is a folder. Choose the .nofs file inside it.":"File not found.")}}
    catch(e){$("#t_preserve_non_acoustic_models").checked=false;say("voice_st","bad",e.message)}
    syncPreserveModelsUI();suggest();refresh();
  },300);
}
async function runDatasetAudit(){
  const path=val("dataset"),box=$("#datasetAudit");
  if(!path){say("dataset_st","warn","Choose a dataset first.");return}
  box.hidden=false;box.textContent="Auditing recording coverage and adjacent phone transitions…";
  try{
    const report=await api("audit-dataset?path="+enc(path));
    const s=report.summary||{};
    const view={
      dataset:report.dataset,
      summary:s,
      fatal_errors:report.fatal_errors||[],
      missing_wavs:(report.missing_wavs||[]).slice(0,20),
      invalid_rows:(report.invalid_rows||[]).slice(0,20),
      duplicate_names:(report.duplicate_names||[]).slice(0,20),
      rare_phones:(report.rare_phones||[]).slice(0,30),
      rare_transitions:(report.rare_transitions||[]).slice(0,30),
      short_phone_examples:(report.short_phone_examples||[]).slice(0,20)
    };
    box.textContent=JSON.stringify(view,null,2);
    const errors=(s.missing_wavs||0)+(s.invalid_rows||0)+(s.duplicate_names||0)+(s.unreadable_wavs||0);
    say("dataset_st",errors?"warn":"ok",
      "Audit complete: "+(s.trainable_rows||0)+"/"+(s.rows||0)+" trainable recordings, "+
      (s.missing_wavs||0)+" missing WAVs, "+(s.invalid_rows||0)+" invalid rows, "+
      (report.rare_transitions||[]).length+" rare transitions. Details below.");
  }catch(e){box.textContent=e.message;say("dataset_st","bad","Dataset audit failed.")}
}
function checkDataset(){
  clearTimeout(tData);tData=setTimeout(async()=>{
    const v=val("dataset");DS=null;
    if(!v){say("dataset_st","","");refresh();return}
    try{const d=await api("check-dataset?path="+enc(v));
      if(!d.exists)say("dataset_st","bad","Folder not found.");
      else if(!d.has_csv||!d.has_wavs)say("dataset_st","bad","This folder needs a “wavs” folder and a “transcriptions.csv” file."+(d.has_csv||d.has_wavs?" One of them is missing.":" If you have .lab files, choose “I have recordings and .lab files” above."));
      else if(d.error)say("dataset_st","bad",d.error);
      else{DS=d;const mins=(d.seconds/60).toFixed(1);
        let t=`✓ ${d.rows} recordings, about ${mins} min of singing. `;
        t+=d.notes===d.rows&&d.rows?"Notes included, so Sound, Timing and Pitch can all be trained.":d.notes?`${d.notes} have notes.`:"No notes yet, so only Sound can be trained.";
        if(d.missing)t+=` Warning: ${d.missing} listed recordings have no .wav file.`;
        say("dataset_st",d.missing?"warn":"ok",t)}
    }catch(e){say("dataset_st","bad",e.message)}
    refresh();
  },350);
}
function checkSource(){
  clearTimeout(tSrc);tSrc=setTimeout(async()=>{
    const src=val("i_src");if(src==="manifest"){say("i_src_st","","");return}
    const dir=src==="folder"?val("i_corpus"):"";
    if(src==="folder"&&!dir){say("i_src_st","","");return}
    if(src==="filenames"&&!val("i_wavs_names")){say("i_src_st","","");return}
    try{
      if(src==="folder"){const f=await api("folder-info?path="+enc(dir));
        say("i_src_st",f.exists&&f.wav&&f.lab?"ok":"warn",!f.exists?"Folder not found.":`Found ${f.wav} .wav and ${f.lab} .lab files.`+(f.wav&&f.lab?"":" Each recording needs both."))}
      else if(src==="filenames"){
        const f=await api("folder-info?path="+enc(val("i_wavs_names")));
        say("i_src_st",f.exists&&f.wav?"ok":"warn",!f.exists?"Folder not found.":`Found ${f.wav||0} WAV files. Labels will be estimated from filenames and equal-duration timing.`)
      }
      else{const w=val("i_wavs"),l=val("i_labs");if(!w||!l){say("i_src_st","","");return}
        const [a,b]=await Promise.all([api("folder-info?path="+enc(w)),api("folder-info?path="+enc(l))]);
        say("i_src_st",a.wav&&b.lab?"ok":"warn",`Found ${a.wav||0} .wav and ${b.lab||0} .lab files.`)}
    }catch(e){say("i_src_st","bad",e.message)}
  },350);
}

/* ---------- file picker ---------- */
let PK=null;
async function openPicker(id,mode,ext){
  if(mode==="dynamic")mode=isInfo(val("voice"))?"dir":"save";
  const cur=val(id);
  PK={id,mode,ext,dir:"",name:mode==="save"?(cur&&/\.\w+$/.test(basename(cur))?basename(cur):(ext==="json"?"history.json":ext==="svpk"?"voice.svpk":ext==="mid"?"audio.mid":"voice.nofs")):""};
  $("#pkTitle").textContent=mode==="dir"?"Choose a folder":mode==="save"?"Choose a folder and file name":"Choose a file";
  await pkLoad(cur&&mode!=="dir"&&/\.\w+$/.test(basename(cur))?dirname(cur):cur);
  $("#picker").showModal();
}
async function pkLoad(path){
  try{
    const r=await api(`browse?path=${enc(path||"")}&files=${PK.mode==="file"?1:0}&ext=${enc(PK.ext||"")}`);
    PK.dir=r.path;$("#pkPath").textContent=r.path+(r.error?" ("+r.error+")":"");
    const ul=$("#pkList");ul.innerHTML="";
    const add=(label,fn)=>{const li=document.createElement("li"),b=document.createElement("button");b.type="button";b.textContent=label;b.onclick=fn;li.append(b);ul.append(li)};
    if(r.parent)add("⬑ Up one folder",()=>pkLoad(r.parent));
    add("🏠 Home folder",()=>pkLoad(r.home));add("📁 TrainSV folder",()=>pkLoad(r.project));
    if(r.roots.length>1||r.roots[0]!=="/")r.roots.forEach(x=>add("💽 "+x,()=>pkLoad(x)));
    r.entries.forEach(e=>add((e.dir?"📁 ":"📄 ")+e.name,()=>{
      if(e.dir)pkLoad(join(r.path,e.name));
      else if(PK.mode==="file")pkPick(join(r.path,e.name));
      else{PK.name=e.name;const n=$("#pkName");if(n)n.value=e.name}
    }));
    pkFoot();
  }catch(e){$("#pkPath").textContent=e.message}
}
function pkFoot(){
  const f=$("#pkFoot");f.innerHTML="";
  const mk=(h)=>{const d=document.createElement("div");d.style.display="contents";d.innerHTML=h;f.append(...d.childNodes)};
  if(PK.mode==="save")mk(`<input type="text" id="pkName" aria-label="File name" value="${esc(PK.name)}"><button class="primary" id="pkOk" type="button">Save here</button>`);
  if(PK.mode==="dir")mk(`<input type="text" id="pkNew" aria-label="New folder name" placeholder="New folder name"><button type="button" id="pkMk">Create</button><button class="primary" id="pkOk" type="button">Use this folder</button>`);
  mk(`<button type="button" id="pkCancel" class="ghost">Cancel</button>`);
  $("#pkCancel").onclick=()=>$("#picker").close();
  $("#pkOk")&&($("#pkOk").onclick=()=>{if(PK.mode==="save"){const n=$("#pkName").value.trim();if(!n)return;pkPick(join(PK.dir,n))}else pkPick(PK.dir)});
  $("#pkMk")&&($("#pkMk").onclick=async()=>{const n=$("#pkNew").value.trim();if(!n)return;const p=join(PK.dir,n);try{await api("mkdir",{path:p});pkPick(p)}catch(e){$("#pkPath").textContent=e.message}});
}
function pkPick(path){
  $("#picker").close();
  if(AUTO.includes(PK.id))EXTRA.manual[PK.id]=true;
  setVal(PK.id,path);
}

/* ---------- running jobs ---------- */
const DONE={
  import(ok){if(ok){setVal("dataset",val("i_output"));$("#d_mode_have").checked=true;setMode();JOB.next={text:"Next: train your voice",go:"train"}}},
  synth(ok){if(ok){setVal("dataset",val("s_output"));JOB.next={text:"Use it: go to Train",go:"train"}}},
  convert(ok){if(ok){setVal("dataset",val("c_output"));JOB.next={text:"Use it: go to Train",go:"train"}}},
  async finetune(ok){if(ok){EXTRA.trained=true;persist();try{const r=await api("resolve-voice?path="+enc(val("t_output")));if(r.path)setVal("e_new",r.path)}catch(e){}JOB.next={text:"Next: check the result",go:"results"};renderJob()}},
  evaluate(ok){if(ok){EXTRA.evaluated=true;persist();loadResults();JOB.next={text:"See the comparison",go:"results",to:"#results"};renderJob();if(JOB.kind==="evaluate"){go("results");showPackageFormatDialog()}}},
  svpk(ok){const el=$("#packageStatus");if(el){el.className="status "+(ok?"warn":"bad");el.textContent=ok?"SVPK diagnostic package created: "+val("svpk_output")+". This package is not installable or suitable for release.":"SVPK diagnostic package generation failed. See the job log for details."}},
  sfpk(ok){const el=$("#packageStatus");if(el){el.className="status "+(ok?"warn":"bad");el.textContent=ok?"SFPK test package created: "+val("sfpk_output")+". Model dependencies were written to the local directory: "+val("sfpk_models_root")+". Import and test rendering in Synthesizer V Studio Flat; runtime compatibility has not yet been verified.":"SFPK generation failed. See the job log for details."}}
};
async function startJob(kind){
  const r=B[kind]();if(r.p.length)return;
  try{
    await api("run",{command:r.cmd,args:r.args,label:r.label});
    JOB={kind,label:r.label,status:"running",lines:[],next:null,since:0,log:!!r.log,logNeedsSync:true,logDropped:0,logRendered:0,step:0,total:0,sps:0,val0:null,valN:null,saved:"",dataLine:"",elapsed:0};
    $("#activity").classList.remove("collapsed");$("#actToggle").textContent="Hide details";$("#actLog").open=false;
    renderJob();refresh();poll();
  }catch(e){alert(e.message)}
}
const pairs=s=>Object.fromEntries([...s.matchAll(/(\w+)=(-?[\d.]+(?:e[+-]?\d+)?|nan)/gi)].map(m=>[m[1],parseFloat(m[2])]));
function ingest(lines){
  for(const l of lines){let m;
    if(m=l.match(/^step (\d+)\/(\d+) \(([\d.]+)s\/step\)/)){JOB.step=+m[1];JOB.total=+m[2];JOB.sps=+m[3]}
    else if(m=l.match(/^step (\d+) validation: (.*)$/)){const v=pairs(m[2]);if(!JOB.val0)JOB.val0=v;JOB.valN=v}
    else if(m=l.match(/^resumed at step (\d+)/))JOB.step=+m[1];
    else if(m=l.match(/^saved (.+?)(?: \(step \d+\))?$/))JOB.saved=m[1];
    else if(/^\d+ utterances,/.test(l))JOB.dataLine=l;
  }
}
async function poll(){
  if(polling)return;polling=true;
  try{
    while(JOB){
      const s=await api("job?since="+(JOB.since||0));
      if(s.status==="idle"){JOB=null;break}
      const was=JOB.status;
      JOB.lines.push(...s.lines);JOB.since=s.next;ingest(s.lines);
      const maxClientLogLines=12000;
      if(JOB.lines.length>maxClientLogLines){
        const drop=JOB.lines.length-maxClientLogLines;
        JOB.lines.splice(0,drop);JOB.logDropped=(JOB.logDropped||0)+drop;JOB.logNeedsSync=true;
      }
      JOB.status=s.status;JOB.elapsed=s.elapsed;JOB.code=s.code;JOB.command=s.command;JOB.label=s.label||JOB.label;
      renderJob();
      if(was==="running"&&s.status!=="running"){
        const k=JOB.kind;
        if(k&&DONE[k])await DONE[k](s.status==="done");
        renderJob();refresh();break;
      }
      if(s.status!=="running")break;
      await new Promise(r=>setTimeout(r,700));
    }
  }catch(e){if(JOB){JOB.status="failed";JOB.err=e.message;renderJob()}}
  polling=false;refresh();
}
function hint(text){
  if(/No module named|ModuleNotFoundError/.test(text))return"A required Python package is missing. In the TrainSV folder run: pip install -e .";
  if(/out of memory/i.test(text))return"The graphics card ran out of memory. Try “Phrases per step” = 1, or choose CPU in Advanced settings.";
  if(/phonemes not in the voice/.test(text))return"Some sounds in your dataset do not exist in this voice. Check the language in step 1, or use a phoneme map (Advanced settings).";
  if(/dataset styles that are not vocal modes/.test(text))return"Your dataset uses a style the voice does not have. Add it under “Vocal modes” in Advanced settings.";
  if(/non-finite loss/.test(text))return"Training became unstable. Try a lower learning rate in Advanced settings.";
  if(/resume/.test(text))return"Untick “Continue the previous run”, or keep the same models and settings as the earlier run.";
  return"";
}
function renderJob(){
  const a=$("#activity");a.hidden=!JOB;if(!JOB)return;
  const st=JOB.status,names={running:"Running",done:"Done",failed:"Failed",stopped:"Stopped"};
  const chip=$("#actChip");chip.textContent=names[st]||st;chip.className="chip "+(st==="done"?"ok":st==="failed"?"bad":st==="stopped"?"warn":"");
  $("#actTitle").textContent=JOB.label||"";
  $("#stopBtn").hidden=st!=="running";
  const bar=$("#bar"),fill=$("#barFill"),pct=JOB.total?JOB.step/JOB.total:0;
  bar.hidden=st==="failed"&&!JOB.total;
  bar.classList.toggle("indet",st==="running"&&!JOB.total);
  fill.style.width=st==="running"?(JOB.total?(pct*100).toFixed(1)+"%":""):(st==="done"?"100%":(JOB.total?(pct*100).toFixed(1)+"%":"0"));
  let now="";
  if(st==="running"){
    if(JOB.total)now=`Step ${JOB.step} of ${JOB.total} (${Math.round(pct*100)}%). About ${fmtTime((JOB.total-JOB.step)*JOB.sps)} left. Elapsed ${fmtTime(JOB.elapsed)}.`;
    else if(JOB.kind==="finetune")now=(JOB.dataLine?JOB.dataLine+". ":"")+"Getting ready: loading the voice and your data. This can take a minute the first time.";
    else now="Working… "+fmtTime(JOB.elapsed)+" so far.";
  }else if(st==="done"){now=JOB.kind==="svpk"?"SVPK diagnostic package created: "+val("svpk_output")+". This package is not installable.":JOB.kind==="sfpk"?"SFPK test package created: "+val("sfpk_output")+". Its model dependencies remain in the local models directory and must be verified in Flat.":JOB.kind==="finetune"&&JOB.saved?"Finished. Your new voice is saved at "+JOB.saved:"Finished in "+fmtTime(JOB.elapsed)+".";if(JOB.kind==="finetune"&&JOB.saved)now+=" ("+fmtTime(JOB.elapsed)+")"}
  else if(st==="stopped")now="You stopped this."+(JOB.kind==="finetune"?" Training progress is kept in the run folder. Tick “Continue the previous run” in Advanced settings to carry on.":"");
  $("#actNow").textContent=now;
  $("#actSummary").textContent=st==="running"&&JOB.total?`step ${JOB.step}/${JOB.total}`:"";
  const eb=$("#actErr");
  if(st==="failed"){
    const last=[...JOB.lines].reverse().find(l=>l.trim())||JOB.err||"The command stopped unexpectedly.";
    const h=hint(JOB.lines.join("\n"));
    eb.hidden=false;eb.innerHTML=esc(translateString("Something went wrong:"))+"\n"+esc(last)+(h?"<span>"+esc(h)+"</span>":"");
  }else eb.hidden=true;
  const mt=$("#actMetrics");
  if(JOB.valN&&JOB.kind==="finetune"){
    mt.hidden=false;
    mt.innerHTML="<tr><th>Test score</th><th>At start</th><th>Latest</th><th></th></tr>"+Object.keys(JOB.valN).map(k=>{
      const [n,h]=METRIC[k]||[k,""];return`<tr><td>${esc(n)}</td><td class="n">${fmtNum(JOB.val0?.[k])}</td><td class="n">${fmtNum(JOB.valN[k])}</td><td>${esc(h)}</td></tr>`}).join("");
  }else mt.hidden=true;
  const nx=$("#actNext");
  if(JOB.next&&st==="done"){nx.hidden=false;nx.textContent=JOB.next.text;nx.onclick=()=>{const n=JOB.next;go(n.go);if(n.to)setTimeout(()=>$(n.to).scrollIntoView({behavior:"smooth",block:"start"}),50)}}else nx.hidden=true;
  const log=$("#logPre"),atEnd=log.scrollTop+log.clientHeight>=log.scrollHeight-30;
  // Append only new lines instead of rebuilding the whole DOM string at every poll.
  if(JOB.logNeedsSync||(JOB.logRendered||0)>JOB.lines.length){
    const omitted=JOB.logDropped?("["+JOB.logDropped+" earlier log lines omitted]\n"):"";
    log.textContent=omitted+JOB.lines.join("\n");
    JOB.logRendered=JOB.lines.length;JOB.logNeedsSync=false;
  }else if((JOB.logRendered||0)<JOB.lines.length){
    const added=JOB.lines.slice(JOB.logRendered||0).join("\n");
    if(added)log.append(document.createTextNode((log.textContent?"\n":"")+added));
    JOB.logRendered=JOB.lines.length;
  }
  if(atEnd)log.scrollTop=log.scrollHeight
  if(JOB.log&&!JOB.logOpened){$("#actLog").open=true;JOB.logOpened=true}
}

/* ---------- results ---------- */
function showPackageFormatDialog(){
  const voicePath=val("e_new")||val("t_output");
  if(!voicePath){alert("Finish training and select the resulting voice.nofs file first.");return;}
  go("results");
  $("#packageFormatDialog").showModal();
}
function packageFileStem(name){
  return String(name||"").trim()
    .replace(/[<>:"\/\\|?*\x00-\x1f]/g,"_")
    .replace(/[. ]+$/g,"") || "voice";
}
function suggestSvpkOutput(){
  const voicePath=val("e_new")||val("t_output"),target=$("#svpk_output");
  if(!voicePath||!target)return;
  const current=val("svpk_output");
  const next=join(dirname(voicePath),packageFileStem(val("svpk_name"))+".svpk");
  if(!current||(SVPK_AUTO_OUTPUT&&current===SVPK_AUTO_OUTPUT)){
    target.value=next;
    SVPK_AUTO_OUTPUT=next;
  }else if(current!==SVPK_AUTO_OUTPUT){
    SVPK_AUTO_OUTPUT="";
  }
}
function suggestSfpkOutput(){
  const voicePath=val("e_new")||val("t_output"),target=$("#sfpk_output");
  if(!voicePath||!target)return;
  const current=val("sfpk_output");
  const next=join(dirname(voicePath),packageFileStem(val("sfpk_name"))+".sfpk");
  if(!current||(SFPK_AUTO_OUTPUT&&current===SFPK_AUTO_OUTPUT)){
    target.value=next;
    SFPK_AUTO_OUTPUT=next;
  }else if(current!==SFPK_AUTO_OUTPUT){
    SFPK_AUTO_OUTPUT="";
  }
}
function openSvpkDialog(){
  const voicePath=val("e_new")||val("t_output");
  if(!voicePath){alert("Select the trained voicebank file first.");return;}
  if(isInfo(voicePath)){alert("This is an info.*.nofs shared-model voicebank. The current SVPK diagnostic exporter supports only a regular single-file voice.nofs.");return;}
  if(!/\.nofs$/i.test(voicePath)){alert("The selected path is not a .nofs file. Choose the trained voice.nofs from Results.");return;}
  if(!val("svpk_name"))document.getElementById("svpk_name").value=val("v_name")||"My Singer";
  if(!val("svpk_version"))document.getElementById("svpk_version").value="101";
  suggestSvpkOutput();
  $("#packageFormError").textContent="";
  $("#svpkDialog").showModal();
}
function openSfpkDialog(){
  const voicePath=val("e_new")||val("t_output");
  $("#packageFormatDialog").close();
  $("#packageStatus").className="status warn";
  $("#packageStatus").textContent="SFPK export is experimental and depends on local model files. Verify the package in Flat after generation.";
  if(!voicePath){alert("Select the trained voicebank file first.");return;}
  if(!/\.nofs$/i.test(voicePath)){alert("Choose the trained .nofs file or an existing info.*.nofs file.");return;}
  if(!val("sfpk_name")&&!isInfo(voicePath))document.getElementById("sfpk_name").value=val("v_name")||"My Singer";
  if(!val("sfpk_vendor")&&!isInfo(voicePath))document.getElementById("sfpk_vendor").value="TrainSV user";
  if(!val("sfpk_version")){
    const m=basename(voicePath).match(/^info\.(.+)\.nofs$/i);
    if(m)document.getElementById("sfpk_version").value=m[1];
  }
  suggestSfpkOutput();
  $("#sfpkFormError").textContent="";
  $("#sfpkDialog").showModal();
}
async function loadResults(){
  const box=$("#results");
  const dir=val("e_out");if(!dir){box.innerHTML="";return}
  try{
    const r=await api("results?path="+enc(dir));
    if(!r.exists||(!r.phrases.length&&!Object.keys(r.metrics).length)){box.innerHTML='<p class="hint">No results saved there yet. Press “Compare voices”.</p>';return}
    let h="<h3>Scores</h3>";
    const labels=Object.keys(r.metrics).sort();
    if(labels.length){
      const keys=[...new Set(labels.flatMap(l=>Object.keys(r.metrics[l]).filter(k=>k!=="voice"&&k!=="gain")))];
      h+='<table class="t"><tr><th>Measure</th>'+labels.map((l,i)=>`<th>${esc(l.replace(/^\d+_/,""))}<br><small>${i===0?"original":"new"}</small></th>`).join("")+"</tr>";
      h+=keys.map(k=>{const [n,t]=METRIC[k]||[k,""];return`<tr><td>${esc(n)}${t?`<br><small>${esc(t)}</small>`:""}</td>`+labels.map(l=>`<td class="n">${fmtNum(r.metrics[l][k])}</td>`).join("")+"</tr>"}).join("");
      h+="</table>";
    }
    const diagnostics=r.timbre_diagnostics||{};
    const diagnosticLabels=Object.keys(diagnostics).sort();
    if(diagnosticLabels.length){
      const fmtDiag=(key,v)=>Number.isFinite(v)?(key==="spectral_centroid_hz"?String(Math.round(v)):v.toFixed(2)):"n/a";
      h+="<h3>Brightness & nasality clues (heuristic)</h3><p class=\"hint\">The spectral measures below compare each render to its source WAV. They are clues, not a reliable automatic nasality score.</p>";
      const measures=[
        ["spectral_centroid_hz","Spectral centroid (Hz)"],
        ["low_mid_ratio_db","Low/mid ratio (dB)"],
        ["high_mid_ratio_db","High/mid ratio (dB)"]
      ];
      h+='<table class="t"><tr><th>Measure</th>'+diagnosticLabels.map(l=>'<th>'+esc(l.replace(/^\d+_/,''))+'</th>').join('')+'</tr>';
      h+=measures.map(([key,label])=>'<tr><td>'+esc(label)+'</td>'+diagnosticLabels.map(l=>{
        const d=diagnostics[l],ref=d.reference||{},out=d.rendered||{},delta=(d.delta||{})[key];
        const ds=Number.isFinite(delta)?((delta>0?"+":"")+fmtDiag(key,delta)):"n/a";
        return '<td class="n"><small>WAV '+fmtDiag(key,ref[key])+' → '+fmtDiag(key,out[key])+'</small><br><b>Δ '+ds+'</b></td>';
      }).join('')+'</tr>').join('');
      h+="</table>";
      const phraseRows=[];
      diagnosticLabels.forEach(l=>(diagnostics[l].per_phrase||[]).forEach(p=>phraseRows.push({voice:l.replace(/^\d+_/,''),phrase:p.name||"",delta:p.delta||{}})));
      if(phraseRows.length){
        h+='<h4>Phrase-level diagnostic deltas</h4><p class="hint">The averages above can hide a local issue; this table shows the difference for each evaluated phrase.</p>';
        h+='<table class="t"><tr><th>Voice</th><th>Phrase</th><th>Centroid Δ (Hz)</th><th>Low/mid Δ (dB)</th><th>High/mid Δ (dB)</th></tr>';
        h+=phraseRows.map(p=>'<tr><td>'+esc(p.voice)+'</td><td>'+esc(p.phrase)+'</td>'+["spectral_centroid_hz","low_mid_ratio_db","high_mid_ratio_db"].map(k=>'<td class="n">'+(Number.isFinite(p.delta[k])?((p.delta[k]>0?"+":"")+fmtDiag(k,p.delta[k])):"n/a")+'</td>').join('')+'</tr>').join('');
        h+="</table>";
      }
      h+=diagnosticLabels.map(l=>'<p class="hint"><b>'+esc(l.replace(/^\d+_/,''))+': </b>'+esc(diagnostics[l].interpretation||"")+' '+esc(diagnostics[l].caveat||"")+'</p>').join('');
    }
    if(r.phrases.length){
      h+="<h3>Listen</h3>";
      h+=r.phrases.map(p=>`<div class="phrase"><h4>${esc(p.name)}</h4><div class="players">`+p.files.map(f=>{
        const nm=f.label==="recording"?"Real recording":f.label.replace(/^\d+_/,"");
        return`<figure><figcaption>${esc(nm)}</figcaption><audio controls preload="none" src="/api/media?token=${enc(TOKEN)}&path=${enc(f.path)}"></audio></figure>`}).join("")+"</div></div>").join("");
    }
    box.innerHTML=h;
  }catch(e){box.innerHTML='<p class="status bad">'+esc(e.message)+"</p>"}
}

/* ---------- start-up ---------- */
async function init(){
  buildFields();
  addStaticHelp();
  wireHelpInteractions();
  try{ENV=await api("env")}catch(e){$("#banner").hidden=false;$("#banner").textContent=e.message;return}
  const savedSettings=await api("settings").catch(()=>({}));
  restore(savedSettings);
  setUiLanguage(val("uiLanguage")||"en");
  watchTranslatedContent();
  // Upgrade legacy VOCALOID tracking-only settings to F0 cleanup without changing Pitch options.
  if(val("t_f0_profile")==="vocaloid"&&String(savedSettings.t_f0_smoothing??"0")==="0"&&!savedSettings.t_f0_octave){
    applyF0ProfileControls("vocaloid",false);
  }
  // a path the user typed earlier stays; empty or untouched ones are re-suggested
  AUTO.forEach(id=>{if(!val(id))EXTRA.manual[id]=false});
  const chips=$("#chips"),add=(t,c)=>{const s=document.createElement("span");s.className="chip "+c;s.textContent=t;chips.append(s);return s};
  const miss=Object.entries(ENV.modules).filter(([,v])=>!v).map(([k])=>k);
  const bn=$("#banner");
  if(!ENV.package){bn.hidden=false;bn.textContent="This file needs to sit inside the TrainSV folder (the one containing opensv_train). Move it there, or start it with: python trainsv_gui.py --project <the TrainSV folder>";}
  else if(miss.length){bn.hidden=false;bn.textContent="Missing Python packages: "+miss.join(", ")+". In the TrainSV folder run: pip install -e .   (use the same Python that started this page: "+ENV.executable+")"}
  add("Python "+ENV.python,"");
  const dev=add("Checking graphics card…","");
  api("device").then(d=>{if(d.ok){dev.textContent=d.kind==="cpu"?"CPU only (training will be slow)":d.name;dev.className="chip "+(d.kind==="cpu"?"warn":"ok")}else{dev.textContent="PyTorch not working";dev.className="chip bad";dev.title=d.error}}).catch(()=>{dev.textContent="Device unknown"});
  setMode();suggest();syncPreserveModelsUI();
  document.addEventListener("input",e=>{
    const t=e.target;if(!t.matches||!t.matches("[data-s]"))return;
    if(AUTO.includes(t.id))EXTRA.manual[t.id]=t.value.trim()!=="";
    if(t.id==="svpk_name")suggestSvpkOutput();
    if(t.id==="svpk_output"&&SVPK_AUTO_OUTPUT&&val("svpk_output")!==SVPK_AUTO_OUTPUT)SVPK_AUTO_OUTPUT="";
    if(t.id==="sfpk_name")suggestSfpkOutput();
    if(t.id==="sfpk_output"&&SFPK_AUTO_OUTPUT&&val("sfpk_output")!==SFPK_AUTO_OUTPUT)SFPK_AUTO_OUTPUT="";
    if(t.id==="voice"){syncPreserveModelsUI();checkVoice();}
    if(t.id==="t_preserve_non_acoustic_models")syncPreserveModelsUI();
    if(t.id==="dataset")checkDataset();
    if(["i_corpus","i_wavs","i_labs","i_wavs_names"].includes(t.id))checkSource();
    if(t.id==="t_steps")$$("input[name=preset]").forEach(r=>r.checked=false);
    if(["v_name","i_corpus","i_wavs","i_wavs_names","i_manifest"].includes(t.id))suggest();
    persist();refresh();
  });
  document.addEventListener("change",e=>{
    const t=e.target;if(!t.matches)return;
    if(t.id==="uiLanguage"){setUiLanguage(val("uiLanguage"));persist();return;}
    if(t.name==="preset"&&PRESETS[t.id]){Object.entries(PRESETS[t.id]).forEach(([k,v])=>{document.getElementById(k).value=v})}
    if(t.id==="t_f0_profile")applyF0ProfileControls(val("t_f0_profile"));
    if(t.id==="t_timbre_preset")applyTimbrePreset(val("t_timbre_preset"));
    if(t.id==="t_preserve_non_acoustic_models")syncPreserveModelsUI();
    if(t.name==="d_mode"||t.id==="i_src"){setMode();suggest();checkSource()}
    if(t.id==="i_src")suggest();
    persist();refresh();
  });
  document.addEventListener("click",e=>{
    const b=e.target.closest("button");if(!b)return;
    if(b.dataset.go)go(b.dataset.go);
    else if(b.dataset.run)startJob(b.dataset.run);
    else if(b.dataset.browse)openPicker(b.dataset.browse,b.dataset.mode,b.dataset.ext);
    else if(b.dataset.audit)runDatasetAudit();
  });
  $("#stopBtn").onclick=()=>api("stop",{}).catch(e=>alert(e.message));
  $("#actToggle").onclick=()=>{const a=$("#activity");a.classList.toggle("collapsed");$("#actToggle").textContent=a.classList.contains("collapsed")?"Show details":"Hide details"};
  $("#copyLog").onclick=()=>{navigator.clipboard?.writeText(JOB?JOB.lines.join("\n"):"")};
  $("#quitBtn").onclick=async()=>{if(!confirm("Stop the TrainSV GUI? Anything running will be stopped."))return;try{await api("quit",{})}catch(e){}document.body.innerHTML='<p style="padding:30px">The TrainSV GUI has stopped. You can close this tab.</p>'};
  $("#useTrained").onclick=async()=>{try{const r=await api("resolve-voice?path="+enc(val("t_output")));if(r.path)setVal("e_new",r.path);else alert(r.error)}catch(e){alert(e.message)}};
  $("#showResults").onclick=loadResults;
  $("#openPackageFormat").onclick=showPackageFormatDialog;
  $("#closePackageFormat").onclick=()=>$("#packageFormatDialog").close();
  $("#chooseSvpk").onclick=()=>{$("#packageFormatDialog").close();openSvpkDialog()};
  $("#chooseSfpk").onclick=openSfpkDialog;
  $("#cancelSvpk").onclick=()=>$("#svpkDialog").close();
  $("#cancelSfpk").onclick=()=>$("#sfpkDialog").close();
  $("#buildSfpk").onclick=async()=>{
    const button=$("#buildSfpk"),r=B.sfpk();
    if(r.p.length){$("#sfpkFormError").textContent=r.p.join("\n");return;}
    $("#sfpkFormError").textContent="";
    button.disabled=true;
    try{
      const target=await api("stat?path="+enc(val("sfpk_output")));
      if(target.isfile&&!confirm("The output file already exists:\n"+val("sfpk_output")+"\n\nOverwrite it?"))return;
      $("#sfpkDialog").close();
      startJob("sfpk");
    }catch(e){alert("Could not check the output path: "+e.message)}
    finally{button.disabled=false}
  };
  $("#buildSvpk").onclick=async()=>{
    const button=$("#buildSvpk"),r=B.svpk();
    if(r.p.length){$("#packageFormError").textContent=r.p.join("\n");return;}
    $("#packageFormError").textContent="";
    button.disabled=true;
    try{
      const target=await api("stat?path="+enc(val("svpk_output")));
      if(target.isfile&&!confirm("The output file already exists:\n"+val("svpk_output")+"\n\nOverwrite it?"))return;
      $("#svpkDialog").close();
      startJob("svpk");
    }catch(e){alert("Could not check the output path: "+e.message)}
    finally{button.disabled=false}
  };
  // resume hint
  const rh=()=>{const r=val("t_run");if(!r)return;api("stat?path="+enc(join(r,"state.pt"))).then(s=>{$("#resumeHint").textContent=s.exists?"A saved run was found here. Set “Steps” higher than what it already did to train further.":"No saved run here yet, so this would start fresh."}).catch(()=>{})};
  $("#t_run").addEventListener("input",rh);rh();
  checkVoice();checkDataset();checkSource();refresh();
  go((location.hash||"#voice").slice(1).replace(/[^a-z]/g,"")||"voice");
  if(!$(".page.show"))go("voice");
  // reattach to a job that is already running (page reloaded)
  try{const s=await api("job?since=0");if(s.status!=="idle"){JOB={kind:"",label:s.label,status:s.status,lines:[],next:null,since:0,logNeedsSync:true,logDropped:0,logRendered:0,step:0,total:0,sps:0,val0:null,valN:null,saved:"",dataLine:"",elapsed:s.elapsed};
    if(/^finetune/.test(s.command))JOB.kind="finetune";poll()}}catch(e){}
  loadResults();
}
init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
