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

ROOT = Path(__file__).resolve().parent
TOKEN = secrets.token_urlsafe(16)
COMMANDS = {"import-lab", "convert-dict", "evaluate", "finetune", "synthesize-dataset", "inspect", "package-svpk", "package-sfpk"}
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
        for e in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if e.name.startswith("."):
                continue
            try:
                is_dir = e.is_dir()
            except OSError:
                continue
            if is_dir:
                entries.append({"name": e.name, "dir": True})
            elif want_files and (not exts or e.suffix.lower().lstrip(".") in exts):
                entries.append({"name": e.name, "dir": False})
            if len(entries) >= 3000:
                break
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
                    if wavs.is_dir() and not (wavs / f"{row.get('name', '')}.wav").exists():
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
        total = len(list(d.glob(f"*.{ext}")))
        for sub in subs:
            if (d / sub).is_dir():
                total += len(list((d / sub).glob(f"*.{ext}")))
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
    phrases = []
    for sub in sorted(x for x in d.iterdir() if x.is_dir()):
        wavs = sorted(sub.glob("*.wav"), key=lambda w: (w.stem != "recording", w.name))
        if wavs:
            phrases.append({"name": sub.name, "files": [{"label": w.stem, "path": str(w)} for w in wavs]})
    return {"exists": True, "metrics": metrics, "phrases": phrases}


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
    <div class="actions"><button class="ghost" data-run="score">Score the original voice on this data (optional)</button></div>
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
      <div class="field" data-id="i_wavs_names" data-kind="path" data-mode="dir" data-label="Folder of WAV files" data-hint="Use spaces between phones, e.g. <code>@_n 7_0.wav</code>. For Mandarin VOCALOID labels, TrainSV converts source phones to the selected voice inventory automatically. Only the final _0/_1 retake suffix is removed."></div>
      <div class="field" data-id="i_phone_table" data-kind="path" data-mode="file" data-ext="json" data-label="phone_table.json (optional)" data-hint="Usually not needed when a voice is selected above. If omitted, keep “Mark word boundaries using my voice” enabled so TrainSV can read the selected voice phone set."></div>
      <p class="note">Approximation: the first and last <code>sil</code> each take 10% of the audio; all middle phones share the remaining 80% equally. This creates labels from filename order, not acoustic forced alignment.</p>
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
  <fieldset>
    <legend>What should it learn?</legend>
    <label class="choice"><input type="checkbox" id="m_acoustic" data-s checked><span><b>Sound</b> (acoustic)<small>How the voice sounds. Works with any dataset. Start here.</small></span></label>
    <label class="choice"><input type="checkbox" id="m_duration" data-s><span><b>Timing</b> (duration)<small>How long each sound lasts. Needs notes in the dataset.</small></span></label>
    <label class="choice"><input type="checkbox" id="m_pitch" data-s><span><b>Pitch</b> (melody and vibrato)<small>How the singer moves between notes. Needs notes in the dataset.</small></span></label>
    <label class="choice"><input type="checkbox" id="m_vocoder" data-s><span><b>Vocoder</b> (final audio stage)<small>Optional and slower. Most people leave this off.</small></span></label>
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
      <div class="field" data-id="t_val" data-label="Recordings held back for testing" data-ph="0.1" data-hint="A fraction (0.1 = 10%) or a count (5)."></div>
      <div class="field" data-id="t_seed" data-label="Random seed" data-ph="0" data-hint="Use the same value when comparing results."></div>
      <div class="field" data-id="t_evalevery" data-label="Test every (steps)" data-ph="250"></div>
      <div class="field" data-id="t_logevery" data-label="Report every (steps)" data-ph="25"></div>
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
    <h3>导出训练后的声库</h3>
    <p class="hint">SVPK 当前仅为格式研究诊断包。全零字段版本在 Sharp 中无反应；把 Hanakuma Chifuyu 的 256 字节字段复制到自制包后，Sharp 能识别包但提示数据损坏。这强烈暗示该字段参与包专属完整性/认证校验（RSA-2048 签名只是未证实的假设）。当前生成结果不可安装或发布。SFPK 为实验性 Flat 格式导出：会将模型依赖写入所选本机 models 目录，包内仅含 config.json 与 info.nofs，暂非自包含包，必须在 Flat 中实际验证。</p>
    <div class="actions"><button class="primary" type="button" id="openPackageFormat">导出声库 / 选择打包格式</button></div>
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
  <div class="pk-head"><b>选择声库打包格式</b><div class="hint">SVPK 目前只生成诊断包，不能安装。SFPK 提供实验性导出：生成 config.json + info.nofs，并将四类 .dnni 依赖写入所选本机 models 目录；不会把模型二进制嵌入 SFPK，迁移到其他电脑时需另行安装对应模型。</div></div>
  <div class="pkg-choice-body">
    <div><button class="primary" type="button" id="chooseSvpk">使用 SVPK 格式打包</button><p class="hint">设置声库名称、供应商、版本、头像和背景后自动开始打包。</p></div>
    <div><button type="button" id="chooseSfpk">使用 SFPK 格式打包（实验）</button><p class="hint">可输入训练后的 voice.nofs 或已有 info.*.nofs；需选择 Synthesizer V 的 models 目录。</p></div>
  </div>
  <div class="pk-foot"><button type="button" class="ghost" id="closePackageFormat">取消</button></div>
</dialog>

<dialog id="svpkDialog">
  <div class="pk-head"><b>使用 SVPK 格式打包</b><div class="hint">无需手动编辑 JSON。图片请选择 PNG 文件。当前仅生成诊断包：全零字段会被 Sharp 忽略；复制 Chifuyu 字段会触发“数据损坏”。这表明字段可能参与包专属完整性/认证校验，算法和签署流程尚未确认，因此此功能不会生成可安装包。</div></div>
  <div class="pkg-form">
    <div class="field" data-id="svpk_name" data-label="声库名称" data-ph="例如：Luo Tianyi"></div>
    <div class="field" data-id="svpk_vendor" data-label="声库供应商 / 作者" data-ph="例如：个人或工作室"></div>
    <div class="field" data-id="svpk_version" data-label="声库版本（整数）" data-value="101" data-ph="101" data-hint="该数值会同时写入 package-config 的 versionString 和 versionNumber；当前版本使用整数版本号，例如 101。"></div>
    <div class="field" data-id="svpk_profile" data-kind="path" data-mode="file" data-ext="png" data-label="头像图片（profile.png）" data-hint="选择声库头像 PNG 文件；打包后将按样例名称存放在 profile.png。"></div>
    <div class="field" data-id="svpk_background" data-kind="path" data-mode="file" data-ext="png" data-label="背景图片（background.png）" data-hint="选择声库背景 PNG 文件；当前会按样例将文件条目命名为 background.png。"></div>
    <div class="field" data-id="svpk_output" data-kind="path" data-mode="save" data-ext="svpk" data-label="SVPK 保存位置" data-hint="输出的 .svpk 文件路径。默认与训练后的 voice.nofs 放在同一目录。"></div>
    <p class="status bad" id="packageFormError" role="alert"></p>
  </div>
  <div class="pk-foot"><button type="button" class="ghost" id="cancelSvpk">取消</button><button type="button" class="primary" id="buildSvpk">生成诊断包（当前不可安装）</button></div>
</dialog>


<dialog id="sfpkDialog">
  <div class="pk-head"><b>使用 SFPK 格式打包（实验版）</b><div class="hint">普通 voice.nofs 将转换为 Flat 使用的 info.nofs 描述，并把 acoustic / vocoder / duration / f0 的 .dnni 文件写入你选定的本机 models 目录。SFPK 内只放 config.json 与 info.nofs，不包含模型二进制；这是本机依赖版，暂未验证 Flat 安装和渲染。</div></div>
  <div class="pkg-form">
    <div class="field" data-id="sfpk_name" data-label="声库名称" data-ph="例如：Luo Tianyi"></div>
    <div class="field" data-id="sfpk_vendor" data-label="作者 / 供应商" data-ph="例如：个人或工作室"></div>
    <div class="field" data-id="sfpk_version" data-label="声库版本（可留空）" data-ph="101" data-hint="留空时保留来源声库已有版本；输入后会覆盖来源版本，可使用 101、106 或 106b1 等标识。"></div>
    <div class="field" data-id="sfpk_models_root" data-kind="path" data-mode="dir" data-label="Synthesizer V 的 models 目录" data-hint="选择包含 acoustic、vocoder、duration、f0 等子目录的现有 models 文件夹。新模型会按内容 MD5 命名写入相应子目录，已有同名但内容不同的文件不会被覆盖。"></div>
    <div class="field" data-id="sfpk_output" data-kind="path" data-mode="save" data-ext="sfpk" data-label="SFPK 保存位置" data-hint="生成的 SFPK 仅包含 config.json 与 info.nofs，所引用的模型仍在上方选定的本机 models 目录。"></div>
    <p class="status bad" id="sfpkFormError" role="alert"></p>
  </div>
  <div class="pk-foot"><button type="button" class="ghost" id="cancelSfpk">取消</button><button type="button" class="primary" id="buildSfpk">生成 SFPK 测试包</button></div>
</dialog>

<dialog id="picker">
  <div class="pk-head"><b id="pkTitle">Choose</b><div id="pkPath"></div></div>
  <ul id="pkList"></ul>
  <div class="pk-foot" id="pkFoot"></div>
</dialog>

<script>
const TOKEN="__TOKEN__";
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const MODELS=["acoustic","duration","pitch","vocoder"];
const MODEL_NAME={acoustic:"Sound",duration:"Timing",pitch:"Pitch",vocoder:"Vocoder"};
const AUTO=["t_output","t_run","e_out","i_output","s_output","c_output"];
const METRIC={mel:["Sound match","lower is closer"],voicing:["Voiced/unvoiced match","lower is closer"],pitch_motion:["Vibrato and motion","1.0 matches the singer"]};
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
function isInfo(p){return /^info\..*\.nofs$/i.test(basename(p||""))}
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
  "i_phone_table": "选择 phone_table.json，用于判断自定义音素是否存在以及复合音素应如何拆分。若留空，TrainSV 会尝试使用所选声库的音素表。",
  "i_labs": "选择存放 .lab 音素标注文件的文件夹。文件名需要与对应的 .wav 录音一致。",
  "i_manifest": "选择 TrainV6 的 manifest.json 文件。文件中应列出录音路径及其时间对齐的音素标签。",
  "i_output": "指定转换后的数据集保存目录。转换成功后，TrainSV 会自动将该目录设为当前数据集。",
  "i_songs": "可选：只导入指定录音。填写不带扩展名的名称，多个名称用逗号分隔；留空表示导入全部录音。",
  "t_steps": "训练总步数。数值越大，训练通常越久；更多步数不保证效果更好。可先用快速测试确认数据和环境正常，再逐步增加。",
  "v_name": "为输出声库设置显示名称。建议使用容易识别的名称；该名称也会用于自动生成部分输出路径。",
  "t_output": "指定训练后声库的保存位置。普通 voice.nofs 声库保存为文件；info.*.nofs 类型的共享模型声库会按工具规则保存到目录中。",
  "t_lr": "学习率，控制每次更新模型参数时的调整幅度。默认值取决于“调整内容”：全部权重通常为 2e-5，仅训练身份向量时通常为 3e-3。过高可能不稳定，过低可能进展缓慢。",
  "t_warmup": "预热步数。大于 0 时，学习率会在指定步数内逐渐升高；留空或为 0 时不启用预热。",
  "t_lrend": "最终学习率。填写后，训练会按余弦衰减逐步降至该值；留空时学习率保持不变（除非设置了预热）。",
  "t_anchor": "控制微调权重向原始声库权重靠拢的强度。值越大，约束通常越强、越能保留原声库特征；具体效果仍取决于数据和训练设置。",
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
  return '<span class="help-wrap"><button class="help-btn" type="button" data-help-toggle aria-label="帮助：' + esc(label) + '" aria-expanded="false" title="点击查看说明">?</button><span class="help-pop" role="tooltip" hidden>' + esc(text) + '</span></span>';
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
    const description = FIELD_HELP[id] || (d.hint ? d.hint.replace(/<[^>]*>/g, "") : "此设置会影响当前操作，请结合训练目标选择。");
    let h='<div class="field-label-row"><label for="' + id + '">' + esc(d.label) + '</label>' + helpMarkup(d.label, description) + '</div>';
    if(d.kind==="path")h+=`<div class="path"><input type="text" id="${id}" data-s placeholder="${esc(d.ph||"")}" spellcheck="false" autocomplete="off"><button type="button" class="ghost" data-browse="${id}" data-mode="${d.mode}" data-ext="${d.ext||""}">Browse…</button></div>`;
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
    else if(src==="filenames"){if(!val("i_wavs_names"))p.push("Choose the folder of WAV files.");a.push("--from-filenames","--wavs",val("i_wavs_names"));if(val("i_phone_table"))a.push("--phone-table",val("i_phone_table"))}
    else{if(!val("i_wavs"))p.push("Choose the folder of .wav files.");if(!val("i_labs"))p.push("Choose the folder of .lab files.");a.push("--labs",val("i_labs"),"--wavs",val("i_wavs"))}
    if(!val("i_output"))p.push("Choose where to save the dataset.");a.push("--output",val("i_output"));
    if(val("i_songs"))a.push("--songs",val("i_songs"));
    if(chk("i_words")){if(!val("voice"))p.push("Choose a voice (step 1), or untick “Mark word boundaries”.");a.push("--voice",val("voice"),"--language",lang())}
    const pairs=val("i_map").split(/[\s,]+/).filter(x=>x.includes("="));if(pairs.length)a.push("--map",...pairs);
    return{cmd:"import-lab",label:"Converting recordings to a dataset",args:a,p}},
  finetune(){const p=[],a=[];
    if(!val("voice"))p.push("Choose a voice (step 1).");dsNeed(p,"Choose a dataset (step 2).");
    if(!val("v_name"))p.push("Name your new voice.");if(!val("t_output"))p.push("Choose where to save the new voice.");
    const models=MODELS.filter(x=>chk("m_"+x));if(!models.length)p.push("Tick at least one thing to learn.");
    a.push("--voice",val("voice"),"--data",val("dataset"),"--output",val("t_output"),"--language",lang(),"--models",models.join(","),"--name",val("v_name"));
    addOpt(a,p,"--steps","t_steps",true);
    if(val("t_params")&&val("t_params")!=="all")a.push("--params",val("t_params"));
    [["--lr","t_lr"],["--warmup","t_warmup"],["--lr-end","t_lrend"],["--anchor","t_anchor"],["--batch","t_batch"],["--consonant-weight","t_cons"],["--validation","t_val"],["--evaluate-every","t_evalevery"],["--log-every","t_logevery"],["--seed","t_seed"],["--gain","t_gain"]].forEach(([f,i])=>addOpt(a,p,f,i));
    [["--device","t_device"],["--phoneme-map","t_pmap"],["--vocal-modes","t_modes"],["--version","t_version"],["--final-version","t_final"],["--level-reference","t_levelref"],["--history","t_history"],["--run","t_run"]].forEach(([f,i])=>{if(val(i))a.push(f,val(i))});
    if(chk("t_keep"))a.push("--keep-level");if(chk("t_savebest"))a.push("--save-best");if(chk("t_resume"))a.push("--resume");
    return{cmd:"finetune",label:"Training",args:a,p}},
  evaluate(){const p=[],a=[];
    if(!val("voice"))p.push("Choose the original voice (step 1).");if(!val("e_new"))p.push("Choose your new voice.");
    dsNeed(p,"Choose a dataset (step 2).");if(!val("e_out"))p.push("Choose where to save the comparison.");
    a.push("--voice",val("voice"),"--voice",val("e_new"),"--data",val("dataset"),"--language",lang(),"--models",trainedModels().join(","));
    addOpt(a,p,"--validation","t_val");addOpt(a,p,"--seed","t_seed");
    if(val("t_pmap"))a.push("--phoneme-map",val("t_pmap"));
    a.push("--out",val("e_out"));addOpt(a,p,"--max-phrases","e_max");
    return{cmd:"evaluate",label:"Comparing voices",args:a,p}},
  svpk(){const p=[],a=[],voicePath=val("e_new")||val("t_output"),version=val("svpk_version");
    if(!voicePath)p.push("先选择训练后的 voice.nofs。");
    else if(isInfo(voicePath))p.push("当前结果是 info.*.nofs 共享模型声库；这个 SVPK 打包器只支持普通单文件 voice.nofs。");
    else if(!/\.nofs$/i.test(voicePath))p.push("请选择训练后的 .nofs 文件，而不是目录或其他文件。");
    if(!val("svpk_name"))p.push("请填写声库名称。");
    if(!val("svpk_vendor"))p.push("请填写声库供应商 / 作者。");
    if(!/^\d+$/.test(version)||!Number.isSafeInteger(Number(version)))p.push("声库版本必须是安全范围内的非负整数，例如 101。");
    if(!val("svpk_profile"))p.push("请选择 profile.png 头像图片。");
    if(!val("svpk_background"))p.push("请选择背景 PNG 图片。");
    let output=val("svpk_output");
    if(!output)p.push("请指定 SVPK 保存位置。");
    else if(!/\.svpk$/i.test(output)){output += ".svpk";const el=document.getElementById("svpk_output");if(el)el.value=output;}
    a.push("--voice",voicePath,"--profile",val("svpk_profile"),"--background",val("svpk_background"),
      "--output",output,"--name",val("svpk_name"),"--vendor",val("svpk_vendor"),
      "--version-string",version,"--version-number",String(Number(version)));
    return{cmd:"package-svpk",label:"打包 SVPK 声库",args:a,p}
  },
  sfpk(){const p=[],a=[],voicePath=val("e_new")||val("t_output"),version=val("sfpk_version");
    if(!voicePath)p.push("先选择训练后的声库文件。");
    else if(!/\.nofs$/i.test(voicePath))p.push("请选择 .nofs 文件，而不是目录或其他文件。");
    if(!val("sfpk_name"))p.push("请填写声库名称。");
    if(!val("sfpk_vendor"))p.push("请填写作者 / 供应商。");
    
    if(!val("sfpk_models_root"))p.push("请选择 Synthesizer V 的 models 目录。");
    let output=val("sfpk_output");
    if(!output)p.push("请指定 SFPK 保存位置。");
    else if(!/\.sfpk$/i.test(output)){output += ".sfpk";const el=document.getElementById("sfpk_output");if(el)el.value=output;}
    a.push("--voice",voicePath,"--models-root",val("sfpk_models_root"),"--output",output,
      "--name",val("sfpk_name"),"--vendor",val("sfpk_vendor"));
    if(version)a.push("--version",version);
    return{cmd:"package-sfpk",label:"打包 SFPK 测试包",args:a,p}
  },
  synth(){const p=[];if(!val("voice"))p.push("Choose a voice (step 1).");if(!val("s_output"))p.push("Choose where to save the data.");
    const a=["--voice",val("voice"),"--output",val("s_output"),"--language",lang()];
    addOpt(a,p,"--phrases","s_phrases");addOpt(a,p,"--notes","s_notes");
    return{cmd:"synthesize-dataset",label:"Making practice data",args:a,p}},
  convert(){const p=[];
    [["c_data","Choose the source dataset."],["c_output","Choose where to save the converted dataset."],["c_src","Choose the source dictionary."],["c_tgt","Choose the Synthesizer V dictionary."],["voice","Choose a voice (step 1)."]].forEach(([i,m])=>{if(!val(i))p.push(m)});
    const a=["--data",val("c_data"),"--output",val("c_output"),"--source-dict",val("c_src"),"--target-dict",val("c_tgt"),"--voice",val("voice"),"--language",lang()];
    addOpt(a,p,"--coda-share","c_coda");
    return{cmd:"convert-dict",label:"Converting the dictionary",args:a,p}}
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
  const running=JOB&&JOB.status==="running";
  $$("[data-run]").forEach(b=>{
    const k=b.dataset.run,r=B[k]();
    b.disabled=r.p.length>0||running;
    $("#needs-"+k).innerHTML=r.p.length?"Before you can continue:<ul>"+r.p.map(x=>"<li>"+esc(x)+"</li>").join("")+"</ul>":"";
    $("#cmd-"+k).textContent="trainsv "+r.cmd+" "+r.args.map(quote).join(" ");
  });
  const warn=$("#trainWarn"),m=MODELS.filter(x=>chk("m_"+x));
  const wantsNotes=m.includes("duration")||m.includes("pitch");
  if(wantsNotes&&DS&&DS.rows&&DS.notes<DS.rows){warn.hidden=false;warn.textContent=DS.notes?`${DS.notes} of ${DS.rows} recordings have notes. The others will train Sound only.`:"Your dataset has no notes yet, so Timing and Pitch cannot learn anything. Add notes first (for example with GAME), or untick them to train Sound only."}
  else if(wantsNotes&&!DS){warn.hidden=false;warn.textContent="Timing and Pitch need a dataset with notes. Check your dataset in step 2."}
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

/* ---------- checks: voice, dataset, recordings ---------- */
function say(id,cls,text){const e=document.getElementById(id);if(e){e.className="status "+cls;e.textContent=text}}
let tVoice,tData,tSrc;
function checkVoice(){
  clearTimeout(tVoice);tVoice=setTimeout(async()=>{
    const v=val("voice");VOICE_OK=false;
    if(!v){say("voice_st","","");refresh();return}
    try{const s=await api("stat?path="+enc(v));
      if(s.isfile){VOICE_OK=true;say("voice_st","ok",isInfo(v)?"✓ Found. This is an info voice (shared models); the result is saved into a folder.":"✓ Found. This is a single-file voice.")}
      else say("voice_st","bad",s.isdir?"That is a folder. Choose the .nofs file inside it.":"File not found.")}catch(e){say("voice_st","bad",e.message)}
    suggest();refresh();
  },300);
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
  PK={id,mode,ext,dir:"",name:mode==="save"?(cur&&/\.\w+$/.test(basename(cur))?basename(cur):(ext==="json"?"history.json":ext==="svpk"?"voice.svpk":"voice.nofs")):""};
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
  svpk(ok){const el=$("#packageStatus");if(el){el.className="status "+(ok?"warn":"bad");el.textContent=ok?"SVPK 诊断包已生成："+val("svpk_output")+"。全零字段版本在 Sharp 中无响应；复制 Chifuyu 字段后提示数据损坏。该包不可安装或发布。":"SVPK 诊断包生成失败；请查看下方任务日志中的具体错误。"}},
  sfpk(ok){const el=$("#packageStatus");if(el){el.className="status "+(ok?"warn":"bad");el.textContent=ok?"SFPK 测试包已生成："+val("sfpk_output")+"。模型依赖已写入本机："+val("sfpk_models_root")+"。请在 Synthesizer V Studio Flat 中安装/导入、选择声库并测试渲染；目前尚未验证运行时兼容性。":"SFPK 生成失败；请查看任务日志中的具体错误。"}}
};
async function startJob(kind){
  const r=B[kind]();if(r.p.length)return;
  try{
    await api("run",{command:r.cmd,args:r.args,label:r.label});
    JOB={kind,label:r.label,status:"running",lines:[],next:null,since:0,log:!!r.log,step:0,total:0,sps:0,val0:null,valN:null,saved:"",dataLine:"",elapsed:0};
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
  }else if(st==="done"){now=JOB.kind==="svpk"?"SVPK 诊断包已生成："+val("svpk_output")+"。全零字段无响应、移植 Chifuyu 字段则提示数据损坏；文件不可安装。":JOB.kind==="sfpk"?"SFPK 测试包已生成："+val("sfpk_output")+"。模型依赖位于本机 models 目录，需在 Flat 中验证。":JOB.kind==="finetune"&&JOB.saved?"Finished. Your new voice is saved at "+JOB.saved:"Finished in "+fmtTime(JOB.elapsed)+".";if(JOB.kind==="finetune"&&JOB.saved)now+=" ("+fmtTime(JOB.elapsed)+")"}
  else if(st==="stopped")now="You stopped this."+(JOB.kind==="finetune"?" Training progress is kept in the run folder. Tick “Continue the previous run” in Advanced settings to carry on.":"");
  $("#actNow").textContent=now;
  $("#actSummary").textContent=st==="running"&&JOB.total?`step ${JOB.step}/${JOB.total}`:"";
  const eb=$("#actErr");
  if(st==="failed"){
    const last=[...JOB.lines].reverse().find(l=>l.trim())||JOB.err||"The command stopped unexpectedly.";
    const h=hint(JOB.lines.join("\n"));
    eb.hidden=false;eb.innerHTML="Something went wrong:\n"+esc(last)+(h?"<span>"+esc(h)+"</span>":"");
  }else eb.hidden=true;
  const mt=$("#actMetrics");
  if(JOB.valN&&JOB.kind==="finetune"){
    mt.hidden=false;
    mt.innerHTML="<tr><th>Test score</th><th>At start</th><th>Latest</th><th></th></tr>"+Object.keys(JOB.valN).map(k=>{
      const [n,h]=METRIC[k]||[k,""];return`<tr><td>${esc(n)}</td><td class="n">${fmtNum(JOB.val0?.[k])}</td><td class="n">${fmtNum(JOB.valN[k])}</td><td>${esc(h)}</td></tr>`}).join("");
  }else mt.hidden=true;
  const nx=$("#actNext");
  if(JOB.next&&st==="done"){nx.hidden=false;nx.textContent=JOB.next.text;nx.onclick=()=>{const n=JOB.next;go(n.go);if(n.to)setTimeout(()=>$(n.to).scrollIntoView({behavior:"smooth",block:"start"}),50)}}else nx.hidden=true;
  const log=$("#logPre"),text=JOB.lines.join("\n"),atEnd=log.scrollTop+log.clientHeight>=log.scrollHeight-30;
  if(log.textContent!==text){log.textContent=text;if(atEnd)log.scrollTop=log.scrollHeight}
  if(JOB.log&&!JOB.logOpened){$("#actLog").open=true;JOB.logOpened=true}
}

/* ---------- results ---------- */
function showPackageFormatDialog(){
  const voicePath=val("e_new")||val("t_output");
  if(!voicePath){alert("请先完成训练，并选择训练后的 voice.nofs。");return;}
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
  if(!voicePath){alert("请先选择训练后的声库文件。");return;}
  if(isInfo(voicePath)){alert("当前结果是 info.*.nofs 共享模型声库。当前 SVPK 打包器仅支持普通单文件 voice.nofs；共享模型声库需要另一套打包方式。");return;}
  if(!/\.nofs$/i.test(voicePath)){alert("当前路径不是 .nofs 文件，请先在 Results 中选择训练后的 voice.nofs。");return;}
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
  $("#packageStatus").textContent="SFPK 是实验性本机依赖版；生成后必须在 Flat 中验证。";
  if(!voicePath){alert("请先选择训练后的声库文件。");return;}
  if(!/\.nofs$/i.test(voicePath)){alert("请选择训练后的 .nofs 文件或已有 info.*.nofs 文件。");return;}
  if(!val("sfpk_name"))document.getElementById("sfpk_name").value=val("v_name")||"My Singer";
  if(!val("sfpk_vendor"))document.getElementById("sfpk_vendor").value="TrainSV user";
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
  restore(await api("settings").catch(()=>({})));
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
  setMode();suggest();
  document.addEventListener("input",e=>{
    const t=e.target;if(!t.matches||!t.matches("[data-s]"))return;
    if(AUTO.includes(t.id))EXTRA.manual[t.id]=t.value.trim()!=="";
    if(t.id==="svpk_name")suggestSvpkOutput();
    if(t.id==="svpk_output"&&SVPK_AUTO_OUTPUT&&val("svpk_output")!==SVPK_AUTO_OUTPUT)SVPK_AUTO_OUTPUT="";
    if(t.id==="sfpk_name")suggestSfpkOutput();
    if(t.id==="sfpk_output"&&SFPK_AUTO_OUTPUT&&val("sfpk_output")!==SFPK_AUTO_OUTPUT)SFPK_AUTO_OUTPUT="";
    if(t.id==="voice")checkVoice();
    if(t.id==="dataset")checkDataset();
    if(["i_corpus","i_wavs","i_labs","i_wavs_names"].includes(t.id))checkSource();
    if(t.id==="t_steps")$$("input[name=preset]").forEach(r=>r.checked=false);
    if(["v_name","i_corpus","i_wavs","i_wavs_names","i_manifest"].includes(t.id))suggest();
    persist();refresh();
  });
  document.addEventListener("change",e=>{
    const t=e.target;if(!t.matches)return;
    if(t.name==="preset"&&PRESETS[t.id]){Object.entries(PRESETS[t.id]).forEach(([k,v])=>{document.getElementById(k).value=v})}
    if(t.name==="d_mode"||t.id==="i_src"){setMode();suggest();checkSource()}
    if(t.id==="i_src")suggest();
    persist();refresh();
  });
  document.addEventListener("click",e=>{
    const b=e.target.closest("button");if(!b)return;
    if(b.dataset.go)go(b.dataset.go);
    else if(b.dataset.run)startJob(b.dataset.run);
    else if(b.dataset.browse)openPicker(b.dataset.browse,b.dataset.mode,b.dataset.ext);
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
      if(target.isfile&&!confirm("输出文件已存在：\n"+val("sfpk_output")+"\n\n是否覆盖它？"))return;
      $("#sfpkDialog").close();
      startJob("sfpk");
    }catch(e){alert("无法检查输出路径："+e.message)}
    finally{button.disabled=false}
  };
  $("#buildSvpk").onclick=async()=>{
    const button=$("#buildSvpk"),r=B.svpk();
    if(r.p.length){$("#packageFormError").textContent=r.p.join("\n");return;}
    $("#packageFormError").textContent="";
    button.disabled=true;
    try{
      const target=await api("stat?path="+enc(val("svpk_output")));
      if(target.isfile&&!confirm("输出文件已存在：\n"+val("svpk_output")+"\n\n是否覆盖它？"))return;
      $("#svpkDialog").close();
      startJob("svpk");
    }catch(e){alert("无法检查输出路径："+e.message)}
    finally{button.disabled=false}
  };
  // resume hint
  const rh=()=>{const r=val("t_run");if(!r)return;api("stat?path="+enc(join(r,"state.pt"))).then(s=>{$("#resumeHint").textContent=s.exists?"A saved run was found here. Set “Steps” higher than what it already did to train further.":"No saved run here yet, so this would start fresh."}).catch(()=>{})};
  $("#t_run").addEventListener("input",rh);rh();
  checkVoice();checkDataset();checkSource();refresh();
  go((location.hash||"#voice").slice(1).replace(/[^a-z]/g,"")||"voice");
  if(!$(".page.show"))go("voice");
  // reattach to a job that is already running (page reloaded)
  try{const s=await api("job?since=0");if(s.status!=="idle"){JOB={kind:"",label:s.label,status:s.status,lines:[],next:null,since:0,step:0,total:0,sps:0,val0:null,valN:null,saved:"",dataLine:"",elapsed:s.elapsed};
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
