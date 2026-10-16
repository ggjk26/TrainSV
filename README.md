# TrainSV

[简体中文](README.zh-CN.md) | [English](README.md)

A Python toolkit that finetunes the four neural models of a SynthV AI voice on a recorded corpus. To sum it up it's a SynthV devkit

Prerequisites: Python 3.10+, PyTorch, and a Synthesizer V voice (`voice.nofs` or `info.<version>.nofs`) to start from. Training uses a CUDA GPU when one is present, then Apple's `mps`, then the CPU; `--device` overrides it.

| Role | DNNI architecture | Trained against |
| --- | --- | --- |
| duration | `gen2a` | each phoneme's share of its note interval |
| pitch | `f0model-dds` | the recording's F0 (voiced frames inside notes) |
| acoustic | `dds` | the recording itself, through the differentiable vocoder (multi-resolution STFT + log-mel) |
| vocoder | `nhv` | the same audio losses (optional |

## Install

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[test]'
```

This installs the `trainsv` command (`python -m opensv_train` is the same). The generic install uses the PyTorch build available from your configured package index; it does not automatically detect whether your computer has an NVIDIA GPU.

### NVIDIA GPU (CUDA)

If `torch.__version__` shows a `+cpu` suffix, that PyTorch build cannot use CUDA. On Windows with an NVIDIA GPU, first install TrainSV as above, then replace PyTorch with the official CUDA wheel in the *same* virtual environment. This example uses the official CUDA 13.0 wheel index for PyTorch 2.14.1:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```

On Linux, use `.venv/bin/python` in place of `.venv\Scripts\python.exe`. Choose a CUDA wheel compatible with your NVIDIA driver from the [official PyTorch installer](https://pytorch.org/get-started/locally/) if CUDA 13.0 is not supported by your system. Installing the CUDA-enabled PyTorch wheel normally does not require installing the full CUDA Toolkit separately; the NVIDIA driver must still be installed and compatible.

Verify the build:

```sh
python -c "import torch; print(torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

A version containing `+cu...` and a non-`None` CUDA build identify a CUDA-enabled PyTorch package. `torch.cuda.is_available()` must return `True` for training to use the GPU.

## Usage

```sh
trainsv <command> [options]
```

Run `trainsv <command> --help` for all options. The pipeline is, in order:

```sh
trainsv import-lab --corpus <corpus> --output <dataset> --voice <voice>
trainsv evaluate --voice <voice> --data <dataset> --language ja
trainsv finetune --voice <voice> --data <dataset> --language ja --output <out-voice> --run <run-directory>
trainsv evaluate --voice <voice> --voice <out-voice> --data <dataset> --language ja --out <eval-directory>
```

For example:

```sh
trainsv finetune \
  --voice /path/to/voice.nofs \
  --data my-dataset --language ja \
  --models acoustic,duration,pitch \
  --output out/voice.nofs --name "My Singer" --run runs/my-singer
```

`--language` takes the voice's phone set name (`japanese`, `mandarin`, `english`, `cantonese`, `spanish`) or its code (`ja`, `zh`, `en`, `yue`, `es`).

## Configs

Every command takes `--config file.json` (repeatable). A config is a JSON object of option values, keyed by option name as written on the command line, with `-` or `_`; later files override earlier ones and command-line options override all of them. Required options such as `--voice` and `--data` can come from a config too.

```json
{"steps": 6000, "lr": 2e-5, "warmup": 100, "lr-end": 2e-6, "models": "acoustic,duration,pitch", "consonant-weight": 4}
```

## Corpus

`import-lab` turns a corpus into a dataset. A corpus is either a directory of `<name>.wav` recordings with `<name>.lab` phoneme labels (also accepted as `wav/` and `lab/` subdirectories), or a TrainV6 `manifest.json`:

```json
{
  "recordings": [
    {"id": "song_000", "wav": "/path/song_000.wav", "split": "train",
     "labels": [{"start_100ns": 0, "end_100ns": 5000000, "phoneme": "sil"}]}
  ]
}
```

`--labs <dir> --wavs <dir>` names the two directories separately instead. Every `.lab` with a matching `.wav` is imported; `--songs a,b` picks some. Labels are HTS/vLabeler alignments (`start end phoneme` in 100 ns units). `pau` and `sil` become silence, `br` breath, and `--map vf=cl` renames others. A manifest's `split` is ignored: `finetune --validation` holds out phrases instead.

If your WAV filenames already contain space-separated phones, you can generate approximate labels directly:

```bash
trainsv import-lab --from-filenames --wavs /path/to/wavs --voice /path/to/voice.nofs --output /path/to/dataset --language mandarin
```

The generator preserves left-to-right order, strips only a terminal `_0` or `_1` retake suffix, and always adds leading and trailing `sil` without deduplicating an internal silence. Each boundary silence receives 10% of the WAV duration; middle phones share the remaining 80% equally. Generated LAB files are saved under the output dataset's `lab/` directory. This is a filename-driven equal-duration approximation, **not acoustic forced alignment**; inspect the boundaries before training.

For `--language mandarin`, `--source-phoneme-system auto` (the default) converts common VOCALOID Chinese phone spellings to the target SynthV Mandarin inventory, supporting both the simplified labels introduced in SynthV 2.3.0 Beta 1 and legacy X-SAMPA-style labels. It prefers phones present in the selected voice. Usually, pass `--voice <voice>` to provide that target inventory; a separate `phone_table.json` is not required. WAVs that cannot be mapped are skipped and recorded with their error in `<output>/unmatched_phones.csv`, while other WAVs continue. Use `--source-phoneme-system none` to disable conversion or set it explicitly to `vocaloid-mandarin`.

X-SAMPA `@\`` and `3\`` are rhoticized vowels. If the complete symbol exists in the target phone table, it is preserved. Otherwise, the generator tries to approximate it as the base vowel plus a rhotic phone available in the table (preferring `r\\`, then `ɹ`, `ɻ`, `r`, etc.). If no compatible phone exists, it raises a specific error rather than silently dropping rhoticity.

Labels carry no notes, so an imported dataset trains the acoustic model. With `--voice <voice>` the importer also writes `ph_num` word spans, which lets stuff like OpenVPI's [GAME](https://github.com/openvpi/GAME) add notes for duration and pitch training.

## Dataset

A dataset is a DiffSinger-style directory:

```
my-dataset/
  wavs/<name>.wav          any sample rate, resampled to 48 kHz
  transcriptions.csv       name, ph_seq, ph_dur[, ph_num, note_seq, note_dur, note_slur, style]
```

`ph_seq` uses the voice's phonemes (`trainsv inspect --voice …` lists them). `SP` and `AP` mean silence and breath respectively.. A `lang/ph` prefix (`ja/a`, `zh/…`, `en/…`) sets the language per phoneme; `--phoneme-map map.json` renames dataset phonemes. `ph_dur` alone is enough for acoustic training. Duration and pitch training also need `ph_num` (DiffSinger words: each starts at a vowel, consonants belong to the previous word), `note_seq`/`note_dur` and, ideally, `note_slur`. The optional `style` column names the vocal mode a recording was sung in; empty means the default style.

Notes with cents (as GAME writes them) are shifted by each song's overall tuning offset before rounding to semitones, and the pitch targets move with them, so a recording tuned away from A440 still trains in-key. F0 is extracted with Praat and cached in `my-dataset/.opensv-cache/`.

Datasets spelled with another syllable dictionary convert with `convert-dict`: `trainsv convert-dict --data src --output dst --source-dict opencpop-extension.txt --target-dict clf-data/mandarin-xsampa-dict.txt --voice <voice> --language zh`. The source stays untouched and `wavs/` is linked.

## Training

`finetune` trains the models named by `--models` (default `acoustic`) for `--steps` steps (default 2000). Utterances are split into phrases at silences of 0.2 s or more, with 0.1 s of silence context, the way OpenSV's renderer builds phrases.

`--params all` (default) trains every weight of the selected models with an L2 pull (`--anchor`) towards the original weights; `--params embeddings` trains only the per-voice speaker, style and vocal-mode vectors. The learning rate (`--lr`) defaults to 2e-5 for all weights and 3e-3 for embeddings, and stays constant unless `--warmup` ramps it in over that many steps or `--lr-end` cosine-decays it to that value by the last step. `--batch` averages that many phrases (audio: windows) per step.

Held-out phrases (`--validation`: a fraction, or a count when 1 or more) are scored every `--val-every` steps (also spelled `--evaluate-every`, default 250). Each model keeps its own best checkpoint (audio models by log-mel distance plus voicing), and those are what is saved. `--save-best` also saves on every improvement; `--final-version` additionally saves the last step's weights as another version.

`--run <directory>` keeps a run directory: `log.jsonl` holds every logged step (`--log-every`, default 25) with its seconds per step and learning rate, and every validation, and `state.pt` is saved at each validation. `--resume` continues from `state.pt` with the same `--models` and `--params`; raise `--steps` to train further. `--history` writes the validation metrics alone as one JSON file.

The audio loss ignores everything 50 dB below each target's peak, so a recording's noise floor is not learned. The pitch loss matches the melody (below 3 Hz) frame by frame but vibrato only by its spectrum, so the model keeps its expression; `pitch_motion` reports its fast motion relative to the singer's (1.0 matches). `--consonant-weight` scales consonant frames in the training audio loss (held-out scores stay unweighted). 

Recordings are scaled to the starting voice's output level, measured from total energy. When continuing from a finetune, pass the stock base as `--level-reference` (or pin `--gain`) so the level cannot ratchet up across runs. `--keep-level` leaves them as recorded.

Vocal modes: `--vocal-modes Solid,High` sets an info voice's modes, adding new ones from the default style. Under `--params all` they share the small weight learning rate and barely leave their starting point, so train them with `--params embeddings` first, then continue from that voice with `--params all`. A trained mode's `extra` field is rewritten as its output level relative to the default style (the natural log of the RMS ratio), which is what the stock values track.

Judge a voice by `evaluate`, not by the training loss. It scores voices on the held-out phrases (the same split as `finetune` with the same `--validation` and `--seed`). Before training, it picks the closest base. After it, pass the base and the finetune as two `--voice`s; `--out` writes `metrics.json` and, for `--max-phrases` held-out phrases (default 4), the recording next to each voice's render with the recorded timing and pitch, all at the recording's level.

Other commands:

- `inspect --voice <voice>`: list a voice's models, parameter counts, vocal modes and phone sets
- `synthesize-dataset --voice <voice> --output <dataset>`: render a labelled dataset with any voice. Finetuning one voice towards another voice's renders is a quick end-to-end check that needs no recordings

## Voice output

A `voice.nofs` voice is written as a `voice.nofs` with its model entries patched in place. An `info.<version>.nofs` voice (shared models) is written to the `databases/<name>/` directory given by `--output`, with changed models added to the sibling `models/<kind>/<md5>.dnni` tree. `--name` sets the singer name and `--version` the info voice's version.

## WebUI packaging (experimental SVPK)

After “Compare voices” finishes in the WebUI Results stage, a format-choice dialog appears. Choose “使用 SVPK 格式打包” to enter the singer name, vendor/author, integer version, avatar PNG and background PNG. The WebUI creates the metadata JSON internally; you do not need to author or import it. This SVPK packager currently accepts a regular single-file `voice.nofs`, not a shared-model `info.*.nofs`.

SFPK export is now an **experimental local-dependency build**. For a regular `voice.nofs`, `trainsv package-sfpk` extracts the four DNNI model payloads to a selected existing Synthesizer V `models` directory using content MD5 IDs, derives the 32-float style embeddings for an `info.nofs` JSON descriptor, and creates an SFPK ZIP with `config.json` and `info.nofs` (the two entries found in the inspected sample). The SFPK does **not** embed model binaries, so the referenced model files must exist in the target installation. Runtime import, singer selection and rendering in Synthesizer V Studio Flat are not yet verified; do not rely on the file until that manual test passes.

The CLI entry point is also available:

```powershell
trainsv package-svpk `
  --voice "out\voice.nofs" `
  --profile "assets\profile.png" `
  --background "assets\background.png" `
  --output "out\MySinger.svpk" `
  --name "My Singer" `
  --vendor "Your name or studio" `
  --version-string 101 `
  --version-number 101
```

**Known limitation: the current output is not installable.** Testing confirmed that Sharp ignores a package with a zero-filled 256-byte field. Copying the 256-byte field from Hanakuma Chifuyu makes Sharp recognize the custom package but then report corrupted data. Along with the differing high-entropy fields in the Teto and Chifuyu samples, this strongly suggests package-bound integrity/authentication data. The field length is consistent with an RSA-2048-sized signature, but the algorithm, signed payload, and official signing process are unconfirmed. The current output is diagnostic only; do not treat it as an installable or distributable voicebank, and do not assume transplanting another package’s field can work. The CLI `--license` argument is optional. If omitted, the packager includes a clearly marked placeholder `license.txt` with `!language en-us` and `!language zh-cn` sections (matching the directive-based format observed in the sample). It does not invent license terms; replace it with actual terms you have the right to distribute before publication.

The archive entries and order match the inspected `KASANE_TETO_AI_104.svpk` sample: `voice.nofs`, `license.txt`, `profile.png`, `background.png`, `package-config`.

### Experimental Flat SFPK export

```powershell
trainsv package-sfpk `
  --voice "out\voice.nofs" `
  --models-root "C:\path\to\Synthesizer V Studio\models" `
  --output "out\MySinger.sfpk" `
  --name "My Singer" `
  --vendor "Your name or studio" `
  --version 101
```

You can also provide an extracted Flat `info.nofs` JSON file as `--voice`; its model ID dependencies must already exist under the selected models root. For a regular binary `voice.nofs`, TrainSV writes acoustic, vocoder, duration, and F0 model files into their corresponding subfolders, then refers to their content MD5 IDs from `info.nofs`. The SFPK itself contains only `config.json` and `info.nofs`; it is not a portable, self-contained package. Choose the actual `models` folder used by Synthesizer V Studio Flat, and make a backup before testing. This format writer follows one inspected SFPK sample and remains experimental until it imports, selects and renders correctly in Flat.

Each 16-bit weight matrix stores a word that OpenSV reads as the original engine's input-quantization setting. Every stock matrix's inputs fit `|x| < 2^(15 - word)`, so before saving, the toolkit measures inputs on the training data and widens that range where finetuning outgrew it.

## Verification

```sh
cmake -S parity -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target OpenSVParity
.venv/bin/python -m pytest
```

The tests use installed voices (`OPENSV_TEST_VOICES`, separated by `:`, overrides the defaults in `tests/conftest.py`).

Now supported Vocaloid-styled X-SAMPA dict. But something wrong with F0.

## Layout

- `opensv_train/`: the CLI, dataset, model, training and export code
- `parity/`: the vendored OpenSV engine and the harness the tests compare against
- `tests/`: parity and training tests
