from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

from .acoustic import TimedPhoneme
from .calibration import fit_input_ranges
from .data import LANGUAGE_CODES, SAMPLE_RATE, load_dataset, make_phrases, read_phoneme_map, split
from .train import MODELS, TrainConfig, Trainer, apply_gain, level_gain, mode_levels, mode_weights
from .voice import ROLES, Voice
from .sfpk import format_report, inspect_sfpk as inspect_sfpk_path

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def finetune(arguments) -> None:
    voice = Voice.load(arguments.voice)
    models = tuple(arguments.models.split(","))
    if unknown := set(models) - set(MODELS):
        raise SystemExit(f"unknown models: {', '.join(sorted(unknown))}")
    utterances = load_dataset(arguments.data, arguments.language, read_phoneme_map(arguments.phoneme_map))
    phrases = [phrase for utterance in utterances for phrase in make_phrases(utterance)]
    problems = [phrase.problem for phrase in phrases if phrase.problem]
    print(f"{len(utterances)} utterances, {len(phrases)} phrases, {sum(p.frames for p in phrases) * 0.005 / 60:.1f} min")
    if problems and set(models) & {"duration", "pitch"}:
        print(f"{len(problems)} phrases lack usable notes and train the acoustic model only; first: {problems[0]}")
    _check_phonemes(voice, phrases)
    if arguments.vocal_modes is not None:
        voice.set_vocal_modes([name for name in arguments.vocal_modes.split(",") if name])
    styles = {utterance.style for utterance in utterances} - {None}
    if unknown := styles - set(voice.acoustic.vocal_modes):
        raise SystemExit(f"dataset styles that are not vocal modes of the voice (add them with --vocal-modes): {', '.join(sorted(unknown))}")
    train, validation = split(phrases, arguments.validation, arguments.seed)
    device = torch.device(arguments.device)
    run = Path(arguments.run) if arguments.run else None
    resume = None
    if arguments.resume:
        if run is None:
            raise SystemExit("--resume continues the state.pt of a --run directory")
        if (run / "state.pt").exists():
            resume = torch.load(run / "state.pt", map_location=device)
            if (resume["models"], resume["params"]) != (list(models), arguments.params):
                raise SystemExit(f"{run / 'state.pt'} trained --models {','.join(resume['models'])} --params {resume['params']}; resume with the same")
    voice.to(device)
    if not arguments.keep_level:
        if arguments.gain is not None:
            gain, source = arguments.gain, "given"
        elif arguments.level_reference:
            gain, source = level_gain(Voice.load(arguments.level_reference).to(device), phrases, device), "reference voice"
        else:
            gain, source = level_gain(voice, phrases, device), "starting voice"
        apply_gain(phrases, gain)
        print(f"recordings scaled by {gain:.3f} ({source})")
    config = TrainConfig(
        models=models, params=arguments.params, steps=arguments.steps, learning_rate=arguments.lr, anchor=arguments.anchor,
        evaluate_every=arguments.evaluate_every, seed=arguments.seed, device=arguments.device, rap_languages=voice.files.rap_languages, batch=arguments.batch,
        consonant_weight=arguments.consonant_weight, warmup=arguments.warmup, lr_end=arguments.lr_end, log_every=arguments.log_every,
    )
    calibration = random.Random(arguments.seed).sample(phrases, min(40, len(phrases)))
    models_root = Path(arguments.output).expanduser().parent.parent / "models"
    preexisting = set(models_root.glob("*/*.dnni"))
    previous: set[Path] = set()

    def save(step=None, version=None):
        nonlocal previous
        for report in fit_input_ranges(voice, calibration, device):
            print(f"widened a {report.role} {report.node_type} input range to ±{report.limit:g} (peak {report.peak:.3g})")
        if styles and voice.files.info is not None:
            voice.files.info.mode_extras = mode_levels(voice, calibration, device)
        saved = voice.save(arguments.output, name=arguments.name, version=version or arguments.version)
        if voice.files.info is not None:
            document = json.loads(saved.read_text("utf-8"))
            current = {models_root / kind / f"{document[kind]}.dnni" for kind in ("duration", "f0", "acoustic", "vocoder")}
            in_use = _referenced_models(models_root)
            for path in previous - current - preexisting - in_use:
                path.unlink(missing_ok=True)
            previous = current
        print(f"saved {saved}" + ("" if step is None else f" (step {step})"), flush=True)

    on_final = (lambda step: save(step, arguments.final_version)) if arguments.final_version else None
    journal = on_checkpoint = None
    if run is not None:
        run.mkdir(parents=True, exist_ok=True)
        log_file = (run / "log.jsonl").open("a")

        def journal(record):
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

        def on_checkpoint(state):
            torch.save({"models": list(models), "params": arguments.params, **state}, run / "state.pt.tmp")
            os.replace(run / "state.pt.tmp", run / "state.pt")

    trainer = Trainer(voice, train, validation, config, on_best=save if arguments.save_best else None, on_final=on_final, journal=journal, on_checkpoint=on_checkpoint)
    history = trainer.run(resume)
    save()
    if arguments.history:
        Path(arguments.history).write_text(json.dumps(history, indent=2))


def _referenced_models(models_root: Path) -> set[Path]:
    referenced = set()
    for info in (models_root.parent / "databases").glob("*/info.*.nofs"):
        try:
            document = json.loads(info.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for kind in ("duration", "f0", "acoustic", "vocoder"):
            if isinstance(document.get(kind), str):
                referenced.add(models_root / kind / f"{document[kind]}.dnni")
    return referenced


def _check_phonemes(voice: Voice, phrases) -> None:
    missing = set()
    for phrase in phrases:
        for phone in phrase.phones:
            try:
                voice.acoustic.phoneme_indices([TimedPhoneme(phone.language, phone.symbol, 1)])
            except ValueError:
                missing.add(f"{phone.language}/{phone.symbol}")
    if missing:
        raise SystemExit(f"phonemes not in the voice's phone sets (map them with --phoneme-map): {', '.join(sorted(missing))}")


def import_lab(arguments) -> None:
    from .data import corpus_recordings, import_labels, lab_recordings

    if arguments.corpus:
        recordings = corpus_recordings(arguments.corpus)
    elif arguments.labs and arguments.wavs:
        recordings = lab_recordings(arguments.labs, arguments.wavs)
    else:
        raise SystemExit("give --corpus, or --labs and --wavs")
    if arguments.songs:
        by_name = {recording[0]: recording for recording in recordings}
        names = arguments.songs.split(",")
        if missing := [name for name in names if name not in by_name]:
            raise SystemExit(f"no .lab with a matching .wav for: {', '.join(missing)}")
        recordings = [by_name[name] for name in names]
    mapping = dict(pair.split("=", 1) for pair in arguments.map)
    vowels = None
    if arguments.voice:
        from .common import find_language

        phone_sets = Voice.load(arguments.voice).acoustic.phone_sets
        phone_set = phone_sets[find_language(phone_sets, arguments.language)]
        vowels = {s for s, c in zip(phone_set.symbols, phone_set.categories) if c in ("vowel", "diphthong")}
    for line in import_labels(recordings, arguments.output, mapping, vowels):
        print(line)


def convert_dict(arguments) -> None:
    from .common import find_language
    from .data import convert_dictionary, read_dictionary

    phone_sets = Voice.load(arguments.voice).acoustic.phone_sets
    phone_set = phone_sets[find_language(phone_sets, arguments.language)]
    nuclei = {s for s, c in zip(phone_set.symbols, phone_set.categories) if c in ("vowel", "diphthong")}
    count, unknown = convert_dictionary(arguments.data, arguments.output, read_dictionary(arguments.source_dict), read_dictionary(arguments.target_dict), nuclei, arguments.coda_share)
    print(f"converted {count} utterances")
    if unknown:
        print("left unconverted:", ", ".join(f"{phone}×{n}" for phone, n in sorted(unknown.items(), key=lambda item: -item[1])))


def evaluate(arguments) -> None:
    device = torch.device(arguments.device)
    output = Path(arguments.out) if arguments.out else None
    report = {}
    for index, path in enumerate(arguments.voice):
        voice = Voice.load(path).to(device)
        phrases = [phrase for utterance in load_dataset(arguments.data, arguments.language, read_phoneme_map(arguments.phoneme_map)) for phrase in make_phrases(utterance)]
        _, validation = split(phrases, arguments.validation, arguments.seed)
        gain = level_gain(voice, phrases, device)
        apply_gain(phrases, gain)
        config = TrainConfig(models=tuple(arguments.models.split(",")), device=arguments.device, rap_languages=voice.files.rap_languages)
        metrics = Trainer(voice, [], validation, config, log=lambda message: None).evaluate()
        name = voice.files.metadata.get(".name", path)
        print(f"{name} (gain {gain:.3f}): " + " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items())))
        if output is not None:
            label = f"{index}_{name}"
            report[label] = {"voice": str(path), "gain": gain, **metrics}
            for phrase in validation[:arguments.max_phrases]:
                directory = output / f"{phrase.utterance.name}_{phrase.start:.2f}"
                directory.mkdir(parents=True, exist_ok=True)
                soundfile.write(str(directory / "recording.wav"), phrase.audio() / gain, SAMPLE_RATE)
                soundfile.write(str(directory / f"{label}.wav"), _render(voice, phrase, device) / gain, SAMPLE_RATE)
    if output is not None:
        (output / "metrics.json").write_text(json.dumps(report, indent=2))
        print(f"wrote {output}")


def _render(voice: Voice, phrase, device: torch.device):
    hz, _ = phrase.frame_f0()
    with torch.no_grad():
        log_f0 = torch.from_numpy(np.log(hz).astype(np.float32)).to(device)
        frames = voice.acoustic(phrase.timed_phonemes(), log_f0, mode_weights=mode_weights(voice.acoustic, phrase.utterance.style)).frames
        return voice.vocoder(frames).samples.cpu().numpy()


def synthesize_dataset(arguments) -> None:
    from .synthetic import render_dataset

    render_dataset(Voice.load(arguments.voice), arguments.output, arguments.phrases, arguments.language, arguments.seed, arguments.notes)


def inspect(arguments) -> None:
    voice = Voice.load(arguments.voice)
    files = voice.files
    print(f"{files.metadata.get('.name', '?')} ({'info voice' if files.info else 'voice.nofs'}) vocal modes: {', '.join(files.vocal_modes) or '-'}")
    for role in ROLES:
        model = voice.models[role]
        count = sum(parameter.numel() for parameter in model.parameters())
        print(f"  {role:9s} DNNI v{files.models[role].version} {count / 1e6:6.2f}M parameters")
    for phone_set in voice.acoustic.phone_sets:
        print(f"  {phone_set.name}: {' '.join(phone_set.symbols)}")


def inspect_sfpk_command(arguments) -> None:
    """Inspect a Flat package or extracted package directory without changing it."""
    report = inspect_sfpk_path(arguments.path)
    print(format_report(report))
    if arguments.json_output:
        destination = Path(arguments.json_output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        print(f"wrote diagnostic report: {destination}")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainsv")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("finetune", help="finetune a voice on a DiffSinger-style dataset")
    command.add_argument("--voice", required=True, help="voice.nofs or info.<version>.nofs")
    command.add_argument("--data", required=True, help="directory with wavs/ and transcriptions.csv")
    command.add_argument("--output", required=True, help="voice.nofs path, or databases/<name>/ directory for info voices")
    command.add_argument("--language", required=True, type=language_name, help="default phoneme language, e.g. japanese or ja")
    command.add_argument("--models", default="acoustic", help=f"comma-separated subset of {','.join(MODELS)}")
    command.add_argument("--params", choices=("all", "embeddings"), default="all", help="train every weight, or only the per-voice speaker/style/vocal-mode vectors")
    command.add_argument("--vocal-modes", help="comma-separated vocal modes of the finetuned voice (info voices): existing ones are kept, new ones start from the default style, others are dropped")
    command.add_argument("--steps", type=int, default=2000)
    command.add_argument("--lr", type=float, help="learning rate (default 2e-5 for all, 3e-3 for embeddings)")
    command.add_argument("--warmup", type=int, default=0, help="steps of linear learning-rate warmup")
    command.add_argument("--lr-end", type=float, help="cosine-decay the learning rate to this by the last step; constant without it")
    command.add_argument("--anchor", type=float, default=1.0, help="L2 pull towards the original weights")
    command.add_argument("--batch", type=int, default=1, help="phrases (audio: windows) averaged per step")
    command.add_argument("--consonant-weight", type=float, default=1.0, help="weight of consonant frames in the training audio loss (held-out scores stay unweighted)")
    command.add_argument("--validation", type=float, default=0.1, help="phrases held out: a fraction, or a count when 1 or more")
    command.add_argument("--evaluate-every", "--val-every", type=int, default=250)
    command.add_argument("--log-every", type=int, default=25)
    command.add_argument("--phoneme-map", help="JSON object mapping dataset phonemes to the voice's phonemes")
    command.add_argument("--keep-level", action="store_true", help="do not rescale the recordings to the base voice's level")
    command.add_argument("--level-reference", help="match the recordings' level to this voice instead of the starting one; use the stock base when continuing from a finetune, so the level cannot drift")
    command.add_argument("--gain", type=float, help="scale the recordings by this factor instead of measuring it")
    command.add_argument("--device", default=DEFAULT_DEVICE, help=f"default {DEFAULT_DEVICE}")
    command.add_argument("--name", help="singer name for the finetuned voice")
    command.add_argument("--version", help="version for info voices (the info.<version>.nofs suffix)")
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--history", help="write validation metrics to this JSON file")
    command.add_argument("--run", help="run directory: log.jsonl of every logged step and validation, and state.pt saved at each validation for --resume")
    command.add_argument("--resume", action="store_true", help="continue from the --run directory's state.pt (starts afresh without one)")
    command.add_argument("--save-best", action="store_true", help="also save the voice whenever the held-out score improves")
    command.add_argument("--final-version", help="also save the last step's weights, unselected, as this version")
    command.set_defaults(handler=finetune)

    command = commands.add_parser("import-lab", help="convert HTS/vLabeler .lab alignments and their wavs into a dataset")
    command.add_argument("--corpus", help="TrainV6 corpus: a directory of <name>.wav and <name>.lab (or wav/ and lab/), or a manifest.json")
    command.add_argument("--labs", help="directory of <name>.lab files (start end phoneme, 100 ns units)")
    command.add_argument("--wavs", help="directory of the matching <name>.wav files")
    command.add_argument("--songs", help="comma-separated names to import; default every .lab with a matching .wav")
    command.add_argument("--output", required=True)
    command.add_argument("--map", nargs="*", default=[], help="phoneme replacements such as vf=cl")
    command.add_argument("--voice", help="also write ph_num word spans, using this voice's phone categories")
    command.add_argument("--language", default="japanese", type=language_name, help="phone set for --voice")
    command.set_defaults(handler=import_lab)

    command = commands.add_parser("convert-dict", help="re-spell a DiffSinger dataset with SV's dictionary for its language")
    command.add_argument("--data", required=True, help="source dataset (read only)")
    command.add_argument("--output", required=True, help="new dataset directory; its wavs/ links to the source")
    command.add_argument("--source-dict", required=True, help="the dataset's syllable dictionary, e.g. DiffSinger's opencpop-extension.txt")
    command.add_argument("--target-dict", required=True, help="SV's dictionary, e.g. clf-data/mandarin-xsampa-dict.txt")
    command.add_argument("--voice", required=True, help="voice whose phone categories mark the target nuclei")
    command.add_argument("--language", required=True, type=language_name)
    command.add_argument("--coda-share", type=float, default=0.3, help="fraction of a rhyme given to its coda (SV's duration models average about 0.3)")
    command.set_defaults(handler=convert_dict)

    command = commands.add_parser("evaluate", help="held-out metrics of one or more voices on a dataset, with renders to listen to")
    command.add_argument("--voice", required=True, action="append", help="repeat to compare several voices, e.g. the base and its finetune")
    command.add_argument("--data", required=True)
    command.add_argument("--language", required=True, type=language_name)
    command.add_argument("--models", default="acoustic")
    command.add_argument("--validation", type=float, default=0.1)
    command.add_argument("--phoneme-map")
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--device", default=DEFAULT_DEVICE, help=f"default {DEFAULT_DEVICE}")
    command.add_argument("--out", help="write metrics.json and, per held-out phrase, the recording next to each voice's render (recorded timing and pitch)")
    command.add_argument("--max-phrases", type=int, default=4, help="held-out phrases rendered to --out")
    command.set_defaults(handler=evaluate)

    command = commands.add_parser("synthesize-dataset", help="render a labelled DiffSinger-style dataset with a voice")
    command.add_argument("--voice", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--phrases", type=int, default=40)
    command.add_argument("--language", default="japanese", type=language_name)
    command.add_argument("--notes", type=int, default=8)
    command.add_argument("--seed", type=int, default=0)
    command.set_defaults(handler=synthesize_dataset)

    command = commands.add_parser("inspect", help="summarise a voice's models and phone sets")
    command.add_argument("--voice", required=True)
    command.set_defaults(handler=inspect)

    command = commands.add_parser("inspect-sfpk", help="inspect an SFPK candidate or extracted directory (read-only)")
    command.add_argument("--path", required=True, help=".sfpk candidate or extracted directory")
    command.add_argument("--json-output", help="optional path to save the diagnostic report as JSON")
    command.set_defaults(handler=inspect_sfpk_command)
    for command in commands.choices.values():
        command.add_argument("--config", action="append", default=[], metavar="FILE", help="JSON file of option values; repeat for several, later files win and command-line options override them")
    return parser


def language_name(value: str) -> str:
    return LANGUAGE_CODES.get(value, value)


def apply_configs(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    reader = argparse.ArgumentParser(add_help=False)
    reader.add_argument("--config", action="append", default=[])
    paths = reader.parse_known_args(argv)[0].config
    if not paths:
        return
    commands = parser._subparsers._group_actions[0].choices
    name = next(argument for argument in argv if argument in commands)
    command = commands[name]
    values = {}
    for path in paths:
        loaded = json.loads(Path(path).read_text("utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit(f"{path}: a config is a JSON object of option values")
        values.update({key.replace("-", "_"): value for key, value in loaded.items()})
    known = {}
    for action in command._actions:
        if action.option_strings and action.dest not in ("help", "config"):
            known[action.dest] = action
            known.update({option.lstrip("-").replace("-", "_"): action for option in action.option_strings})
    if unknown := sorted(set(values) - known.keys()):
        raise SystemExit(f"{name} has no such options: {', '.join(unknown)}")
    for key, value in values.items():
        known[key].required = False
        command.set_defaults(**{known[key].dest: value})


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = parser()
    apply_configs(root, argv)
    arguments = root.parse_args(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
