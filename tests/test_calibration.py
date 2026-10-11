import struct

import torch

from opensv_train.calibration import fit_input_ranges
from opensv_train.data import load_dataset, make_phrases
from opensv_train.synthetic import render_dataset
from opensv_train.voice import Voice


def test_ranges_fit_stock_voice_and_widen_after_growth(voice_path, tmp_path):
    voice = Voice.load(voice_path)
    render_dataset(voice, tmp_path, phrases=2, seed=9, log=lambda message: None)
    phrases = [phrase for utterance in load_dataset(tmp_path, "japanese") for phrase in make_phrases(utterance)]
    assert fit_input_ranges(voice, phrases, torch.device("cpu")) == []

    encoder = voice.acoustic.phoneme_encoder.layers
    downstream = encoder[2].taps[0].node
    before = struct.unpack_from("<I", downstream.payload, 4)[0]
    with torch.no_grad():
        for tap in encoder[0].taps:
            tap.weight.mul_(2.0 ** (16 - before))
    reports = fit_input_ranges(voice, phrases, torch.device("cpu"))
    after = struct.unpack_from("<I", downstream.payload, 4)[0]
    assert any(report.role == "acoustic" for report in reports)
    assert after < before
    assert all(report.peak < report.limit for report in reports)
