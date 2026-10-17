from __future__ import annotations

import numpy as np
import soundfile

from opensv_train import audio_io


def test_primary_reader_returns_float_audio_with_channel_shape(tmp_path):
    path = tmp_path / "stereo.wav"
    source = np.column_stack([
        np.full(480, -0.5, dtype=np.float32),
        np.full(480, 0.5, dtype=np.float32),
    ])
    soundfile.write(str(path), source, 48000)
    frames, rate = audio_io.read_wav_frames(path)
    assert rate == 48000
    assert frames.shape == source.shape
    assert frames.dtype == np.float32
    assert np.allclose(frames, source, atol=1e-4)


def test_scipy_fallback_normalizes_signed_pcm_wav(monkeypatch, tmp_path):
    path = tmp_path / "pcm.wav"
    source = np.array([[-32768], [-16384], [0], [16384], [32767]], dtype=np.int16)
    from scipy.io import wavfile
    wavfile.write(str(path), 48000, source)

    def fail_primary(*_args, **_kwargs):
        raise RuntimeError("simulated unsupported subtype")

    monkeypatch.setattr(audio_io.soundfile, "read", fail_primary)
    frames, rate = audio_io.read_wav_frames(path)
    assert rate == 48000
    assert frames.shape == source.shape
    assert np.allclose(frames[:, 0], [-1.0, -0.5, 0.0, 0.5, 32767 / 32768], atol=1e-6)


def test_true_decode_failure_reports_both_decoders(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not a wav")
    try:
        audio_io.read_wav_frames(path)
    except ValueError as error:
        assert "libsndfile" in str(error)
        assert "SciPy WAV reader" in str(error)
    else:
        raise AssertionError("broken WAV should fail both decoders")
