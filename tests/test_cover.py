from types import SimpleNamespace

import pytest

from yue2.cli import parser
from yue2.cover import MELODY_PROMPTS, CoverDisabled, MelodyTranscriber, require_melody_abc


def test_melody_prompts_omit_chords():
    assert "chord_full" not in MELODY_PROMPTS
    assert "melody_full" in MELODY_PROMPTS


def test_require_melody_abc_rejects_empty_scores():
    assert require_melody_abc({"abc": "X:1\n", "abc_error": None}) == "X:1\n"
    with pytest.raises(ValueError):
        require_melody_abc({"abc": "  ", "abc_error": None})
    with pytest.raises(ValueError):
        require_melody_abc({"abc": None, "abc_error": "no beats"})


def test_transcribe_requests_melody_only_without_writing_a_score(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"audio")
    seen = {}

    class Model:
        def transcribe(self, source, **kwargs):
            seen["source"] = source
            seen.update(kwargs)
            return {"abc": "X:1\n", "abc_error": None}

    transcriber = MelodyTranscriber(device="cpu")
    transcriber._model = Model()
    transcriber._torch_device = SimpleNamespace(type="cpu")
    assert transcriber.transcribe(audio) == "X:1\n"
    assert seen["source"] == str(audio)
    assert seen["melody_only"] is True
    assert seen["prompts"] == list(MELODY_PROMPTS)
    assert seen["dtype"] == "fp32"
    assert "output_dir" not in seen


def test_service_defaults_sheetsage_to_auto():
    from yue2.service import Settings
    assert Settings(api_key="k" * 16).sheetsage_device == "auto"


def test_cover_cli_requires_audio_and_pins_sheetsage():
    with pytest.raises(SystemExit):
        parser().parse_args(["cover"])
    args = parser().parse_args(["cover", "--audio", "song.mp3", "--style", "jazz"])
    assert args.command == "cover"
    assert args.audio.name == "song.mp3"
    assert args.sheetsage == "m-a-p/SheetSage2"
    assert args.sheetsage_device == "auto"
    assert args.sheetsage_min_free_gib == 8
    assert not hasattr(args, "stage")


def _fake_cuda(monkeypatch, free_bytes):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (free_bytes, 32 * 1024 ** 3))
    return torch


def test_auto_uses_cpu_when_free_memory_is_below_threshold(monkeypatch):
    _fake_cuda(monkeypatch, 4 * 1024 ** 3)
    transcriber = MelodyTranscriber(device="auto", min_free_gib=8)
    assert transcriber.resolve_device() == "cpu"
    assert transcriber.resolve_device() == "cpu"


def test_auto_uses_cuda_when_free_memory_is_enough(monkeypatch):
    _fake_cuda(monkeypatch, 10 * 1024 ** 3)
    assert MelodyTranscriber(device="auto", min_free_gib=8).resolve_device() == "cuda"


def test_auto_without_cuda_uses_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert MelodyTranscriber(device="auto").resolve_device() == "cpu"


def test_explicit_cuda_without_gpu_fails(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        MelodyTranscriber(device="cuda").resolve_device()


def test_off_refuses_to_load():
    with pytest.raises(CoverDisabled, match="YUE2_SHEETSAGE_DEVICE"):
        MelodyTranscriber(device="off").resolve_device()


def test_low_memory_transcribe_stays_on_cpu_fp32(monkeypatch, tmp_path):
    _fake_cuda(monkeypatch, 4 * 1024 ** 3)
    moved = []

    class Model:
        def eval(self):
            return self

        def to(self, device):
            moved.append(("to", str(device)))
            return self

        def transcribe(self, source, **kwargs):
            moved.append(("dtype", kwargs["dtype"]))
            return {"abc": "X:1\n", "abc_error": None}

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")
    transcriber = MelodyTranscriber(device="auto", min_free_gib=8)
    transcriber._load_model = lambda: Model()
    assert transcriber.transcribe(audio) == "X:1\n"
    assert moved == [("dtype", "fp32")]
