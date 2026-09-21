"""Real tiny-decoder equivalence plus residency/cancellation contract checks on CPU."""
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
from yue2.pipeline import ARResult, YuE2Pipeline


def tiny_decoder():
    return YuE2VAE(YuE2VAEConfig(
        decoder_config=dict(channels=1, c_mults=[1] * 6, strides=[2, 2, 4, 4, 5, 6],
                            use_snake=True, out_channels=2, latent_dim=64,
                            snake_type="vanilla", final_tanh=False), latent_dim=64), decoder_only=True)


def pipe_with(decoder, resident):
    pipe = object.__new__(YuE2Pipeline)
    pipe.device, pipe.progress = torch.device("cpu"), False
    pipe.resident_models, pipe.vae_core_frames = resident, 4
    pipe._model, pipe._vae = Mock(), decoder
    return pipe


def test_nar_admission_translates_model_load_oom():
    pipe = object.__new__(YuE2Pipeline)
    pipe._load_model = Mock(side_effect=torch.OutOfMemoryError("simulated"))
    result = ARResult(None, {}, "request", 0)
    with patch("torch.cuda.empty_cache") as empty:
        with pytest.raises(MemoryError, match="admission"):
            pipe.nar_batch_admission([result])
    empty.assert_called_once()


def test_residency_eliminates_offloads_and_preserves_real_decoder_output():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    decoder = tiny_decoder()
    latent = np.random.default_rng(42).normal(size=(9, 64)).astype(np.float32)
    baseline, resident = pipe_with(decoder, False), pipe_with(decoder, True)
    with patch.object(decoder, "to", wraps=decoder.to) as move:
        expected = baseline.decode(latent)
        assert move.call_count == 2  # to device, then offload
    baseline._model.to.assert_called_once_with("cpu")
    with patch.object(decoder, "to", wraps=decoder.to) as move:
        actual = resident.decode(latent)
        assert move.call_count == 1  # keep resident after decode
    resident._model.to.assert_not_called()
    np.testing.assert_array_equal(actual, expected)


def test_decode_cancel_stops_between_tiles():
    torch.set_num_threads(1)
    decoder = tiny_decoder()
    pipe = pipe_with(decoder, True)
    checks = iter([False, True])
    with patch.object(decoder, "decode", wraps=decoder.decode) as decode:
        with pytest.raises(InterruptedError, match="during audio decode"):
            pipe.decode(np.zeros((12, 64), dtype=np.float32), cancelled=lambda: next(checks))
        assert decode.call_count == 1


def test_residency_rejects_ar_offload():
    with pytest.raises(ValueError, match="resident_models requires"):
        YuE2Pipeline("missing-model", "missing-vae", resident_models=True, offload_ar=True)


def test_vllm_residency_preloads_acoustic_model_and_vae(monkeypatch):
    from yue2 import fast, modeling_vae, pipeline
    subject = object.__new__(YuE2Pipeline)
    subject.backend, subject.resident_models = "vllm", True
    subject.device, subject.vae_dir = torch.device("cpu"), "vae"
    subject._model = subject._vae = None
    subject._load_model = Mock(return_value=Mock())
    decoder = Mock()
    monkeypatch.setattr(fast, "preload_vllm", Mock())
    monkeypatch.setattr(modeling_vae.YuE2VAE, "from_pretrained", Mock(return_value=decoder))
    monkeypatch.setattr(pipeline, "synchronize", Mock())

    subject.preload()

    fast.preload_vllm.assert_called_once_with(subject)
    subject._load_model.assert_called_once_with(for_nar=True)
    modeling_vae.YuE2VAE.from_pretrained.assert_called_once_with(
        "vae", decoder_only=True, device="cpu", local_files_only=True)
    decoder.to.assert_called_once_with(subject.device)


def test_stage_callback_failure_stops_before_model_work():
    pipe = object.__new__(YuE2Pipeline)
    pipe._request = Mock(return_value=Mock(to_dict=lambda: {}))
    pipe.effective_config = Mock(return_value={})
    pipe.weights = {}
    pipe.plan = Mock()
    def reject(stage):
        assert stage == "planning"
        raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        pipe(style="piano", lyrics="words", on_stage=reject)
    pipe.plan.assert_not_called()
