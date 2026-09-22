"""Melody transcription for covers. SheetSage2 stays lazy and out of the import path."""
from __future__ import annotations

import gc
import logging

MELODY_PROMPTS = ("timestamp", "downbeat_meter", "structure", "key", "melody_full")
COVER_AUDIO_KEY = "_cover_audio"
COVER_SHA_KEY = "_cover_sha256"
GENERATION_FIELDS = (
    "style", "lyrics", "cot", "seed", "abc", "cfg_scale", "id",
    "abc_sampling", "semantic_sampling",
)
SHEETSAGE_DEVICES = ("off", "cpu", "cuda", "auto")
COVER_DISABLED = "翻唱未启用。将 YUE2_SHEETSAGE_DEVICE 设为 auto 或 cpu 后重启服务。"

log = logging.getLogger("yue2.cover")


class CoverDisabled(RuntimeError):
    """SheetSage2 was not enabled for this process."""


def generation_fields(request):
    """Fields the YuE2 pipeline accepts. Cover bookkeeping stays out of generation."""
    return {key: request[key] for key in GENERATION_FIELDS if key in request}


def require_melody_abc(result):
    if not isinstance(result, dict):
        raise ValueError("Transcription did not return a melody score")
    abc = result.get("abc")
    if result.get("abc_error") or not isinstance(abc, str) or not abc.strip():
        detail = result.get("abc_error") or "no ABC score"
        raise ValueError(f"Transcription produced no usable melody: {detail}")
    return abc


class MelodyTranscriber:
    """Load SheetSage2 on first use and return a chord-free melody ABC."""

    def __init__(self, model="m-a-p/SheetSage2", revision=None, device="off",
                 local_files_only=False, min_free_gib=8):
        if device not in SHEETSAGE_DEVICES:
            raise ValueError("sheetsage device must be off, cpu, cuda, or auto")
        if not min_free_gib > 0:
            raise ValueError("sheetsage min free memory must be positive")
        self.model_id = model
        self.revision = revision
        self.device = device
        self.local_files_only = local_files_only
        self.min_free_gib = float(min_free_gib)
        self._model = None
        self._torch_device = None
        self._resolved = None

    def resolve_device(self):
        """Pick cpu or cuda once. auto uses CPU when free VRAM is below the threshold."""
        if self._resolved is not None:
            return self._resolved
        if self.device == "off":
            raise CoverDisabled(COVER_DISABLED)
        import torch
        if self.device == "cpu":
            chosen = "cpu"
        elif self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("SheetSage2 device is cuda but CUDA is not available")
            chosen = "cuda"
        elif torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            chosen = "cuda" if free >= self.min_free_gib * 1024 ** 3 else "cpu"
        else:
            chosen = "cpu"
        log.info("SheetSage2 transcription device: %s (requested %s)", chosen, self.device)
        self._resolved = chosen
        return chosen

    def _load_model(self):
        from transformers import AutoModel
        loader = {"trust_remote_code": True, "local_files_only": self.local_files_only}
        if self.revision:
            loader.update(revision=self.revision, code_revision=self.revision)
        return AutoModel.from_pretrained(self.model_id, **loader)

    def load(self):
        if self._model is not None:
            return self._model
        import torch
        device = self.resolve_device()
        model = self._load_model().eval()
        self._torch_device = torch.device(device)
        if device == "cuda":
            model = model.to(self._torch_device)
        self._model = model
        return self._model

    def transcribe(self, audio):
        model = self.load()
        device_type = self._torch_device.type if self._torch_device is not None else "cpu"
        result = model.transcribe(
            str(audio),
            prompts=list(MELODY_PROMPTS),
            melody_only=True,
            dtype="bf16" if device_type == "cuda" else "fp32",
        )
        return require_melody_abc(result)

    def close(self):
        """Drop weights before YuE2 generation so the two models do not share the GPU."""
        self._model = None
        self._torch_device = None
        self._resolved = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
