"""One request protocol for local and Hugging Face song generation."""
from __future__ import annotations
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
from pathlib import Path
import dataclasses
import json
import threading
import time
import numpy as np
import torch

from .protocol import SongRequest, GenerationConfig, Sampling, token_prefixes, negative_prefix, CODEC_OFFSET, resolve_sampling
from .storage import resolve_model, model_identity, identity, write_json, collect_hashes, sha256_file, copy_model_files
from .tokenization_yue2 import YuE2TextTokenizer
from .sampling import generate_tokens, synchronize
from .progress import Progress


@dataclass
class SymbolicPlan:
    request: SongRequest
    abc: str | None
    abc_ids: list[int]
    prefix: list[int]
    timing: dict = field(default_factory=dict)
    truncated: bool = False

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.abc is not None:
            (directory / "score.abc").write_bytes(self.abc.encode("utf-8"))
        np.save(directory / "abc_tokens.npy", np.asarray(self.abc_ids, dtype=np.int32))
        np.save(directory / "prefix.npy", np.asarray(self.prefix, dtype=np.int32))
        write_json(directory / "plan.json", {"request": self.request.to_dict(), "timing": self.timing,
                                              "truncated": self.truncated, "prefix": self.prefix,
                                              "abc_ids": self.abc_ids, "abc": self.abc})


        names = ["plan.json", "abc_tokens.npy", "prefix.npy"] + (["score.abc"] if self.abc is not None else [])
        write_json(directory / "plan_manifest.json", {name: sha256_file(directory / name) for name in names})

    @classmethod
    def load(cls, directory):
        """Restore exact planner output without decoding and retokenizing its ABC."""
        directory = Path(directory)
        hashes = json.loads((directory / "plan_manifest.json").read_text())
        if not {"plan.json", "abc_tokens.npy", "prefix.npy"} <= hashes.keys():
            raise ValueError("Incomplete saved plan")
        for name, digest in hashes.items():
            if name not in {"plan.json", "abc_tokens.npy", "prefix.npy", "score.abc"} or (directory / name).is_symlink():
                raise ValueError("Invalid plan artifact")
            if sha256_file(directory / name) != digest:
                raise ValueError("Saved plan changed; supply modified ABC as an external planner input")
        data = json.loads((directory / "plan.json").read_text())
        for field, filename in (("abc_ids", "abc_tokens.npy"), ("prefix", "prefix.npy")):
            array = np.load(directory / filename, allow_pickle=False)
            if array.ndim != 1 or array.dtype.kind not in "iu" or array.tolist() != data[field]:
                raise ValueError("Saved plan token array mismatch")
        if data["abc"] is not None and ("score.abc" not in hashes or (directory / "score.abc").read_bytes() != data["abc"].encode("utf-8")):
            raise ValueError("Saved ABC text mismatch")
        return cls(SongRequest(**data["request"]), data["abc"], data["abc_ids"], data["prefix"],
                   data["timing"], data["truncated"])


@dataclass
class SemanticResult:
    plan: SymbolicPlan
    tokens: list[int]
    timing: dict
    truncated: bool


@dataclass
class ARResult:
    """Completed autoregressive stages, ready for serialized NAR/VAE rendering."""
    semantic: SemanticResult
    config: dict
    request_identity: str
    started_at: float


@dataclass
class NARResult:
    """Completed acoustic flow matching, ready for serial VAE decoding."""
    ar: ARResult
    latents: np.ndarray
    seconds: float
    batch_size: int = 1


@dataclass
class SongResult:
    audio: np.ndarray
    sample_rate: int
    semantic: SemanticResult
    latents: np.ndarray
    config: dict
    weights: dict
    timing: dict
    request_identity: str

    @property
    def abc(self):
        return self.semantic.plan.abc

    @property
    def truncated(self):
        return {"abc": self.semantic.plan.truncated, "semantic": self.semantic.truncated}

    def save(self, path):
        """Write audio; use save_artifacts(directory) to retain a reproducible run."""
        import soundfile as sf
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() not in {".flac", ".wav"}:
            raise ValueError("Use .flac or .wav; MP3 is an optional delivery conversion")
        sf.write(path, self.audio, self.sample_rate, subtype="PCM_24" if path.suffix.lower() == ".flac" else "FLOAT")
        return str(path)

    def save_artifacts(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.semantic.plan.save(directory)
        self.save(directory / "audio.flac")
        np.save(directory / "semantic.npy", np.asarray(self.semantic.tokens, dtype=np.int32))
        np.save(directory / "latent.npy", self.latents.astype(np.float32))
        write_json(directory / "request.json", self.semantic.plan.request.to_dict())
        write_json(directory / "config.json", self.config)
        result = {"status": "complete", "identity": self.request_identity,
                  "truncated": self.truncated, "sample_rate": self.sample_rate,
                  "audio_seconds": len(self.audio) / self.sample_rate,
                  "weights": self.weights, "timing": self.timing,
                  "artifacts": collect_hashes(directory)}
        write_json(directory / "result.json", result)
        return result


class YuE2Pipeline:
    def __init__(self, model_dir, vae_dir, *, device="auto", memory_budget_gib=24,
                 backend="torch", generation_config=None, verify_hashes=True,
                 vae_core_frames=None, quantization="none", offload_ar=False, progress=True,
                 resident_models=False, vllm_max_num_seqs=4,
                 vllm_max_num_batched_tokens=8192,
                 vllm_gpu_memory_utilization=.3):
        if not isinstance(resident_models, bool):
            raise TypeError("resident_models must be True or False")
        if resident_models and offload_ar:
            raise ValueError("resident_models requires offload_ar=False")
        self.resident_models = resident_models
        if not isinstance(progress, bool):
            raise TypeError("progress must be True or False")
        self.progress = progress
        if backend not in {"torch", "torch-eager", "vllm"}:
            raise ValueError("backend must be torch, torch-eager, or vllm")
        if quantization not in {"none", "fp8"}:
            raise ValueError("quantization must be none or fp8")
        if not 0 < memory_budget_gib:
            raise ValueError("memory_budget_gib must be positive")
        if isinstance(vllm_max_num_seqs, bool) or not isinstance(vllm_max_num_seqs, int) or vllm_max_num_seqs < 1:
            raise ValueError("vllm_max_num_seqs must be a positive integer")
        if (isinstance(vllm_max_num_batched_tokens, bool)
                or not isinstance(vllm_max_num_batched_tokens, int)
                or not 1 <= vllm_max_num_batched_tokens <= 24576):
            raise ValueError("vllm_max_num_batched_tokens must be in [1, 24576]")
        if not 0 < float(vllm_gpu_memory_utilization) <= .9:
            raise ValueError("vllm_gpu_memory_utilization must be in (0, 0.9]")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("highest")
        self.model_dir, self.vae_dir = Path(model_dir), Path(vae_dir)
        self.backend, self.quantization = backend, quantization
        self.vllm_max_num_seqs = vllm_max_num_seqs
        self.vllm_max_num_batched_tokens = vllm_max_num_batched_tokens
        self.vllm_gpu_memory_utilization = float(vllm_gpu_memory_utilization)
        self._vllm_start_lock = threading.Lock()
        self.memory_budget_gib = float(memory_budget_gib)
        self.vae_core_frames = vae_core_frames if vae_core_frames is not None else (512 if memory_budget_gib <= 12 else 1024)
        self.offload_ar = offload_ar
        self.generation_config = generation_config or GenerationConfig()
        self.tokenizer = YuE2TextTokenizer(self.model_dir / "qwen.tiktoken")
        with self._status("Verifying model files"):
            self.weights = {"mot": model_identity(self.model_dir, verify_hashes),
                            "vae": model_identity(self.vae_dir, verify_hashes)}
        self.runtime_sha256 = identity({p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
        self._model, self._vae = None, None
        self.load_timing = {}
        if self.device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The unquantized preset requires CUDA BF16 support")
            total = torch.cuda.get_device_properties(self.device).total_memory
            budget = min((self.memory_budget_gib - 2) * 2**30, total - 2 * 2**30)
            if budget <= 0:
                raise ValueError("Memory budget must leave room for a 2GiB reserve")
            torch.cuda.set_per_process_memory_fraction(min(budget / total, 1), self.device)

    @classmethod
    def from_pretrained(cls, model="m-a-p/YuE2-3B", *, vae="m-a-p/YuE2-Vae",
                        revision=None, vae_revision=None, local_files_only=False,
                        token=None, cache_dir=None, progress=True, **kwargs):
        """Load a song pipeline with English progress on stderr; set progress=False to hide it."""
        if not isinstance(progress, bool):
            raise TypeError("progress must be True or False")
        start = time.perf_counter()
        saved = Path(model) / "pipeline.json"
        if saved.is_file():
            metadata = json.loads(saved.read_text())
            parent = Path(model)
            model = parent / metadata["model"]
            if vae == "m-a-p/YuE2-Vae":
                vae = parent / metadata["vae"]
            kwargs.setdefault("generation_config", GenerationConfig.from_dict(metadata["generation_config"]))
        hub = dict(local_files_only=local_files_only, token=token, cache_dir=cache_dir)
        with Progress(enabled=progress).stage("Resolving model files"):
            model_path = resolve_model(model, revision=revision, **hub)
            vae_path = resolve_model(vae, revision=vae_revision, **hub)
        result = cls(model_path, vae_path, progress=progress, **kwargs)
        result.load_timing["resolve_and_integrity_seconds"] = time.perf_counter() - start
        return result

    @contextmanager
    def _status(self, label, *, total=None, unit=None):
        # Display state never enters the request/config identity or model RNG.
        with Progress(enabled=self.progress) as reporter:
            with reporter.stage(label, total=total, unit=unit) as stage:
                yield stage

    def save_pretrained(self, directory):
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("save_pretrained needs an empty pipeline destination")
        for name, source in (("YuE2-3B", self.model_dir), ("YuE2-Vae", self.vae_dir)):
            destination = directory / name
            if destination.resolve() == source.resolve():
                continue
            copy_model_files(source, destination)
        write_json(directory / "pipeline.json", {"model": "YuE2-3B", "vae": "YuE2-Vae",
                   "generation_config": self.generation_config.to_dict(), "source_weights": self.weights})

    def _load_model(self, for_nar=False):
        loading = self._model is None or next(self._model.parameters()).device != self.device
        with self._status("Loading model") if loading else nullcontext():
            if self._model is None:
                from .modeling_yue2 import YuE2ForCausalLM
                start = time.perf_counter()
                self._model = YuE2ForCausalLM.from_pretrained(self.model_dir, local_files_only=True,
                              torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
                self.load_timing["mot_load_seconds"] = time.perf_counter() - start
            if self.quantization == "fp8" and not for_nar:
                from .quantization import prepare_fp8_ar
                prepare_fp8_ar(self._model, self.device)
            self._model.to(self.device)
        return self._model

    def _request(self, style=None, lyrics=None, *, tags=None, **kwargs):
        if style is not None and tags is not None and style != tags:
            raise ValueError("style and tags are aliases and cannot disagree")
        style = tags if style is None else style
        if style is None or lyrics is None:
            raise ValueError("Provide style and lyrics")
        return SongRequest(style=style, lyrics=lyrics, **kwargs)

    def _generate(self, prefix, sampling, seed, phase, **kwargs):
        model = self._load_model() if self.backend != "vllm" else None
        callback = kwargs.pop("on_token", None)
        label = "Planning score" if phase == "abc" else "Generating song"
        with self._status(label, unit="tokens") as status:
            def on_output(token_phase, token):
                # The backend emits each real output once, including its end token.
                # Prefixes, provided scores and extra CFG branches are not output.
                status.advance()
                if callback is not None:
                    callback(token_phase, token)
            observed = on_output if self.progress else callback
            if self.backend == "vllm":
                from .fast import generate_vllm
                result = generate_vllm(self, prefix, sampling, seed, phase, on_token=observed, **kwargs)
            else:
                result = generate_tokens(model, prefix, sampling, seed, phase,
                                         use_cuda_graph=self.backend != "torch-eager", on_token=observed, **kwargs)
            if result[2]:
                status.finish(status="truncated")
            return result

    def plan(self, style=None, lyrics=None, *, tags=None, request=None, abc_sampling=None,
             cancelled=None, on_token=None, **kwargs):
        request = request or self._request(style, lyrics, tags=tags, **kwargs)
        if request.cot == "off":
            return SymbolicPlan(request, None, [], token_prefixes(request, self.tokenizer))
        if request.abc is not None:
            with self._status("Using provided score"):
                ids = self.tokenizer.encode(request.abc)
                return SymbolicPlan(request, request.abc, ids, token_prefixes(request, self.tokenizer, ids),
                                    {"seconds": 0., "output_tokens": 0, "external_prefix_tokens": len(ids)})
        sampling = resolve_sampling(abc_sampling, self.generation_config.abc)
        ids, timing, truncated = self._generate(token_prefixes(request, self.tokenizer), sampling,
                            request.seed, "abc", cancelled=cancelled, on_token=on_token)
        return SymbolicPlan(request, self.tokenizer.decode(ids), ids,
                            token_prefixes(request, self.tokenizer, ids), timing, truncated)

    def generate_semantic(self, plan, *, sampling=None, cancelled=None, on_token=None):
        if not isinstance(plan, SymbolicPlan):
            raise TypeError("Pass the SymbolicPlan returned by pipe.plan()")
        request = plan.request
        expected = token_prefixes(request, self.tokenizer, plan.abc_ids)
        if expected != plan.prefix:
            raise ValueError("Plan prefix disagrees with request/exact ABC IDs")
        sampling = resolve_sampling(sampling, self.generation_config.semantic)
        negative = negative_prefix(request, self.tokenizer, plan.abc_ids) if request.guidance != 1 else None
        ids, timing, truncated = self._generate(plan.prefix, sampling, request.seed, "semantic",
                        negative=negative, cfg_scale=request.guidance, legacy_off=request.cot == "off",
                        cancelled=cancelled, on_token=on_token)
        return SemanticResult(plan, [int(t) - CODEC_OFFSET for t in ids], timing, truncated)

    def synthesize(self, semantic, *, cancelled=None):
        from .nar import synthesize
        if not isinstance(semantic, SemanticResult):
            raise TypeError("Pass the SemanticResult returned by generate_semantic()")
        if token_prefixes(semantic.plan.request, self.tokenizer, semantic.plan.abc_ids) != semantic.plan.prefix:
            raise ValueError("Semantic result does not retain the request's exact prefix")
        if self.quantization != "none":
            from .quantization import restore_ar
            restore_ar(self._model)
        model = self._load_model(for_nar=True)
        with self._status("Synthesizing audio", unit="steps") as status:
            report = (lambda completed, total: status.update(completed, total=total)) if self.progress else None
            result = synthesize(model, semantic.plan.prefix, semantic.tokens,
                                semantic.plan.request.seed, steps=self.generation_config.ode_steps,
                                context=self.generation_config.context, offload_ar=self.offload_ar,
                                cancelled=cancelled, on_progress=report)
            return result.detach().float().cpu().numpy()

    def close(self):
        if self.backend == "vllm":
            from .fast import close_vllm
            close_vllm(self)
        self._model, self._vae = None, None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def preload(self):
        """Load reusable weights and start the persistent AR engine before traffic."""
        from .modeling_vae import YuE2VAE
        if self.backend == "vllm":
            from .fast import preload_vllm
            preload_vllm(self)
            if self.resident_models:
                # Acoustic synthesis needs the MoT AR path for prefix prefill as
                # well as its NAR path, so keep the complete PyTorch model.
                self._load_model(for_nar=True)
        else:
            self._load_model()
        if self._vae is None:
            self._vae = YuE2VAE.from_pretrained(self.vae_dir, decoder_only=True,
                                              device="cpu", local_files_only=True)
        if self.resident_models:
            self._vae.to(self.device)
        synchronize(self.device)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def decode(self, latents, *, full=False, vae=None, cancelled=None):
        from .modeling_vae import YuE2VAE
        resident = getattr(self, "resident_models", False)
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before audio decode")
        with self._status("Loading audio decoder"):
            if self._model is not None and not resident:
                self._model.to("cpu")
            if self.device.type == "cuda" and not resident:
                torch.cuda.empty_cache()
            if vae is not None:
                model = YuE2VAE.from_pretrained(vae, decoder_only=True, device=self.device)
            else:
                if self._vae is None:
                    self._vae = YuE2VAE.from_pretrained(self.vae_dir, decoder_only=True, device="cpu",
                                                        local_files_only=True)
                model = self._vae.to(self.device)
        z = torch.as_tensor(latents, dtype=torch.float32)
        if z.ndim == 2 and z.shape[1] == 64:
            z = z.T.unsqueeze(0)
        if z.ndim != 3 or z.shape[0] != 1 or z.shape[1] != 64:
            raise ValueError("Expected latents [T,64] or [1,64,T]")
        try:
            tiles = 1 if full else (z.shape[-1] + self.vae_core_frames - 1) // self.vae_core_frames
            with self._status("Decoding audio", total=tiles, unit="chunks") as status:
                report = None
                if self.progress or cancelled is not None:
                    def report(completed, total):
                        if cancelled is not None and cancelled():
                            raise InterruptedError("Cancelled during audio decode")
                        if self.progress:
                            status.update(completed, total=total)
                with torch.inference_mode():
                    if full:
                        audio = model.decode(z.to(self.device)).cpu()
                        status.update(1)
                    else:
                        audio = model.decode_tiled(z, core_frames=self.vae_core_frames, halo_frames=16,
                                                   output_device="cpu", on_progress=report)
                if not torch.isfinite(audio).all():
                    raise ValueError("VAE produced non-finite audio")
                return audio[0].float().clamp(-1, 1).T.contiguous().numpy()
        finally:
            if not resident or vae is not None:
                model.to("cpu")
            if self.device.type == "cuda" and not resident:
                torch.cuda.empty_cache()

    def effective_config(self, request, abc_sampling=None, semantic_sampling=None):
        config = self.generation_config.to_dict()
        for key, sampling in (("abc", abc_sampling), ("semantic", semantic_sampling)):
            if sampling is not None:
                config[key] = dataclasses.asdict(resolve_sampling(sampling, getattr(self.generation_config, key)))
        defaults = GenerationConfig().to_dict()
        overrides = {k: v for k, v in config.items() if defaults.get(k) != v}
        if request.guidance != (1.01 if request.cot == "off" else 1.0):
            overrides["cfg_scale"] = request.guidance
        return {"generation": config, "overrides": overrides,
                "cot": request.cot, "cfg_scale": request.guidance,
                "cfg_negative": "instruction_only" if request.cot == "off" else "same_instruction_and_exact_abc",
                "backend": self.backend, "quantization": self.quantization,
                "model_dtype": "bfloat16", "vae_dtype": "float32", "vae_decode": "halo_crop",
                "vae_core_frames": self.vae_core_frames, "vae_halo_frames": 16,
                "device": str(self.device), "memory_budget_gib": self.memory_budget_gib,
                "offload_ar": self.offload_ar, "resident_models": getattr(self, "resident_models", False),
                "vllm_max_num_seqs": self.vllm_max_num_seqs if self.backend == "vllm" else None,
                "vllm_max_num_batched_tokens": (
                    self.vllm_max_num_batched_tokens if self.backend == "vllm" else None),
                "vllm_gpu_memory_utilization": (
                    self.vllm_gpu_memory_utilization if self.backend == "vllm" else None),
                "runtime_sha256": self.runtime_sha256,
                "decoder_release": json.loads((self.vae_dir / "config.json").read_text()).get("release_variant"),
                "validation_status": "unvalidated"}

    def parallel_ar_eligible(self, request):
        """Whether this request can stay entirely on the concurrent vLLM AR path."""
        if not isinstance(request, SongRequest):
            request = SongRequest(**request)
        return (self.backend == "vllm" and self.device.type == "cuda" and self.quantization == "none"
                and request.cot in {"full", "melody"} and request.guidance == 1)

    def generate_ar(self, style=None, lyrics=None, *, tags=None, abc_sampling=None,
                    semantic_sampling=None, cancelled=None, on_token=None, on_stage=None, **kwargs):
        """Run score planning and semantic-token AR; safe to batch through vLLM."""
        request = self._request(style, lyrics, tags=tags, **kwargs)
        config = self.effective_config(request, abc_sampling, semantic_sampling)
        request_id = identity({"request": request.to_dict(), "config": config, "weights": self.weights})
        start = time.perf_counter()
        if on_stage is not None:
            on_stage("planning")
        plan = self.plan(request=request, abc_sampling=abc_sampling, cancelled=cancelled, on_token=on_token)
        if on_stage is not None:
            on_stage("semantic")
        semantic = self.generate_semantic(plan, sampling=semantic_sampling, cancelled=cancelled, on_token=on_token)
        return ARResult(semantic, config, request_id, start)

    def render_ar(self, ar_result, *, cancelled=None, on_stage=None):
        """Run the non-AR acoustic stages for a completed AR result."""
        if not isinstance(ar_result, ARResult):
            raise TypeError("Pass the ARResult returned by generate_ar()")
        if on_stage is not None:
            on_stage("synthesis")
        nar_start = time.perf_counter()
        try:
            latents = self.synthesize(ar_result.semantic, cancelled=cancelled)
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise MemoryError("NAR exceeded its runtime GPU allocation") from error
        nar_result = NARResult(ar_result, latents, time.perf_counter() - nar_start)
        return self.render_nar(nar_result, cancelled=cancelled, on_stage=on_stage)

    def nar_batch_admission(self, ar_results):
        """Estimate a FIFO NAR window before allocating padded KV caches."""
        from .nar import nar_batch_memory_estimate, song_chunks
        if not ar_results or any(not isinstance(result, ARResult) for result in ar_results):
            raise ValueError("NAR admission requires at least one ARResult")
        try:
            model = self._load_model(for_nar=True)
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise MemoryError("NAR admission could not load the acoustic model") from error
        songs = [
            song_chunks(result.semantic.plan.prefix, result.semantic.tokens,
                        result.semantic.plan.request.seed, self.generation_config.context)
            for result in ar_results
        ]
        return nar_batch_memory_estimate(model, songs)

    def generate_nar_batch(self, ar_results, *, cancelled=None, on_stage=None):
        """Run padded shared-forward NAR and preserve row order."""
        from .nar import synthesize_batch
        ar_results = list(ar_results)
        if len(ar_results) < 2 or any(not isinstance(result, ARResult) for result in ar_results):
            raise ValueError("NAR batching requires at least two ARResult objects")
        callbacks = list(cancelled or [None] * len(ar_results))
        stages = list(on_stage or [None] * len(ar_results))
        if len(callbacks) != len(ar_results) or len(stages) != len(ar_results):
            raise ValueError("NAR batch callbacks must align with results")
        admission = self.nar_batch_admission(ar_results)
        if not admission["allowed"]:
            raise MemoryError("NAR batch does not leave the required GPU memory reserve")
        values, active = [None] * len(ar_results), []
        for row, (callback, stage) in enumerate(zip(callbacks, stages)):
            try:
                if callback is not None and callback():
                    raise InterruptedError("Cancelled before acoustic flow matching")
                if stage is not None:
                    stage("synthesis")
            except Exception as error:
                values[row] = error
            else:
                active.append(row)
        if not active:
            return values
        model = self._load_model(for_nar=True)
        started = time.perf_counter()
        try:
            latents = synthesize_batch(
                model,
                [ar_results[row].semantic.plan.prefix for row in active],
                [ar_results[row].semantic.tokens for row in active],
                [ar_results[row].semantic.plan.request.seed for row in active],
                steps=self.generation_config.ode_steps, context=self.generation_config.context,
                cancelled=[callbacks[row] for row in active],
            )
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise MemoryError("NAR batch exceeded its runtime GPU allocation") from error
        seconds = time.perf_counter() - started
        for row, value in zip(active, latents):
            values[row] = (
                value if isinstance(value, Exception)
                else NARResult(ar_results[row], value.detach().float().cpu().numpy(),
                               seconds, len(active))
            )
        return values

    def render_nar(self, nar_result, *, cancelled=None, on_stage=None):
        """Decode one completed NAR result; VAE intentionally remains serial."""
        if not isinstance(nar_result, NARResult):
            raise TypeError("Pass the NARResult returned by the acoustic stage")
        ar_result, latents = nar_result.ar, nar_result.latents
        semantic = ar_result.semantic
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before VAE")
        vae_start = time.perf_counter()
        if on_stage is not None:
            on_stage("decode")
        try:
            audio = self.decode(
                latents, cancelled=cancelled) if cancelled is not None else self.decode(latents)
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise MemoryError("VAE exceeded its runtime GPU allocation") from error
        timing = {"abc": semantic.plan.timing, "semantic": semantic.timing, "nar_seconds": nar_result.seconds,
                  "vae_seconds": time.perf_counter() - vae_start, "load": dict(self.load_timing),
                  "e2e_seconds": time.perf_counter() - ar_result.started_at}
        config = dict(ar_result.config)
        config.update(nar_execution="batched_padded" if nar_result.batch_size > 1 else "sequential",
                      nar_batch_size=nar_result.batch_size, vae_execution="sequential")
        Progress(enabled=self.progress).complete(len(audio) / 48000, timing["e2e_seconds"],
                                                truncated=semantic.plan.truncated or semantic.truncated)
        result = SongResult(audio, 48000, semantic, latents, config, self.weights,
                            timing, ar_result.request_identity)
        return result

    def generate_batch(self, requests, **kwargs):
        """Experimental batched AR forwards; see yue2.batching for limits."""
        from .batching import generate_batch
        return generate_batch(self, requests, **kwargs)

    def __call__(self, style=None, lyrics=None, *, tags=None, abc_sampling=None,
                 semantic_sampling=None, cancelled=None, on_token=None, on_stage=None, **kwargs):
        ar_result = self.generate_ar(style, lyrics, tags=tags, abc_sampling=abc_sampling,
                    semantic_sampling=semantic_sampling, cancelled=cancelled,
                    on_token=on_token, on_stage=on_stage, **kwargs)
        return self.render_ar(ar_result, cancelled=cancelled, on_stage=on_stage)
