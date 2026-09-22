"""FastAPI + durable queue + concurrent vLLM AR and serialized acoustic stages.

The API process imports no torch until its inference worker starts. A test can
inject a tiny pipeline without downloading weights or requiring a GPU.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
import fcntl
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import threading
import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .cover import COVER_AUDIO_KEY, COVER_DISABLED, COVER_SHA_KEY, generation_fields
from .service_store import IdempotencyConflict, JobStore, QueueFull, TERMINAL

JSON_BODY_LIMIT = 256 * 1024
COVER_BODY_LIMIT = 40 * 1024 * 1024

log = logging.getLogger("yue2.service")


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str = Field(min_length=16, repr=False)
    data_dir: Path = Path("outputs/service")
    model: str = "m-a-p/YuE2-3B"
    vae: str = "m-a-p/YuE2-Vae"
    revision: str | None = None
    vae_revision: str | None = None
    device: str = "cuda"
    backend: Literal["torch", "torch-eager", "vllm"] = "vllm"
    resident_models: bool = True
    memory_budget_gib: float = Field(default=30, gt=2, allow_inf_nan=False)
    quantization: Literal["none", "fp8"] = "none"
    ar_concurrency: int = Field(default=4, ge=1, le=16)
    vllm_max_num_seqs: int = Field(default=4, ge=1, le=16)
    vllm_max_num_batched_tokens: int = Field(default=8192, ge=1, le=24576)
    vllm_gpu_memory_utilization: float = Field(default=.3, gt=0, le=.9, allow_inf_nan=False)
    nar_batch_size: int = Field(default=2, ge=1, le=2)
    ar_nar_overlap: bool = True
    ar_batch_wait_ms: int = Field(default=50, ge=0, le=1000)
    ode_steps: int = Field(default=32, ge=1, le=64)
    vae_core_frames: int = Field(default=1024, ge=64, le=4096)
    max_pending: int = Field(default=16, ge=1, le=10000)
    task_timeout_seconds: float = Field(default=1200, gt=0, allow_inf_nan=False)
    artifact_retention_seconds: int = Field(default=86400, ge=60)
    artifact_max_gib: float = Field(default=5, gt=0, allow_inf_nan=False)
    artifact_cleanup_interval_seconds: int = Field(default=300, ge=10)
    warmup: bool = True
    local_files_only: bool = False
    sheetsage: str = "m-a-p/SheetSage2"
    sheetsage_revision: str | None = None
    sheetsage_device: Literal["off", "cpu", "cuda", "auto"] = "auto"
    sheetsage_min_free_gib: float = Field(default=8, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def compatible_backend(self):
        if self.backend == "vllm" and self.quantization != "none":
            raise ValueError("The current vLLM adapter does not support FP8")
        if self.backend == "vllm" and self.ar_concurrency > self.vllm_max_num_seqs:
            raise ValueError("YUE2_AR_CONCURRENCY cannot exceed YUE2_VLLM_MAX_NUM_SEQS")
        return self

    @classmethod
    def from_env(cls):
        return cls(**{name: os.environ[f"YUE2_{name.upper()}"] for name in cls.model_fields
                      if f"YUE2_{name.upper()}" in os.environ})


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    n: int = Field(default=1, ge=1, le=2, strict=True)
    style: str = Field(min_length=1, max_length=2000)
    lyrics: str = Field(min_length=1, max_length=16000)
    cot: Literal["full", "melody", "off"] = "full"
    seed: int = Field(default=831001, ge=0, lt=2**63, strict=True)
    abc: str | None = Field(default=None, min_length=1, max_length=64000)
    cfg_scale: float | None = Field(default=None, ge=0, le=20)

    @field_validator("style", "lyrics", "abc")
    @classmethod
    def nonblank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Text must not be blank")
        return value

    @model_validator(mode="after")
    def score_mode(self):
        if self.abc is not None and self.cot == "off":
            raise ValueError("ABC requires cot=full or melody")
        return self


def build_pipeline(settings):
    from .pipeline import YuE2Pipeline
    from .protocol import GenerationConfig
    return YuE2Pipeline.from_pretrained(
        settings.model, vae=settings.vae, revision=settings.revision, vae_revision=settings.vae_revision,
        device=settings.device, backend=settings.backend, resident_models=settings.resident_models,
        memory_budget_gib=settings.memory_budget_gib, quantization=settings.quantization,
        vllm_max_num_seqs=settings.vllm_max_num_seqs,
        vllm_max_num_batched_tokens=settings.vllm_max_num_batched_tokens,
        vllm_gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
        vae_core_frames=settings.vae_core_frames, local_files_only=settings.local_files_only,
        generation_config=GenerationConfig(ode_steps=settings.ode_steps), progress=False)


class _AROverlapControl:
    """Coordinate one prefetching AR wave with the parent acoustic lane."""

    def __init__(self):
        self.condition = threading.Condition()
        self.paused = False
        self.cancelled = False
        self.ar_active = False

    def begin_ar(self, stop_event):
        with self.condition:
            while self.paused and not self.cancelled and not stop_event.is_set():
                self.condition.wait(.1)
            if self.cancelled or stop_event.is_set():
                return False
            self.ar_active = True
            self.condition.notify_all()
            return True

    def wait_resumed(self, stop_event):
        with self.condition:
            while self.paused and not self.cancelled and not stop_event.is_set():
                self.condition.wait(.1)
            return not self.cancelled and not stop_event.is_set()

    def finish_ar(self):
        with self.condition:
            self.ar_active = False
            self.condition.notify_all()

    def pause_and_wait(self):
        with self.condition:
            self.paused = True
            while self.ar_active:
                self.condition.wait(.1)

    def resume(self):
        with self.condition:
            self.paused = False
            self.condition.notify_all()

    def cancel_and_wait(self):
        with self.condition:
            self.cancelled = True
            self.paused = True
            self.condition.notify_all()
            while self.ar_active:
                self.condition.wait(.1)

    def restart(self):
        with self.condition:
            self.cancelled = False
            self.paused = False
            self.condition.notify_all()

    def active(self):
        with self.condition:
            return self.ar_active


class JobWorker:
    def __init__(self, settings, factory, transcriber=None):
        self.settings, self.factory = settings, factory
        self.transcriber = transcriber
        self._transcriber_lock = threading.Lock()
        self.root = settings.data_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(self.root / "jobs.sqlite3")
        self.stop_event, self.wake = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.active = {}
        self.ready = False
        self.startup_error = False
        self.boot_id = uuid.uuid4().hex
        self.pipeline = None
        self.thread = None
        self.lock_file = None
        self.overlap_control = None

    def start(self):
        self.lock_file = (self.root / "worker.lock").open("a+")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close()
            self.lock_file = None
            raise RuntimeError("This data directory already has a worker. Use one Uvicorn worker per GPU.") from None
        try:
            self.store.recover()
            self.thread = threading.Thread(target=self._run, name="yue2-gpu", daemon=True)
            self.thread.start()
        except BaseException:
            self.lock_file.close()
            self.lock_file = None
            raise

    def stop(self):
        self.ready = False
        self.stop_event.set()
        with self.lock:
            for cancel_event in self.active.values():
                cancel_event.set()
        if self.overlap_control is not None:
            self.overlap_control.resume()
        self.wake.set()
        if self.thread is not None:
            self.thread.join()  # cooperative cancellation; supervisor may force-stop a stuck CUDA driver
        if self.lock_file is not None:
            self.lock_file.close()
            self.lock_file = None

    @staticmethod
    def _directory_size(path):
        total = 0
        for directory, _, names in os.walk(path, followlinks=False):
            for name in names:
                try:
                    total += (Path(directory) / name).lstat().st_size
                except FileNotFoundError:
                    pass
        return total

    def cleanup_artifacts(self, now=None):
        """Expire terminal-job files by age and enforce a per-data-directory cap."""
        now = time.time() if now is None else now
        artifacts = self.root / "artifacts"
        if not artifacts.is_dir():
            return {"removed": 0, "bytes": 0, "remaining_bytes": 0}

        records = self.store.artifact_records()
        cutoff = now - self.settings.artifact_retention_seconds
        candidates = []
        total = 0
        for path in artifacts.iterdir():
            if path.is_symlink() or not path.is_dir():
                continue
            size = self._directory_size(path)
            total += size
            record = records.get(path.name)
            if record is None:
                completed_at = path.stat().st_mtime
                if completed_at < cutoff:
                    candidates.append((completed_at, path, size))
                continue
            if record["status"] not in TERMINAL:
                continue
            completed_at = (record["finished_at"] or record["updated_at"]
                            or record["created_at"] or path.stat().st_mtime)
            candidates.append((completed_at, path, size))

        selected = {path for completed_at, path, _ in candidates if completed_at < cutoff}
        remaining = total - sum(size for _, path, size in candidates if path in selected)
        limit = int(self.settings.artifact_max_gib * 1024 ** 3)
        for _, path, size in sorted(candidates):
            if remaining <= limit:
                break
            if path not in selected:
                selected.add(path)
                remaining -= size

        removed = removed_bytes = 0
        for _, path, size in sorted(candidates):
            if path not in selected:
                continue
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                continue
            except OSError:
                log.warning("Could not remove expired artifact directory %s", path, exc_info=True)
                continue
            removed += 1
            removed_bytes += size
        removed_covers, cover_bytes = self._cleanup_cover_audio(now)
        return {"removed": removed + removed_covers, "bytes": removed_bytes + cover_bytes,
                "remaining_bytes": max(0, total - removed_bytes)}

    def _cleanup_cover_audio(self, now):
        covers = self.root / "covers"
        if not covers.is_dir():
            return 0, 0
        in_use = self.store.active_cover_audio()
        cutoff = now - self.settings.artifact_retention_seconds
        removed = removed_bytes = 0
        for path in covers.iterdir():
            if path.is_symlink() or not path.is_file():
                continue
            try:
                resolved = str(path.resolve())
                if resolved in in_use or path.stat().st_mtime >= cutoff:
                    continue
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                log.warning("Could not remove expired cover audio %s", path, exc_info=True)
                continue
            removed += 1
            removed_bytes += size
        return removed, removed_bytes

    def cancel(self, job_id):
        job = self.store.cancel(job_id)
        with self.lock:
            ids = job.get("candidate_ids", [job_id]) if job else [job_id]
            for child_id in ids:
                cancel_event = self.active.get(child_id)
                if cancel_event is not None:
                    cancel_event.set()
        self.wake.set()
        return job

    def _load(self):
        self.ready = False
        self.pipeline = self.factory(self.settings)
        self.pipeline.preload()
        if self.settings.warmup and not self.stop_event.is_set():
            # Short disposable generation warms libraries, not a cached production song.
            self.pipeline(style="English piano pop", lyrics="[Verse]\nA new day begins",
                          cot="full", seed=42, abc_sampling={"min_tokens": 16, "max_tokens": 32},
                          semantic_sampling={"min_tokens": 16, "max_tokens": 32},
                          cancelled=self.stop_event.is_set)
        self.ready = not self.stop_event.is_set()

    def _reload_pipeline(self):
        self.ready = False
        self.pipeline.close()
        self.pipeline = None
        gc.collect()
        self._load()

    def _claim_next(self, control=None):
        while not self.stop_event.is_set():
            if control is not None and not control.wait_resumed(self.stop_event):
                return None
            self.wake.clear()
            if self.settings.backend == "vllm" and self.settings.ar_concurrency > 1:
                # Briefly coalesce simultaneous submissions into one vLLM scheduling wave.
                if self.stop_event.wait(self.settings.ar_batch_wait_ms / 1000):
                    return None
            claimed = self.store.claim_many(self.settings.ar_concurrency)
            if claimed:
                return claimed
            self.wake.wait(0.5)
        return None

    def _claim_and_prepare(self, control):
        claimed = self._claim_next(control)
        if claimed is None:
            return None
        contexts = [self._context(job, request) for job, request in claimed]
        if not all(self._parallel_ready(context) for context in contexts):
            return {"kind": "exclusive", "contexts": contexts}
        if not control.begin_ar(self.stop_event):
            return {"kind": "exclusive", "contexts": contexts}
        try:
            prepared, rebuild = self._prepare_parallel_segment(contexts)
            return {"kind": "prepared", "contexts": contexts,
                    "prepared": prepared, "rebuild": rebuild}
        finally:
            control.finish_ar()

    def _cancel_wave(self, wave):
        if wave is None:
            return
        for context in wave["contexts"]:
            with self.lock:
                active = context["job"]["id"] in self.active
            if active:
                self._handle_error(context, InterruptedError("Worker stopped"))
                self._release_context(context)

    def _quiesce_prefetch(self, control, future):
        control.cancel_and_wait()
        self.wake.set()
        return future.result() if future is not None else None

    def _run_barrier(self):
        while not self.stop_event.is_set():
            claimed = self._claim_next()
            if claimed is None:
                return
            rebuild = self._execute_claimed(claimed)
            if rebuild and not self.stop_event.is_set():
                self._reload_pipeline()

    def _run_overlap(self):
        control = self.overlap_control = _AROverlapControl()
        pending, future = None, None
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="yue2-ar-prefetch") as prefetch:
                future = prefetch.submit(self._claim_and_prepare, control)
                while not self.stop_event.is_set():
                    wave = pending if pending is not None else future.result()
                    pending, future = None, None
                    if wave is None:
                        break
                    if wave["kind"] == "exclusive":
                        deadline = time.monotonic() + self.settings.task_timeout_seconds
                        for context in wave["contexts"]:
                            context["deadline"] = deadline
                        rebuild = self._execute_contexts(wave["contexts"])
                        if rebuild and not self.stop_event.is_set():
                            self._reload_pipeline()
                        if not self.stop_event.is_set():
                            future = prefetch.submit(self._claim_and_prepare, control)
                        continue

                    # Do not prefetch through a known-bad AR wave before rebuilding.
                    if wave["rebuild"]:
                        rebuild = self._render_prepared_segment(
                            wave["contexts"], wave["prepared"], None)
                        if (rebuild or wave["rebuild"]) and not self.stop_event.is_set():
                            self._reload_pipeline()
                        if not self.stop_event.is_set():
                            future = prefetch.submit(self._claim_and_prepare, control)
                        continue

                    future = prefetch.submit(self._claim_and_prepare, control)
                    rebuild = self._render_prepared_segment(
                        wave["contexts"], wave["prepared"], control)
                    if rebuild and not self.stop_event.is_set():
                        pending = self._quiesce_prefetch(control, future)
                        future = None
                        try:
                            self._reload_pipeline()
                        except BaseException:
                            self._cancel_wave(pending)
                            pending = None
                            raise
                        control.restart()
                    elif control.paused:
                        control.resume()
                    if pending is None and future is None and not self.stop_event.is_set():
                        future = prefetch.submit(self._claim_and_prepare, control)
                if future is not None:
                    self._cancel_wave(future.result())
                self._cancel_wave(pending)
        finally:
            control.resume()
            self.overlap_control = None

    def _run(self):
        try:
            self._load()
            if (self.settings.ar_nar_overlap and self.settings.backend == "vllm"
                    and self.settings.ar_concurrency > 1):
                self._run_overlap()
            else:
                self._run_barrier()
        except Exception:
            if not self.stop_event.is_set():
                self.startup_error = True
                log.exception("Inference worker unavailable")
        finally:
            self.ready = False
            if self.pipeline is not None:
                self.pipeline.close()

    def _context(self, job, request):
        job_id, cancel_event = job["id"], threading.Event()
        with self.lock:
            self.active[job_id] = cancel_event
        if self.store.get(job_id)["cancel_requested"]:
            cancel_event.set()
        context = {
            "job": job, "request": request, "cancel_event": cancel_event,
            "deadline": time.monotonic() + self.settings.task_timeout_seconds,
            "counts": {"abc": 0, "semantic": 0}, "last_report": 0.0,
            "pipeline": {"ar_nar_overlap": bool(self.settings.ar_nar_overlap),
                         "ar_active_at_acoustic_start": False,
                         "pressure_waited": False},
        }

        def cancelled():
            return (cancel_event.is_set() or self.stop_event.is_set()
                    or time.monotonic() >= context["deadline"])

        def stage(name):
            if cancelled():
                raise InterruptedError("Cancelled at stage boundary")
            self.store.progress(job_id, stage=name, tokens=dict(context["counts"]))

        def token(phase, value):
            context["counts"][phase] = context["counts"].get(phase, 0) + 1
            now = time.monotonic()
            if now - context["last_report"] >= 1:
                self.store.progress(job_id, tokens=dict(context["counts"]))
                context["last_report"] = now

        context.update(cancelled=cancelled, stage=stage, token=token)
        return context

    def _save_result(self, context, result):
        job_id = context["job"]["id"]
        context["stage"]("saving")
        output = self.root / "artifacts" / job_id
        output.mkdir(parents=True, exist_ok=False)
        if isinstance(getattr(result, "config", None), dict):
            result.config["service_pipeline"] = dict(context["pipeline"])
        saved = result.save_artifacts(output)
        if context["cancelled"]():
            raise InterruptedError("Cancelled after saving")
        truncated = any(saved["truncated"].values())
        self.store.finish(job_id, "truncated" if truncated else "succeeded", result={
            "audio_url": f"/v1/jobs/{job_id}/audio",
            "score_url": f"/v1/jobs/{job_id}/score" if result.abc else None,
            "audio_seconds": saved["audio_seconds"], "sample_rate": saved["sample_rate"],
            "truncated": saved["truncated"], "timing": saved["timing"],
            "configuration": result.config,
        })

    def _handle_error(self, context, error):
        job_id = context["job"]["id"]
        if isinstance(error, InterruptedError):
            if (time.monotonic() >= context["deadline"] and not context["cancel_event"].is_set()
                    and not self.stop_event.is_set()):
                self.store.finish(job_id, "failed", error={
                    "code": "timeout", "message": "Generation exceeded its execution deadline."})
            else:
                self.store.finish(job_id, "cancelled")
            return False
        # Do not pass the exception object to logging handlers: handlers/mocks
        # may retain traceback frames and the tensors referenced by them.
        log.exception("Generation failed: job=%s", job_id)
        invalid = isinstance(error, (ValueError, TypeError))
        self.store.finish(job_id, "failed", error={
            "code": "invalid_generation" if invalid else "inference_failed",
            "message": ("Input could not be generated; check score and context length." if invalid
                        else "Inference failed; see worker logs with this job ID."),
        })
        return not invalid

    def _release_context(self, context):
        with self.lock:
            self.active.pop(context["job"]["id"], None)

    def _melody_transcriber(self):
        if self.transcriber is None:
            with self._transcriber_lock:
                if self.transcriber is None:
                    from .cover import MelodyTranscriber
                    self.transcriber = MelodyTranscriber(
                        self.settings.sheetsage, revision=self.settings.sheetsage_revision,
                        device=self.settings.sheetsage_device,
                        local_files_only=self.settings.local_files_only,
                        min_free_gib=self.settings.sheetsage_min_free_gib)
        return self.transcriber

    def _cover_audio_file(self, raw):
        path = Path(raw)
        root = (self.root / "covers").resolve()
        if path.is_symlink():
            raise ValueError("Cover audio is missing")
        resolved = path.resolve()
        if resolved.parent != root or resolved.suffix != ".bin" or not resolved.is_file():
            raise ValueError("Cover audio is missing")
        return resolved

    def _transcribe_cover(self, context):
        request = context["request"]
        raw = request.get(COVER_AUDIO_KEY)
        if not raw:
            return
        context["stage"]("transcribing")
        if context["cancelled"]():
            raise InterruptedError("Cancelled before transcription")
        path = self._cover_audio_file(raw)
        abc = self._melody_transcriber().transcribe(path)
        request.pop(COVER_AUDIO_KEY, None)
        request.pop(COVER_SHA_KEY, None)
        request["abc"] = abc
        request["cot"] = "melody"

    def _parallel_ready(self, context):
        supports = all(hasattr(self.pipeline, name) for name in (
            "parallel_ar_eligible", "generate_ar", "render_ar"))
        if not supports or context["request"].get(COVER_AUDIO_KEY):
            return False
        return self.pipeline.parallel_ar_eligible(generation_fields(context["request"]))

    def _execute_serial(self, context):
        try:
            if context["cancelled"]():
                raise InterruptedError("Cancelled before generation")
            self._transcribe_cover(context)
            if context["cancelled"]():
                raise InterruptedError("Cancelled before generation")
            result = self.pipeline(**generation_fields(context["request"]), cancelled=context["cancelled"],
                                   on_token=context["token"], on_stage=context["stage"])
            self._save_result(context, result)
            return False
        except Exception as error:
            return self._handle_error(context, error)
        finally:
            self._release_context(context)

    def _pause_for_pressure(self, control, contexts):
        if control is None:
            return False
        for context in contexts:
            context["pipeline"]["pressure_waited"] = True
        control.pause_and_wait()
        return True

    def _mark_acoustic_start(self, control, contexts):
        active = control is not None and control.active()
        for context in contexts:
            context["pipeline"]["ar_active_at_acoustic_start"] |= active

    def _prepare_parallel_segment(self, contexts):
        rebuild = False
        prepared = {}
        with ThreadPoolExecutor(max_workers=min(len(contexts), self.settings.ar_concurrency),
                                thread_name_prefix="yue2-ar") as executor:
            futures = {
                context["job"]["id"]: executor.submit(
                    self.pipeline.generate_ar, **generation_fields(context["request"]),
                    cancelled=context["cancelled"], on_token=context["token"],
                    on_stage=context["stage"])
                for context in contexts
            }
            # The segment's AR barrier lets vLLM continuously batch before NAR.
            for context in contexts:
                try:
                    prepared[context["job"]["id"]] = futures[context["job"]["id"]].result()
                except Exception as error:
                    rebuild |= self._handle_error(context, error)
                    self._release_context(context)
        return prepared, rebuild

    def _render_prepared(self, context, ar_result, control=None):
        self._mark_acoustic_start(control, [context])
        try:
            result = None
            for attempt in range(2 if control is not None else 1):
                try:
                    result = self.pipeline.render_ar(
                        ar_result, cancelled=context["cancelled"], on_stage=context["stage"])
                except MemoryError:
                    if attempt == 0 and self._pause_for_pressure(control, [context]):
                        continue
                    raise
                break
            self._save_result(context, result)
            return False
        except Exception as error:
            return self._handle_error(context, error)
        finally:
            self._release_context(context)

    def _render_nar_result(self, context, nar_result, control=None):
        try:
            result = None
            for attempt in range(2 if control is not None else 1):
                try:
                    result = self.pipeline.render_nar(
                        nar_result, cancelled=context["cancelled"], on_stage=context["stage"])
                except MemoryError:
                    if attempt == 0 and self._pause_for_pressure(control, [context]):
                        continue
                    raise
                break
            self._save_result(context, result)
            return False
        except Exception as error:
            return self._handle_error(context, error)
        finally:
            self._release_context(context)

    def _nar_admitted(self, ar_results):
        try:
            return self.pipeline.nar_batch_admission(ar_results)["allowed"]
        except (MemoryError, ValueError):
            return False

    def _render_prepared_segment(self, contexts, prepared, control=None):
        rebuild = False
        index = 0
        supports_nar_batch = all(hasattr(self.pipeline, name) for name in (
            "nar_batch_admission", "generate_nar_batch", "render_nar"))
        while index < len(contexts):
            context = contexts[index]
            job_id = context["job"]["id"]
            if job_id not in prepared:
                index += 1
                continue

            window = []
            for candidate in contexts[index:index + self.settings.nar_batch_size]:
                candidate_id = candidate["job"]["id"]
                if candidate_id not in prepared:
                    break
                window.append(candidate)
            batch_count = len(window)
            if not supports_nar_batch:
                batch_count = 1
            elif batch_count >= 2:
                results = [prepared[item["job"]["id"]] for item in window[:batch_count]]
                self._mark_acoustic_start(control, window[:batch_count])
                if not self._nar_admitted(results):
                    self._pause_for_pressure(control, window[:batch_count])
                if not self._nar_admitted(results):
                    batch_count = 1
            if batch_count < 2:
                if supports_nar_batch and control is not None:
                    self._mark_acoustic_start(control, [context])
                    if not self._nar_admitted([prepared[job_id]]):
                        self._pause_for_pressure(control, [context])
                rebuild |= self._render_prepared(context, prepared[job_id], control)
                index += 1
                continue

            batch = window[:batch_count]
            batch_args = {
                "cancelled": [item["cancelled"] for item in batch],
                "on_stage": [item["stage"] for item in batch],
            }
            try:
                ar_results = [prepared[item["job"]["id"]] for item in batch]
                try:
                    nar_results = self.pipeline.generate_nar_batch(ar_results, **batch_args)
                except MemoryError:
                    if not self._pause_for_pressure(control, batch):
                        raise
                    nar_results = self.pipeline.generate_nar_batch(ar_results, **batch_args)
            except MemoryError:
                for item in batch:
                    rebuild |= self._render_prepared(
                        item, prepared[item["job"]["id"]], control)
                index += batch_count
                continue
            except Exception as error:
                for item in batch:
                    rebuild |= self._handle_error(item, error)
                    self._release_context(item)
                index += batch_count
                continue
            for item, nar_result in zip(batch, nar_results):
                if isinstance(nar_result, Exception):
                    try:
                        raise nar_result
                    except Exception as error:
                        rebuild |= self._handle_error(item, error)
                    finally:
                        self._release_context(item)
                else:
                    rebuild |= self._render_nar_result(item, nar_result, control)
            index += batch_count
        return rebuild

    def _execute_parallel_segment(self, contexts):
        prepared, rebuild = self._prepare_parallel_segment(contexts)
        return rebuild | self._render_prepared_segment(contexts, prepared)

    def _execute_contexts(self, contexts):
        rebuild, index = False, 0
        while index < len(contexts):
            context = contexts[index]
            if not self._parallel_ready(context):
                rebuild |= self._execute_serial(context)
                index += 1
                continue
            end = index + 1
            while end < len(contexts) and self._parallel_ready(contexts[end]):
                end += 1
            rebuild |= self._execute_parallel_segment(contexts[index:end])
            index = end
        return rebuild

    def _execute_claimed(self, claimed):
        contexts = [self._context(job, request) for job, request in claimed]
        return self._execute_contexts(contexts)


class BodyLimit:
    """Bound request buffering before validation, including chunked requests."""
    def __init__(self, app, limit=JSON_BODY_LIMIT, cover_limit=COVER_BODY_LIMIT):
        self.app, self.limit, self.cover_limit = app, limit, cover_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        cover = scope.get("path") == "/v1/covers"
        limit = self.cover_limit if cover else self.limit
        detail = "Request body exceeds 40 MiB" if cover else "Request body exceeds 256 KiB"
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > limit:
                return await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(settings=None, pipeline_factory=None, transcriber=None):
    settings = settings or Settings.from_env()
    worker = JobWorker(settings, pipeline_factory or build_pipeline, transcriber=transcriber)

    @asynccontextmanager
    async def lifespan(app):
        worker.start()
        async def publish_load():
            while True:
                snapshot = load()
                path = worker.root / "load.json"
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(snapshot))
                tmp.chmod(0o640)
                tmp.replace(path)
                await asyncio.sleep(1)
        async def maintain_artifacts():
            while True:
                try:
                    result = await asyncio.to_thread(worker.cleanup_artifacts)
                    if result["removed"]:
                        log.info("Removed %d expired artifact directories (%.2f GiB)",
                                 result["removed"], result["bytes"] / 1024 ** 3)
                except Exception:
                    log.exception("Artifact cleanup failed")
                await asyncio.sleep(settings.artifact_cleanup_interval_seconds)
        publisher = asyncio.create_task(publish_load())
        janitor = asyncio.create_task(maintain_artifacts())
        try:
            yield
        finally:
            janitor.cancel()
            with suppress(asyncio.CancelledError):
                await janitor
            publisher.cancel()
            with suppress(asyncio.CancelledError):
                await publisher
            await asyncio.to_thread(worker.stop)

    app = FastAPI(title="Noiz YuE2", version="0.1.0", lifespan=lifespan)
    app.state.worker = worker
    app.add_middleware(BodyLimit)
    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    def get_job(job_id):
        job = worker.store.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        if not worker.ready or (worker.root / "draining").exists():
            return JSONResponse({"status": "failed" if worker.startup_error else "loading"}, status_code=503)
        return {"status": "ready"}

    @app.get("/internal/load", dependencies=[Depends(authorize)])
    def load():
        return {**worker.store.load_snapshot(), "schema_version": 1,
                "boot_id": worker.boot_id,
                "ready": worker.ready and not (worker.root / "draining").exists(),
                "capacity": settings.max_pending, "ar_concurrency": settings.ar_concurrency,
                "nar_batch_size": settings.nar_batch_size}

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(authorize)])
    def submit(request: GenerateRequest, idempotency_key: str | None = Header(default=None, min_length=1, max_length=128), x_admission_id: str | None = Header(default=None, max_length=128)):
        if (worker.root / "draining").exists() or not worker.ready or worker.stop_event.is_set():
            raise HTTPException(503, "Inference worker is not ready", headers={"Retry-After": "5"})
        try:
            job, created = worker.store.submit(request.model_dump(exclude={"n"}), settings.max_pending, idempotency_key, n=request.n, admission_id=x_admission_id)
        except QueueFull:
            raise HTTPException(429, "Queue is full", headers={"Retry-After": "5"}) from None
        except IdempotencyConflict:
            raise HTTPException(409, "Idempotency-Key was already used with different input") from None
        worker.wake.set()
        return JSONResponse(job, status_code=202 if created else 200, headers={"Location": f"/v1/jobs/{job['id']}"})

    @app.post("/v1/covers", status_code=202, dependencies=[Depends(authorize)])
    async def cover(
            audio: UploadFile = File(),
            style: str = Form(),
            lyrics: str = Form(),
            seed: int = Form(default=831001),
            cfg_scale: float | None = Form(default=None),
            idempotency_key: str | None = Header(default=None, min_length=1, max_length=128),
            x_admission_id: str | None = Header(default=None, max_length=128)):
        if (worker.root / "draining").exists() or not worker.ready or worker.stop_event.is_set():
            raise HTTPException(503, "Inference worker is not ready", headers={"Retry-After": "5"})
        if settings.sheetsage_device == "off":
            raise HTTPException(503, COVER_DISABLED)
        payload = await audio.read()
        if not payload:
            raise HTTPException(422, "Audio file is empty")
        if len(payload) > COVER_BODY_LIMIT:
            raise HTTPException(413, "Request body exceeds 40 MiB")
        try:
            public = GenerateRequest(style=style, lyrics=lyrics, cot="melody", seed=seed, cfg_scale=cfg_scale)
        except ValidationError as exc:
            message = exc.errors()[0]["msg"] if exc.errors() else "Invalid cover request"
            raise HTTPException(422, message) from None
        digest = hashlib.sha256(payload).hexdigest()
        covers = worker.root / "covers"
        covers.mkdir(parents=True, exist_ok=True)
        path = covers / f"{digest}.bin"
        if not path.exists():
            temporary = covers / f".{digest}.{uuid.uuid4().hex}.tmp"
            temporary.write_bytes(payload)
            temporary.replace(path)
        request = public.model_dump(exclude={"n"})
        request[COVER_AUDIO_KEY] = str(path)
        request[COVER_SHA_KEY] = digest
        try:
            job, created = worker.store.submit(
                request, settings.max_pending, idempotency_key, admission_id=x_admission_id)
        except QueueFull:
            raise HTTPException(429, "Queue is full", headers={"Retry-After": "5"}) from None
        except IdempotencyConflict:
            raise HTTPException(409, "Idempotency-Key was already used with different input") from None
        worker.wake.set()
        return JSONResponse(job, status_code=202 if created else 200, headers={"Location": f"/v1/jobs/{job['id']}"})

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def status(job_id: str):
        return get_job(job_id)

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(authorize)])
    def cancel(job_id: str):
        job = worker.cancel(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    def artifact(job_id, name, media):
        job = get_job(job_id)
        if job["status"] not in {"succeeded", "truncated"}:
            raise HTTPException(409, "Artifact is not available for this job state")
        path = worker.root / "artifacts" / job["id"] / name
        if not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, media_type=media, filename=f"{job['id']}-{name}")

    @app.get("/v1/jobs/{job_id}/audio", dependencies=[Depends(authorize)])
    def audio(job_id: str):
        return artifact(job_id, "audio.flac", "audio/flac")

    @app.get("/v1/jobs/{job_id}/score", dependencies=[Depends(authorize)])
    def score(job_id: str):
        return artifact(job_id, "score.abc", "text/plain; charset=utf-8")

    return app


def main():
    import uvicorn
    uvicorn.run("yue2.service:create_app", factory=True,
                host=os.environ.get("YUE2_HOST", "127.0.0.1"),
                port=int(os.environ.get("YUE2_PORT", "8000")), workers=1)


if __name__ == "__main__":
    main()
