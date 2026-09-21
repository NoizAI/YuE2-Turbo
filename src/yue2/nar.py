"""Memory-bounded acoustic flow matching with one AR prefill per original chunk.

Only PyTorch is required. The reference 32-step midpoint solver, full-song CPU
FP32 noise draw, boundary positions, and original context chunks are preserved.
Attention query tiling changes temporary storage, never the visible key set.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .protocol import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END, chunk_ranges


@dataclass
class Chunk:
    ar_tokens: list[int]
    noise: torch.Tensor
    nar_cond_end: int = 0


def _integers(values, name):
    result = list(values)
    if not result or any(isinstance(v, bool) or not isinstance(v, Integral) for v in result):
        raise ValueError(f"{name} must be a nonempty sequence of integer token IDs")
    return [int(v) for v in result]


def song_chunks(prefix, codec, seed, context=CONTEXT):
    """Draw the complete noise tensor once, then take views at historical cuts."""
    prefix = _integers(prefix, "prefix")
    codec = _integers(codec, "codec")
    if min(prefix) < 0 or min(codec) < 0 or max(codec) >= CODEC_SIZE:
        raise ValueError("Token IDs are outside their allowed vocabulary")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(context, bool) or not isinstance(context, Integral) or not 1 <= context <= CONTEXT:
        raise ValueError(f"context must be an integer in 1..{CONTEXT}")
    ranges = chunk_ranges(len(codec), len(prefix), int(context))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)
    return [Chunk(prefix + [value + CODEC_OFFSET for value in codec[a:b]] + [MUSIC_END], noise[a:b])
            for a, b in ranges]


def attention(q, k, v, *, causal=False, backend="sdpa", query_chunk_size=None):
    """Attend [tokens, heads, dim] tensors without materializing a song mask.

    CPU/MPS bound the number of query rows for a potential math SDPA fallback.
    CUDA normally uses PyTorch's fused SDPA without an external flash package.
    """
    if backend not in {"sdpa", "math", "flash"}:
        raise ValueError("attention must be sdpa, math, or flash")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Expected Q/K/V [tokens, heads, dim] with matching K/V")
    if min(q.shape) < 1 or min(k.shape) < 1 or q.shape[1] % k.shape[1]:
        raise ValueError("Invalid attention lengths or grouped-query head count")
    if causal and len(q) != len(k):
        raise ValueError("Causal prefill requires matching Q/K sequence lengths")
    if backend == "flash" and q.device.type != "cuda":
        raise ValueError("Explicit flash SDPA requires a CUDA device")
    if query_chunk_size is not None and (isinstance(query_chunk_size, bool) or
                                        not isinstance(query_chunk_size, Integral) or query_chunk_size < 1):
        raise ValueError("query_chunk_size must be a positive integer")
    block = query_chunk_size or (len(q) if q.device.type == "cuda" and backend != "math" else 256)
    query = q.transpose(0, 1).unsqueeze(0)
    key = k.transpose(0, 1).unsqueeze(0)
    value = v.transpose(0, 1).unsqueeze(0)
    grouped = query.shape[1] != key.shape[1]
    if grouped and q.device.type == "mps":
        groups = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
        grouped = False
    context = nullcontext()
    if backend != "sdpa":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        context = sdpa_kernel(SDPBackend.MATH if backend == "math" else SDPBackend.FLASH_ATTENTION)
    outputs = []
    with context:
        for start in range(0, len(q), block):
            end = min(start + block, len(q))
            used_key = key[..., :end, :] if causal else key
            used_value = value[..., :end, :] if causal else value
            # is_causal on a rectangular Q/K uses an upper-left triangle, so a
            # later query block needs its absolute query positions explicitly.
            mask = None
            if causal and start:
                mask = (torch.arange(end, device=q.device)[None, :] <=
                        torch.arange(start, end, device=q.device)[:, None])
            outputs.append(F.scaled_dot_product_attention(
                query[..., start:end, :], used_key, used_value,
                attn_mask=mask, is_causal=causal and start == 0, enable_gqa=grouped,
            ))
    return torch.cat(outputs, dim=-2)[0].transpose(0, 1)


class CachedNAR:
    """One original acoustic chunk; AR prefix KV is invariant during the ODE."""

    def __init__(self, model, chunk: Chunk, attention="sdpa", query_chunk_size=None):
        self.model, self.chunk = model, chunk
        self.backend, self.query_chunk_size = attention, query_chunk_size
        weight = next(model.vae2llm.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        if chunk.noise.ndim != 2 or chunk.noise.shape[1] != 64 or len(chunk.noise) < 1:
            raise ValueError("Expected nonempty acoustic noise [frames,64]")
        if not torch.isfinite(chunk.noise).all():
            raise ValueError("Acoustic noise contains non-finite values")
        self.ar_length, self.nar_length = len(chunk.ar_tokens), len(chunk.noise) + 2
        if self.ar_length < 1 or min(chunk.ar_tokens) < 0 or max(chunk.ar_tokens) >= model.config.vocab_size:
            raise ValueError("AR prefix is empty or outside the model vocabulary")
        if self.ar_length + self.nar_length > model.config.max_position_embeddings:
            raise ValueError("Original acoustic chunk exceeds the model context")
        if chunk.nar_cond_end < 0:
            raise ValueError("nar_cond_end must be nonnegative")
        self.visible_length = min(chunk.nar_cond_end, self.ar_length) if chunk.nar_cond_end else self.ar_length
        positions = torch.arange(self.ar_length, self.ar_length + self.nar_length, device=self.device)[None]
        self.cos, self.sin = model.model.rotary_emb(positions)
        local = torch.arange(self.nar_length, device=self.device).clamp(max=model.config.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.cache = []
        self._prefill()

    def _attention(self, q, k, v, causal=False):
        return attention(q, k, v, causal=causal, backend=self.backend, query_chunk_size=self.query_chunk_size)

    @torch.inference_mode()
    def _prefill(self):
        backbone = self.model.model
        ids = torch.tensor([self.chunk.ar_tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(self.ar_length, device=self.device)[None]
        cos, sin = backbone.rotary_emb(positions)
        x = backbone.embed_tokens(ids)
        for layer in backbone.layers:
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            # Clone only for restricted visibility; a slice would retain the
            # storage of invisible codec tokens for every layer.
            cached = (k[0, :self.visible_length], v[0, :self.visible_length])
            if self.visible_length != self.ar_length:
                cached = tuple(t.clone() for t in cached)
            self.cache.append(cached)
            h = self._attention(q[0], k[0], v[0], causal=True)
            x = x + layer.self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.mlp(layer.post_attention_layernorm(x))

    @torch.inference_mode()
    def velocity(self, state, raw_t):
        model = self.model
        if tuple(state.shape) != tuple(self.chunk.noise.shape):
            raise ValueError("ODE state shape changed")
        x_nar = F.pad(state, (0, 0, 1, 1))
        shifted = model._shift_t_value(raw_t, self.device, self.dtype)
        x = model.vae2llm(x_nar[None])
        x = x + model.time_embedder(shifted.expand(self.nar_length))[None]
        x = x + self.pos_emb
        for layer, (ar_k, ar_v) in zip(model.model.layers, self.cache):
            q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), self.cos, self.sin)
            k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
            h = self._attention(q[0], k, v)
            x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        return model.llm2vae(model.model.norm(x))[0, 1:-1]

    @torch.inference_mode()
    def solve(self, steps=32, cancelled: Callable[[], bool] | None = None,
              on_progress: Callable[[int, int], None] | None = None):
        """Solve a chunk, reporting each submitted midpoint step without syncing.

        CUDA work may still be executing when ``on_progress`` runs. The existing
        CPU result transfer completes that work before this method returns.
        Callback exceptions propagate to the caller.
        """
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        state = self.chunk.noise.to(device=self.device, dtype=self.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            raw = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            first = self.velocity(state, raw)
            mid = state - first * (dt / 2)
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            state = state - self.velocity(mid, raw_mid) * dt
            if on_progress is not None:
                on_progress(step + 1, int(steps))
        result = state.float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self):
        self.cache.clear()
        self.cos = self.sin = self.pos_emb = None


def _pad_first_dim(tensors):
    if not tensors:
        raise ValueError("Expected at least one tensor")
    width = max(len(tensor) for tensor in tensors)
    result = tensors[0].new_zeros((len(tensors), width, *tensors[0].shape[1:]))
    for row, tensor in enumerate(tensors):
        result[row, :len(tensor)] = tensor
    return result


def _batched_attention(q, ar_k, ar_v, nar_k, nar_v, query_lengths, ar_lengths,
                       backend="sdpa", query_chunk_size=None):
    """Row-wise unmasked SDPA avoids masked-GQA's quadratic fallback on CUDA."""
    if (q.ndim != 4 or ar_k.ndim != 4 or nar_k.ndim != 4
            or ar_v.shape != ar_k.shape or nar_v.shape != nar_k.shape):
        raise ValueError("Expected batched Q/K/V [batch,tokens,heads,dim]")
    if len(query_lengths) != len(q) or len(ar_lengths) != len(q):
        raise ValueError("Attention lengths must align with batch rows")
    result = torch.zeros_like(q)
    for row, (query_length, ar_length) in enumerate(zip(query_lengths, ar_lengths)):
        key = torch.cat((ar_k[row, :ar_length], nar_k[row, :query_length]))
        value = torch.cat((ar_v[row, :ar_length], nar_v[row, :query_length]))
        result[row, :query_length] = attention(
            q[row, :query_length], key, value, backend=backend,
            query_chunk_size=query_chunk_size)
    return result


class BatchedCachedNAR:
    """Padded NAR rows with serial prefix prefill and shared ODE forwards."""

    def __init__(self, model, chunks, attention="sdpa", query_chunk_size=None):
        if len(chunks) < 2:
            raise ValueError("BatchedCachedNAR requires at least two chunks")
        if attention not in {"sdpa", "math", "flash"}:
            raise ValueError("attention must be sdpa, math, or flash")
        self.model, self.chunks = model, list(chunks)
        self.backend, self.query_chunk_size = attention, query_chunk_size
        weight = next(model.vae2llm.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        engines = []
        try:
            engines = [CachedNAR(model, chunk, attention, query_chunk_size) for chunk in self.chunks]
            self.frame_lengths = [len(chunk.noise) for chunk in self.chunks]
            self.nar_lengths = [engine.nar_length for engine in engines]
            self.max_frames, self.max_nar = max(self.frame_lengths), max(self.nar_lengths)
            self.cos = _pad_first_dim([engine.cos[0] for engine in engines])
            self.sin = _pad_first_dim([engine.sin[0] for engine in engines])
            self.pos_emb = _pad_first_dim([engine.pos_emb[0] for engine in engines])
            visible = [len(engine.cache[0][0]) for engine in engines]
            self.cache = []
            for layer in range(len(engines[0].cache)):
                self.cache.append((
                    _pad_first_dim([engine.cache[layer][0] for engine in engines]),
                    _pad_first_dim([engine.cache[layer][1] for engine in engines]),
                ))
                for engine in engines:
                    engine.cache[layer] = (None, None)
            frame_positions = torch.arange(self.max_frames, device=self.device)
            self.frame_valid = frame_positions[None] < torch.tensor(
                self.frame_lengths, device=self.device)[:, None]
            self.ar_lengths = visible
        finally:
            for engine in engines:
                engine.close()

    @torch.inference_mode()
    def velocity(self, state, raw_t):
        expected = (len(self.chunks), self.max_frames, 64)
        if tuple(state.shape) != expected:
            raise ValueError(f"Expected padded batch state {expected}")
        model = self.model
        x_nar = state.new_zeros((len(self.chunks), self.max_nar, 64), dtype=self.dtype)
        x_nar[:, 1:1 + self.max_frames] = state.to(self.dtype)
        shifted = model._shift_t_value(raw_t, self.device, self.dtype)
        x = model.vae2llm(x_nar)
        x = x + model.time_embedder(shifted.expand(len(self.chunks), self.max_nar))
        x = x + self.pos_emb
        for layer, (ar_k, ar_v) in zip(model.model.layers, self.cache):
            q, k, v = layer.nar_self_attn.project_qkv(
                layer.nar_input_layernorm(x), self.cos, self.sin)
            h = _batched_attention(
                q, ar_k, ar_v, k, v, self.nar_lengths, self.ar_lengths,
                self.backend, self.query_chunk_size)
            x = x + layer.nar_self_attn.o_proj(h.flatten(2))
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        result = model.llm2vae(model.model.norm(x))[:, 1:1 + self.max_frames]
        return result.masked_fill(~self.frame_valid[..., None], 0)

    @torch.inference_mode()
    def solve(self, steps=32, cancelled=None, on_progress=None):
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        callbacks = list(cancelled or [None] * len(self.chunks))
        progress = list(on_progress or [None] * len(self.chunks))
        if len(callbacks) != len(self.chunks) or len(progress) != len(self.chunks):
            raise ValueError("Callbacks must align with batch rows")
        state = _pad_first_dim([
            chunk.noise.to(device=self.device, dtype=self.dtype) for chunk in self.chunks])
        active = [not (callback is not None and callback()) for callback in callbacks]
        dt = 1.0 / steps
        for step in range(steps):
            if not any(active):
                break
            t = 1.0 - step * dt
            raw = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            first = self.velocity(state, raw)
            update = (torch.tensor(active, device=self.device)[:, None]
                      & self.frame_valid)[..., None]
            midpoint = torch.where(update, state - first * (dt / 2), state)
            for row, callback in enumerate(callbacks):
                if active[row] and callback is not None and callback():
                    active[row] = False
            if not any(active):
                break
            raw_mid = torch.logit(
                torch.tensor(t - dt / 2, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            second = self.velocity(midpoint, raw_mid)
            update = (torch.tensor(active, device=self.device)[:, None]
                      & self.frame_valid)[..., None]
            state = torch.where(update, state - second * dt, state)
            for row, callback in enumerate(progress):
                if active[row] and callback is not None:
                    callback(step + 1, int(steps))
        results = []
        for row, length in enumerate(self.frame_lengths):
            result = state[row, :length].float().cpu() if active[row] else None
            if result is not None and not torch.isfinite(result).all():
                raise FloatingPointError("Acoustic flow matching produced non-finite latents")
            results.append(result)
        return results

    def close(self):
        self.cache.clear()
        self.cos = self.sin = self.pos_emb = None
        self.frame_valid = None


def nar_batch_memory_estimate(model, songs, reserve_gib=4):
    """Padded-cache admission bounded by physical and process memory limits."""
    chunks = [chunk for song in songs for chunk in song]
    batch = len(songs)
    if not chunks or batch < 1:
        raise ValueError("Expected nonempty song chunks")
    config = model.config
    dtype_bytes = next(model.parameters()).element_size()
    max_visible = max(
        min(chunk.nar_cond_end, len(chunk.ar_tokens)) if chunk.nar_cond_end
        else len(chunk.ar_tokens) for chunk in chunks)
    max_nar = max(len(chunk.noise) + 2 for chunk in chunks)
    cache_bytes = (2 * config.num_hidden_layers * batch * max_visible
                   * config.num_key_value_heads * config.head_dim * dtype_bytes)
    qkv = (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim
    activation_width = qkv + 4 * config.hidden_size + 2 * config.intermediate_size
    activation_bytes = batch * max_nar * activation_width * dtype_bytes
    required = int(cache_bytes * 1.2 + activation_bytes * 2 + 2 * 2**30)
    result = {
        "batch_size": batch, "max_visible_tokens": max_visible, "max_nar_tokens": max_nar,
        "cache_bytes": cache_bytes, "activation_bytes": activation_bytes,
        "required_bytes": required, "reserve_bytes": int(reserve_gib * 2**30),
    }
    device = next(model.parameters()).device
    if device.type != "cuda":
        return dict(result, available_bytes=None, allowed=True)
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reclaimable = max(torch.cuda.memory_reserved(device) - allocated, 0)
    fraction = torch.cuda.get_per_process_memory_fraction(device)
    process_available = total * fraction - allocated - 2**30
    physical_available = free + reclaimable - result["reserve_bytes"]
    available = min(process_available, physical_available)
    return dict(result, available_bytes=available, free_bytes=free,
                allocated_bytes=allocated, reclaimable_cache_bytes=reclaimable,
                process_limit_bytes=int(total * fraction),
                allowed=required <= available)


@torch.inference_mode()
def synthesize_batch(model, prefixes, codecs, seeds, steps=32, context=CONTEXT,
                     attention="sdpa", cancelled=None, query_chunk_size=None,
                     on_progress=None):
    """Batch FIFO songs without length sorting; return latents or row cancellation."""
    prefixes, codecs, seeds = list(prefixes), list(codecs), list(seeds)
    if not prefixes or not (len(prefixes) == len(codecs) == len(seeds)):
        raise ValueError("Batch inputs must be nonempty and aligned")
    callbacks = list(cancelled or [None] * len(prefixes))
    progress = list(on_progress or [None] * len(prefixes))
    if len(callbacks) != len(prefixes) or len(progress) != len(prefixes):
        raise ValueError("Callbacks must align with songs")
    songs = [song_chunks(prefix, codec, seed, context)
             for prefix, codec, seed in zip(prefixes, codecs, seeds)]
    completed, outputs = [0] * len(songs), [[] for _ in songs]
    stopped = [False] * len(songs)
    while any(completed[row] < len(songs[row]) and not stopped[row] for row in range(len(songs))):
        rows = [row for row in range(len(songs))
                if completed[row] < len(songs[row]) and not stopped[row]]
        for row in list(rows):
            if callbacks[row] is not None and callbacks[row]():
                stopped[row] = True
                rows.remove(row)
        if not rows:
            break
        chunks = [songs[row][completed[row]] for row in rows]
        row_progress = []
        for row in rows:
            offset, total = completed[row] * steps, len(songs[row]) * steps
            callback = progress[row]
            row_progress.append(
                None if callback is None
                else lambda done, _steps, callback=callback, offset=offset, total=total:
                    callback(offset + done, total))
        if len(rows) == 1:
            row = rows[0]
            engine = CachedNAR(model, chunks[0], attention, query_chunk_size)
            try:
                result = engine.solve(steps, callbacks[row], row_progress[0])
            except InterruptedError:
                stopped[row] = True
            else:
                outputs[row].append(result)
                completed[row] += 1
            finally:
                engine.close()
            continue
        engine = BatchedCachedNAR(model, chunks, attention, query_chunk_size)
        try:
            results = engine.solve(
                steps, [callbacks[row] for row in rows], row_progress)
            for row, result in zip(rows, results):
                if result is None:
                    stopped[row] = True
                else:
                    outputs[row].append(result)
                    completed[row] += 1
        finally:
            engine.close()
    return [
        InterruptedError("Cancelled during acoustic flow matching") if stopped[row]
        else torch.cat(outputs[row], dim=0)
        for row in range(len(songs))
    ]


@contextmanager
def _offload_ar(model, enabled):
    """Temporarily move unused AR modules; this model cannot serve concurrently."""
    modules = [model.model.embed_tokens, model.lm_head]
    for layer in model.model.layers:
        modules.extend((layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp))
    moved = []
    try:
        if enabled:
            for module in modules:
                device = next(module.parameters()).device
                if device.type != "cpu":
                    module.to(device="cpu")
                    moved.append((module, device))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        yield
    finally:
        for module, device in moved:
            module.to(device=device)


@torch.inference_mode()
def synthesize(model, prefix: Sequence[int], codec: Sequence[int], seed: int,
               steps=32, context=CONTEXT, attention="sdpa", offload_ar=False,
               cancelled=None, query_chunk_size=None,
               on_progress: Callable[[int, int], None] | None = None):
    """Return CPU FP32 [frames,64] latents, solving original chunks serially.

    Defaults preserve the release protocol. Explicit steps/context overrides
    belong in the caller's effective configuration record. ``offload_ar`` is
    an optional memory tradeoff and requires exclusive access to ``model``.
    Progress counts submitted midpoint steps across all original chunks; it
    introduces no device synchronization. Callback exceptions propagate after
    the current chunk's cache is released and any offloaded weights restored.
    """
    if model.training:
        raise ValueError("synthesize requires model.eval()")
    chunks = song_chunks(prefix, codec, seed, context)
    output = []
    for chunk_index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        engine = CachedNAR(model, chunk, attention, query_chunk_size)
        # Drop the prefix cache before restoring AR weights, including on
        # cancellation/failure, to keep the restoration memory peak bounded.
        with _offload_ar(model, offload_ar):
            try:
                progress = None
                if on_progress is not None:
                    def progress(completed, total):
                        on_progress(chunk_index * total + completed, total * len(chunks))
                output.append(engine.solve(steps, cancelled, on_progress=progress))
            finally:
                engine.close()
        del engine
    return torch.cat(output, dim=0)
