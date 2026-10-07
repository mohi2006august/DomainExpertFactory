"""TinyLM: a NumPy causal language model with LoRA, merge and k-bit quantisation.

This is the CPU reference student. It is tiny (under a million parameters) and will not
match a real 3B model on quality, but every factory step runs on it for real, so the
whole pipeline (guarded training, checkpoint selection, merge, post-quant gate,
memorisation probes) can be exercised in CI and on a laptop in seconds:

* Base "pretraining": PPMI-SVD word embeddings ``E`` plus a softmax readout ``W, b``
  trained on the raw domain documents and a bundled general corpus.
* Features for position t: ``[E[x_t], E[x_{t-1}], mean(E[context])]``, where context is
  the instruction (SFT) or the previous ``context_window`` tokens (language modelling).
* LoRA: ``W`` (D x V) is frozen and a low-rank update ``(alpha / r) * A @ B`` is trained
  with Adam on response tokens only. ``B`` starts at zero, so step 0 equals the base.
* Merge: ``W + (alpha / r) * A @ B`` stored in fp16 (never merged into a quantised base).
* Quantisation: group-wise symmetric round-to-nearest, bit-packed to k bits per weight.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from expert_factory.artifacts import read_json, write_json
from expert_factory.canaries import CANARY_WORDS, alternatives
from expert_factory.log import get_logger
from expert_factory.resources import data_path
from expert_factory.schemas import Canary, EvalPoint, TrainRunResult
from expert_factory.students.base import (
    Hyperparams,
    QuantizerUnavailableError,
    StudentBackend,
    StudentModel,
    TrainData,
)
from expert_factory.text import normalize

if TYPE_CHECKING:
    from expert_factory.pipeline.assets.guard import OverfitGuard

log = get_logger("student.tinylm")

SPECIALS = ("<pad>", "<unk>", "<bos>", "<sep>", "<eos>")
PAD, UNK, BOS, SEP, EOS = range(len(SPECIALS))
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'.][a-z0-9]+)*|[^\sa-z0-9]")
QUANT_BITS = {"int4": 4, "int5": 5, "int6": 6, "int8": 8}
GROUP_SIZE = 32


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text).lower())


class Vocab:
    def __init__(self, itos: Sequence[str]) -> None:
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: Sequence[str], max_size: int, required: Sequence[str] = ()) -> Vocab:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        itos = list(SPECIALS)
        seen = set(itos)
        for word in required:
            if word not in seen:
                itos.append(word)
                seen.add(word)
        for word, _ in counts.most_common():
            if len(itos) >= max_size:
                break
            if word not in seen:
                itos.append(word)
                seen.add(word)
        return cls(itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(t, UNK) for t in tokenize(text)]

    def decode(self, ids: Sequence[int]) -> str:
        text = " ".join(self.itos[i] for i in ids if i >= len(SPECIALS))
        text = re.sub(r" ([.,;:!?)%])", r"\1", text)
        text = re.sub(r"\( ", "(", text)
        return re.sub(r"(^|[.!?] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


# --------------------------------------------------------------------------- math utils
def _unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    out: np.ndarray = x / np.where(norms == 0, 1.0, norms)
    return out


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    out: np.ndarray = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    out: np.ndarray = shifted / shifted.sum(axis=-1, keepdims=True)
    return out


def randomized_svd(
    m: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    omega = rng.standard_normal((m.shape[1], k + 10)).astype(np.float32)
    y = m @ omega
    for _ in range(3):
        y = m @ (m.T @ y)
    q, _ = np.linalg.qr(y)
    u_small, s, _ = np.linalg.svd(q.T @ m, full_matrices=False)
    return (q @ u_small)[:, :k], s[:k]


def ppmi_svd_embeddings(
    seqs: Sequence[np.ndarray], vocab_size: int, dim: int, rng: np.random.Generator, window: int = 4
) -> np.ndarray:
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for seq in seqs:
        for off in range(1, window + 1):
            if len(seq) <= off:
                break
            a, b = seq[:-off], seq[off:]
            np.add.at(counts, (a, b), 1.0 / off)
            np.add.at(counts, (b, a), 1.0 / off)
    total = counts.sum()
    rows = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(counts * total / (rows * rows.T))
    ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0).astype(np.float32)
    del counts, pmi
    u, s = randomized_svd(ppmi, dim, rng)
    semantic = _unit((u * np.sqrt(s)).astype(np.float32))
    # Random directions keep tokens never seen in pretraining distinguishable.
    noise = _unit(rng.standard_normal((vocab_size, dim)).astype(np.float32))
    return _unit(semantic + 0.5 * noise).astype(np.float32)


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float, weight_decay: float = 0.0) -> None:
        self.lr, self.wd, self.t = lr, weight_decay, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for key, grad in grads.items():
            self.m[key] = b1 * self.m[key] + (1 - b1) * grad
            self.v[key] = b2 * self.v[key] + (1 - b2) * grad * grad
            m_hat = self.m[key] / (1 - b1**self.t)
            v_hat = self.v[key] / (1 - b2**self.t)
            params[key] -= self.lr * (m_hat / (np.sqrt(v_hat) + eps) + self.wd * params[key])


# --------------------------------------------------------------------------- bit packing
def pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    v = values.astype(np.uint16).ravel()
    matrix = ((v[:, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(matrix.ravel())


def unpack_bits(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    matrix = np.unpackbits(packed)[: count * bits].reshape(count, bits).astype(np.uint16)
    out: np.ndarray = (matrix << np.arange(bits, dtype=np.uint16)).sum(axis=1)
    return out


def quantize_tensor(w: np.ndarray, bits: int, group_size: int = GROUP_SIZE) -> dict[str, Any]:
    flat = w.astype(np.float32).ravel()
    pad = (-flat.size) % group_size
    groups = np.pad(flat, (0, pad)).reshape(-1, group_size)
    qmax = 2 ** (bits - 1) - 1
    scales = (np.abs(groups).max(axis=1) / qmax).astype(np.float16)
    scales[scales == 0] = np.float16(1e-4)
    q = np.clip(np.rint(groups / scales.astype(np.float32)[:, None]), -qmax, qmax)
    return {
        "packed": pack_bits((q + qmax).astype(np.int32), bits),
        "scales": scales,
        "shape": np.array(w.shape, dtype=np.int64),
        "bits": np.array(bits),
    }


def dequantize_tensor(t: dict[str, Any], group_size: int = GROUP_SIZE) -> np.ndarray:
    bits = int(t["bits"])
    shape = tuple(int(s) for s in t["shape"])
    count = int(np.prod(shape))
    n_groups = len(t["scales"])
    qmax = 2 ** (bits - 1) - 1
    q = unpack_bits(t["packed"], bits, n_groups * group_size).astype(np.float32) - qmax
    groups = q.reshape(n_groups, group_size) * t["scales"].astype(np.float32)[:, None]
    out: np.ndarray = groups.ravel()[:count].reshape(shape)
    return out


# --------------------------------------------------------------------------- the model
@dataclass
class LoRA:
    A: np.ndarray  # (D, r)
    B: np.ndarray  # (r, V)
    alpha: float

    @property
    def rank(self) -> int:
        return int(self.A.shape[1])

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def delta(self) -> np.ndarray:
        out: np.ndarray = self.scale * (self.A @ self.B)
        return out


class TinyLM:
    def __init__(
        self,
        vocab: Vocab,
        E: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        context_window: int,
        adapter: LoRA | None = None,
    ) -> None:
        self.vocab = vocab
        self.E = E.astype(np.float32)
        self.W = W.astype(np.float32)
        self.b = b.astype(np.float32)
        self.context_window = context_window
        self.adapter = adapter
        self._w_eff: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return int(self.E.shape[1])

    @property
    def n_params(self) -> int:
        return int(self.E.size + self.W.size + self.b.size)

    def W_eff(self) -> np.ndarray:
        if self.adapter is None:
            return self.W
        if self._w_eff is None:
            self._w_eff = (self.W + self.adapter.delta()).astype(np.float32)
        return self._w_eff

    def invalidate(self) -> None:
        self._w_eff = None

    # ---------------------------------------------------------------- features
    def _ctx(self, ids: Sequence[int]) -> np.ndarray:
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        out: np.ndarray = self.E[np.asarray(ids)].mean(axis=0)
        return out

    def sft_features(self, instruction: str, response: str) -> tuple[np.ndarray, np.ndarray]:
        instr = self.vocab.encode(instruction)
        seq = np.array([BOS, *instr, SEP, *self.vocab.encode(response), EOS])
        start = len(instr) + 1  # index of SEP
        t = np.arange(start, len(seq) - 1)
        ctx = np.broadcast_to(self._ctx(instr), (len(t), self.dim))
        h = np.concatenate([self.E[seq[t]], self.E[seq[t - 1]], ctx], axis=1)
        return h, seq[t + 1]

    def lm_features(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        emb = self.E[seq]
        csum = np.concatenate([np.zeros((1, self.dim), np.float32), np.cumsum(emb, axis=0)])
        t = np.arange(0, len(seq) - 1)
        lo = np.maximum(0, t + 1 - self.context_window)
        ctx = (csum[t + 1] - csum[lo]) / (t + 1 - lo)[:, None]
        prev = np.where(t > 0, seq[np.maximum(t - 1, 0)], BOS)
        h = np.concatenate([emb[t], self.E[prev], ctx], axis=1)
        return h.astype(np.float32), seq[t + 1]

    def stack_sft(self, pairs: Sequence[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        feats = [self.sft_features(q, a) for q, a in pairs]
        if not feats:
            return np.zeros((0, 3 * self.dim), np.float32), np.zeros(0, dtype=np.int64)
        return (
            np.concatenate([f[0] for f in feats]).astype(np.float32),
            np.concatenate([f[1] for f in feats]),
        )

    # ---------------------------------------------------------------- scoring
    def mean_nll(self, h: np.ndarray, y: np.ndarray, batch: int = 2048) -> float:
        if len(y) == 0:
            return float("nan")
        w, total = self.W_eff(), 0.0
        for i in range(0, len(y), batch):
            lp = _log_softmax(h[i : i + batch] @ w + self.b)
            total -= float(lp[np.arange(len(lp)), y[i : i + batch]].sum())
        return total / len(y)

    def continuation_logprobs(
        self, instruction: str, prefix: str, options: Sequence[str]
    ) -> np.ndarray:
        instr = self.vocab.encode(instruction)
        base_seq = [BOS, *instr, SEP, *self.vocab.encode(prefix)]
        ctx = self._ctx(instr)
        rows, targets, owner = [], [], []
        for k, option in enumerate(options):
            seq = base_seq + self.vocab.encode(option)
            for t in range(len(base_seq) - 1, len(seq) - 1):
                rows.append(np.concatenate([self.E[seq[t]], self.E[seq[t - 1]], ctx]))
                targets.append(seq[t + 1])
                owner.append(k)
        lp = _log_softmax(np.stack(rows) @ self.W_eff() + self.b)
        picked = lp[np.arange(len(targets)), np.array(targets)]
        out: np.ndarray = np.bincount(np.array(owner), weights=picked, minlength=len(options))
        return out

    def generate(
        self, prompts: Sequence[str], max_new_tokens: int, min_tokens: int = 3
    ) -> list[str]:
        if not prompts:
            return []
        w = self.W_eff()
        instrs = [self.vocab.encode(p) for p in prompts]
        ctx = np.stack([self._ctx(ids) for ids in instrs])
        last = np.array([SEP] * len(prompts))
        prev = np.array([ids[-1] if ids else BOS for ids in instrs])
        outputs: list[list[int]] = [[] for _ in prompts]
        seen: list[set[tuple[int, int]]] = [set() for _ in prompts]
        done = np.zeros(len(prompts), dtype=bool)
        banned = [PAD, UNK, BOS, SEP]
        for step in range(max_new_tokens):
            h = np.concatenate([self.E[last], self.E[prev], ctx], axis=1)
            logits = h @ w + self.b
            logits[:, banned] = -np.inf
            if step < min_tokens:
                logits[:, EOS] = -np.inf
            top = np.argsort(-logits, axis=1)[:, :12]
            for i in range(len(prompts)):
                if done[i]:
                    continue
                # Greedy with no-repeat-bigram blocking (a tiny model loops otherwise).
                pick = next(
                    (
                        int(t)
                        for t in top[i]
                        if t != last[i] and (int(last[i]), int(t)) not in seen[i]
                    ),
                    int(top[i][0]),
                )
                if pick == EOS:
                    done[i] = True
                    continue
                seen[i].add((int(last[i]), pick))
                outputs[i].append(pick)
                prev[i], last[i] = last[i], pick
            if done.all():
                break
        return [self.vocab.decode(o) for o in outputs]

    # ---------------------------------------------------------------- io
    def save(
        self, directory: Path, dtype: str = "fp16", extra_meta: dict[str, Any] | None = None
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        cast = np.float16 if dtype == "fp16" else np.float32
        w = self.W_eff() if self.adapter is not None else self.W
        np.savez(
            directory / "model.npz", E=self.E.astype(cast), W=w.astype(cast), b=self.b.astype(cast)
        )
        (directory / "vocab.json").write_text(json.dumps(self.vocab.itos), encoding="utf-8")
        write_json(
            directory / "meta.json",
            {
                "format": dtype,
                "context_window": self.context_window,
                "params": self.n_params,
                **(extra_meta or {}),
            },
        )

    @classmethod
    def load(cls, directory: Path) -> TinyLM:
        meta = read_json(directory / "meta.json")
        vocab = Vocab(json.loads((directory / "vocab.json").read_text(encoding="utf-8")))
        if str(meta["format"]).startswith("rtn-"):
            with np.load(directory / "model.npz") as z:
                tensors = {
                    name: {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"{name}__")}
                    for name in ("E", "W")
                }
                E = dequantize_tensor(tensors["E"])
                W = dequantize_tensor(tensors["W"])
                b = z["b"].astype(np.float32)
        else:
            with np.load(directory / "model.npz") as z:
                E, W, b = z["E"], z["W"], z["b"]
        return cls(vocab, E, W, b, int(meta["context_window"]))


def _copy_side_files(src: Path, dst: Path) -> None:
    for name in ("vocab.json", "general_probe.txt"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


class TinyLMModel(StudentModel):
    def __init__(self, lm: TinyLM, label: str, path: Path) -> None:
        self.lm, self.label, self.path = lm, label, path

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        return self.lm.generate(prompts, max_new_tokens)

    def canary_ranks(self, canaries: Sequence[Canary], n_alternatives: int, seed: int) -> list[int]:
        return canary_ranks(self.lm, canaries, n_alternatives, seed)

    def mean_nll(self, pairs: Sequence[tuple[str, str]]) -> float:
        h, y = self.lm.stack_sft(pairs)
        return self.lm.mean_nll(h, y)

    def general_perplexity(self) -> float | None:
        probe = self.path / "general_probe.txt"
        if not probe.exists():
            return None
        seqs = [
            np.array([BOS, *self.lm.vocab.encode(p), EOS])
            for p in _paragraphs(probe.read_text("utf-8"))
        ]
        feats = [self.lm.lm_features(s) for s in seqs if len(s) > 2]
        h = np.concatenate([f[0] for f in feats])
        y = np.concatenate([f[1] for f in feats])
        return float(math.exp(self.lm.mean_nll(h, y)))

    @property
    def size_bytes(self) -> int:
        return int((self.path / "model.npz").stat().st_size)


def canary_ranks(
    lm: TinyLM, canaries: Sequence[Canary], n_alternatives: int, seed: int
) -> list[int]:
    ranks = []
    for canary in canaries:
        options = [canary.secret, *alternatives(canary, n_alternatives, seed)]
        scores = lm.continuation_logprobs(canary.instruction, canary.prefix, options)
        ranks.append(1 + int((scores[1:] > scores[0]).sum()))
    return ranks


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# --------------------------------------------------------------------------- backend
class TinyLMBackend(StudentBackend):
    name = "tinylm"

    def __init__(self, info: dict[str, Any], seed: int, work_dir: Path | None = None) -> None:
        self.info = info
        self.seed = seed
        self.work_dir = work_dir

    def bind(self, work_dir: Path) -> TinyLMBackend:
        self.work_dir = work_dir
        return self

    @property
    def base_dir(self) -> Path:
        if self.work_dir is None:
            raise RuntimeError("TinyLMBackend.bind(work_dir) must be called first")
        return self.work_dir

    # ---------------------------------------------------------------- base model
    def prepare(self, data: TrainData, work_dir: Path) -> dict[str, Any]:
        self.bind(work_dir)
        if (work_dir / "model.npz").exists():
            return dict(read_json(work_dir / "meta.json"))
        rng = np.random.default_rng(self.seed)
        corpus = _paragraphs(
            (data_path("configs") / "tinylm_general_corpus.txt").read_text("utf-8")
        )
        n_probe = max(1, len(corpus) // 5)
        general_train, general_probe = corpus[n_probe:], corpus[:n_probe]
        domain = [c.text for c in data.chunks]
        sft_text = [t for ex in data.train for t in (ex.instruction, ex.response)]
        vocab = Vocab.build(
            domain + general_train + sft_text,
            max_size=int(self.info.get("vocab_size", 4000)),
            required=CANARY_WORDS,
        )
        dim = int(self.info.get("embed_dim", 48))
        window = int(self.info.get("context_window", 16))
        seqs = [np.array([BOS, *vocab.encode(t), EOS]) for t in domain + general_train]
        start = time.perf_counter()
        E = ppmi_svd_embeddings(seqs, len(vocab), dim, rng)
        lm = TinyLM(
            vocab,
            E,
            np.zeros((3 * dim, len(vocab)), np.float32),
            np.zeros(len(vocab), np.float32),
            window,
        )
        feats = [lm.lm_features(s) for s in seqs if len(s) > 2]
        h = np.concatenate([f[0] for f in feats])
        y = np.concatenate([f[1] for f in feats])
        unigram = np.bincount(y, minlength=len(vocab)) + 1.0
        lm.b = np.log(unigram / unigram.sum()).astype(np.float32)
        lm.W = (rng.standard_normal(lm.W.shape) * 0.01).astype(np.float32)
        params = {"W": lm.W, "b": lm.b}
        opt = Adam(params, lr=0.01, weight_decay=1e-5)
        epochs = int(self.info.get("pretrain_epochs", 4))
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for i in range(0, len(y), 512):
                idx = order[i : i + 512]
                hb = h[idx]
                g = _softmax(hb @ params["W"] + params["b"])
                g[np.arange(len(idx)), y[idx]] -= 1.0
                g /= len(idx)
                opt.step(params, {"W": hb.T @ g, "b": g.sum(axis=0)})
        lm.W, lm.b = params["W"], params["b"]
        pretrain_loss = lm.mean_nll(h, y)
        meta = {
            "vocab_size": len(vocab),
            "embed_dim": dim,
            "context_window": window,
            "params": lm.n_params,
            "pretrain_tokens": len(y),
            "pretrain_loss": round(pretrain_loss, 4),
            "pretrain_seconds": round(time.perf_counter() - start, 2),
        }
        lm.save(work_dir, dtype="fp32", extra_meta=meta)
        (work_dir / "general_probe.txt").write_text("\n\n".join(general_probe), encoding="utf-8")
        log.info("tinylm_base_ready", **meta)
        return dict(read_json(work_dir / "meta.json"))

    def _base(self) -> TinyLM:
        return TinyLM.load(self.base_dir)

    # ---------------------------------------------------------------- LoRA training
    def train(
        self, data: TrainData, hp: Hyperparams, guard: OverfitGuard, out_dir: Path
    ) -> TrainRunResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(hp.seed + hp.rank)
        lm = self._base()
        D, V = lm.W.shape
        h, y = lm.stack_sft([(e.instruction, e.response) for e in data.train])
        h_real, y_real = lm.stack_sft([(s.question, s.answer) for s in data.real_val])
        h_syn, y_syn = lm.stack_sft([(e.instruction, e.response) for e in data.synth_val])
        sub = rng.choice(len(y), size=min(len(y), 4096), replace=False)
        h_sub, y_sub = h[sub], y[sub]

        bound = 1.0 / math.sqrt(D)
        params = {
            "A": rng.uniform(-bound, bound, size=(D, hp.rank)).astype(np.float32),
            "B": np.zeros((hp.rank, V), dtype=np.float32),
        }
        adapter = LoRA(params["A"], params["B"], hp.alpha)
        lm.adapter = adapter
        opt = Adam(params, lr=hp.learning_rate, weight_decay=hp.weight_decay)
        steps_per_epoch = max(1, math.ceil(len(y) / hp.batch_size))
        eval_every = max(1, steps_per_epoch // hp.evals_per_epoch)
        scale = adapter.scale
        start, step = time.perf_counter(), 0

        def evaluate() -> EvalPoint:
            lm.invalidate()
            real, synth = lm.mean_nll(h_real, y_real), lm.mean_nll(h_syn, y_syn)
            hit_rate = None
            if data.canaries:
                ranks = canary_ranks(lm, data.canaries, hp.canary_alternatives, hp.seed)
                hit_rate = sum(r == 1 for r in ranks) / len(ranks)
            return EvalPoint(
                step=step,
                epoch=round(step / steps_per_epoch, 3),
                train_loss=lm.mean_nll(h_sub, y_sub),
                real_val_loss=real,
                synth_val_loss=synth,
                gap=real - synth,
                canary_hit_rate=hit_rate,
            )

        # Evaluate the untouched base first so the guard sees the starting point.
        guard.update(evaluate())
        np.savez(out_dir / "ckpt-0.npz", A=params["A"], B=params["B"], alpha=hp.alpha)
        stop = False
        for _epoch in range(hp.epochs):
            order = rng.permutation(len(y))
            for i in range(steps_per_epoch):
                idx = order[i * hp.batch_size : (i + 1) * hp.batch_size]
                hb = h[idx]
                hd = hb
                if hp.dropout > 0:
                    hd = hb * (rng.random(hb.shape) >= hp.dropout) / (1.0 - hp.dropout)
                ha = hd @ params["A"]
                g = _softmax(hb @ lm.W + lm.b + scale * (ha @ params["B"]))
                g[np.arange(len(idx)), y[idx]] -= 1.0
                g /= len(idx)
                grads = {"B": scale * (ha.T @ g), "A": scale * (hd.T @ (g @ params["B"].T))}
                opt.step(params, grads)
                adapter.A, adapter.B = params["A"], params["B"]
                step += 1
                if step % eval_every == 0 or i == steps_per_epoch - 1:
                    point = evaluate()
                    np.savez(
                        out_dir / f"ckpt-{step}.npz", A=params["A"], B=params["B"], alpha=hp.alpha
                    )
                    if guard.update(point).stop:
                        stop = True
                        break
            if stop:
                break

        best_ckpt = out_dir / f"ckpt-{guard.best_step}.npz"
        return TrainRunResult(
            rank=hp.rank,
            alpha=hp.alpha,
            curve=guard.history,
            events=guard.events,
            stopped_early=guard.stopped,
            stop_reason=guard.stop_reason,
            best_step=guard.best_step,
            best_epoch=guard.best_epoch,
            best_real_val_loss=guard.best_loss,
            best_checkpoint=str(best_ckpt),
            trainable_params=int(params["A"].size + params["B"].size),
            total_params=lm.n_params,
            duration_s=round(time.perf_counter() - start, 3),
        )

    # ---------------------------------------------------------------- merge / quantise
    def merge(self, checkpoint: Path, out_dir: Path) -> Path:
        lm = self._base()
        with np.load(checkpoint) as z:
            lm.adapter = LoRA(z["A"], z["B"], float(z["alpha"]))
        # Merge into the full-precision base, then store fp16 (CLAUDE.md gotcha).
        lm.save(
            out_dir,
            dtype="fp16",
            extra_meta={"merged_from": checkpoint.name, "rank": lm.adapter.rank},
        )
        _copy_side_files(self.base_dir, out_dir)
        return out_dir

    def quantize(self, merged: Path, fmt: str, out_dir: Path, calibration: Sequence[str]) -> Path:
        if fmt not in QUANT_BITS:
            raise QuantizerUnavailableError(f"tinylm supports {sorted(QUANT_BITS)}, not {fmt!r}")
        bits = QUANT_BITS[fmt]
        lm = TinyLM.load(merged)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"b": lm.b.astype(np.float16)}
        for name, tensor in (("E", lm.E), ("W", lm.W)):
            for key, value in quantize_tensor(tensor, bits).items():
                arrays[f"{name}__{key}"] = np.asarray(value)
        np.savez(out_dir / "model.npz", **arrays)  # type: ignore[arg-type]
        meta = read_json(merged / "meta.json")
        meta.update({"format": f"rtn-{fmt}", "bits": bits, "group_size": GROUP_SIZE})
        write_json(out_dir / "meta.json", meta)
        _copy_side_files(merged, out_dir)
        return out_dir

    def load(self, artifact: Path, label: str) -> StudentModel:
        return TinyLMModel(TinyLM.load(artifact), label, artifact)

    def load_base(self) -> StudentModel:
        return TinyLMModel(self._base(), "base", self.base_dir)

    def quant_families(self, device: dict[str, Any]) -> dict[str, list[str]]:
        return {"rtn": ["int4", "int5", "int8"]}
