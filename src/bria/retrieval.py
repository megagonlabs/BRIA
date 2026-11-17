"""Embedding configuration, Qwen encoding, and exact tool retrieval."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import yaml

from bria.state import (
    Config,
    PublicTaskView,
    RetrievalRequest,
    RetrievalResult,
    RetrievedTool,
    UnifiedToolRecord,
)

DEFAULT_QUERY_INSTRUCTION = "Given a user task or a missing capability in a tool-use plan, retrieve public tool schemas that can help satisfy the request."
DOCUMENT_TEMPLATE = "public_schema"


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str = "Qwen/Qwen3-Embedding-8B"
    model_revision: str = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
    backend: str = "sentence_transformers"
    dtype: str = "bfloat16"
    device: str | list[str] = "auto"
    batch_size: int = 16
    max_length: int = 4096
    normalize: bool = True
    truncate_dim: int | None = None
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION

    def validate(self):
        if self.model_id != "Qwen/Qwen3-Embedding-8B":
            raise ValueError("This retriever requires Qwen/Qwen3-Embedding-8B")
        if not self.model_revision or self.model_revision in {"main", "latest"}:
            raise ValueError("Pin an exact embedding model revision")
        if self.backend != "sentence_transformers" or self.normalize is not True:
            raise ValueError("Use normalized sentence-transformers embeddings")
        for name in ("batch_size", "max_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.device, list) and (
            not self.device or not all((isinstance(value, str) and value for value in self.device))
        ):
            raise ValueError("device must contain device names")


@dataclass(frozen=True)
class AppConfig:
    policy: Config = Config()
    embedding: EmbeddingConfig = EmbeddingConfig()


def load_config(path: str | Path) -> AppConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or set(payload) - {"policy", "embedding"}:
        raise ValueError("Configuration supports only policy and embedding sections")

    def construct(cls, values):
        if not isinstance(values, dict) or set(values) - {item.name for item in fields(cls)}:
            raise ValueError(f"Unknown {cls.__name__} configuration fields")
        return cls(**values)

    config = AppConfig(
        policy=construct(Config, payload.get("policy", {})),
        embedding=construct(EmbeddingConfig, payload.get("embedding", {})),
    )
    config.embedding.validate()
    return config


class TextEmbedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def normalize_query_text(text: str) -> str:
    return " ".join(text.strip().split())


def qwen_query(instruction: str, query_text: str) -> str:
    return f"Instruct: {instruction.strip()}\nQuery: {normalize_query_text(query_text)}"


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional embedding matrix, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Embedding matrix contains non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Embedding backend returned a zero vector")
    normalized = array / norms
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized embedding matrix contains non-finite values")
    return normalized


class Qwen3Embedder:
    """Lazy sentence-transformers backend; it never substitutes another model."""

    def __init__(self, config: EmbeddingConfig, *, query_cache_size: int = 1024) -> None:
        config.validate()
        self.config = config
        self._model = None
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_size = query_cache_size

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def model_revision(self) -> str:
        return self.config.model_revision

    @property
    def query_instruction(self) -> str:
        return self.config.query_instruction

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the 'qwen' extra to use Qwen/Qwen3-Embedding-8B") from exc
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.config.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported embedding dtype: {self.config.dtype}")
        multiple_devices = self.config.device if isinstance(self.config.device, list) else None
        device = (
            None
            if self.config.device == "auto" or multiple_devices is not None
            else self.config.device
        )
        kwargs = {
            "revision": self.config.model_revision,
            "model_kwargs": {"torch_dtype": dtype},
            "trust_remote_code": False,
        }
        if device is not None:
            kwargs["device"] = device
        self._model = SentenceTransformer(self.config.model_id, **kwargs)
        self._model.max_seq_length = self.config.max_length
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dimension = int(self.config.truncate_dim or 0)
            return np.empty((0, dimension), dtype=np.float32)
        model = self._load_model()
        kwargs = {
            "batch_size": self.config.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": False,
            "show_progress_bar": False,
        }
        if isinstance(self.config.device, list):
            kwargs["device"] = self.config.device
        if self.config.truncate_dim is not None:
            kwargs["truncate_dim"] = self.config.truncate_dim
        embeddings = np.asarray(model.encode(list(texts), **kwargs), dtype=np.float32)
        if self.config.truncate_dim is not None:
            embeddings = embeddings[:, : self.config.truncate_dim]
        return l2_normalize(embeddings)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        normalized = [normalize_query_text(text) for text in texts]
        results: list[np.ndarray | None] = [None] * len(normalized)
        missing_keys: list[str] = []
        missing_positions: dict[str, list[int]] = {}
        for position, key in enumerate(normalized):
            cached = self._query_cache.get(key)
            if cached is not None:
                self._query_cache.move_to_end(key)
                results[position] = cached.copy()
            else:
                if key not in missing_positions:
                    missing_keys.append(key)
                    missing_positions[key] = []
                missing_positions[key].append(position)
        if missing_keys:
            encoded = self._encode(
                [qwen_query(self.query_instruction, key) for key in missing_keys]
            )
            for key, vector in zip(missing_keys, encoded):
                self._query_cache[key] = vector.copy()
                self._query_cache.move_to_end(key)
                while len(self._query_cache) > self._query_cache_size:
                    self._query_cache.popitem(last=False)
                for position in missing_positions[key]:
                    results[position] = vector.copy()
        if not results:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([item for item in results if item is not None]).astype(np.float32)


@dataclass(frozen=True, slots=True)
class IndexManifest:
    model_id: str
    model_revision: str
    query_instruction: str
    normalize_embeddings: bool
    embedding_dimension: int
    corpus_sha256: str
    index_type: str = "IndexFlatIP"
    document_template_version: str = "public_schema"
    embedding_sha256: str = ""
    ids_sha256: str = ""
    index_sha256: str = ""
    faiss_sha256: str = ""


class StaleIndexError(ValueError):
    pass


class ExactVectorIndex:
    def __init__(
        self, ids: Sequence[str], embeddings: np.ndarray, *, faiss_index: object | None = None
    ) -> None:
        self.ids = tuple((str(item) for item in ids))
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Index IDs must be unique")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.ids):
            raise ValueError("Embedding rows must match index IDs")
        self.embeddings = l2_normalize(matrix)
        self._faiss_index = faiss_index

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def _rank(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        order = sorted(range(len(self.ids)), key=lambda i: (-float(values[i]), self.ids[i]))
        return [(self.ids[i], float(values[i])) for i in order[: max(0, k)]]

    def search_numpy(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        vector = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if vector.shape[1] != self.dimension:
            raise ValueError("Query/index embedding dimensions differ")
        return self._rank(self.embeddings @ vector[0], min(k, len(self.ids)))

    def search_faiss(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS is not installed") from exc
        vector = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
        index = self._faiss_index
        if index is None:
            index = faiss.IndexFlatIP(self.dimension)
            index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))
        scores, indices = index.search(vector, len(self.ids))
        dense = np.full(len(self.ids), -np.inf, dtype=np.float64)
        for score, index_position in zip(scores[0], indices[0]):
            if index_position >= 0:
                dense[int(index_position)] = float(score)
        return self._rank(dense, min(k, len(self.ids)))

    def search(
        self, query: np.ndarray, k: int, *, backend: Literal["auto", "numpy", "faiss"] = "auto"
    ) -> list[tuple[str, float]]:
        if backend == "numpy":
            return self.search_numpy(query, k)
        if backend == "faiss":
            return self.search_faiss(query, k)
        try:
            return self.search_faiss(query, k)
        except RuntimeError:
            return self.search_numpy(query, k)

    def save(self, output_dir: str | Path, manifest: IndexManifest) -> IndexManifest:
        output = Path(output_dir)
        if manifest.embedding_dimension != self.dimension:
            raise ValueError("Manifest embedding dimension does not match index")
        output.mkdir(parents=True, exist_ok=False)
        f16 = self.embeddings.astype(np.float16)
        embedding_path = output / "embeddings.f16.npy"
        np.save(embedding_path, f16, allow_pickle=False)
        embedding_bytes = embedding_path.read_bytes()
        embedding_sha = hashlib.sha256(embedding_bytes).hexdigest()
        ids_bytes = (json.dumps(list(self.ids), ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        ids_path = output / "ids.json"
        ids_path.write_bytes(ids_bytes)
        ids_sha = hashlib.sha256(ids_bytes).hexdigest()
        index_sha = hashlib.sha256(ids_bytes + embedding_bytes).hexdigest()
        faiss_sha = ""
        try:
            import faiss

            stored_embeddings = l2_normalize(f16.astype(np.float32))
            index = faiss.IndexFlatIP(self.dimension)
            index.add(np.ascontiguousarray(stored_embeddings, dtype=np.float32))
            faiss_path = output / "index.faiss"
            faiss.write_index(index, str(faiss_path))
            faiss_sha = hashlib.sha256(faiss_path.read_bytes()).hexdigest()
        except ImportError:
            pass
        complete = IndexManifest(
            **{
                **asdict(manifest),
                "embedding_sha256": embedding_sha,
                "ids_sha256": ids_sha,
                "index_sha256": index_sha,
                "faiss_sha256": faiss_sha,
            }
        )
        (output / "index_manifest.json").write_text(
            json.dumps(asdict(complete), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return complete

    @classmethod
    def load(
        cls,
        output_dir: str | Path,
        *,
        expected_corpus_sha256: str | None = None,
        expected_model_id: str | None = None,
        expected_model_revision: str | None = None,
        expected_query_instruction: str | None = None,
        expected_document_template_version: str | None = None,
    ) -> tuple[ExactVectorIndex, IndexManifest]:
        output = Path(output_dir)
        raw_manifest = json.loads((output / "index_manifest.json").read_text(encoding="utf-8"))
        manifest = IndexManifest(**raw_manifest)
        expectations = {
            "corpus_sha256": expected_corpus_sha256,
            "model_id": expected_model_id,
            "model_revision": expected_model_revision,
            "query_instruction": expected_query_instruction,
            "document_template_version": expected_document_template_version,
        }
        stale = {
            key: (expected, getattr(manifest, key))
            for key, expected in expectations.items()
            if expected is not None and expected != getattr(manifest, key)
        }
        if stale:
            raise StaleIndexError(f"Stale index manifest: {stale}")
        embedding_path = output / "embeddings.f16.npy"
        if hashlib.sha256(embedding_path.read_bytes()).hexdigest() != manifest.embedding_sha256:
            raise ValueError("Embedding artifact checksum mismatch")
        ids_path = output / "ids.json"
        ids_bytes = ids_path.read_bytes()
        if hashlib.sha256(ids_bytes).hexdigest() != manifest.ids_sha256:
            raise ValueError("Index ID artifact checksum mismatch")
        if (
            hashlib.sha256(ids_bytes + embedding_path.read_bytes()).hexdigest()
            != manifest.index_sha256
        ):
            raise ValueError("Logical index checksum mismatch")
        embeddings = np.load(embedding_path, allow_pickle=False).astype(np.float32)
        ids = json.loads(ids_bytes.decode("utf-8"))
        faiss_index = None
        if manifest.faiss_sha256:
            faiss_path = output / "index.faiss"
            if hashlib.sha256(faiss_path.read_bytes()).hexdigest() != manifest.faiss_sha256:
                raise ValueError("FAISS artifact checksum mismatch")
            try:
                import faiss
            except ImportError:
                pass
            else:
                faiss_index = faiss.read_index(str(faiss_path))
        index = cls(ids, embeddings, faiss_index=faiss_index)
        if faiss_index is not None and (
            faiss_index.d != index.dimension or faiss_index.ntotal != len(index.ids)
        ):
            raise ValueError("FAISS artifact shape does not match manifest IDs")
        if index.dimension != manifest.embedding_dimension:
            raise ValueError("Loaded embedding dimension does not match manifest")
        return (index, manifest)


class UnifiedRetriever:
    """The only inference-time component with access to the complete corpus."""

    def __init__(
        self,
        *,
        records: Sequence[UnifiedToolRecord],
        index: ExactVectorIndex,
        embedder: TextEmbedder,
        query_cache_size: int = 1024,
        allow_exact_duplicates: bool = False,
    ) -> None:
        self._records = {record.global_tool_id: record for record in records}
        if set(index.ids) != set(self._records):
            raise ValueError("Index IDs and corpus records differ")
        self._index = index
        self._embedder = embedder
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_size = query_cache_size
        self._allow_exact_duplicates = allow_exact_duplicates
        self._query_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._embedder.model_id

    @property
    def model_revision(self) -> str:
        return self._embedder.model_revision

    def _encode_query(self, query_text: str) -> np.ndarray:
        with self._query_lock:
            return self._encode_query_locked(query_text)

    def _encode_query_locked(self, query_text: str) -> np.ndarray:
        key = normalize_query_text(query_text)
        cached = self._query_cache.get(key)
        if cached is not None:
            self._query_cache.move_to_end(key)
            return cached.copy()
        matrix = self._embedder.encode_queries([key])
        if matrix.shape != (1, self._index.dimension):
            raise ValueError(
                f"Query embedder returned {matrix.shape}, expected (1, {self._index.dimension})"
            )
        vector = matrix[0].copy()
        self._query_cache[key] = vector
        while len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return vector.copy()

    def retrieve(self, request: RetrievalRequest, task: PublicTaskView) -> RetrievalResult:
        if request.task_id != task.task_id or request.active_benchmark != task.benchmark:
            raise ValueError("Retrieval request/task identity mismatch")
        if request.top_k < 1 or request.global_search_depth < 1:
            raise ValueError("Retrieval depths must be positive")
        started = time.perf_counter()
        vector = self._encode_query(request.query_text)
        encoded = time.perf_counter()
        depth = min(
            len(self._records),
            max(request.top_k, request.global_search_depth, request.repeated_context_top_k),
        )
        rank_seen = (
            request.repeated_context_from_seen
            and request.repeated_context_top_k > 0
            and bool(request.seen_tool_ids)
        )
        ranked = self._index.search(vector, len(self._records) if rank_seen else depth)
        searched = time.perf_counter()
        blocked = set(request.exclude_tool_ids)
        seen_ids = set(request.seen_tool_ids)
        repeated_context: list[RetrievedTool] = []
        if request.exclude_seen_tools or rank_seen:
            for global_rank, (tool_id, score) in enumerate(
                ranked if rank_seen else ranked[: request.repeated_context_top_k], start=1
            ):
                if tool_id in seen_ids and tool_id not in blocked:
                    repeated_context.append(
                        RetrievedTool(self._records[tool_id], global_rank, score, global_rank)
                    )
                    if len(repeated_context) >= request.repeated_context_top_k:
                        break
        excluded = set(blocked)
        seen_groups: set[str] = set()
        if request.exclude_seen_tools:
            excluded.update(seen_ids)
            seen_groups.update(
                (
                    record.duplicate_group_id
                    for tool_id in request.seen_tool_ids
                    if (record := self._records.get(tool_id)) is not None
                    and record.duplicate_group_id is not None
                )
            )
        selected: list[RetrievedTool] = []
        for global_rank, (tool_id, score) in enumerate(ranked[:depth], start=1):
            if tool_id in excluded:
                continue
            record = self._records[tool_id]
            group = record.duplicate_group_id
            if group and (not self._allow_exact_duplicates) and (group in seen_groups):
                continue
            if group:
                seen_groups.add(group)
            selected.append(RetrievedTool(record, len(selected) + 1, score, global_rank))
            if len(selected) >= request.top_k:
                break
        raw = tuple(selected)
        return RetrievalResult(
            request=request,
            raw_global_top_k=raw,
            visible_top_k=raw,
            query_encoding_ms=(encoded - started) * 1000.0,
            index_search_ms=(searched - encoded) * 1000.0,
            shortfall=max(0, request.top_k - len(raw)),
            repeated_context_top_k=tuple(repeated_context),
        )
