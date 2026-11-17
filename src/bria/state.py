"""Task and tool types, budgets, evidence access, and execution records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import NoSuchResource

METHOD_ID = "bria"
PROJECTOR_ID = "bria_context"


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def clone(value):
    return json.loads(canonical(value))


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


class ValidationError(ValueError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class Interrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    max_tool_calls: int = 100
    max_retrieval_queries: int = 20
    retrieval_top_k: int = 50
    global_search_depth: int = 10000
    max_candidates_per_checkpoint: int = 16
    max_llm_calls: int | None = 200
    max_total_tokens: int | None = None
    max_wall_seconds: float | None = None
    max_internal_actions: int = 200

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is None and name not in {
                "max_llm_calls",
                "max_total_tokens",
                "max_wall_seconds",
            }:
                raise ValueError(f"Invalid limit: {name}")
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (not math.isfinite(value))
                or (value < 1)
                or (name != "max_wall_seconds" and type(value) is not int)
            ):
                raise ValueError(f"Invalid limit: {name}")


class Budget:
    def __init__(self, config, audit):
        self.config, self.audit = (config, audit)
        self.started = time.monotonic()
        self.counts = dict(tool=0, retrieval=0, llm=0, internal=0)
        self.known_tokens = 0
        self.unknown_usage_calls = 0
        self.usage_by_role = {}
        self.backend_dispatches_known = 0
        self.backend_dispatches_unknown = 0
        self.backend_not_dispatched = 0

    def check(self):
        if (
            self.config.max_wall_seconds is not None
            and time.monotonic() - self.started >= self.config.max_wall_seconds
        ):
            raise BudgetExhausted("wall_clock")
        if self.config.max_total_tokens is not None:
            if self.unknown_usage_calls:
                raise BudgetExhausted("token_usage_unknown")
            if self.known_tokens >= self.config.max_total_tokens:
                raise BudgetExhausted("tokens")

    def reserve(self, kind, role=None):
        self.check()
        limit = {
            "tool": self.config.max_tool_calls,
            "retrieval": self.config.max_retrieval_queries,
            "llm": self.config.max_llm_calls,
            "internal": self.config.max_internal_actions,
        }[kind]
        if limit is not None and self.counts[kind] >= limit:
            raise BudgetExhausted(kind)
        self.counts[kind] += 1
        return self.audit.append_event(
            "budget_event", dict(stage="reserved", kind=kind, role=role, counts=self.counts)
        )

    def settle_llm(self, role, usage):
        bucket = self.usage_by_role.setdefault(
            role, dict(calls=0, known_tokens=0, unknown_usage_calls=0)
        )
        bucket["calls"] += 1
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        if total is None and isinstance(usage, dict):
            prompt, output = (
                usage.get("prompt_tokens", usage.get("input_tokens")),
                usage.get("completion_tokens", usage.get("output_tokens")),
            )
            if type(prompt) is int and prompt >= 0 and (type(output) is int) and (output >= 0):
                total = prompt + output
        if type(total) is int and total >= 0:
            self.known_tokens += total
            bucket["known_tokens"] += total
        else:
            self.unknown_usage_calls += 1
            bucket["unknown_usage_calls"] += 1
        self.audit.append_event(
            "budget_event",
            dict(stage="settled", kind="llm", role=role, usage=usage, totals=self.view()),
        )

    def settle_tool(self, dispatched):
        if dispatched is True:
            self.backend_dispatches_known += 1
        elif dispatched is False:
            self.backend_not_dispatched += 1
        else:
            self.backend_dispatches_unknown += 1

    def view(self):
        limits = dict(
            tool=self.config.max_tool_calls,
            retrieval=self.config.max_retrieval_queries,
            llm=self.config.max_llm_calls,
            internal=self.config.max_internal_actions,
        )
        return dict(
            counts=clone(self.counts),
            backend_dispatches_known=self.backend_dispatches_known,
            backend_dispatches=None
            if self.backend_dispatches_unknown
            else self.backend_dispatches_known,
            backend_dispatches_unknown=self.backend_dispatches_unknown,
            backend_not_dispatched=self.backend_not_dispatched,
            remaining={
                k: None if v is None else max(0, v - self.counts[k]) for k, v in limits.items()
            },
            known_tokens=self.known_tokens,
            total_tokens=None if self.unknown_usage_calls else self.known_tokens,
            unknown_usage_calls=self.unknown_usage_calls,
            usage_by_role=clone(self.usage_by_role),
        )


SourceFamily: TypeAlias = Literal["planbench_xl", "mcp_atlas", "custom"]
Benchmark: TypeAlias = Literal["planbench_xl", "mcp_atlas", "custom"]
RetrievalMode: TypeAlias = Literal["union_raw_topk"]


@dataclass(frozen=True, slots=True)
class UnifiedToolRecord:
    global_tool_id: str
    tool_name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object] | None
    output_description: str | None
    source_family: SourceFamily
    source_id: str
    server_name: str | None
    executor_key: str
    public_schema_hash: str
    retrieval_document: str
    duplicate_group_id: str | None = None


@dataclass(frozen=True, slots=True)
class PublicTaskView:
    """The controller-visible portion of a task; deliberately excludes gold data."""

    task_id: str
    benchmark: Benchmark
    user_query: str
    allowed_tool_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query_text: str
    top_k: int
    task_id: str
    active_benchmark: Benchmark
    exclude_tool_ids: tuple[str, ...] = ()
    mode: RetrievalMode = "union_raw_topk"
    global_search_depth: int = 10000
    seen_tool_ids: tuple[str, ...] = ()
    exclude_seen_tools: bool = False
    repeated_context_top_k: int = 0
    repeated_context_from_seen: bool = False

    def __post_init__(self) -> None:
        if type(self.repeated_context_from_seen) is not bool:
            raise ValueError("repeated_context_from_seen must be a boolean")
        if not isinstance(self.exclude_seen_tools, bool):
            raise ValueError("exclude_seen_tools must be a boolean")
        if type(self.repeated_context_top_k) is not int or self.repeated_context_top_k < 0:
            raise ValueError("repeated_context_top_k must be a nonnegative integer")
        if self.mode != "union_raw_topk":
            raise ValueError("Only union_raw_topk retrieval is supported")


@dataclass(frozen=True, slots=True)
class RetrievedTool:
    record: UnifiedToolRecord
    rank: int
    score: float
    global_rank: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    request: RetrievalRequest
    raw_global_top_k: tuple[RetrievedTool, ...]
    visible_top_k: tuple[RetrievedTool, ...]
    query_encoding_ms: float = 0.0
    index_search_ms: float = 0.0
    shortfall: int = 0
    repeated_context_top_k: tuple[RetrievedTool, ...] = ()


def stable_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


@dataclass(frozen=True, slots=True, order=True)
class CallKey:
    tool_id: str
    arguments_json: str

    @classmethod
    def from_arguments(cls, tool_id: str, arguments: Mapping[str, object]) -> CallKey:
        return cls(tool_id=tool_id, arguments_json=stable_json(dict(arguments)))

    @property
    def arguments(self) -> dict[str, object]:
        decoded = json.loads(self.arguments_json)
        if not isinstance(decoded, dict):
            raise TypeError("CallKey arguments must decode to an object")
        return decoded

    def stable_string(self) -> str:
        return f"{self.tool_id}:{self.arguments_json}"


@dataclass(frozen=True, slots=True)
class PublicToolObservation:
    success: bool
    result: object | None
    error_code: str | None = None
    message: str | None = None
    unexpected_error: bool = False
    http_status: int | None = None
    raw_response: object | None = None
    interrupts_task: bool = False
    error_origin: str | None = None
    dispatched: bool | None = None
    server_name: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    call_key: CallKey
    observation: PublicToolObservation
    step_id: int


class ToolExecutor(Protocol):
    def execute(self, task: PublicTaskView, call: CallKey) -> PublicToolObservation: ...


def _no_remote(uri):
    raise NoSuchResource(ref=uri)


def schema_errors(value: object, schema: Mapping[str, object] | None) -> tuple[str, ...]:
    if not schema:
        return ()
    try:
        cls = validator_for(schema)
        cls.check_schema(schema)
        validator = cls(schema, registry=Registry(retrieve=_no_remote))
        return tuple(
            (
                f"/{'/'.join((str(x) for x in error.absolute_path))}: {error.message}"
                for error in validator.iter_errors(value)
            )
        )
    except Exception as exc:
        return (f"Public schema could not be validated: {type(exc).__name__}: {exc}",)


def json_pointer(value: object, pointer: str) -> object:
    if pointer == "":
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("An evidence path must be an RFC 6901 JSON pointer")
    current = value
    for encoded in pointer[1:].split("/"):
        import re

        if re.search("~(?![01])", encoded):
            raise ValueError("Invalid JSON pointer escape")
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise ValueError(f"Observation path does not exist: {pointer}")
            current = current[token]
        elif (
            isinstance(current, (list, tuple))
            and token.isdigit()
            and (token == "0" or not token.startswith("0"))
        ):
            index = int(token)
            if index >= len(current):
                raise ValueError(f"Observation index does not exist: {pointer}")
            current = current[index]
        else:
            raise ValueError(f"Observation path does not exist: {pointer}")
    return current


class AuditStore:
    def __init__(self, directory, sink=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "blobs").mkdir()
        self.path = self.directory / "events.jsonl"
        self._events = []
        self.sink = sink

    def blob(self, value):
        key = digest(value)
        path = self.directory / "blobs" / (key.split(":")[1] + ".json")
        text = canonical(value)
        if path.exists():
            if path.read_text() != text:
                raise ValidationError("Immutable blob mismatch")
        else:
            with path.open("x") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        return key

    def append_event(self, kind, data):
        event = dict(
            event_id=f"event_{len(self._events) + 1}",
            kind=kind,
            data=clone(data),
            previous_hash=self._events[-1]["event_hash"] if self._events else None,
        )
        event["event_hash"] = digest(event)
        with self.path.open("a") as stream:
            stream.write(canonical(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._events.append(event)
        if self.sink is not None:
            self.sink(clone(event))
        return event["event_id"]

    @property
    def events(self):
        return clone(self._events)

    @staticmethod
    def replay(directory):
        """Validate the chain/blobs and recover persisted controller state/projections.

        Replay is read-only, not permission to redispatch an unknown tool outcome.
        """
        root = Path(directory)
        previous = None
        state = None
        projections, observations, pending, unknown = ([], {}, {}, {})
        for index, line in enumerate((root / "events.jsonl").read_text().splitlines(), 1):
            event = json.loads(line)
            event_hash = event.pop("event_hash")
            if (
                digest(event) != event_hash
                or event["previous_hash"] != previous
                or event["event_id"] != f"event_{index}"
            ):
                raise ValidationError("Audit chain mismatch")
            previous = event_hash
            data = event["data"]
            if event["kind"] == "state_saved":
                state = data
            if event["kind"] == "observation_saved":
                observations[data["observation_id"]] = data
            if event["kind"] == "projection_created":
                projections.append(data)
            if event["kind"] == "tool_dispatch":
                pending[event["event_id"]] = data
            if event["kind"] == "dispatched_unknown_outcome":
                unknown[data["dispatch_event_id"]] = data
            if event["kind"] in {"tool_result", "tool_error", "dispatched_unknown_outcome"}:
                pending.pop(data["dispatch_event_id"], None)
        for path in (root / "blobs").glob("*.json"):
            if digest(json.loads(path.read_text())) != "sha256:" + path.stem:
                raise ValidationError("Blob hash mismatch")
        for projection in projections:
            messages = json.loads(
                (
                    root
                    / "blobs"
                    / (projection["rendered_message_blob_ref"].split(":")[1] + ".json")
                ).read_text()
            )
            schemas = json.loads(
                (
                    root
                    / "blobs"
                    / (projection["rendered_schema_blob_ref"].split(":")[1] + ".json")
                ).read_text()
            )
            if digest(dict(messages=messages, schemas=schemas)) != projection["input_hash"]:
                raise ValidationError("Projection hash mismatch")
        return dict(
            state=state,
            observations=observations,
            projections=projections,
            dispatched_unknown_outcome={**unknown, **pending},
        )


def tokens(pointer):
    if pointer == "":
        return []
    if (
        not isinstance(pointer, str)
        or not pointer.startswith("/")
        or re.search("~(?![01])", pointer)
    ):
        raise ValidationError("Invalid RFC 6901 pointer")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def pointer_get(value, pointer):
    current = value
    for token in tokens(pointer):
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif (
            isinstance(current, list)
            and re.fullmatch("0|[1-9][0-9]*", token)
            and (int(token) < len(current))
        ):
            current = current[int(token)]
        else:
            raise ValidationError("Pointer does not exist")
    return clone(current)


def below(pointer, prefix):
    child, parent = (tokens(pointer), tokens(prefix))
    return child[: len(parent)] == parent


def join(pointer, field):
    tokens(pointer)
    tokens(field)
    return pointer + field


def strict_json(text):

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError("Duplicate JSON object key")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValidationError("Non-finite JSON")),
    )


class EvidenceStore:
    def __init__(self, audit):
        self.audit = audit
        self._snapshots = {}
        self._payloads = {}
        self.grants = {}
        self.invalidated = set()
        self.dependencies = {}

    def save(
        self,
        *,
        tool_id,
        schema_hash,
        arguments,
        raw,
        parsed,
        owner,
        event_id,
        epoch,
        success=True,
        source_refs=(),
        dependencies=(),
    ):
        parse_status = "json"
        try:
            if isinstance(raw, str):
                try:
                    strict_json(raw)
                except json.JSONDecodeError:
                    if not isinstance(parsed, str):
                        raise ValidationError(
                            "Non-JSON raw text cannot become extracted JSON evidence"
                        )
            parsed = clone(parsed)
        except (ValueError, TypeError):
            parsed, parse_status = (None, "invalid_json")
        obs_id = f"obs_{len(self._snapshots) + 1}"
        snapshot = dict(
            observation_id=obs_id,
            producing_event_id=event_id,
            tool_id=tool_id,
            tool_schema_hash=schema_hash,
            resolved_arguments=clone(arguments),
            raw_payload_blob_ref=self.audit.blob(raw),
            parsed_payload_blob_ref=self.audit.blob(parsed),
            payload_hash=digest(parsed),
            observed_at=datetime.now(timezone.utc).isoformat(),
            owner_scope=owner,
            environment_epoch=epoch,
            parse_status=parse_status,
            success=success,
            source_refs=clone(list(source_refs)),
        )
        self._snapshots[obs_id] = clone(snapshot)
        self._payloads[obs_id] = clone(parsed)
        self.dependencies[obs_id] = set(dependencies) | {r["observation_id"] for r in source_refs}
        self.grant(owner, obs_id, "")
        self.audit.append_event("observation_saved", snapshot)
        return self.ref(obs_id)

    def snapshot(self, obs_id):
        if obs_id not in self._snapshots:
            raise ValidationError("Unknown observation")
        return clone(self._snapshots[obs_id])

    def ref(self, obs_id, pointer=""):
        snapshot = self.snapshot(obs_id)
        pointer_get(self._payloads[obs_id], pointer)
        return dict(
            observation_id=obs_id,
            pointer=pointer,
            payload_hash=snapshot["payload_hash"],
            owner_scope=snapshot["owner_scope"],
        )

    def grant(self, actor, obs_id, pointer):
        self.grants.setdefault(actor, {}).setdefault(obs_id, set()).add(pointer)

    def restrict(self, actor, obs_id, pointers):
        self.grants.setdefault(actor, {})[obs_id] = set(pointers)

    def read_evidence(self, ref, actor_scope):
        snapshot = self.snapshot(ref["observation_id"])
        if snapshot["observation_id"] in self.invalidated:
            raise ValidationError("Invalidated evidence")
        if snapshot["parse_status"] != "json" or not snapshot["success"]:
            raise ValidationError("Observation is not successful JSON evidence")
        if (
            ref.get("payload_hash") != snapshot["payload_hash"]
            or ref.get("owner_scope") != snapshot["owner_scope"]
        ):
            raise ValidationError("Evidence hash or owner mismatch")
        if not any(
            (
                below(ref["pointer"], p)
                for p in self.grants.get(actor_scope, {}).get(ref["observation_id"], ())
            )
        ):
            raise ValidationError("Evidence access denied")
        return pointer_get(self._payloads[ref["observation_id"]], ref["pointer"])

    def closure(self, obs_ids):
        result = set(obs_ids)
        while True:
            enlarged = result | {d for obs in result for d in self.dependencies.get(obs, ())}
            if enlarged == result:
                return sorted(result)
            result = enlarged

    def invalidate(self, ids):
        invalid = set(ids)
        while True:
            enlarged = invalid | {obs for obs, deps in self.dependencies.items() if deps & invalid}
            if enlarged == invalid:
                break
            invalid = enlarged
        self.invalidated |= invalid
        return sorted(invalid)
