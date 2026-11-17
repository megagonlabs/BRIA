"""External PlanBench-XL and MCP-Atlas loading, execution, and evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from bria.corpus import is_read_only_mcp_tool
from bria.state import (
    CallKey,
    ExecutionRecord,
    PublicTaskView,
    PublicToolObservation,
    UnifiedToolRecord,
    schema_errors,
)


def redact(value):
    if isinstance(value, dict):
        return {
            k: "[REDACTED]"
            if re.search("(?i)api.?key|authorization|access.?token|secret|password", k)
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(
            "(?i)((?:api_?key|access_token|token|key)=)[^&\\s'\"<>]+", "\\1[REDACTED]", value
        )
        value = re.sub("(?i)(bearer\\s+)[A-Za-z0-9._~+/=-]+", "\\1[REDACTED]", value)
    return value


def classify(tool, message):
    """Return diagnostic kind/code/rule; the client policy decides whether to stop."""
    text = message.lower()
    if "zero_results" in text:
        return ("tool", "no_results", "empty_search")
    if "access denied" in text and "allowed director" in text:
        return ("tool", "path_denied", "filesystem_boundary")
    if tool.startswith("filesystem_") and any(
        (
            s in text
            for s in ("enoent", "enotdir", "no such file", "not a directory", "is a directory")
        )
    ):
        return ("tool", "invalid_path", "filesystem_path")
    if tool.startswith("git_") and (
        message.startswith("/")
        or any(
            (s in text for s in ("not a git repository", "invalidgitrepository", "no such path"))
        )
    ):
        return ("tool", "invalid_repository", "git_path")
    if tool.startswith("desktop-commander_") and "path not allowed:" in text:
        return ("tool", "path_denied", "desktop_boundary")
    if tool.startswith("clinicaltrial") and any(
        (s in text for s in ("invalid field", "not a valid field"))
    ):
        return ("tool", "invalid_arguments", "clinicaltrials_fields")
    if tool.startswith("calculator_") and any(
        (
            s in text
            for s in (
                "invalid syntax",
                "unsupported",
                "not allowed",
                "division by zero",
                "unknown identifier:",
            )
        )
    ):
        return ("tool", "invalid_arguments", "calculator_expression")
    if tool.startswith("slack_") and any(
        (
            s in text
            for s in (
                "not found",
                "unable to parse date",
                "invalid 'during'",
                "invalid channel format:",
            )
        )
    ):
        return ("tool", "invalid_lookup", "slack_lookup")
    if "api key is limited" in text or any(
        (s in text for s in ("upgrade to", "payment required", "insufficient credits"))
    ):
        return ("api", "api_entitlement", "provider_entitlement")
    if any(
        (
            s in text
            for s in ("429", "rate limit", "rate-limit", "too many requests", "quota exceeded")
        )
    ):
        return ("api", "rate_limit", "provider_rate_limit")
    if any(
        (
            s in text
            for s in ("unauthorized", "invalid api key", "authentication", "401", "403 forbidden")
        )
    ):
        return ("api", "api_authentication", "provider_authentication")
    if any(
        (
            s in text
            for s in (
                "cannot connect",
                "connection",
                "timed out",
                "timeout",
                "ssl:",
                "nearby pois: 406",
            )
        )
    ):
        return ("api", "api_connection", "provider_connection")
    if "invalid response shape" in text:
        return ("api", "invalid_api_response", "response_validation")
    if (
        tool.startswith("twelvedata_")
        and "twelve data http 404:" in text
        and ("**symbol** or **figi** parameter is missing or invalid" in text)
    ):
        return ("tool", "invalid_lookup", "twelvedata_symbol_lookup")
    if re.search(
        "client error ['\"]?[45]\\d\\d|server error ['\"]?5\\d\\d|http[^\\n]{0,12}[45]\\d\\d", text
    ):
        return ("api", "upstream_http_error", "upstream_http")
    if any(
        (
            s in text
            for s in (
                "validation failed",
                "input validation",
                "invalid argument",
                "required argument",
                "validation error for",
            )
        )
    ):
        return ("tool", "invalid_arguments", "input_validation")
    return ("api", "unclassified_tool_error", "unrecognized_error_conservative_stop")


class MCPCallGate:
    def __init__(self, max_in_flight=1):
        if type(max_in_flight) is not int or max_in_flight < 1:
            raise ValueError("max_in_flight must be a positive integer")
        self._slots = threading.BoundedSemaphore(max_in_flight)
        self._lock = threading.Lock()
        self._servers = {}
        self._failure = None

    @property
    def interruption(self):
        with self._lock:
            return self._failure

    def call(self, server, operation):
        with self._lock:
            server_lock = self._servers.setdefault(server, threading.Lock())
        with server_lock, self._slots:
            if self.interruption:
                return PublicToolObservation(
                    False,
                    None,
                    "dispatch_blocked",
                    "MCP transport requires reconciliation",
                    interrupts_task=True,
                    dispatched=False,
                    error_origin="transport",
                )
            try:
                observation = operation()
            except BaseException:
                with self._lock:
                    self._failure = PublicToolObservation(
                        False,
                        None,
                        "unknown_outcome",
                        "MCP dispatch outcome is unknown",
                        interrupts_task=True,
                        error_origin="transport",
                    )
                raise
            if observation.error_origin in {"tool", "api"} and observation.dispatched is not False:
                observation = replace(observation, interrupts_task=False)
            elif observation.interrupts_task or observation.error_origin in {
                "transport",
                "gateway",
            }:
                observation = replace(observation, interrupts_task=True)
                with self._lock:
                    self._failure = observation
            return observation


class PlanBenchTaskLoader:
    """Creates controller views without copying target types, answers, or paths."""

    def __init__(self, query_path: str | Path, records: Sequence[UnifiedToolRecord]) -> None:
        payload = json.loads(Path(query_path).read_text(encoding="utf-8"))
        self.dataset_sha256 = hashlib.sha256(Path(query_path).read_bytes()).hexdigest()
        if not isinstance(payload, list):
            raise TypeError("PlanBench query file must contain an array")
        self._queries = {
            str(item["query_id"]): item for item in payload if isinstance(item, Mapping)
        }
        self._data_dir = Path(query_path).parent
        self._corpus_ids = tuple((record.global_tool_id for record in records))
        self._allowed = frozenset(
            (record.global_tool_id for record in records if record.source_family == "planbench_xl")
        )

    def public_task(self, query_id: str) -> PublicTaskView:
        raw = self._queries[query_id]
        query_text = str(raw["query_text"])
        return PublicTaskView(
            task_id=query_id,
            benchmark="planbench_xl",
            user_query=query_text,
            allowed_tool_ids=self._allowed,
        )

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._queries)

    def evaluation_record(self, query_id: str) -> Mapping[str, object]:
        """Evaluation-only access; never pass this object to a controller or planner."""
        return self._queries[query_id]

    def native_execution_state(self, query_id: str) -> Mapping[str, object]:
        """Private runtime inputs for the native gate, excluding targets and answers."""
        raw = self._queries[query_id]
        initial_values = raw.get("input_values")
        return {
            "input_datatypes": tuple((str(value) for value in raw.get("input_datatypes", ()))),
            "input_values": dict(initial_values) if isinstance(initial_values, Mapping) else {},
        }


@dataclass
class _NativeRuntimeState:
    current_datatypes: set[str]
    trusted_values_by_datatype: dict[str, set[str]] = field(default_factory=dict)
    untrusted_values_by_datatype: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_initial(cls, initial: Mapping[str, object]) -> "_NativeRuntimeState":
        from env.core.utils import normalize_runtime_value

        trusted: dict[str, set[str]] = {}
        values = initial.get("input_values")
        if isinstance(values, Mapping):
            for datatype, value in values.items():
                if isinstance(datatype, str):
                    trusted.setdefault(datatype, set()).add(normalize_runtime_value(value))
        return cls(
            current_datatypes={str(value) for value in initial.get("input_datatypes", ())},
            trusted_values_by_datatype=trusted,
        )

    def clone(self) -> "_NativeRuntimeState":
        return _NativeRuntimeState(
            current_datatypes=set(self.current_datatypes),
            trusted_values_by_datatype={
                key: set(values) for key, values in self.trusted_values_by_datatype.items()
            },
            untrusted_values_by_datatype={
                key: set(values) for key, values in self.untrusted_values_by_datatype.items()
            },
        )

    def rejected_inputs(
        self, input_datatypes: Sequence[str], arguments: Mapping[str, object]
    ) -> tuple[list[str], list[str]]:
        from env.core.utils import normalize_runtime_value

        missing = [
            datatype for datatype in input_datatypes if datatype not in self.current_datatypes
        ]
        unsafe = []
        for datatype in input_datatypes:
            normalized = normalize_runtime_value(arguments.get(datatype))
            if normalized in self.untrusted_values_by_datatype.get(
                datatype, set()
            ) and normalized not in self.trusted_values_by_datatype.get(datatype, set()):
                unsafe.append(datatype)
        return (missing, unsafe)

    def record(self, result: Mapping[str, object]) -> None:
        from env.core.utils import normalize_runtime_value

        value = result.get("output_value")
        if value is None:
            return
        datatype = str(result["output_datatype"])
        normalized = normalize_runtime_value(value)
        if result.get("output_provenance") == "trusted":
            self.current_datatypes.add(datatype)
            self.trusted_values_by_datatype.setdefault(datatype, set()).add(normalized)
        elif result.get("output_provenance") == "untrusted":
            self.untrusted_values_by_datatype.setdefault(datatype, set()).add(normalized)


class PlanBenchExecutor:
    """Official backend plus the native pre-execution datatype/provenance gate."""

    def __init__(
        self,
        planbench_root: str | Path,
        *,
        native_initial_state: Mapping[str, object] | None = None,
    ) -> None:
        root = Path(planbench_root).resolve()
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from env.domains.executor import DomainToolExecutor
        from env.domains.registry import load_all_databases, load_tool_universe

        data_root = src / "data"
        universe = load_tool_universe(
            data_root, categories=("baseline", "noisy", "blocker"), strict=True
        )
        self._tools = {f"planbench_xl::{name}": dict(spec) for name, spec in universe.tools.items()}
        self._executor = DomainToolExecutor(load_all_databases(data_root))
        self._initial_state = (
            _NativeRuntimeState.from_initial(native_initial_state)
            if native_initial_state is not None
            else None
        )
        self._state = self._initial_state.clone() if self._initial_state is not None else None
        self.evaluation_log: list[dict] = []

    def _state_for_ancestor_attempts(
        self, attempt_ids: Sequence[int]
    ) -> _NativeRuntimeState | None:
        if self._initial_state is None:
            return None
        selected = set(attempt_ids)
        state = self._initial_state.clone()
        for event in self.evaluation_log:
            if event.get("attempt_id") not in selected or event.get("gate_status") != "accepted":
                continue
            result = event.get("result")
            if isinstance(result, Mapping):
                state.record(result)
        return state

    def execute_at_node(
        self,
        task: PublicTaskView,
        call: CallKey,
        *,
        step_id: int,
        node_id: str,
        ancestor_attempt_ids: Sequence[int],
    ) -> PublicToolObservation:
        """Execute against the selected actor lineage, never against sibling state."""
        return self._execute(
            task,
            call,
            state=self._state_for_ancestor_attempts(ancestor_attempt_ids),
            event_metadata={"attempt_id": step_id, "actor_scope": node_id},
        )

    def execute(self, task: PublicTaskView, call: CallKey) -> PublicToolObservation:
        return self._execute(task, call, state=self._state)

    def _execute(
        self,
        task: PublicTaskView,
        call: CallKey,
        *,
        state: _NativeRuntimeState | None,
        event_metadata: Mapping[str, object] | None = None,
    ) -> PublicToolObservation:
        spec = self._tools.get(call.tool_id)
        if spec is None:
            return PublicToolObservation(
                False, None, "outside_task_boundary", "Unknown PlanBench tool"
            )
        if task.benchmark == "planbench_xl" and state is None:
            return PublicToolObservation(
                False,
                None,
                "native_state_unavailable",
                "PlanBench native execution state is unavailable",
                unexpected_error=True,
                error_origin="planbench_adapter",
                dispatched=False,
            )
        mapping = spec.get("surface_argument_to_datatype") or {}
        expected_keys = list(mapping)
        if not expected_keys:
            expected_keys = list((spec.get("parameters") or {}).get("properties") or {})
        if not expected_keys:
            expected_keys = list(spec["input_datatypes"])
        if set(call.arguments) != set(expected_keys):
            return PublicToolObservation(
                False,
                None,
                "schema_rejection",
                "Arguments do not match the public input schema",
                dispatched=False,
            )
        canonical_arguments = {
            mapping.get(key, key): value for key, value in call.arguments.items()
        }
        metadata = dict(event_metadata or {})
        if state is not None:
            missing, unsafe = state.rejected_inputs(spec["input_datatypes"], canonical_arguments)
            if missing or unsafe:
                self.evaluation_log.append(
                    {
                        "tool_id": call.tool_id,
                        "input_datatypes": spec["input_datatypes"],
                        "canonical_arguments": canonical_arguments,
                        "gate_status": "rejected",
                        "gate_error_type": "untrusted_input_rejected"
                        if unsafe
                        else "missing_required_input",
                        "missing_input_datatypes": sorted(missing),
                        "untrusted_input_datatypes": sorted(unsafe),
                        "backend_dispatched": False,
                        "result": None,
                        **metadata,
                    }
                )
                return PublicToolObservation(
                    False,
                    None,
                    "required_input_unavailable",
                    "Required trusted input is unavailable",
                    error_origin="planbench_native_gate",
                    dispatched=False,
                )
        try:
            result = self._executor.execute_tool(spec, canonical_arguments)
        except (KeyError, TypeError, ValueError) as exc:
            return PublicToolObservation(False, None, "schema_rejection", str(exc))
        except Exception as exc:
            return PublicToolObservation(
                False, None, "unclassified_exception", str(exc), unexpected_error=True
            )
        serialized = asdict(result)
        if state is not None:
            state.record(serialized)
        self.evaluation_log.append(
            {
                "tool_id": call.tool_id,
                "input_datatypes": spec["input_datatypes"],
                "canonical_arguments": canonical_arguments,
                "gate_status": "accepted" if state is not None else "not_applicable",
                "backend_dispatched": True,
                "result": serialized,
                **metadata,
            }
        )
        if not result.success:
            return PublicToolObservation(
                False,
                None,
                "tool_execution_failed",
                "Official executor reported failure",
                dispatched=True,
            )
        if result.output_value is None:
            return PublicToolObservation(
                False, None, "no_result", "No result exists for this binding", dispatched=True
            )
        return PublicToolObservation(True, result.output_value, dispatched=True)


class PlanBenchEvaluator:
    """Evaluation-only gold access kept outside the controller dependency graph."""

    def __init__(self, planbench_root: str | Path) -> None:
        root = Path(planbench_root).resolve()
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from env.core.answer_judges import build_answer_judge

        self._judge = build_answer_judge("normalized_contains")

    def answer_correct(self, evaluation_record: Mapping[str, object], answer: object) -> bool:
        return self._judge.is_correct(evaluation_record.get("correct_answer"), answer)

    def native_checks(
        self,
        evaluation_record: Mapping[str, object],
        answer: object,
        executions: Sequence[Mapping[str, object]],
    ) -> dict:
        """Audit online native gates and apply final-answer provenance checks.

        This does not simulate native retrieval, error feedback, blockers or step budgets.
        Canonical metadata is evaluation-only and never returned to the solver.
        """
        from env.core.answer_judges import normalize_answer
        from env.core.utils import normalize_runtime_value

        datatypes = set(evaluation_record.get("input_datatypes", []))
        initial = evaluation_record.get("input_values") or {}
        trusted = {key: {normalize_runtime_value(value)} for key, value in initial.items()}
        untrusted, sources, rejected, disagreements = ({}, {}, [], [])
        for event in executions:
            arguments = event["canonical_arguments"]
            missing = set(event["input_datatypes"]) - datatypes
            unsafe = [
                key
                for key in event["input_datatypes"]
                if normalize_runtime_value(arguments.get(key)) in untrusted.get(key, set())
                and normalize_runtime_value(arguments.get(key)) not in trusted.get(key, set())
            ]
            expected_rejection = bool(missing or unsafe)
            gate_status = event.get("gate_status")
            if gate_status == "rejected" and (not expected_rejection):
                disagreements.append(
                    {"tool_id": event["tool_id"], "reason": "unexpected_online_rejection"}
                )
                continue
            if gate_status == "accepted" and expected_rejection:
                disagreements.append(
                    {"tool_id": event["tool_id"], "reason": "online_gate_accepted_invalid_call"}
                )
            if expected_rejection:
                rejected.append(
                    {
                        "tool_id": event["tool_id"],
                        "missing_datatypes": sorted(missing),
                        "untrusted_input_datatypes": unsafe,
                    }
                )
                continue
            if gate_status == "rejected":
                continue
            result = event["result"]
            if not isinstance(result, Mapping):
                disagreements.append(
                    {"tool_id": event["tool_id"], "reason": "accepted_event_missing_result"}
                )
                continue
            value = result.get("output_value")
            if value is None:
                continue
            datatype = result["output_datatype"]
            normalized = normalize_runtime_value(value)
            if result.get("output_provenance") == "trusted":
                datatypes.add(datatype)
                trusted.setdefault(datatype, set()).add(normalized)
            elif result.get("output_provenance") == "untrusted":
                untrusted.setdefault(datatype, set()).add(normalized)
                source = result.get("untrusted_source_type")
                if source is not None and normalize_answer(value):
                    sources.setdefault(datatype, {}).setdefault(normalize_answer(value), set()).add(
                        source
                    )
        target = evaluation_record.get("target_datatype")
        matched = {
            source
            for value, origin in sources.get(target, {}).items()
            if value in normalize_answer(answer)
            for source in origin
        }
        reached = target in datatypes
        correct = self.answer_correct(evaluation_record, answer)
        consistent = not disagreements
        return {
            "answer_correct": correct,
            "target_datatype_reached": reached,
            "matched_untrusted_source_types": sorted(matched),
            "native_input_gate_rejections": rejected,
            "online_gate_consistent": consistent,
            "online_gate_disagreements": disagreements,
            "datatype_check_passed": reached and (not matched),
            "native_criteria_success": bool(correct and reached and (not matched) and consistent),
        }


def _post_json(url: str, payload: Mapping[str, object], timeout: float) -> object:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    token = os.environ.get("SANDBOX_AUTH_TOKEN")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_schema_envelope(
    result: object, output_schema: Mapping[str, object] | None
) -> object:
    """Add a uniquely implied public-schema envelope without semantic repair."""
    if not output_schema or not schema_errors(result, output_schema):
        return result
    if output_schema.get("type") != "object":
        return result
    properties = output_schema.get("properties")
    required = output_schema.get("required")
    if (
        not isinstance(properties, Mapping)
        or not isinstance(required, list)
        or len(required) != 1
        or (not isinstance(required[0], str))
        or (required[0] not in properties)
    ):
        return result
    wrapper_key = required[0]
    if isinstance(result, Mapping) and wrapper_key in result:
        return result
    if not isinstance(properties[wrapper_key], Mapping):
        return result
    candidates: list[object] = [{wrapper_key: result}, {wrapper_key: [result]}]
    valid = [candidate for candidate in candidates if not schema_errors(candidate, output_schema)]
    return valid[0] if len(valid) == 1 else result


class MCPAtlasTaskLoader:
    """Public task loader with evaluation-only gold access kept separate."""

    def __init__(self, path: str | Path, records: Sequence[UnifiedToolRecord]) -> None:
        task_path = Path(path)
        self.dataset_sha256 = hashlib.sha256(task_path.read_bytes()).hexdigest()
        if task_path.suffix.casefold() == ".parquet":
            try:
                from pyarrow import parquet
            except ImportError as exc:
                raise RuntimeError(
                    "Install the mcp-data extra to read the official Parquet dataset"
                ) from exc
            raw_rows = parquet.read_table(task_path).to_pylist()
        else:
            with task_path.open(newline="", encoding="utf-8") as handle:
                raw_rows = list(csv.DictReader(handle))
        self._rows = {str(row["TASK"]): dict(row) for row in raw_rows}
        self._corpus_ids = tuple((record.global_tool_id for record in records))
        self._id_by_executor = {
            record.executor_key: record.global_tool_id
            for record in records
            if record.source_family == "mcp_atlas" and is_read_only_mcp_tool(record.executor_key)
        }

    def public_task(self, task_id: str) -> PublicTaskView:
        row = self._rows[task_id]
        prompt = str(row["PROMPT"])
        enabled_raw = row.get("ENABLED_TOOLS", [])
        if isinstance(enabled_raw, str):
            try:
                enabled = json.loads(enabled_raw)
            except json.JSONDecodeError:
                enabled = []
        else:
            enabled = enabled_raw
        enabled_names = (
            [
                item if isinstance(item, str) else item.get("name")
                for item in enabled
                if isinstance(item, (str, Mapping))
            ]
            if isinstance(enabled, list)
            else []
        )
        allowed_ids = frozenset(
            (
                self._id_by_executor[name]
                for name in enabled_names
                if isinstance(name, str) and name in self._id_by_executor
            )
        )
        return PublicTaskView(
            task_id=task_id, benchmark="mcp_atlas", user_query=prompt, allowed_tool_ids=allowed_ids
        )

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._rows)

    def evaluation_record(self, task_id: str) -> Mapping[str, object]:
        """Evaluation-only row containing claims/trajectory; never pass to planning."""
        return self._rows[task_id]


class MCPAtlasExecutor:
    def __init__(
        self,
        records: Sequence[UnifiedToolRecord],
        *,
        base_url: str = "http://localhost:1984",
        timeout: float = 120.0,
        call_gate: MCPCallGate | None = None,
    ) -> None:
        self._executor_by_id = {
            record.global_tool_id: record.executor_key
            for record in records
            if record.source_family == "mcp_atlas"
        }
        self._output_schema_by_id = {
            record.global_tool_id: record.output_schema
            for record in records
            if record.source_family == "mcp_atlas"
        }
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._call_gate = call_gate or MCPCallGate()
        self._interruption = None
        mcp_records = [record for record in records if record.source_family == "mcp_atlas"]
        named_servers = all(
            (
                record.server_name and record.executor_key.startswith(record.server_name + "_")
                for record in mcp_records
            )
        )
        self._server_by_id = {
            record.global_tool_id: record.server_name if named_servers else "unknown-server"
            for record in mcp_records
        }

    def execute(self, task: PublicTaskView, call: CallKey) -> PublicToolObservation:
        executor_key = self._executor_by_id.get(call.tool_id)
        if executor_key is None:
            return PublicToolObservation(False, None, "outside_task_boundary", "Unknown MCP tool")
        if not is_read_only_mcp_tool(executor_key):
            return PublicToolObservation(
                False,
                None,
                "permission_unavailable",
                "MCP tool is blocked by the BRIA read-only policy",
            )
        observation = self._call_gate.call(
            self._server_by_id.get(call.tool_id, "unknown-server"),
            lambda: self._request(
                executor_key, call.arguments, self._output_schema_by_id.get(call.tool_id)
            ),
        )
        if observation.interrupts_task:
            self._interruption = observation
        return observation

    @property
    def interruption(self):
        return self._interruption or self._call_gate.interruption

    def _request(
        self,
        executor_key: str,
        arguments: Mapping[str, object],
        output_schema: Mapping[str, object] | None,
    ) -> PublicToolObservation:
        try:
            payload = _post_json(
                f"{self.base_url}/call-tool",
                {"tool_name": executor_key, "tool_args": arguments, "use_cache": False},
                self.timeout,
            )
        except urllib.error.HTTPError as exc:
            message = redact(exc.read().decode("utf-8", errors="replace"))
            code = {
                400: "schema_rejection",
                403: "permission_unavailable",
                404: "outside_task_boundary",
                422: "schema_rejection",
                429: "rate_limit",
            }.get(exc.code, "server_error" if exc.code >= 500 else "transport_error")
            return PublicToolObservation(
                False,
                None,
                code,
                message,
                unexpected_error=exc.code not in {400, 422},
                http_status=exc.code,
                raw_response=message,
                interrupts_task=exc.code not in {400, 422},
                error_origin="tool" if exc.code in {400, 422} else "gateway",
                dispatched=True,
            )
        except TimeoutError as exc:
            return PublicToolObservation(
                False,
                None,
                "timeout",
                redact(str(exc)),
                unexpected_error=True,
                interrupts_task=True,
                error_origin="transport",
                dispatched=True,
            )
        except (OSError, urllib.error.URLError) as exc:
            return PublicToolObservation(
                False,
                None,
                "transport_error",
                redact(str(exc)),
                unexpected_error=True,
                interrupts_task=True,
                error_origin="transport",
                dispatched=True,
            )
        except (ValueError, TypeError) as exc:
            return PublicToolObservation(
                False,
                None,
                "invalid_gateway_response",
                redact(str(exc)),
                unexpected_error=True,
                interrupts_task=True,
                error_origin="gateway",
                dispatched=True,
            )
        payload = redact(payload)
        if isinstance(payload, Mapping) and payload.get("isError"):
            message = json.dumps(payload, ensure_ascii=False)
            kind, code, _ = classify(executor_key, message)
            return PublicToolObservation(
                False,
                None,
                code,
                message,
                unexpected_error=kind != "tool",
                raw_response=payload,
                interrupts_task=kind != "tool",
                error_origin=kind,
                dispatched=True,
            )
        raw_payload = payload
        if isinstance(payload, Mapping) and payload.get("structuredContent") is not None:
            result: object = payload["structuredContent"]
        else:
            if isinstance(payload, Mapping) and "content" in payload:
                payload = payload["content"]
            result = payload
            if isinstance(payload, list):
                texts = [
                    item.get("text")
                    for item in payload
                    if isinstance(item, Mapping)
                    and item.get("type") == "text"
                    and (item.get("text") is not None)
                ]
                if texts:
                    if len(texts) == 1:
                        try:
                            result = json.loads(str(texts[0]))
                        except json.JSONDecodeError:
                            result = texts[0]
                    else:
                        result = texts
        result = _normalize_schema_envelope(result, output_schema)
        return PublicToolObservation(True, result, raw_response=raw_payload, dispatched=True)


def trajectory_from_executions(
    executions: Sequence[ExecutionRecord], records: Sequence[UnifiedToolRecord]
) -> list[dict[str, object]]:
    executor_by_id = {record.global_tool_id: record.executor_key for record in records}
    trajectory: list[dict[str, object]] = []
    for execution in executions:
        observation = execution.observation
        response: str | None = None
        if observation.result is not None:
            try:
                response = json.dumps(observation.result, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                response = str(observation.result)
        error = None
        if not observation.success or observation.error_code:
            error = (
                ": ".join(
                    (value for value in (observation.error_code, observation.message) if value)
                )
                or "tool call failed"
            )
        trajectory.append(
            {
                "tool_name": executor_by_id.get(
                    execution.call_key.tool_id, execution.call_key.tool_id
                ),
                "parameters": execution.call_key.arguments,
                "response": response,
                "error": error,
                "error_code": observation.error_code,
                "unexpected_error": observation.unexpected_error,
                "http_status": observation.http_status,
            }
        )
    return trajectory


_NATIVE_IMPORT_LOCK = threading.Lock()


class UnionExecutor:
    def __init__(
        self,
        records,
        *,
        planbench_root,
        base_url="http://localhost:1984",
        planbench_native_state=None,
        trust,
        call_gate=None,
    ):
        self._records = {record.global_tool_id: record for record in records}
        self._planbench_root = planbench_root
        self._native_state = planbench_native_state
        self._trust = trust
        self._planbench = None
        self._mcp = MCPAtlasExecutor(records, base_url=base_url, call_gate=call_gate)

    @property
    def interruption(self):
        return self._mcp.interruption

    @property
    def evaluation_log(self):
        return self._planbench.evaluation_log if self._planbench else []

    def _native(self):
        if self._planbench is None:
            with _NATIVE_IMPORT_LOCK:
                if self._planbench is None:
                    self._planbench = PlanBenchExecutor(
                        self._planbench_root, native_initial_state=self._native_state
                    )
        return self._planbench

    def execute(self, task, call):
        record = self._records.get(call.tool_id)
        if record is None:
            return PublicToolObservation(False, None, "unknown_tool", dispatched=False)
        if record.source_family == "mcp_atlas":
            if not self._trust.permits(record):
                return PublicToolObservation(
                    False,
                    None,
                    "unreviewed_tool",
                    "The host has not attested this tool schema and cache contract",
                    dispatched=False,
                )
            return self._mcp.execute(task, call)
        if record.source_family == "planbench_xl":
            return self._native().execute(task, call)
        return PublicToolObservation(False, None, "unsupported_backend", dispatched=False)

    def execute_at_node(self, task, call, *, step_id, node_id, ancestor_attempt_ids):
        record = self._records.get(call.tool_id)
        if record is not None and record.source_family == "planbench_xl":
            return self._native().execute_at_node(
                task,
                call,
                step_id=step_id,
                node_id=node_id,
                ancestor_attempt_ids=ancestor_attempt_ids,
            )
        return self.execute(task, call)
