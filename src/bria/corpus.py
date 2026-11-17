"""Build a public tool corpus from separately obtained benchmark schemas."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

from bria.retrieval import DOCUMENT_TEMPLATE
from bria.state import UnifiedToolRecord

MCPToolAccess = Literal["read", "write", "exec"]
MCP_READ_ONLY_POLICY_VERSION = "name_rules"
_EXEC_TOOLS = frozenset(
    {
        "cli-mcp-server_run_command",
        "e2b-server_run_code",
        "mcp-server-code-runner_run-code",
        "mcp-code-executor_execute_code",
        "mcp-code-executor_execute_code_file",
        "desktop-commander_start_process",
        "desktop-commander_interact_with_process",
    }
)
_WRITE = re.compile(
    "(^|_)(create|update|delete|insert|write|append|edit|move_file|remove|push_files|fork|reset|init|checkout|configure_environment|install_dependencies|initialize|kill|force_terminate|give_feedback|set_config|drop|rename|duplicate|patch|commit)(_|$)|add_observations|add_issue_comment|add_reaction|add_message|post_message|post_page|post_comment|upload|send_email",
    re.IGNORECASE,
)
_READ = re.compile(
    "(^|_)(get|list|search|read|find|show|status|log|diff|schema|describe|count|history|geocode|view|fetch)(_|$)",
    re.IGNORECASE,
)
_HARD_WRITE = re.compile(
    "(^|_)(create|write|update|delete|insert|edit|move_file|push_files|fork|merge|drop|patch|rename)(_|$)|post_message|post_page|post_comment|add_message|add_reaction|send_email|upload|add_observations|add_issue_comment",
    re.IGNORECASE,
)


def classify_mcp_tool(tool_name: str) -> MCPToolAccess:
    """Classify an MCP tool name using the frozen MCP-Atlas read-only policy."""
    normalized = tool_name.replace("-", "_")
    if tool_name in _EXEC_TOOLS:
        return "exec"
    if _READ.search(normalized) and (not _HARD_WRITE.search(normalized)):
        return "read"
    if _WRITE.search(normalized) or re.search("merge_pull_request|create_branch", normalized):
        return "write"
    return "read"


def is_read_only_mcp_tool(tool_name: str) -> bool:
    """Return whether a tool is admitted to the BRIA MCP-Atlas tool universe."""
    return classify_mcp_tool(tool_name) == "read"


DOCUMENT_TEMPLATE_VERSION = "public_schema"


def _clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _type_label(spec: Mapping[str, object]) -> str:
    raw = spec.get("type")
    if isinstance(raw, list):
        return "|".join((str(item) for item in raw))
    if isinstance(raw, str) and raw:
        return raw
    for keyword in ("anyOf", "oneOf"):
        choices = spec.get(keyword)
        if isinstance(choices, list):
            labels = [_type_label(item) for item in choices if isinstance(item, Mapping)]
            labels = [label for label in labels if label != "any"]
            if labels:
                return "|".join(dict.fromkeys(labels))
    return "any"


def _property_lines(schema: Mapping[str, object]) -> list[str]:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return []
    required_raw = schema.get("required")
    required = set(required_raw) if isinstance(required_raw, list) else set()
    lines: list[str] = []
    for name in sorted(properties, key=str):
        raw = properties[name]
        spec = raw if isinstance(raw, Mapping) else {}
        fields = [f"- {_clean(name)}", f"type: {_type_label(spec)}"]
        description = _clean(spec.get("description"))
        if description:
            fields.append(description)
        if name not in required:
            fields.append("optional")
        lines.append(" | ".join(fields))
    return lines


def build_retrieval_document(
    *,
    tool_name: str,
    description: str,
    input_schema: Mapping[str, object],
    output_description: str | None = None,
    output_schema: Mapping[str, object] | None = None,
    server_name: str | None = None,
) -> str:
    """Render only public schema fields with one deterministic cross-benchmark template."""
    sections = [f"Tool name: {_clean(tool_name)}"]
    cleaned_description = _clean(description)
    if cleaned_description:
        sections.append(f"Description: {cleaned_description}")
    inputs = _property_lines(input_schema)
    if inputs:
        sections.extend(["", "Inputs:", *inputs])
    output = _clean(output_description)
    if not output and output_schema:
        output = _clean(output_schema.get("description"))
        if not output:
            output = f"JSON type: {_type_label(output_schema)}"
    if output:
        sections.extend(["", "Outputs:", f"- {output}"])
    cleaned_server = _clean(server_name)
    if cleaned_server:
        sections.extend(["", "Server:", f"- {cleaned_server}"])
    return "\n".join(sections).strip()


def normalized_document(document: str) -> str:
    return re.sub("\\s+", " ", document.strip()).casefold()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def record_to_dict(record: UnifiedToolRecord) -> dict[str, object]:
    return asdict(record)


def record_from_dict(value: Mapping[str, object]) -> UnifiedToolRecord:
    return UnifiedToolRecord(
        global_tool_id=str(value["global_tool_id"]),
        tool_name=str(value["tool_name"]),
        description=str(value.get("description") or ""),
        input_schema=dict(value.get("input_schema") or {}),
        output_schema=dict(value["output_schema"])
        if isinstance(value.get("output_schema"), Mapping)
        else None,
        output_description=str(value["output_description"])
        if value.get("output_description") is not None
        else None,
        source_family=str(value["source_family"]),
        source_id=str(value["source_id"]),
        server_name=str(value["server_name"]) if value.get("server_name") is not None else None,
        executor_key=str(value["executor_key"]),
        public_schema_hash=str(value["public_schema_hash"]),
        retrieval_document=str(value["retrieval_document"]),
        duplicate_group_id=str(value["duplicate_group_id"])
        if value.get("duplicate_group_id") is not None
        else None,
    )


def assign_duplicate_groups(records: Iterable[UnifiedToolRecord]) -> list[UnifiedToolRecord]:
    records_list = list(records)
    by_document: dict[str, list[int]] = {}
    for index, record in enumerate(records_list):
        digest = hashlib.sha256(
            normalized_document(record.retrieval_document).encode("utf-8")
        ).hexdigest()
        by_document.setdefault(digest, []).append(index)
    result = list(records_list)
    for digest, indices in by_document.items():
        if len(indices) < 2:
            continue
        group_id = f"exact::{digest[:20]}"
        for index in indices:
            result[index] = replace(result[index], duplicate_group_id=group_id)
    return result


DEFAULT_CATEGORIES = ("baseline", "noisy", "blocker")
_FILENAMES = {
    "baseline": "baseline_tools.json",
    "noisy": "noisy_tools.json",
    "blocker": "blocker_tools.json",
}
FORBIDDEN_PUBLIC_FIELDS = frozenset(
    {
        "input_datatypes",
        "output_datatype",
        "surface_argument_to_datatype",
        "datatype_to_surface_argument",
        "internal_tool_name",
        "internal_baseline_tool_name",
        "baseline_tool_name",
        "noise_type",
        "tool_type",
        "return_value",
        "correct_answer",
        "target_datatype",
        "one_available_path",
        "paths_set_catalog",
        "output_provenance",
        "blocker",
    }
)


def _load_array(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected an array in {path}")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _surface_output(raw: Mapping[str, object]) -> tuple[dict[str, object] | None, str | None]:
    value = raw.get("surface_output")
    if not isinstance(value, Mapping):
        return (None, None)
    schema: dict[str, object] = {}
    for key in ("type", "description"):
        if value.get(key) is not None:
            schema[key] = value[key]
    description = value.get("description") or value.get("label")
    return (schema or None, str(description) if description else None)


def _record(raw: Mapping[str, object], *, domain: str, category: str) -> UnifiedToolRecord:
    name = raw.get("name") or raw.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{domain}/{category} tool is missing a public name")
    description = str(raw.get("description") or "")
    parameters = raw.get("parameters")
    input_schema = dict(parameters) if isinstance(parameters, Mapping) else {"type": "object"}
    output_schema, output_description = _surface_output(raw)
    retrieval_document = build_retrieval_document(
        tool_name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        output_description=output_description,
    )
    public_schema = {
        "name": name,
        "description": description,
        "parameters": input_schema,
        "output_schema": output_schema,
        "output_description": output_description,
    }
    return UnifiedToolRecord(
        global_tool_id=f"planbench_xl::{name}",
        tool_name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        output_description=output_description,
        source_family="planbench_xl",
        source_id=f"{domain}:{category}",
        server_name=None,
        executor_key=name,
        public_schema_hash=sha256_json(public_schema),
        retrieval_document=retrieval_document,
    )


def ingest_planbench(
    data_root: str | Path,
    *,
    domains: Iterable[str] | None = None,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
) -> list[UnifiedToolRecord]:
    root = Path(data_root)
    requested_domains = set(domains) if domains is not None else None
    records: list[UnifiedToolRecord] = []
    for domain_dir in sorted((path for path in root.iterdir() if path.is_dir())):
        if requested_domains is not None and domain_dir.name not in requested_domains:
            continue
        for category in categories:
            if category not in _FILENAMES:
                raise ValueError(f"Unknown PlanBench category: {category}")
            path = domain_dir / _FILENAMES[category]
            if not path.exists():
                continue
            records.extend(
                (
                    _record(raw, domain=domain_dir.name, category=category)
                    for raw in _load_array(path)
                )
            )
    records.sort(key=lambda item: item.global_tool_id)
    ids = [record.global_tool_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("PlanBench tool names do not produce collision-free global IDs")
    return records


FORBIDDEN_MCP_FIELDS = frozenset(
    {
        "claims",
        "GTFA_CLAIMS",
        "reference_trajectory",
        "TRAJECTORY",
        "reference_answer",
        "evaluator_diagnostics",
    }
)


def _fetch_json(url: str, *, method: str = "GET", timeout: float = 60.0) -> object:
    request = urllib.request.Request(url, data=b"" if method == "POST" else None, method=method)
    request.add_header("Content-Type", "application/json")
    token = os.environ.get("SANDBOX_AUTH_TOKEN")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_live_schemas(base_url: str, *, timeout: float = 60.0) -> list[dict[str, object]]:
    payload = _fetch_json(f"{base_url.rstrip('/')}/list-tools", method="POST", timeout=timeout)
    if not isinstance(payload, list):
        raise TypeError("MCP /list-tools response must be an array")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def fetch_enabled_servers(base_url: str, *, timeout: float = 10.0) -> list[str]:
    payload = _fetch_json(f"{base_url.rstrip('/')}/enabled-servers", timeout=timeout)
    if not isinstance(payload, Mapping):
        return []
    servers = payload.get("servers")
    if isinstance(servers, list):
        return sorted(
            (
                str(row[0])
                for row in servers
                if isinstance(row, list) and len(row) >= 2 and (row[1] == "OK")
            )
        )
    enabled = payload.get("enabled_servers")
    return sorted((str(item) for item in enabled)) if isinstance(enabled, list) else []


def write_schema_snapshot(schemas: Sequence[Mapping[str, object]], path: str | Path) -> None:
    ordered = sorted((dict(item) for item in schemas), key=lambda item: str(item.get("name")))
    Path(path).write_text(
        json.dumps(ordered, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def load_schema_snapshot(path: str | Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("MCP schema snapshot must contain an array")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def split_namespaced_tool(tool_name: str, server_names: Sequence[str]) -> tuple[str, str]:
    matches = [server for server in server_names if tool_name.startswith(f"{server}_")]
    if not matches:
        explicit_server, separator, bare_name = tool_name.partition("::")
        if separator and explicit_server and bare_name:
            return (explicit_server, bare_name)
        raise ValueError(
            f"Cannot infer MCP server for {tool_name!r}; pass the live enabled server list"
        )
    server = max(matches, key=lambda item: (len(item), item))
    return (server, tool_name[len(server) + 1 :])


def ingest_mcp_atlas_schemas(
    schemas: Iterable[Mapping[str, object]],
    *,
    server_names: Sequence[str],
    excluded_servers: Iterable[str] = ("planbench-xl",),
    access_policy: str = "read_only",
) -> list[UnifiedToolRecord]:
    if access_policy != "read_only":
        raise ValueError("Unknown MCP access policy")
    admitted = is_read_only_mcp_tool
    excluded = set(excluded_servers)
    records: list[UnifiedToolRecord] = []
    for raw in schemas:
        tool_name = raw.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            continue
        if not admitted(tool_name):
            continue
        explicit_server = raw.get("server")
        if isinstance(explicit_server, str) and explicit_server:
            server = explicit_server
            prefix = f"{server}_"
            bare_name = tool_name.removeprefix(prefix)
        else:
            server, bare_name = split_namespaced_tool(tool_name, server_names)
        if server in excluded:
            continue
        description = str(raw.get("description") or "")
        input_raw = raw.get("inputSchema") or raw.get("input_schema")
        input_schema = dict(input_raw) if isinstance(input_raw, Mapping) else {"type": "object"}
        output_raw = raw.get("outputSchema") or raw.get("output_schema")
        output_schema = dict(output_raw) if isinstance(output_raw, Mapping) else None
        output_description = None
        if output_schema and output_schema.get("description"):
            output_description = str(output_schema["description"])
        retrieval_document = build_retrieval_document(
            tool_name=bare_name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            output_description=output_description,
            server_name=server,
        )
        public_schema = {
            "server": server,
            "name": bare_name,
            "description": description,
            "inputSchema": input_schema,
            "outputSchema": output_schema,
        }
        records.append(
            UnifiedToolRecord(
                global_tool_id=f"mcp_atlas::{server}::{bare_name}",
                tool_name=bare_name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                output_description=output_description,
                source_family="mcp_atlas",
                source_id=server,
                server_name=server,
                executor_key=tool_name,
                public_schema_hash=sha256_json(public_schema),
                retrieval_document=retrieval_document,
            )
        )
    records.sort(key=lambda item: item.global_tool_id)
    ids = [record.global_tool_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("MCP schemas do not produce collision-free global IDs")
    return records


def _validate(records):
    ids = [record.global_tool_id for record in records]
    if ids != sorted(set(ids)):
        raise ValueError("Corpus tool IDs must be unique and sorted")
    if any((record.source_family not in {"planbench_xl", "mcp_atlas"} for record in records)):
        raise ValueError("Unknown benchmark source")
    if any(
        (
            record.source_family == "mcp_atlas" and (not is_read_only_mcp_tool(record.executor_key))
            for record in records
        )
    ):
        raise ValueError("Corpus contains a mutating or code-execution MCP tool")


def build_union_corpus(planbench_records, mcp_records):
    records = sorted(
        assign_duplicate_groups([*planbench_records, *mcp_records]),
        key=lambda record: record.global_tool_id,
    )
    _validate(records)
    if {record.source_family for record in records} != {"planbench_xl", "mcp_atlas"}:
        raise ValueError("The shared corpus requires both benchmark families")
    return records


def write_corpus(records, output_dir, *, provenance):
    records = list(records)
    _validate(records)
    body = "".join((canonical_json(record_to_dict(record)) + "\n" for record in records)).encode()
    manifest = dict(
        corpus_sha256=hashlib.sha256(body).hexdigest(),
        document_template_version=DOCUMENT_TEMPLATE,
        access_policy="read_only",
        tool_count=len(records),
        provenance=provenance,
        tool_count_planbench=sum((r.source_family == "planbench_xl" for r in records)),
        tool_count_mcp_atlas=sum((r.source_family == "mcp_atlas" for r in records)),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "union_tools.jsonl").write_bytes(body)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_verified_corpus(path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    body = Path(path).read_bytes()
    if hashlib.sha256(body).hexdigest() != manifest.get("corpus_sha256"):
        raise ValueError("Corpus checksum mismatch")
    if (
        manifest.get("document_template_version") != DOCUMENT_TEMPLATE
        or manifest.get("access_policy") != "read_only"
    ):
        raise ValueError("Corpus template or access policy mismatch")
    records = [
        record_from_dict(json.loads(line)) for line in body.decode().splitlines() if line.strip()
    ]
    _validate(records)
    counts = dict(
        tool_count=len(records),
        tool_count_planbench=sum((r.source_family == "planbench_xl" for r in records)),
        tool_count_mcp_atlas=sum((r.source_family == "mcp_atlas" for r in records)),
    )
    if any((manifest.get(key) != value for key, value in counts.items())):
        raise ValueError("Corpus manifest counts differ from records")
    return (records, manifest)
