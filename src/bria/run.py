"""Model transport and command-line entry points for BRIA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path

import httpx

from bria import Controller, ExecutionTrust
from bria.corpus import (
    build_union_corpus,
    fetch_enabled_servers,
    fetch_live_schemas,
    ingest_mcp_atlas_schemas,
    ingest_planbench,
    load_schema_snapshot,
    load_verified_corpus,
    write_corpus,
)
from bria.retrieval import (
    ExactVectorIndex,
    IndexManifest,
    Qwen3Embedder,
    UnifiedRetriever,
    load_config,
)
from bria.state import digest, strict_json


class JSONCompletion:
    def __init__(
        self,
        *,
        model,
        base_url,
        api_key,
        max_tokens=8192,
        temperature=None,
        reasoning_effort=None,
        timeout=1800.0,
        client=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)
        self.last_usage = None
        self.last_content = None
        self.last_reasoning = None
        self.last_finish_reason = None
        self.last_http_error = None

    def complete(self, *, messages, response_format):
        if response_format != {"type": "json_object"}:
            raise ValueError("BRIA requires the provider JSON-object response format")
        self.last_usage = self.last_content = self.last_reasoning = None
        self.last_finish_reason = self.last_http_error = None
        body = dict(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
        )
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        try:
            response = self._client.post(
                self.base_url + "/chat/completions",
                json=body,
                headers={"Authorization": "Bearer " + self._api_key},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self.last_http_error = {"status": exc.response.status_code}
            raise RuntimeError(f"Provider HTTP {exc.response.status_code}") from None
        except httpx.TransportError:
            self.last_http_error = {"status": None, "kind": "transport_error"}
            raise RuntimeError("Provider transport failed") from None
        payload = response.json()
        self.last_usage = payload.get("usage")
        choice = payload["choices"][0]
        self.last_finish_reason = choice.get("finish_reason")
        message = choice["message"]
        self.last_content = message.get("content")
        self.last_reasoning = message.get("reasoning_content") or message.get("reasoning")
        if self.last_finish_reason == "length":
            raise ValueError("Provider output reached its token limit")
        if not isinstance(self.last_content, str):
            raise ValueError("Provider returned no JSON text")
        result = strict_json(self.last_content)
        if not isinstance(result, dict):
            raise ValueError("Provider response must be a JSON object")
        return result

    def close(self):
        self._client.close()


def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return result


def _save(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _commit(root):
    top = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != Path(root).resolve():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _fresh(path):
    if path.exists():
        raise ValueError(f"Choose a fresh output directory: {path}")


def _planbench_digest(root):
    root = Path(root)
    if not (root / "src/env").is_dir() or not (root / "src/data").is_dir():
        raise ValueError("Supply the full external PlanBench environment and data")
    paths = sorted([*(root / "src/env").rglob("*.py"), *(root / "src/data").rglob("*.json")])
    return digest(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )


def build_corpus(args):
    _fresh(args.output_dir)
    if args.mcp_schema_snapshot:
        schemas = load_schema_snapshot(args.mcp_schema_snapshot)
        servers = (
            args.mcp_servers.split(",")
            if args.mcp_servers
            else sorted(
                {schema["server"] for schema in schemas if isinstance(schema.get("server"), str)}
            )
        )
        if not servers:
            raise ValueError("Supply --mcp-servers for a snapshot without server metadata")
    else:
        if not args.mcp_url:
            raise ValueError("Supply --mcp-url or --mcp-schema-snapshot")
        schemas = fetch_live_schemas(args.mcp_url)
        servers = fetch_enabled_servers(args.mcp_url)
    records = build_union_corpus(
        ingest_planbench(args.planbench_root / "src/data"),
        ingest_mcp_atlas_schemas(schemas, server_names=servers),
    )
    provenance = dict(
        planbench_commit=_commit(args.planbench_root),
        planbench_tree_sha256=_planbench_digest(args.planbench_root),
        schema_snapshot_sha256=digest(schemas),
    )
    manifest = write_corpus(records, args.output_dir, provenance=provenance)
    _save(args.output_dir / "mcp_schemas.json", schemas)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_index(args):
    _fresh(args.output_dir)
    config = load_config(args.config)
    records, corpus = load_verified_corpus(args.corpus, args.corpus_manifest)
    embedding = config.embedding
    if args.device:
        devices = [item.strip() for item in args.device.split(",") if item.strip()]
        if not devices:
            raise ValueError("Supply at least one encoding device")
        embedding = replace(embedding, device=devices if len(devices) > 1 else devices[0])
    if args.batch_size:
        embedding = replace(embedding, batch_size=args.batch_size)
    embedding.validate()
    embedder = Qwen3Embedder(embedding)
    vectors = embedder.encode_documents([record.retrieval_document for record in records])
    index = ExactVectorIndex([record.global_tool_id for record in records], vectors)
    manifest = IndexManifest(
        model_id=embedder.model_id,
        model_revision=embedder.model_revision,
        query_instruction=embedding.query_instruction,
        normalize_embeddings=True,
        embedding_dimension=index.dimension,
        corpus_sha256=corpus["corpus_sha256"],
        document_template_version=corpus["document_template_version"],
    )
    saved = index.save(args.output_dir, manifest)
    print(json.dumps(asdict(saved), indent=2, sort_keys=True))


def _task_loader(benchmark, dataset, records):
    if benchmark == "planbench_xl":
        from bria.benchmarks import PlanBenchTaskLoader

        return PlanBenchTaskLoader(dataset, records)
    from bria.benchmarks import MCPAtlasTaskLoader

    return MCPAtlasTaskLoader(dataset, records)


def _select_task_ids(benchmark, available_ids, cohort=None, task_ids=(), task_list=None):
    if cohort and (task_ids or task_list is not None):
        raise ValueError("Choose --cohort or custom task IDs (--task-id/--task-list)")
    if cohort == "full":
        selected = list(available_ids)
    elif cohort == "75tasks":
        cohorts = json.loads(files("bria").joinpath("75tasks.json").read_text(encoding="utf-8"))
        selected = cohorts[benchmark]
        if len(selected) != 75 or len(set(selected)) != 75:
            raise ValueError(f"Expected 75 unique bundled task IDs for {benchmark}")
    else:
        selected = list(task_ids)
        if task_list is not None:
            selected.extend(
                line.strip() for line in task_list.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select one or more unique tasks with --cohort, --task-id, or --task-list")
    missing = set(selected) - set(available_ids)
    if missing:
        label = f"The {cohort} cohort" if cohort else "The custom selection"
        raise ValueError(f"{label} contains IDs missing from --dataset: {', '.join(sorted(missing))}")
    return selected


def list_tasks(args):
    loader = _task_loader(args.benchmark, args.dataset, [])
    print("\n".join(_select_task_ids(args.benchmark, loader.task_ids, args.cohort)))


def _load_trust(path, records):
    supplied = json.loads(path.read_text(encoding="utf-8")) if path else {}
    allowed = {"non_mutating_schema_hashes", "cache_policy", "environment_epoch", "contract_id"}
    if not isinstance(supplied, dict) or set(supplied) - allowed:
        raise ValueError("Unknown execution-contract fields")
    hashes = supplied.get("non_mutating_schema_hashes", [])
    if not isinstance(hashes, list) or any((not isinstance(value, str) for value in hashes)):
        raise ValueError("non_mutating_schema_hashes must be an array of schema hashes")
    if hashes and supplied.get("cache_policy") != "disabled":
        raise ValueError("The MCP adapter requires an attested disabled cache contract")
    if hashes and any(
        (
            not isinstance(supplied.get(key), str) or not supplied[key].strip()
            for key in ("contract_id", "environment_epoch")
        )
    ):
        raise ValueError("Reviewed MCP tools require a contract ID and environment identity")
    known = {record.public_schema_hash for record in records if record.source_family == "mcp_atlas"}
    if set(hashes) - known:
        raise ValueError("Execution contract contains schemas absent from this corpus")
    trusted = set(hashes) | {
        record.public_schema_hash for record in records if record.source_family == "planbench_xl"
    }
    return ExecutionTrust(
        non_mutating_schema_hashes=frozenset(trusted),
        cache_policy="disabled",
        environment_epoch=str(supplied.get("environment_epoch", "external_environment")),
        contract_id=str(supplied.get("contract_id", "native_planbench_only")),
    )


def run_benchmark(args):
    from bria.benchmarks import UnionExecutor

    _fresh(args.output_dir)
    config = load_config(args.config)
    records, corpus = load_verified_corpus(args.corpus, args.corpus_manifest)
    native_digest = _planbench_digest(args.planbench_root)
    if corpus.get("provenance", {}).get("planbench_tree_sha256") != native_digest:
        raise ValueError("PlanBench environment differs from the corpus source")
    index, _ = ExactVectorIndex.load(
        args.index_dir,
        expected_corpus_sha256=corpus["corpus_sha256"],
        expected_model_id=config.embedding.model_id,
        expected_model_revision=config.embedding.model_revision,
        expected_query_instruction=config.embedding.query_instruction,
        expected_document_template_version=corpus["document_template_version"],
    )
    benchmark = "planbench_xl" if args.command == "run-planbench" else "mcp_atlas"
    loader = _task_loader(benchmark, args.dataset, records)
    task_ids = _select_task_ids(
        benchmark, loader.task_ids, args.cohort, args.task_id or (), args.task_list
    )
    trust = _load_trust(args.execution_contract, records)
    mcp_records = [record for record in records if record.source_family == "mcp_atlas"]
    admitted_mcp = [record for record in mcp_records if trust.permits(record)]
    if benchmark == "mcp_atlas" and (not admitted_mcp):
        raise ValueError("MCP tasks require a reviewed --execution-contract")
    if admitted_mcp and (not args.mcp_url):
        raise ValueError("Supply --mcp-url for reviewed MCP tools")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} for the solver provider")
    base_url = args.llm_base_url or os.environ.get("LLM_BASE_URL")
    if not base_url:
        raise ValueError("Set --llm-base-url or LLM_BASE_URL")
    completion = JSONCompletion(
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=args.max_output_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
    )
    retriever = UnifiedRetriever(
        records=records,
        index=index,
        embedder=Qwen3Embedder(config.embedding),
        allow_exact_duplicates=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _save(
        args.output_dir / "run_config.json",
        dict(
            method="bria",
            benchmark=benchmark,
            policy=asdict(config.policy),
            embedding=asdict(config.embedding),
            model=args.model,
            response_format={"type": "json_object"},
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            corpus_sha256=corpus["corpus_sha256"],
            planbench_tree_sha256=native_digest,
            dataset_sha256=loader.dataset_sha256,
            index_manifest_sha256=hashlib.sha256(
                (args.index_dir / "index_manifest.json").read_bytes()
            ).hexdigest(),
            cohort=args.cohort or "custom",
            task_ids=task_ids,
            execution_contract_id=trust.contract_id,
            reviewed_schema_hashes=sorted(trust.non_mutating_schema_hashes),
        ),
    )
    try:
        with (args.output_dir / "predictions.csv").open(
            "x", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["task_id", "response"])
            writer.writeheader()
            for sequence, task_id in enumerate(task_ids, 1):
                task = loader.public_task(task_id)
                native = (
                    loader.native_execution_state(task_id) if benchmark == "planbench_xl" else None
                )
                executor = UnionExecutor(
                    records,
                    planbench_root=args.planbench_root,
                    base_url=args.mcp_url or "http://localhost:1984",
                    planbench_native_state=native,
                    trust=trust,
                )
                task_dir = args.output_dir / f"task_{sequence:04d}"
                controller = Controller(
                    task=task,
                    config=config.policy,
                    completion=completion,
                    executor=executor,
                    retriever=retriever,
                    audit_dir=task_dir / "audit",
                    trust=trust,
                )
                result = controller.run()
                _save(task_dir / "result.json", result)
                _save(task_dir / "native_execution.json", executor.evaluation_log)
                if result["stop_reason"] != "completed":
                    raise RuntimeError(
                        f"Task {task_id} stopped: {result['stop_reason']}; saved output requires review"
                    )
                writer.writerow(dict(task_id=task_id, response=result["answer"]))
                handle.flush()
                print(f"{sequence}/{len(task_ids)}: {task_id} completed", flush=True)
    finally:
        completion.close()


def score_planbench(args):
    from bria.benchmarks import PlanBenchEvaluator, PlanBenchTaskLoader

    _fresh(args.output_dir)
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    if run_config.get("benchmark") != "planbench_xl":
        raise ValueError("Scoring requires a saved PlanBench run")
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != run_config.get("dataset_sha256"):
        raise ValueError("Scoring dataset differs from the solver dataset")
    if _planbench_digest(args.planbench_root) != run_config.get("planbench_tree_sha256"):
        raise ValueError("Scoring environment differs from the solver environment")
    evaluator = PlanBenchEvaluator(args.planbench_root)
    loader = PlanBenchTaskLoader(args.dataset, [])
    scores = []
    for task_dir in sorted(args.run_dir.glob("task_*")):
        result = json.loads((task_dir / "result.json").read_text())
        if result["stop_reason"] != "completed":
            raise ValueError("An incomplete task cannot be counted as a completed evaluation")
        selected = set(result["selected_tool_attempt_ids"])
        executions = json.loads((task_dir / "native_execution.json").read_text())
        executions = [event for event in executions if event["attempt_id"] in selected]
        score = evaluator.native_checks(
            loader.evaluation_record(result["task_id"]), result["answer"], executions
        )
        scores.append(dict(task_id=result["task_id"], **score))
    if not scores:
        raise ValueError("No completed task outputs found")
    scored_ids = [row["task_id"] for row in scores]
    if len(scored_ids) != len(set(scored_ids)) or set(scored_ids) != set(run_config["task_ids"]):
        raise ValueError("Saved task outputs do not exactly match the selected run tasks")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _save(args.output_dir / "scores.json", scores)
    print(
        json.dumps(
            dict(
                tasks=len(scores),
                native_success_rate=sum((row["native_criteria_success"] for row in scores))
                / len(scores),
            ),
            indent=2,
        )
    )


def parser():
    result = argparse.ArgumentParser(description="BRIA: BRanch–Inspect–Adopt")
    commands = result.add_subparsers(dest="command", required=True)
    corpus = commands.add_parser(
        "build-corpus", help="Read external public schemas into a local shared corpus"
    )
    corpus.add_argument("--planbench-root", type=Path, required=True)
    schema = corpus.add_mutually_exclusive_group(required=True)
    schema.add_argument("--mcp-url")
    schema.add_argument("--mcp-schema-snapshot", type=Path)
    corpus.add_argument(
        "--mcp-servers", help="Comma-separated server names for an offline schema snapshot"
    )
    corpus.add_argument("--output-dir", type=Path, required=True)
    corpus.set_defaults(function=build_corpus)
    index = commands.add_parser("build-index", help="Encode public schemas with Qwen3-Embedding-8B")
    index.add_argument("--config", type=Path, required=True)
    index.add_argument("--corpus", type=Path, required=True)
    index.add_argument("--corpus-manifest", type=Path, required=True)
    index.add_argument("--output-dir", type=Path, required=True)
    index.add_argument("--device")
    index.add_argument("--batch-size", type=_positive)
    index.set_defaults(function=build_index)
    listing = commands.add_parser("list-tasks", help="List IDs from an external dataset")
    listing.add_argument("--benchmark", choices=["planbench_xl", "mcp_atlas"], required=True)
    listing.add_argument("--dataset", type=Path, required=True)
    listing.add_argument("--cohort", choices=["full", "75tasks"], default="full")
    listing.set_defaults(function=list_tasks)
    for name in ("run-planbench", "run-mcp-atlas"):
        run = commands.add_parser(name, help="Run BRIA on explicitly selected external tasks")
        for flag in (
            "config",
            "corpus",
            "corpus-manifest",
            "index-dir",
            "dataset",
            "planbench-root",
            "output-dir",
        ):
            run.add_argument("--" + flag, type=Path, required=True)
        run.add_argument("--mcp-url")
        run.add_argument("--execution-contract", type=Path)
        run.add_argument(
            "--cohort",
            choices=["full", "75tasks"],
            help="Run all tasks in the supplied dataset or the fixed 75-task subset",
        )
        run.add_argument("--task-id", action="append")
        run.add_argument("--task-list", type=Path)
        run.add_argument("--model", required=True)
        run.add_argument("--llm-base-url")
        run.add_argument("--api-key-env", default="LLM_API_KEY")
        run.add_argument(
            "--response-format", choices=["json_object"], action="append", required=True
        )
        run.add_argument("--max-output-tokens", type=_positive, default=8192)
        run.add_argument("--temperature", type=float)
        run.add_argument("--reasoning-effort")
        run.add_argument("--timeout", type=float, default=1800.0)
        run.set_defaults(function=run_benchmark)
    score = commands.add_parser(
        "score-planbench", help="Evaluate saved answers and selected native execution lineages"
    )
    for flag in ("planbench-root", "dataset", "run-dir", "output-dir"):
        score.add_argument("--" + flag, type=Path, required=True)
    score.set_defaults(function=score_planbench)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if hasattr(args, "response_format") and args.response_format != ["json_object"]:
        cli.error("Pass exactly one --response-format json_object")
    try:
        args.function(args)
    except (ValueError, RuntimeError, OSError) as exc:
        cli.exit(1, f"bria: {exc}\n")
