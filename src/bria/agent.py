"""BRIA control flow: branch, inspect, adopt, and continue."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass

from bria.prompts import PROMPTS, ContextProjector
from bria.state import (
    METHOD_ID,
    PROJECTOR_ID,
    AuditStore,
    Budget,
    BudgetExhausted,
    CallKey,
    EvidenceStore,
    Interrupted,
    RetrievalRequest,
    ValidationError,
    canonical,
    clone,
    digest,
    schema_errors,
)


class Controller:
    """Follow run() for the task loop, then the candidate and adoption methods.

    Retrieval and tool execution follow those methods; evidence and event
    bookkeeping are at the end of the class.
    """

    def __init__(
        self,
        *,
        task,
        config,
        completion,
        executor,
        retriever,
        audit_dir,
        trust=None,
        manifest=None,
        audit_sink=None,
        http_504_retries=0,
        http_504_retry_delays=(10, 20),
    ):
        self.task, self.config, self.completion = (task, config, completion)
        self.executor, self.retriever = (executor, retriever)
        self.trust = trust or ExecutionTrust()
        if type(http_504_retries) is not int or http_504_retries < 0:
            raise ValueError("http_504_retries must be a non-negative integer")
        if http_504_retries > len(http_504_retry_delays):
            raise ValueError("Each HTTP 504 retry requires a recorded delay")
        self.http_504_retries = http_504_retries
        self.http_504_retry_delays = tuple(http_504_retry_delays)
        if config.max_total_tokens is not None and (
            not callable(getattr(completion, "request_token_upper_bound", None))
        ):
            raise ValueError(
                "A hard token budget requires a public request_token_upper_bound adapter; estimates are not accepted"
            )
        if any(
            (
                getattr(completion, name, 0)
                for name in ("max_http_429_retries", "max_http_504_retries")
            )
        ):
            raise ValueError("BRIA requires exposed single-attempt completion transport")
        if self.trust.cache_policy not in {"disabled", "immutable_public_snapshot", "unknown"}:
            raise ValueError(
                "Candidate adoption requires disabled cache or an immutable public snapshot"
            )
        if task.benchmark == "planbench_xl" and (
            not callable(getattr(executor, "execute_at_node", None))
        ):
            raise ValueError("PlanBench multi-adoption requires lineage-scoped execution")
        self.audit = AuditStore(audit_dir, sink=audit_sink)
        self.store = EvidenceStore(self.audit)
        self.budget = Budget(config, self.audit)
        self.projector = ContextProjector(self.audit)
        self.visible = {"parent": {}}
        self.retrieval_queries = {"parent": set()}
        self.tried = {"parent": set()}
        self.success_observations = {"parent": 0}
        self.gate_retry_failures = {"parent": {}}
        self.schema_views = {"parent": dict(latest=[], used=[], relevant=[], query=None)}
        self.active = None
        self.state = dict(parent_history=[], checkpoints=[], comparisons=[], status="running")
        self.state.update(continuation_history=[], selection_pending=None, adoptions=[])
        self.lineage_attempt_ids = {"parent": []}
        self.next_attempt_id = 0
        self.audit.append_event(
            "run_started",
            dict(
                policy_id=METHOD_ID,
                config=asdict(config),
                task=dict(
                    task_id=task.task_id, benchmark=task.benchmark, user_query=task.user_query
                ),
                model=getattr(completion, "model", None),
                reasoning_effort=getattr(completion, "reasoning_effort", None),
                provider=getattr(completion, "base_url", None),
                prompt_hashes={r: digest(p) for r, p in PROMPTS.items()},
                projector_id=PROJECTOR_ID,
                trust={
                    **asdict(self.trust),
                    "non_mutating_schema_hashes": sorted(self.trust.non_mutating_schema_hashes),
                },
                infrastructure_manifest=manifest or {},
                transport_retries=http_504_retries,
                transport_retry_delays_seconds=list(self.http_504_retry_delays[:http_504_retries]),
                external_judge="not_run_by_policy",
            ),
        )
        self.save_state()

    # Main task loop.
    def run(self):
        answer, error = (None, None)
        started = time.monotonic()
        try:
            self.search(self.task.user_query)
            while self.state["status"] == "running":
                action = self.call_model()
                name = action.get("action")
                try:
                    if self.state["selection_pending"]:
                        if name != "adopt_candidates":
                            raise ValidationError("Adopt candidates before parent continuation")
                        self.adopt_candidates(action)
                    elif name == "adopt_candidates":
                        raise ValidationError("No candidate checkpoint awaits adoption")
                    elif name == "final_answer":
                        if not isinstance(action.get("answer"), str):
                            raise ValidationError("Answer must be text")
                        answer = action["answer"]
                        self.state["status"] = "completed"
                    elif name == "open_candidates":
                        result = self.open_checkpoint(action)
                        if result["status"] == "opened":
                            self.process_checkpoint(result["checkpoint_id"])
                    else:
                        self.ordinary_action(action)
                except (ValidationError, KeyError, TypeError) as exc:
                    self.budget.reserve("internal")
                    self.note("action_rejected", dict(reason=str(exc)))
        except BudgetExhausted as exc:
            self.state["status"], error = ("budget_exhausted", str(exc))
        except (Interrupted, KeyboardInterrupt) as exc:
            self.state["status"], error = ("interrupted", str(exc))
            self.audit.append_event(
                "interruption", dict(reason=error, scoring_status="pending_unscored")
            )
        except Exception as exc:
            self.state["status"], error = ("execution_error", type(exc).__name__ + ": " + str(exc))
        if self.active and self.state["status"] in {
            "budget_exhausted",
            "interrupted",
            "execution_error",
        }:
            checkpoint = next(
                (item for item in self.state["checkpoints"] if item["checkpoint_id"] == self.active)
            )
            checkpoint["status"] = self.state["status"]
            self.audit.append_event(
                "checkpoint_outcome",
                dict(
                    checkpoint_id=self.active,
                    status=checkpoint["status"],
                    reason=error,
                    counts=dict(
                        total=len(checkpoint["candidates"]),
                        resolved=sum(
                            (
                                (candidate.get("result") or {}).get("status") == "resolved"
                                for candidate in checkpoint["candidates"]
                            )
                        ),
                        unresolved=sum(
                            (
                                (candidate.get("result") or {}).get("status") == "unresolved"
                                for candidate in checkpoint["candidates"]
                            )
                        ),
                        pending=sum(
                            (
                                candidate.get("result") is None
                                for candidate in checkpoint["candidates"]
                            )
                        ),
                    ),
                ),
            )
        result = dict(
            policy_id=METHOD_ID,
            task_id=self.task.task_id,
            answer=answer,
            stop_reason=self.state["status"],
            execution_status="interrupted"
            if self.state["status"] == "interrupted"
            else "error"
            if self.state["status"] == "execution_error"
            else "completed",
            scoring_status="pending_unscored"
            if self.state["status"] in {"interrupted", "execution_error"}
            else "awaiting_external_evaluation",
            task_success=None,
            error=error,
            budget=self.budget.view(),
            wall_seconds=time.monotonic() - started,
        )
        result["selected_tool_attempt_ids"] = list(self.lineage_attempt_ids["parent"])
        result["adoptions"] = clone(self.state["adoptions"])
        self.save_state()
        self.audit.append_event("run_finished", result)
        return result

    # Independent candidate investigations, followed by inspection and adoption.
    def open_checkpoint(self, proposal):
        self.budget.reserve("internal")
        if self.active:
            raise ValidationError("Nested checkpoints are unsupported")
        try:
            if self.trust.cache_policy not in {"disabled", "immutable_public_snapshot"}:
                raise ValidationError("Executor cache isolation is unknown")
            goal, candidates = open_source(proposal, self.config.max_candidates_per_checkpoint)
        except (ValidationError, KeyError, TypeError) as exc:
            self.note("open_candidates_failed", dict(reason=str(exc)))
            self.save_state()
            return dict(status="rejected", reason=str(exc))
        cp_id = f"cp_{len(self.state['checkpoints']) + 1}"
        common_parent_observations = self.common_parent_observations()
        prefix = None
        prefix = clone(self.state["continuation_history"])
        if (
            prefix
            and prefix[-1]["kind"] == "policy_action"
            and ((prefix[-1]["data"] or {}).get("action") == "open_candidates")
        ):
            prefix.pop()
        cp = dict(
            checkpoint_id=cp_id,
            checkpoint_contract="candidate_investigation",
            goal=goal,
            continuation_prefix=prefix,
            candidate_origin="model_supplied_from_visible_context",
            candidate_set_digest=digest(candidates),
            visible_tool_ids_at_open=list(self.visible["parent"]),
            common_parent_observations=common_parent_observations,
            candidates=candidates,
            status="active",
        )
        for index, candidate in enumerate(candidates):
            cid = f"{cp_id}:c{index + 1}"
            actor = "branch:" + cid
            candidate.update(
                candidate_id=cid,
                checkpoint_id=cp_id,
                branch_id=actor,
                execution_status="pending",
                history=[],
                observation_views=[],
                branch_observation_ids=[],
                tool_calls=0,
                policy_calls=0,
                result=None,
            )
            candidate["candidate_view"] = dict(
                candidate_id=cid, value=clone(candidate["candidate_value"])
            )
            self.visible[actor] = dict(self.visible["parent"])
            self.retrieval_queries[actor] = set(self.retrieval_queries["parent"])
            self.lineage_attempt_ids[actor] = list(self.lineage_attempt_ids["parent"])
            self.tried[actor] = set()
            self.success_observations[actor] = 0
            self.gate_retry_failures[actor] = {}
            self.schema_views[actor] = clone(self.schema_views["parent"])
            for observation in common_parent_observations:
                if observation["success"]:
                    ref = observation["evidence_ref"]
                    self.store.grant(actor, ref["observation_id"], ref["pointer"])
        self.state["checkpoints"].append(cp)
        self.active = cp_id
        cp["created_event_id"] = self.audit.append_event("checkpoint_opened", clone(cp))
        self.save_state()
        return dict(status="opened", checkpoint_id=cp_id)

    def process_checkpoint(self, checkpoint_id):
        checkpoint = next(
            (item for item in self.state["checkpoints"] if item["checkpoint_id"] == checkpoint_id)
        )
        for candidate in checkpoint["candidates"]:
            self.run_candidate(checkpoint_id, candidate["candidate_id"])
        return self.compare_candidates(checkpoint)

    def run_candidate(self, checkpoint_id, candidate_id):
        checkpoint = next(
            (item for item in self.state["checkpoints"] if item["checkpoint_id"] == checkpoint_id)
        )
        candidate = next(
            (item for item in checkpoint["candidates"] if item["candidate_id"] == candidate_id)
        )
        if candidate["execution_status"] != "pending":
            raise ValidationError("Candidate has already been processed")
        candidate["execution_status"] = "running"
        self.audit.append_event("candidate_started", dict(candidate_id=candidate_id))
        try:
            while True:
                action = self.call_model(candidate["branch_id"], checkpoint, candidate)
                if action.get("action") == "candidate_done":
                    try:
                        return self.finish_candidate(checkpoint, candidate, action)
                    except (ValidationError, KeyError, TypeError) as exc:
                        self.note("action_rejected", dict(reason=str(exc)), candidate)
                        continue
                try:
                    self.ordinary_action(action, candidate["branch_id"], candidate)
                except (ValidationError, KeyError, TypeError) as exc:
                    self.note("action_rejected", dict(reason=str(exc)), candidate)
        except Interrupted:
            candidate["execution_status"] = "interrupted"
            raise
        except (ValueError, TypeError) as exc:
            return self.unresolved(checkpoint, candidate, "model_error:" + type(exc).__name__)
        finally:
            self.audit.append_event(
                "candidate_finished",
                dict(candidate_id=candidate_id, execution_status=candidate["execution_status"]),
            )
            self.save_state()

    def finish_candidate(self, checkpoint, candidate, action):
        status = action.get("status")
        refs = action.get("evidence_refs", [])
        if status not in {"resolved", "unresolved"} or not isinstance(refs, list):
            raise ValidationError(
                "candidate_done requires resolved/unresolved status and evidence_refs"
            )
        summary = action.get("summary")
        reason = action.get("reason")
        if status == "resolved" and (
            not isinstance(summary, str) or not summary.strip() or (not refs)
        ):
            raise ValidationError("Resolved candidate requires a summary and evidence")
        if status == "unresolved" and (not isinstance(reason, str) or not reason.strip()):
            raise ValidationError("Unresolved candidate requires a reason")
        evidence = []
        for ref in refs:
            if not isinstance(ref, dict):
                raise ValidationError("Candidate evidence references must be objects")
            value = branch_evidence(ref, candidate, self.store, candidate["branch_id"])
            evidence.append(dict(ref=clone(ref), value=value))
        result = dict(
            candidate_id=candidate["candidate_id"],
            status=status,
            model_summary=summary.strip() if status == "resolved" else None,
            reason=reason.strip() if status == "unresolved" else None,
            tool_executions=self.branch_tool_executions(candidate),
            evidence=evidence,
        )
        candidate["result"] = result
        candidate["execution_status"] = "finished"
        self.audit.append_event("candidate_result", clone(result))
        self.save_state()
        return result

    def unresolved(self, checkpoint, candidate, reason):
        result = dict(
            candidate_id=candidate["candidate_id"],
            status="unresolved",
            model_summary=None,
            reason=str(reason),
            tool_executions=self.branch_tool_executions(candidate),
            evidence=[],
        )
        candidate["result"] = result
        candidate["execution_status"] = "finished"
        self.audit.append_event("candidate_result", clone(result))
        self.save_state()
        return result

    def compare_candidates(self, checkpoint):
        if any((candidate.get("result") is None for candidate in checkpoint["candidates"])):
            raise ValidationError("Every candidate must have a result before comparison")
        results = []
        for candidate in checkpoint["candidates"]:
            row = clone(candidate["result"])
            row["candidate"] = clone(candidate["candidate_value"])
            results.append(row)
        capsule = dict(
            checkpoint_id=checkpoint["checkpoint_id"],
            goal=checkpoint["goal"],
            candidate_set_digest=checkpoint["candidate_set_digest"],
            results=results,
            counts=dict(
                total=len(results),
                resolved=sum((row["status"] == "resolved" for row in results)),
                unresolved=sum((row["status"] == "unresolved" for row in results)),
            ),
        )
        event_id = self.note("candidate_results", capsule)
        self.state["comparisons"].append(dict(event_id=event_id, **clone(capsule)))
        checkpoint["status"] = "awaiting_adoption"
        self.state["selection_pending"] = checkpoint["checkpoint_id"]
        self.save_state()
        return dict(
            status="awaiting_adoption",
            checkpoint_id=checkpoint["checkpoint_id"],
            counts=clone(capsule["counts"]),
        )

    def adopt_candidates(self, action):
        checkpoint_id = self.state["selection_pending"]
        if checkpoint_id is None or action.get("checkpoint_id") != checkpoint_id:
            raise ValidationError("No matching checkpoint awaits adoption")
        candidate_ids = action.get("candidate_ids")
        if (
            not isinstance(candidate_ids, list)
            or any((not isinstance(cid, str) for cid in candidate_ids))
            or len(candidate_ids) != len(set(candidate_ids))
        ):
            raise ValidationError("Adoption requires distinct candidate IDs")
        checkpoint = next(
            (cp for cp in self.state["checkpoints"] if cp["checkpoint_id"] == checkpoint_id)
        )
        by_id = {candidate["candidate_id"]: candidate for candidate in checkpoint["candidates"]}
        if any((cid not in by_id for cid in candidate_ids)):
            raise ValidationError("Adoption references an unknown candidate")
        self.budget.reserve("internal")
        selected = [by_id[cid] for cid in candidate_ids]
        rows = []
        for candidate in selected:
            row = clone(candidate["result"])
            row["candidate"] = clone(candidate["candidate_value"])
            rows.append(row)
            actor = candidate["branch_id"]
            for execution in row["tool_executions"]:
                if execution["success"]:
                    ref = execution["evidence_ref"]
                    self.store.grant("parent", ref["observation_id"], ref["pointer"])
            self.visible["parent"].update(self.visible[actor])
            self.retrieval_queries["parent"].update(self.retrieval_queries[actor])
            self.tried["parent"].update(self.tried[actor])
            for name in ("latest", "used", "relevant"):
                for tool_id in self.schema_views[actor][name]:
                    if tool_id not in self.schema_views["parent"][name]:
                        self.schema_views["parent"][name].append(tool_id)
        adopted = dict(
            checkpoint_id=checkpoint_id,
            goal=checkpoint["goal"],
            selected_candidate_ids=clone(candidate_ids),
            selected_branches=rows,
        )
        event_id = self.note("adopted_candidates", adopted)
        item = clone(self.state["parent_history"][-1])
        self.state["continuation_history"] = clone(checkpoint["continuation_prefix"]) + [item]
        self.state["adoptions"].append(
            dict(
                event_id=event_id,
                checkpoint_id=checkpoint_id,
                selected_candidate_ids=clone(candidate_ids),
            )
        )
        self.lineage_attempt_ids["parent"] = sorted(
            set(self.lineage_attempt_ids["parent"]).union(
                *(self.lineage_attempt_ids[candidate["branch_id"]] for candidate in selected)
            )
        )
        self.success_observations["parent"] += sum(
            (self.success_observations[candidate["branch_id"]] for candidate in selected)
        )
        for candidate in selected:
            for key, failure in self.gate_retry_failures[candidate["branch_id"]].items():
                self.gate_retry_failures["parent"][key] = dict(
                    failure, success_observations=self.success_observations["parent"]
                )
        checkpoint["status"] = "adopted"
        self.state["selection_pending"] = None
        self.active = None
        self.save_state()
        return adopted

    # Retrieval and execution available to both the parent and each branch.
    def ordinary_action(self, action, actor="parent", candidate=None):
        name = action.get("action")
        if name == "call_tool":
            return self.execute(action, actor, candidate)
        if name == "read_evidence":
            return self.read_evidence(action, actor, candidate)
        if name == "retrieve":
            thought = action.get("thought")
            if not isinstance(thought, str) or not thought.strip():
                raise ValidationError(
                    "Retrieve requires a brief reason why displayed tools cannot advance the next step"
                )

            def leaves(value):
                if isinstance(value, dict):
                    return [item for nested in value.values() for item in leaves(nested)]
                if isinstance(value, list):
                    return [item for nested in value for item in leaves(nested)]
                return [value]

            forbidden = leaves(candidate["candidate_value"]) if candidate else []
            query = retrieval_query(action["need"], forbidden)
            if any((tool_id.casefold() in query.casefold() for tool_id in self.visible[actor])):
                raise ValidationError("Tool identifiers are forbidden in retrieval need")
            return self.search(query, actor, candidate, action["need"])
        raise ValidationError("Unsupported action in this scope")

    def search(self, query, actor="parent", candidate=None, need=None):
        self.budget.reserve("retrieval")
        repeat_query = query in self.retrieval_queries[actor]
        excluded_ids = tuple(self.visible[actor]) if repeat_query else ()
        event = self.audit.append_event(
            "retrieval_request",
            dict(
                actor_scope=actor,
                query=query,
                need=need,
                top_k=self.config.retrieval_top_k,
                tried_tool_ids=sorted(self.tried[actor]),
                repeated_query=repeat_query,
                excluded_tool_ids=list(excluded_ids),
            ),
        )
        try:
            result = retrieve(
                self.retriever,
                self.task,
                self.config,
                query,
                self.visible[actor],
                self.tried[actor],
                exclude_tool_ids=excluded_ids,
            )
        except Exception as exc:
            self.audit.append_event(
                "retrieval_result",
                dict(request_event_id=event, error=type(exc).__name__, usage=None),
            )
            raise
        prior_ids = set(self.visible[actor])
        self.retrieval_queries[actor].add(query)
        relevant = [
            x.record.global_tool_id
            for x in result.repeated_context_top_k
            if x.record.global_tool_id in prior_ids
        ][:0]
        for item in result.visible_top_k:
            self.visible[actor][item.record.global_tool_id] = item.record
        self.schema_views[actor].update(
            latest=[x.record.global_tool_id for x in result.visible_top_k],
            relevant=relevant,
            query=query,
        )
        self.note(
            "retrieval_result",
            dict(
                request_event_id=event,
                tools=[public_schema(x.record) for x in result.visible_top_k],
                repeated_tool_ids=relevant,
                query_encoding_ms=result.query_encoding_ms,
                index_search_ms=result.index_search_ms,
                embedding_usage=None,
            ),
            candidate,
        )
        self.save_state()

    def execute(self, action, actor="parent", candidate=None):
        call = resolve_action(action, actor, self.visible[actor], self.trust)
        refs = []
        key = digest(dict(tool_id=call.tool_id, arguments=call.arguments))
        failed = self.gate_retry_failures[actor].get(key)
        if failed and failed["success_observations"] == self.success_observations[actor]:
            raise ValidationError("Identical gate-rejected call has no new successful observation")
        if getattr(self.executor, "interruption", None):
            raise Interrupted("executor_interrupted")
        self.budget.reserve("tool")
        if candidate:
            candidate["tool_calls"] += 1
        self.tried[actor].add(call.tool_id)
        used = self.schema_views[actor]["used"]
        if call.tool_id not in used:
            used.append(call.tool_id)
        attempt_id = None
        ancestors = []
        self.next_attempt_id += 1
        attempt_id = self.next_attempt_id
        ancestors = list(self.lineage_attempt_ids[actor])
        self.lineage_attempt_ids[actor].append(attempt_id)
        event = self.audit.append_event(
            "tool_dispatch",
            dict(
                actor_scope=actor,
                proposal=action,
                tool_id=call.tool_id,
                resolved_arguments=call.arguments,
                source_refs=refs,
                cache_policy=self.trust.cache_policy,
                lineage_attempt_id=attempt_id,
            ),
        )
        try:
            if callable(getattr(self.executor, "execute_at_node", None)):
                observation = self.executor.execute_at_node(
                    self.task,
                    call,
                    step_id=attempt_id,
                    node_id=actor,
                    ancestor_attempt_ids=ancestors,
                )
            else:
                observation = self.executor.execute(self.task, call)
        except BaseException as exc:
            self.budget.settle_tool(None)
            self.audit.append_event(
                "dispatched_unknown_outcome",
                dict(dispatch_event_id=event, error=type(exc).__name__, actor_scope=actor),
            )
            raise Interrupted("dispatched_unknown_outcome") from exc
        result_event = self.audit.append_event(
            "tool_result" if observation.success else "tool_error",
            dict(dispatch_event_id=event, actor_scope=actor, observation=asdict(observation)),
        )
        self.budget.settle_tool(observation.dispatched)
        self.audit.append_event(
            "budget_event",
            dict(
                stage="settled",
                kind="tool",
                dispatch_event_id=event,
                backend_dispatched=observation.dispatched,
                usage=None,
            ),
        )
        if observation.interrupts_task:
            raise Interrupted(True)
        record = self.visible[actor][call.tool_id]
        errors = (
            schema_errors(observation.result, record.output_schema) if observation.success else ()
        )
        dependencies = []
        ref = self.store.save(
            tool_id=call.tool_id,
            schema_hash=record.public_schema_hash,
            arguments=call.arguments,
            raw=observation.raw_response
            if observation.raw_response is not None
            else observation.result,
            parsed=observation.result,
            owner=actor,
            event_id=result_event,
            epoch=self.trust.environment_epoch,
            success=observation.success and (not errors),
            source_refs=refs,
            dependencies=dependencies,
        )
        view = dict(
            ref=ref,
            result=clone(observation.result),
            success=observation.success and (not errors),
            error_code=observation.error_code,
            message=observation.message,
            schema_errors=list(errors),
        )
        self.note("observation", view, candidate)
        if observation.success and (not errors):
            self.success_observations[actor] += 1
        elif (
            observation.error_code == "required_input_unavailable"
            and observation.dispatched is False
        ):
            self.gate_retry_failures[actor][key] = dict(
                tool_id=call.tool_id,
                arguments=clone(call.arguments),
                observation_id=ref["observation_id"],
                success_observations=self.success_observations[actor],
            )
        if candidate:
            candidate["branch_observation_ids"].append(ref["observation_id"])
            candidate["observation_views"].append(view)
        self.save_state()
        return ref

    def read_evidence(self, action, actor="parent", candidate=None):
        self.budget.reserve("internal")
        ref = self.store.ref(action["observation_id"], action.get("pointer", ""))
        value = self.store.read_evidence(ref, actor)
        self.note("evidence_read", dict(actor_scope=actor, ref=ref, value=value), candidate)
        self.save_state()
        return value

    def call_model(self, actor="parent", checkpoint=None, candidate=None):
        if getattr(self.executor, "interruption", None):
            raise Interrupted("executor_interrupted")
        self.budget.check()
        if candidate:
            candidate["policy_calls"] += 1
        role = "parent" if actor == "parent" else "branch"
        messages, projection = self.project(actor, checkpoint, candidate)
        request = dict(messages=messages, response_format={"type": "json_object"})
        if self.config.max_total_tokens is not None:
            bound = self.completion.request_token_upper_bound(**clone(request))
            if (
                type(bound) is not int
                or bound < 1
                or self.budget.known_tokens + bound > self.config.max_total_tokens
            ):
                raise BudgetExhausted("token_reservation")
            self.audit.append_event(
                "budget_event", dict(stage="reserved", kind="tokens", upper_bound=bound, role=role)
            )
        self.budget.reserve("llm", role)
        for transport_attempt in range(1, self.http_504_retries + 2):
            event = self.audit.append_event(
                "llm_request",
                dict(
                    role=role,
                    actor_scope=actor,
                    projection_id=projection["projection_id"],
                    request_blob_ref=self.audit.blob(request),
                    model=getattr(self.completion, "model", None),
                    transport_attempt=transport_attempt,
                    generation_settings={
                        key: getattr(self.completion, key, None)
                        for key in ("temperature", "max_tokens", "reasoning_effort", "timeout")
                    },
                ),
            )
            result = None
            error = None
            raised = None
            started = time.monotonic()
            try:
                result = self.completion.complete(**clone(request))
                if not isinstance(result, dict):
                    raise ValidationError("Model response must be an object")
                result = clone(result)
            except Exception as exc:
                error = type(exc).__name__
                raised = exc
            usage = getattr(self.completion, "last_usage", None)
            response_event = self.audit.append_event(
                "llm_response",
                dict(
                    request_event_id=event,
                    actor_scope=actor,
                    role=role,
                    transport_attempt=transport_attempt,
                    output=result,
                    error=error,
                    provider_transport_error=clone(
                        getattr(self.completion, "last_http_error", None)
                    ),
                    usage=usage,
                    latency_ms=(time.monotonic() - started) * 1000,
                    content=getattr(self.completion, "last_content", None),
                    reasoning=getattr(self.completion, "last_reasoning", None),
                    finish_reason=getattr(self.completion, "last_finish_reason", None),
                ),
            )
            diagnostic = getattr(self.completion, "last_http_error", None)
            retryable = (
                raised is not None
                and isinstance(diagnostic, dict)
                and (diagnostic.get("status") == 504)
                and (transport_attempt <= self.http_504_retries)
            )
            if retryable:
                delay = self.http_504_retry_delays[transport_attempt - 1]
                self.audit.append_event(
                    "llm_transport_retry",
                    dict(
                        request_event_id=event,
                        response_event_id=response_event,
                        transport_attempt=transport_attempt,
                        status=504,
                        retry_delay_seconds=delay,
                    ),
                )
                self.save_state()
                time.sleep(delay)
                continue
            self.budget.settle_llm(role, usage)
            item = dict(
                event_id=response_event,
                kind="policy_action",
                data=result if result is not None else {"error": error},
            )
            self.history(candidate).append(item)
            if candidate is None:
                self.state["continuation_history"].append(clone(item))
            self.save_state()
            if raised is not None:
                raise raised
            return result

    # Context and evidence bookkeeping.
    def project(self, actor="parent", checkpoint=None, candidate=None):
        records = self.visible[actor]
        view = self.schema_views[actor]
        displayed = list(dict.fromkeys(view["latest"] + view["used"] + view["relevant"]))
        tool_groups = None
        tool_groups = dict(
            latest_retrieval_ids=list(view["latest"]),
            previously_used_ids=list(view["used"]),
            reexposed_prior_ids=list(view["relevant"]),
        )
        return self.projector.project(
            task=self.task.user_query,
            actor_scope=actor,
            state=self.state,
            schemas=[public_schema(records[tid]) for tid in displayed],
            budget=self.budget.view(),
            checkpoint=checkpoint,
            candidate=candidate,
            tool_groups=tool_groups,
            recovery_context=self.recovery_context(actor),
        )

    def recovery_context(self, actor):
        current = self.success_observations[actor]
        return [
            dict(
                tool_id=item["tool_id"],
                arguments=clone(item["arguments"]),
                error_code="required_input_unavailable",
                observation_id=item["observation_id"],
                new_successful_observation_since_rejection=False,
            )
            for item in self.gate_retry_failures[actor].values()
            if item["success_observations"] == current
        ][-3:]

    def common_parent_observations(self):
        """Freeze visible pre-fork executions, including previously adopted branches."""
        observations = []
        events = self.state["continuation_history"]
        for event in events:
            if event.get("kind") == "adopted_candidates":
                for branch in event["data"]["selected_branches"]:
                    for execution in branch["tool_executions"]:
                        observations.append(
                            clone(
                                {
                                    key: execution[key]
                                    for key in (
                                        "observation_id",
                                        "tool_id",
                                        "arguments",
                                        "success",
                                        "result",
                                        "error_code",
                                        "message",
                                        "schema_errors",
                                        "evidence_ref",
                                    )
                                }
                            )
                        )
                continue
            if event.get("kind") != "observation":
                continue
            view = event["data"]
            evidence_ref = clone(view["ref"])
            snapshot = self.store.snapshot(evidence_ref["observation_id"])
            if snapshot["owner_scope"] != "parent":
                continue
            observations.append(
                dict(
                    observation_id=snapshot["observation_id"],
                    tool_id=snapshot["tool_id"],
                    arguments=clone(snapshot["resolved_arguments"]),
                    success=view["success"],
                    result=clone(view["result"]),
                    error_code=view.get("error_code"),
                    message=view.get("message"),
                    schema_errors=clone(view.get("schema_errors", [])),
                    evidence_ref=evidence_ref,
                )
            )
        return observations

    def branch_tool_executions(self, candidate):
        """Project the ordered executed-tool transcript, without model reasoning."""
        executions = []
        if len(candidate["observation_views"]) != len(candidate["branch_observation_ids"]):
            raise ValidationError("Branch observation ledger is inconsistent")
        for step, view in enumerate(candidate["observation_views"], 1):
            evidence_ref = clone(view["ref"])
            observation_id = evidence_ref["observation_id"]
            snapshot = self.store.snapshot(observation_id)
            if (
                observation_id != candidate["branch_observation_ids"][step - 1]
                or snapshot["owner_scope"] != candidate["branch_id"]
            ):
                raise ValidationError("Branch observation ownership is inconsistent")
            executions.append(
                dict(
                    step=step,
                    observation_id=observation_id,
                    tool_id=snapshot["tool_id"],
                    arguments=clone(snapshot["resolved_arguments"]),
                    success=view["success"],
                    result=clone(view["result"]),
                    error_code=view.get("error_code"),
                    message=view.get("message"),
                    schema_errors=clone(view.get("schema_errors", [])),
                    evidence_ref=evidence_ref,
                )
            )
        return executions

    def note(self, kind, data, candidate=None):
        event = self.audit.append_event(kind, data)
        item = dict(event_id=event, kind=kind, data=clone(data))
        self.history(candidate).append(item)
        if candidate is None:
            self.state["continuation_history"].append(clone(item))
        return event

    def history(self, candidate=None):
        return candidate["history"] if candidate else self.state["parent_history"]

    def save_state(self):
        self.audit.append_event(
            "state_saved",
            dict(
                ledger=self.state,
                budget=self.budget.view(),
                active_checkpoint=self.active,
                visible_tool_ids={actor: list(records) for actor, records in self.visible.items()},
                retrieval_queries={
                    actor: sorted(queries) for actor, queries in self.retrieval_queries.items()
                },
                schema_views=clone(self.schema_views),
                tried_tool_ids={actor: sorted(ids) for actor, ids in self.tried.items()},
                success_observations=clone(self.success_observations),
                gate_retry_failures=clone(self.gate_retry_failures),
                grants={
                    a: {obs: sorted(pointers) for obs, pointers in grants.items()}
                    for a, grants in self.store.grants.items()
                },
                invalidated_observations=sorted(self.store.invalidated),
                dependencies={k: sorted(v) for k, v in self.store.dependencies.items()},
                lineage_attempt_ids=clone(self.lineage_attempt_ids),
                next_attempt_id=self.next_attempt_id,
            ),
        )


@dataclass(frozen=True)
class ExecutionTrust:
    """Host-supplied attestations, never model output or readOnlyHint inference.

    Unknown effects/cache behavior fail closed. Schema hashes bind reviewed contracts.
    Contracts are supplied by the host and are bound to public schema hashes.
    """

    non_mutating_schema_hashes: frozenset[str] = frozenset()
    cache_policy: str = "unknown"
    environment_epoch: str = "live"
    contract_id: str = "unattested"

    def permits(self, record):
        return (
            record.public_schema_hash in self.non_mutating_schema_hashes
            and self.cache_policy in {"disabled", "immutable_public_snapshot"}
        )


def public_schema(record):
    return {
        name: clone(getattr(record, name))
        for name in (
            "global_tool_id",
            "tool_name",
            "description",
            "input_schema",
            "output_schema",
            "output_description",
        )
    }


def resolve_action(action, actor, visible, trust):
    tool_id = action.get("tool_id")
    if tool_id not in visible:
        raise ValidationError("Tool is not visible to this actor")
    record = visible[tool_id]
    if actor != "parent" and (not trust.permits(record)):
        raise ValidationError("Branch tool effects/cache isolation are not attested")
    if "literal_arguments" in action or "argument_bindings" in action:
        raise ValidationError("Use one direct arguments object")
    arguments = clone(action.get("arguments"))
    if not isinstance(arguments, dict):
        raise ValidationError("Arguments must be an object")
    errors = schema_errors(arguments, record.input_schema)
    if errors:
        raise ValidationError("Public input schema: " + "; ".join(errors))
    return CallKey.from_arguments(tool_id, arguments)


def retrieve(retriever, task, config, query, visible, tried, *, exclude_tool_ids=()):
    request = RetrievalRequest(
        query_text=query,
        top_k=config.retrieval_top_k,
        task_id=task.task_id,
        active_benchmark=task.benchmark,
        global_search_depth=config.global_search_depth,
        seen_tool_ids=tuple(visible),
        exclude_tool_ids=tuple(exclude_tool_ids),
        exclude_seen_tools=False,
        repeated_context_top_k=0,
        repeated_context_from_seen=True,
    )
    return retriever.retrieve(request, task)


def retrieval_query(need, candidate_values=()):
    if not isinstance(need, dict) or set(need) - {
        "goal",
        "semantic_input_types",
        "required_outputs",
        "domain",
        "failure_class",
    }:
        raise ValidationError("Invalid RetrievalNeed")
    for name in ("goal", "semantic_input_types", "required_outputs"):
        if not need.get(name):
            raise ValidationError("Incomplete RetrievalNeed")
    if not isinstance(need["goal"], str) or not all(
        (
            isinstance(need[k], list) and all((isinstance(x, str) and x.strip() for x in need[k]))
            for k in ("semantic_input_types", "required_outputs")
        )
    ):
        raise ValidationError("Invalid semantic retrieval fields")
    query = f"Capability: {need['goal']}; inputs: {', '.join(need['semantic_input_types'])}; outputs: {', '.join(need['required_outputs'])}"
    if need.get("domain"):
        query += "; domain: " + str(need["domain"])
    for value in candidate_values:
        if (
            isinstance(value, (str, int, float))
            and (not isinstance(value, bool))
            and str(value)
            and re.search("(?<!\\w)" + re.escape(str(value)) + "(?!\\w)", query, re.IGNORECASE)
        ):
            raise ValidationError("Candidate literal is forbidden in retrieval query")
    return query


def open_source(proposal, cap):
    """Normalize model-supplied JSON candidates without source provenance gates."""
    goal = proposal.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValidationError("A nonempty candidate investigation goal is required")
    values = proposal.get("candidates")
    if not isinstance(values, list) or not values:
        raise ValidationError("candidates must be a nonempty list")
    candidates = []
    seen = set()
    for index, value in enumerate(values):
        if value is None:
            raise ValidationError("Null is not a candidate")
        value = clone(value)
        key = canonical(value)
        if key in seen:
            raise ValidationError("Duplicate candidates are not allowed")
        seen.add(key)
        candidates.append(dict(candidate_value=value, source_order_index=index))
    if len(candidates) < 2:
        raise ValidationError("At least two candidates are required")
    if len(candidates) > cap:
        raise ValidationError("Candidate cap exceeded; no silent truncation")
    return (goal.strip(), candidates)


def branch_evidence(ref, candidate, store, actor):
    """Return successful JSON evidence produced during this candidate branch."""
    value = store.read_evidence(ref, actor)
    observation_id = ref["observation_id"]
    snapshot = store.snapshot(observation_id)
    if (
        snapshot["owner_scope"] != actor
        or observation_id not in candidate["branch_observation_ids"]
    ):
        raise ValidationError("Candidate evidence must be produced in this branch")
    return clone(value)
