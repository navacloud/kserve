<!--
GitHub issue draft for multi-runtime (SGLang + TensorRT-LLM) support in LLMInferenceService.
Edit freely, then paste the body into a new KServe issue (or use: gh issue create -F docs/multi_runtime_support_issue.md).
Suggested title:  feat(llmisvc): support SGLang and TensorRT-LLM (trtllm-serve) runtimes in LLMInferenceService
Suggested labels: kind/feature, area/llmisvc, priority/important-longterm
-->

# feat(llmisvc): support SGLang and TensorRT-LLM (trtllm-serve) runtimes in LLMInferenceService

## Summary

`LLMInferenceService` (LLMISVC) is effectively **vLLM-only** today — preset commands, the workload Service port, health probes, scheduler metrics, and LoRA args all encode vLLM. This issue tracks making LLMISVC **multi-runtime**, starting with **SGLang** (already an officially supported llm-d engine since v0.6.0) and **NVIDIA TensorRT-LLM via `trtllm-serve`** (the modern OpenAI-compatible server; `--backend pytorch` loads HF checkpoints directly with no `trtllm-build` step).

Detailed analysis and the file-by-file change set live in:
- `docs/multi_runtime_support_analysis.md`
- `docs/multi_runtime_support_required_changes.md`

## Motivation

As KServe positions LLMISVC as the standard LLM control plane, it should be runtime-agnostic and support the major open inference engines, not just vLLM. The Gateway API Inference Extension (GIE) / llm-d routing layer is already engine-agnostic (OpenAI API + Prometheus metrics selected by an `engine-type` pod label), so the work is mostly **additive presets + contained controller plumbing**, not a rewrite.

## Current state (where vLLM is baked in)

- Presets in `config/llmisvcconfig/config-llm-*.yaml` hardcode `vllm serve` + `ghcr.io/llm-d/llm-d-cuda`.
- Workload Service + InferencePool `targetPorts` hardcoded to **8000** (`workload.go:130`, `config-llm-scheduler.yaml:23`) — a KServe self-imposed constraint, not an llm-d one (the InferencePool protocol allows any `targetPorts`).
- Scheduler metrics: the `model-server-protocol-metrics` plugin is injected **only** in the v0.7.0 deprecated-flag migration path and hardcodes `defaultEngine: "vllm"` (`scheduler.go:1483-1529`); the default scheduler config ships **no** explicit metrics plugin. (The EPP binary auto-injects `core-metrics-extractor` by default, but with vLLM metric names.)
- LoRA: `attachLoRAAdapters` emits vLLM-only flags unconditionally (`workload_lora.go:214`).
- Preset selection is vLLM-only (`config_merge.go`).

## Goals

- Run SGLang and `trtllm-serve` as first-class runtimes via a `spec.runtime` discriminator (default `vllm`, fully backward compatible).
- Smart routing (queue / load / KV-cache-utilization via EPP) works for both, driven by the `engine-type` label + per-engine metric names.
- SGLang multi-node (LeaderWorkerSet) and P/D disaggregation (reusing the existing `nixlv2` routing sidecar).
- Speculative decoding support (launch-flag feature, all runtimes).

## Non-goals (this issue)

- TensorRT-LLM **Triton** backend (`triton-tensorrt-llm`, v2 protocol) — separate follow-up.
- TensorRT-LLM P/D disaggregation (no llm-d sidecar connector; Dynamo-mediated) — deferred.
- LoRA-affinity scoring for non-vLLM engines (upstream GIE is vLLM-biased) — deferred.
- Full port indirection (un-hardcoding 8000) — deferred; v1 forces port 8000.

## Design overview

1. Add `spec.runtime` enum (`vllm` | `sglang` | `trtllm-serve`), default `vllm`.
2. Stamp an `engine-type` label on workload pods so the EPP selects the right per-engine metric map. **Resolve the label-key mismatch** (EPP default reads `llm-d.ai/engine-type`; KServe's injected plugin reads `inference.networking.k8s.io/engine-type`).
3. Generate a `model-server-protocol-metrics` plugin into the default scheduler config to pin exact metric names + label key for non-vLLM runtimes.
4. Ship per-runtime presets selected by `spec.runtime` in `config_merge.go`.
5. Guard LoRA: reject `spec.model.lora` for non-vLLM runtimes (v1).

### Per-engine metric map (GIE / llm-d model-server protocol)

| Semantic | vLLM (default) | SGLang | trtllm-serve |
|---|---|---|---|
| Queued requests | `vllm:num_requests_waiting` | `sglang:num_queue_reqs` | `trtllm_num_requests_waiting` |
| Running requests | `vllm:num_requests_running` | `sglang:num_running_reqs` | `trtllm_num_requests_running` |
| KV-cache utilization | `vllm:kv_cache_usage_perc` | `sglang:token_usage` | `trtllm_kv_cache_utilization` |

---

## Tasks

### Phase 1 — SGLang single-node + `spec.runtime` API + routing

**API: `spec.runtime` (full-stack change)**
- [ ] Add `Runtime` field to `LLMInferenceServiceSpec` — `pkg/apis/serving/v1alpha2/llm_inference_service_types.go`
- [ ] Default empty → `vllm` — `llm_inference_service_defaults.go`
- [ ] Enum validation + reject unknown values — `pkg/apis/serving/v1alpha2/llm_inference_service_validation.go`
- [ ] v1alpha1 conversion (decide round-trip: annotation stash or drop, don't silently lose) — `pkg/apis/serving/v1alpha1/llm_inference_service_conversion.go`
- [ ] Regenerate CRDs (`config/crd/full`, `config/crd/minimal`, `charts/kserve-llmisvc-crd*`) via `make manifests`
- [ ] Regenerate deepcopy/openapi/clients via `make generate`
- [ ] Unit tests: types / defaults / validation / conversion

**Engine-type label + EPP metrics**
- [ ] Stamp `engine-type` label on workload pods from `spec.runtime` — `workload_single_node.go`, `workload_multi_node.go`
- [ ] **Investigate & resolve label-key mismatch** (`llm-d.ai/engine-type` vs `inference.networking.k8s.io/engine-type`) against the pinned EPP image; stamp the correct key (or both)
- [ ] Generate `model-server-protocol-metrics` plugin into the default scheduler `EndpointPickerConfig` for non-vLLM runtimes (independent of the migration path) — `scheduler.go`, `config-llm-scheduler.yaml`
- [ ] Set `defaultEngine` / `engineConfigs[].name` / `engineLabelKey` from `spec.runtime`
- [ ] Add per-engine metric-name table (see map above)

**SGLang single-node preset**
- [ ] New `config/llmisvcconfig/config-llm-sglang-template.yaml` — `lmsysorg/sglang:<pinned>`, `sglang.launch_server`, `--port 8000`, `--enable-metrics`
- [ ] Readiness probe on `/health_generate`, liveness/startup on `/health` (validate per pinned version, ref sglang#12731)
- [ ] Map `spec.parallelism` → `--tp-size` / `--dp-size` / `--pp-size` (guarded)
- [ ] Preset selection by runtime — `config_merge.go`

**LoRA guard**
- [ ] Reject `spec.model.lora` when `spec.runtime != vllm` — `llm_inference_service_validation.go`

**Packaging / tests / docs**
- [ ] Add preset to `charts/kserve-runtime-configs/files/llmisvcconfigs/` + `config/llmisvcconfig/kustomization.yaml`
- [ ] Sample under `docs/samples/llmisvc/sglang/`
- [ ] E2E smoke test (SGLang single-node) — `test/e2e/llmisvc`

### Phase 2 — SGLang multi-node (LeaderWorkerSet)
- [ ] New `config/llmisvcconfig/config-llm-sglang-worker.yaml` mirroring `config-llm-worker-data-parallel.yaml`
- [ ] Parallelism mapping per existing contract: worker requires data **or** pipeline parallelism (`validation.go:262`); node count from `ParallelismSpec.GetSize()` (Data/DataLocal or Pipeline), **never** Tensor; all fields `{{- if }}`-guarded
- [ ] Decide how to expose `GetSize()` to templates (template func vs precompute as env) — ref existing data-parallel preset
- [ ] Validate smart routing across multi-node pods
- [ ] E2E multi-node test

### Phase 3 — TensorRT-LLM (`trtllm-serve`) single-node
- [ ] New `config/llmisvcconfig/config-llm-trtllm-template.yaml` — **pinned** `nvcr.io/nvidia/tensorrt-llm/release:<tag>`, `trtllm-serve /mnt/models --backend pytorch --port 8000`
- [ ] Document NGC `imagePullSecret` requirement
- [ ] Map `spec.parallelism` → `--tp_size` / `--pp_size` / `--ep_size` (guarded)
- [ ] Health probe on `/health`
- [ ] EPP metric wiring — decide metric source (bare `trtllm-serve /metrics` is weak; use Triton `:8002` or Dynamo `trtllm_*`); set `engine-type: trtllm-serve` + metric names
- [ ] Sample + e2e

### Phase 4 — Advanced (SGLang)
- [ ] SGLang P/D presets (`config-llm-sglang-prefill-template.yaml` / `-decode-template.yaml`) mirroring the vLLM P/D presets, **reusing the existing `nixlv2` routing sidecar** (no `--connector=sglang`); launch with `--disaggregation-mode prefill|decode --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998`
- [ ] RDMA/IB device wiring (`--disaggregation-ib-device`) reusing the RoCE auto-inference preamble
- [ ] Integration test the SGLang↔sidecar NIXL handshake against KServe's preset (PR #580 validated llm-d's helm path, not KServe's)
- [ ] Speculative decoding preset variant / `SGLANG_ADDITIONAL_ARGS` (`--speculative-algorithm EAGLE3 ...`); optional draft-model download (reuse LoRA-style storage-init plumbing)
- [ ] Precise prefix-cache routing config for SGLang (`precise-prefix-cache-scorer`, ZMQ KV events) mirroring `e2e-gpt-oss/llmisvc_config_prefix_cache.yaml`

### Phase 5 — Deferred / follow-up (separate issues)
- [ ] **Port indirection** (un-hardcode 8000 across Service + probes + InferencePool + routing sidecar). The maintainers documented the rationale for the current 8000 constraint in `workload.go`:
  > The rationale for not supporting changing the vLLM port is because it requires a lot of changes beyond this service (presets, routing sidecar flags, routing sidecar container spec, etc) or selecting the exact port is not trivial, for example, when there is P/D, the request goes through the routing sidecar, so here we would need to use the routing sidecar container ports. The presets we bundle assume 8000 for the "main receiver" (being it vLLM or P/D routing sidecar) and, for now, I'd discourage that use case unless we have a use case that requires not assuming "main receiver" port.

  Multi-runtime is precisely that use case (SGLang defaults to 30000, Triton metrics on 8002). **We will attempt the change and validate it end-to-end** — single-node, multi-node (LWS), and P/D-with-routing-sidecar (where the "main receiver" port is the sidecar's, not the model server's) — before proposing it for merge. If E2E shows the P/D sidecar contract can't cleanly carry an arbitrary port, we keep forcing 8000 and document it as a permanent requirement. (Phases 1–4 assume forced 8000 and are unblocked regardless.)
- [ ] Runtime-aware LoRA injection (SGLang `--lora-paths` / `--max-loras-per-batch`) instead of reject
- [ ] TensorRT-LLM Triton backend (`triton-tensorrt-llm`, v2 protocol, leader/worker multi-node)
- [ ] TensorRT-LLM P/D (Dynamo-mediated)

## Key decisions / open questions

- [ ] **Label key**: which `engine-type` key does the pinned EPP image actually read? Stamp that one (or both).
- [ ] **Port**: v1 forces all runtimes to 8000 (SGLang `--port 8000` overriding its 30000 default), honoring the documented `workload.go` rationale. Port indirection is a Phase 5 follow-up that we commit to attempting and **testing end-to-end (incl. the P/D routing-sidecar "main receiver" path)** before merge; if it can't be done cleanly, 8000 stays permanent.
- [ ] **v1alpha1 conversion** of `spec.runtime`: annotation stash vs drop.
- [ ] **TRT-LLM metric source**: bare `trtllm-serve` vs Triton vs Dynamo for production routing signals.

## References

- `docs/multi_runtime_support_analysis.md`, `docs/multi_runtime_support_required_changes.md`
- llm-d model-server protocol + engine-type label: https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/model-servers.md
- llm-d InferencePool: https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/inferencepool.md
- llm-d EPP configuration / scheduling: https://github.com/llm-d/llm-d/tree/main/docs/architecture/core/router/epp
- llm-d SGLang EPIC #403; P/D PR #580; TensorRT-LLM request #1550
- GIE model-server protocol: https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/docs/proposals/003-model-server-protocol/README.md
- SGLang server args: https://docs.sglang.io/advanced_features/server_arguments.html
- trtllm-serve: https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html
