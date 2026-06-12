# KServe: Engineering Changes Required to Support SGLang & TensorRT-LLM Runtimes

* **Author:** Mohan Kumar
* **Status:** Proposed
* **Target Version:** v0.17+
* **Companion Doc:** [multi_runtime_support_analysis.md](./multi_runtime_support_analysis.md)

---

## 1. Strategy: Preset-First, Minimal Core Change

The `LLMInferenceService` controller already delegates the entire `PodSpec` to the user/preset (`d.Spec.Template.Spec = *llmSvc.Spec.Template.DeepCopy()`). Because of this, a **large share of multi-runtime support ships as data (new `LLMInferenceServiceConfig` presets).** However — correcting an earlier, too-optimistic framing — the Go side is **not trivial**: it includes a **full `spec.runtime` API change** (field + v1alpha1 conversion + validation + CRD/client regen), a **new scheduler-config generation path** to inject the engine-aware metrics plugin (which is *absent* from the default config today), and a **LoRA guard** (the current LoRA path emits vLLM-only flags unconditionally). Treat Phase 1 as a Medium-sized change, not a YAML drop-in.

We organize the work into three tiers:

| Tier | What | Code vs Config |
|---|---|---|
| **A. Presets** | New `LLMInferenceServiceConfig` objects for SGLang & TRT-LLM (single-node, multi-node, scheduler/metrics) | Config only |
| **B. Controller plumbing** | Engine-type label, port indirection, probe indirection, EPP metric defaults | Small Go changes |
| **C. Advanced** | P/D connectors, prefix-cache KV events, LoRA affinity for non-vLLM | Larger / upstream-dependent |

> [!NOTE]
> A user can run SGLang/TRT-LLM **today** by hand-writing a full `spec.template`. Everything below is about making it a **supported, ergonomic, routing-integrated** path rather than a DIY override.

---

## 2. Change Set Overview

| # | Area | File(s) | Type | Tier |
|---|---|---|---|---|
| 1 | SGLang single-node preset | `config/llmisvcconfig/config-llm-sglang-template.yaml` (new) | Config | A |
| 2 | SGLang multi-node worker preset | `config/llmisvcconfig/config-llm-sglang-worker.yaml` (new) | Config | A |
| 3 | TRT-LLM single-node preset | `config/llmisvcconfig/config-llm-trtllm-template.yaml` (new) | Config | A |
| 4 | `spec.runtime` field + engine-type label (full API change: conversion, CRD, validation) | types, `v1alpha1` conversion, validation, generated clients, `workload*.go` | Go | B |
| 5 | **Generate** `model-server-protocol-metrics` plugin into default scheduler config (sglang/trtllm) | `scheduler.go`, `config-llm-scheduler.yaml` | Go | B |
| 6 | v1: force port 8000 (presets); later: port indirection (un-hardcode) | presets; `workload.go`, `config-llm-scheduler.yaml` | Config (v1) / Go (later) | B |
| 7 | Probe paths carried in presets (no core change) | presets | Config | A |
| 8 | Parallelism flag mapping per engine (guarded; matches `GetSize` contract) | presets (templating) | Config | A/B |
| 9 | Preset selection wiring by runtime | `config_merge.go` | Go | B |
| 10 | LoRA: reject `spec.model.lora` for non-vLLM (v1) or runtime-aware args (later) | `llm_inference_service_validation.go`, `workload_lora.go` | Go | B/C |
| 11 | SGLang P/D presets (reuse `nixlv2` sidecar) + speculative-decoding variant | decode/prefill presets (new), spec-decoding preset (§6.1, §6.2) | Config | C |
| 12 | Charts/manifests packaging + regenerate CRDs | `charts/...`, `config/` kustomize, `make manifests` | Config | A |
| 13 | E2E + docs + samples | `test/e2e/llmisvc`, `docs/samples/llmisvc` | Test/Docs | A/B |

---

## 3. Tier A — New Runtime Presets

### 3.1 SGLang single-node preset (new)

`config/llmisvcconfig/config-llm-sglang-template.yaml`:

```yaml
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceServiceConfig
metadata:
  name: kserve-config-llm-sglang-template
spec:
  annotations:
    serving.kserve.io/model-based-routing-enabled: "true"
  template:
    containers:
      - name: main
        image: lmsysorg/sglang:v0.5.10.post1        # upstream — no native llm-d image (see analysis §5)
        imagePullPolicy: IfNotPresent
        ports:
          - containerPort: 8000                      # align to KServe's expected 8000 (override SGLang's 30000)
            protocol: TCP
        command: ["/bin/bash", "-c"]
        args:
          - |-
            exec python3 -m sglang.launch_server \
              --model-path /mnt/models \
              --served-model-name {{ .Spec.Model.Name }} \
              --host 0.0.0.0 \
              --port 8000 \
              --enable-metrics \
              {{- if .Spec.Parallelism }}
              {{- if .Spec.Parallelism.Tensor }}--tp-size {{ .Spec.Parallelism.Tensor }}{{ end }} \
              {{- if .Spec.Parallelism.Data }}--dp-size {{ .Spec.Parallelism.Data }}{{ end }} \
              {{- if .Spec.Parallelism.Pipeline }}--pp-size {{ .Spec.Parallelism.Pipeline }}{{ end }} \
              {{- end }}
              ${SGLANG_ADDITIONAL_ARGS:-}
        env:
          - name: HF_HUB_CACHE
            value: /models
        livenessProbe:
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 10
          failureThreshold: 10
        readinessProbe:
          httpGet: { path: /health_generate, port: 8000 }   # real forward-pass readiness
          periodSeconds: 5
          failureThreshold: 3
        startupProbe:
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 10
          failureThreshold: 60
```

> [!WARNING]
> **SGLang-specific gotchas to encode and document:**
> 1. **Port:** SGLang defaults to **30000**; we force `--port 8000` to match KServe's hardcoded Service and InferencePool `targetPorts` (see change #6). Port indirection (using 30000 natively) is deferred.
> 2. **Metric names:** SGLang uses the **`sglang:` colon prefix** (`sglang:num_queue_reqs`, `sglang:num_running_reqs`, `sglang:token_usage`), confirmed against the [SGLang production-metrics doc](https://github.com/sgl-project/sglang/blob/main/docs/references/production_metrics.md). Pin the image tag and the EPP metric names (change #5) together and re-verify on upgrade.
> 3. **`/health` vs `/health_generate`:** validate against the pinned version ([sglang#12731](https://github.com/sgl-project/sglang/issues/12731)) — on some builds `/health` also runs generation and can flap under load.

### 3.2 SGLang multi-node worker preset (new)

`config/llmisvcconfig/config-llm-sglang-worker.yaml` — mirrors `config-llm-worker-data-parallel.yaml` but with SGLang's distributed flags. **The parallelism mapping must match KServe's existing contract — do not invent one.**

> [!WARNING]
> **The earlier draft of this preset was wrong on three counts** and would fail validation or produce incorrect args:
> 1. **Worker requires data *or* pipeline parallelism, not tensor.** `llm_inference_service_validation.go:262` rejects a `worker` unless `parallelism.IsDataParallel()` *or* `parallelism.IsPipelineParallel()`. A worker preset keyed only on `--tp-size` is invalid. Tensor parallelism alone never creates a `LeaderWorkerSet`.
> 2. **`DataLocal` is GPUs-per-node, not node count.** `GetSize()` (`llm_inference_service_types_func.go:78`) derives the LWS group size as `Data / DataLocal` (data-parallel) or `Pipeline` (pipeline-parallel) — it never derives size from `Tensor`. `--nnodes` must be wired from `GetSize()`, not from `DataLocal`.
> 3. **Unguarded field access** (`{{ .Spec.Parallelism.Tensor }}`) panics the template when the field is nil. Every field must be `{{- if }}`-guarded.
>
> The SGLang↔KServe mapping must be defined explicitly: `Tensor → --tp-size` (intra-node), `Data/DataLocal → --dp-size` + node fan-out, `Pipeline → --pp-size`, node count = `GetSize()`. SGLang `--tp-size` is the *total* TP across all nodes, so reconcile that with how KServe counts replicas before finalizing.

```yaml
# leader (spec.template) + worker (spec.worker); both share dist-init-addr from LWS_LEADER_ADDRESS.
# nnodes is derived from ParallelismSpec.GetSize() (Data/DataLocal or Pipeline) — NEVER from Tensor.
args:
  - |-
    exec python3 -m sglang.launch_server \
      --model-path /mnt/models \
      {{- if .Spec.Parallelism }}
      {{- if .Spec.Parallelism.Tensor }} --tp-size {{ .Spec.Parallelism.Tensor }}{{ end }} \
      {{- if .Spec.Parallelism.Data }} --dp-size {{ .Spec.Parallelism.Data }}{{ end }} \
      {{- if .Spec.Parallelism.Pipeline }} --pp-size {{ .Spec.Parallelism.Pipeline }}{{ end }} \
      {{- end }}
      --nnodes {{ .Spec.Parallelism.GetSize }} \
      --node-rank ${LWS_WORKER_INDEX:-0} \
      --dist-init-addr ${LWS_LEADER_ADDRESS}:50000 \
      --host 0.0.0.0 --port 8000 --enable-metrics
```

> [!NOTE]
> `GetSize` is a method, not a field — exposing it to the preset template requires either a template func or precomputing the value in the controller and injecting it as an env var (the existing data-parallel preset uses `${LWS_*}` env vars rather than templating struct methods). Decide the mechanism during implementation; the existing `config-llm-worker-data-parallel.yaml` is the reference.

### 3.3 TensorRT-LLM single-node preset (new)

`config/llmisvcconfig/config-llm-trtllm-template.yaml`:

```yaml
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceServiceConfig
metadata:
  name: kserve-config-llm-trtllm-template
spec:
  template:
    containers:
      - name: main
        image: nvcr.io/nvidia/tensorrt-llm/release:1.0.0   # PIN a tag — NGC has no stable :latest; NGC auth (imagePullSecret) required
        ports:
          - containerPort: 8000
        command: ["/bin/bash", "-c"]
        args:
          - |-
            exec trtllm-serve /mnt/models \
              --backend pytorch \
              --host 0.0.0.0 --port 8000 \
              {{- if .Spec.Parallelism.Tensor }}--tp_size {{ .Spec.Parallelism.Tensor }}{{ end }} \
              {{- if .Spec.Parallelism.Pipeline }}--pp_size {{ .Spec.Parallelism.Pipeline }}{{ end }} \
              ${TRTLLM_ADDITIONAL_ARGS:-}
        livenessProbe:
          httpGet: { path: /health, port: 8000 }
        readinessProbe:
          httpGet: { path: /health, port: 8000 }
```

> [!IMPORTANT]
> - `--backend pytorch` loads an **HF checkpoint directly — no `trtllm-build` step**. This is what makes a vLLM-style preset viable. The `tensorrt` backend would require an extra engine-build init-container and is out of scope for v1.
> - **NGC image pull:** TRT-LLM images live on `nvcr.io` and need an `imagePullSecret`; document this prominently.
> - **Metrics caveat:** bare `trtllm-serve` `/metrics` is weak. For real routing signals either run a **Triton** variant (metrics on `:8002`) or run under **Dynamo** (`trtllm_*` metrics). The EPP metric config (change #5) must point at whichever source is in use.

---

## 4. Tier B — Controller Plumbing

### 4.1 Engine-type label propagation (change #4)

The EPP already reads `inference.networking.k8s.io/engine-type` and defaults to `vllm` ([`scheduler.go:1510-1529`](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/scheduler.go)). We need the controller to **stamp this label on workload pods** based on the runtime.

**Proposed:** add an optional discriminator to the API and propagate it as a pod label + EPP `defaultEngine`:

```go
// pkg/apis/serving/v1alpha2/llm_inference_service_types.go
type LLMInferenceServiceSpec struct {
    // +optional
    // Runtime selects the inference engine for metric/label wiring.
    // One of: vllm (default), sglang, trtllm-serve, triton-tensorrt-llm.
    Runtime string `json:"runtime,omitempty"`
    // ...
}
```

- Workload reconcilers (`workload_single_node.go`, `workload_multi_node.go`) add `inference.networking.k8s.io/engine-type: <runtime>` to the pod template labels.
- `scheduler.go` sets `"defaultEngine": <runtime>` instead of the hardcoded `"vllm"`.
- Default empty → `vllm`, preserving current behavior (no breaking change).

> [!IMPORTANT]
> **Adding `spec.runtime` is a full-stack API change, not a one-line struct edit.** The complete checklist:
>
> | Touch point | File / artifact |
> |---|---|
> | Field definition | `pkg/apis/serving/v1alpha2/llm_inference_service_types.go` (`LLMInferenceServiceSpec`, ~line 65) |
> | **v1alpha1 compatibility + conversion** | `pkg/apis/serving/v1alpha1/llm_inference_service_conversion.go` — `convertSpecToV1Alpha2` / `convertSpecFromV1Alpha2` (~line 130). If v1alpha1 lacks `runtime`, decide round-trip behavior (annotation stash, like criticality, or drop). **Do not silently lose the field on conversion.** |
> | Defaults | `pkg/apis/serving/v1alpha2/llm_inference_service_defaults.go` (default empty→vllm) |
> | **Validation (enum)** | `pkg/apis/serving/v1alpha2/llm_inference_service_validation.go` — **this is the correct validation location, not the webhook package.** Reject unknown runtime values; cross-validate runtime vs incompatible features (see LoRA, §4.6). |
> | Generated CRDs | regenerate `config/crd/full/` + `config/crd/minimal/` and the Helm chart CRDs (`charts/kserve-llmisvc-crd*`) via `make manifests` |
> | Generated clients | `make generate` → deepcopy, openapi, clientset/listers/informers |
> | Tests | types/defaults/validation/conversion unit tests + e2e |

### 4.2 EPP per-engine metric config (change #5) — *bigger than a defaults table*

> [!IMPORTANT]
> **Scope — primary lever is the label; the plugin is the override.** Per the llm-d [EPP configuration doc](https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/configuration.md), the EPP **auto-injects** `metrics-data-source` + `core-metrics-extractor` into its `dataLayer` **by default** — so metric scraping happens out of the box, but with **vLLM names** (the [model-servers doc](https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/model-servers.md): *"pods without the engine-type label will default to vLLM metrics names"*). Therefore:
> - **Change #4 (the engine-type pod label) is the primary enabler** — it flips the per-engine metric-name map without touching the scheduler config, *if* the EPP image's built-in map covers the runtime and you stamp the key it reads.
> - **This change #5 (explicit plugin generation) is the robust override** — needed to (a) **pin exact metric names** rather than trust the image's built-in map across upgrades, and (b) **control the `engineLabelKey`** (the KServe-injected plugin reads `inference.networking.k8s.io/engine-type`; the EPP default reads `llm-d.ai/engine-type` — they differ).
>
> Note the **existing** controller plugin path (`withCoreMetricsExtractorPlugin`, `scheduler.go:1483`) is only reachable in the one-time v0.7.0 deprecated-flag migration, is a no-op when nothing was extracted (`scheduler.go:1493`), and hardcodes `name: "vllm"`. KServe's **default** `config-llm-scheduler.yaml` contains no explicit metrics plugin. So the override path must be **generated fresh**, not reused from the migration code.

Required work, in order:

1. **Stamp the engine-type label** on workload pods (change #4) — *the* critical step. Set the key(s) the active EPP extractor reads (`llm-d.ai/engine-type`, and/or `inference.networking.k8s.io/engine-type` if relying on KServe's plugin). Validate which one the pinned EPP image honors.
2. **Add a generation step** (not a migration step) that injects a `model-server-protocol-metrics` plugin into the default scheduler `EndpointPickerConfig` whenever `spec.runtime != "" && spec.runtime != "vllm"`, to pin names + label key. Must run independent of `schedulerTransform`'s version/`writable`/`extracted` gates.
3. **Set `defaultEngine` / `engineConfigs[].name`** and `engineLabelKey` from `spec.runtime`.
4. **Supply the per-engine metric names** via a table (correct names below — note `sglang:` colon prefix):

```go
var engineMetricDefaults = map[string]map[string]string{
    "vllm": { // current EPP built-in default; only needed if we make it explicit
        "queuedRequestsSpec":  "vllm:num_requests_waiting",
        "runningRequestsSpec": "vllm:num_requests_running",
        "kvUsageSpec":         "vllm:kv_cache_usage_perc",
    },
    "sglang": {
        "queuedRequestsSpec":  "sglang:num_queue_reqs",
        "runningRequestsSpec": "sglang:num_running_reqs",
        "kvUsageSpec":         "sglang:token_usage",
    },
    "trtllm-serve": {
        "queuedRequestsSpec":  "trtllm_num_requests_waiting",
        "runningRequestsSpec": "trtllm_num_requests_running",
        "kvUsageSpec":         "trtllm_kv_cache_utilization",
    },
}
```

The user can still override via the inline `EndpointPickerConfig` mechanism; this provides correct defaults. **No EPP/GIE *image* change is needed** — the GIE/EPP binary is engine-agnostic and already scrapes metrics by default (analysis §4) — but the **KServe controller change is non-trivial** (label stamping + a fresh config-generation path, not a lookup-table tweak).

### 4.3 Un-hardcode the workload Service port (change #6)

[`workload.go:131-148`](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/workload.go) pins the Service to `8000`. The maintainers documented the rationale directly in the code, and it is worth quoting in full because it scopes the work precisely:

> The rationale for not supporting changing the vLLM port is because it requires a lot of changes beyond this service (presets, routing sidecar flags, routing sidecar container spec, etc) or selecting the exact port is not trivial, for example, when there is P/D, the request goes through the routing sidecar, so here we would need to use the routing sidecar container ports.
>
> The presets we bundle assume 8000 for the "main receiver" (being it vLLM or P/D routing sidecar) and, for now, I'd discourage that use case unless we have a use case that requires not assuming "main receiver" port.

**This multi-runtime effort is exactly that use case** (SGLang defaults to 30000; Triton's metrics live on 8002), so the constraint is now load-bearing rather than hypothetical. Two options:

- **Option A (minimal, recommended for v1):** keep `8000` and require runtime presets to bind their server to `8000` (as the SGLang preset above does with `--port 8000`). Zero core change; documented constraint. **This is what Phase 1–4 assume.**
- **Option B (clean, later):** derive the port from the main container's first `containerPort` (or a new `spec.router.targetPort`) and thread it through the Service, probes, InferencePool `targetPorts`, and — the hard part the comment calls out — the **P/D routing sidecar's `--port` / `--vllm-port` flags and the "main receiver" selection** when a sidecar is present.

Recommend shipping **Option A** first, tracking **Option B** as a follow-up.

> [!NOTE]
> **Commitment for Option B:** we will attempt the port-indirection change and **validate it end-to-end** — single-node, multi-node (LWS), *and* P/D-with-routing-sidecar (where the "main receiver" port is the sidecar's, not the model server's) — before proposing it for merge. Given the maintainer's stated concern that the P/D sidecar case is the tricky one, this E2E coverage (especially the sidecar "main receiver" path) is a hard prerequisite, not a nice-to-have. If E2E surfaces that the sidecar contract can't cleanly carry an arbitrary port, we keep Option A and document the 8000 requirement as permanent.

### 4.4 Probe indirection (change #7)

Health-path differences (`/health` vs `/health_generate` vs `/v2/health/ready`) are fully expressible in the preset's `PodSpec` (as shown in §3), so **no core change is strictly required** — the presets carry the right probes. Only if we add server-side probe defaulting would Go change be needed; not recommended for v1.

### 4.5 Preset selection wiring (change #9)

`config_merge.go` auto-selects the vLLM presets by deployment shape. Extend the selection so that when `spec.runtime` (change #4.1) is `sglang`/`trtllm-serve`, the controller merges the corresponding `kserve-config-llm-<runtime>-*` preset instead of the vLLM one. Keep `WellKnownDefaultConfigs` additive and backward-compatible.

### 4.6 LoRA is vLLM-only — make it runtime-aware or reject it (change #10)

> [!WARNING]
> **`spec.model.lora` currently emits vLLM flags unconditionally.** When LoRA adapters are set, the controller resolves them (`workload_storage.go`) and `attachLoRAAdapters` appends vLLM-specific flags via `addLoRAVLLMArgs` (`workload_lora.go:214`): `--lora-modules`, `--enable-lora`, `--max-loras`, `--max-lora-rank`, `--max-cpu-loras`. SGLang's flags differ (`--lora-paths`, `--max-loras-per-batch`, `--max-lora-rank`) and TensorRT-LLM's LoRA is engine-build-bound. Feeding vLLM flags to a SGLang/TRT-LLM binary breaks startup.

Two options, pick per phase:

- **v1 (safe, recommended): validate and reject.** In `llm_inference_service_validation.go`, error if `spec.model.lora` is set and `spec.runtime` is not `vllm` (or empty). Clear failure beats a broken pod.
- **Later: runtime-aware LoRA injection.** Generalize `attachLoRAAdapters` to branch on `spec.runtime` and emit the correct flag set (and gate the LoRA-affinity scorer, which is vLLM-biased). Larger change; defer to Phase 4.

---

## 5. Parallelism & LoRA Flag Mapping (reference for preset authors)

| Concept | vLLM | SGLang | TensorRT-LLM (`trtllm-serve`) |
|---|---|---|---|
| Tensor parallel | `--tensor-parallel-size` | `--tp-size` | `--tp_size` |
| Pipeline parallel | `--pipeline-parallel-size` | `--pp-size` | `--pp_size` |
| Data parallel | `--data-parallel-size` | `--dp-size` | (n/a; use ep/replicas) |
| Expert parallel | `--enable-expert-parallel` | `--ep-size` | `--ep_size` |
| Multi-node addr | `--data-parallel-address` | `--dist-init-addr` | MPI/`mpirun` |
| Node count / rank | (LWS env) | `--nnodes` / `--node-rank` | MPI hostfile |
| Served name | `--served-model-name` | `--served-model-name` | (model id) |
| Max LoRA rank | `--max-lora-rank` | `--max-lora-rank` | (engine-build) |
| LoRA adapters | `--max-loras` / `--max-cpu-loras` | `--max-loras-per-batch` / `--lora-paths` | limited |
| Port | `--port` | `--port` (def 30000) | `--port` (def 8000) |
| Metrics enable | (always on) | `--enable-metrics` | (Triton/Dynamo) |

The `spec.parallelism.*` fields ([`llm_inference_service_types.go`](file:///Users/mohankumar/kluisz/kserve/pkg/apis/serving/v1alpha2/llm_inference_service_types.go)) already capture the *intent*; only the **flag spelling in the preset template** differs per engine. No API change needed for parallelism.

---

## 6. Tier C — Advanced Features (Phase 2+)

| Feature | SGLang | TensorRT-LLM | Required work |
|---|---|---|---|
| **P/D disaggregation** | `--disaggregation-mode prefill\|decode` + `--disaggregation-transfer-backend nixl`; **reuses the existing `nixlv2` routing sidecar** (no `--connector=sglang`) | `trtllm-serve disaggregated` (NIXL/UCX) | New SGLang decode/prefill presets mirroring the vLLM ones (§6.1); for TRT-LLM no llm-d connector → Dynamo-mediated, defer |
| **Precise prefix-cache routing** | ZMQ KV events supported | not wired | SGLang: add `precise-prefix-cache-scorer` config (mirror `e2e-gpt-oss/llmisvc_config_prefix_cache.yaml`); TRT-LLM: blocked |
| **LoRA-affinity scoring** | partial | ❌ | Upstream GIE LoRA scorer is vLLM-biased; defer |
| **Multi-node TRT-LLM** | n/a | MPI/Triton leader-worker (LWS) | Separate Triton-based preset + MPI bootstrap; higher effort |

### 6.1 SGLang Prefill/Decode disaggregation — **Medium effort, reuses the existing sidecar**

**How easy is it?** Easier than it first appears, because **SGLang P/D rides the same `llm-d-routing-sidecar` (`nixlv2`) that vLLM already uses** — there is no SGLang-specific connector to build. llm-d [PR #580](https://github.com/llm-d/llm-d/pull/580) proved this path: *"NIXL is used as the data transfer backend."* The KServe work is therefore **two new presets that mirror the existing vLLM `config-llm-prefill-template.yaml` / `config-llm-decode-template.yaml`, swapping the `vllm serve` command for `sglang.launch_server` with disaggregation flags and keeping the sidecar block almost verbatim.**

Mapping the existing vLLM decode preset to SGLang (KServe's decode preset puts the model server on **8001** and the sidecar on **8000** with `--vllm-port=8001 --kv-connector=nixlv2`):

```yaml
# config/llmisvcconfig/config-llm-sglang-decode-template.yaml (new) — mirrors config-llm-decode-template.yaml
spec:
  template:
    containers:
      - name: main                                   # SGLang decode server on 8001
        image: lmsysorg/sglang:v0.5.10.post1
        ports: [{ containerPort: 8001 }]
        command: ["/bin/bash", "-c"]
        args:
          - |-
            exec python3 -m sglang.launch_server --model-path /mnt/models \
              --host 0.0.0.0 --port 8001 --enable-metrics \
              --disaggregation-mode decode \
              --disaggregation-transfer-backend nixl \
              --disaggregation-bootstrap-port 8998 \
              ${KSERVE_INFER_IB_DEVICE:+--disaggregation-ib-device ${KSERVE_INFER_IB_DEVICE}} \
              ${SGLANG_ADDITIONAL_ARGS:-}
        # probes on 8001 (use /health_generate for readiness)
      - name: llm-d-routing-sidecar                  # UNCHANGED from the vLLM preset
        image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.7.1
        ports: [{ containerPort: 8000 }]
        args:
          - "--port=8000"
          - "--vllm-port=8001"                       # flag is vLLM-named but engine-neutral: just the decode port
          - "--kv-connector=nixlv2"                  # SAME connector as vLLM
```

The prefill preset mirrors `config-llm-prefill-template.yaml`: SGLang `main` on **8000** with `--disaggregation-mode prefill --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998`, no sidecar.

> [!IMPORTANT]
> **What makes it Medium, not Low:**
> 1. **RDMA/IB device wiring.** NIXL KV transfer needs RDMA NICs. SGLang's `--disaggregation-ib-device` (e.g. `mlx5_0,mlx5_1`) must be set; the existing vLLM presets already do RoCE auto-inference in their bash preamble — reuse that to populate `KSERVE_INFER_IB_DEVICE`. This is the fiddliest part and is cluster-specific.
> 2. **Handshake validation.** The sidecar's `nixlv2` connector and SGLang's NIXL transceiver must agree on the bootstrap protocol. PR #580 validated this in llm-d's own helm path (`ms-pd/values_sglang.yaml`), **not** against KServe's exact decode preset — so an integration test is required, not assumed.
> 3. **`--vllm-port` is misleadingly named** but engine-neutral (it's just "the decode server port"). Leave it; don't rename in the sidecar.
> 4. **Heterogeneous TP is allowed** (prefill and decode pools may use different `--tp-size`; SGLang stages KV via `SGLANG_DISAGG_STAGING_BUFFER=1`), so the two presets need not share parallelism.
>
> **Net effort:** ~2 presets + IB-device plumbing reuse + 1 e2e test. No new controller code beyond preset selection (change #9 already routes by `spec.runtime`). The mooncake backend is SGLang's *default* but NIXL is what integrates with the existing sidecar — **always pass `--disaggregation-transfer-backend nixl`.**

### 6.2 Speculative decoding — **Low effort, no controller change (all runtimes)**

**How easy is it?** Speculative decoding is **purely a set of launch-server flags** on the model container — it does not touch the controller, the Service, the EPP, or the routing layer. It is therefore the *cheapest* feature to add: it ships as **preset variants (or just extra `*_ADDITIONAL_ARGS`)**, not code.

SGLang speculative decoding flags (verified against `server_args.py`):

```yaml
# add to the SGLang single-node preset args, or expose as a preset variant
exec python3 -m sglang.launch_server --model-path /mnt/models \
  --host 0.0.0.0 --port 8000 --enable-metrics \
  --speculative-algorithm EAGLE3 \                    # EAGLE | EAGLE3 | NEXTN | STANDALONE | NGRAM | DFLASH
  --speculative-draft-model-path <eagle-head-repo> \  # required for EAGLE/EAGLE3/STANDALONE (separate checkpoint)
  --speculative-num-steps 5 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 8
```

Equivalent flags exist in the other runtimes (one launch-time config each):
- **vLLM:** `--speculative-config '{...}'` (JSON; supersedes the old `--speculative_model`/`--num_speculative_tokens`) — n-gram, draft-model, EAGLE, Medusa.
- **TensorRT-LLM:** speculative config at `trtllm-serve` / LLM-API launch (draft-target, EAGLE/EAGLE3, Medusa, MTP/ReDrafter, Lookahead) — but generally needs the draft/EAGLE **engine** specified.

> [!NOTE]
> **The only real consideration is the draft-model artifact.** EAGLE/EAGLE3/STANDALONE need a *second* checkpoint (the EAGLE head / draft model) on disk alongside the base model. Options, in increasing effort:
> 1. **Bake it into the image or model dir** — zero KServe change; user supplies a model URI that already contains the draft weights.
> 2. **Mount via an extra volume / second storage-initializer** — small preset change; reuses the LoRA-style download plumbing (`collectLoRADownloadPairs` pattern in `workload_lora.go`).
> 3. *(optional, later)* **A first-class `spec.model.speculativeDecoding` API field** (algorithm + draftModelURI + numSteps) that the controller expands into the right per-engine flags and a draft-model download. Nice ergonomics, but **not required** — flags + a volume cover it today.
>
> `NGRAM` needs **no** draft model (pure n-gram cache) and `NEXTN`/MTP weights are usually bundled in the base model — those are truly zero-artifact. Constraints to document: `STANDALONE`/`NGRAM`/`DFLASH` conflict with `--enable-dp-attention`; the V2 overlap scheduler only supports `--speculative-eagle-topk 1`. None of these interact with the routing layer.
>
> **Net effort:** Low. Add an optional preset variant (or document `SGLANG_ADDITIONAL_ARGS`) + optionally a draft-model download. No API change for v1.

---

## 7. Packaging, Testing & Docs (change #11, #12)

- **Charts/kustomize:** add the new presets to `charts/kserve-runtime-configs/files/llmisvcconfigs/` and `config/llmisvcconfig/kustomization.yaml` so they install alongside the vLLM presets.
- **Samples:** add `docs/samples/llmisvc/sglang-*` and `docs/samples/llmisvc/trtllm-*` mirroring the existing `e2e-gpt-oss/` and `qwen-llm-d-envoy-ai-gateway/` examples (including the Envoy AI Gateway `AIGatewayRoute` wiring, which is engine-agnostic and needs no change).
- **E2E:** extend `test/e2e/llmisvc` with an SGLang single-node smoke test first (cheapest, llm-d-proven), then TRT-LLM.
- **Validation:** if `spec.runtime` is added, validate the enum in `pkg/apis/serving/v1alpha2/llm_inference_service_validation.go` (the spec-validation layer the admission webhook invokes), **not** directly in `pkg/webhook/admission/llminferenceservice`. Add the LoRA-vs-runtime cross-check here too (§4.6).

---

## 8. Phased Roadmap & Effort

| Phase | Scope | Effort | Risk |
|---|---|---|---|
| **1** | SGLang single-node preset (port 8000) + `spec.runtime` API (incl. conversion/CRD/validation) + engine-type label + **generated** EPP metrics plugin + LoRA-reject validation + sample + e2e | 1 preset, full API change, scheduler config-gen | **Med** (API surface is the bulk; routing is llm-d-proven) |
| **2** | SGLang multi-node (LWS) preset with guarded `GetSize`-based parallelism + smart-routing validation | 1 preset | Med |
| **3** | TensorRT-LLM single-node (`trtllm-serve --backend pytorch`, pinned NGC tag) + Triton/Dynamo metric source wiring + NGC pull docs | 1 preset + metric config | Med |
| **4** | SGLang P/D disaggregation (NIXL via existing `nixlv2` sidecar, §6.1) + speculative decoding presets (§6.2) + precise prefix-cache routing + runtime-aware LoRA | 2 P/D presets + spec-decoding variants + EPP config + `workload_lora.go` | Med–High |
| **5** | Port indirection (un-hardcode 8000 across Service/probes/InferencePool/sidecar), TRT-LLM multi-node & P/D (Dynamo), LoRA affinity | Core change + upstream deps | High |

> [!NOTE]
> Phases 1–3 deliver "serve any of the three major runtimes with smart (queue/load/KV-util) routing" — the headline goal. **Phase 1 is sized "Medium," not "small": the `spec.runtime` API change (conversion + CRD regen + validation) and the new scheduler-config generation path are the real cost — not the preset YAML.** Phases 4–5 chase advanced disaggregation/caching/LoRA and are gated on upstream llm-d/Dynamo maturity.

---

## 9. Summary of Required Code Changes (the short list)

1. **`spec.runtime` enum field** (default `vllm`) — **full API change**: `llm_inference_service_types.go`, v1alpha1 **conversion** (`llm_inference_service_conversion.go`), defaults, **validation** (`llm_inference_service_validation.go` — *not* the webhook package), regenerated CRDs + clients, tests.
2. **Stamp `inference.networking.k8s.io/engine-type` pod label** from `spec.runtime` — `workload_single_node.go`, `workload_multi_node.go`.
3. **Generate** the `model-server-protocol-metrics` plugin (with per-engine names + `defaultEngine` from `spec.runtime`) into the **default** scheduler `EndpointPickerConfig` — `scheduler.go`, `config-llm-scheduler.yaml`. *(Not just a defaults table — the plugin is absent from the default config today.)*
4. **Preset selection by runtime** — `config_merge.go`.
5. **LoRA guard** — reject `spec.model.lora` for non-vLLM runtimes in `llm_inference_service_validation.go` (v1); runtime-aware args in `workload_lora.go` (later).
6. *(Phase 5)* **Port indirection** — `workload.go` + `config-llm-scheduler.yaml` + routing-sidecar flags.

Everything else is **net-new YAML presets, samples, charts, e2e tests, and docs** — no rearchitecting of the controller.
