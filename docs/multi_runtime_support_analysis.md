# KServe Analysis: Multi-Runtime Support in LLMInferenceService (SGLang & TensorRT-LLM)

* **Author:** Mohan Kumar
* **Status:** Analysis / Discovery
* **Target Version:** v0.17+
* **Companion Doc:** [multi_runtime_support_required_changes.md](./multi_runtime_support_required_changes.md)

---

## 1. Overview & Problem Statement

KServe's `LLMInferenceService` (LLMISVC, `serving.kserve.io/v1alpha2`) is becoming our standard control plane for serving large language models on Kubernetes. Today it is implemented and tested almost exclusively around **vLLM** (packaged inside the `ghcr.io/llm-d/llm-d-cuda` images) and the **llm-d** routing stack (Endpoint Picker / EPP, routing sidecar, tokenizer) layered on top of the **Gateway API Inference Extension (GIE)**.

To position KServe as a neutral, runtime-agnostic serving platform, we want first-class support for the other two dominant open inference engines:

1. **SGLang** (`sgl-project/sglang`) — high-throughput engine with RadixAttention prefix caching, strong P/D disaggregation, and broad hardware coverage.
2. **NVIDIA TensorRT-LLM** (`NVIDIA/TensorRT-LLM`) — NVIDIA's optimized engine, now shipping a modern OpenAI-compatible server (`trtllm-serve`) with a PyTorch backend.

This document answers four questions the team raised:

1. **What changes are required** to integrate SGLang and TensorRT-LLM with `LLMInferenceService`? *(High-level here; mechanics in the companion doc.)*
2. **Can the llm-d features be driven purely by metrics scraping** (rather than runtime-specific code)?
3. **Does llm-d publish native images** for SGLang and TensorRT-LLM the way it does for vLLM?
4. **What is the realistic feature gap** per runtime (single-node, multi-node, P/D disaggregation, prefix-cache routing, LoRA)?

> [!NOTE]
> **Bottom line up front:** The architecture is *structurally* runtime-agnostic — the controller delegates the full `PodSpec` to the user and the GIE routing layer talks to backends over the OpenAI HTTP API plus standard Prometheus metrics. **SGLang is already an officially supported llm-d engine (since llm-d v0.6.0).** **TensorRT-LLM is recognized in the GIE metric protocol but is not yet a tested path in llm-d** (tracked upstream in [llm-d#1550](https://github.com/llm-d/llm-d/issues/1550)). The KServe work is **new preset `LLMInferenceServiceConfig` objects plus a real (if contained) set of controller changes** — *not* a rewrite, but also *not* purely config: it requires a `spec.runtime` API addition (with v1alpha1 conversion + validation + CRD regen), a new path to generate the engine-aware metrics plugin into the default scheduler config, and a LoRA guard. The current implementation is effectively **vLLM-only** — preset selection, command templates, scheduler metrics, LoRA args, and the 8000 port assumption all encode vLLM today.

---

## 2. The Runtime Landscape

| Capability | vLLM (today) | SGLang | TensorRT-LLM |
|---|---|---|---|
| OpenAI server entrypoint | `vllm serve` | `python -m sglang.launch_server` | `trtllm-serve` |
| Default port | 8000 | **30000** | 8000 |
| OpenAI endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) | ✅ | ✅ | ✅ |
| Health endpoint | `/health` | `/health` (live), **`/health_generate`** (ready) | `/health` (Triton: `/v2/health/ready`) |
| Prometheus `/metrics` | ✅ always | ✅ via `--enable-metrics` | ⚠️ weak on bare `trtllm-serve`; full via Triton (`:8002`) or Dynamo |
| Loads HF checkpoint directly | ✅ | ✅ | ✅ **with `--backend pytorch`** (no `trtllm-build`) |
| Official image | `vllm/vllm-openai`, `ghcr.io/llm-d/llm-d-*` | `lmsysorg/sglang` | `nvcr.io/nvidia/tensorrt-llm/release`, `nvcr.io/nvidia/tritonserver` |
| Multi-node | LWS + Ray/native | `--nnodes/--node-rank/--dist-init-addr` | MPI/`mpirun` + LWS (Triton leader/worker) |
| P/D disaggregation | NIXL connector | Mooncake **or** NIXL (`--disaggregation-mode`) | `trtllm-serve disaggregated` (NIXL/UCX/MPI) |
| Prefix-cache (KV-event) routing | ✅ (ZMQ events) | ✅ (ZMQ events) | Protocol-aware, not wired in llm-d |
| LoRA dynamic serving | ✅ | ✅ (`--lora-paths`, `--max-loras-per-batch`) | Limited / engine-build bound |

**Key takeaway:** SGLang and `trtllm-serve` (PyTorch backend) both behave like vLLM at the contract level — point them at an HF model, they expose OpenAI endpoints and a health probe. The differences are in **flag names, ports, probe paths, metric names, and KV-transfer connectors** — exactly the things KServe currently hardcodes for vLLM.

---

## 3. How `LLMInferenceService` Works Today (and Where vLLM Is Baked In)

### 3.1 Template delegation — the runtime-agnostic core

The controller does **not** synthesize containers from scratch. It copies the user's (or a preset's) `PodSpec` verbatim into the generated `Deployment`/`LeaderWorkerSet`:

```go
// pkg/controller/v1alpha2/llmisvc/workload_single_node.go (and workload_multi_node.go)
d.Spec.Template.Spec = *llmSvc.Spec.Template.DeepCopy()
```

The API surface that feeds this ([`llm_inference_service_types.go`](file:///Users/mohankumar/kluisz/kserve/pkg/apis/serving/v1alpha2/llm_inference_service_types.go)):

- `spec.template` — main/decode/single-node leader `PodSpec`
- `spec.worker` — multi-node worker `PodSpec` (triggers a `LeaderWorkerSet`)
- `spec.prefill` — disaggregated prefill workload
- `spec.model` — model URI + name + LoRA spec
- `spec.parallelism` — tensor/pipeline/data parallel, expert parallel
- `spec.router` — `{ gateway, route, scheduler }` (the GIE/EPP layer)
- `spec.baseRefs` — references to `LLMInferenceServiceConfig` presets that get merged in

**Implication:** A user *can* run SGLang or TensorRT-LLM **today** by supplying a complete `spec.template` with the right image, command, port, and probes — no code change required. This is the same property that made the [NVIDIA DRA proposal](./nvidia_dra_support_proposal.md) a no-code-change feature. What's missing is the *ergonomic* path: presets, defaults, and the routing-layer glue.

### 3.2 The presets are vLLM-only

The merge logic ([`config_merge.go`](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/config_merge.go)) auto-applies "well-known" `LLMInferenceServiceConfig` presets based on the deployment shape. **Every shipped preset hardcodes vLLM:**

| Preset (`config/llmisvcconfig/`) | vLLM assumption |
|---|---|
| `config-llm-template.yaml` | `image: ghcr.io/llm-d/llm-d-cuda:v0.6.0`, `exec vllm serve /mnt/models --port 8000`, `containerPort: 8000`, probes on `/health` |
| `config-llm-decode-template.yaml` | vLLM + `ghcr.io/llm-d/llm-d-routing-sidecar` on 8000, decode on 8001 |
| `config-llm-prefill-template.yaml` | vLLM prefill on 8000 |
| `config-llm-worker-data-parallel.yaml` | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port 5555`, RoCE inference shell |
| `config-llm-scheduler.yaml` | EPP `ghcr.io/llm-d/llm-d-inference-scheduler` + `llm-d-uds-tokenizer`, `defaultEngine: vllm` |

### 3.3 vLLM-specific assumptions inventory

| Assumption | Location | Notes |
|---|---|---|
| Workload Service port **8000** (hardcoded, deliberately) | [`workload.go:131-148`](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/workload.go) | Comment explicitly says changing it "requires a lot of changes beyond this service (presets, routing sidecar flags…)". SGLang defaults to **30000**. |
| `vllm serve` command + flags | all `config-llm-*` presets | SGLang/TRT-LLM use different binaries and flag spellings |
| `/health` probe path | all presets | SGLang readiness is best served by `/health_generate`; Triton uses `/v2/health/ready` |
| `--tensor-parallel-size`, `--data-parallel-size`, `--data-parallel-rpc-port` | worker presets | map from `spec.parallelism.*`; flag names differ per engine |
| LoRA flags `--max-lora-rank/--max-loras/--max-cpu-loras` | presets | SGLang: `--max-lora-rank/--max-loras-per-batch/--lora-paths` |
| EPP metric plugin `defaultEngine: "vllm"`, and *only* in the v0.7.0 migration path | [`scheduler.go:1370-1529`](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/scheduler.go) | See §4.1 — the metrics plugin is **not** in the default scheduler config; it is injected only during deprecated-flag migration and hardcodes vLLM |
| Routing sidecar `--kv-connector=nixlv2`, `--vllm-port=8001` | decode presets | The sidecar only accepts `nixl`/`nixlv2`/`lmcache` (no `sglang` value); SGLang P/D rides **NIXL** too, so the preset reuses `nixlv2` and points `--vllm-port` at the SGLang decode port. TRT-LLM has no llm-d connector. |

---

## 4. Can the llm-d Features Be Driven by Metrics Scraping? — **Architecturally Yes; Not Wired By Default**

This is the most important architectural finding. The llm-d routing intelligence (load-aware, queue-aware, KV-cache-aware scheduling) is built on the **Gateway API Inference Extension (GIE) "Model Server Protocol"** ([proposal 003](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/docs/proposals/003-model-server-protocol/README.md)), which is **explicitly engine-agnostic**:

> *"The model server MUST implement OpenAI's Completions and Chat APIs"* and expose three core Prometheus gauges. *"The exact metric names don't necessarily need to be the same as the recommended names here; however the metric types and semantics MUST follow this doc."*

So the EPP doesn't *care* that vLLM emits `vllm:num_requests_waiting` — it cares that **some** gauge represents the queue depth. The mapping from semantic → concrete metric name is **configuration, selected by a pod label.** That is the protocol-level reason multi-runtime routing is *possible*. Whether KServe *exercises* it today is a separate question — and the answer is no.

### 4.1 KServe is NOT engine-aware by default today

> [!WARNING]
> **Correction to an earlier draft of this analysis.** KServe does *not* inject an engine-aware metrics plugin into the scheduler config it generates. The `model-server-protocol-metrics` plugin exists in the controller only inside a **one-time v0.7.0 migration path**, and only fires under narrow conditions:
>
> ```go
> // pkg/controller/v1alpha2/llmisvc/scheduler.go  (schedulerTransform, ≈1370-1406)
> if v.Compare(*semver.New("0.7.0")) < 0 { return nil }      // skipped for < 0.7.0
> writable := hasWritableConfigText(d)
> if writable {
>     extracted := extractDeprecatedMetricFlags(d)            // pulls 5 deprecated CLI flags
>     opts = append(opts, withCoreMetricsExtractorPlugin(extracted))
> }
> ```
>
> And `withCoreMetricsExtractorPlugin` is a **no-op when no deprecated flags were extracted** (`if len(extracted) == 0 { return nil }`), and even when it does fire it hardcodes `engineConfig["name"] = "vllm"` / `"defaultEngine": "vllm"` (`scheduler.go:1510-1529`). The **default generated scheduler config** (`config/llmisvcconfig/config-llm-scheduler.yaml`) ships **no `model-server-protocol-metrics` plugin at all**.
>
> **Refinement (per the llm-d [EPP configuration doc](https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/configuration.md)):** the EPP binary *does* auto-inject `metrics-data-source` + `core-metrics-extractor` into its `dataLayer` **by default** — but with **vLLM metric names**, because per the [model-servers doc](https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/model-servers.md) *"pods without the engine-type label will default to vLLM metrics names."* So metric *scraping* happens regardless; what's missing for SGLang/TRT-LLM is **the engine-type label that flips the per-engine name map**.
>
> **Net (revised):** the **primary** lever is **stamping the engine-type label on workload pods** (change #4) — that alone may make SGLang routing work if the EPP image's built-in per-engine map is used. The **secondary** lever is generating an explicit `model-server-protocol-metrics` plugin (change #5) to *pin* exact metric names and control the label key (KServe's controller plugin reads `inference.networking.k8s.io/engine-type`, while the EPP's default extractor reads `llm-d.ai/engine-type` — see the label-key caveat in §4.2). Either way it is a real change, not a defaults-table flip.

The migration-path plugin does, however, prove the *shape* of the solution. The fields it understands (`scheduler.go:1484-1490`) map 1:1 to the GIE per-engine spec, and a from-scratch generator must populate the same fields:

| Semantic | engineConfig field |
|---|---|
| Total queued requests | `queuedRequestsSpec` |
| Total running requests | `runningRequestsSpec` |
| KV-cache utilization | `kvUsageSpec` |
| LoRA info | `loraSpec` |
| Cache info (blocks/size) | `cacheInfoSpec` |

### 4.2 The per-runtime metric map (from the GIE/llm-d protocol)

| Semantic (GIE) | vLLM | SGLang | trtllm-serve | Triton TRT-LLM |
|---|---|---|---|---|
| Queued requests | `vllm:num_requests_waiting` | `sglang:num_queue_reqs` | `trtllm_num_requests_waiting` | `nv_trt_llm_request_metrics{request_type=waiting}` |
| Running requests | `vllm:num_requests_running` | `sglang:num_running_reqs` | `trtllm_num_requests_running` | `nv_trt_llm_request_metrics{request_type=scheduled}` |
| KV-cache utilization | `vllm:kv_cache_usage_perc` | `sglang:token_usage` | `trtllm_kv_cache_utilization` | `nv_trt_llm_kv_cache_block_metrics{...=fraction}` |

The metric names above use the **`sglang:` / `trtllm_` prefixes verbatim from the SGLang and llm-d model-server references** (e.g. `sglang:num_queue_reqs`); pin the engine image tag and these names together and re-verify against the running version. llm-d selects which row applies via the pod label **`llm-d.ai/engine-type: {vllm|sglang|trtllm-serve|triton-tensorrt-llm}`** (pods with no label default to vLLM).

> [!WARNING]
> **Label-key mismatch to resolve during implementation.** The llm-d EPP's *default* `core-metrics-extractor` reads the label key **`llm-d.ai/engine-type`**, but KServe's controller-injected `model-server-protocol-metrics` plugin sets `engineLabelKey: "inference.networking.k8s.io/engine-type"` (`scheduler.go:1523`). These are **different keys**. Whichever extractor is actually active in the pinned EPP image determines which key is read — so the controller should stamp the key the active extractor expects (or **both** keys defensively, and/or generate an explicit plugin that pins `engineLabelKey`). This is exactly the kind of detail that silently breaks routing if assumed.

> [!IMPORTANT]
> **Conclusion:** Queue-aware, load-aware, and KV-cache-utilization routing — the bulk of llm-d's "smart routing" value — are achievable for SGLang and TensorRT-LLM **through metrics scraping alone**, with no change to the **EPP/GIE binary**. What it *does* require on the KServe side: (1) generate the `model-server-protocol-metrics` plugin into the default scheduler config (§4.1), (2) stamp the `engine-type` pod label, (3) supply the per-engine metric names. The routing features that are *genuinely* engine-coupled (and out of reach of pure metrics scraping) are:
> - **LoRA — even basic serving.** KServe's LoRA path is vLLM-only: `attachLoRAAdapters` unconditionally appends vLLM flags (`--lora-modules`, `--enable-lora`, `--max-loras`) via `addLoRAVLLMArgs` (`workload_lora.go:214`) whenever `spec.model.lora` is set (`workload_storage.go`). For SGLang/TRT-LLM this emits wrong flags. LoRA-*affinity scoring* is additionally vLLM-biased (`vllm:lora_requests_info`).
> - **Precise prefix-cache routing** — needs the engine to publish KV-block events over ZMQ. vLLM and SGLang do; TensorRT-LLM does not in the llm-d wiring.
> - **P/D routing sidecar** — needs a KV-transfer connector the sidecar understands. The sidecar accepts only `nixl`/`nixlv2`/`lmcache`; **both vLLM and SGLang use `nixlv2`** (SGLang launches with `--disaggregation-transfer-backend nixl`), so this is *reusable*, not per-engine code. TensorRT-LLM has no llm-d sidecar connector (Dynamo path). See §6.3.

---

## 5. Does llm-d Publish Native Images for SGLang / TensorRT-LLM? — **No**

llm-d's custom-built model-server images are **vLLM-only**. There is **no `llm-d-sglang` or `llm-d-tensorrt-llm` image.** ([llm-d artifacts doc](https://github.com/llm-d/llm-d/blob/main/docs/getting-started/artifacts.md))

**llm-d-built model-server images (all vLLM):**

| Image | Target |
|---|---|
| `ghcr.io/llm-d/llm-d-cuda` | NVIDIA GPU (vLLM) |
| `ghcr.io/llm-d/llm-d-cuda-gb200` | GB200 / DeepEP |
| `ghcr.io/llm-d/llm-d-aws` | NVIDIA + EFA/libfabric + NIXL |
| `ghcr.io/llm-d/llm-d-rocm` | AMD ROCm |
| `ghcr.io/llm-d/llm-d-xpu` / `llm-d-hpu` / `llm-d-cpu` | Intel XPU / Gaudi / CPU |

These are described as "vLLM images with features not yet merged upstream." For non-vLLM engines, llm-d points users at the **upstream** images:

- **SGLang → `lmsysorg/sglang`** (e.g. `lmsysorg/sglang:v0.5.10.post1`) — upstream, *not* an llm-d image.
- **TensorRT-LLM → no llm-d path at all** today.

**Routing/control-plane images llm-d does publish (engine-neutral):** `llm-d-inference-scheduler` (EPP), `llm-d-routing-sidecar`, `llm-d-uds-tokenizer`. KServe already consumes these in `config-llm-scheduler.yaml` / decode presets.

> [!NOTE]
> Because there is no native image, KServe presets for SGLang must reference `lmsysorg/sglang:<tag>`, and TensorRT-LLM presets must reference `nvcr.io/nvidia/tensorrt-llm/release` (NGC auth required) or a Triton image. This also means KServe inherits each engine's image cadence and CVE posture directly from upstream, with no llm-d hardening layer.

---

## 6. Per-Runtime Deep Dive

### 6.1 SGLang — *officially supported by llm-d (v0.6.0)*

- **Status in llm-d:** Delivered under EPIC [llm-d#403](https://github.com/llm-d/llm-d/issues/403). SGLang options exist across the inference-scheduling, P/D-disaggregation, optimized-baseline, tiered-prefix-cache, and precise-prefix-cache-routing "well-lit paths." A `llm-d-sglang-overview` Grafana dashboard ships in-repo. (Caveat: GPU-only.)
- **Launch:** `python -m sglang.launch_server --model-path /mnt/models --served-model-name <name> --host 0.0.0.0 --port 30000 [--tp-size N --dp-size N --pp-size N]`.
- **Health:** `/health` (liveness, lightweight) and `/health_generate` (readiness, runs a real forward pass). ⚠️ On some versions these behaved identically ([sglang#12731](https://github.com/sgl-project/sglang/issues/12731)) — probe config must be validated against the pinned version.
- **Metrics:** `--enable-metrics`, served at `/metrics`. Queue gauge is **`sglang:num_queue_reqs`**, running is `sglang:num_running_reqs`, KV/token usage is `sglang:token_usage` / `sglang:num_used_tokens` (colon prefix, per the [SGLang production-metrics doc](https://github.com/sgl-project/sglang/blob/main/docs/references/production_metrics.md)). Always re-verify the exact names against the pinned image tag before wiring the EPP.
- **Multi-node:** `--nnodes`, `--node-rank`, `--dist-init-addr`; `--tp-size` is the *total* TP across nodes. Maps cleanly to LeaderWorkerSet.
- **P/D:** native, `--disaggregation-mode prefill|decode` with `--disaggregation-transfer-backend mooncake|nixl` (mooncake is SGLang's default; **NIXL** is what llm-d/KServe use). SGLang can be unified by its own `sglang_router`, but in llm-d/KServe the existing EPP + `llm-d-routing-sidecar` does the routing — the sidecar transfers KV over **`nixlv2`**, the same connector as vLLM (llm-d [PR #580](https://github.com/llm-d/llm-d/pull/580): *"NIXL is used as the data transfer backend"*). There is **no `--connector=sglang`** — the sidecar only accepts `nixl`/`nixlv2`/`lmcache`. See §6.3 for integration effort.
- **LoRA:** `--lora-paths`, `--max-loras-per-batch`, `--max-lora-rank`.

### 6.2 TensorRT-LLM — *recognized in protocol, not yet a tested llm-d path*

- **Status in llm-d:** **No recipe, no CI, no guide** — open enhancement [llm-d#1550](https://github.com/llm-d/llm-d/issues/1550). The GIE metric mapping (`trtllm-serve`, `triton-tensorrt-llm`) is defined, and design docs name TRT-LLM as a target engine, but nothing is wired end-to-end.
- **Modern path:** `trtllm-serve <model> --backend pytorch [--tp_size N --pp_size N --ep_size N --host 0.0.0.0 --port 8000]`. **With `--backend pytorch` it loads an HF checkpoint directly — no `trtllm-build` AOT compile step.** This removes the single biggest historical obstacle to treating TRT-LLM like vLLM. (The `--backend tensorrt` path still needs the AOT engine build.)
- **Health:** `trtllm-serve` exposes `/health`; Triton exposes `/v2/health/ready` + `/v2/health/live`.
- **Metrics — the weak spot:** bare `trtllm-serve` `/metrics` returns iteration stats only and has historically blocked under load; robust Prometheus comes from **Triton** (`:8002/metrics`, `nv_inference_*` / `nv_trt_llm_*`) or from running under **NVIDIA Dynamo** (`trtllm_*`). Native `llm_*` Prometheus metrics are tracked in [TensorRT-LLM#4926](https://github.com/NVIDIA/TensorRT-LLM/issues/4926).
- **Multi-node:** MPI/`mpirun`; Triton leader/worker via LeaderWorkerSet.
- **P/D:** `trtllm-serve disaggregated` with separate context/generation servers; KV transfer via NIXL/UCX/MPI. **No llm-d routing-sidecar connector exists**, so KServe's P/D presets cannot drive TRT-LLM P/D without new glue.
- **Ecosystem note — Dynamo:** NVIDIA pushes **Dynamo** as its K8s orchestration/disaggregation layer *above* the engines, and is **collaborating** (not purely competing) with llm-d — notably contributing **NIXL** and committing to integrate the Dynamo Planner/KV-Cache-Manager into llm-d. For KServe, TRT-LLM is the engine; Dynamo is the NVIDIA-preferred orchestration option; llm-d shares components with it.

### 6.3 SGLang Prefill/Decode disaggregation — *how hard to integrate?* → **Medium, mostly preset work**

The encouraging finding: **SGLang P/D reuses KServe's existing routing stack rather than needing new machinery.** KServe's current vLLM P/D presets put the model server on port 8001 and inject an `llm-d-routing-sidecar` on 8000 with `--vllm-port=8001 --kv-connector=nixlv2`. SGLang slots into the *same* pattern because:

- SGLang P/D uses **NIXL** as the KV-transfer backend (`--disaggregation-mode prefill|decode --disaggregation-transfer-backend nixl`), and the routing sidecar's `nixlv2` connector is exactly that — llm-d [PR #580](https://github.com/llm-d/llm-d/pull/580) ("NIXL is used as the data transfer backend") shipped SGLang P/D on this path in llm-d v0.6.0.
- The sidecar's `--vllm-port` flag is **vLLM-named but engine-neutral** (just "the decode server's port"), so the sidecar block is reused almost verbatim.
- **Heterogeneous TP** between prefill and decode pools is allowed (SGLang stages mismatched KV layouts via `SGLANG_DISAGG_STAGING_BUFFER=1`), so the two pools need not share parallelism.
- **The scheduler side is engine-agnostic.** Per the llm-d [EPP scheduling doc](https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/scheduling.md), the `disagg-profile-handler` runs separate prefill and decode scheduling profiles (via `prefill-endpoints-filter` / `decode-endpoints-filter` label selectors), returns the **decode endpoint as the primary destination**, and **injects the chosen prefill endpoint as a request header**; the decode pod's sidecar then coordinates the remote prefill. None of this is vLLM-specific — so SGLang P/D needs **no scheduler change**, only the engine flags + the (reused) sidecar.

What keeps it *Medium* rather than *Low*: the **RDMA/InfiniBand device wiring** (`--disaggregation-ib-device mlx5_*`, bootstrap port 8998) is cluster-specific and the fiddliest piece, and the SGLang↔sidecar NIXL handshake must be **integration-tested against KServe's exact preset** (PR #580 validated llm-d's own helm path, not KServe's). Concrete presets and the effort breakdown are in the companion doc, [§6.1](./multi_runtime_support_required_changes.md). Net: ~2 new presets reusing the IB auto-inference preamble + 1 e2e test; **no new controller code** beyond the runtime-based preset selection already required for single-node.

### 6.4 Speculative decoding — *how hard to integrate?* → **Low, no controller change at all**

Speculative decoding is the **cheapest** advanced feature to support because it is **entirely launch-time flags on the model container** — it never touches the Service, probes, EPP, or routing layer. SGLang exposes `--speculative-algorithm {EAGLE|EAGLE3|NEXTN|STANDALONE|NGRAM|DFLASH}` plus `--speculative-draft-model-path`, `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens`. vLLM (`--speculative-config`) and TensorRT-LLM (LLM-API speculative config) have equivalents.

The only wrinkle is **artifacts, not code**: EAGLE/EAGLE3/STANDALONE need a *second* checkpoint (the draft/EAGLE head) on disk; `NGRAM` needs none and `NEXTN`/MTP weights are usually bundled in the base model. So it ships as a **preset variant (or `*_ADDITIONAL_ARGS`)** plus, optionally, a draft-model download reusing the LoRA-style storage-initializer plumbing. A first-class `spec.model.speculativeDecoding` API field is a *nice-to-have*, not a requirement. Effort breakdown and example flags: companion doc [§6.2](./multi_runtime_support_required_changes.md).

---

## 7. Feature Support Matrix (Target State in KServe)

| Feature | vLLM (today) | SGLang (achievable) | TensorRT-LLM (achievable) |
|---|---|---|---|
| Single-node serve | ✅ shipped | ✅ new preset | ✅ new preset (`--backend pytorch`) |
| Multi-node (LWS) | ✅ | ✅ new worker preset | ⚠️ MPI/Triton — harder, Triton or Dynamo |
| OpenAI routing via EPP | ✅ | ✅ engine-type=sglang | ✅ engine-type=trtllm-serve |
| Load/queue/KV-util routing (metrics) | ✅ | ✅ metric-name config | ✅ via Triton/Dynamo metrics |
| Precise prefix-cache routing (ZMQ KV events) | ✅ | ✅ | ❌ not wired |
| LoRA-affinity routing | ✅ | ⚠️ partial (no affinity scorer) | ❌ |
| P/D disaggregation | ✅ NIXL (`nixlv2` sidecar) | ✅ **Medium** — NIXL reuses the `nixlv2` sidecar (§6.3) | ❌ no llm-d connector (Dynamo path) |
| Speculative decoding (EAGLE/Medusa/n-gram/MTP) | ✅ launch flags | ✅ **Low** — launch flags only, no controller change (§6.4) | ✅ launch config (draft engine) |
| Autoscale / scale-to-zero (HPA/KEDA) | ✅ | ✅ (metric-name change) | ✅ (metric source change) |
| Native llm-d hardened image | ✅ | ❌ upstream `lmsysorg/sglang` | ❌ `nvcr.io` |

---

## 8. Gaps, Risks & Recommendation

**Low-risk / high-value (do first):**
- SGLang single-node + multi-node presets and engine-type metric wiring. This is "follow the vLLM pattern with different flags," and llm-d has already proven the path.
- **Speculative decoding (any runtime)** — pure launch-flag feature, no controller change (§6.4). Genuinely the lowest-effort item; ship it as a preset variant whenever convenient.
- TensorRT-LLM single-node preset via `trtllm-serve --backend pytorch`, with EPP metrics sourced appropriately.

**Medium-risk:**
- **Port:** the workload Service *and* the scheduler InferencePool `targetPorts` are hardcoded to **8000** (`workload.go:130`, `config-llm-scheduler.yaml:23`). Note this is a **KServe self-imposed constraint, not an llm-d requirement** — the [InferencePool doc](https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/inferencepool.md) treats 8000 as a vLLM *example* and lets `targetPorts` be any port. **Milestone decision: v1 forces every runtime onto port 8000** (the SGLang preset passes `--port 8000` to override its 30000 default; probes target 8000). Port *indirection* (deriving the port and threading it through Service + probes + InferencePool + routing-sidecar flags) is deferred to a later phase. The two docs are aligned on this.
- **Probes:** path differences (`/health` vs `/health_generate` vs `/v2/health/ready`) are carried entirely in each preset's `PodSpec` — no core change needed; just correct presets.
- Metric-name maturity (TRT-LLM Prometheus support is still maturing; SGLang names verified as `sglang:*`) — pin engine image tags and metric names together.

**Medium-risk:**
- **SGLang P/D disaggregation** — *not* high-risk, because it reuses the existing `nixlv2` routing sidecar (§6.3). The real cost is RDMA/IB device wiring + an integration test of the SGLang↔sidecar NIXL handshake against KServe's preset. ~2 presets, no new controller code.

**High-risk / defer:**
- TensorRT-LLM P/D disaggregation and prefix-cache routing — no llm-d connector / no KV-event publishing wired. Likely a Dynamo-mediated path, out of scope for the first milestone.
- LoRA-affinity routing for non-vLLM engines — upstream-blocked.

**Recommendation:** Pursue a **phased adoption** — (1) SGLang single + multi-node + smart routing (+ speculative-decoding variant, essentially free), (2) TensorRT-LLM single-node + smart routing via Triton/Dynamo metrics, (3) **SGLang P/D disaggregation** (reuses the `nixlv2` sidecar) + prefix-cache, (4) TRT-LLM P/D as upstream/Dynamo support matures. The engineering mechanics, file-by-file, are detailed in the companion document: [multi_runtime_support_required_changes.md](./multi_runtime_support_required_changes.md).

---

## 9. References

- KServe LLMISVC types — `pkg/apis/serving/v1alpha2/llm_inference_service_types.go`
- Preset configs — `config/llmisvcconfig/config-llm-*.yaml`
- Workload reconcilers — `pkg/controller/v1alpha2/llmisvc/workload*.go`
- Scheduler/EPP reconciler — `pkg/controller/v1alpha2/llmisvc/scheduler.go`, `router.go`
- GIE Model Server Protocol — https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/docs/proposals/003-model-server-protocol/README.md
- llm-d artifacts/images — https://github.com/llm-d/llm-d/blob/main/docs/getting-started/artifacts.md
- llm-d model-server protocol + engine-type label + per-engine metric map — https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/model-servers.md
- llm-d InferencePool (selector, targetPorts, endpointPickerRef; no mandated port) — https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/inferencepool.md
- llm-d EPP configuration (dataLayer auto-injects core-metrics-extractor; plugins/profiles) — https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/configuration.md
- llm-d EPP scheduling (disagg-profile-handler, filters/scorers/picker) — https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/scheduling.md
- llm-d SGLang EPIC — https://github.com/llm-d/llm-d/issues/403
- llm-d TensorRT-LLM request — https://github.com/llm-d/llm-d/issues/1550
- SGLang server args — https://docs.sglang.io/advanced_features/server_arguments.html
- SGLang production metrics — https://github.com/sgl-project/sglang/blob/main/docs/references/production_metrics.md
- SGLang P/D disaggregation — https://docs.sglang.io/advanced_features/pd_disaggregation.html
- trtllm-serve — https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html
- TensorRT-LLM Prometheus metrics issue — https://github.com/NVIDIA/TensorRT-LLM/issues/4926
- NVIDIA Dynamo ↔ llm-d — https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference/
