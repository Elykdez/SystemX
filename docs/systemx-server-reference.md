# SystemX server reference

SystemX is a local HTTP service for bounded System One decisions. Give it observed
state and typed questions; it returns choices, yes/no probabilities or ordinal
scores. Applications validate the answer and execute actions.

Model identity and revision are pinned in `model_config.py`; backend-specific
loading lives in `backend.py`. The setup dependencies select the compatible
adapter. Replacing the provider requires qualification, not just changing a
model name. See the [backend boundary](system-one-model.md).

Everything runs locally after setup. The server needs no game or browser, and
any HTTP client can use it. Run examples from the repository root. Return to
the [quick-start README](../README.md).

## Quick start on Windows

Requirements: Python 3.12, Git, and an NVIDIA driver compatible with CUDA 12.6 for GPU use. CPU mode is available. Allow several GB for dependencies plus about 9.3 GB for the BF16 checkpoint.

On this prepared machine, **double-click `start.bat`**. It opens a visible console titled `SystemX Server`, shows loading and request logs, and keeps startup errors visible. Stop the server with Ctrl+C or close its console window. The launcher works regardless of the current directory.

From a terminal, the batch launcher also accepts the PowerShell options: `start.bat -Device cpu -Port 8001`.

For a fresh checkout, **double-click `setup.bat` first**, then `start.bat`.
Setup keeps its window open to show the result. From a terminal:

```powershell
# Run from your SystemX checkout root.
& ./setup.bat
& ./start.bat
```

Setup creates this folder's `.venv`, installs pinned dependencies, and downloads missing model files into `.cache/huggingface` by default. Both `.venv` and `.cache` are excluded from Git, so a fresh checkout needs setup and internet access. A complete cached model is reused. Setup does not load the model onto the GPU. Run setup on each new machine instead of moving a virtual environment between Python installations.

`start.bat` downloads missing model files when dependencies are already installed,
but does not install Python packages or create the environment. The PowerShell
scripts remain available for direct use. Python 3.12 and Git must be installed
before running setup.

When the complete pinned model is cached, the launcher automatically uses offline
mode to avoid waiting on Hugging Face during startup. It prints whether it will
use the local model or download missing files. An explicitly set
`HF_HUB_OFFLINE` is respected and its value is restored when the script exits.

`start.ps1` runs in the foreground at `http://127.0.0.1:8000`. In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Wait for `ready: true` and `status: ready` before sending decisions. Loading and warmup happen after the HTTP listener starts. A failed startup reports `status: failed`; inspect the server console. Stop with **Ctrl+C** to release model memory. Nothing starts automatically when this folder is opened.

CPU-only installation and launch:

```powershell
& ./setup.bat -Device cpu
& ./start.bat -Device cpu -Port 8001
```

CUDA errors are reported instead of silently falling back to CPU. `start.ps1` defaults to CUDA; pass `-Device cpu` explicitly when using it. The direct Python entry point also accepts `--device cpu` and defaults to `SYSTEMX_DEVICE` or CUDA.

## Make a decision

Send JSON to `POST /v1/systemone`. Here is a complete PowerShell request:

```powershell
$payload = @{
    model = 'StartLux-Decision-v1.1-4B-Instant'
    state = 'I was charged twice for one order. Please refund the duplicate charge.'
    questions = @{
        route = @{
            type = 'choice'
            instructions = 'Which support team should handle this ticket?'
            criteria = @{
                billing = 'Payment, duplicate charge, invoice or refund problems.'
                technical = 'Software crashes, access problems or broken features.'
            }
        }
        refund = @{
            type = 'noul'
            instructions = 'Did the customer explicitly request a refund?'
        }
        impact = @{
            type = 'score'
            instructions = "Rate the customer's financial impact."
            criteria = @('No financial loss.', 'An incorrect charge needs correction.', 'Repeated major financial losses.')
        }
    }
} | ConvertTo-Json -Depth 8
$result = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/systemone -ContentType 'application/json' -Body $payload
$result.answers | ConvertTo-Json -Depth 8
```

Python, using only the standard library:

```python
import json
from urllib.request import Request, urlopen

payload = {
    "model": "StartLux-Decision-v1.1-4B-Instant",
    "state": "I was charged twice for one order.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which support team should handle this ticket?",
            "criteria": {
                "billing": "Payment, duplicate charge, invoice or refund problems.",
                "technical": "Software crashes, access problems or broken features.",
            },
        },
    },
}
request = Request(
    "http://127.0.0.1:8000/v1/systemone",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=30) as response:
    result = json.load(response)
print(result["answers"]["route"]["choice"])
```

`state` can be any JSON value, including a structured object. Question IDs and candidate keys are application-defined. Describe what each candidate means; opaque IDs alone provide little useful context. The application remains responsible for validating and carrying out the selected action.

## API contract

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Status JSON: `ready`, `status`, `model`, `revision`, `device`, and `workers` (`total`, `ready`, `busy`, `loading`, `failed`); includes `load_seconds` after successful initialization. |
| `GET /ready` | Same status; HTTP 200 when ready, otherwise 503. |
| `GET /v1/models` | Lists the pinned model. |
| `POST /v1/systemone` | Evaluates the supplied questions. Optional `parallel` (default `true`) spreads several questions across free workers. |
| `POST /v1/systemone/batch` | `{"requests": [...]}` with 1-8 independent decision requests, evaluated concurrently across workers; returns `results` in request order, each a normal response or `{"error": {"status", "detail"}}`. |
| `GET /v1/vision/models` | Image models kept in memory (see below): `folder`, `device`, and per model its `name`, `sha256`, `vram_mb`, `warmup_ms`, `calls`, last `ms`, and the `meta` from its JSON file. |
| `POST /v1/vision/reload` | Loads the vision folder's models again (after an addon put a new one there). |
| `POST /v1/vision/detect?model=NAME&width=W&height=H[&min_score=S]` | Body: one raw uint8 image, `H x W x 3` bytes. Returns `rows` (for a detector `[x0, y0, x1, y1, score, class]`; rows scoring under `min_score` are left out) and `ms`. |
| `GET /docs` | Interactive API documentation. |
| `GET /openapi.json` | Machine-readable request schema. |

Image models: small TorchScript models that addons export into the server's vision folder
(`SYSTEMX_VISION_DIR`, default `.cache/vision`) are loaded after the decision model, into the same process and
CUDA context, and stay in memory while the server runs; each takes one uint8 image `(H, W, 3)`
and returns rows of numbers. `vision.py` knows no game: a JSON file beside each model gives its
name, classes, and input. The Chrono Divide addon's unit detector (about 0.5 M parameters, 2 MB)
answered in 4.7 ms per 850x736 frame including the upload on an RTX 3070, with about 2 MB of
weights resident and about 40 MB at its peak while running (measured 2026-09-28, outside the
server; its client falls back to the CPU, about 70 ms, when no server runs).

The decision response contains `model`, `answers`, `usage`, and `latency_ms`. Each answer uses the corresponding question ID:

| Question type | Input criteria | Answer fields |
| --- | --- | --- |
| `choice` | Object with 2-16 candidate keys and string descriptions (null uses the key). | `type`, `choice`, `probabilities` keyed by candidate, `confidence`. |
| `noul` | Optional object with `true` and/or `false` descriptions. | `type`, `noul`: probability of yes from 0 to 1. |
| `score` | Ordered list of 2-10 string descriptions, from lowest to highest. | `type`, `score`: expected level index from 0 to N-1; `legend`, `probabilities`, `confidence`. |

`confidence` is a top-two probability gap, not calibrated correctness. Instant returns `output_tokens: 0` because it generates no reasoning. `latency_ms` excludes transport and initial loading.

Only `StartLux-Decision-v1.1-4B-Instant` is accepted. AutoThink and legacy model aliases are rejected.

## Limits and client behavior

- Maximum request body: 64 KiB (16 MiB for `/v1/vision/detect`, one raw image); 1-8 questions per request.
- Maximum instruction/criterion length: 2,000 characters; question and candidate keys: 128 characters.
- Maximum packed tokens: 2,048 per question, 8,192 total. Oversized input is rejected, not truncated.
- Reserved tokenizer symbols such as `[MASK]` are rejected.
- CUDA defaults to int8 weights with FP16 computation and accelerated FLA linear attention. Use `--quantization none`, `-Quantization none`, or `SYSTEMX_QUANTIZATION=none` for the BF16 reference. CPU defaults to unquantized FP32; int8 requires CUDA.
- Parallel workers: three int8 replicas by default on GPUs with at least 24 GB VRAM, one otherwise (including the RTX 3070). BF16 defaults to two on 24 GB GPUs. Use `--workers N`, `-Workers N`, or `SYSTEMX_WORKERS` (1-3) to override. Replicas improve concurrent throughput, not the latency of a single decision.
- Int8 measured approximately 5.3 GiB per worker on short prompts and 6.1 GiB near the token limit, including CUDA overhead, on a 4090. Reserve room for the desktop, games, and optional vision models. Actual 3070 latency and total application memory still require validation on that card; a 7 GB model budget does not guarantee room for a demanding game on an 8 GB GPU.
- Concurrent calls are served by free workers. When every worker is busy, a call receives HTTP 429 with `Retry-After: 1`; requests are never queued implicitly. `/v1/systemone/batch` waits up to 10 s per item for a free worker instead.
- With multiple free workers, `parallel: true` can split questions across replicas and adds `workers_used` to the response.
- Benchmark the configured checkpoint on the target hardware; measurements from other checkpoints do not apply.
- HTTP 401 means invalid/missing configured key; 413 means oversized body; 422 means invalid input/model/token budget; 503 means loading, failed initialization, or inference failure.

Clients should set a timeout, check readiness, handle errors, and retry busy responses with bounded backoff. If the application state changes while waiting, discard stale results and send the current state. Test decision quality on representative data; valid output alone does not establish accuracy.

## Model cache and background download

The default cache is `.cache/huggingface`, ignored by Git. The StartLux safetensors checkpoint is pinned in `model_config.py`; the upstream implementation is pinned in `requirements.txt`.

Check or prepare files without loading Torch or using the GPU:

```powershell
& ./.venv/Scripts/python.exe ./model_download.py --check-only
& ./.venv/Scripts/python.exe ./model_download.py
```

The worker writes JSON lines on stdout with stages `checking`, `missing`, `downloading`, `ready`, or `error`. Diagnostics go to stderr. Exit 0 means ready; check-only exits 2 for missing/incomplete files; exit 1 means failure. This is stage reporting, not a measured byte percentage. A client can run the worker in a separate process and use those events to update its UI without blocking.

Cancel with Ctrl+C or terminate the download worker. Partial downloads remain for retry. The cache check verifies required metadata and checkpoint files before reporting readiness.

To choose a different cache or start offline after download:

```powershell
$env:SYSTEMX_MODEL_CACHE = 'D:\Models\SystemX'
# Download into this cache before enabling offline mode:
& ./.venv/Scripts/python.exe ./model_download.py
$env:HF_HUB_OFFLINE = '1'
& ./start.ps1
```

Set cache variables before launching the Python process. `SYSTEMX_MODEL_CACHE` controls HF_HOME; existing `HF_HUB_CACHE` or `HUGGINGFACE_HUB_CACHE` variables can override the hub subdirectory. Clear those overrides when expecting the folder-local cache. A machine-level `HF_HOME` alone does not override this server's default.

## Network access

Loopback is the default. For access from another machine:

```powershell
$env:SYSTEMX_API_KEY = 'replace-with-a-private-random-key'
& ./start.ps1 -BindAddress 0.0.0.0 -Port 8000
```

Clients send `Authorization: Bearer <your-key>` with decision requests. Non-loopback binding requires a key. Health, model-list, and documentation endpoints remain unauthenticated. Use a trusted private network or a TLS reverse proxy with access controls. The scripts do not configure firewall rules, HTTPS, public hosting, or browser CORS.

## Optional application-owned lifecycle

Manual servers run until stopped. An application can instead start `server.py` as its child process, poll `/ready`, then stop only the process it owns. On Windows, set these in the child's environment:

| Variable | Purpose |
| --- | --- |
| `SYSTEMX_OWNER_PID` | PID of the owning application. The server binds a process handle before heavy imports and exits if that owner dies. Omit for a manually managed service. |
| `SYSTEMX_MANAGEMENT_TOKEN` | Random private token enabling `POST /internal/shutdown`. Send it as `X-SystemX-Management-Token`; only loopback clients are accepted. Keep separate from the inference API key. |
| `SYSTEMX_SHOW_CONSOLE` | Windows defaults to `1`: reuse a visible console or allocate one if launched without it. Set to `0` for background integrations that already launch Python hidden. This setting does not hide an existing terminal. |
| `SYSTEMX_DEVICE` | Default `cuda` or `cpu` for direct `server.py` invocation. Explicit CLI options take priority. |
| `SYSTEMX_VISION_DIR` | Folder of image models (`*.pt` TorchScript plus a JSON file each) loaded at start and on `/v1/vision/reload`; default `.cache/vision`. |

The shutdown endpoint returns 202 then terminates the process, even during loading. Stop sending requests first. Without the management token it returns 404. Owner monitoring and the optional console are Windows-only; standalone inference does not require them. Do not set owner variables globally for unrelated services.

An idle loaded server retains model memory. Stop it to release that memory. Each additional worker loads another full copy of the model. Loading may take several seconds; show an indeterminate activity indicator until readiness instead of inventing a percentage.

## Linux setup

Windows CUDA is the tested target. Linux commands for a Python 3.12 host:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-cuda.txt
.venv/bin/python model_download.py
.venv/bin/python server.py --device cuda
```

For CPU, use the PyTorch index `https://download.pytorch.org/whl/cpu` and `--device cpu`. Validate on the target platform before deployment.

## Verification and performance

Run the tests without loading model weights or allocating model VRAM:

```powershell
& ./.venv/Scripts/python.exe -m pytest -q
```

They cover request validation, readiness, authentication, concurrency, shutdown ownership, Windows owner termination, and the background download protocol. Windows console tests briefly open windows to verify default and explicit console behavior.

For an explicit real-model check, start the server, wait for readiness, then run:

```powershell
& ./.venv/Scripts/python.exe ./smoke_test.py --samples 5
```

This sends support-routing examples through all three primitives and measures warm single-Choice latency. It writes ignored `benchmark-results.json`. Actual choices are reported, not hardcoded or guaranteed. Stop the server afterwards if you do not need it resident.

Latency and memory measurements apply only to their recorded model configuration. Run the smoke test to measure the configured model on the target machine.

## Source map and provenance

- `server.py`: FastAPI schema, bounded requests, readiness, authentication, concurrency, optional shutdown.
- `vision.py`: image models kept in memory for addons (TorchScript, loaded from the vision folder).
- `backend.py`: pinned checkpoint loader, token limits, upstream model evaluation and answer mapping.
- `model_download.py`: CPU-only cache check/download with JSON status events.
- `owner_watchdog.py`: optional Windows owner monitoring and separate console.
- `setup.ps1`, `start.ps1`: local installation and foreground launch.
- `test_*.py`: tests without real inference; `smoke_test.py`: explicit real-model verification.

Upstream: [StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision). Check `model_config.py` for the pinned model revision and `requirements.txt` for the adapter source revision.

The adapter uses the upstream Instant evaluator with bidirectional attention and no generated reasoning. It preserves the bounded Choice/Noul/Score response contract.

Windows PyTorch wheels may lack the fused grouped-query attention kernel required
by upstream. On Windows the adapter allows SDPA's math fallback while preserving
bidirectional attention and its startup self-check. This can be slower and use
more memory than fused attention; each question is bounded to 2,048 tokens.

Model weights use CC BY-NC 4.0; commercial licensing is separate. See `THIRD_PARTY_NOTICES.md` and the upstream license terms.

## Optional addons

Application-specific clients live under `addons/`, with their own launchers,
dependencies, and instructions. They call the server's HTTP API and keep their
application logic separate from the core model service.

The [desktop visual player design](desktop-visual-player-design.md) describes
shared screen capture and mouse control. Its first Minesweeper implementation has
[setup instructions](../addons/minesweeper/DESKTOP.md) and a
[validation and VRAM report](minesweeper-validation.md).

### Minesweeper player

The [Minesweeper addon](../addons/minesweeper/README.md) reads the visible board,
applies Minesweeper rules, requests model decisions, and clicks cells through
Python Playwright and installed Google Chrome. The game itself is a separately
hosted webpage; its source is not included here.

Start the model server with the root `start.bat`, then launch the player with
`addons/minesweeper/play-minesweeper.bat`. Starting the server alone does not
open a game or browser, and stopping the player leaves the server running.
See the addon README for setup, controls, and verification details. Future
games belong in separate sibling addon folders using the same server API.

### Red Alert 2 player (Chrono Divide)

The [Chrono Divide addon](../addons/chronodivide/README.md) plays Red Alert 2 from the visible
screen with mouse and keyboard input, asking System One for strategy. Like the Minesweeper desktop
player it picks the game's window, finds the game view, and recognizes what it sees with taught
profiles, so it is not tied to one client. Its unit detector is one of the server's image models.

The [System 2 strategic supervisor prototype](system-two-strategic-supervisor-design.md)
proposes an occasional local LLM above vision (System 0) and a replaceable
System One decision model (System 1), with a qualified adapter
boundary for backends such as Jev. It covers independent model modes, bounded
steering, and an evaluation plan. It is
design only; no System 2 implementation or model execution is included.
