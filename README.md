# SystemOne server

A standalone HTTP service for the local [Von OptionMarker model](https://github.com/wfzyx/von). Give it a state and typed questions; it returns choices, yes/no probabilities, or ordinal scores. Use it for routing, classification, ranking a bounded set of actions, or application decisions. It does not generate chat responses or execute actions.

Everything runs locally after dependencies and model files are installed. This is Von, not the hosted Jev service. The server needs no game, browser, Codex, or ChatGPT plugin. Any application can call its HTTP API.

## Quick start on Windows

Requirements: Python 3.12, Git, and an NVIDIA driver compatible with CUDA 12.6 for GPU use. CPU mode is available. Allow several GB for dependencies plus approximately 1.58 GB for the checkpoint.

On this prepared machine, **double-click `start.bat`**. It opens a visible console titled `SystemOne Von Server`, shows loading and request logs, and keeps startup errors visible. Stop the server with Ctrl+C or close its console window. The launcher works regardless of the current directory.

From a terminal, the batch launcher also accepts the PowerShell options: `start.bat -Device cpu -Port 8001`.

For a fresh checkout, **double-click `setup.bat` first**, then `start.bat`.
Setup keeps its window open to show the result. From a terminal:

```powershell
Set-Location E:\Workspace\Python\SystemOne
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

CUDA errors are reported instead of silently falling back to CPU. `start.ps1` defaults to CUDA; pass `-Device cpu` explicitly when using it. The direct Python entry point also accepts `--device cpu` and defaults to `VON_DEVICE` or CUDA.

## Make a decision

Send JSON to `POST /v1/systemone`. Here is a complete PowerShell request:

```powershell
$payload = @{
    model = 'von-option-marker'
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
    "model": "von-option-marker",
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
| `GET /health` | Status JSON: `ready`, `status`, `model`, `revision`, `device`; includes `load_seconds` after successful initialization. |
| `GET /ready` | Same status; HTTP 200 when ready, otherwise 503. |
| `GET /v1/models` | Lists the pinned model. |
| `POST /v1/systemone` | Evaluates the supplied questions. |
| `GET /docs` | Interactive API documentation. |
| `GET /openapi.json` | Machine-readable request schema. |

The decision response contains `model`, `answers`, `usage`, and `latency_ms`. Each answer uses the corresponding question ID:

| Question type | Input criteria | Answer fields |
| --- | --- | --- |
| `choice` | Object with 2-16 candidate keys and string descriptions (null uses the key). | `type`, `choice`, `probabilities` keyed by candidate, `confidence`. |
| `noul` | Optional object with `true` and/or `false` descriptions. | `type`, `noul`: probability of yes from 0 to 1. |
| `score` | Ordered list of 2-10 string descriptions, from lowest to highest. | `type`, `score`: expected level index from 0 to N-1; `legend`, `probabilities`, `confidence`. |

`confidence` is the upstream top-two probability gap, not a calibrated probability that an application decision is correct. `usage.input_tokens` counts packed question tokens; it excludes Noul's extra context-free debiasing pass when explicit criteria are absent. `output_tokens` counts answers, not generated text tokens. `latency_ms` is server evaluation time, excluding transport and initial model loading.

Model names `von-option-marker`, `von-latest`, and `von-1.0.0` all use the same pinned checkpoint. Jev model names are rejected. The API is a limited state/questions/answers interface, not a full Jev or OpenAI API implementation; instructions and criteria descriptions must be strings.

## Limits and client behavior

- Maximum request body: 64 KiB; 1-8 questions per request.
- Maximum instruction/criterion length: 2,000 characters; question and candidate keys: 128 characters.
- Maximum packed tokens: 2,048 per question, 8,192 total. Oversized input is rejected, not truncated.
- Reserved tokenizer symbols such as `[MASK]` are rejected.
- One active inference at a time. Additional concurrent calls receive HTTP 429 with `Retry-After: 1`.
- Questions within a request run serially, not as one shared forward pass. Noul without explicit criteria adds another pass.
- HTTP 401 means invalid/missing configured key; 413 means oversized body; 422 means invalid input/model/token budget; 503 means loading, failed initialization, or inference failure.

Clients should set a timeout, check readiness, handle errors, and retry busy responses with bounded backoff. If the application state changes while waiting, discard stale results and send the current state. Test decision quality on representative data; valid output alone does not establish accuracy.

## Model cache and background download

The default cache is `.cache/huggingface` under this folder. It is ignored by Git. The model checkpoint is `option_marker.pt`, not a safetensors file. The copied snapshot is approximately 1.58 GB and uses revision `aa2fdc9630ecdadef32c56073553b3a69bed38bf`.

Check or prepare files without loading Torch or using the GPU:

```powershell
& ./.venv/Scripts/python.exe ./model_download.py --check-only
& ./.venv/Scripts/python.exe ./model_download.py
```

The worker writes JSON lines on stdout with stages `checking`, `missing`, `downloading`, `ready`, or `error`. Diagnostics go to stderr. Exit 0 means ready; check-only exits 2 for missing/incomplete files; exit 1 means failure. This is stage reporting, not a measured byte percentage. A client can run the worker in a separate process and use those events to update its UI without blocking.

Cancel with Ctrl+C or terminate the worker process your application started. Partial downloads remain for retry. Readiness requires all pinned files, valid JSON metadata, and the full 1,581,316,050-byte checkpoint. The quick cache check validates size and metadata, not a cryptographic checksum.

To choose a different cache or start offline after download:

```powershell
$env:VON_MODEL_CACHE = 'D:\Models\SystemOne'
# Download into this cache before enabling offline mode:
& ./.venv/Scripts/python.exe ./model_download.py
$env:HF_HUB_OFFLINE = '1'
& ./start.ps1
```

Set cache variables before launching the Python process. `VON_MODEL_CACHE` controls HF_HOME; existing `HF_HUB_CACHE` or `HUGGINGFACE_HUB_CACHE` variables can override the hub subdirectory. Clear those overrides when expecting the folder-local cache. A machine-level `HF_HOME` alone does not override this server's default.

## Network access

Loopback is the default. For access from another machine:

```powershell
$env:VON_API_KEY = 'replace-with-a-private-random-key'
& ./start.ps1 -BindAddress 0.0.0.0 -Port 8000
```

Clients send `Authorization: Bearer <your-key>` with decision requests. Non-loopback binding requires a key. Health, model-list, and documentation endpoints remain unauthenticated. Use a trusted private network or a TLS reverse proxy with access controls. The scripts do not configure firewall rules, HTTPS, public hosting, or browser CORS.

## Optional application-owned lifecycle

Manual servers run until stopped. An application can instead start `server.py` as its child process, poll `/ready`, then stop only the process it owns. On Windows, set these in the child's environment:

| Variable | Purpose |
| --- | --- |
| `VON_OWNER_PID` | PID of the owning application. The server binds a process handle before heavy imports and exits if that owner dies. Omit for a manually managed service. |
| `VON_MANAGEMENT_TOKEN` | Random private token enabling `POST /internal/shutdown`. Send it as `X-Von-Management-Token`; only loopback clients are accepted. Keep separate from the inference API key. |
| `VON_SHOW_CONSOLE` | Windows defaults to `1`: reuse a visible console or allocate one if launched without it. Set to `0` for background integrations that already launch Python hidden. This setting does not hide an existing terminal. |
| `VON_DEVICE` | Default `cuda` or `cpu` for direct `server.py` invocation. Explicit CLI options take priority. |

The shutdown endpoint returns 202 then terminates the process, even during loading. Stop sending requests first. Without the management token it returns 404. Owner monitoring and the optional console are Windows-only; standalone inference does not require them. Do not set owner variables globally for unrelated services.

An idle loaded server retains model memory. Stop it to release that memory. Use one worker: additional processes load additional copies of the model. Loading may take several seconds; show an indeterminate activity indicator until readiness instead of inventing a percentage.

## Linux setup

Windows CUDA is the tested target. Linux commands for a Python 3.12 host:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
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

Prior measurements of this adapter on an RTX 3070 8 GB, Windows, Python 3.12.8, PyTorch 2.9.0+cu126, and Transformers 4.57.3: cached load/warmup about 10.9 seconds; five warm single-Choice calls had a 44.51 ms median; CUDA warmup memory was 763.1 MiB allocated / 852 MiB reserved, with roughly 1.2 GiB additional whole-GPU usage including CUDA overhead. These are earlier short-prompt measurements, not a new benchmark of this extracted copy. Input size, hardware, and competing workloads affect results.

## Source map and provenance

- `server.py`: FastAPI schema, bounded requests, readiness, authentication, concurrency, optional shutdown.
- `backend.py`: pinned checkpoint loader, token limits, upstream model evaluation and answer mapping.
- `model_download.py`: CPU-only cache check/download with JSON status events.
- `owner_watchdog.py`: optional Windows owner monitoring and separate console.
- `setup.ps1`, `start.ps1`: local installation and foreground launch.
- `test_*.py`: tests without real inference; `smoke_test.py`: explicit real-model verification.

Upstream SDK commit: [`2656a69be2ef0ebf2f3058302e52791f99cbd8de`](https://github.com/wfzyx/von/tree/2656a69be2ef0ebf2f3058302e52791f99cbd8de). Model snapshot: [`wfzyx/von-1.0@aa2fdc9630ecdadef32c56073553b3a69bed38bf`](https://huggingface.co/wfzyx/von-1.0/tree/aa2fdc9630ecdadef32c56073553b3a69bed38bf). Pinned checkpoint SHA256: `52202f176080d1efe38ca62c45264894fc7e444d332f4af6bac86f4e3146e0f2`.

The adapter uses the upstream architecture, packing, scorer, and Choice/Noul/Score inference. It loads the full checkpoint strictly with `weights_only=True`, uses SDPA attention, and disables optional ModernBERT compilation for Windows compatibility. GPU inference uses BF16 where supported, otherwise FP16; CPU uses FP32. There is no randomly initialized fallback head or hosted inference fallback.

Upstream code and model are Apache-2.0 licensed. Preserve the upstream license and attribution when redistributing them; dependency packages retain their own licenses. See `THIRD_PARTY_NOTICES.md` for bundled attribution. The wrapper is a separately maintained extraction; upstream's license does not automatically determine the wrapper's distribution license.

## Optional addons

Application-specific clients live under `addons/`, with their own launchers,
dependencies, and instructions. They call the server's HTTP API and keep their
application logic separate from the core model service.

### Minesweeper player

The [Minesweeper addon](addons/minesweeper/README.md) reads the visible board,
applies Minesweeper rules, requests model decisions, and clicks cells through
Python Playwright and installed Google Chrome. The game itself is a separately
hosted webpage; its source is not included here.

Start the model server with the root `start.bat`, then launch the player with
`addons/minesweeper/play-minesweeper.bat`. Starting the server alone does not
open a game or browser, and stopping the player leaves the server running.
See the addon README for setup, controls, and verification details. Future
games belong in separate sibling addon folders using the same server API.
