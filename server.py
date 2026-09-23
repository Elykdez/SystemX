"""SystemX: standalone, bounded HTTP adapter for the current System One decision model."""

from owner_watchdog import configure_console, start_owner_watchdog

if __name__ == "__main__":
    start_owner_watchdog()
    configure_console()

import argparse
import asyncio
from contextlib import asynccontextmanager
import hmac
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from backend import MODEL_NAME, MODEL_REVISION, ModelBackend, configure_cache
from model_config import MODEL_DESCRIPTION, MODEL_OWNER
from vision import VisionModels
from workers import ProcessBackend, WorkerSlots, default_workers, merge_responses

LOG = logging.getLogger("systemx-server")
MAX_BODY = 65536
VISION_BODY = 16 * 2 ** 20            # one raw image for /v1/vision/detect
ShortText = Annotated[str, Field(min_length=1, max_length=2000)]
Key = Annotated[str, Field(min_length=1, max_length=128)]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice", "noul", "score"]
    instructions: ShortText
    criteria: dict[Key, ShortText | None] | list[ShortText] | None = None

    @model_validator(mode="after")
    def validate_criteria(self):
        if self.type == "choice" and not (isinstance(self.criteria, dict) and 2 <= len(self.criteria) <= 16):
            raise ValueError("Choice needs 2-16 named criteria.")
        if self.type == "score" and not (isinstance(self.criteria, list) and 2 <= len(self.criteria) <= 10):
            raise ValueError("Score needs 2-10 ordered string criteria.")
        if self.type == "noul" and self.criteria is not None:
            if not isinstance(self.criteria, dict) or set(self.criteria) - {"true", "false"}:
                raise ValueError("Noul criteria can only contain true and false.")
        return self


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal[MODEL_NAME] = MODEL_NAME
    state: JsonValue
    questions: dict[Key, Question] = Field(min_length=1, max_length=8)
    # Spread several questions across free workers; answers are identical to serial evaluation.
    parallel: bool = True


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: list[DecisionRequest] = Field(min_length=1, max_length=8)


def create_app(backend=None, api_key=None, management_token=None, shutdown_callback=None, workers=None,
               vision=None):
    """`backend` may be one engine or a list of interchangeable replicas (tests, custom hosts).
    `vision` holds the image models kept in memory (vision.py); the real server always has one."""
    if backend is None:
        device = os.environ.get("SYSTEMX_DEVICE", "cuda")
        count = workers or default_workers(device)
        engines = [ModelBackend(device)] + [ProcessBackend(device) for _ in range(count - 1)]
        vision = vision or VisionModels(device=device)
    else:
        engines = list(backend) if isinstance(backend, (list, tuple)) else [backend]
    key = api_key if api_key is not None else os.environ.get("SYSTEMX_API_KEY", "")
    slots = WorkerSlots(engines)
    pool = ThreadPoolExecutor(max_workers=max(2, 2 * len(engines)), thread_name_prefix="systemone-dispatch")
    status = {"ready": False, "status": "loading", "model": MODEL_NAME,
              "revision": MODEL_REVISION, "device": engines[0].device,
              "quantization": getattr(engines[0], "quantization", "none")}

    async def load(index):
        try:
            await asyncio.to_thread(engines[index].load)
            slots.mark(index, "ready")
            return True
        except Exception:
            slots.mark(index, "failed")
            LOG.exception("System One worker %d failed to load", index)
            return False

    async def load_vision():
        try:
            names = await asyncio.to_thread(vision.load)
            status["vision"] = names
        except Exception:
            LOG.exception("Vision models failed to load")

    async def initialize():
        started = time.perf_counter()
        # The primary loads first so a missing model is downloaded once, then replicas load together.
        # Image models load after it, into the same process and CUDA context.
        if not await load(0):
            status.update(ready=False, status="failed", error="Model initialization failed; inspect server logs.")
            if vision is not None:
                await load_vision()
            return
        if vision is not None:
            await load_vision()
        status.update(ready=True, status="ready", load_seconds=round(time.perf_counter() - started, 2))
        LOG.info("System One ready: %s", status)
        if len(engines) > 1:
            await asyncio.gather(*(load(i) for i in range(1, len(engines))))
            status["workers_seconds"] = round(time.perf_counter() - started, 2)
            LOG.info("System One workers: %s", slots.status())

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(initialize())
        yield
        await task
        pool.shutdown(wait=False, cancel_futures=True)
        slots.close()

    app = FastAPI(title="SystemX Server", version="1.1.0", lifespan=lifespan)

    @app.post("/internal/shutdown", include_in_schema=False)
    def shutdown(request: Request, x_systemx_management_token: Annotated[str | None, Header()] = None):
        if not management_token or shutdown_callback is None:
            raise HTTPException(404, "Management is disabled for this server.")
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "Management is loopback-only.")
        if not hmac.compare_digest((x_systemx_management_token or "").encode(), management_token.encode()):
            raise HTTPException(401, "Invalid management token.")
        return JSONResponse({"status": "stopping"}, status_code=202, background=BackgroundTask(shutdown_callback))

    @app.middleware("http")
    async def limit_request(request: Request, call_next):
        if request.method == "POST":
            limit = VISION_BODY if request.url.path == "/v1/vision/detect" else MAX_BODY
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > limit:
                    return JSONResponse(status_code=413, content={"detail": f"Request exceeds {limit // 1024} KiB."})
            request._body = bytes(body)
        return await call_next(request)

    def current_status():
        return dict(status, workers=slots.status())

    @app.get("/health")
    def health():
        return current_status()

    @app.get("/ready")
    def ready():
        return JSONResponse(status_code=200 if status["ready"] else 503, content=current_status())

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": MODEL_NAME, "description": MODEL_DESCRIPTION}],
                "object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": MODEL_OWNER}]}

    def check_key(authorization):
        if key and not hmac.compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
            raise HTTPException(401, "Missing or invalid Bearer token.")

    def authorize(authorization):
        check_key(authorization)
        if not status["ready"]:
            raise HTTPException(503, "System One is not ready. Check /health and server logs.")

    def image_models(authorization):
        check_key(authorization)
        if vision is None:
            raise HTTPException(404, "This server keeps no image models.")
        return vision

    @app.get("/v1/vision/models")
    def vision_models(authorization: Annotated[str | None, Header()] = None):
        """Image models in memory; `folder` is where a local addon puts its models for a reload."""
        models = image_models(authorization)
        return {"folder": str(models.folder), "device": models.device, "models": models.describe(),
                "error": models.error}

    @app.post("/v1/vision/reload")
    def vision_reload(authorization: Annotated[str | None, Header()] = None):
        """Load the folder's models again (after an addon copied a new one there)."""
        models = image_models(authorization)
        models.load()
        return {"folder": str(models.folder), "models": models.describe(), "error": models.error}

    @app.post("/v1/vision/detect")
    async def vision_detect(request: Request, model: str, width: int, height: int, min_score: float | None = None,
                            authorization: Annotated[str | None, Header()] = None):
        """One raw uint8 image (height x width x 3 bytes) through a model; its rows back."""
        models = image_models(authorization)
        try:
            rows, ms = await asyncio.to_thread(models.run, model, await request.body(), width, height, min_score)
        except KeyError:
            raise HTTPException(404, f"No image model {model!r} is loaded.")
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {"model": model, "rows": rows, "ms": round(ms, 2)}

    def run(indices, request):
        """Evaluate on already acquired workers, splitting questions across several; releases them."""
        started = time.perf_counter()
        try:
            questions = {qid: q.model_dump(exclude_none=True) for qid, q in request.questions.items()}
            if len(indices) == 1:
                response = engines[indices[0]].evaluate(request.state, questions)
            else:
                ids = list(questions)
                shares = [ids[i::len(indices)] for i in range(len(indices))]
                futures = [pool.submit(engines[index].evaluate, request.state, {q: questions[q] for q in share})
                           for index, share in zip(indices, shares)]
                response = merge_responses(questions, [future.result() for future in futures])
                response["workers_used"] = len(indices)
            response["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return response
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            LOG.exception("System One inference failed")
            raise HTTPException(503, "Model inference failed; inspect server logs.") from error
        finally:
            for index in indices:
                slots.release(index)

    @app.post("/v1/systemone")
    def decide(request: DecisionRequest, authorization: Annotated[str | None, Header()] = None):
        authorize(authorization)
        wanted = min(len(request.questions), len(engines)) if request.parallel else 1
        indices = slots.acquire(wanted)
        if not indices:
            raise HTTPException(429, "All System One workers are busy; retry with fresh state.", headers={"Retry-After": "1"})
        return run(indices, request)

    @app.post("/v1/systemone/batch")
    def batch(request: BatchRequest, authorization: Annotated[str | None, Header()] = None):
        """Independent decisions evaluated concurrently across workers; results keep request order."""
        authorize(authorization)
        started = time.perf_counter()

        def one(item):
            indices = slots.acquire(1, timeout=10)
            if not indices:
                return {"error": {"status": 429, "detail": "No System One worker became free in time."}}
            try:
                return run(indices, item)
            except HTTPException as error:
                return {"error": {"status": error.status_code, "detail": error.detail}}

        results = list(pool.map(one, request.requests))
        return {"results": results, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=os.environ.get("SYSTEMX_DEVICE", "cuda"))
    parser.add_argument("--quantization", choices=["none", "int8"], default=os.environ.get("SYSTEMX_QUANTIZATION"),
                        help="Default: int8 on CUDA, none on CPU.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel model replicas (1-3; default: 3 int8 / 2 BF16 on 24 GB GPUs, 1 otherwise).")
    args = parser.parse_args()
    if args.workers is not None and not 1 <= args.workers <= 3:
        parser.error("--workers must be 1-3.")
    if args.quantization is not None:
        os.environ["SYSTEMX_QUANTIZATION"] = args.quantization
    if args.device == "cpu" and args.quantization == "int8":
        parser.error("Int8 inference requires CUDA.")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("SYSTEMX_API_KEY"):
        parser.error("Set SYSTEMX_API_KEY before binding a network interface.")
    configure_cache()
    logging.basicConfig(level=logging.INFO)
    import uvicorn
    management_token = os.environ.get("SYSTEMX_MANAGEMENT_TOKEN")

    def owned_shutdown():
        # The owning application has cancelled its calls. Exit even if a model download
        # or initialization thread is still blocked, releasing the CUDA context.
        timer = threading.Timer(0.2, lambda: os._exit(0))
        timer.daemon = True
        timer.start()

    count = args.workers or default_workers(args.device)
    engines = [ModelBackend(args.device)] + [ProcessBackend(args.device) for _ in range(count - 1)]
    LOG.info("Starting %d System One worker(s) on %s", count, args.device)
    uvicorn.run(create_app(engines, management_token=management_token, shutdown_callback=owned_shutdown,
                           vision=VisionModels(device=args.device)), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
