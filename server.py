"""Standalone, bounded HTTP adapter for the real Von decision model."""

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
from typing import Annotated, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from backend import MODEL_NAME, MODEL_REVISION, VonBackend, configure_cache

LOG = logging.getLogger("von-server")
MAX_BODY = 65536
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
    model: Literal["von-option-marker", "von-latest", "von-1.0.0"] = MODEL_NAME
    state: JsonValue
    questions: dict[Key, Question] = Field(min_length=1, max_length=8)


def create_app(backend=None, api_key=None, management_token=None, shutdown_callback=None):
    engine = backend if backend is not None else VonBackend(os.environ.get("VON_DEVICE", "cuda"))
    key = api_key if api_key is not None else os.environ.get("VON_API_KEY", "")
    gate = threading.Lock()
    status = {"ready": False, "status": "loading", "model": MODEL_NAME,
              "revision": MODEL_REVISION, "device": engine.device}

    async def initialize():
        started = time.perf_counter()
        try:
            await asyncio.to_thread(engine.load)
            status.update(ready=True, status="ready", load_seconds=round(time.perf_counter() - started, 2))
            LOG.info("Von ready: %s", status)
        except Exception:
            status.update(ready=False, status="failed", error="Model initialization failed; inspect server logs.")
            LOG.exception("Model initialization failed")

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(initialize())
        yield
        await task

    app = FastAPI(title="SystemOne Von Server", version="1.0.0", lifespan=lifespan)

    @app.post("/internal/shutdown", include_in_schema=False)
    def shutdown(request: Request, x_von_management_token: Annotated[str | None, Header()] = None):
        if not management_token or shutdown_callback is None:
            raise HTTPException(404, "Management is disabled for this server.")
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "Management is loopback-only.")
        if not hmac.compare_digest((x_von_management_token or "").encode(), management_token.encode()):
            raise HTTPException(401, "Invalid management token.")
        return JSONResponse({"status": "stopping"}, status_code=202, background=BackgroundTask(shutdown_callback))

    @app.middleware("http")
    async def limit_request(request: Request, call_next):
        if request.method == "POST":
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_BODY:
                    return JSONResponse(status_code=413, content={"detail": "Request exceeds 64 KiB."})
            request._body = bytes(body)
        return await call_next(request)

    @app.get("/health")
    def health():
        return dict(status)

    @app.get("/ready")
    def ready():
        return JSONResponse(status_code=200 if status["ready"] else 503, content=dict(status))

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": MODEL_NAME, "description": "Pinned local Von OptionMarker"}],
                "object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "wfzyx"}]}

    @app.post("/v1/systemone")
    def decide(request: DecisionRequest, authorization: Annotated[str | None, Header()] = None):
        if key and not hmac.compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
            raise HTTPException(401, "Missing or invalid Bearer token.")
        if not status["ready"]:
            raise HTTPException(503, "Von is not ready. Check /health and server logs.")
        if not gate.acquire(blocking=False):
            raise HTTPException(429, "Von is busy; retry with fresh state.", headers={"Retry-After": "1"})
        started = time.perf_counter()
        try:
            questions = {qid: q.model_dump(exclude_none=True) for qid, q in request.questions.items()}
            response = engine.evaluate(request.state, questions)
            response["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return response
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            LOG.exception("Von inference failed")
            raise HTTPException(503, "Model inference failed; inspect server logs.") from error
        finally:
            gate.release()

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=os.environ.get("VON_DEVICE", "cuda"))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("VON_API_KEY"):
        parser.error("Set VON_API_KEY before binding a network interface.")
    configure_cache()
    logging.basicConfig(level=logging.INFO)
    import uvicorn
    management_token = os.environ.get("VON_MANAGEMENT_TOKEN")

    def owned_shutdown():
        # The owning application has cancelled its calls. Exit even if a model download
        # or initialization thread is still blocked, releasing the CUDA context.
        timer = threading.Timer(0.2, lambda: os._exit(0))
        timer.daemon = True
        timer.start()

    uvicorn.run(create_app(VonBackend(args.device), management_token=management_token,
                          shutdown_callback=owned_shutdown), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
