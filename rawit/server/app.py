# Copyright 2026 Wisu Suntoyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Server architecture derived from Laya (Apache-2.0):
# https://github.com/convaiinnovations/laya

"""FastAPI application exposing POST /v1/systemone.

Start with:
    rawit-serve
    # or
    uvicorn rawit.server.app:app --host 0.0.0.0 --port 8080

Environment variables:
    RAWIT_MODEL      HuggingFace model id or local path (default: bigwisu/rawit-300m)
    RAWIT_DEVICE     cuda | mps | cpu  (auto-detected when unset)
    RAWIT_MAX_LEN    Maximum sequence length (default: 1024)
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from ..pipeline import RawitPipeline
from .schemas import SystemOneRequest, SystemOneResponse

_log = logging.getLogger("rawit.serve")

_DEFAULT_MODEL = "bigwisu/rawit-300m"
_pool = ThreadPoolExecutor(max_workers=1)
_pipeline: Optional[RawitPipeline] = None
_gate: Optional[asyncio.Lock] = None


def _get_pipeline() -> RawitPipeline:
    global _pipeline
    if _pipeline is None:
        model_id = os.environ.get("RAWIT_MODEL", _DEFAULT_MODEL)
        device = os.environ.get("RAWIT_DEVICE", None)
        max_len = int(os.environ.get("RAWIT_MAX_LEN", "1024"))
        _log.info("Loading Rawit model: %s  device=%s", model_id, device or "auto")
        _pipeline = RawitPipeline.from_pretrained(model_id, device=device, max_len=max_len)
        _log.info("Model loaded.")
    return _pipeline


app = FastAPI(title="Rawit System 1 API", version="0.1.0")


@app.on_event("startup")
async def _startup():
    global _gate
    _gate = asyncio.Lock()
    # Eagerly load the model at startup so the first request isn't slow.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_pool, _get_pipeline)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/systemone", response_model=SystemOneResponse)
async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
    global _gate
    if _gate is None:
        _gate = asyncio.Lock()

    body = await request.json()
    try:
        req = SystemOneRequest(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    options = [{"id": o.id, "label": o.label} for o in req.options]

    try:
        async with _gate:
            loop = asyncio.get_running_loop()
            pipe = _get_pipeline()
            kw = {}
            if req.max_len:
                kw["max_len"] = req.max_len

            def _run():
                return pipe.decide(
                    context=req.context,
                    rubric=req.question,
                    options=options,
                    qtype=req.type,
                    escalate_threshold=req.escalate_threshold,
                )

            result = await loop.run_in_executor(_pool, _run)

        return JSONResponse(
            content={
                "decision": result.decision,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "escalate": result.escalate,
                "escalation_score": result.escalation_score,
                "execution_time_ms": result.execution_time_ms,
                "model": req.model or result.model,
            },
            headers={"X-Inference-Time-Ms": str(result.execution_time_ms)},
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        _log.exception("inference failed")
        raise HTTPException(status_code=500, detail="inference failed")


def main():
    import uvicorn
    port = int(os.environ.get("RAWIT_PORT", "8080"))
    uvicorn.run("rawit.server.app:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
