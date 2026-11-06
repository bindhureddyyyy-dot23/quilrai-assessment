"""Two mock OpenAI-compatible providers whose behaviour can be flipped at runtime for demos.

    python mock_providers.py            # primary on :9301, secondary on :9302

    curl -X POST localhost:9301/__behavior -d '{"mode":"429"}'   # ok | 429 | slow | 500 | 400
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def make_provider(name: str, mode: str = "ok", slow_seconds: float = 5.0) -> FastAPI:
    app = FastAPI(title=f"mock-{name}")
    app.state.mode = mode
    app.state.hits = 0

    @app.post("/__behavior")
    async def set_behavior(request: Request):
        app.state.mode = (await request.json())["mode"]
        return {"provider": name, "mode": app.state.mode}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        app.state.hits += 1
        body = await request.json()
        mode = app.state.mode
        if mode == "429":
            return JSONResponse({"error": {"message": "rate limit"}}, status_code=429, headers={"retry-after": "7"})
        if mode == "500":
            return JSONResponse(
                {"error": "Traceback (most recent call last): File /srv/internal/llm.py line 42 KeyError: 'db-password'"},
                status_code=500,
            )
        if mode == "400":
            return JSONResponse({"error": {"message": "context_length_exceeded on node gpu-17.internal"}}, status_code=400)
        if mode == "slow":
            await asyncio.sleep(slow_seconds)
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in body["messages"]) // 4 + 5
        completion = f"Hello from the {name} provider ({body.get('model')})."
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": completion}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 12, "total_tokens": prompt_tokens + 12},
        }

    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    def serve(app, port):
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    primary = make_provider("primary", os.getenv("PRIMARY_MODE", "ok"))
    secondary = make_provider("secondary", os.getenv("SECONDARY_MODE", "ok"))
    threading.Thread(target=serve, args=(secondary, 9302), daemon=True).start()
    print("primary :9301  secondary :9302  (POST /__behavior to change mode)")
    serve(primary, 9301)
