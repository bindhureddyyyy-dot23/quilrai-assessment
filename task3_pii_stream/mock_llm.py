"""Mock OpenAI-compatible LLM provider that streams PII in deliberately awkward fragments.

The reply is the last user message (echo mode) or, if that is empty, a canned answer that
contains an email, an SSN and a card number. Text is cut into random 1-6 character deltas
so PII is split across SSE events - the worst case for a streaming guardrail.

Env: MOCK_CHUNK_DELAY_MS (default 15), MOCK_LLM_PORT (default 9200)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

CANNED = (
    "Sure! I found the account. The customer email is jane.doe+billing@example.com, "
    "SSN on file is 123-45-6789 and the card ending is part of 4111 1111 1111 1111. "
    "Order number 1234567890123 shipped yesterday. Let me know if you need anything else."
)

app = FastAPI(title="Mock LLM provider")


def _pieces(text: str, seed: int | None = None) -> list[str]:
    rnd = random.Random(seed)
    out, i = [], 0
    while i < len(text):
        n = rnd.randint(1, 6)
        out.append(text[i : i + n])
        i += n
    return out


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    last_user = next((m.get("content") for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
    text = last_user if isinstance(last_user, str) and last_user.strip() and last_user != "canned" else CANNED
    model = body.get("model", "mock-1")
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
    delay = float(os.getenv("MOCK_CHUNK_DELAY_MS", "15")) / 1000

    if not body.get("stream"):
        return JSONResponse(
            {
                "id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
        )

    async def gen():
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
        for piece in _pieces(text, body.get("seed")):
            if delay:
                await asyncio.sleep(delay)
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("MOCK_LLM_PORT", "9200")))
