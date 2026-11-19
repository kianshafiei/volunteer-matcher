"""FastAPI service: POST a volunteer's description, get ranked opportunities with reasons."""
from dataclasses import asdict
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import load
from .matcher import Matcher

app = FastAPI(title="Volunteer matcher", version="0.1.0")


class Query(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000, description="What the volunteer wants, in their own words")
    k: int = Field(5, ge=1, le=20)
    method: str = Field("full", pattern="^(bm25|dense|hybrid|full)$")


@lru_cache(maxsize=1)
def matcher() -> Matcher:
    return Matcher(load())


@app.get("/health")
def health():
    return {"status": "ok", "opportunities": len(matcher().df)}


@app.post("/match")
def match(q: Query):
    try:
        res = matcher().search(q.text, q.k, q.method)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    # Only the request text is used; it is not stored or logged.
    return {"query": q.text, "method": q.method, "results": [asdict(r) for r in res]}
