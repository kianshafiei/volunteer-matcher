"""Hybrid volunteer-opportunity matcher.

Pipeline: keyword retrieval (BM25) + semantic retrieval (MiniLM embeddings), fused with
reciprocal rank fusion, then a light preference step (one-off vs ongoing) and an optional
diversity step (MMR) so one organisation cannot fill the whole list. Every result comes with a
short explanation: the sentence in the listing closest to the volunteer's request.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .embed import Embedder

TOKEN = re.compile(r"[a-z0-9]+")
ONE_OFF = re.compile(r"\b(one[- ]off|one time|one[- ]day|single day|this (weekend|saturday|sunday)|just once)\b", re.I)
HARD = re.compile(r"\b(only|can't commit|cannot commit|not an? (ongoing|regular))\b", re.I)
REGULAR = re.compile(r"\b(every week|weekly|regular(ly)?|ongoing|long[- ]term|each week)\b", re.I)


def tokenize(s: str) -> list[str]:
    return [t for t in TOKEN.findall(s.lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 1]


def split_sentences(s: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if len(p.strip()) > 15]
    return parts or [s]


@dataclass
class Match:
    doc_id: str
    title: str
    org: str
    category: str
    recurrence: str
    score: float
    why: str


class Matcher:
    def __init__(self, df, embedder: Embedder | None = None, rrf_k: int = 60):
        self.df = df.reset_index(drop=True)
        self.emb = embedder or Embedder()
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenize(t) for t in self.df.text])
        self.vecs = self.emb.encode(self.df.text.tolist())
        self.orgs = np.asarray(self.df.org_title.astype(str).tolist(), dtype=object)
        self.recur = np.asarray(self.df.recurrence_type.astype(str).tolist(), dtype=object)

    # ---------- individual retrievers: return document indices, best first ----------
    def rank_bm25(self, q: str) -> np.ndarray:
        return np.argsort(-self.bm25.get_scores(tokenize(q)), kind="stable")

    def rank_dense(self, q: str, qv: np.ndarray | None = None) -> np.ndarray:
        qv = self.emb.encode([q])[0] if qv is None else qv
        return np.argsort(-(self.vecs @ qv), kind="stable")

    def fused_scores(self, q: str, qv: np.ndarray | None = None) -> np.ndarray:
        s = np.zeros(len(self.df))
        for order in (self.rank_bm25(q), self.rank_dense(q, qv)):
            s[order] += 1.0 / (self.rrf_k + np.arange(1, len(order) + 1))
        return s

    @staticmethod
    def commitment(q: str) -> str | None:
        if ONE_OFF.search(q):
            return "onetime"
        if REGULAR.search(q):
            return "ongoing"
        return None

    # ---------- full pipeline ----------
    def search(self, q: str, k: int = 10, method: str = "full", mmr_lambda: float = 0.8,
               max_per_org: int = 3) -> list[Match]:
        if not q or not q.strip():
            raise ValueError("empty query")
        qv = self.emb.encode([q])[0]
        if method == "bm25":
            order = self.rank_bm25(q)[:k]
        elif method == "dense":
            order = self.rank_dense(q, qv)[:k]
        else:
            s = self.fused_scores(q, qv)
            if method == "full":
                want = self.commitment(q)
                if want:
                    # a stated constraint ("I can only...") filters; a mild preference only nudges
                    factor = 0.0 if HARD.search(q) else 1.0 / 1.25
                    s = s * np.where(self.recur == want, 1.0, factor)
            order = np.argsort(-s, kind="stable")
            if method == "full":
                order = order[s[order] > 0]
                order = self._mmr(order[:80], s, k, mmr_lambda, max_per_org)
            else:
                order = order[:k]
        return [self._to_match(i, q, qv, r) for r, i in enumerate(order)]

    def _mmr(self, cand: np.ndarray, s: np.ndarray, k: int, lam: float, max_per_org: int) -> list[int]:
        """Maximal marginal relevance: trade relevance against similarity to what is already picked."""
        rel = s[cand] / s[cand].max()
        chosen: list[int] = []
        pool = list(range(len(cand)))
        while pool and len(chosen) < k:
            if chosen:
                sim = self.vecs[cand[pool]] @ self.vecs[np.array(cand)[chosen]].T
                same_org = self.orgs[cand[pool]][:, None] == self.orgs[np.array(cand)[chosen]][None, :]
                red = np.maximum(sim, same_org * 0.9).max(1)
            else:
                red = np.zeros(len(pool))
            best = pool[int(np.argmax(lam * rel[pool] - (1 - lam) * red))]
            pool.remove(best)
            picked_orgs = [self.orgs[cand[c]] for c in chosen]
            if picked_orgs.count(self.orgs[cand[best]]) >= max_per_org:
                continue   # cap: no single organisation fills the list
            chosen.append(best)
        return [int(cand[c]) for c in chosen]

    def _to_match(self, i: int, q: str, qv: np.ndarray, rank: int) -> Match:
        r = self.df.iloc[int(i)]
        sents = split_sentences(r.summary) + [r.title]
        sv = self.emb.encode(sents)
        why = sents[int(np.argmax(sv @ qv))]
        return Match(r.doc_id, r.title, r.org_title, r.category, r.recurrence_type,
                     round(float(self.vecs[int(i)] @ qv), 3), why)
