"""Memory-strength micro-benchmark across memory providers.

Compares retrieval quality of pluggable memory backends on a fixed
fact/query set. Metrics: recall@1, recall@3, MRR@5, and store/query latency.

Covers cognicore's own backends (TF-IDF, SQLite-FTS5, Hybrid, dense MiniLM,
MultiHop) plus external providers mem0 and mempalace, all offline.
Cloud systems (zep-cloud, supermemory) are SKIPPED unless their API keys are set.

Usage:
    python benchmarks/memory_strength_bench.py
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
import warnings
from typing import Callable, List, Optional, Tuple

warnings.filterwarnings("ignore")
# Cache-first, but allow downloads: cognicore's MultiHop backend fetches a
# cross-encoder + NLTK data on first use. all-MiniLM-L6-v2 is already cached.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Windows consoles default to cp1252 and choke on ✓/✗ marks.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class Skip(Exception):
    """Raised by an adapter's setup() to mark it SKIPPED (e.g. no API key)."""


# ----------------------------------------------------------------------
# Dataset: (fact_id, category, text)
# ----------------------------------------------------------------------
#
# Distractor clusters (multiple near-neighbors per topic) force real top-k
# discrimination. Queries are split into two difficulty tiers:
#   [L] lexical  — shares surface words with the target fact
#   [S] semantic — paraphrased, minimal/zero word overlap (tests embeddings)
# A flat, easy set scores 1.00 everywhere and tells you nothing; this doesn't.
FACTS: List[Tuple[str, str, str]] = [
    # capitals (with negation distractors)
    ("f01", "capital", "The capital of France is Paris."),
    ("f02", "capital", "The capital of Japan is Tokyo."),
    ("f03", "capital", "The capital of Australia is Canberra, not Sydney."),
    ("f04", "capital", "The capital of Brazil is Brasilia, not Rio de Janeiro."),
    ("f05", "capital", "The capital of Canada is Ottawa."),
    # programming languages / creators (dense distractor cluster)
    ("f06", "lang",    "Python was created by Guido van Rossum in 1991."),
    ("f07", "lang",    "Rust was originally developed by Graydon Hoare at Mozilla."),
    ("f08", "lang",    "JavaScript was created by Brendan Eich in 1995."),
    ("f09", "lang",    "C++ was designed by Bjarne Stroustrup starting in 1985."),
    ("f10", "lang",    "Go was created at Google by Rob Pike, Ken Thompson, and Robert Griesemer."),
    # planets
    ("f11", "space",   "Mars is the fourth planet from the Sun and is called the Red Planet."),
    ("f12", "space",   "Jupiter is the largest planet in the solar system."),
    ("f13", "space",   "The Moon is Earth's only natural satellite."),
    ("f14", "space",   "Saturn is best known for its prominent ring system."),
    ("f15", "space",   "Venus is the hottest planet in the solar system."),
    # user preferences (paraphrase / negation targets)
    ("f16", "pref",    "The user prefers dark mode in their code editor."),
    ("f17", "pref",    "The user is allergic to peanuts and avoids them."),
    ("f18", "pref",    "The user's favorite programming language is Rust."),
    ("f19", "pref",    "In the morning the user drinks tea, not coffee."),
    ("f20", "pref",    "The user currently lives in Berlin."),
    # biology
    ("f21", "bio",     "Mitochondria are the powerhouse of the cell."),
    ("f22", "bio",     "DNA carries genetic information in a double helix."),
    ("f23", "bio",     "Photosynthesis converts sunlight into chemical energy in plants."),
    # history
    ("f24", "hist",    "The Berlin Wall fell in 1989."),
    ("f25", "hist",    "World War II ended in 1945."),
    ("f26", "hist",    "The first Moon landing was in 1969 by Apollo 11."),
    # tech (HTTP-code distractor cluster)
    ("f27", "tech",    "HTTP status code 404 means Not Found."),
    ("f28", "tech",    "HTTP status code 500 means Internal Server Error."),
    ("f29", "tech",    "TCP is a connection-oriented transport protocol."),
    ("f30", "tech",    "The speed of light is about 299,792 kilometers per second."),
]

# (query_text, expected_fact_id)
QUERIES: List[Tuple[str, str]] = [
    # --- [L] lexical: shares words with the target ---
    ("capital of Australia", "f03"),
    ("largest planet in the solar system", "f12"),
    ("HTTP status code 404", "f27"),
    ("when did the Berlin Wall fall", "f24"),
    ("speed of light in kilometers per second", "f30"),
    ("HTTP status code 500 meaning", "f28"),
    # --- [S] semantic: paraphrased, little/no word overlap ---
    ("Who designed the Python programming language?", "f06"),
    ("Which planet is nicknamed the Red Planet?", "f11"),
    ("What is the user unable to eat safely?", "f17"),          # allergic -> peanuts
    ("Which coding language does the user love the most?", "f18"),
    ("Where does the user reside?", "f20"),                     # reside -> lives in Berlin
    ("What beverage does the user have each morning?", "f19"),  # beverage -> tea
    ("Which organelle produces energy inside a cell?", "f21"),  # organelle -> mitochondria
    ("Which country has Brasilia as its capital?", "f04"),
    ("Who were the creators of the Go language at Google?", "f10"),
    ("Which planet has a famous set of rings?", "f14"),         # rings -> Saturn
]

TOP_K = 5


# ----------------------------------------------------------------------
# Adapter protocol: each adapter exposes
#   .name
#   .setup()                 -> None  (build store, add all FACTS)
#   .query(text, k) -> List[str]  (ranked fact_ids)
#   .teardown()              -> None
# A ValueError raised in setup() marks the adapter SKIPPED (e.g. no key).
# ----------------------------------------------------------------------

class Mem0Adapter:
    name = "mem0 (local, HF embed)"

    def setup(self):
        # Dummy key so the default OpenAI LLM *constructs*; never called (infer=False).
        os.environ.setdefault("OPENAI_API_KEY", "sk-noop-offline")
        from mem0 import Memory
        self._path = tempfile.mkdtemp(prefix="mem0_bench_")
        cfg = {
            "embedder": {"provider": "huggingface",
                         "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}},
            "vector_store": {"provider": "chroma",
                             "config": {"collection_name": "bench", "path": self._path}},
        }
        self.m = Memory.from_config(cfg)
        self._text2id = {t: fid for fid, _, t in FACTS}
        for fid, cat, text in FACTS:
            self.m.add(text, user_id="bench", infer=False, metadata={"fid": fid, "category": cat})

    def query(self, text: str, k: int) -> List[str]:
        # mem0 v2: top-level user_id removed from search(); use filters=.
        r = self.m.search(text, top_k=k, filters={"user_id": "bench"}, threshold=0.0)
        rows = r.get("results", r) if isinstance(r, dict) else r
        ids: List[str] = []
        for row in rows:
            fid = (row.get("metadata") or {}).get("fid")
            if not fid:
                fid = self._text2id.get(row.get("memory", ""))
            if fid:
                ids.append(fid)
        return ids

    def teardown(self):
        pass


class MemPalaceAdapter:
    name = "mempalace (local, chroma)"

    def setup(self):
        from mempalace.backends.chroma import ChromaBackend
        from mempalace.backends.base import PalaceRef
        self._path = tempfile.mkdtemp(prefix="mempalace_bench_")
        self.backend = ChromaBackend()
        self.ref = PalaceRef(id="bench", local_path=self._path)
        self.col = self.backend.get_collection(
            palace=self.ref, collection_name="bench", create=True
        )
        self.col.add(
            documents=[t for _, _, t in FACTS],
            ids=[fid for fid, _, _ in FACTS],
            metadatas=[{"category": c} for _, c, _ in FACTS],
        )

    def query(self, text: str, k: int) -> List[str]:
        res = self.col.query(query_texts=[text], n_results=k)
        ids = res.get("ids") if isinstance(res, dict) else getattr(res, "ids", None)
        if ids and isinstance(ids[0], list):
            return list(ids[0])
        return list(ids or [])

    def teardown(self):
        try:
            self.col.close()
            self.backend.close()
        except Exception:
            pass


class _CogniCoreBase:
    """Shared store/query logic for cognicore's MemoryBackend implementations."""
    name = "cognicore"

    def _build(self):
        raise NotImplementedError

    def setup(self):
        from cognicore.memory.base import MemoryEntry
        self._Entry = MemoryEntry
        self.backend = self._build()
        self._text2id = {t: fid for fid, _, t in FACTS}
        for fid, cat, text in FACTS:
            self.backend.store(MemoryEntry(text=text, category=cat, metadata={"fid": fid}))

    def query(self, text: str, k: int) -> List[str]:
        results = self.backend.search(text, top_k=k)
        ids: List[str] = []
        for r in results:
            entry = getattr(r, "entry", None)
            if entry is None:
                continue
            fid = (getattr(entry, "metadata", None) or {}).get("fid")
            if not fid:
                fid = self._text2id.get(getattr(entry, "text", ""))
            if fid:
                ids.append(fid)
        return ids

    def teardown(self):
        pass


class CogniTfidfAdapter(_CogniCoreBase):
    name = "cognicore TF-IDF (default)"

    def _build(self):
        from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
        return TFIDFMemoryBackend()


class CogniSqliteAdapter(_CogniCoreBase):
    name = "cognicore SQLite-FTS5"

    def _build(self):
        from cognicore.memory.sqlite_backend import SQLiteMemoryBackend
        self._path = os.path.join(tempfile.mkdtemp(prefix="cogni_sqlite_"), "bench.sqlite")
        return SQLiteMemoryBackend(self._path)


class CogniHybridAdapter(_CogniCoreBase):
    name = "cognicore Hybrid (RRF)"

    def _build(self):
        from cognicore.memory.hybrid_backend import HybridMemoryBackend
        return HybridMemoryBackend()  # dense=None -> offline TF-IDF fusion


class CogniEmbedAdapter(_CogniCoreBase):
    name = "cognicore Dense (MiniLM)"

    def _build(self):
        # Same embedding model as mem0 (all-MiniLM-L6-v2) -> fair apples-to-apples.
        from cognicore.memory.embedding_backend import BasicEmbeddingBackend
        from cognicore.memory.providers.sentence_transformers import SentenceTransformerProvider
        return BasicEmbeddingBackend(provider=SentenceTransformerProvider("all-MiniLM-L6-v2"))


class CogniMultiHopAdapter(_CogniCoreBase):
    name = "cognicore MultiHop (flagship)"

    def _build(self):
        from cognicore.memory.multihop_backend import MultiHopMemoryBackend
        return MultiHopMemoryBackend(top_k=TOP_K)

    def setup(self):
        super().setup()
        # Heavy deps (cross-encoder + NLTK) load lazily on first search; trigger
        # them here so a missing model surfaces as SKIP, not a mid-run failure.
        try:
            self.backend.search("warmup", top_k=1)
        except Exception as e:
            raise Skip(f"MultiHop deps unavailable ({type(e).__name__}: "
                       f"{str(e)[:80]}); needs cross-encoder + nltk")


class ZepAdapter:
    name = "zep-cloud (cloud)"

    def setup(self):
        key = os.environ.get("ZEP_API_KEY")
        if not key:
            raise Skip("ZEP_API_KEY not set")
        from zep_cloud.client import Zep
        self.client = Zep(api_key=key)
        # NOTE: Zep is a temporal knowledge-graph API; wiring a fair fact/query
        # roundtrip requires graph ingestion + search. Left as a live TODO.
        raise Skip("live Zep adapter not implemented (needs key + graph wiring)")

    def query(self, text: str, k: int) -> List[str]:
        return []

    def teardown(self):
        pass


class SupermemoryAdapter:
    name = "supermemory (cloud)"

    def setup(self):
        key = os.environ.get("SUPERMEMORY_API_KEY")
        if not key:
            raise Skip("SUPERMEMORY_API_KEY not set")
        from supermemory import Supermemory
        self.client = Supermemory(api_key=key)
        raise Skip("live Supermemory adapter not implemented (needs key + wiring)")

    def query(self, text: str, k: int) -> List[str]:
        return []

    def teardown(self):
        pass


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def evaluate(adapter) -> dict:
    t0 = time.perf_counter()
    adapter.setup()
    store_ms = (time.perf_counter() - t0) * 1000.0

    hits1 = hits3 = 0
    rr_sum = 0.0
    q_times: List[float] = []
    detail: List[str] = []

    for text, expected in QUERIES:
        qt0 = time.perf_counter()
        ranked = adapter.query(text, TOP_K)
        q_times.append((time.perf_counter() - qt0) * 1000.0)

        rank = ranked.index(expected) + 1 if expected in ranked else 0
        if rank == 1:
            hits1 += 1
        if 1 <= rank <= 3:
            hits3 += 1
        if rank:
            rr_sum += 1.0 / rank
        mark = "✓" if rank == 1 else ("~" if 1 <= rank <= 3 else "✗")
        detail.append(f"      {mark} rank={rank or '-':>1}  q={text!r} -> got {ranked[:3]}")

    adapter.teardown()
    n = len(QUERIES)
    return {
        "recall@1": hits1 / n,
        "recall@3": hits3 / n,
        "mrr@5": rr_sum / n,
        "store_ms": store_ms,
        "query_ms": sum(q_times) / len(q_times),
        "detail": detail,
    }


def main():
    adapters = [
        # cognicore's own memory stack
        CogniTfidfAdapter(), CogniSqliteAdapter(), CogniHybridAdapter(),
        CogniEmbedAdapter(), CogniMultiHopAdapter(),
        # external providers
        Mem0Adapter(), MemPalaceAdapter(), ZepAdapter(), SupermemoryAdapter(),
    ]
    results = []
    print(f"\nMemory-strength benchmark — {len(FACTS)} facts, {len(QUERIES)} queries, top_k={TOP_K}\n")
    for a in adapters:
        try:
            r = evaluate(a)
            results.append((a.name, r))
            print(f"[RAN ]  {a.name}")
            for line in r["detail"]:
                print(line)
        except Skip as e:
            results.append((a.name, {"skipped": str(e)}))
            print(f"[SKIP]  {a.name}: {e}")
        except Exception as e:
            results.append((a.name, {"error": f"{type(e).__name__}: {e}"}))
            print(f"[FAIL]  {a.name}: {type(e).__name__}: {e}")
        print()

    # Results table
    W = 82
    print("=" * W)
    print(f"{'backend':<32}{'R@1':>7}{'R@3':>7}{'MRR':>7}{'store_ms':>11}{'q_ms':>9}")
    print("-" * W)
    for name, r in results:
        if "recall@1" in r:
            print(f"{name:<32}{r['recall@1']:>7.2f}{r['recall@3']:>7.2f}{r['mrr@5']:>7.2f}"
                  f"{r['store_ms']:>11.0f}{r['query_ms']:>9.1f}")
        else:
            reason = r.get("skipped") or r.get("error") or "?"
            print(f"{name:<32}— {reason}")
    print("=" * W)
    print(f"\n{len(FACTS)} facts, {len(QUERIES)} queries "
          f"(6 lexical + 10 semantic-paraphrase), top_k={TOP_K}.")


if __name__ == "__main__":
    main()
