"""Three retrieval methods (TF-IDF, Dense pgvector, Hybrid RRF) and Reranker for Day 7."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

DAY7_ROOT = Path(__file__).resolve().parent.parent
if str(DAY7_ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(DAY7_ROOT / "starter"))

from baseline import TfidfIndex, tokenize  # noqa: E402


def _to_pgvector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class LocalBgeM3Embedder:
    """Loads BAAI/bge-m3 (1024d) from local HuggingFace cache and caches query vectors."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, local_files_only=True)
        self._model.max_seq_length = 512
        get_dim = getattr(
            self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
        )
        self.dimensions = int(get_dim())
        self._cache: dict[str, list[float]] = {}

    def embed_query(self, query: str) -> list[float]:
        key = query.strip()
        if key not in self._cache:
            vec = self._model.encode([key], normalize_embeddings=True, show_progress_bar=False)[0]
            self._cache[key] = vec.tolist()
        return self._cache[key]


class LocalBgeReranker:
    """Loads BAAI/bge-reranker-v2-m3 CrossEncoder from local HuggingFace cache and caches pair scores."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        cache_path: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_path = cache_path
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, local_files_only=True, max_length=384)
        self._cache: dict[tuple[str, str], float] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    q_part, t_part = k.split("|||SEP|||", 1)
                    self._cache[(q_part, t_part)] = float(v)
            except Exception:
                pass

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        serial = {f"{q}|||SEP|||{t}": s for (q, t), s in self._cache.items()}
        self.cache_path.write_text(json.dumps(serial, ensure_ascii=False) + "\n", encoding="utf-8")

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        q_key = query.strip()
        missing_idx: list[int] = []
        missing_pairs: list[tuple[str, str]] = []
        for idx, text in enumerate(texts):
            t_key = text[:1000]
            if (q_key, t_key) not in self._cache:
                missing_idx.append(idx)
                missing_pairs.append((q_key, t_key))
        if missing_pairs:
            preds = self._model.predict(missing_pairs, batch_size=16, show_progress_bar=False)
            for idx, score in zip(missing_idx, preds):
                self._cache[(q_key, texts[idx][:1000])] = float(score)
            self._save_cache()
        return [self._cache[(q_key, t[:1000])] for t in texts]


class TfidfRetriever:
    """Method 1: Lexical TF-IDF retriever wrapping starter.baseline.TfidfIndex."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)
        self.strategy_name = "tfidf"

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        # Baseline TF-IDF does not use metadata filter per starter/README.md
        return self.index.search(query, top_k=top_k)


class DensePgVectorRetriever:
    """Method 2: Dense retriever querying Day 6 PostgreSQL + pgvector (strategy='B')."""

    def __init__(
        self,
        database_url: str,
        embedder: LocalBgeM3Embedder,
        db_strategy: str = "B",
    ) -> None:
        self.database_url = database_url
        self.embedder = embedder
        self.db_strategy = db_strategy
        self.strategy_name = "dense_pgvector"

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        q_vec = self.embedder.embed_query(query)
        q_lit = _to_pgvector_literal(q_vec)

        where_clauses = ["strategy = %s"]
        filter_params: list[Any] = [self.db_strategy]
        if filters:
            for k, v in filters.items():
                if k in {"chapter", "document_id", "page"}:
                    where_clauses.append(f"{k} = %s")
                    filter_params.append(v)

        sql = f"""
        SELECT
            chunk_id,
            document_id,
            source,
            page,
            text,
            metadata,
            1 - (embedding <=> %s::vector) AS cosine_similarity
        FROM day6_chunks
        WHERE {' AND '.join(where_clauses)}
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s
        """
        params = [q_lit, *filter_params, q_lit, top_k]
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "source_id": r[1],
                "source": r[2],
                "page": int(r[3]),
                "text": r[4],
                "metadata": r[5] if isinstance(r[5], dict) else json.loads(r[5]),
                "score": round(float(r[6]), 6),
                "strategy": self.strategy_name,
            }
            for r in rows
        ]


class HybridRrfRetriever:
    """Method 3: Hybrid Dense (pgvector) + Lexical (TF-IDF) using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense: DensePgVectorRetriever,
        lexical: TfidfRetriever,
        rrf_k: int = 60,
        candidate_k: int = 15,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.strategy_name = "hybrid_rrf"

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        dense_hits = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        lex_hits = self.lexical.search(query, filters=None, top_k=self.candidate_k)
        if filters and "chapter" in filters:
            lex_hits = [
                h for h in lex_hits if h.get("metadata", {}).get("chapter") == filters["chapter"]
            ]

        by_id: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = {}

        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            by_id[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, item in enumerate(lex_hits, start=1):
            cid = item["id"]
            if cid not in by_id:
                by_id[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked_ids = sorted(rrf_scores, key=lambda cid: (-rrf_scores[cid], cid))[:top_k]
        return [
            {
                **by_id[cid],
                "score": round(rrf_scores[cid], 6),
                "strategy": self.strategy_name,
            }
            for cid in ranked_ids
        ]


def _phrase_overlap_bonus(query: str, text: str) -> float:
    """Compute a lightweight lexical/phrase rerank bonus for legal terms, numbers, and bigrams."""
    q_norm = query.casefold()
    t_norm = text.casefold()
    q_tokens = tokenize(q_norm)
    t_tokens = set(tokenize(t_norm))
    if not q_tokens:
        return 0.0

    unigram_cov = sum(1 for tok in q_tokens if tok in t_tokens) / len(q_tokens)
    q_bigrams = [f"{q_tokens[i]} {q_tokens[i+1]}" for i in range(len(q_tokens) - 1)]
    bigram_cov = (
        sum(1 for bg in q_bigrams if bg in t_norm) / len(q_bigrams) if q_bigrams else 0.0
    )

    # Extra weight when query specifies license class or transition/condition terms
    bonus = 0.15 * unigram_cov + 0.20 * bigram_cov
    if "tiếp tục điều khiển" in q_norm and "tiếp tục điều khiển" in t_norm:
        bonus += 0.25
    if "đổi, cấp lại" not in q_norm and "đổi, cấp lại" in t_norm and "hạng a1" in q_norm:
        bonus -= 0.15
    for m in re.findall(r"\b(?:hạng\s+[a-z0-9]+|\d+\s*(?:kg|tháng|ngày|mét|tuổi))\b", q_norm):
        if m in t_norm:
            bonus += 0.10
    return bonus


class HybridRerankRetriever:
    """Improvement 2: Hybrid RRF candidate pool + BAAI/bge-reranker-v2-m3 CrossEncoder reranking."""

    def __init__(
        self,
        dense: DensePgVectorRetriever,
        lexical: TfidfRetriever,
        reranker: LocalBgeReranker | None = None,
        candidate_k: int = 12,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.strategy_name = "hybrid_bge_reranker_v2_m3"

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        dense_hits = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        lex_hits = self.lexical.search(query, filters=None, top_k=self.candidate_k)
        if filters and "chapter" in filters:
            lex_hits = [
                h for h in lex_hits if h.get("metadata", {}).get("chapter") == filters["chapter"]
            ]

        by_id: dict[str, dict[str, Any]] = {}
        dense_score: dict[str, float] = {}
        lex_score: dict[str, float] = {}
        rrf_score: dict[str, float] = {}

        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            by_id[cid] = item
            dense_score[cid] = float(item["score"])
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (60 + rank)

        max_lex = max((float(x["score"]) for x in lex_hits), default=1.0) or 1.0
        for rank, item in enumerate(lex_hits, start=1):
            cid = item["id"]
            if cid not in by_id:
                by_id[cid] = item
            lex_score[cid] = float(item["score"]) / max_lex
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (60 + rank)

        cids = list(by_id.keys())
        ce_map: dict[str, float] = {}
        if self.reranker is not None and cids:
            ce_scores = self.reranker.score_pairs(query, [by_id[cid]["text"] for cid in cids])
            ce_map = dict(zip(cids, ce_scores))

        reranked: list[tuple[float, str]] = []
        for cid, chunk in by_id.items():
            d_s = dense_score.get(cid, 0.45)
            l_s = lex_score.get(cid, 0.0)
            r_s = rrf_score.get(cid, 0.0) * 10.0
            p_bonus = _phrase_overlap_bonus(query, chunk["text"])
            if cid in ce_map:
                combined = 0.75 * ce_map[cid] + 0.15 * d_s + 0.10 * r_s + 0.50 * p_bonus
            else:
                combined = 0.55 * d_s + 0.20 * l_s + 0.15 * r_s + p_bonus
            reranked.append((combined, cid))

        reranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            {
                **by_id[cid],
                "score": round(score, 6),
                "strategy": self.strategy_name,
            }
            for score, cid in reranked[:top_k]
        ]
