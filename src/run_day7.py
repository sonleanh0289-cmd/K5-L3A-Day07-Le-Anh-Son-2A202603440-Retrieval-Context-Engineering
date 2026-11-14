"""Run all Day 7 Retrieval/Context benchmarks and Agentic RAG evaluations."""
from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

DAY7_ROOT = Path(__file__).resolve().parent.parent
if str(DAY7_ROOT) not in sys.path:
    sys.path.insert(0, str(DAY7_ROOT))
if str(DAY7_ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(DAY7_ROOT / "starter"))

from agent_metrics import aggregate  # noqa: E402
from agentic import run_agent  # noqa: E402
from baseline import hit, pack_context, validate  # noqa: E402

from src.controller import QwenOllamaController  # noqa: E402
from src.retrievers import (  # noqa: E402
    DensePgVectorRetriever,
    HybridRerankRetriever,
    HybridRrfRetriever,
    LocalBgeM3Embedder,
    LocalBgeReranker,
    TfidfRetriever,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"


def evaluate_retrieval_config(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    split: str,
    *,
    top_k: int = 3,
    budget: int = 800,
    use_filter: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [q for q in questions if q.get("split") == split]
    traces: list[dict[str, Any]] = []
    for q in selected:
        flt = q.get("filter") if use_filter else None
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=flt, top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        top3 = results[:3]
        marked = [{**c, "hit": hit(c, q["gold"])} for c in top3]
        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "strategy": name,
                "top_k": top_k,
                "budget_chars": budget,
                "filter_used": flt,
                "gold": q["gold"],
                "results": marked,
                "context": context,
                "context_ids": ids,
                "context_chars": len(context),
                "elapsed_ms": round(elapsed_ms, 3),
                "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(c["hit"] for c in marked),
                "context_hit": any(hit(c, q["gold"]) and c["id"] in ids for c in results),
            }
        )

    n = len(traces)
    summary = {
        "strategy": name,
        "split": split,
        "top_k": top_k,
        "budget_chars": budget,
        "use_filter": use_filter,
        "n": n,
        "hit1": sum(t["hit1"] for t in traces),
        "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4) if n else 0.0,
        "hit3": sum(t["hit3"] for t in traces),
        "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4) if n else 0.0,
        "context_hit": sum(t["context_hit"] for t in traces),
        "context_hit_rate": round(sum(t["context_hit"] for t in traces) / n, 4) if n else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1)
        if n
        else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2) if n else 0.0,
        "service_cost_usd": 0.0,
    }
    return summary, traces


def _save_pair(out_dir: Path, summary: dict[str, Any], traces: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _build_review(case: dict[str, Any], trace: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Verify each citation against the gold evidence and grade the agent output."""
    cid = case["id"]
    citations: list[str] = trace.get("citations", [])
    status = trace.get("status")

    if not case["answerable"]:
        return {
            "query_id": cid,
            "citation_support": [False for _ in citations],
            "abstention_correct": status == "abstain",
            "notes": "Câu hỏi ngoài phạm vi Luật 36/2024/QH15; từ chối đúng." if status == "abstain" else "Không từ chối câu thiếu bằng chứng.",
        }

    gold = case.get("gold", [])
    support: list[bool] = []
    for c_id in citations:
        chunk = chunks_by_id.get(c_id)
        is_supported = bool(chunk and hit(chunk, gold))
        support.append(is_supported)

    if status == "abstain":
        grade = 0
        note = "Từ chối nhầm câu có đáp án trong tài liệu."
    elif any(support):
        grade = 2
        note = "Trả lời đúng nội dung điều luật và trích dẫn đúng chunk chứa bằng chứng gold."
    else:
        # Check if cited chunk is on the same gold page or partially relevant
        same_page = any(
            chunks_by_id.get(c_id, {}).get("page") == g["page"]
            for c_id in citations
            for g in gold
        )
        grade = 1 if same_page else 0
        note = "Trích dẫn cùng trang nhưng thiếu đoạn bằng chứng chính." if grade == 1 else "Trích dẫn sai chunk/trang nguồn."

    return {
        "query_id": cid,
        "answer_grade": grade,
        "citation_support": support,
        "notes": note,
    }


def run_agent_suite(
    suite_name: str,
    controller: QwenOllamaController,
    retriever: Any,
    cases: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    split: str,
    *,
    top_k: int,
    budget: int,
    use_filter: bool,
    out_dirs: list[Path],
) -> dict[str, Any]:
    selected_cases = [c for c in cases if c.get("split") == split]
    traces: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []

    for c in selected_cases:
        flt = c.get("filter") if use_filter else None
        t0 = time.perf_counter()
        try:
            res = run_agent(
                c["question"],
                controller,
                retriever,
                filters=flt,
                max_searches=3,
                top_k=top_k,
                budget=budget,
            )
        except Exception as exc:
            logger.error("Agent error on %s: %s", c["id"], exc)
            res = {"status": "error", "answer": "", "citations": [], "events": [{"event": "error", "detail": str(exc)}]}
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        trace_item = {
            "query_id": c["id"],
            "split": split,
            "question": c["question"],
            "answerable": c["answerable"],
            "strategy": suite_name,
            "model": controller.model_name,
            "elapsed_ms": elapsed_ms,
            "service_cost_usd": 0.0,
            **res,
        }
        traces.append(trace_item)
        if trace_item["status"] != "error":
            reviews.append(_build_review(c, trace_item, chunks_by_id))
        logger.info(
            "[%s | %s] %s -> status=%s, citations=%s (%sms)",
            suite_name,
            split,
            c["id"],
            trace_item["status"],
            trace_item["citations"],
            elapsed_ms,
        )

    metrics = {"strategy": suite_name, "split": split, "model": controller.model_name, **aggregate(selected_cases, traces, reviews)}
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (d / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    database_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    chunks = json.loads((DAY7_ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((DAY7_ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
    cases = json.loads((DAY7_ROOT / "data" / "cases.json").read_text(encoding="utf-8"))
    validate(chunks, questions)
    chunks_by_id = {c["id"]: c for c in chunks}

    logger.info("Initializing local BAAI/bge-m3 embedder, BAAI/bge-reranker-v2-m3, and Day 6 pgvector store...")
    embedder = LocalBgeM3Embedder("BAAI/bge-m3")
    reranker = LocalBgeReranker("BAAI/bge-reranker-v2-m3")
    tfidf_ret = TfidfRetriever(chunks)
    dense_ret = DensePgVectorRetriever(database_url, embedder, db_strategy="B")
    hybrid_ret = HybridRrfRetriever(dense_ret, tfidf_ret)
    rerank_ret = HybridRerankRetriever(dense_ret, tfidf_ret, reranker=reranker)

    dev_configs = [
        ("1_baseline_tfidf_b800", tfidf_ret, 3, 800, False),
        ("2_dense_pgvector_b800", dense_ret, 3, 800, False),
        ("3_hybrid_rrf_b800", hybrid_ret, 3, 800, False),
        ("4_imp1_hybrid_rrf_b1800", hybrid_ret, 3, 1800, False),
        ("5_imp2_hybrid_rerank_filter_b1800", rerank_ret, 3, 1800, True),
    ]

    all_retrieval_summaries: list[dict[str, Any]] = []
    for cfg_name, ret, top_k, budget, use_flt in dev_configs:
        summary, traces = evaluate_retrieval_config(
            cfg_name, ret, questions, "dev", top_k=top_k, budget=budget, use_filter=use_flt
        )
        all_retrieval_summaries.append(summary)
        _save_pair(DAY7_ROOT / "starter" / "results" / f"dev-{cfg_name}", summary, traces)
        _save_pair(DAY7_ROOT / "results" / f"dev-{cfg_name}", summary, traces)
        logger.info("Dev retrieval [%s]: %s", cfg_name, summary)

    # Evaluate Baseline and Selected Configuration (5_imp2_hybrid_rerank_filter_b1800) on held-out TEST
    for cfg_name, ret, top_k, budget, use_flt in [
        ("1_baseline_tfidf_b800", tfidf_ret, 3, 800, False),
        ("5_imp2_hybrid_rerank_filter_b1800", rerank_ret, 3, 1800, True),
    ]:
        summary, traces = evaluate_retrieval_config(
            cfg_name, ret, questions, "test", top_k=top_k, budget=budget, use_filter=use_flt
        )
        all_retrieval_summaries.append(summary)
        _save_pair(DAY7_ROOT / "starter" / "results" / f"test-{cfg_name}", summary, traces)
        _save_pair(DAY7_ROOT / "results" / f"test-{cfg_name}", summary, traces)
        logger.info("Test retrieval [%s]: %s", cfg_name, summary)

    # Run Real Agent with Qwen 2.5 7B on dev & test for Baseline vs Selected Configuration
    rerank_only = "--rerank-only" in sys.argv
    controller = QwenOllamaController(model_name="qwen2.5:7b")
    agent_summaries: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        for suite_name, ret, top_k, budget, use_flt in [
            ("agent_baseline_tfidf_b800", tfidf_ret, 3, 800, False),
            ("agent_selected_hybrid_rerank_b1800", rerank_ret, 3, 1800, True),
        ]:
            existing_summary = DAY7_ROOT / "results" / f"{split}-{suite_name}" / "summary.json"
            if rerank_only and suite_name == "agent_baseline_tfidf_b800" and existing_summary.exists():
                m = json.loads(existing_summary.read_text(encoding="utf-8"))
                agent_summaries.append(m)
                continue
            m = run_agent_suite(
                suite_name,
                controller,
                ret,
                cases,
                chunks_by_id,
                split,
                top_k=top_k,
                budget=budget,
                use_filter=use_flt,
                out_dirs=[
                    DAY7_ROOT / "starter" / "results" / f"{split}-{suite_name}",
                    DAY7_ROOT / "results" / f"{split}-{suite_name}",
                ],
            )
            agent_summaries.append(m)
            logger.info("Agent summary [%s | %s]: %s", suite_name, split, json.dumps(m, ensure_ascii=False))

    overview = {
        "retrieval_benchmarks": all_retrieval_summaries,
        "agent_benchmarks": agent_summaries,
    }
    for root_d in [DAY7_ROOT / "starter" / "results", DAY7_ROOT / "results"]:
        (root_d / "day7_overview.json").write_text(
            json.dumps(overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

