"""Unit tests for Day 7 custom retrievers (TF-IDF, Hybrid RRF, Hybrid Rerank)."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrievers import HybridRerankRetriever, HybridRrfRetriever, TfidfRetriever


class _FakeDense:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, *, filters=None, top_k=3):
        items = self.chunks
        if filters and "chapter" in filters:
            items = [c for c in items if c["metadata"].get("chapter") == filters["chapter"]]
        return [{**c, "score": 0.9 - 0.1 * i, "strategy": "dense_pgvector"} for i, c in enumerate(items[:top_k])]


class _FakeReranker:
    def score_pairs(self, query, texts):
        q_norm = query.casefold()
        return [0.99 if "hạng a1" in q_norm and "hạng a1" in t.casefold() else 0.05 for t in texts]


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "c1",
                "source_id": "doc1",
                "source": "law.pdf",
                "page": 8,
                "text": "Người lái xe và người được chở trên xe ô tô phải thắt dây đai an toàn.",
                "metadata": {"chapter": "Chương II"},
            },
            {
                "id": "c2",
                "source_id": "doc1",
                "source": "law.pdf",
                "page": 41,
                "text": "Sau thời hạn ít nhất là 06 tháng kể từ ngày bị trừ hết điểm được kiểm tra.",
                "metadata": {"chapter": "Chương IV"},
            },
            {
                "id": "c3",
                "source_id": "doc1",
                "source": "law.pdf",
                "page": 60,
                "text": "Giấy phép lái xe hạng A1 được tiếp tục điều khiển xe mô tô hai bánh.",
                "metadata": {"chapter": "Chương IX"},
            },
        ]

    def test_tfidf_and_hybrid_rrf_and_rerank(self):
        tfidf = TfidfRetriever(self.chunks)
        dense = _FakeDense(self.chunks)
        hybrid = HybridRrfRetriever(dense, tfidf)
        rerank = HybridRerankRetriever(dense, tfidf, reranker=_FakeReranker())

        r1 = tfidf.search("thắt dây đai an toàn", top_k=2)
        self.assertEqual(r1[0]["id"], "c1")

        r2 = hybrid.search("trừ hết điểm 06 tháng", filters={"chapter": "Chương IV"}, top_k=2)
        self.assertEqual(r2[0]["id"], "c2")

        r3 = rerank.search("hạng A1 tiếp tục điều khiển", top_k=2)
        self.assertEqual(r3[0]["id"], "c3")


if __name__ == "__main__":
    unittest.main()

