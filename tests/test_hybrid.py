"""Unit tests for hybrid retrieval helpers."""

import unittest

from retrieval.hybrid import fuse_hybrid_candidates, reciprocal_rank_fusion
from retrieval.metadata_filter import (
    build_chroma_where,
    chunk_matches_filters,
    merge_metadata_filters,
    normalize_metadata_filters,
)


class TestReciprocalRankFusion(unittest.TestCase):
    def test_fuses_overlapping_and_unique_results(self):
        dense = [
            {"chunk_id": "a", "similarity_score": 0.9, "text": "alpha"},
            {"chunk_id": "b", "similarity_score": 0.8, "text": "beta"},
        ]
        sparse = [
            {"chunk_id": "b", "bm25_score": 4.0, "bm25_score_normalized": 1.0, "text": "beta"},
            {"chunk_id": "c", "bm25_score": 2.0, "bm25_score_normalized": 0.5, "text": "gamma"},
        ]

        fused = reciprocal_rank_fusion(
            [dense, sparse],
            weights=[0.7, 0.3],
            rrf_k=60,
        )

        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["chunk_id"], "b")
        self.assertIn("rrf_score", fused[0])
        self.assertEqual(fused[0]["dense_rank"], 2)
        self.assertEqual(fused[0]["bm25_rank"], 1)

    def test_dense_only_weight_preserves_dense_order(self):
        dense = [
            {"chunk_id": "a", "similarity_score": 0.95},
            {"chunk_id": "b", "similarity_score": 0.85},
        ]
        fused = reciprocal_rank_fusion([dense], weights=[1.0], rrf_k=60)
        self.assertEqual([c["chunk_id"] for c in fused], ["a", "b"])


class TestFuseHybridCandidates(unittest.TestCase):
    def test_falls_back_to_dense_when_sparse_missing(self):
        dense = [{"chunk_id": "a", "similarity_score": 0.9}]
        fused, stats = fuse_hybrid_candidates(dense, [], alpha=0.7, rrf_k=60)
        self.assertEqual([c["chunk_id"] for c in fused], ["a"])
        self.assertEqual(stats["fusion_mode"], "dense_only")
        self.assertIn("bm25_returned_no_candidates", stats["warnings"])

    def test_falls_back_to_sparse_when_dense_missing(self):
        sparse = [{"chunk_id": "b", "bm25_score_normalized": 0.8}]
        fused, stats = fuse_hybrid_candidates([], sparse, alpha=0.7, rrf_k=60)
        self.assertEqual([c["chunk_id"] for c in fused], ["b"])
        self.assertEqual(stats["fusion_mode"], "sparse_only")

    def test_rejects_invalid_alpha(self):
        with self.assertRaises(ValueError):
            fuse_hybrid_candidates([], [], alpha=1.5, rrf_k=60)


class TestMetadataFilters(unittest.TestCase):
    def test_build_chroma_where_single_field(self):
        where = build_chroma_where({"source_file": "policy.md"})
        self.assertEqual(where, {"source_file": {"$eq": "policy.md"}})

    def test_build_chroma_where_multiple_fields(self):
        where = build_chroma_where(
            {"source_file": "policy.md", "doc_type": ["policy", "runbook"]}
        )
        self.assertEqual(
            where,
            {
                "$and": [
                    {"source_file": {"$eq": "policy.md"}},
                    {"doc_type": {"$in": ["policy", "runbook"]}},
                ]
            },
        )

    def test_merge_metadata_filters_body_and_query_shortcuts(self):
        merged = merge_metadata_filters(
            {"doc_type": "policy"},
            {"source_file": "access_control_policy.md"},
        )
        self.assertEqual(
            merged,
            {
                "doc_type": "policy",
                "source_file": "access_control_policy.md",
            },
        )

    def test_chunk_matches_filters(self):
        chunk = {
            "source_file": "access_control_policy.md",
            "doc_type": "policy",
            "section_title": "Purpose",
        }
        self.assertTrue(chunk_matches_filters(chunk, {"doc_type": "policy"}))
        self.assertFalse(chunk_matches_filters(chunk, {"doc_type": "runbook"}))

    def test_rejects_unknown_filter_keys(self):
        with self.assertRaises(ValueError):
            normalize_metadata_filters({"unknown_field": "x"})


if __name__ == "__main__":
    unittest.main()
