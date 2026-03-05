"""Tests for retrieval / virtual screening functionality."""

import numpy as np
import polars as pl
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from aev_plig.config import RetrievalConfig
from aev_plig.datasets import TargetAwareBatchSampler, get_target_labels
from aev_plig.models import get_model
from aev_plig.training import pairwise_ranking_loss

NODE_DIM = 25
EDGE_DIM = 4


# ==================== Helpers ====================


def _make_model(node_dim=NODE_DIM, edge_dim=EDGE_DIM):
    """Create a small GATv2Net for testing."""
    return get_model('GATv2Net', node_feature_dim=node_dim, edge_feature_dim=edge_dim)


def _make_data(unique_id, pk, n_nodes=5, node_dim=NODE_DIM, edge_dim=EDGE_DIM):
    """Create a minimal PyG Data object for testing."""
    x = torch.randn(n_nodes, node_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, edge_dim)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([pk], dtype=torch.float32))
    data.unique_id = unique_id
    data.pK = pk
    return data


def _make_multi_target_dataset():
    """Create a dataset with 3 targets, each with 3-4 complexes."""
    dataset = []
    # Target A: 4 complexes
    for i, pk in enumerate([8.0, 7.0, 6.0, 5.0]):
        dataset.append(_make_data(f"targetA", pk))
    # Target B: 3 complexes
    for i, pk in enumerate([9.0, 7.5, 6.5]):
        dataset.append(_make_data(f"targetB", pk))
    # Target C: 3 complexes
    for i, pk in enumerate([7.0, 5.5, 4.0]):
        dataset.append(_make_data(f"targetC", pk))
    return dataset


# ==================== Pairwise Ranking Loss ====================


class TestPairwiseRankingLoss:
    def test_basic_output(self):
        """Loss should be a scalar tensor with gradients."""
        scores = torch.tensor([[3.0], [1.0], [2.0], [0.5]], requires_grad=True)
        target_labels = torch.tensor([0, 0, 1, 1])
        affinities = torch.tensor([8.0, 6.0, 7.0, 5.0])

        loss = pairwise_ranking_loss(scores, target_labels, affinities)
        assert loss.shape == ()
        assert loss.requires_grad

        loss.backward()
        assert scores.grad is not None

    def test_correct_ranking_zero_loss(self):
        """Loss should be 0 when rankings are correct with sufficient margin."""
        scores = torch.tensor([[10.0], [5.0]], requires_grad=True)
        target_labels = torch.tensor([0, 0])
        affinities = torch.tensor([8.0, 6.0])

        loss = pairwise_ranking_loss(scores, target_labels, affinities, margin=1.0)
        # score diff = 5.0 >> margin=1.0, so loss should be 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_no_same_target_pairs(self):
        """Returns 0.0 when no same-target pairs exist."""
        scores = torch.tensor([[3.0], [1.0]], requires_grad=True)
        target_labels = torch.tensor([0, 1])  # different targets
        affinities = torch.tensor([8.0, 6.0])

        loss = pairwise_ranking_loss(scores, target_labels, affinities)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_wrong_ranking_nonzero_loss(self):
        """Loss should be positive when ranking is wrong."""
        # Higher affinity complex gets lower score
        scores = torch.tensor([[1.0], [3.0]], requires_grad=True)
        target_labels = torch.tensor([0, 0])
        affinities = torch.tensor([8.0, 6.0])  # first has higher affinity

        loss = pairwise_ranking_loss(scores, target_labels, affinities, margin=1.0)
        assert loss.item() > 0


# ==================== Target-Aware Batch Sampler ====================


class TestTargetAwareBatchSampler:
    def test_batches_have_multiple_per_target(self):
        """Each batch should have ≥2 complexes per target."""
        labels = ['A'] * 5 + ['B'] * 5 + ['C'] * 5
        sampler = TargetAwareBatchSampler(labels, complexes_per_target=3, batch_size=12)

        for batch_indices in sampler:
            batch_labels = [labels[i] for i in batch_indices]
            from collections import Counter
            counts = Counter(batch_labels)
            for target, count in counts.items():
                assert count >= 2, f"Target {target} has only {count} in batch"

    def test_excludes_singleton_targets(self):
        """Targets with <2 complexes should be excluded."""
        labels = ['A'] * 5 + ['B'] * 5 + ['singleton']
        sampler = TargetAwareBatchSampler(labels, complexes_per_target=2, batch_size=8)

        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        # Index 10 (singleton) should never appear
        assert 10 not in all_indices

    def test_len(self):
        """__len__ should return number of batches."""
        labels = ['A'] * 10 + ['B'] * 10
        sampler = TargetAwareBatchSampler(labels, complexes_per_target=4, batch_size=8)
        assert len(sampler) > 0


# ==================== get_target_labels ====================


class TestGetTargetLabels:
    def test_returns_unique_ids(self):
        dataset = [_make_data("target1", 7.0), _make_data("target2", 6.0)]
        labels = get_target_labels(dataset)
        assert labels == ["target1", "target2"]


# ==================== Predict Retrieval ====================


class TestPredictRetrieval:
    def test_output_format(self):
        """Output should have correct columns and per-protein ranks."""
        from aev_plig.prediction import predict_retrieval

        device = torch.device('cpu')
        model = _make_model()
        model.eval()

        dataset = _make_multi_target_dataset()

        df = predict_retrieval(model, dataset, device)

        assert 'protein' in df.columns
        assert 'ligand' in df.columns
        assert 'predicted_score' in df.columns
        assert 'predicted_rank' in df.columns
        assert 'actual_rank' in df.columns

        # Ranks should be 1-indexed
        assert df['predicted_rank'].min() == 1
        assert df['actual_rank'].min() == 1


# ==================== Format Retrieval Scores ====================


class TestFormatRetrievalScores:
    def test_sorted_descending(self):
        from aev_plig.prediction import format_retrieval_scores

        scores = [1.0, 3.0, 2.0]
        is_active = [0, 1, 1]
        result = format_retrieval_scores(scores, is_active)

        # Should be sorted by score descending
        assert result[0][0] == 3.0
        assert result[1][0] == 2.0
        assert result[2][0] == 1.0

        # Active flags should follow
        assert result[0][1] == 1
        assert result[1][1] == 1
        assert result[2][1] == 0

    def test_correct_shape(self):
        from aev_plig.prediction import format_retrieval_scores

        scores = [5.0, 3.0, 4.0, 1.0]
        is_active = [1, 0, 1, 0]
        result = format_retrieval_scores(scores, is_active)

        assert len(result) == 4
        assert all(len(row) == 2 for row in result)


# ==================== RDKit Scoring Integration ====================


class TestRDKitScoringIntegration:
    def test_calc_enrichment(self):
        from rdkit.ML.Scoring.Scoring import CalcEnrichment

        # Perfect ranking: all actives at top
        scores = [[5.0, 1], [4.0, 1], [3.0, 0], [2.0, 0], [1.0, 0]]
        ef = CalcEnrichment(scores, 1, [0.2])
        # At 20% (1 out of 5), we find 1 active out of 2 total = 50% of actives
        # EF = (actives found / actives total) / fraction = 0.5 / 0.2 = 2.5
        assert ef[0] > 1.0  # Better than random

    def test_calc_bedroc(self):
        from rdkit.ML.Scoring.Scoring import CalcBEDROC

        # Perfect ranking
        scores = [[5.0, 1], [4.0, 1], [3.0, 0], [2.0, 0], [1.0, 0]]
        bedroc = CalcBEDROC(scores, 1, 20.0)
        assert 0.0 <= bedroc <= 1.0
        assert bedroc > 0.5  # Should be high for perfect ranking


# ==================== Training Step ====================


class TestTrainingStep:
    def test_forward_backward(self):
        """One forward + backward pass should work."""
        device = torch.device('cpu')
        model = _make_model()
        model.to(device)
        model.train()

        data = []
        for uid, pk in [("t1", 8.0), ("t1", 6.0), ("t2", 7.0), ("t2", 5.0)]:
            data.append(_make_data(uid, pk))

        loader = DataLoader(data, batch_size=4)
        batch = next(iter(loader))
        batch = batch.to(device)

        scores = model(batch)
        assert scores.shape == (4, 1)

        target_ids = torch.tensor([0, 0, 1, 1], device=device)
        affinities = torch.tensor([8.0, 6.0, 7.0, 5.0], device=device)
        loss = pairwise_ranking_loss(scores, target_ids, affinities)

        loss.backward()
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break


# ==================== Evaluate Retrieval (end-to-end) ====================


class TestEvaluateRetrieval:
    def test_end_to_end(self):
        """evaluate_retrieval should produce a DataFrame with metrics."""
        from aev_plig.prediction import evaluate_retrieval

        device = torch.device('cpu')
        model = _make_model()
        model.eval()

        # Need enough data per target for meaningful metrics
        data = []
        for pk in [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]:
            data.append(_make_data("targetX", pk))

        df = evaluate_retrieval(model, data, device)

        # Should have metric columns
        assert 'protein' in df.columns
        assert 'bedroc' in df.columns
        assert 'rie' in df.columns
        assert 'n_ligands' in df.columns
        assert 'n_actives' in df.columns
