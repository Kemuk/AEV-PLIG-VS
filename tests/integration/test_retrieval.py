"""Tests for retrieval / virtual screening functionality."""

import numpy as np
import polars as pl
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from aev_plig.config import RetrievalConfig
from aev_plig.models import get_model
from aev_plig.training import inbatch_contrastive_loss

NODE_DIM = 25
EDGE_DIM = 4


# ==================== Helpers ====================


def _make_retrieval_model(node_dim=NODE_DIM, edge_dim=EDGE_DIM):
    """Create a small GATv2NetRetrieval for testing."""
    return get_model('GATv2NetRetrieval', node_feature_dim=node_dim, edge_feature_dim=edge_dim)


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


# ==================== In-Batch Contrastive Loss ====================


class TestInbatchContrastiveLoss:
    def test_basic_output(self):
        """Loss should be a scalar tensor with gradients."""
        protein_emb = torch.randn(4, 128, requires_grad=True)
        ligand_emb = torch.randn(4, 128, requires_grad=True)

        loss = inbatch_contrastive_loss(protein_emb, ligand_emb)
        assert loss.shape == ()
        assert loss.requires_grad

        loss.backward()
        assert protein_emb.grad is not None
        assert ligand_emb.grad is not None

    def test_perfect_alignment_low_loss(self):
        """Loss should be low when embeddings are perfectly aligned on diagonal."""
        # Make protein and ligand embeddings identical (perfect match on diagonal)
        emb = torch.randn(4, 128)
        protein_emb = emb.clone().requires_grad_(True)
        ligand_emb = emb.clone().requires_grad_(True)

        loss = inbatch_contrastive_loss(protein_emb, ligand_emb, temperature=1.0)
        # With identical embeddings, diagonal has score 1.0, off-diagonal < 1.0
        # Loss should be relatively low
        assert loss.item() < 2.0  # cross-entropy on near-identity matrix

    def test_random_embeddings_higher_loss(self):
        """Random embeddings should have higher loss than aligned ones."""
        torch.manual_seed(42)
        emb = torch.randn(8, 128)

        # Aligned
        loss_aligned = inbatch_contrastive_loss(
            emb.clone(), emb.clone(), temperature=1.0
        ).item()

        # Random
        loss_random = inbatch_contrastive_loss(
            torch.randn(8, 128), torch.randn(8, 128), temperature=1.0
        ).item()

        assert loss_random > loss_aligned

    def test_temperature_scaling(self):
        """Lower temperature should produce sharper distributions (higher loss for random)."""
        torch.manual_seed(42)
        p = torch.randn(4, 128)
        l = torch.randn(4, 128)

        loss_high_temp = inbatch_contrastive_loss(p, l, temperature=1.0).item()
        loss_low_temp = inbatch_contrastive_loss(p, l, temperature=0.07).item()

        # Lower temperature amplifies differences
        assert loss_low_temp > loss_high_temp


# ==================== GATv2NetRetrieval Model ====================


class TestGATv2NetRetrieval:
    def test_forward_returns_tuple(self):
        """forward() should return (protein_emb, ligand_emb) tuple."""
        model = _make_retrieval_model()
        data = [_make_data("t1", 7.0) for _ in range(3)]
        loader = DataLoader(data, batch_size=3)
        batch = next(iter(loader))

        output = model(batch)
        assert isinstance(output, tuple)
        assert len(output) == 2

        protein_emb, ligand_emb = output
        assert protein_emb.shape == (3, RetrievalConfig.EMBEDDING_DIM)
        assert ligand_emb.shape == (3, RetrievalConfig.EMBEDDING_DIM)

    def test_predict_returns_scalar(self):
        """predict() should return scalar scores."""
        model = _make_retrieval_model()
        data = [_make_data("t1", 7.0) for _ in range(3)]
        loader = DataLoader(data, batch_size=3)
        batch = next(iter(loader))

        scores = model.predict(batch)
        assert scores.shape == (3, 1)

    def test_custom_embed_dim(self):
        """Should respect custom embed_dim from config."""
        from types import SimpleNamespace
        config = SimpleNamespace(embed_dim=64)
        model = get_model('GATv2NetRetrieval',
                          node_feature_dim=NODE_DIM, edge_feature_dim=EDGE_DIM,
                          config=config)
        data = [_make_data("t1", 7.0) for _ in range(2)]
        loader = DataLoader(data, batch_size=2)
        batch = next(iter(loader))

        p, l = model(batch)
        assert p.shape == (2, 64)
        assert l.shape == (2, 64)


# ==================== Predict Retrieval ====================


class TestPredictRetrieval:
    def test_output_format(self):
        """Output should have correct columns and per-protein ranks."""
        from aev_plig.prediction import predict_retrieval

        device = torch.device('cpu')
        model = _make_retrieval_model()
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

    def test_works_with_affinity_model(self):
        """predict_retrieval should also work with standard GATv2Net."""
        from aev_plig.prediction import predict_retrieval

        device = torch.device('cpu')
        model = _make_model()
        model.eval()

        dataset = _make_multi_target_dataset()
        df = predict_retrieval(model, dataset, device)
        assert len(df) == len(dataset)


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
        """One forward + backward pass through contrastive loss should work."""
        device = torch.device('cpu')
        model = _make_retrieval_model()
        model.to(device)
        model.train()

        data = []
        for uid, pk in [("t1", 8.0), ("t1", 6.0), ("t2", 7.0), ("t2", 5.0)]:
            data.append(_make_data(uid, pk))

        loader = DataLoader(data, batch_size=4)
        batch = next(iter(loader))
        batch = batch.to(device)

        protein_emb, ligand_emb = model(batch)
        loss = inbatch_contrastive_loss(protein_emb, ligand_emb)

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
        model = _make_retrieval_model()
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
