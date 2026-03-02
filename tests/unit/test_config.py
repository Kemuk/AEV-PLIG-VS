"""
Unit tests for Config class, especially seed validation.
"""

import os
import pytest
from aev_plig.config import Config


class TestConfigGetCpuCount:
    """Tests for Config.get_cpu_count()"""

    def test_returns_slurm_cpus_when_set(self, monkeypatch):
        """get_cpu_count() returns SLURM_CPUS_PER_TASK when set."""
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
        assert Config.get_cpu_count() == 4

    def test_falls_back_to_os_cpu_count(self, monkeypatch):
        """get_cpu_count() falls back to os.cpu_count() when SLURM var is unset."""
        monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
        assert Config.get_cpu_count() == os.cpu_count()

    def test_returns_int(self, monkeypatch):
        """get_cpu_count() always returns an int."""
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')
        result = Config.get_cpu_count()
        assert isinstance(result, int)


class TestConfigSeedValidation:
    """Tests for Config.validate_ensemble_seeds()"""

    def test_validate_seeds_no_duplicates(self):
        """Test that validation passes with unique seeds."""
        # Should not raise any exception
        Config.validate_ensemble_seeds()

    def test_validate_seeds_with_duplicates(self, monkeypatch):
        """Test that validation fails with duplicate seeds."""
        # Mock Config.ENSEMBLE_SEEDS with duplicates
        duplicated_seeds = [100, 123, 15, 100, 200]  # 100 appears twice
        monkeypatch.setattr(Config, 'ENSEMBLE_SEEDS', duplicated_seeds)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Duplicate seeds found"):
            Config.validate_ensemble_seeds()

    def test_validate_seeds_all_same(self, monkeypatch):
        """Test that validation fails when all seeds are the same."""
        duplicated_seeds = [42, 42, 42, 42]
        monkeypatch.setattr(Config, 'ENSEMBLE_SEEDS', duplicated_seeds)

        with pytest.raises(ValueError, match="Duplicate seeds found"):
            Config.validate_ensemble_seeds()

    def test_validate_seeds_single_seed(self, monkeypatch):
        """Test that validation passes with a single seed."""
        single_seed = [100]
        monkeypatch.setattr(Config, 'ENSEMBLE_SEEDS', single_seed)

        # Should not raise any exception
        Config.validate_ensemble_seeds()

    def test_validate_seeds_empty_list(self, monkeypatch):
        """Test that validation passes with empty seed list."""
        empty_seeds = []
        monkeypatch.setattr(Config, 'ENSEMBLE_SEEDS', empty_seeds)

        # Should not raise (len([]) == len(set([])) == 0)
        Config.validate_ensemble_seeds()


class TestConfigEnsembleSeeds:
    """Tests for default ensemble seeds configuration."""

    def test_ensemble_seeds_are_unique(self):
        """Test that default ensemble seeds have no duplicates."""
        seeds = Config.ENSEMBLE_SEEDS
        assert len(seeds) == len(set(seeds)), "Default ENSEMBLE_SEEDS contains duplicates!"

    def test_ensemble_seeds_match_ensemble_size(self):
        """Test that number of seeds matches ENSEMBLE_SIZE."""
        assert len(Config.ENSEMBLE_SEEDS) == Config.ENSEMBLE_SIZE

    def test_ensemble_seeds_are_integers(self):
        """Test that all seeds are integers."""
        for seed in Config.ENSEMBLE_SEEDS:
            assert isinstance(seed, int), f"Seed {seed} is not an integer!"

    def test_ensemble_seeds_are_positive(self):
        """Test that all seeds are positive integers."""
        for seed in Config.ENSEMBLE_SEEDS:
            assert seed > 0, f"Seed {seed} is not positive!"
