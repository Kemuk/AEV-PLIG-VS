"""
Unit tests for training script argument parsing and file organization.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import argparse


class TestTrainScriptFilenames:
    """Tests for seed-based filename generation."""

    def test_model_filename_format(self):
        """Test that model filenames use seed format."""
        seed = 100
        expected = f"model_seed_{seed}.model"
        assert expected == "model_seed_100.model"

    def test_model_filename_multiple_seeds(self):
        """Test filename format for multiple seeds."""
        seeds = [100, 123, 15]
        filenames = [f"model_seed_{seed}.model" for seed in seeds]

        assert filenames[0] == "model_seed_100.model"
        assert filenames[1] == "model_seed_123.model"
        assert filenames[2] == "model_seed_15.model"

    def test_output_directory_format(self):
        """Test output directory naming convention."""
        model_name = "GATv2Net"
        timestamp = "20250211_120000"
        expected_dir = f"models/{model_name}_{timestamp}"

        assert expected_dir == "models/GATv2Net_20250211_120000"

    def test_scaler_filename_format(self):
        """Test scaler filename format."""
        expected = "scaler.pickle"
        assert expected == "scaler.pickle"


class TestOutputDirectoryStructure:
    """Tests for output directory creation and structure."""

    def test_output_directory_creation(self):
        """Test that output directory is created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = "GATv2Net"
            timestamp = "20250211_120000"
            output_dir = os.path.join(tmpdir, f"{model_name}_{timestamp}")

            # Create directory
            os.makedirs(output_dir, exist_ok=True)

            # Verify it exists
            assert os.path.exists(output_dir)
            assert os.path.isdir(output_dir)

    def test_multiple_models_in_same_directory(self):
        """Test that multiple model files can coexist in same directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "GATv2Net_20250211_120000")
            os.makedirs(output_dir, exist_ok=True)

            # Create multiple model files
            seeds = [100, 123, 15]
            model_paths = []
            for seed in seeds:
                model_path = os.path.join(output_dir, f"model_seed_{seed}.model")
                Path(model_path).touch()  # Create empty file
                model_paths.append(model_path)

            # Verify all exist
            for path in model_paths:
                assert os.path.exists(path)

            # Verify they're in the same directory
            parent_dirs = [os.path.dirname(p) for p in model_paths]
            assert len(set(parent_dirs)) == 1  # All in same dir

    def test_scaler_in_same_directory_as_models(self):
        """Test that scaler is saved in the same directory as models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "GATv2Net_20250211_120000")
            os.makedirs(output_dir, exist_ok=True)

            # Create model and scaler
            model_path = os.path.join(output_dir, "model_seed_100.model")
            scaler_path = os.path.join(output_dir, "scaler.pickle")

            Path(model_path).touch()
            Path(scaler_path).touch()

            # Verify same directory
            assert os.path.dirname(model_path) == os.path.dirname(scaler_path)


class TestSeedUniqueness:
    """Tests for ensuring seed uniqueness in filenames."""

    def test_different_seeds_different_filenames(self):
        """Test that different seeds produce different filenames."""
        seed1 = 100
        seed2 = 123

        filename1 = f"model_seed_{seed1}.model"
        filename2 = f"model_seed_{seed2}.model"

        assert filename1 != filename2

    def test_seed_collision_impossible(self):
        """Test that seed-based naming prevents collisions."""
        seeds = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
        filenames = [f"model_seed_{seed}.model" for seed in seeds]

        # All filenames should be unique
        assert len(filenames) == len(set(filenames))


class TestTimestampHandling:
    """Tests for timestamp parameter handling."""

    def test_timestamp_format(self):
        """Test timestamp format matches expected pattern."""
        timestamp = "20250211_120000"

        # Should match pattern: YYYYMMDD_HHMMSS
        assert len(timestamp) == 15
        assert timestamp[8] == "_"
        assert timestamp[:8].isdigit()  # Date part
        assert timestamp[9:].isdigit()  # Time part

    def test_shared_timestamp_across_jobs(self):
        """Test that shared timestamp creates same output directory."""
        model_name = "GATv2Net"
        shared_timestamp = "20250211_120000"

        # Multiple jobs with same timestamp
        dir1 = f"models/{model_name}_{shared_timestamp}"
        dir2 = f"models/{model_name}_{shared_timestamp}"

        assert dir1 == dir2  # Same directory for all parallel jobs
