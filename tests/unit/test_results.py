import math

import polars as pl
import pytest

from aev_plig import results


def test_load_predictions_requires_existing_parquet(tmp_path):
    missing = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        results.load_predictions(str(missing))


def test_load_predictions_rejects_non_parquet(tmp_path):
    bad = tmp_path / "predictions.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        results.load_predictions(str(bad))


def test_metrics_primitives():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]

    assert math.isclose(results.rmse(y_true, y_pred), math.sqrt(1 / 3), rel_tol=1e-8)
    assert math.isclose(results.pearson_r(y_true, y_true), 1.0, rel_tol=1e-8)
    assert math.isclose(results.kendall_tau(y_true, y_true), 1.0, rel_tol=1e-8)


def test_overall_metrics():
    df = pl.DataFrame({"pK": [1.0, 2.0, 3.0], "preds": [1.0, 2.0, 4.0]})
    out = results.overall_metrics(df, truth_col="pK", pred_col="preds")

    assert out.keys() == {"n", "rmse", "pearson_r", "kendall_tau"}
    assert out["n"] == 3.0


def test_per_target_metrics_columns_and_filtering():
    df = pl.DataFrame(
        {
            "target": ["A", "A", "B"],
            "pK": [1.0, 2.0, 1.0],
            "preds": [1.0, 1.5, 1.2],
        }
    )

    out = results.per_target_metrics(
        df,
        target_col="target",
        truth_col="pK",
        pred_col="preds",
        min_samples=2,
    )

    assert out.columns == ["target", "n", "rmse", "pearson_r", "kendall_tau"]
    assert out.height == 1
    assert out["target"][0] == "A"
    assert out["n"][0] == 2
