import numpy as np
import pandas as pd
import pytest

from src.models.data import feature_cols, load
from src.models.evaluate import baseline, metrics


def test_metrics_hand_computed():
    m = metrics([0, 2, 4], [1, 2, 1])  # errors +1, 0, -3; sum actual 6
    assert m["mae"] == pytest.approx(4 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(10 / 3))
    assert m["wape"] == pytest.approx(4 / 6)
    assert m["bias"] == pytest.approx(-2 / 6)


def test_baseline_fallback_chain():
    df = pd.DataFrame({"pu_lag168": [1, None, None], "pu_same_hour_4w": [9, 2, None],
                       "pu_recent_profile": [None, 3, None], "pu_profile": [5, 9, None]})
    assert baseline(df, "pickups", "fresh").tolist() == [1, 2, 0]
    assert baseline(df, "pickups", "published").tolist() == [5, 3, 0]


def test_test_split_guarded():
    with pytest.raises(PermissionError):
        load("test", ["pickups"])
    with pytest.raises(PermissionError):
        load("val,test", ["pickups"])


def test_feature_sets_from_tiers():
    cols = feature_cols("A+C+sid")
    assert cols[-1] == "station_id" and "pu_recent_profile" in cols and "pu_lag168" not in cols
    assert set(cols) - set(feature_cols("A+C+sid-season")) == {"doy_sin", "doy_cos", "daylight_h"}


def test_blend_weights_pick_the_better_forecast_per_hour():
    from src.models.evaluate import fit_blend

    y = np.array([4, 0, 6, 2] * 50, dtype=float)
    df = pd.DataFrame({"hour": [0, 0, 1, 1] * 50, "pickups": y,
                       "pred": np.where(np.arange(200) % 4 < 2, y, 3.0),  # model exact at hour 0
                       "b_published": np.where(np.arange(200) % 4 >= 2, y, 3.0)})  # profile exact at hour 1
    w = fit_blend(df, "pickups")
    assert w[0] == 1 and w[1] == 0
