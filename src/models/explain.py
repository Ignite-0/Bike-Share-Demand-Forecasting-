"""SHAP for a saved LightGBM model (step 7): global importance on a 50k-row val sample, plus three local cases.

Values come from LightGBM's own TreeSHAP (pred_contrib=True) and are in the model's raw output space: for
Tweedie/Poisson that is log(expected count), so contributions add up to log(prediction), not trips.
    python -m src.models.explain pickups_A+sid_tweedie1.2_lightgbm_cpu_tuned
"""
import sys

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

from src.models.data import load  # noqa: E402

FIG = "reports/figures"
CASES = {  # (title, row filter); the busiest matching station is shown
    "rainy_weekday_8am_business": ("Rainy weekday 8:00, business station",
                                   lambda d: (d.cluster == 3) & (d.hour == 8) & ~d.is_weekend & ~d.is_holiday
                                   & (d.precip_mm >= 1)),
    "summer_saturday_afternoon_leisure": ("Summer Saturday 15:00, evening-heavy (leisure-type) station",
                                          lambda d: (d.cluster == 1) & (d.hour == 15) & (d.dow == 6)
                                          & (d.precip_mm < 0.1)),
    "quiet_station_night": ("Weekday 3:00, quiet station (5-20 trips/day)",
                            lambda d: (d.volume_tier == "5-20/day") & (d.hour == 3) & ~d.is_weekend & ~d.is_holiday),
}


def explanation(model, x):
    contrib = model.predict(x, pred_contrib=True)
    data = x.assign(station_id=x["station_id"].cat.codes) if "station_id" in x else x
    return shap.Explanation(contrib[:, :-1], contrib[:, -1], data=data.to_numpy(dtype=float),
                            feature_names=list(x.columns))


def main(name: str) -> None:
    target = name.split("_")[0]
    model = lgb.Booster(model_file=f"data/processed/models/{name}.txt")
    feats = model.feature_name()
    val = load("val", feats + [target, "cluster", "volume_tier", "ts_local"])

    sample = val.sample(50_000, random_state=0)
    ex = explanation(model, sample[feats])
    imp = np.abs(ex.values).mean(0)
    print("mean |SHAP| (log space):")
    for f, v in sorted(zip(feats, imp), key=lambda t: -t[1]):
        print(f"  {f:18s} {v:.3f}")
    shap.plots.beeswarm(ex, max_display=15, show=False)
    plt.title(f"SHAP, {target} model (log-count space), 50k val rows")
    plt.savefig(f"{FIG}/shap_beeswarm_{target}.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    for key, (title, cond) in CASES.items():
        rows = val[cond(val)]
        busiest = rows.groupby("station_id", observed=True)[target].sum().idxmax()
        row = rows[rows.station_id == busiest].iloc[[0]]
        e = explanation(model, row[feats])[0]
        pred = float(np.exp(e.base_values + e.values.sum()))
        print(f"\n{key}: station {busiest} at {row.ts_local.iloc[0]}, actual {row[target].iloc[0]}, pred {pred:.2f}")
        for f, v in sorted(zip(feats, e.values), key=lambda t: -abs(t[1]))[:6]:
            print(f"  {f:18s} {v:+.3f}  (value {row[f].iloc[0]})")
        shap.plots.waterfall(e, max_display=10, show=False)
        plt.title(f"{title}\nstation {busiest}, {row.ts_local.iloc[0]:%a %b %d %H:%M}, actual {row[target].iloc[0]}, "
                  f"predicted {pred:.1f} (log-count space)", fontsize=9)
        plt.savefig(f"{FIG}/shap_local_{target}_{key}.png", dpi=120, bbox_inches="tight")
        plt.close("all")


if __name__ == "__main__":
    main(sys.argv[1])
