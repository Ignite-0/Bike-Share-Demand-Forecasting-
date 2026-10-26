"""Dashboard: stations at risk of running empty/full in the next 3 h, recommended bike moves, station forecasts, and
the track record (alerts vs real stockouts, projected vs actual bikes). Reads the API only.
    .venv/Scripts/python -m streamlit run app/streamlit_app.py   (API_URL defaults to localhost:8000)
"""
import os

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")  # Streamlit Cloud: set it as a secret
NY = "America/New_York"
COLORS = {"empty": [214, 39, 40], "full": [31, 119, 180]}

st.set_page_config(page_title="Citi Bike stockout alerts", layout="wide")


@st.cache_data(ttl=60)
def get(path: str):
    r = requests.get(API + path, timeout=120)  # a free host may take ~1 min to wake up
    if r.status_code in (404, 503):
        return None
    r.raise_for_status()
    return r.json()


def local(ts) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(NY).dt.tz_localize(None)


st.title("Citi Bike: stations about to run empty or full")
alerts, rec = get("/alerts"), get("/recommendations")
if alerts is None or rec is None:
    st.info("The API is starting up (it loads the latest published forecasts). Refresh in a minute.")
    st.stop()
a = pd.DataFrame(alerts["stations"])
names = dict(zip(a["name"] + " (" + a["station_id"] + ")", a["station_id"])) if len(a) else {}
live, record = st.tabs(["Live alerts", "Track record"])

with live:
    snap = pd.Timestamp(alerts["snapshot_at"]).tz_convert(NY)
    st.caption(f"Live status at {snap:%a %b %d, %H:%M} New York time · {alerts['renting_stations']} renting stations · "
               "look-ahead: next 3 h · point = expected demand, cautious = high (P90) demand")
    for col, k in zip(st.columns(4), ["empty", "full", "empty_cautious", "full_cautious"]):
        col.metric(k.replace("_", " ").capitalize(), int(a[k].sum()) if len(a) else 0)

    show = st.multiselect("Show", ["empty", "full"], default=["empty", "full"])
    cautious = st.toggle("Include cautious-only alerts", value=True)
    m = a.copy()
    if len(m):
        m["kind"] = m["empty"].map({True: "empty", False: "full"})
        m.loc[~m["empty"] & ~m["full"], "kind"] = m["empty_cautious"].map({True: "empty", False: "full"})
        m["level"] = (m["empty"] | m["full"]).map({True: "point", False: "cautious only"})
        m = m[m["kind"].isin(show) & (cautious | (m["level"] == "point"))]
        m["color"] = [COLORS[k] + [230 if lv == "point" else 90] for k, lv in zip(m["kind"], m["level"])]
        m["bikes_6h"] = m["bikes_6h"].round(1)
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", m, get_position="[lng, lat]", get_fill_color="color", get_radius=60,
                          radius_min_pixels=4, pickable=True)],
        initial_view_state=pdk.ViewState(latitude=40.73, longitude=-73.96, zoom=11),
        tooltip={"text": "{name} ({station_id})\n{kind} risk, {level}\n{bikes} bikes / {slots} slots\nin 6 h: {bikes_6h} bikes"},
        map_style=None,
    ))
    st.caption("Red = empty risk, blue = full risk; solid = point alert, faint = cautious only.")

    st.subheader(f"Recommended moves ({len(rec['moves'])})")
    mv = pd.DataFrame(rec["moves"])
    if len(mv):
        st.dataframe(mv[["from_name", "to_name", "bikes", "km", "reason", "from_id", "to_id"]], hide_index=True,
                     width="stretch")
        st.caption(f"{mv.bikes.sum()} bikes. One move per at-risk station, nearest partner within 2 km that stays 30-70% full.")
    else:
        st.write("No moves needed right now.")

    st.subheader("Station forecast")
    if names:
        pick = st.selectbox("At-risk station", list(names), key="fc_station")
        f = get(f"/forecast/{names[pick]}")
        if f:
            h = pd.DataFrame(f["hours"])
            h["time"] = local(h["ts_utc"])
            st.line_chart(h.set_index("time")[["pu_mean", "pu_p90", "dr_mean", "dr_p90"]]
                          .rename(columns={"pu_mean": "pick-ups", "pu_p90": "pick-ups P90",
                                           "dr_mean": "drop-offs", "dr_p90": "drop-offs P90"}))

with record:
    card = get("/accuracy")
    if not card:
        st.info("The track record isn't published yet.")
        st.stop()
    w = [pd.Timestamp(t).tz_convert(NY) for t in card["scored_window_utc"]]
    st.markdown(f"How the alerts did against **what really happened**, from {w[0]:%b %d %H:%M} to {w[1]:%b %d %H:%M} "
                f"({card['origins']} hourly checks). Ground truth: live station counts recorded every 10 minutes. "
                "Only alerts made after the alert levels were tuned are counted.")
    s = pd.DataFrame(card["summary"])
    for kind, col in zip(("empty", "full"), st.columns(2)):
        r = s[(s.kind == kind)].set_index("variant")
        p, base = r.loc["point"], r.loc["current stock"]
        col.metric(f"Warned to run {kind}: came true", f"{p.precision:.0%}",
                   f"{p.precision - base.precision:+.0%} vs current-stock rule")
        col.metric(f"Ran {kind}: we had warned", f"{p.recall:.0%}", f"{p.recall - base.recall:+.0%} vs current-stock rule")
        col.metric(f"Ran {kind} from a healthy level: caught early", f"{p.caught_early:.0%}",
                   "current-stock rule: 0%", delta_color="off")
    st.caption(card["note"])
    st.dataframe(s.assign(**{c: s[c].map("{:.0%}".format) for c in ("precision", "recall", "caught_early", "f1")}),
                 hide_index=True, width="stretch")

    kind = st.radio("Daily", ["empty", "full"], horizontal=True)
    d = pd.DataFrame(card["daily"])
    d = d[(d.kind == kind) & (d.variant != "cautious")].replace({"variant": {"point": "forecast"}})
    d = d.melt(["date", "variant"], ["precision", "recall"])
    st.line_chart(d.assign(series=d.variant + " " + d.variable).pivot(index="date", columns="series", values="value"))

    st.subheader("Projected vs actual bikes")
    if names:
        pick = st.selectbox("At-risk station", list(names), key="track_station")
        t = get(f"/track/{names[pick]}")
        if t:
            t = pd.DataFrame(t)
            t["time"] = local(t["ts_utc"])
            st.line_chart(t.set_index("time")[["bikes", "proj_1h", "proj_3h"]]
                          .rename(columns={"bikes": "actual bikes", "proj_1h": "projected 1 h before",
                                           "proj_3h": "projected 3 h before"}))
            st.caption("Last 48 h, ending 3 h ago (newer hours aren't scored yet). Citi Bike trucks also move bikes, "
                       "which shows up as jumps the forecast can't know about.")
        else:
            st.write("No track record for this station yet.")
