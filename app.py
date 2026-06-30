# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  Shantanu's Options Analysis Dashboard  Streamlit Edition          ║
║  Data: Dhan API (primary) | Demo Mode (fallback)                    ║
║  NIFTY 50 + NIFTY Futures ONLY                                      ║
║  Bias Score: -100 to +100 | Regime | Strategy Engine               ║
║  v6 — Hardened Edition: CI #2-10 + H1-H26 audit fixes applied      ║
║  All data and calculations are LIVE during market hours             ║
║  (Mon-Fri 09:1515:30 IST). Outside market hours: DEMO/CACHED.      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, json, time, warnings, threading
# ── v4: combined_bias_engine merged inline (no external dependency) ──
from datetime import date, timedelta, datetime
import pytz

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import plotly.graph_objs as go
import plotly.io as _pio

# Force explicit colours for all charts — fixes white labels on iOS Safari / iPad
_pio.templates["_mobile_fix"] = go.layout.Template(
    layout=go.Layout(
        font=dict(color="#1A1A2E"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#F9FAFB",
        xaxis=dict(tickfont=dict(color="#1A1A2E"), title_font=dict(color="#1A1A2E"),
                   linecolor="#E5E7EB", gridcolor="#F3F4F6"),
        yaxis=dict(tickfont=dict(color="#1A1A2E"), title_font=dict(color="#1A1A2E"),
                   linecolor="#E5E7EB", gridcolor="#F3F4F6"),
        legend=dict(font=dict(color="#1A1A2E")),
        hoverlabel=dict(bgcolor="#ffffff", font_color="#1A1A2E"),
        title=dict(font=dict(color="#1A1A2E")),
    )
)
_pio.templates.default = "plotly+_mobile_fix"
import streamlit as st
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")

# ─── Timezone ─────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)

def ist_str(fmt="%d-%m-%Y  %H:%M:%S IST"):
    return now_ist().strftime(fmt)

def is_market_hours():
    n = now_ist()
    return n.weekday() < 5 and (9, 15) <= (n.hour, n.minute) <= (15, 30)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shantanu's Options Dashboard — NIFTY",   # H21 fix: was mojibake (\x97 where em-dash should be)
    page_icon="📊",   # H21 fix: was empty — browser showed default favicon
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CI #5 fix: register auto-refresh IMMEDIATELY after set_page_config, BEFORE any
# st.stop() can fire downstream. The previous location (very end of script) meant
# that if Dhan returned an error → `st.stop()` halted execution →
# `st_autorefresh` never registered → the page would not auto-recover from a
# transient API failure; the user had to manually reload the browser.
# Registering early guarantees the page re-runs every 60s regardless of any
# downstream st.stop() or exception.
#
# H18 FOLLOW-UP (not applied in this pass): migrating from `streamlit_autorefresh`
# to native `st.fragment(run_every=60.0)` (Streamlit ≥1.33) would cut per-rerun
# work by ~80% — only the data-fetch + history-append logic would re-execute,
# not the entire 7000-line script. This is a larger refactor (requires wrapping
# the fetch logic in a fragment function and reading results from
# st.session_state in the main body) and should be done as a separate PR.
# For now, st_autorefresh at the top is the safe fix that preserves all existing
# behavior while solving the st.stop() recovery problem.
st_autorefresh(interval=60_000, key="nifty_autorefresh")


# ─── Credentials ──────────────────────────────────────────────────────────────
def _get_secret(key, default=""):
    try:
        return st.secrets.get(key, default) or os.environ.get(key, default)
    except Exception:
        return os.environ.get(key, default)

DHAN_CLIENT_ID    = _get_secret("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = _get_secret("DHAN_ACCESS_TOKEN")
USE_DHAN          = bool(DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN)
USE_DEMO_MODE     = not USE_DHAN

# ─── Constants ────────────────────────────────────────────────────────────────
APP_TITLE        = "Shantanu's Options Analysis  NIFTY 50"
RISK_FREE_RATE   = 0.065
STRUCTURAL_BAND  = 10
SIGNAL_BAND      = 5
NIFTY_STEP       = 50
NIFTY_LOT_SIZE   = 65   # Current NIFTY F&O lot size — multiplies GEX to match industry standard (OI×Γ×LotSize×S²×0.01)
REFRESH_SECONDS  = 60   # default; overridden at runtime via owner settings

# ─── Section 9 — Δ-Weighted Flow Engine constants (v23-4) ────────────────────
DW_FLOW_DECAY       = 0.85   # Graph 1: decay factor per 15-min bucket (0.90=slow/trend, 0.70=fast/pre-move)
PCR_MIN_BUCKET_OI   = 500    # Graph 3: min lots (abs call + abs put) for a reliable PCR ratio
NORM_WINDOW_BUCKETS = 12     # Graph 1 rolling-window normaliser: buckets to look back (~3 hours)

# ─── Owner sidebar  PIN-protected advanced controls ──────────────────────────
# Dashboard is publicly readable. Owner PIN unlocks expiry, refresh, manual reload.
_REFRESH_OPTIONS = {
    "⚡ 1 min   Live market":       60,
    " 5 min   Active monitoring": 300,
    " 15 min  Slow watch":        900,
    " 30 min  Market closed":     1800,
    " 1 hour  Weekend / idle":    3600,
}

def _render_owner_sidebar(expiry_list):
    """
    Renders the sidebar for all visitors.
    - Locked state  → PIN entry form only.
    - Unlocked state → expiry selector, refresh interval, manual refresh button.
    Returns (sel_expiry, manual_refresh_clicked).

    CI #3 fix: if OWNER_PIN secret is not configured, the unlock form is hidden
    entirely (fail-closed) — there is NO default PIN. A misconfigured deploy
    cannot expose owner controls to public visitors.
    """
    import hmac
    correct_pin = _get_secret("OWNER_PIN", None)   # CI #3 fix: was "12345"

    if "owner_unlocked" not in st.session_state:
        st.session_state.owner_unlocked = False

    # CI #3 fix: brute-force lockout — 5 failed attempts → 5-min cooldown
    if "owner_pin_fail_count" not in st.session_state:
        st.session_state.owner_pin_fail_count = 0
    if "owner_pin_lock_until" not in st.session_state:
        st.session_state.owner_pin_lock_until = 0.0

    # Smart default for refresh (set once, survives reruns)
    if "refresh_seconds" not in st.session_state:
        _n = now_ist()
        _mh = _n.weekday() < 5 and (9, 15) <= (_n.hour, _n.minute) <= (15, 30)
        st.session_state.refresh_seconds = 60 if _mh else 1800

    sel_expiry            = None
    manual_refresh_clicked = False

    with st.sidebar:
        st.markdown("### ⚙️ Owner Controls")

        # CI #3 fix: if PIN isn't configured, hide the entire owner section.
        if correct_pin is None:
            st.caption("🔒 Owner controls disabled (no OWNER_PIN configured).")
            st.caption(" Dashboard is in **read-only** mode for all visitors.")
            return sel_expiry, manual_refresh_clicked

        if not st.session_state.owner_unlocked:
            # CI #3 fix: brute-force lockout check
            _now_ts = time.time()
            if _now_ts < st.session_state.owner_pin_lock_until:
                _mins_left = int((st.session_state.owner_pin_lock_until - _now_ts) / 60) + 1
                st.warning(f"⏱️ Too many failed attempts. Try again in {_mins_left} min.")
            else:
                st.caption("Enter PIN to access advanced settings.")
                pin = st.text_input("Owner PIN", type="password",
                                    key="owner_pin_input", placeholder="Enter owner PIN")
                if st.button("Unlock", width='stretch', type="primary", key="owner_unlock_btn"):
                    # CI #3 fix: constant-time comparison (timing-attack resistance)
                    if pin and hmac.compare_digest(pin, correct_pin):
                        st.session_state.owner_unlocked = True
                        st.session_state.owner_pin_fail_count = 0
                        st.rerun()
                    else:
                        st.session_state.owner_pin_fail_count += 1
                        if st.session_state.owner_pin_fail_count >= 5:
                            st.session_state.owner_pin_lock_until = _now_ts + 300   # 5 min
                            st.session_state.owner_pin_fail_count = 0
                            st.error("❌ Too many failed attempts. Locked for 5 minutes.")
                        else:
                            st.error(f"❌ Incorrect PIN (attempt "
                                     f"{st.session_state.owner_pin_fail_count}/5)")
            st.divider()
            st.caption(" Dashboard is in **read-only** mode for guests.")

        else:
            # ── Unlocked: show advanced controls ──────────────────────────────
            st.success(" Owner mode")

            # Expiry selector
            _cur_settings = _load_owner_settings()
            if expiry_list:
                _saved_expiry = _cur_settings.get("selected_expiry") or expiry_list[0]
                _exp_idx = expiry_list.index(_saved_expiry) if _saved_expiry in expiry_list else 0
                sel_expiry = st.selectbox(" Expiry", expiry_list,
                                          index=_exp_idx, key="owner_expiry")
                # Persist if owner changed expiry
                if sel_expiry != _cur_settings.get("selected_expiry"):
                    _cur_settings["selected_expiry"] = sel_expiry
                    _save_owner_settings(_cur_settings)
                    _force_server_refresh()   # expire cache so new expiry is fetched next cycle
            else:
                sel_expiry = None
                st.caption("Expiry: Auto (nearest)")

            st.divider()

            # Refresh interval  (data refresh, NOT page refresh)
            _saved_interval = _cur_settings.get("refresh_interval", REFRESH_SECONDS)
            st.session_state.refresh_seconds = _saved_interval
            current_label = next(
                (k for k, v in _REFRESH_OPTIONS.items()
                 if v == _saved_interval),
                list(_REFRESH_OPTIONS.keys())[0]
            )
            chosen = st.selectbox(
                " Data refresh interval",
                list(_REFRESH_OPTIONS.keys()),
                index=list(_REFRESH_OPTIONS.keys()).index(current_label),
                key="refresh_selector",
            )
            new_interval = _REFRESH_OPTIONS[chosen]
            if new_interval != _saved_interval:
                _cur_settings["refresh_interval"] = new_interval
                _save_owner_settings(_cur_settings)
                st.session_state.refresh_seconds = new_interval
            mins = new_interval // 60
            st.info(f"Data refresh: **{new_interval}s** ({mins} min)\nPage refresh: **60s** (always)")

            st.divider()

            # Vega band width selector
            # Controls how many strikes either side of ATM are included in the
            # OI-weighted band vega sum (ATM Vega Diff chart + Net Vega per Strike).
            # ±1 = tight (expiry-day); ±2 = default; ±3/4 = monthlies with wide smile.
            _VEGA_BAND_OPTIONS = {"±1 strike  (50 pts)": 1,
                                  "±2 strikes (100 pts) — default": 2,
                                  "±3 strikes (150 pts)": 3,
                                  "±4 strikes (200 pts)": 4}
            _saved_vb = _cur_settings.get("vega_band_strikes", 2)
            _vb_label = next((k for k, v in _VEGA_BAND_OPTIONS.items() if v == _saved_vb),
                             "±2 strikes (100 pts) — default")
            _chosen_vb = st.selectbox(
                "📐 ATM Vega band width",
                list(_VEGA_BAND_OPTIONS.keys()),
                index=list(_VEGA_BAND_OPTIONS.keys()).index(_vb_label),
                key="vega_band_selector",
                help="Number of strikes either side of ATM used for the OI-weighted "
                     "Vega Diff chart. Wider = smoother but includes more OTM noise.",
            )
            _new_vb = _VEGA_BAND_OPTIONS[_chosen_vb]
            if _new_vb != _saved_vb:
                _cur_settings["vega_band_strikes"] = _new_vb
                _save_owner_settings(_cur_settings)
            st.caption(f"Band: ATM ± {_new_vb} × 50 = ±{_new_vb * NIFTY_STEP} pts")

            st.divider()

            # Manual refresh  — OWNER ONLY
            if st.button("⟳ Refresh Now", width='stretch',
                         type="primary", key="owner_refresh_btn"):
                manual_refresh_clicked = True

            st.divider()
            if st.button(" Lock", key="owner_lock_btn", width='stretch'):
                st.session_state.owner_unlocked = False
                # CI #3 fix: clear the PIN input so the next visitor can't re-unlock
                # by clicking Unlock without re-entering the PIN.
                if "owner_pin_input" in st.session_state:
                    del st.session_state["owner_pin_input"]
                st.rerun()

    return sel_expiry, manual_refresh_clicked

# Resolve effective data-refresh  reads owner-controlled value from server settings.
# NOTE: page auto-refresh is always 60 s for ALL visitors regardless of this value.
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nifty_settings.json"), "r") as _sf:
        _effective_refresh = json.load(_sf).get("refresh_interval", REFRESH_SECONDS)
except Exception:
    _effective_refresh = REFRESH_SECONDS

SYMBOL           = "NIFTY"   # NIFTY 50 only

DHAN_SECURITY    = {"NIFTY": {"id": 13, "seg": "IDX_I"}}

# Bias weights  identical to Dash app
BIAS_WEIGHTS = {
    "net_delta":               16,
    "momentum":                22,
    "ev_ratio":                10,
    "atm_pressure":            10,
    "skew_slope":               8,
    "vanna":                    4,
    "regime_range":            22,
    "regime_trend":            25,
    "regime_transition":       10,
    "near_oi_concentration":   12,
    "near_oichg_concentration":12,
    "wall_proximity":           8,
    "wall_shift":               8,
    "max_pain_drift":           6,
    "range_compression":        6,
    "expansion_building":       6,
    "persistence":             10,
    "skew_slope_threshold":  0.15,
    "ev_ratio_bull":         1.05,
    "ev_ratio_bear":         0.95,
    "momentum_gex_threshold":500,
    "near_oi_min":           0.55,
    "near_oichg_min":        0.50,
    "wall_proximity_pts":      75,
    "bias_bull_threshold":     15,
    "bias_bear_threshold":    -15,
    "confidence_min_strategy": 35,
}

METRIC_EXPLAIN = {
    "Bias Score":      "Hedge-flow bias score (-100..+100) from the legacy compute_nifty_bias engine — uses SIGNED delta x OI (net_delta), so it reads dealer hedge-flow pressure, not writer positioning. Use the S3/4 / Combined Decision panels for the authoritative directional call.",
    "Confidence":      "Signal quality score based on regime, persistence, concentration, and wall behavior.",
    "Regime":          "Range/pin, trend/expansion, or transition inferred from gamma, IV, walls, and persistence.",
    "EV Ratio":        "Call vs put time value near spot; higher means call premium stronger, lower means put premium stronger.",
    "Net Delta":       "Directional lean from near-spot open interest weighted by delta.",
    "Momentum":        "Fresh intraday open-interest change weighted by delta near spot.",
    "GEX":             "Gamma exposure from the structural band; positive tends to pin, negative tends to expand moves.",
    "PCR":             "Put-call ratio from the structural band.",
    "G/T Ratio":       "Gamma-to-theta ratio; higher values often align with unstable or directional conditions.",
    "ATM Pressure":    "Near-ATM put OI change minus call OI change.",
    "Skew Slope":      "Difference between downside put-IV slope and upside call-IV slope.",
    "Near OI %":       "Share of structural-band OI concentrated near ATM.",
    "Near OI Chg %":   "Share of fresh OI activity concentrated near ATM.",
    "Wall Width":      "Distance between the strongest put wall and strongest call wall.",
    "Max Pain":        "Strike where option writers lose the least money.",
    "Support":         "Strongest put OI wall in the structural band.",
    "Resistance":      "Strongest call OI wall in the structural band.",
    "IV Rank":         "ATM IV rank within smile range (0=low, 100=high).",
    "Gamma Flip":      "Strike where cumulative GEX turns zero. Spot below flip = trend amplification zone.",
}

# ─── Colours ──────────────────────────────────────────────────────────────────
GREEN  = "#059669"
RED    = "#DC2626"
AMBER  = "#D97706"
BLUE   = "#2563EB"
CYAN   = "#0891B2"
PINK   = "#9333EA"
MUTED  = "#6B7280"
ACCENT = "#5C35CC"
TEXT   = "#1A1A2E"
GOLD   = "#B45309"

# ─── Black-Scholes helpers  EXACT same as Dash app ──────────────────────────
def _bs_price(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_greeks(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    delta = norm.cdf(d1) if opt == "CE" else -norm.cdf(-d1)
    gamma = nd1 / (S * sigma * np.sqrt(T))
    if opt == "CE":
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) -
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    vega = S * nd1 * np.sqrt(T) / 100
    return delta, gamma, theta, vega


def _solve_iv(mkt_price, S, K, T, r, opt):
    if T <= 0 or mkt_price <= 0 or S <= 0 or K <= 0:
        return np.nan
    try:
        return brentq(
            lambda v: (_bs_price(S, K, T, r, v, opt) - mkt_price),
            1e-4, 5.0, xtol=1e-5, maxiter=100
        )
    except Exception:
        return np.nan


def safe_num(x, d=0.0):
    """H5 fix: also reject ±inf (was only None / NaN before).

    ±inf in metrics propagates to json.dump which (with default allow_nan=True)
    writes the literal `Infinity` / `NaN` token — invalid per RFC 8259 and
    unparseable by non-Python tools. Combined with allow_nan=False in
    _atomic_json_write (CI #6), this guarantees persisted history stays clean.
    """
    try:
        if x is None:
            return d
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return d
        return v
    except (TypeError, ValueError):
        return d


# ─── GEX + IV Rank + Gamma Regime  IDENTICAL to Dash app ────────────────────
def compute_true_gex(df, spot):
    """Standard (unweighted) GEX per strike.

    Formula matches industry standard (Perfiliev / SpotGamma / StockMojo):
      GEX = OI × Gamma × LotSize × Spot² × 0.01
    Calls add to GEX (dealers long gamma → pinning).
    Puts subtract from GEX (dealers short gamma → amplifying).
    Gaussian weighting removed — no tool uses it; it made the headline
    metric incomparable to any published Indian options tool.
    """
    if df is None or df.empty:
        return 0.0, pd.Series(dtype=float), None
    strikes  = df["strike"].values
    call_arr = df["call_oi"].values * df["call_gamma"].values * NIFTY_LOT_SIZE * (spot ** 2) * 0.01
    put_arr  = df["put_oi"].values  * df["put_gamma"].values  * NIFTY_LOT_SIZE * (spot ** 2) * 0.01
    net_arr  = call_arr - put_arr
    total_gex  = float(net_arr.sum())
    gex_series = pd.Series(net_arr, index=strikes)
    cumulative = gex_series.sort_index().cumsum()
    flip_cands = cumulative[cumulative <= 0].index
    gamma_flip = float(flip_cands[-1]) if len(flip_cands) > 0 else None
    return total_gex, gex_series, gamma_flip


def compute_iv_rank(df, atm):
    """Cross-sectional IV rank within the current smile.

    Returns (smile_position_pct, iv_pct). This is a SMILE-POSITION metric, NOT a
    temporal IV rank — for a normal NIFTY smile where ATM sits near the IV
    minimum, this value is close to 0 most of the time. Use compute_temporal_iv_rank()
    for any vol-regime / strategy decision that depends on whether IV is
    historically high or low.
    """
    if df is None or df.empty:
        return 0.0, 0.0
    avg_iv = ((df["call_iv"].replace(0, np.nan) + df["put_iv"].replace(0, np.nan)) / 2).dropna()
    if avg_iv.empty:
        return 0.0, 0.0
    row = df[df["strike"] == atm]
    if row.empty:
        row = df.iloc[[(df["strike"] - atm).abs().idxmin()]]
    # CI #2 fix part A: robust non-zero average for ATM IV (was biased by zeros).
    _c = safe_num(row["call_iv"].iloc[0])
    _p = safe_num(row["put_iv"].iloc[0])
    if _c > 0 or _p > 0:
        atm_iv = (_c + _p) / max(1, (_c > 0) + (_p > 0))
    else:
        atm_iv = 0.0
    iv_min, iv_max = float(avg_iv.min()), float(avg_iv.max())
    if iv_max <= iv_min:
        return 0.0, 0.0
    iv_rank = round((atm_iv - iv_min) / (iv_max - iv_min) * 100, 1)
    iv_pct  = round(float((avg_iv <= atm_iv).mean()) * 100, 1)
    return iv_rank, iv_pct


# ── CI #2 fix: True temporal IV rank from persisted history ─────────────────
# `nifty_history.json` records `atm_iv` per tick (see build_history_entry, L3239).
# We compute a trailing-N-day rank: rank = (now - min) / (max - min) * 100.
# Falls back to the cross-sectional smile_position (with a flag) if history is
# too short or contains no IV variation.
TEMPORAL_IVR_LOOKBACK_DAYS = 20    # ~1 trading month
TEMPORAL_IVR_MIN_SAMPLES   = 10   # need at least this many distinct days

def compute_temporal_iv_rank(current_atm_iv, history=None):
    """
    Returns (temporal_iv_rank, is_temporal).
      temporal_iv_rank: 0-100 rank of current_atm_iv vs trailing ~20d window.
      is_temporal:      True if computed from real history,
                        False if fell back to None (caller should use cross-sectional).
    """
    if current_atm_iv is None or current_atm_iv <= 0:
        return None, False
    if not history or len(history) < 2:
        return None, False

    # Build a series of (date, atm_iv) from history, deduplicating by date
    # (one IV value per calendar day, taking the last sample of that day).
    by_date = {}
    for h in history:
        iv = safe_num(h.get("atm_iv", 0))
        if iv <= 0:
            continue
        ts = h.get("ts", "")
        # ts format: "YYYY-MM-DDTHH:MM:SS"
        d = ts.split("T")[0] if "T" in ts else ts[:10]
        if d:
            by_date[d] = iv   # later entries overwrite earlier — keeps last sample of day

    if len(by_date) < TEMPORAL_IVR_MIN_SAMPLES:
        return None, False

    # Take the trailing N days (sorted by date)
    sorted_dates = sorted(by_date.keys())
    window_dates = sorted_dates[-TEMPORAL_IVR_LOOKBACK_DAYS:]
    window_ivs   = [by_date[d] for d in window_dates]

    iv_min = min(window_ivs)
    iv_max = max(window_ivs)
    if iv_max - iv_min < 0.5:    # <0.5 vol-point spread — insufficient variation
        return None, False

    rank = round((current_atm_iv - iv_min) / (iv_max - iv_min) * 100, 1)
    rank = max(0.0, min(100.0, rank))
    return rank, True


def classify_gamma_regime(gex, wall_width, momentum, atm_iv, iv_rank, spot, gamma_flip):
    # H13 fix: convert absolute-point thresholds to % of spot. As NIFTY drifts
    # (22k → 25k), 300 pts went from 1.36% to 1.20% of spot — the regime
    # classifier became arbitrarily stricter simply because the index rose.
    # Now: 300 pts ≈ 1.3% band; 400 pts ≈ 1.7% band; 500 momentum units stays
    # absolute (it's an OI-delta unit, not a price unit).
    spot_pct = lambda abs_pts: (abs_pts / max(spot, 1)) * 100.0   # converts pts → % of spot
    flip_dist = abs(spot - gamma_flip) if gamma_flip is not None else 9999
    _strike_step = max(wall_width / 20, 50) if wall_width > 0 else 50
    near_flip = flip_dist < max(2.0 * _strike_step, 100) if gamma_flip is not None else False
    if iv_rank >= 70:   vol_regime = "HIGH_VOL"
    elif iv_rank <= 30: vol_regime = "LOW_VOL"
    else:               vol_regime = "MID_VOL"
    wall_width_pct = spot_pct(wall_width)
    if   gex > 0 and wall_width_pct <= 1.3 and vol_regime == "LOW_VOL":
        return "PINNED / RANGE",       vol_regime, near_flip
    elif gex > 0 and wall_width_pct <= 1.7:
        return "RANGE / PIN",          vol_regime, near_flip
    elif gex < 0 and abs(momentum) > 500 and vol_regime in ("MID_VOL", "HIGH_VOL"):
        return "TREND / EXPANSION",    vol_regime, near_flip
    elif near_flip:
        return "FLIP ZONE / UNSTABLE", vol_regime, near_flip
    else:
        return "TRANSITION",           vol_regime, near_flip


# ─── Data fetchers ────────────────────────────────────────────────────────────
# H1+H2+H3 fix: shared Dhan HTTP helper. Centralizes:
#   - status-code checking via raise_for_status (H1)
#   - status=="success" body validation (H2)
#   - retry+backoff via urllib3 Retry (H3) — 3 retries, exponential backoff,
#     honors 429 / 5xx. Mounted on a module-level Session so connection pooling
#     works across fetchers.
import requests as _requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except ImportError:                                # pragma: no cover
    Retry = None

_dhan_session = None
_dhan_session_lock = threading.Lock()

def _get_dhan_session():
    """H3 fix: return a Session with retry/backoff mounted. Thread-safe singleton."""
    global _dhan_session
    if _dhan_session is not None:
        return _dhan_session
    with _dhan_session_lock:
        if _dhan_session is None:
            s = _requests.Session()
            if Retry is not None:
                retry = Retry(
                    total=3,
                    backoff_factor=0.5,            # 0.5, 1, 2 seconds
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=("GET", "POST"),
                    respect_retry_after_header=True,
                )
                adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
                s.mount("https://", adapter)
                s.mount("http://",  adapter)
            _dhan_session = s
    return _dhan_session


class DhanAPIError(Exception):
    """Raised when Dhan returns a non-success response body or HTTP error."""
    pass


def _dhan_post(url, payload, timeout=15):
    """H1+H2+H3 fix: POST to a Dhan endpoint with full validation + retry.

    Returns the parsed JSON body on success.
    Raises DhanAPIError on HTTP error, non-success body, or network failure.
    """
    if not USE_DHAN:
        raise DhanAPIError("Dhan credentials not configured")
    sec = DHAN_SECURITY["NIFTY"]
    headers = {
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id":    str(DHAN_CLIENT_ID),
        "Content-Type": "application/json",
    }
    sess = _get_dhan_session()
    try:
        resp = sess.post(url, headers=headers, json=payload, timeout=timeout)
    except _requests.RequestException as e:
        raise DhanAPIError(f"network error: {e}")

    # H1 fix: check HTTP status code (was missing — JSONDecodeError was silently swallowed)
    if resp.status_code >= 400:
        # Try to extract error message from body, fall back to status text
        try:
            body = resp.json()
            msg = body.get("errorMessage") or body.get("message") or body
        except Exception:
            msg = resp.text[:200] if resp.text else resp.reason
        raise DhanAPIError(f"HTTP {resp.status_code}: {msg}")

    try:
        data = resp.json()
    except ValueError as e:
        raise DhanAPIError(f"invalid JSON response: {e}")

    # H2 fix: validate status=="success" envelope (only for /v2 endpoints;
    # the scrip-master CSV is fetched separately via _load_dhan_instrument_master)
    if "status" in data and data.get("status") != "success":
        err = data.get("errorMessage") or data.get("errorCode") or data.get("status")
        raise DhanAPIError(f"Dhan status={data.get('status')}: {err}")

    return data


def _dhan_get_csv(url, timeout=25):
    """H1+H3 fix: GET a CSV (used for the instrument master). Returns the text."""
    sess = _get_dhan_session()
    try:
        resp = sess.get(url, timeout=timeout)
        resp.raise_for_status()           # H1 fix
        return resp.text
    except _requests.RequestException as e:
        raise DhanAPIError(f"network error fetching CSV: {e}")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_dhan_expiry_list():
    if not USE_DHAN:
        return []
    sec = DHAN_SECURITY["NIFTY"]
    try:
        data = _dhan_post(
            "https://api.dhan.co/v2/optionchain/expirylist",
            {"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"]},
            timeout=15,
        )
        expiries = data.get("data", []) or []
        today = date.today().isoformat()
        return [e for e in expiries if e >= today]
    except (DhanAPIError, Exception) as e:
        # H1+M9 fix: log the exception (was silently swallowed)
        try:
            print(f"[fetch_dhan_expiry_list] error: {e}", flush=True)
        except Exception:
            pass
        return []


def fetch_dhan_option_chain(expiry=None):
    if not USE_DHAN:
        return pd.DataFrame(), 0.0, ""

    sec = DHAN_SECURITY["NIFTY"]

    if expiry is None:
        try:
            exp_data = _dhan_post(
                "https://api.dhan.co/v2/optionchain/expirylist",
                {"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"]},
                timeout=15,
            )
            expiries = exp_data.get("data", []) or []
            today = date.today().isoformat()
            future = [e for e in expiries if e >= today]
            expiry = future[0] if future else ""
        except (DhanAPIError, Exception) as e:
            try:
                print(f"[fetch_dhan_option_chain] expiry list error: {e}", flush=True)
            except Exception:
                pass
            return pd.DataFrame(), 0.0, ""

    if not expiry:
        return pd.DataFrame(), 0.0, ""

    try:
        resp = _dhan_post(
            "https://api.dhan.co/v2/optionchain",
            {"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"], "Expiry": expiry},
            timeout=20,
        )
    except (DhanAPIError, Exception) as e:
        try:
            print(f"[fetch_dhan_option_chain] chain fetch error: {e}", flush=True)
        except Exception:
            pass
        return pd.DataFrame(), 0.0, expiry

    data = resp.get("data", {}) or {}
    spot = float(data.get("last_price") or data.get("lastPrice") or
                 data.get("last_traded_price") or data.get("ltp") or 0)

    oc = data.get("oc", {}) or {}
    rows = []
    # H6 fix: use int(float(...)) for OI parsing. Previously int(ce.get("oi", 0) or 0)
    # raised ValueError on schema drift (e.g., "12345.0" string), discarding the
    # entire option chain for that cycle.
    def _safe_int(v):
        try:
            return int(float(v or 0))
        except (TypeError, ValueError):
            return 0

    for strike_str, chain in oc.items():
        K = safe_num(strike_str, 0)
        ce = (chain or {}).get("ce", {}) or {}
        pe = (chain or {}).get("pe", {}) or {}
        cg = ce.get("greeks", {}) or {}
        pg = pe.get("greeks", {}) or {}
        rows.append({
            "strike": K,
            "call_ltp": safe_num(ce.get("last_price", 0)),
            "call_oi": _safe_int(ce.get("oi", 0)),                       # H6 fix
            "call_prev_oi": _safe_int(ce.get("previous_oi", 0)),         # H6 fix
            "call_oi_chg": _safe_int(ce.get("oi", 0)) - _safe_int(ce.get("previous_oi", 0)),
            "call_vol": _safe_int(ce.get("volume", 0)),                  # H6 fix
            "call_bid": safe_num(ce.get("top_bid_price", 0)),
            "call_ask": safe_num(ce.get("top_ask_price", 0)),
            "call_iv": safe_num(ce.get("implied_volatility", 0)),
            "call_delta": safe_num(cg.get("delta", 0)),
            "call_gamma": safe_num(cg.get("gamma", 0)),
            "call_theta": safe_num(cg.get("theta", 0)),
            "call_vega": safe_num(cg.get("vega", 0)),
            "put_ltp": safe_num(pe.get("last_price", 0)),
            "put_oi": _safe_int(pe.get("oi", 0)),                        # H6 fix
            "put_prev_oi": _safe_int(pe.get("previous_oi", 0)),          # H6 fix
            "put_oi_chg": _safe_int(pe.get("oi", 0)) - _safe_int(pe.get("previous_oi", 0)),
            "put_vol": _safe_int(pe.get("volume", 0)),                   # H6 fix
            "put_bid": safe_num(pe.get("top_bid_price", 0)),
            "put_ask": safe_num(pe.get("top_ask_price", 0)),
            "put_iv": safe_num(pe.get("implied_volatility", 0)),
            "put_delta": safe_num(pg.get("delta", 0)),
            "put_gamma": safe_num(pg.get("gamma", 0)),
            "put_theta": safe_num(pg.get("theta", 0)),
            "put_vega": safe_num(pg.get("vega", 0)),
        })

    if spot == 0 and oc:
        # H8 fix: use proper put-call parity S = C - P + K (median across
        # strikes with both sides having positive LTP). The previous logic
        # returned the strike K where |C-P| was smallest, which is just the
        # ATM strike (rounded to NIFTY_STEP) — a 17-pt error at spot 24517.
        try:
            spot_estimates = []
            for strike_str, chain in oc.items():
                K = safe_num(strike_str, 0)
                ce_ltp = safe_num((chain or {}).get("ce", {}).get("last_price", 0))
                pe_ltp = safe_num((chain or {}).get("pe", {}).get("last_price", 0))
                if ce_ltp > 0 and pe_ltp > 0 and K > 0:
                    # put-call parity: C - P = S - K*exp(-rT) ≈ S - K for short T
                    # → S ≈ C - P + K
                    spot_estimates.append(ce_ltp - pe_ltp + K)
            if spot_estimates:
                # Use median to be robust against wide-spread / illiquid strikes
                spot = float(np.median(spot_estimates))
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), 0.0, expiry
    df = df.sort_values("strike").reset_index(drop=True)

    # BS greek backfill when Dhan returns sparse greeks
    try:
        _T = max((datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days, 0) / 365.0
        _r = RISK_FREE_RATE
        for i, row in df.iterrows():
            _K = row["strike"]
            if row["call_delta"] == 0 and row["call_iv"] > 0.5 and spot > 0 and _T > 0:
                _d, _g, _th, _ve = _bs_greeks(spot, _K, _T, _r, row["call_iv"] / 100.0, "CE")
                df.at[i, "call_delta"] = _d
                df.at[i, "call_gamma"] = _g
                df.at[i, "call_theta"] = _th
                df.at[i, "call_vega"]  = _ve
            if row["put_delta"] == 0 and row["put_iv"] > 0.5 and spot > 0 and _T > 0:
                _d, _g, _th, _ve = _bs_greeks(spot, _K, _T, _r, row["put_iv"] / 100.0, "PE")
                df.at[i, "put_delta"] = _d
                df.at[i, "put_gamma"] = _g
                df.at[i, "put_theta"] = _th
                df.at[i, "put_vega"]  = _ve
    except Exception:
        pass

    return df, spot, expiry


# CI #8 fix: cached wrapper around fetch_dhan_option_chain for callers that
# fetch the BACK expiry every 60s visitor refresh (e.g., the v4 #6 Inter-Expiry
# Roll Signal at ~L4630). The bare function has no caching — every visitor
# triggered a fresh Dhan POST, violating the ~1-req/3s rate limit and doubling
# API spend. 5-min TTL is sufficient because roll-detection doesn't need
# per-minute granularity.
@st.cache_data(ttl=300, show_spinner=False)
def fetch_dhan_option_chain_cached(expiry=None):
    return fetch_dhan_option_chain(expiry)


def fetch_demo_option_chain():
    np.random.seed(int(time.time()) // 60)
    step = NIFTY_STEP
    base_spot = 24500
    spot = base_spot * (1 + np.random.normal(0, 0.003))
    atm = round(spot / step) * step
    strikes = np.arange(atm - 15 * step, atm + 16 * step, step)
    T = 3 / 365.0
    r = RISK_FREE_RATE
    base_iv = 14.0 + np.random.normal(0, 1.5)
    rows = []
    for K in strikes:
        mono = (K - spot) / max(spot, 1)
        iv_c = max(0.05, (base_iv / 100) + 0.015 * mono**2 + abs(mono) * 0.04 + np.random.normal(0, 0.005))
        iv_p = max(0.05, (base_iv / 100) + 0.025 * mono**2 - mono * 0.03 + np.random.normal(0, 0.005))
        c_price = _bs_price(spot, K, T, r, iv_c, "CE")
        p_price = _bs_price(spot, K, T, r, iv_p, "PE")
        cd, cg, ct, cv = _bs_greeks(spot, K, T, r, iv_c, "CE")
        pd2, pg, pt, pv = _bs_greeks(spot, K, T, r, iv_p, "PE")
        put_oi_fac  = max(0.2, 2 - mono * 12) * np.random.lognormal(0, 0.4)
        call_oi_fac = max(0.2, 2 + mono * 10) * np.random.lognormal(0, 0.4)
        call_oi = int(max(500, call_oi_fac * 40000))
        put_oi  = int(max(500, put_oi_fac  * 55000))
        call_prev_oi = max(0, call_oi - int(np.random.normal(800, 4000)))
        put_prev_oi  = max(0, put_oi  - int(np.random.normal(-300, 4500)))
        rows.append({
            "strike": float(K),
            "call_ltp": round(max(0.05, c_price + np.random.normal(0, 0.3)), 2),
            "call_oi": call_oi, "call_prev_oi": call_prev_oi,
            "call_oi_chg": call_oi - call_prev_oi,
            "call_vol": int(abs(np.random.normal(15000, 8000))),
            "call_bid": round(max(0.05, c_price - 0.25), 2),
            "call_ask": round(c_price + 0.25, 2),
            "call_iv": round(iv_c * 100, 2),
            "call_delta": round(cd, 4), "call_gamma": round(cg, 6),
            "call_theta": round(ct, 4), "call_vega": round(cv, 4),
            "put_ltp": round(max(0.05, p_price + np.random.normal(0, 0.3)), 2),
            "put_oi": put_oi, "put_prev_oi": put_prev_oi,
            "put_oi_chg": put_oi - put_prev_oi,
            "put_vol": int(abs(np.random.normal(18000, 9000))),
            "put_bid": round(max(0.05, p_price - 0.25), 2),
            "put_ask": round(p_price + 0.25, 2),
            "put_iv": round(iv_p * 100, 2),
            "put_delta": round(pd2, 4), "put_gamma": round(pg, 6),
            "put_theta": round(pt, 4), "put_vega": round(pv, 4),
        })
    expiry = (date.today() + timedelta(days=3)).strftime("%Y-%m-%d")
    return pd.DataFrame(rows), round(spot, 2), expiry


def get_option_chain(expiry=None):
    """CI #10 fix: distinguish "API error" from "no credentials / demo mode".

    Previously: any transient Dhan failure (timeout, 429, 5xx) returned an empty
    DataFrame, which silently triggered `fetch_demo_option_chain` and was labeled
    "DEMO MODE (No API credentials)". Users saw fabricated spot/OI/IV values
    presented as if their credentials were missing — a data-integrity / trust
    issue.

    Now: returns one of three source strings:
      - " LIVE — Dhan API"           : successful live fetch
      - " API ERROR — using stale demo data"  : live fetch failed, fell back to demo
                                                (clearly labeled as NOT real data)
      - " DEMO MODE (No API credentials)"     : USE_DHAN is False (genuine demo)
    """
    if USE_DHAN:
        df, spot, exp = fetch_dhan_option_chain(expiry)
        if not df.empty:
            return df, spot, exp, " LIVE — Dhan API"
        # Live fetch returned empty — this is an API ERROR, not "no credentials".
        # Fall back to demo data (so the dashboard remains visually populated)
        # but tag the source so the banner can surface a clear warning.
        df, spot, exp = fetch_demo_option_chain()
        return df, spot, exp, " API ERROR — using stale demo data"
    # USE_DHAN is False — genuine demo mode (no credentials configured)
    df, spot, exp = fetch_demo_option_chain()
    return df, spot, exp, " DEMO MODE (No API credentials)"


# ─── NIFTY Futures LTP ────────────────────────────────────────────────────────
_fut_master_df   = None
_fut_id_cache    = {}
_fut_master_lock = threading.Lock()

def _load_dhan_instrument_master():
    global _fut_master_df
    with _fut_master_lock:
        if _fut_master_df is not None:
            return _fut_master_df
        try:
            import io
            # H1+H3 fix: use the shared session (with retry/backoff) via _dhan_get_csv.
            csv_text = _dhan_get_csv("https://images.dhan.co/api-data/api-scrip-master.csv", timeout=25)
            df = pd.read_csv(io.StringIO(csv_text), low_memory=False)
            df.columns = [c.strip().upper() for c in df.columns]
            _fut_master_df = df
            return _fut_master_df
        except (DhanAPIError, Exception) as e:
            try:
                print(f"[_load_dhan_instrument_master] error: {e}", flush=True)
            except Exception:
                pass
            return None

def _resolve_futures_id(near_expiry_str=None):
    today = date.today()
    master = _load_dhan_instrument_master()
    if master is None:
        return None, None
    try:
        cols  = set(master.columns)
        exch_col  = next((c for c in ["SEM_EXM_EXCH_ID"] if c in cols), None)
        instr_col = next((c for c in ["SEM_INSTRUMENT_NAME"] if c in cols), None)
        tsym_col  = next((c for c in ["SEM_TRADING_SYMBOL","SM_SYMBOL_NAME"] if c in cols), None)
        expdt_col = next((c for c in ["SEM_EXPIRY_DATE","SM_EXPIRY_DATE"] if c in cols), None)
        secid_col = next((c for c in ["SEM_SMST_SECURITY_ID","SEM_SECURITY_ID"] if c in cols), None)
        if not all([tsym_col, expdt_col, secid_col]):
            return None, None
        df = master.copy()
        if exch_col:
            df = df[df[exch_col].astype(str).str.strip().str.upper() == "NSE"]
        if instr_col:
            df = df[df[instr_col].astype(str).str.strip().str.upper().isin(["FUTIDX"])]
        tsym_s = df[tsym_col].astype(str).str.strip().str.upper()
        mask = tsym_s.str.startswith("NIFTY") & tsym_s.str.endswith("FUT")
        df = df[mask].copy()
        if df.empty:
            return None, None
        df["_expdt"] = pd.to_datetime(df[expdt_col], dayfirst=True, errors="coerce").dt.date
        df = df[df["_expdt"] >= today].sort_values("_expdt")
        if df.empty:
            return None, None
        chosen = df.iloc[0]
        return str(int(float(chosen[secid_col]))), chosen["_expdt"]
    except Exception:
        return None, None

_fut_ltp_cache = {"ltp": 0.0, "ts": 0.0}
_FUT_CACHE_SEC = 58

def fetch_futures_ltp(near_expiry_str=None):
    if not USE_DHAN:
        return 0.0
    now = time.time()
    if now - _fut_ltp_cache["ts"] < _FUT_CACHE_SEC:
        return _fut_ltp_cache["ltp"]
    cached = _fut_id_cache.get("NIFTY")
    if not cached:
        sec_id, exp_dt = _resolve_futures_id(near_expiry_str)
        if not sec_id:
            return 0.0
        _fut_id_cache["NIFTY"] = {"id": sec_id, "expiry": exp_dt}
        cached = _fut_id_cache["NIFTY"]
    try:
        # H1+H2+H3 fix: route through shared _dhan_post helper (status check,
        # status=="success" validation, retry+backoff).
        rjson = _dhan_post(
            "https://api.dhan.co/v2/marketfeed/ltp",
            {"NSE_FNO": [int(cached["id"])]},
            timeout=8,
        )
        seg = (rjson.get("data") or {}).get("NSE_FNO") or {}
        for _, info in seg.items():
            ltp = float(info.get("last_price") or info.get("ltp") or 0)
            if ltp > 0:
                _fut_ltp_cache["ltp"] = ltp
                _fut_ltp_cache["ts"]  = now
                return ltp
    except (DhanAPIError, Exception) as e:
        try:
            print(f"[fetch_futures_ltp] error: {e}", flush=True)
        except Exception:
            pass
    return _fut_ltp_cache.get("ltp", 0.0)


# ─── Metrics engine  identical logic to Dash app ─────────────────────────────
def select_atm_band(df, spot, band=SIGNAL_BAND):
    if df.empty:
        return pd.DataFrame(), 0.0
    strikes = sorted(df["strike"].dropna().unique())
    if not strikes:
        return pd.DataFrame(), 0.0
    atm = min(strikes, key=lambda x: abs(x - spot))
    lo, hi = atm - band * NIFTY_STEP, atm + band * NIFTY_STEP
    out = df[df["strike"].between(lo, hi)].copy()
    return out.sort_values("strike").reset_index(drop=True), float(atm)


def compute_max_pain(df):
    if df.empty:
        return 0.0
    results = {}
    strikes = list(df["strike"].values)
    for K in strikes:
        lower = df[df["strike"] < K]
        upper = df[df["strike"] > K]
        cl = (lower["call_oi"] * (K - lower["strike"])).sum()
        pl = (upper["put_oi"] * (upper["strike"] - K)).sum()
        results[K] = cl + pl
    return float(min(results, key=results.get)) if results else 0.0


def compute_metrics(df, spot, expiry=None, history=None):
    if df.empty:
        return {}

    wide_df, atm = select_atm_band(df, spot, STRUCTURAL_BAND)
    tight_df, _  = select_atm_band(df, spot, SIGNAL_BAND)
    if wide_df.empty or tight_df.empty:
        return {}

    t = tight_df.copy()
    t["intr_c"] = np.maximum(0, spot - t["strike"])
    t["ev_c"]   = np.maximum(0, t["call_ltp"] - t["intr_c"])
    t["intr_p"] = np.maximum(0, t["strike"] - spot)
    t["ev_p"]   = np.maximum(0, t["put_ltp"] - t["intr_p"])

    ev_sum_c = float(t["ev_c"].sum())
    ev_sum_p = float(t["ev_p"].sum())
    ev_ratio = ev_sum_c / ev_sum_p if ev_sum_p > 0 else 1.0

    net_delta = float((t["call_oi"] * t["call_delta"]).sum() + (t["put_oi"] * t["put_delta"]).sum())
    net_gamma = float((t["call_oi"] * t["call_gamma"]).sum() + (t["put_oi"] * t["put_gamma"]).sum())
    net_theta = float((t["call_oi"] * t["call_theta"]).sum() + (t["put_oi"] * t["put_theta"]).sum())
    momentum  = float((t["call_oi_chg"] * t["call_delta"]).sum() + (t["put_oi_chg"] * t["put_delta"]).sum())

    # H10 fix (renamed v2): metric previously called "vanna" is NOT textbook vanna
    # (∂Δ/∂σ). It is an OI-weighted vega×delta triple product / spot — a custom
    # directional-vega pressure indicator. Renamed to oi_vega_delta_flow for clarity.
    #   oi_vega_delta_flow = OI-weighted vega×delta product / spot
    #                        (sign tracks call-vega-dominant vs put-vega-dominant)
    #   vega_skew          = OI-weighted call-vega / put-vega ratio (NOT IV skew;
    #                        the actual IV skew is `skew_slope` computed elsewhere)
    oi_vega_delta_flow = float(
        ((t["call_oi"] * t["call_vega"] * t["call_delta"]).sum() +
         (t["put_oi"] * t["put_vega"] * t["put_delta"]).sum()) / max(spot, 1)
    )
    sum_vega_c = float((t["call_oi"] * t["call_vega"]).sum())
    sum_vega_p = float((t["put_oi"] * t["put_vega"]).sum())
    vega_skew  = sum_vega_c / sum_vega_p if sum_vega_p > 0 else 1.0

    w = wide_df.copy()
    true_gex, _gex_series, gamma_flip = compute_true_gex(w, spot)
    gex = true_gex
    iv_rank, iv_pct = compute_iv_rank(w, atm)
    gt_ratio = abs(net_gamma) / max(abs(net_theta), 1e-6)
    total_coi = float(w["call_oi"].sum())
    total_poi = float(w["put_oi"].sum())
    pcr = total_poi / total_coi if total_coi > 0 else 1.0

    atm_row = w[w["strike"] == atm]
    if not atm_row.empty:
        atm_iv = float((safe_num(atm_row["call_iv"].iloc[0]) + safe_num(atm_row["put_iv"].iloc[0])) / 2)
    else:
        atm_iv = 0.0
    # ATM band vega captured AFTER IV/greek backfill (see below) so we always
    # read post-backfill values. Initialise here; overwritten after backfill.
    _atm_cv = 0.0
    _atm_pv = 0.0

    # IV backfill if Dhan returns zero IVs
    try:
        T_iv = 7 / 365
        if expiry:
            for _fmt in ("%Y-%m-%d", "%d-%b-%Y"):
                try:
                    _exp_dt = datetime.strptime(str(expiry), _fmt).date()
                    T_iv = max(1/365, (_exp_dt - date.today()).days / 365)
                    break
                except ValueError:
                    continue
        if w["call_iv"].max() < 0.01 and spot > 0:
            def _bs_iv_row(row, side, T):
                ltp = safe_num(row[f"{side}_ltp"])
                K   = safe_num(row["strike"])
                if ltp <= 0.05 or K <= 0:
                    return 0.0
                try:
                    return round(_solve_iv(ltp, spot, K, T, 0.065, side) * 100, 2)
                except Exception:
                    return 0.0
            w = w.copy()
            w["call_iv"] = w.apply(lambda r: _bs_iv_row(r, "call", T_iv), axis=1)
            w["put_iv"]  = w.apply(lambda r: _bs_iv_row(r, "put",  T_iv), axis=1)
        atm_row2 = w[w["strike"] == atm]
        if not atm_row2.empty:
            _c = safe_num(atm_row2["call_iv"].iloc[0])
            _p = safe_num(atm_row2["put_iv"].iloc[0])
            if _c > 0 or _p > 0:
                atm_iv = round(float((_c + _p) / max(1, (_c > 0) + (_p > 0))), 2)
        if atm_iv > 0:
            iv_rank, iv_pct = compute_iv_rank(w, atm)
    except Exception:
        pass

    # ── Band vega capture (post-backfill) ────────────────────────────────────────
    # Read vega_band_strikes from persisted owner settings (default 2 = ±100 pts).
    # OI-weighted sum across the band gives a stable, positioning-aware vega signal
    # that doesn't jump discontinuously when ATM shifts by one strike.
    # Placed here (after IV+greek backfill) so we always read filled vega values.
    try:
        _vb_settings  = _load_owner_settings()
        _vega_band_n  = int(_vb_settings.get("vega_band_strikes", 2))
    except Exception:
        _vega_band_n  = 2
    _vb_lo = atm - _vega_band_n * NIFTY_STEP
    _vb_hi = atm + _vega_band_n * NIFTY_STEP
    _vb_df = w[w["strike"].between(_vb_lo, _vb_hi)].copy()
    if not _vb_df.empty and "call_vega" in _vb_df.columns and "put_vega" in _vb_df.columns:
        _atm_cv = float((_vb_df["call_oi"] * _vb_df["call_vega"]).sum())
        _atm_pv = float((_vb_df["put_oi"]  * _vb_df["put_vega"]).sum())
    # _atm_cv / _atm_pv remain 0.0 if band is empty or columns missing

    # ── CI #2 fix: override cross-sectional iv_rank with TEMPORAL iv_rank
    # when sufficient history is available. The cross-sectional value is a
    # smile-position metric (close to 0 for normal NIFTY smiles where ATM sits
    # near the IV min), which makes the HIGH_VOL / LOW_VOL regime classifier
    # and the IV-rank-driven strategy branches (Iron Condor ≥65 / Iron Fly ≤35)
    # systematically misfire. The temporal rank uses a trailing ~20-day window
    # of ATM IV from `nifty_history.json`.
    _smile_position = iv_rank  # preserve original for diagnostics
    _iv_rank_is_temporal = False
    if history and atm_iv > 0:
        _temp_ivr, _is_ok = compute_temporal_iv_rank(atm_iv, history)
        if _is_ok and _temp_ivr is not None:
            iv_rank = _temp_ivr
            _iv_rank_is_temporal = True

    # Structural features
    step = NIFTY_STEP
    support    = float(wide_df.loc[wide_df["put_oi"].idxmax(), "strike"])
    resistance = float(wide_df.loc[wide_df["call_oi"].idxmax(), "strike"])
    max_pain   = compute_max_pain(df)

    atm_window = wide_df[wide_df["strike"].between(atm - 2*step, atm + 2*step)]
    atm_pressure = float(atm_window["put_oi_chg"].sum() - atm_window["call_oi_chg"].sum())

    near = wide_df[wide_df["strike"].between(atm - 2*step, atm + 2*step)].copy()
    total_oi  = float((wide_df["call_oi"] + wide_df["put_oi"]).sum())
    near_oi   = float((near["call_oi"] + near["put_oi"]).sum())
    near_oi_concentration = near_oi / total_oi if total_oi > 0 else 0.0

    total_oichg = float((wide_df["call_oi_chg"].abs() + wide_df["put_oi_chg"].abs()).sum())
    near_oichg  = float((near["call_oi_chg"].abs() + near["put_oi_chg"].abs()).sum())
    near_oichg_concentration = near_oichg / total_oichg if total_oichg > 0 else 0.0

    put_side  = wide_df[wide_df["strike"] <= atm][["strike","put_iv"]].dropna()
    put_side  = put_side[put_side["put_iv"] > 0.5]
    call_side = wide_df[wide_df["strike"] >= atm][["strike","call_iv"]].dropna()
    call_side = call_side[call_side["call_iv"] > 0.5]
    put_iv_slope = call_iv_slope = 0.0
    if len(put_side) >= 2:
        px = (put_side["strike"] - atm) / max(step, 1)
        put_iv_slope = float(np.polyfit(px, put_side["put_iv"], 1)[0])
    if len(call_side) >= 2:
        cx = (call_side["strike"] - atm) / max(step, 1)
        call_iv_slope = float(np.polyfit(cx, call_side["call_iv"], 1)[0])
    skew_slope = put_iv_slope - call_iv_slope

    return {
        "ev_ratio": round(ev_ratio, 3),
        "net_delta": round(net_delta, 0),
        "net_gamma": round(net_gamma, 6),
        "net_theta": round(net_theta, 0),
        "gex": round(gex, 0),
        "gamma_flip": round(gamma_flip, 0) if gamma_flip is not None else None,
        "iv_rank": iv_rank,
        "iv_pct": iv_pct,
        "iv_rank_is_temporal": _iv_rank_is_temporal,   # CI #2 fix
        "smile_position": _smile_position,             # CI #2 fix (diagnostic)
        "vanna": round(oi_vega_delta_flow, 2),            # key kept for downstream compat; value is oi_vega_delta_flow
        "oi_vega_delta_flow": round(oi_vega_delta_flow, 2),
        "gt_ratio": round(gt_ratio, 4),
        "momentum": round(momentum, 0),
        "vega_skew": round(vega_skew, 3),
        "pcr": round(pcr, 2),
        "atm_iv":         round(atm_iv, 2),
        "atm_call_vega":  round(_atm_cv, 4),   # OI-weighted ΣCall(OI×Vega) across ATM band
        "atm_put_vega":   round(_atm_pv, 4),   # OI-weighted ΣPut(OI×Vega) across ATM band
        "vega_band_strikes": _vega_band_n,       # band half-width used this tick
        "atm": float(atm),
        "support": support,
        "resistance": resistance,
        "max_pain": float(max_pain),
        "dist_to_support": round(spot - support, 2),
        "dist_to_resistance": round(resistance - spot, 2),
        "wall_width": round(resistance - support, 2),
        "atm_pressure": round(atm_pressure, 0),
        "near_oi_concentration": round(near_oi_concentration, 3),
        "near_oichg_concentration": round(near_oichg_concentration, 3),
        "skew_slope": round(skew_slope, 3),
        "call_oi_total": float(wide_df["call_oi"].sum()),
        "put_oi_total":  float(wide_df["put_oi"].sum()),
        "df_band": wide_df,
        "df_signal": tight_df,
    }



# ─── IV Smile Scenario Classifier  (IV Smile Master Reference Algorithm) ─────
def classify_iv_smile_scenario(df_band, m, spot, iv_smile_history=None):
    """
    Classifies the live IV smile shape into one of 12 scenarios.

    iv_smile_history: list of dicts [{ts, atm_iv, put_wing_excess, call_wing_excess}, ...]
    collected intraday by session_state — used for trend-aware disambiguation.
    Returns a dict with scenario metadata, signals, strategies, confidence, and trend info.
    """
    if df_band is None or len(df_band) == 0:
        return None
    df = pd.DataFrame(df_band) if isinstance(df_band, list) else df_band.copy()
    if df.empty or "call_iv" not in df.columns or "put_iv" not in df.columns:
        return None

    atm     = safe_num(m.get("atm", spot))
    atm_iv  = safe_num(m.get("atm_iv", 0))
    iv_rank = safe_num(m.get("iv_rank", 50))
    step    = NIFTY_STEP

    if atm_iv <= 0:
        return None

    # OTM wings: 2-6 steps from ATM
    otm_put_iv = df.loc[
        df["strike"].between(atm - 6*step, atm - 2*step) & (df["put_iv"]  > 0.5), "put_iv"
    ]
    otm_call_iv = df.loc[
        df["strike"].between(atm + 2*step, atm + 6*step) & (df["call_iv"] > 0.5), "call_iv"
    ]
    if len(otm_put_iv) < 2 or len(otm_call_iv) < 2:
        return None

    put_wing_excess  = float(otm_put_iv.mean())  - atm_iv
    call_wing_excess = float(otm_call_iv.mean()) - atm_iv
    skew_asymmetry   = put_wing_excess - call_wing_excess  # +ve = put skew, -ve = call skew

    # ── Intraday trend info from session history ──────────────────────────────
    trend_info = {"has_trend": False}
    if iv_smile_history and len(iv_smile_history) >= 3:
        lookback = min(len(iv_smile_history) - 1, 8)   # ~2 hours at 15-min refresh
        # v4: Exponentially weighted trend — recent ticks matter more
        _WEIGHTS = [0.30, 0.22, 0.15, 0.10, 0.07, 0.06, 0.05, 0.05]
        _wts = _WEIGHTS[:lookback]
        _hist_slice = iv_smile_history[-(lookback + 1):]
        ref = _hist_slice[0]
        d_atm = atm_iv - safe_num(ref.get("atm_iv", atm_iv))
        d_put = put_wing_excess - safe_num(ref.get("put_wing_excess", put_wing_excess))
        d_call = call_wing_excess - safe_num(ref.get("call_wing_excess", call_wing_excess))
        # Weighted multi-tick trend detection
        w_d_atm = 0.0; w_d_put = 0.0; w_d_call = 0.0
        for wi, wt in enumerate(_wts):
            h = _hist_slice[wi] if wi < len(_hist_slice) else _hist_slice[-1]
            h_next = _hist_slice[wi + 1] if wi + 1 < len(_hist_slice) else _hist_slice[-1]
            w_d_atm  += wt * (safe_num(h_next.get("atm_iv", 0)) - safe_num(h.get("atm_iv", 0)))
            w_d_put  += wt * (safe_num(h_next.get("put_wing_excess", 0)) - safe_num(h.get("put_wing_excess", 0)))
            w_d_call += wt * (safe_num(h_next.get("call_wing_excess", 0)) - safe_num(h.get("call_wing_excess", 0)))
        peak_atm = max(r.get("atm_iv", 0) for r in iv_smile_history)
        # CI #9 fix: previously a SECOND `trend_info = {...}` assignment here
        # silently overwrote this richer weighted dict, discarding
        # `weighted_d_atm/put/call` and `lookback`. The weighted computation ran
        # every tick and was thrown away. We now keep the weighted version and
        # surface all fields. Downstream consumers (_sc closure + Scenario 8
        # check at L1130) only read `has_trend` and `peak_atm_iv`, which both
        # versions provided — so this fix preserves behavior while exposing
        # the weighted fields for future scenario rules.
        trend_info = {
            "has_trend":      True,
            "d_atm_iv":       round(d_atm, 2),
            "d_put_wing":     round(d_put, 2),
            "d_call_wing":    round(d_call, 2),
            "peak_atm_iv":    round(peak_atm, 2),
            "ticks":          len(iv_smile_history),
            "weighted_d_atm":  round(w_d_atm, 2),
            "weighted_d_put":  round(w_d_put, 2),
            "weighted_d_call": round(w_d_call, 2),
            "lookback":       lookback,
        }

    def _sc(sid, name, badge, bc, signals, strategies, confidence, description):
        return {
            "scenario_id":     sid,
            "scenario_name":   name,
            "badge":           badge,
            "badge_color":     bc,
            "atm_iv":          round(atm_iv, 2),
            "put_wing_excess": round(put_wing_excess,  2),
            "call_wing_excess":round(call_wing_excess, 2),
            "skew_asymmetry":  round(skew_asymmetry,   2),
            "iv_rank":         round(iv_rank, 1),
            "signals":         signals,
            "strategies":      strategies,
            "confidence":      confidence,
            "description":     description,
            "trend":           trend_info,
        }

    # Scenario 12: Inverted Smile
    if put_wing_excess < -2.0 and call_wing_excess < -2.0:
        return _sc(12, "Inverted Smile / Vol Smirk", "ANOMALY", "#F59E0B",
            ["ATM IV exceeds OTM wings on both sides — extremely rare",
             "Likely a data artifact, bid-ask spread illusion, or illiquid strikes",
             "Check strike-level prices before acting on any signal here"],
            ["VERIFY DATA FIRST", "BUY WINGS IF REAL", "RATIO SPREAD"], 45,
            "ATM IV higher than both wings. Almost certainly a data artifact. Verify prices.")

    # Scenario 8: Compressed IV / Coiled Spring
    if iv_rank <= 20 and abs(put_wing_excess) < 3.0 and abs(call_wing_excess) < 3.0:
        if trend_info["has_trend"] and trend_info["peak_atm_iv"] > atm_iv * 1.5:
            return _sc(11, "Post-Event IV Crush", "VOL COLLAPSE", "#6B7280",
                ["ATM IV crushed from session peak {:.1f}% to now {:.1f}% (rank {:.0f}%ile)".format(
                    trend_info['peak_atm_iv'], atm_iv, iv_rank),
                 "Entire surface deflated — the event has resolved",
                 "Straddle buyers lose even if direction was right (vega loss dominates)",
                 "Crush confirmed by intraday session history"],
                ["SELL STRADDLES", "IRON CONDOR", "COLLECT THETA"], 92,
                "Post-event IV crush confirmed by intraday history. Peak {:.1f}% to now {:.1f}%. Premium-selling regime.".format(
                    trend_info['peak_atm_iv'], atm_iv))
        return _sc(8, "Compressed IV / Coiled Spring", "BREAKOUT ALERT", "#D97706",
            ["ATM IV at multi-week low (rank {:.0f}%ile) — market in deep sleep".format(iv_rank),
             "Both wings flat — total smile compression",
             "Market underpricing future moves — vol explosion imminent",
             "Direction unknown — but the magnitude of the move will be large",
             "Theta decay is the enemy: do not buy too early, manage timing"],
            ["BUY STRADDLE", "BUY STRANGLE", "AVOID SELLING VOL"], 85,
            "IV rank {:.0f}%ile, both wings flat. Classic coiled spring. Buy vol but manage theta.".format(iv_rank))

    # Scenario 11: Post-Event IV Crush (snapshot only)
    if iv_rank <= 15 and put_wing_excess < 2.0 and call_wing_excess < 2.0:
        return _sc(11, "Post-Event IV Crush", "VOL COLLAPSE", "#6B7280",
            ["Entire surface deflated — event has resolved",
             "Straddle buyers lose even if direction was right (vega loss dominates)",
             "Good time to sell premium going forward — vol regime has reset"],
            ["SELL STRADDLES", "IRON CONDOR", "COLLECT THETA"], 88,
            "Post-event IV crush (rank {:.0f}%ile). Everything deflated. Premium-selling regime.".format(iv_rank))

    # Scenario 2: Crash Fear / Panic
    if put_wing_excess > 12.0 and iv_rank >= 65:
        extra = []
        if trend_info["has_trend"]:
            if trend_info["d_put_wing"] > 3.0:
                extra.append("TREND: PUT WING ACCELERATING +{:.1f}pts in ~30 min — panic building".format(trend_info['d_put_wing']))
            elif trend_info["d_put_wing"] < -2.0:
                extra.append("TREND: PUT WING DEFLATING {:.1f}pts — panic may be peaking/reversing".format(trend_info['d_put_wing']))
        return _sc(2, "Crash Fear / Panic Mode", "EXTREME BEAR", "#DC2626",
            ["OTM puts {:.1f} vol pts above ATM — explosive put demand".format(put_wing_excess),
             "ATM IV very elevated (rank {:.0f}%ile) — whole surface lifted".format(iv_rank),
             "Massive put buying: retail panic + institutional hedging simultaneously",
             "PCR likely spiking above 1.3-1.5 (verify vs recent rolling baseline)",
             "Contrarian: calls are SKEW-cheap relative to puts — NOT cheap in absolute vega terms"] + extra,
            ["BUY CALL (SKEW-CHEAP, NOT VOL-CHEAP)", "SELL PUT SPREADS", "DO NOT SELL NAKED PUTS"], 88,
            "Panic: puts {:.1f}pts above ATM, IV rank {:.0f}%ile. Calls skew-cheap but expensive in absolute vega.".format(put_wing_excess, iv_rank))

    # Scenario 10: Pre-Event Volatility Spike
    if put_wing_excess > 5.0 and call_wing_excess > 5.0 and abs(skew_asymmetry) < 4.0 and iv_rank >= 55:
        extra = []
        if trend_info["has_trend"] and trend_info["d_put_wing"] > 1.5 and trend_info["d_call_wing"] > 1.5:
            extra.append("TREND: BOTH WINGS BUILDING put+{:.1f}pts call+{:.1f}pts last 30 min — vol accumulation active".format(
                trend_info['d_put_wing'], trend_info['d_call_wing']))
        return _sc(10, "Pre-Event Volatility Spike", "EVENT RISK", "#D97706",
            ["Both wings elevated symmetrically — binary event pricing",
             "Market cannot pick direction: straddle is the only honest trade",
             "WARNING: IV already peaked at rank {:.0f}%ile — pre-event straddles often lose".format(iv_rank),
             "Only buy straddle if realized move will exceed the implied move priced in",
             "Best edge: sell vol AFTER the event resolves, not before"] + extra,
            ["STRADDLE PRE-EVENT (IF MOVE > IMPLIED)", "SELL VOL POST-EVENT", "AVOID NAKED WRITES"], 90,
            "Binary event: both wings +{:.1f}/{:.1f}pts symmetrically at IV rank {:.0f}%ile.".format(put_wing_excess, call_wing_excess, iv_rank))

    # Scenario 5: Melt-Up / Euphoric Rally
    if call_wing_excess > 4.0 and put_wing_excess < 1.5:
        return _sc(5, "Melt-Up / Euphoric Rally", "EXTREME BULL", "#22C55E",
            ["Put wing collapsed — nobody hedging downside (pure euphoria state)",
             "OTM calls sharply bid: {:.1f}pts above ATM — lottery-ticket call buying".format(call_wing_excess),
             "ATM IV often paradoxically LOW despite rising market (IV crush on drift up)",
             "DANGER: put collapse = no downside protection = violent reversal risk",
             "Smart money typically fades by selling OTM calls into the melt-up"],
            ["SELL OTM CALLS", "BUY CHEAP PUTS", "TRAIL STOPS TIGHT"], 78,
            "Euphoria: put wing flat ({:.1f}pts), calls steeply bid {:.1f}pts. Reversal risk rising.".format(put_wing_excess, call_wing_excess))

    # Scenario 4: Call Skew Dominant — RARE on NIFTY
    if call_wing_excess > put_wing_excess and call_wing_excess > 3.0 and skew_asymmetry < -2.0:
        return _sc(4, "Call Skew Dominant (Positive Skew)", "BULLISH", "#22C55E",
            ["Call IV exceeds put IV — rare positive skew on NIFTY",
             "Aggressive OTM call buying — breakout or budget/event anticipation",
             "Typical before budget, results seasons, or global melt-up phases",
             "RARE on NIFTY — treat as high-conviction bullish signal when seen"],
            ["BUY CALL SPREADS", "SELL PUT SKEW", "LONG FUTURES"], 82,
            "Positive skew: call wing {:.1f}pts vs put wing {:.1f}pts. Rare on NIFTY — high conviction bullish.".format(call_wing_excess, put_wing_excess))

    # Scenario 9: Two-Sided Fork
    if put_wing_excess > 3.5 and call_wing_excess > 2.5 and 0.5 < skew_asymmetry < 6.0 and iv_rank < 50:
        return _sc(9, "Two-Sided Fork", "MILD BEAR", "#D97706",
            ["Both wings bid but ATM compressed: put+{:.1f}pts call+{:.1f}pts".format(put_wing_excess, call_wing_excess),
             "Call wing bid = bulls not fully exiting, two-sided uncertainty",
             "Net put skew ({:+.1f}pts) = directional bearish positioning warranted".format(skew_asymmetry),
             "Both wings elevated does NOT mean sell both wings — use asymmetric plays",
             "Avoid symmetric writes (strangle/condor) — use directional bearish setups"],
            ["BEAR PUT SPREAD", "LONG PUT / SHORT CALL", "AVOID SYMMETRIC WRITES"], 78,
            "Fork: put+{:.1f}pts call+{:.1f}pts, ATM compressed. Net put skew — directional bearish plays.".format(put_wing_excess, call_wing_excess))

    # Scenario 6: Post-Crash Relief Rally
    if put_wing_excess > 4.0 and call_wing_excess > 1.0 and 2.0 < skew_asymmetry <= 8.0 and 20 < iv_rank < 55:
        extra = []
        conf = 65
        if trend_info["has_trend"]:
            if trend_info["d_put_wing"] < -1.5:
                extra.append("TREND: PUT WING DEFLATING {:.1f}pts — relief confirmed by intraday trend".format(trend_info['d_put_wing']))
                conf = 80
            if trend_info["d_call_wing"] > 1.0:
                extra.append("TREND: CALL WING RISING +{:.1f}pts — early call buying confirms recovery".format(trend_info['d_call_wing']))
                conf = min(conf + 5, 85)
        return _sc(6, "Post-Crash Relief Rally", "RECOVERING", "#2563EB",
            ["Put wing still elevated at {:.1f}pts but potentially deflating from panic peak".format(put_wing_excess),
             "Call wing rising ({:.1f}pts) — early call buying as market stabilizes".format(call_wing_excess),
             "IV contraction underway — IV crush is a tailwind for long delta positions",
             "Long calls gain doubly: delta gain + vega gain (falling IV = vol tailwind)",
             "Confirm with market breadth — not all relief rallies sustain to new highs"] + extra,
            ["BUY CALLS (IV FALLING)", "CLOSE PUT HEDGES", "LONG FUTURES"], conf,
            "Relief: puts still {:.1f}pts elevated but deflating, calls {:.1f}pts and rising.".format(put_wing_excess, call_wing_excess))

    # Scenario 1: Classic Put Skew (Negative Skew)
    if put_wing_excess >= 4.0 and call_wing_excess < 3.5 and skew_asymmetry >= 2.0:
        extra = []
        if trend_info["has_trend"] and trend_info["d_put_wing"] > 1.5:
            extra.append("TREND: SKEW BUILDING put wing+{:.1f}pts last 30 min — skew pressure increasing".format(trend_info['d_put_wing']))
        return _sc(1, "Classic Put Skew (Negative Skew)", "BEARISH", "#EF4444",
            ["OTM puts {:.1f} vol pts above ATM — standard hedging mode".format(put_wing_excess),
             "Call wing flat/cheap — calls not in demand",
             "Institutions buying downside protection (collars/spreads/put spreads)",
             "Market assigning fat left-tail probability — orderly, not panic",
             "NOT a crash signal — systematic hedging, not capitulation"] + extra,
            ["BUY PUT SPREAD", "SELL CALL SKEW", "RISK REVERSAL SHORT"], 80,
            "Classic negative skew: OTM puts {:.1f}pts above ATM, calls flat. Orderly hedging, not panic.".format(put_wing_excess))

    # Scenario 3: Bearish Drift / Slow Bleed
    if 2.0 <= put_wing_excess < 4.0 and call_wing_excess < 2.0:
        return _sc(3, "Bearish Drift / Slow Bleed", "MILD BEAR", "#D97706",
            ["Put wing moderately elevated ({:.1f}pts) — gradual slope, not steep".format(put_wing_excess),
             "Call wing flat — bulls absent or covering long positions",
             "Systematic put buying — hedgers protecting long portfolios",
             "Watch for ATM IV expansion — that signals acceleration toward panic mode"],
            ["BEAR PUT SPREAD", "SHORT FUTURES", "AVOID CALLS"], 72,
            "Slow bleed: put wing {:.1f}pts, calls flat. Watch ATM IV for acceleration signal.".format(put_wing_excess))

    # Scenario 7: Flat / Symmetric Smile
    if abs(skew_asymmetry) <= 2.0 and put_wing_excess > 0 and call_wing_excess > 0:
        return _sc(7, "Flat / Symmetric Smile", "NEUTRAL", "#6B7280",
            ["Put wing {:.1f}pts ≈ call wing {:.1f}pts — balanced book".format(put_wing_excess, call_wing_excess),
             "Market sees equal probability of up and down moves",
             "No directional preference — balanced (relatively rare in practice on NIFTY)",
             "Seen post-resolution of major events (budget, Fed, results seasons)"],
            ["IRON CONDOR", "SHORT STRANGLE (IF IV ELEVATED)", "SELL BOTH WINGS"], 65,
            "Symmetric smile: put+{:.1f}pts approx call+{:.1f}pts. No directional edge.".format(put_wing_excess, call_wing_excess))

    # Fallback
    return _sc(0, "Indeterminate / Transitional", "NEUTRAL", "#6B7280",
        ["Pattern does not cleanly match any known scenario",
         "put_wing={:.1f}pts, call_wing={:.1f}pts, skew={:.1f}pts".format(put_wing_excess, call_wing_excess, skew_asymmetry),
         "May be a transitional state — collect more data ticks for trend context"],
        ["WAIT FOR CLARITY"], 0,
        "No clean scenario match. Wings: put={:.1f}pts, call={:.1f}pts.".format(put_wing_excess, call_wing_excess))

# ─── Bias engine  IDENTICAL to Dash app ─────────────────────────────────────
def compute_nifty_bias(m, history=None):
    if not m:
        return {"bias_score": 0.0, "confidence": 0.0, "regime": "NO DATA",
                "direction": "NEUTRAL", "factors": []}
    history = history or []
    factors = []
    direction = 0.0
    confidence = 0.0
    BW = BIAS_WEIGHTS

    if m["net_delta"] > 0:
        direction += BW["net_delta"]; factors.append("Net delta bullish")
    elif m["net_delta"] < 0:
        direction -= BW["net_delta"]; factors.append("Net delta bearish")

    if m["momentum"] > 0:
        direction += BW["momentum"]; factors.append("OI momentum bullish")
    elif m["momentum"] < 0:
        direction -= BW["momentum"]; factors.append("OI momentum bearish")

    if m["ev_ratio"] >= BW["ev_ratio_bull"]:
        direction += BW["ev_ratio"]; factors.append("Call premium stronger")
    elif m["ev_ratio"] <= BW["ev_ratio_bear"]:
        direction -= BW["ev_ratio"]; factors.append("Put premium stronger")

    if m["atm_pressure"] > 0:
        direction += BW["atm_pressure"]; factors.append("ATM put support stronger")
    elif m["atm_pressure"] < 0:
        direction -= BW["atm_pressure"]; factors.append("ATM call pressure stronger")

    if m["skew_slope"] > BW["skew_slope_threshold"]:
        direction -= BW["skew_slope"]; factors.append("Downside IV skew stronger")
    elif m["skew_slope"] < -BW["skew_slope_threshold"]:
        direction += BW["skew_slope"]; factors.append("Upside call skew improving")

    if m["vanna"] > 0:
        direction += BW["vanna"]
    elif m["vanna"] < 0:
        direction -= BW["vanna"]

    regime, vol_regime, near_flip = classify_gamma_regime(
        gex=m["gex"], wall_width=m["wall_width"], momentum=m["momentum"],
        atm_iv=m["atm_iv"], iv_rank=m.get("iv_rank", 50),
        spot=m.get("atm", 0), gamma_flip=m.get("gamma_flip"),
    )
    if "PINNED" in regime or "RANGE" in regime:
        confidence += BW["regime_range"]; factors.append(f"GEX regime: {regime} [{vol_regime}]")
    elif "TREND" in regime:
        confidence += BW["regime_trend"]; factors.append(f"GEX regime: {regime} [{vol_regime}]")
    elif "FLIP" in regime:
        confidence += BW["regime_transition"]; factors.append(f"⚠ Gamma Flip Zone [{vol_regime}]")
    else:
        confidence += BW["regime_transition"]
    if near_flip:
        factors.append("⚠ Near Gamma Flip  elevated breakout risk")

    if m["near_oi_concentration"] >= BW["near_oi_min"]:
        confidence += BW["near_oi_concentration"]; factors.append("Near-ATM OI concentrated")
    if m["near_oichg_concentration"] >= BW["near_oichg_min"]:
        confidence += BW["near_oichg_concentration"]; factors.append("Fresh OI active near ATM")
    if abs(m["dist_to_support"]) < BW["wall_proximity_pts"] or abs(m["dist_to_resistance"]) < BW["wall_proximity_pts"]:
        confidence += BW["wall_proximity"]; factors.append("Spot close to active wall")

    if history:
        prev = history[-1]
        prev_support    = safe_num(prev.get("support",    m["support"]))
        prev_resistance = safe_num(prev.get("resistance", m["resistance"]))
        prev_max_pain   = safe_num(prev.get("max_pain",   m["max_pain"]))
        prev_wall_width = safe_num(prev.get("wall_width", m["wall_width"]))
        if m["support"] > prev_support and m["resistance"] >= prev_resistance:
            direction += BW["wall_shift"]; factors.append("Walls shifting higher")
        elif m["support"] <= prev_support and m["resistance"] < prev_resistance:
            direction -= BW["wall_shift"]; factors.append("Walls shifting lower")
        if m["max_pain"] > prev_max_pain:
            direction += BW["max_pain_drift"]; factors.append("Max pain drifting up")
        elif m["max_pain"] < prev_max_pain:
            direction -= BW["max_pain_drift"]; factors.append("Max pain drifting down")
        if m["wall_width"] < prev_wall_width and m["gex"] > 0:
            confidence += BW["range_compression"]; factors.append("Range compressing")
        elif m["wall_width"] > prev_wall_width and m["gex"] < 0:
            confidence += BW["expansion_building"]; factors.append("Expansion structure building")

    if len(history) >= 2:
        recent = history[-2:]
        nds  = [safe_num(x.get("net_delta", 0)) for x in recent] + [safe_num(m["net_delta"])]
        moms = [safe_num(x.get("momentum",  0)) for x in recent] + [safe_num(m["momentum"])]
        if all(x > 0 for x in nds) and all(x > 0 for x in moms):
            confidence += BW["persistence"]; factors.append("Bullish persistence")
        elif all(x < 0 for x in nds) and all(x < 0 for x in moms):
            confidence += BW["persistence"]; factors.append("Bearish persistence")

    bias_score = max(-100, min(100, round(direction, 1)))

    # FIX (Issue 3): raw confidence sum has a structural ceiling of ~73
    # (regime_trend 25 + near_oi 12 + near_oichg 12 + wall_proximity 8 +
    #  range_compression/expansion 6 + persistence 10 = 73).
    # Normalise against that max so a fully confirming market reads ~100%.
    _CONF_MAX = (
        BW["regime_trend"]
        + BW["near_oi_concentration"]
        + BW["near_oichg_concentration"]
        + BW["wall_proximity"]
        + max(BW["range_compression"], BW["expansion_building"])
        + BW["persistence"]
    )  # = 73
    confidence = round(max(0.0, min(100.0, (confidence / _CONF_MAX) * 100)), 1)

    if bias_score >= BW["bias_bull_threshold"]:
        direction_label = "BULLISH"
    elif bias_score <= BW["bias_bear_threshold"]:
        direction_label = "BEARISH"
    else:
        direction_label = "NEUTRAL"

    return {
        "bias_score": bias_score,
        "confidence": confidence,
        "regime":     regime,
        "vol_regime": vol_regime,
        "near_flip":  near_flip,
        "direction":  direction_label,
        "factors":    factors[:6],
    }


# ─── Section 3 & 4 Bias Engine ───────────────────────────────────────────────
def compute_section34_bias(df_band_records, m, spot, roll_discount=1.0, front_expiry=None, skew_baseline=None):
    """
    5-signal market bias from Section 3 (metrics) + Section 4 (±10 strike band).
    Score range: -100 (strongly bearish) to +100 (strongly bullish).

    Signal 1 — Moneyness-Adjusted Δ-Weighted Net OI    (±35 pts)
    Signal 2 — Moneyness-Adjusted OI Momentum          (±25 pts)
    Signal 3 — Key Level Position (S/R + GEX×Flip + Max Pain) (±20 pts)
    Signal 4 — OTM IV Skew                             (±12 pts)
    Signal 5 — PCR Contrarian                          (±8  pts)
    """
    if not df_band_records or not m or spot <= 0:
        return {"bias_score": 0.0, "direction": "NEUTRAL", "signal_breakdown": {}}

    df = pd.DataFrame(df_band_records).copy()
    if df.empty:
        return {"bias_score": 0.0, "direction": "NEUTRAL", "signal_breakdown": {}}

    for col in ["strike", "call_oi", "put_oi", "call_oi_chg", "put_oi_chg",
                "call_delta", "put_delta", "call_iv", "put_iv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Section 3 anchors
    support    = safe_num(m.get("support",    spot))
    resistance = safe_num(m.get("resistance", spot))
    max_pain   = safe_num(m.get("max_pain",   spot))
    gamma_flip = m.get("gamma_flip")
    gex        = safe_num(m.get("gex", 0))
    if gamma_flip is not None:
        gamma_flip = safe_num(gamma_flip)

    # ATM band: ±0.3% of spot (≈ ±70 pts at Nifty 23 000)
    atm_band = spot * 0.003

    call_delta_abs = df["call_delta"].abs()
    put_delta_abs  = df["put_delta"].abs()

    # Moneyness weights:
    #   OTM calls (k > spot) and ATM → full weight (1.0); ITM calls (k < spot−band) → 0.35
    #   OTM puts  (k < spot) and ATM → full weight (1.0); ITM puts  (k > spot+band) → 0.35
    call_wt = df["strike"].apply(lambda k: 1.0 if k >= (spot - atm_band) else 0.35)
    put_wt  = df["strike"].apply(lambda k: 1.0 if k <= (spot + atm_band) else 0.35)

    # ── v4 #3: Smart Money OI Quality Filter ────────────────────────────
    # Filter out retail noise: only count OI changes >= significance floor
    # and within 5 strikes of ATM (where institutional activity concentrates)
    _OI_MOMENTUM_FLOOR = max(100, df["call_oi_chg"].abs().median() * 0.1)
    _near_mask = df["strike"].between(spot - 5 * NIFTY_STEP, spot + 5 * NIFTY_STEP)
    _smart_mask = (df["call_oi_chg"].abs() >= _OI_MOMENTUM_FLOOR) | \
                 (df["put_oi_chg"].abs() >= _OI_MOMENTUM_FLOOR)
    _quality_mask = _near_mask & _smart_mask
    _quality_df = df[_quality_mask].copy() if _quality_mask.any() else df.copy()
    # Use _quality_df for Signal 1 and Signal 2 instead of full df
    _qf = _quality_df

    # ── Signal 1: Moneyness-Adjusted Δ-Weighted Net OI (±35 pts) ─────────────
    # Positive = put-side dominant = BULLISH (floor > ceiling)
    # Negative = call-side dominant = BEARISH (ceiling > floor)
    # Uses smart-money quality filter (_qf) — near-ATM, significant OI changes only
    sum_call_s1 = (_qf["call_oi"] * _qf["call_delta"].abs() * _qf["strike"].apply(lambda k: 1.0 if k >= (spot - atm_band) else 0.35)).sum()
    sum_put_s1  = (_qf["put_oi"]  * _qf["put_delta"].abs()  * _qf["strike"].apply(lambda k: 1.0 if k <= (spot + atm_band) else 0.35)).sum()
    denom_s1    = sum_call_s1 + sum_put_s1
    s1 = ((sum_put_s1 - sum_call_s1) / denom_s1 * 35.0) if denom_s1 > 0 else 0.0
    s1 = max(-35.0, min(35.0, s1))

    # ── Signal 2: Moneyness-Adjusted OI Momentum (±25 pts) ───────────────────
    net_mom   = (
        (_qf["put_oi_chg"]  * _qf["put_delta"].abs()  * _qf["strike"].apply(lambda k: 1.0 if k <= (spot + atm_band) else 0.35)).sum()
      - (_qf["call_oi_chg"] * _qf["call_delta"].abs() * _qf["strike"].apply(lambda k: 1.0 if k >= (spot - atm_band) else 0.35)).sum()
    )
    abs_mom   = (
        (_qf["put_oi_chg"].abs()  * _qf["put_delta"].abs()  * _qf["strike"].apply(lambda k: 1.0 if k <= (spot + atm_band) else 0.35)).sum()
      + (_qf["call_oi_chg"].abs() * _qf["call_delta"].abs() * _qf["strike"].apply(lambda k: 1.0 if k >= (spot - atm_band) else 0.35)).sum()
    )
    s2 = (net_mom / abs_mom * 25.0) if abs_mom > 0 else 0.0
    # roll_discount < 1.0 when detect_roll_activity() identifies mechanical expiry
    # roll as the dominant cause of OI change — prevents institutional roll from
    # being misread as genuine directional selling / covering pressure.
    s2 = round(max(-25.0, min(25.0, s2)) * roll_discount, 1)

    # ── Signal 3: Key Level Position (±20 pts) ────────────────────────────────
    # Sub-A: Spot vs S/R band (±6 pts)
    if resistance > support > 0:
        midpoint = (support + resistance) / 2.0
        if spot > resistance:
            s3a = 6.0    # breakout above resistance
        elif spot > midpoint:
            s3a = 3.0    # upper half of range
        elif spot > support:
            s3a = -3.0   # lower half of range
        else:
            s3a = -6.0   # breakdown below support
    else:
        s3a = 0.0

    # Sub-B: Gamma Flip × GEX quadrant (±10 pts)
    if gamma_flip and gamma_flip > 0:
        above_flip = spot > gamma_flip
        pos_gex    = gex >= 0
        if above_flip and pos_gex:
            s3b = 10.0    # stable + supported: ideal bullish backdrop
        elif above_flip and not pos_gex:
            s3b = 4.0     # trending up but volatile (negative GEX amplifies)
        elif not above_flip and pos_gex:
            s3b = -4.0    # oscillating below pivot
        else:
            s3b = -10.0   # trending down, dealer flow amplifies falls
    else:
        s3b = 3.0 if gex >= 0 else -3.0  # fallback: GEX sign only

    # Sub-C: Max Pain gravitational pull (±4 pts)
    if max_pain > 0 and spot > 0:
        pull_pct = (max_pain - spot) / spot * 100.0
        if   pull_pct >  0.5: s3c =  4.0
        elif pull_pct >  0.15: s3c =  2.0
        elif pull_pct < -0.5: s3c = -4.0
        elif pull_pct < -0.15: s3c = -2.0
        else:                  s3c =  0.0
    else:
        s3c = 0.0

    s3 = max(-20.0, min(20.0, s3a + s3b + s3c))

    # ── Signal 4: Normalised IV Skew — ±9 pts level + ±3 pts intraday = ±12 total ─
    # FIX (v4.1): Raw skew is always positive in normal Nifty sessions, causing a
    # persistent ~5–7 pt bearish drag even in calm markets.
    #
    # Solution (Option 1 — ATM IV normalisation):
    #   norm_skew = (put_OTM_IV − call_OTM_IV) / ATM_IV × 100
    # Normal Nifty range ≈ 20–40% (baseline ~30%). Signal fires only on EXCESS above
    # that baseline, removing the spurious chronic bearish tilt.
    #
    # The remaining ±3 pts (intraday momentum, Option 2) are added by the caller
    # using session_state to anchor against the opening norm_skew each session.
    otm_call_mask = df["strike"] > (spot + atm_band)
    otm_put_mask  = df["strike"] < (spot - atm_band)
    atm_mask      = df["strike"].between(spot - atm_band, spot + atm_band)

    valid_civ = df.loc[otm_call_mask, "call_iv"][df.loc[otm_call_mask, "call_iv"] > 0.5]
    valid_piv = df.loc[otm_put_mask,  "put_iv" ][df.loc[otm_put_mask,  "put_iv" ] > 0.5]

    # ATM IV: regime baseline — average of ATM calls + puts (both sides for robustness)
    atm_c = df.loc[atm_mask, "call_iv"][df.loc[atm_mask, "call_iv"] > 0.5]
    atm_p = df.loc[atm_mask, "put_iv" ][df.loc[atm_mask, "put_iv" ] > 0.5]
    atm_iv_level = (
        pd.concat([atm_c, atm_p]).mean()
        if (len(atm_c) + len(atm_p)) > 0 else 15.0
    )

    if len(valid_civ) > 0 and len(valid_piv) > 0 and atm_iv_level > 1.0:
        raw_skew  = valid_piv.mean() - valid_civ.mean()   # positive = put fear premium
        norm_skew = (raw_skew / atm_iv_level) * 100.0     # as % of ATM IV

        # Fix #4: Use caller-supplied adaptive baseline (trailing median of norm_skew
        # over recent neutral sessions from bias history). Falls back to 30.0 when
        # insufficient history exists (first boot, cold start, etc.).
        NORMAL_SKEW_BASELINE = skew_baseline if skew_baseline is not None else 30.0
        excess = norm_skew - NORMAL_SKEW_BASELINE
        # 20 pct-pt excess maps to full ±9 level score
        s4 = max(-9.0, min(9.0, -(excess / 20.0) * 9.0))
    else:
        raw_skew  = 0.0
        norm_skew = 30.0   # assume neutral when IV data unavailable
        s4        = 0.0

    # ── Signal 5: IV Term Structure (±8 pts) — placeholder, injected by caller ──
    # REPLACED from PCR Contrarian (v4.1) to eliminate the S1/S5 OI-data redundancy.
    # Both S1 and PCR derive from put_OI / call_OI quantities — information is
    # double-counted at effectively ±43 pts on the same underlying measure.
    #
    # The IV Term Structure signal uses front vs back expiry ATM IV (not OI at all),
    # making it genuinely orthogonal to S1. Normal contango (back IV > front IV)
    # signals calm market. Backwardation (front IV > back IV) signals acute near-term
    # fear. Scored via the existing compute_term_structure_signal() ts_score field,
    # with an automatic DTE confidence weight applied by the caller to suppress the
    # signal when front-expiry DTE ≤ 2 days (IV noise-dominated near expiry).
    s5 = 0.0   # always 0 here; real value injected in main flow after _ts_data available

    # ── Aggregate ─────────────────────────────────────────────────────────────
    bias_score = max(-100.0, min(100.0, round(s1 + s2 + s3 + s4 + s5, 1)))

    # ── Signal 6: Bias Velocity / Acceleration (±10 pts) ────────────────
    # v4 NEW: Rate-of-change of composite bias — the most LEADING signal.
    # Requires bias_history (session state) passed via function param.
    s6 = 0.0
    _bias_hist = None  # will be populated by caller
    # We store a placeholder; the actual velocity is computed in the
    # main flow where bias_history is available, then injected.
    bias_score_raw = s1 + s2 + s3 + s4 + s5

    if   bias_score >=  15: direction = "BULLISH"
    elif bias_score <= -15: direction = "BEARISH"
    else:                   direction = "NEUTRAL"

    # v4: expose smart money quality strikes count for the Leading Signals panel
    _qcount = int(_quality_mask.sum())

    return {
        "bias_score":  bias_score,
        "direction":   direction,
        "quality_strikes_count":  _qcount,
        "roll_discount_applied":  round(roll_discount, 2),
        "roll_window_active":     is_roll_window(front_expiry),  # Fix #3: pass expiry for dynamic weekday
        # v4.1: norm_skew returned so caller can anchor intraday delta (Option 2)
        "norm_skew":   round(norm_skew, 2),
        "signal_breakdown": {
            "S1 Net OI":     round(s1, 1),
            "S2 Momentum":   round(s2, 1),   # already roll-discounted
            "S3 Key Levels": round(s3, 1),
            "S4 IV Skew":    round(s4, 1),   # level component only (±9 pts)
            "S5 Term Str":   round(s5, 1),   # 0.0 here; real value injected by caller
            "S6 Velocity":   0.0,   # populated by caller with bias_history
        },
    }


def strategy_recommendation(bias, m, history=None):
    history = history or []
    atm        = int(m.get("atm", 0))
    support    = int(m.get("support", 0))
    resistance = int(m.get("resistance", 0))
    step       = NIFTY_STEP
    gamma_flip = m.get("gamma_flip")
    iv_rank    = m.get("iv_rank", 50)
    momentum   = m.get("momentum", 0)
    ev_ratio   = m.get("ev_ratio", 1.0)
    pcr        = m.get("pcr", 1.0)
    skew_slope = m.get("skew_slope", 0)
    near_oichg = m.get("near_oichg_concentration", 0.5)
    gex        = m.get("gex", 0)
    confidence = bias.get("confidence", 0)
    regime     = bias.get("regime", "TRANSITION")
    direction  = bias.get("direction", "NEUTRAL")
    vol_regime = bias.get("vol_regime", "MID_VOL")
    near_flip  = bias.get("near_flip", False)
    iv_ctx     = f"IV Rank {iv_rank:.0f} ({vol_regime})"

    if not atm or not support or not resistance:
        return {"name": "WAIT", "legs": "ATM/wall data unavailable", "color": BLUE,
                "market_mode": regime, "mode_color": BLUE, "iv_context": iv_ctx}
    if confidence < BIAS_WEIGHTS["confidence_min_strategy"]:
        return {"name": "WAIT", "legs": "No clear edge  await regime confirmation", "color": BLUE,
                "market_mode": regime, "mode_color": BLUE, "iv_context": iv_ctx}

    if near_flip or "FLIP" in regime:
        flip_str = f"{int(gamma_flip)}" if gamma_flip is not None else "N/A"
        return {"name": "⚠ Long Straddle / Strangle",
                "legs": f"Buy {atm} CE + {atm} PE (Straddle) OR Buy {atm+step} CE + {atm-step} PE (Strangle). Flip @ {flip_str}.",
                "color": PINK, "market_mode": regime, "mode_color": PINK,
                "iv_context": f"FLIP ZONE  IV Rank {iv_rank:.0f}"}

    if gex < 0:
        recent_gex_neg = sum(1 for h in history[-3:] if safe_num(h.get("gex", 1)) < 0)
        if recent_gex_neg >= 2:
            if direction == "BEARISH":
                return {"name": "Bear Put Spread (GEX Trend Day)",
                        "legs": f"Buy {atm} PE | Sell {atm-2*step} PE",
                        "color": RED, "market_mode": regime, "mode_color": RED,
                        "iv_context": f"GEX negative  {iv_ctx}"}
            elif direction == "BULLISH":
                return {"name": "Bull Call Spread (GEX Trend Day)",
                        "legs": f"Buy {atm} CE | Sell {atm+2*step} CE",
                        "color": GREEN, "market_mode": regime, "mode_color": GREEN,
                        "iv_context": f"GEX negative  {iv_ctx}"}
            else:
                return {"name": "WAIT  GEX Negative",
                        "legs": "Await directional confirmation.",
                        "color": BLUE, "market_mode": regime, "mode_color": BLUE,
                        "iv_context": f"GEX negative  {iv_ctx}"}

    if near_oichg < 0.40:
        if direction == "BULLISH" and momentum > 0:
            return {"name": "Bull Call Spread (Weak Pin)",
                    "legs": f"Buy {atm} CE | Sell {atm+2*step} CE",
                    "color": GREEN, "market_mode": regime, "mode_color": GREEN,
                    "iv_context": f"Near OI Chg {near_oichg*100:.0f}%  pin soft"}
        elif direction == "BEARISH" and momentum < 0:
            return {"name": "Bear Put Spread (Weak Pin)",
                    "legs": f"Buy {atm} PE | Sell {atm-2*step} PE",
                    "color": RED, "market_mode": regime, "mode_color": RED,
                    "iv_context": f"Near OI Chg {near_oichg*100:.0f}%  pin soft"}
        else:
            return {"name": "WAIT  Pin Weakening",
                    "legs": f"Near OI Chg% {near_oichg*100:.0f}%  pin soft, avoid IC.",
                    "color": BLUE, "market_mode": regime, "mode_color": BLUE, "iv_context": iv_ctx}

    if gamma_flip is not None and history:
        prev_atm = safe_num(history[-1].get("atm", atm))
        if atm > gamma_flip and prev_atm < gamma_flip:
            return {"name": "Bull Call Spread (Flip Recapture)",
                    "legs": f"Buy {atm} CE | Sell {atm+2*step} CE. Confirm with GEX.",
                    "color": GREEN, "market_mode": regime, "mode_color": GREEN,
                    "iv_context": f"Flip recapture @ {int(gamma_flip)}"}

    mom_bull = momentum > 0; mom_bear = momentum < 0
    ev_bull  = ev_ratio >= BIAS_WEIGHTS["ev_ratio_bull"]
    ev_bear  = ev_ratio <= BIAS_WEIGHTS["ev_ratio_bear"]
    mec      = (mom_bull and ev_bear) or (mom_bear and ev_bull)

    if pcr > 1.4 and momentum > 0 and not mec:
        return {"name": "Bull Call Spread (PCR Contrarian)",
                "legs": f"Buy {atm} CE | Sell {atm+2*step} CE  PCR {pcr:.2f} extreme",
                "color": GREEN, "market_mode": regime, "mode_color": GREEN,
                "iv_context": f"PCR contrarian bull  {iv_ctx}"}
    if pcr < 0.55 and momentum < 0 and not mec:
        return {"name": "Bear Put Spread (PCR Contrarian)",
                "legs": f"Buy {atm} PE | Sell {atm-2*step} PE  PCR {pcr:.2f} extreme",
                "color": RED, "market_mode": regime, "mode_color": RED,
                "iv_context": f"PCR contrarian bear  {iv_ctx}"}

    wide_put_wing = skew_slope > 0.3
    put_sell = support + step
    put_buy  = support - (2 * step if wide_put_wing else step)
    size_note = " ⚠ Half size (conflict)" if mec else ""

    if "RANGE" in regime or "PINNED" in regime:
        gex_pos_ticks = sum(1 for h in history[-3:] if safe_num(h.get("gex", -1)) > 0)
        gex_confirmed = gex_pos_ticks >= 2 or len(history) < 2
        if not gex_confirmed:
            return {"name": "WAIT  GEX Not Yet Confirmed",
                    "legs": f"GEX positive for only {gex_pos_ticks}/3 ticks",
                    "color": AMBER, "market_mode": regime, "mode_color": AMBER, "iv_context": iv_ctx}
        if iv_rank >= 65:
            return {"name": f"Iron Condor  High IV{size_note}",
                    "legs": f"Sell {put_sell} PE / Buy {put_buy} PE + Sell {resistance-step} CE / Buy {resistance+step} CE",
                    "color": AMBER, "market_mode": regime, "mode_color": AMBER,
                    "iv_context": f"IV Rank {iv_rank:.0f}  ideal IC setup"}
        elif iv_rank <= 35:
            return {"name": f"Iron Fly  Low IV{size_note}",
                    "legs": f"Sell {atm} CE / Buy {atm+2*step} CE + Sell {atm} PE / Buy {atm-2*step} PE",
                    "color": GOLD, "market_mode": regime, "mode_color": GOLD,
                    "iv_context": f"IV Rank {iv_rank:.0f}  Iron Fly preferred"}
        else:
            return {"name": f"Iron Condor / Iron Fly{size_note}",
                    "legs": f"Sell {put_sell} PE / Buy {put_buy} PE + Sell {resistance-step} CE / Buy {resistance+step} CE",
                    "color": AMBER, "market_mode": regime, "mode_color": AMBER, "iv_context": iv_ctx}

    if direction == "BULLISH":
        if iv_rank <= 40:
            return {"name": f"Bull Call Spread (Debit){size_note}",
                    "legs": f"Buy {atm} CE | Sell {atm+2*step} CE",
                    "color": GREEN, "market_mode": regime, "mode_color": GREEN,
                    "iv_context": f"IV Rank {iv_rank:.0f}  cheap debit"}
        else:
            return {"name": f"Bull Put Spread (Credit){size_note}",
                    "legs": f"Sell {support} PE | Buy {support-step} PE",
                    "color": GREEN, "market_mode": regime, "mode_color": GREEN,
                    "iv_context": f"IV Rank {iv_rank:.0f}  sell premium below support"}
    if direction == "BEARISH":
        if iv_rank <= 40:
            return {"name": f"Bear Put Spread (Debit){size_note}",
                    "legs": f"Buy {atm} PE | Sell {atm-2*step} PE",
                    "color": RED, "market_mode": regime, "mode_color": RED,
                    "iv_context": f"IV Rank {iv_rank:.0f}  cheap debit"}
        else:
            return {"name": f"Bear Call Spread (Credit){size_note}",
                    "legs": f"Sell {resistance} CE | Buy {resistance+step} CE",
                    "color": RED, "market_mode": regime, "mode_color": RED,
                    "iv_context": f"IV Rank {iv_rank:.0f}  sell above resistance"}

    return {"name": "WAIT / BREAKOUT WATCH",
            "legs": f"Watch {support} support / {resistance} resistance",
            "color": BLUE, "market_mode": regime, "mode_color": BLUE, "iv_context": iv_ctx}


# ─── OI Velocity  IDENTICAL to Dash app ──────────────────────────────────────
def _zscore(arr, window):
    if len(arr) < 2:
        return 0.0
    w = arr[-window:]
    mean = w.mean(); std = w.std()
    if std < 1e-9:
        return 0.0
    return float((w[-1] - mean) / std)


def compute_oi_velocity(history):
    sym_history = history if isinstance(history, list) else []
    if len(sym_history) < 3:
        return {"call_oi_velocity":0,"put_oi_velocity":0,"call_oi_accel":0,"put_oi_accel":0,
                "call_vel_zscore":0,"put_vel_zscore":0,"alert_level":"NONE","alert_text":"Collecting data","n_ticks":0}

    call_oi = np.array([safe_num(x.get("call_oi_total",0)) for x in sym_history], dtype=float)
    put_oi  = np.array([safe_num(x.get("put_oi_total",0))  for x in sym_history], dtype=float)
    if call_oi.max() == 0 and put_oi.max() == 0:
        nd_arr  = np.array([safe_num(x.get("net_delta",0))    for x in sym_history], dtype=float)
        mom_arr = np.array([safe_num(x.get("oi_net_delta",0)) for x in sym_history], dtype=float)
        call_oi = np.maximum(nd_arr, 0) + np.maximum(mom_arr, 0)
        put_oi  = np.maximum(-nd_arr,0) + np.maximum(-mom_arr,0)

    c_vel = np.diff(call_oi); p_vel = np.diff(put_oi)
    if len(c_vel) < 2:
        return {"call_oi_velocity":0,"put_oi_velocity":0,"call_oi_accel":0,"put_oi_accel":0,
                "call_vel_zscore":0,"put_vel_zscore":0,"alert_level":"NONE","alert_text":"Collecting data","n_ticks":len(sym_history)}

    c_accel = float(c_vel[-1] - c_vel[-2]) if len(c_vel) >= 2 else 0.0
    p_accel = float(p_vel[-1] - p_vel[-2]) if len(p_vel) >= 2 else 0.0
    window  = min(10, len(c_vel))
    c_vel_z = _zscore(c_vel, window)
    p_vel_z = _zscore(p_vel, window)
    max_z   = max(abs(c_vel_z), abs(p_vel_z))

    if max_z >= 2.0:
        alert_level = "DANGER"
        side = "CALL" if abs(c_vel_z) > abs(p_vel_z) else "PUT"
        direction = "surge" if (c_vel_z if side=="CALL" else p_vel_z) > 0 else "unwind"
        alert_text = f"⚡ {side} OI {direction} detected  velocity {max_z:.1f}σ above norm."
    elif max_z >= 1.2:
        alert_level = "WATCH"
        side = "CALL" if abs(c_vel_z) > abs(p_vel_z) else "PUT"
        alert_text = f"⚠ {side} OI velocity elevated ({max_z:.1f}σ). Monitor closely."
    else:
        alert_level = "NONE"
        alert_text  = "OI velocity within normal range."

    return {"call_oi_velocity":float(c_vel[-1]),"put_oi_velocity":float(p_vel[-1]),
            "call_oi_accel":c_accel,"put_oi_accel":p_accel,
            "call_vel_zscore":round(c_vel_z,2),"put_vel_zscore":round(p_vel_z,2),
            "alert_level":alert_level,"alert_text":alert_text,"n_ticks":len(sym_history)}


# ─── Pre-move alert  IDENTICAL to Dash app ───────────────────────────────────
def compute_pre_move_alert(m, history):
    sym_history = history if isinstance(history, list) else []
    fires = []; details = []
    atm_iv = safe_num(m.get("atm_iv", 0))
    if len(sym_history) >= 2:
        prev_iv = safe_num(sym_history[-2].get("atm_iv", atm_iv))
        iv_jump = atm_iv - prev_iv
        if iv_jump >= 2.0:
            fires.append("IV SPIKE"); details.append(f"ATM IV jumped +{iv_jump:.2f}pp this tick")
        elif iv_jump >= 1.0:
            details.append(f"ATM IV up +{iv_jump:.2f}pp (watch)")
    else:
        details.append("IV: collecting data")

    oi_vel = compute_oi_velocity(sym_history)
    if oi_vel["alert_level"] == "DANGER":
        fires.append("OI VELOCITY"); details.append(oi_vel["alert_text"])
    elif oi_vel["alert_level"] == "WATCH":
        details.append(oi_vel["alert_text"])

    gex = safe_num(m.get("gex", 0))
    gamma_flip = m.get("gamma_flip")
    atm = safe_num(m.get("atm", 0))
    if gamma_flip and atm > 0:
        flip_distance = abs(atm - gamma_flip)
        wall_width = safe_num(m.get("wall_width", 400))
        _step = max(wall_width / 20, 50)
        proximity_threshold = max(2.0 * _step, 100)
        # H11 fix: original `(0 <= gex < 1000 or gex < 0)` simplifies to `gex < 1000`
        # (the `or gex < 0` clause is dead since 0<=gex<1000 already covers the
        # 0..1000 range and `gex < 0` covers everything below 0). Intent appears to
        # be "small positive GEX [0,1000) OR any negative GEX" → `gex < 1000`.
        if gex < 1000 and flip_distance < proximity_threshold:
            fires.append("GEX FLIP RISK")
            side_note = "below flip" if atm < gamma_flip else "above flip"
            details.append(f"GEX={'positive' if gex>=0 else 'NEGATIVE'} ({gex:,.0f}), spot {flip_distance:.0f}pts from flip @ {int(gamma_flip)} [{side_note}]")

    gt_ratio = safe_num(m.get("gt_ratio", 0))
    if gt_ratio >= 0.08 and gex < 0:
        fires.append("HIGH G/T"); details.append(f"G/T ratio {gt_ratio:.4f} (high) with negative GEX  unstable conditions")
    elif gt_ratio >= 0.05:
        details.append(f"G/T ratio {gt_ratio:.4f}  gamma starting to dominate theta")

    n = len(fires)
    if n >= 3:     alert_level = "DANGER"
    elif n >= 2:   alert_level = "WATCH"
    elif n == 1:   alert_level = "MONITOR"
    else:          alert_level = "NONE"
    return {"alert_level": alert_level, "fires": fires, "details": details, "pre_move_score": n}


# ─── Fake Breakout Score  IDENTICAL to Dash app ─────────────────────────────
def compute_fake_breakout_score(m, history):
    sym_history = history if isinstance(history, list) else []
    spot       = safe_num(m.get("atm", 0))
    support    = safe_num(m.get("support", 0))
    resistance = safe_num(m.get("resistance", 0))
    wall_width = safe_num(m.get("wall_width", 400))
    step       = wall_width / 20 if wall_width > 0 else 50

    if spot == 0 or support == 0 or resistance == 0:
        return {"score":0,"side":"NONE","alert_level":"NONE","alert_text":"No data.","factor_breakdown":{}}

    dist_to_res = resistance - spot; dist_to_sup = spot - support
    near_res    = 0 < dist_to_res < 1.5 * step
    near_sup    = 0 < dist_to_sup < 1.5 * step
    above_res   = spot > resistance
    below_sup   = spot < support

    if not (near_res or near_sup or above_res or below_sup):
        return {"score":0,"side":"NONE","alert_level":"NONE","alert_text":"Spot not near any wall.","factor_breakdown":{}}

    side = "CALL_WALL" if (near_res or above_res) else "PUT_WALL"
    factors = {}; score = 0

    if len(sym_history) >= 2:
        prev = sym_history[-2]
        curr_res = safe_num(m.get("resistance", 0)); prev_res = safe_num(prev.get("resistance", curr_res))
        curr_sup = safe_num(m.get("support", 0));    prev_sup = safe_num(prev.get("support", curr_sup))
        _tol = max(wall_width / 20, 50)
        if side == "CALL_WALL":
            wall_held = abs(curr_res - prev_res) < _tol
            f1 = 30 if wall_held else (0 if curr_res > prev_res else 15)
            factors["Wall OI"] = (f1, "Call wall defending" if wall_held else ("Resistance shifted UP  genuine break?" if curr_res > prev_res else "Wall eroding"))
        else:
            wall_held = abs(curr_sup - prev_sup) < _tol
            f1 = 30 if wall_held else (0 if curr_sup < prev_sup else 15)
            factors["Wall OI"] = (f1, "Put wall defending" if wall_held else ("Support shifted DOWN  genuine breakdown?" if curr_sup < prev_sup else "Wall eroding"))
    else:
        f1 = 15; factors["Wall OI"] = (f1, "Insufficient history")
    score += f1

    atm_iv = safe_num(m.get("atm_iv", 16))
    if len(sym_history) >= 2:
        prev_iv = safe_num(sym_history[-2].get("atm_iv", atm_iv))
        iv_delta = atm_iv - prev_iv
        if above_res or below_sup:
            if side == "CALL_WALL" and iv_delta < 0.3:
                f2 = 25; factors["IV Confirm"] = (f2, f"IV flat/falling ({iv_delta:+.2f}pp) on upside break  fake signal")
            elif side == "PUT_WALL" and iv_delta < 0.3:
                f2 = 25; factors["IV Confirm"] = (f2, f"IV flat/falling ({iv_delta:+.2f}pp) on downside break  fake signal")
            elif iv_delta >= 1.5:
                f2 = 0; factors["IV Confirm"] = (0, f"IV surging (+{iv_delta:.2f}pp)  genuine break likely")
            else:
                f2 = 10; factors["IV Confirm"] = (10, f"IV mildly rising (+{iv_delta:.2f}pp)  ambiguous")
        else:
            f2 = 0; factors["IV Confirm"] = (0, "Spot approaching wall  IV signal pending")
    else:
        f2 = 0; factors["IV Confirm"] = (0, "No IV history")
    score += f2

    atm_pressure = safe_num(m.get("atm_pressure", 0))
    if above_res and atm_pressure > 200:
        f3 = 20; factors["ATM Pressure"] = (20, f"ATM put building ({atm_pressure:+.0f}) after upside break  fade signal")
    elif below_sup and atm_pressure < -200:
        f3 = 20; factors["ATM Pressure"] = (20, f"ATM call building ({atm_pressure:+.0f}) after downside break  fade signal")
    elif abs(atm_pressure) > 100:
        f3 = 10; factors["ATM Pressure"] = (10, f"Moderate ATM pressure ({atm_pressure:+.0f})")
    else:
        f3 = 0; factors["ATM Pressure"] = (0, f"ATM pressure neutral ({atm_pressure:+.0f})")
    score += f3

    gex = safe_num(m.get("gex", 0))
    gamma_flip = m.get("gamma_flip")
    if gex > 0 and (above_res or below_sup):
        f4 = 15; factors["GEX Regime"] = (15, "GEX positive after wall break  dealers in range mode = FAKE")
    elif gex > 0 and (near_res or near_sup):
        f4 = 10; factors["GEX Regime"] = (10, "GEX positive approaching wall  range support active")
    elif gamma_flip and spot < gamma_flip and above_res:
        f4 = 5; factors["GEX Regime"] = (5, "Spot above resistance but below gamma flip  unstable")
    else:
        f4 = 0; factors["GEX Regime"] = (0, "GEX negative  trend regime supports genuine break")
    score += f4

    if len(sym_history) >= 3:
        pcr_arr = [safe_num(x.get("pcr", 1)) for x in sym_history[-4:]]
        pcr_delta = pcr_arr[-1] - pcr_arr[0] if len(pcr_arr) >= 2 else 0
        if side == "CALL_WALL" and pcr_delta > 0.05:
            f5 = 10; factors["PCR Velocity"] = (10, f"PCR rising (+{pcr_delta:.3f}) during upside break  FAKE signal")
        elif side == "PUT_WALL" and pcr_delta < -0.05:
            f5 = 10; factors["PCR Velocity"] = (10, f"PCR falling ({pcr_delta:.3f}) during downside break  FAKE signal")
        else:
            f5 = 0; factors["PCR Velocity"] = (0, f"PCR velocity neutral ({pcr_delta:+.3f})")
    else:
        f5 = 0; factors["PCR Velocity"] = (0, "Insufficient PCR history")
    score += f5
    score = max(0, min(100, score))

    if score >= 65:
        alert_level = "DANGER"
        alert_text  = f" FAKE {side.replace('_',' ')} RISK  Score {score}/100. Consider fading."
    elif score >= 40:
        alert_level = "WATCH"
        alert_text  = f"⚠ {side.replace('_',' ')} breakout unconfirmed  Score {score}/100. Wait for confirmation."
    else:
        alert_level = "NONE"
        alert_text  = f"Wall approach normal  Score {score}/100."

    return {"score":score,"side":side,"alert_level":alert_level,"alert_text":alert_text,"factor_breakdown":factors}


# ── v4: Wall Strength Index REMOVED (replaced by Leading Signals panel) ──





# ─── Market Sentiments  IDENTICAL to Dash app ────────────────────────────────
N_TICKS_SENTIMENT = 15

def compute_market_sentiments(history):
    if not history or len(history) < 3:
        return None
    recent  = history[-N_TICKS_SENTIMENT:]
    n       = len(recent)
    last    = recent[-1]
    warming = n < 5

    def safe_arr(key):
        return np.array([safe_num(x.get(key, 0)) for x in recent], dtype=float)

    def zscore_latest(arr):
        if len(arr) < 2: return 0.0
        std = arr.std()
        return float((arr[-1] - arr.mean()) / std) if std > 1e-9 else 0.0

    def classify_pcr_abs(pcr):
        if   pcr > 1.3: return "Bullish", GREEN
        elif pcr > 0.8: return "SideWays", AMBER
        else:           return "Bearish", RED

    def classify_nd_abs(nd):
        # CHANGE 3 (audit fix): flipped to WRITING convention.
        # net_delta = sum(call_oi*call_delta) + sum(put_oi*put_delta) with put_delta<0.
        # nd > 0  =>  call-side delta-weighted OI dominant  =>  CALL WRITING  =>  BEARISH.
        # nd < 0  =>  put-side  delta-weighted OI dominant  =>  PUT  WRITING  =>  BULLISH.
        # (was previously treated as a buying-convention signal, which contradicts the
        # S3/4 writer-positioning engine and the Combined Decision panel.)
        if   nd >  5000: return "Bearish", RED
        elif nd < -5000: return "Bullish", GREEN
        else:            return "SideWays", AMBER

    def classify_gex_abs(gex):
        if   gex >  500: return "SideWays", AMBER
        elif gex < -500: return "Bearish", RED
        else:            return "SideWays", AMBER

    def classify_iv_abs(iv):
        if   iv > 18: return "Bearish", RED
        elif iv > 14: return "SideWays", AMBER
        else:         return "Bullish", GREEN

    def classify_z(z):
        if z >  0.8: return "Bullish", GREEN
        if z < -0.8: return "Bearish", RED
        return "SideWays", AMBER

    iv_arr  = safe_arr("atm_iv");    pcr_arr = safe_arr("pcr")
    nd_arr  = safe_arr("net_delta"); gex_arr = safe_arr("gex")
    iv_z    = zscore_latest(iv_arr); pcr_z   = zscore_latest(pcr_arr)
    nd_z    = zscore_latest(nd_arr); gex_z   = zscore_latest(gex_arr)

    raw_iv  = safe_num(last.get("atm_iv", 0));   raw_pcr = safe_num(last.get("pcr", 1))
    raw_nd  = safe_num(last.get("net_delta", 0)); raw_gex = safe_num(last.get("gex", 0))

    use_abs_iv  = iv_arr.std()  < 1e-9; use_abs_pcr = pcr_arr.std() < 1e-9
    use_abs_nd  = nd_arr.std()  < 1e-9; use_abs_gex = gex_arr.std() < 1e-9

    vega_label,     vega_color     = classify_iv_abs(raw_iv)   if use_abs_iv  else classify_z(iv_z)
    oi_label,       oi_color       = classify_pcr_abs(raw_pcr) if use_abs_pcr else classify_z(-pcr_z)
    # CHANGE 3 (audit fix): negate nd_z to match writing convention (see classify_nd_abs).
    strength_label, strength_color = classify_nd_abs(raw_nd)   if use_abs_nd  else classify_z(-nd_z)
    theta_label,    theta_color    = classify_gex_abs(raw_gex) if use_abs_gex else classify_z(gex_z)

    abs_score = (
        (1 if oi_label=="Bullish" else -1 if oi_label=="Bearish" else 0) * 2.5 +
        (1 if strength_label=="Bullish" else -1 if strength_label=="Bearish" else 0) * 3.0 +
        (1 if vega_label=="Bullish" else -1 if vega_label=="Bearish" else 0) * 2.5 +
        (1 if theta_label=="Bullish" else -1 if theta_label=="Bearish" else 0) * 2.0
    )
    # CHANGE 3 (audit fix): -nd_z to flip from buying convention to writing convention.
    z_score_raw = (0.30 * (-nd_z) + 0.25 * iv_z + 0.25 * (-pcr_z) + 0.20 * gex_z) * 5
    all_flat = use_abs_iv and use_abs_pcr and use_abs_nd and use_abs_gex
    pos_score = round(float(np.clip(abs_score if all_flat else z_score_raw, -10, 10)), 2)

    if   pos_score >  2.0: pos_dot = GREEN
    elif pos_score < -2.0: pos_dot = RED
    else:                  pos_dot = AMBER

    labels = [vega_label, theta_label, oi_label, strength_label]
    bulls  = labels.count("Bullish"); bears = labels.count("Bearish")

    if   bulls >= 3 and pos_score >  2.0: overall, ov_color = "BUY",      GREEN
    elif bears >= 3 and pos_score < -2.0: overall, ov_color = "SELL",     RED
    elif oi_label == "Bearish" and bulls >= 2:
        overall, ov_color = "No Trade", AMBER
    else:
        overall, ov_color = "No Trade", MUTED

    if   pos_score >  4.0: pos_caption = "POS is strongly bullish"
    elif pos_score >  1.0: pos_caption = "POS is neutral to +ve"
    elif pos_score > -1.0: pos_caption = "POS is neutral"
    elif pos_score > -4.0: pos_caption = "POS is neutral to -ve"
    else:                  pos_caption = "POS is strongly bearish"

    mode_note = " · Absolute mode (warming up)" if all_flat else " · Z-score mode"
    return dict(
        vega_label=vega_label, vega_color=vega_color,
        theta_label=theta_label, theta_color=theta_color,
        oi_label=oi_label, oi_color=oi_color,
        strength_label=strength_label, strength_color=strength_color,
        pos_score=pos_score, pos_dot=pos_dot,
        overall=overall, overall_color=ov_color,
        pos_caption=pos_caption + mode_note,
        n_ticks=n, warming=warming,
    )


# ─── Synthetic Future / Basis Triangulation  IDENTICAL to Dash app ───────────
def compute_synthetic_future(df_band, spot, atm, expiry_str, r=0.065):
    if df_band is None or len(df_band) == 0 or spot <= 0 or atm <= 0:
        return None
    if isinstance(df_band, list):
        df_s10 = pd.DataFrame(df_band)
    else:
        df_s10 = df_band.copy()
    if df_s10.empty or "strike" not in df_s10.columns:
        return None
    atm_rows = df_s10[df_s10["strike"] == atm]
    if atm_rows.empty:
        idx = (df_s10["strike"] - atm).abs().idxmin()
        atm_rows = df_s10.iloc[[idx]]
    row      = atm_rows.iloc[0]
    call_ltp = safe_num(row.get("call_ltp", 0))
    put_ltp  = safe_num(row.get("put_ltp",  0))
    if call_ltp <= 0 or put_ltp <= 0:
        return None
    synthetic = call_ltp - put_ltp + atm
    T = 7.0 / 365
    try:
        for _fmt in ("%Y-%m-%d", "%d-%b-%Y"):
            try:
                _exp = datetime.strptime(str(expiry_str), _fmt).date()
                T = max((_exp - date.today()).days, 0) / 365
                break
            except ValueError:
                continue
    except Exception:
        pass
    fair_future  = spot * np.exp(r * T)   # H9 fix: was spot*(1+r*T) (simple interest);
                                           # the synthetic call-put+K is a continuous-compounding
                                           # forward per BS assumptions. Mismatch was numerically
                                           # tiny for short DTE but methodologically inconsistent.
    fair_basis   = fair_future - spot
    synth_basis  = synthetic  - spot
    synth_excess = synthetic  - fair_future
    return {"spot":round(spot,2),"synthetic":round(synthetic,2),"fair_future":round(fair_future,2),
            "fair_basis":round(fair_basis,2),"synth_basis":round(synth_basis,2),
            "synth_excess":round(synth_excess,2),"call_ltp":round(call_ltp,2),
            "put_ltp":round(put_ltp,2),"atm":int(atm),"T_days":round(T*365,1)}


def compute_basis_signals(sf, traded_future):
    if sf is None:
        return None
    spot         = sf["spot"]; synthetic = sf["synthetic"]; fair_future = sf["fair_future"]
    fair_basis   = sf["fair_basis"]; synth_basis = sf["synth_basis"]; synth_excess = sf["synth_excess"]
    has_traded   = (traded_future is not None and safe_num(traded_future, 0) > 0)
    if has_traded:
        traded_future = safe_num(traded_future)
        traded_basis  = traded_future - spot
        traded_excess = traded_future - fair_future
        basis_gap     = traded_future - synthetic
    else:
        traded_future = traded_basis = traded_excess = basis_gap = None

    thr = max(spot * 0.0004, 5.0)
    signals = []; net_bias = 0
    if synth_excess > thr:
        signals.append(("Call premium bid above carry  options leaning bullish", GREEN)); net_bias += 1
    elif synth_excess < -thr:
        signals.append(("Put premium bid above carry  options leaning bearish", RED));    net_bias -= 1
    else:
        signals.append(("Options pricing near fair carry  no directional lean", MUTED))

    if has_traded:
        if traded_excess > thr:
            signals.append(("Futures premium above fair carry  institutional longs dominant", GREEN)); net_bias += 1
        elif traded_excess < -thr:
            signals.append(("Futures below fair carry  hedge/short pressure present", RED));           net_bias -= 1
        else:
            signals.append(("Futures at fair carry  carry-neutral positioning", MUTED))
        if basis_gap > thr:
            signals.append((f"Futures leads synthetic by {basis_gap:+.1f}pts  bullish lead", GREEN)); net_bias += 2
        elif basis_gap < -thr:
            signals.append((f"Futures lags synthetic by {abs(basis_gap):.1f}pts  bearish lean", RED)); net_bias -= 2
        else:
            signals.append((f"Futures and synthetic aligned (gap {basis_gap:+.1f}pts)", MUTED))

    if   net_bias >= 3:  sc, sl = GREEN,   "STRONGLY BULLISH"
    elif net_bias == 2:  sc, sl = GREEN,   "BULLISH"
    elif net_bias == 1:  sc, sl = "#10B981","MILDLY BULLISH"
    elif net_bias == -1: sc, sl = AMBER,   "MILDLY BEARISH"
    elif net_bias == -2: sc, sl = RED,     "BEARISH"
    elif net_bias <= -3: sc, sl = RED,     "STRONGLY BEARISH"
    else:                sc, sl = MUTED,   "NEUTRAL"

    return {"spot":spot,"synthetic":synthetic,"fair_future":fair_future,
            "traded_future":traded_future,"has_traded":has_traded,
            "fair_basis":fair_basis,"synth_basis":synth_basis,"synth_excess":synth_excess,
            "traded_basis":traded_basis,"traded_excess":traded_excess,"basis_gap":basis_gap,
            "signals":signals,"net_bias":net_bias,"summary_label":sl,"summary_color":sc,
            "T_days":sf["T_days"],"atm":sf["atm"],"call_ltp":sf["call_ltp"],"put_ltp":sf["put_ltp"]}


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCED PRICE CONFIRMATION LAYER  (v7 addition — surgical)
#   Module A: VWAP + Opening Range  (Dhan /v2/charts/intraday)
#   Module B: Term Structure         (second expiry from /v2/optionchain)
#   Module C: India VIX              (Dhan instrument master + LTP feed)
# All three are read-only additions. No existing variables modified.
# ═════════════════════════════════════════════════════════════════════════════

# ── Module A: VWAP + Opening Range ───────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_nifty_intraday_candles():
    """
    Fetch today's 1-min OHLCV for the near-month NIFTY Futures contract from
    Dhan /v2/charts/intraday.  Uses the actual futures security ID resolved via
    _resolve_futures_id() so VWAP is computed on real traded volume, not the
    synthetic index tick count.
    Returns list of dicts {ts, open, high, low, close, volume} or [] on failure.
    """
    if not USE_DHAN:
        return []
    # Resolve near-month NIFTY futures security ID (reuses existing helper)
    _fut_sec_id, _fut_expiry = _resolve_futures_id()
    if not _fut_sec_id:
        return []
    today_str = date.today().strftime("%Y-%m-%d")
    try:
        # H1+H2+H3 fix: shared helper — status check, success validation, retry/backoff.
        data = _dhan_post(
            "https://api.dhan.co/v2/charts/intraday",
            {
                "securityId":      str(_fut_sec_id),
                "exchangeSegment": "NSE_FNO",
                "instrument":      "FUTIDX",
                "interval":        "1",
                "oi":              False,
                "fromDate":        f"{today_str} 09:15:00",
                "toDate":          f"{today_str} 15:30:00",
            },
            timeout=15,
        )
        ts_arr  = data.get("timestamp", [])
        op_arr  = data.get("open",      [])
        hi_arr  = data.get("high",      [])
        lo_arr  = data.get("low",       [])
        cl_arr  = data.get("close",     [])
        vo_arr  = data.get("volume",    [])
        if not ts_arr:
            return []
        # M7 fix: validate array lengths — partial responses were silently zero-filled,
        # producing fake 0-volume / 0-price candles that corrupt VWAP & Opening Range.
        n = len(ts_arr)
        for arr_name, arr in (("open", op_arr), ("high", hi_arr),
                              ("low", lo_arr), ("close", cl_arr),
                              ("volume", vo_arr)):
            if len(arr) != n:
                try:
                    print(f"[fetch_nifty_intraday_candles] array length mismatch: "
                          f"ts={n} {arr_name}={len(arr)} — discarding response", flush=True)
                except Exception:
                    pass
                return []
        candles = []
        for i, ts in enumerate(ts_arr):
            candles.append({
                "ts":     int(ts),
                "open":   float(op_arr[i]),
                "high":   float(hi_arr[i]),
                "low":    float(lo_arr[i]),
                "close":  float(cl_arr[i]),
                "volume": float(vo_arr[i]),
            })
        return candles
    except (DhanAPIError, Exception) as e:
        try:
            print(f"[fetch_nifty_intraday_candles] error: {e}", flush=True)
        except Exception:
            pass
        return []


def compute_vwap_opening_range(candles):
    """
    From today's 1-min candles compute:
      - Opening Range high/low  (first 15 candles = 9:15–9:30)
      - Running VWAP (cumulative typical-price × volume / cumulative volume)
      - Current candle vs VWAP position
    Returns dict or None if data insufficient.
    """
    if not candles or len(candles) < 2:
        return None
    # Opening Range: first 15 1-min candles (9:15–9:29)
    or_candles = candles[:15]
    or_high  = max(c["high"]  for c in or_candles)
    or_low   = min(c["low"]   for c in or_candles)
    or_mid   = (or_high + or_low) / 2.0
    # VWAP: cumulative (typical_price × volume) / cumulative_volume
    cum_tpv = 0.0
    cum_vol = 0.0
    for c in candles:
        tp = (c["high"] + c["low"] + c["close"]) / 3.0
        cum_tpv += tp * c["volume"]
        cum_vol += c["volume"]
    vwap = cum_tpv / cum_vol if cum_vol > 0 else 0.0
    last_close = candles[-1]["close"]
    # Signal derivation
    above_vwap = last_close > vwap if vwap > 0 else None
    or_position = (
        "ABOVE_OR"  if last_close > or_high else
        "BELOW_OR"  if last_close < or_low  else
        "INSIDE_OR"
    )
    # Score contribution: +10 (above both), -10 (below both), else proportional
    if vwap > 0 and above_vwap and or_position == "ABOVE_OR":
        price_score = 10.0
        price_label = "Bullish — price above VWAP & Opening Range"
        price_color = "#059669"
    elif vwap > 0 and (not above_vwap) and or_position == "BELOW_OR":
        price_score = -10.0
        price_label = "Bearish — price below VWAP & Opening Range"
        price_color = "#DC2626"
    elif vwap > 0 and above_vwap:
        price_score = 5.0
        price_label = "Mildly bullish — above VWAP, inside Opening Range"
        price_color = "#10B981"
    elif vwap > 0 and not above_vwap:
        price_score = -5.0
        price_label = "Mildly bearish — below VWAP, inside/above Opening Range"
        price_color = "#F59E0B"
    else:
        price_score = 0.0
        price_label = "VWAP unavailable (pre-session or no volume)"
        price_color = "#6B7280"
    return {
        "vwap":        round(vwap, 2),
        "or_high":     round(or_high, 2),
        "or_low":      round(or_low, 2),
        "or_mid":      round(or_mid, 2),
        "last_close":  round(last_close, 2),
        "above_vwap":  above_vwap,
        "or_position": or_position,
        "price_score": price_score,
        "price_label": price_label,
        "price_color": price_color,
        "n_candles":   len(candles),
    }


# ── Module B: Term Structure (front vs back expiry ATM IV) ───────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_back_expiry_atm_iv(back_expiry: str):
    """
    Fetch the back-month option chain and return ATM IV only.
    Runs concurrently with the main front-expiry fetch; same Dhan endpoint.
    Rate limit note: 1 unique request per 3s — this is a different expiry
    so qualifies as a unique request per Dhan docs.
    """
    if not USE_DHAN or not back_expiry:
        return None
    sec = DHAN_SECURITY["NIFTY"]
    try:
        # H1+H2+H3 fix: shared helper.
        resp = _dhan_post(
            "https://api.dhan.co/v2/optionchain",
            {"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"],
             "Expiry": back_expiry},
            timeout=15,
        )
        data = resp.get("data", {}) or {}
        spot = float(data.get("last_price") or data.get("ltp") or 0)
        oc   = data.get("oc", {}) or {}
        if spot <= 0 or not oc:
            return None
        # Find ATM strike
        strikes = [safe_num(k) for k in oc.keys() if safe_num(k) > 0]
        if not strikes:
            return None
        atm_k = min(strikes, key=lambda x: abs(x - spot))
        chain = oc.get(str(float(atm_k)), oc.get(f"{atm_k:.6f}", {})) or {}
        ce_iv = safe_num((chain.get("ce", {}) or {}).get("implied_volatility", 0))
        pe_iv = safe_num((chain.get("pe", {}) or {}).get("implied_volatility", 0))
        atm_iv_back = 0.0
        if ce_iv > 0.5 and pe_iv > 0.5:
            atm_iv_back = (ce_iv + pe_iv) / 2.0
        elif ce_iv > 0.5:
            atm_iv_back = ce_iv
        elif pe_iv > 0.5:
            atm_iv_back = pe_iv
        return round(atm_iv_back, 2) if atm_iv_back > 0 else None
    except (DhanAPIError, Exception) as e:
        try:
            print(f"[fetch_back_expiry_atm_iv] error: {e}", flush=True)
        except Exception:
            pass
        return None


# ─── Roll Detection — Inter-Expiry OI Comparison ────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)  # Fix #5: was ttl=60; 5-min matches fetch_back_expiry_atm_iv and avoids rate-limit pressure
def fetch_back_expiry_oi_band(back_expiry: str):
    """
    Fetch strike-level OI + OI change for the back expiry — lightweight version.
    Returns only the columns needed for roll detection:
        strike, call_oi, put_oi, call_oi_chg, put_oi_chg
    Unlike fetch_back_expiry_atm_iv(), this fetches the FULL OI band so we can
    compare OI changes at shared strikes between front and back expiry.
    """
    if not USE_DHAN or not back_expiry:
        return pd.DataFrame()
    sec = DHAN_SECURITY["NIFTY"]
    try:
        # H1+H2+H3 fix: shared helper.
        resp = _dhan_post(
            "https://api.dhan.co/v2/optionchain",
            {"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"],
             "Expiry": back_expiry},
            timeout=15,
        )
        data = resp.get("data", {}) or {}
        oc   = data.get("oc", {}) or {}
        if not oc:
            return pd.DataFrame()
        # H6 fix: use int(float(...)) for OI parsing
        def _safe_int(v):
            try:
                return int(float(v or 0))
            except (TypeError, ValueError):
                return 0
        rows = []
        for strike_str, chain in oc.items():
            K  = safe_num(strike_str, 0)
            if K <= 0:
                continue
            ce = (chain or {}).get("ce", {}) or {}
            pe = (chain or {}).get("pe", {}) or {}
            c_oi      = _safe_int(ce.get("oi", 0))
            c_prev_oi = _safe_int(ce.get("previous_oi", 0))
            p_oi      = _safe_int(pe.get("oi", 0))
            p_prev_oi = _safe_int(pe.get("previous_oi", 0))
            rows.append({
                "strike":      K,
                "call_oi":     c_oi,
                "put_oi":      p_oi,
                "call_oi_chg": c_oi - c_prev_oi,
                "put_oi_chg":  p_oi - p_prev_oi,
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values("strike").reset_index(drop=True)
    except (DhanAPIError, Exception) as e:
        try:
            print(f"[fetch_back_expiry_oi_band] error: {e}", flush=True)
        except Exception:
            pass
        return pd.DataFrame()


def is_roll_window(front_expiry_str: str = None) -> bool:
    """
    Fix #3: Derive roll window from actual expiry date rather than hardcoded Tuesday.

    When front_expiry_str is supplied the expiry weekday is read from the date,
    so the function stays correct if SEBI ever moves the weekly settlement day again.
    Falls back to the current hardcoded schedule (Tuesday) when no expiry is given.

    Roll window = afternoon of the day-before-expiry (14:00 IST → EOD) + all of expiry day.
    """
    n  = now_ist()
    wd = n.weekday()   # 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri
    hm = (n.hour, n.minute)

    if front_expiry_str:
        try:
            _exp_dt   = date.fromisoformat(front_expiry_str)
            expiry_wd = _exp_dt.weekday()         # weekday of expiry (e.g. 1 = Tuesday)
            prev_wd   = (expiry_wd - 1) % 7      # day before expiry (e.g. 0 = Monday)
            if wd == prev_wd and hm >= (14, 0):  # afternoon roll-start
                return True
            if wd == expiry_wd:                  # all of expiry day
                return True
            return False
        except Exception:
            pass  # malformed expiry string — fall through to hardcoded default

    # Hardcoded fallback: NIFTY 50 expiry = Tuesday (SEBI schedule since Sep 2023)
    if wd == 1 and hm >= (14, 0):   # Tuesday afternoon
        return True
    if wd == 2:                      # All day Wednesday
        return True
    return False


def detect_roll_activity(front_df, back_df, spot: float) -> dict:
    """
    Distinguish mechanical OI roll from genuine directional liquidation.

    Roll signature   : front OI falling AND back OI rising at the SAME strike.
    Liquidation sig  : front OI falling with NO corresponding back-expiry build.

    Returns:
      roll_fraction     (float 0–1) : share of unwinding front strikes that have
                                      a matching back-expiry build.
      roll_detected     (bool)      : True if roll_fraction >= 0.50
      momentum_discount (float 0–1) : multiplier applied to S2 — scales linearly
                                      from 1.0 (no roll) → 0.20 (pure roll).
                                      Never fully zero: real positioning co-exists.
      roll_pts_call / _put (int)    : net call/put OI draining from front expiry
      details           (list[str]) : human-readable lines for the UI card
    """
    _base = {
        "roll_fraction": 0.0, "roll_detected": False,
        "momentum_discount": 1.0,
        "roll_pts_call": 0, "roll_pts_put": 0,
        "details": [],
    }
    if front_df is None or front_df.empty or back_df is None or back_df.empty:
        _base["details"] = ["Back-expiry OI unavailable — roll detection inactive"]
        return _base

    # Near-ATM band: ±10 strikes where rolls concentrate
    lo = spot - 10 * NIFTY_STEP
    hi = spot + 10 * NIFTY_STEP
    f = front_df[front_df["strike"].between(lo, hi)].copy()
    b = back_df[back_df["strike"].between(lo, hi)].copy()
    if f.empty or b.empty:
        _base["details"] = ["Insufficient near-ATM OI data for roll detection"]
        return _base

    shared = pd.merge(
        f[["strike", "call_oi_chg", "put_oi_chg"]],
        b[["strike", "call_oi_chg", "put_oi_chg"]],
        on="strike", suffixes=("_f", "_b"),
    )
    if shared.empty:
        _base["details"] = ["No shared strikes between front and back expiry"]
        return _base

    # Significance threshold: ignore trivial lot movements
    _MIN_LOTS = 100
    call_unwind = shared["call_oi_chg_f"] < -_MIN_LOTS
    call_roll   = call_unwind & (shared["call_oi_chg_b"] >  _MIN_LOTS // 2)
    put_unwind  = shared["put_oi_chg_f"]  < -_MIN_LOTS
    put_roll    = put_unwind  & (shared["put_oi_chg_b"]  >  _MIN_LOTS // 2)

    total_unwind   = call_unwind.sum() + put_unwind.sum()
    total_roll     = call_roll.sum()   + put_roll.sum()
    roll_fraction  = (total_roll / total_unwind) if total_unwind > 0 else 0.0

    roll_pts_call = int(shared.loc[call_unwind, "call_oi_chg_f"].sum())
    roll_pts_put  = int(shared.loc[put_unwind,  "put_oi_chg_f"].sum())

    # Linear discount: 1.0 at 0% roll → 0.20 at 100% roll
    momentum_discount = round(max(0.20, 1.0 - roll_fraction * 0.80), 2)
    roll_detected     = roll_fraction >= 0.50

    details = []
    if roll_detected:
        details.append(
            f"⚠ ROLL: {roll_fraction*100:.0f}% of near-ATM OI unwind matched "
            f"by back-expiry build — S2 discounted to {momentum_discount*100:.0f}%"
        )
        if roll_pts_call < -500:
            details.append(f"Call roll: {roll_pts_call:,} lots draining front expiry")
        if roll_pts_put < -500:
            details.append(f"Put roll: {roll_pts_put:,} lots draining front expiry")
    elif total_unwind > 0:
        details.append(
            f"Liquidation dominant ({roll_fraction*100:.0f}% roll fraction) — "
            f"OI unwind is genuine; S2 at full strength"
        )
    else:
        details.append("No significant near-ATM OI unwind detected")

    return {
        "roll_fraction":      round(roll_fraction, 3),
        "roll_detected":      roll_detected,
        "momentum_discount":  momentum_discount,
        "roll_pts_call":      roll_pts_call,
        "roll_pts_put":       roll_pts_put,
        "details":            details,
    }


def compute_term_structure_signal(front_atm_iv: float, back_atm_iv, front_expiry: str = None):
    """
    Compute term structure slope and derive a bias signal.
    front_atm_iv: ATM IV of nearest expiry (already in main metrics)
    back_atm_iv:  ATM IV of second expiry from fetch_back_expiry_atm_iv()
    front_expiry: ISO date string of front expiry (YYYY-MM-DD) from Dhan API.
                  Used to detect expiry day dynamically — no weekday hardcoding.

    Normal contango  (front < back): market calm, range bias confirmed.
    Flat term structure (|slope| < 1): transitional.
    Backwardation (front > back): near-term event risk, directional move likely.

    NOTE: On expiry day the front IV collapses to near-zero, making the slope
    unreliable. Signal is suppressed when DTE ≤ 0 (i.e., today IS expiry day).
    Previously hardcoded to Thursday; now dynamically derived from front_expiry
    so it works correctly for Tuesday expiry (and any future schedule changes).
    """
    from datetime import date as _date
    # Expiry-day guard: suppress when today is the actual front expiry date.
    # DTE is computed from the real expiry string — no day-of-week assumption.
    _suppress = False
    if front_expiry:
        try:
            _dte = (_date.fromisoformat(front_expiry) - _date.today()).days
            if _dte <= 0:
                _suppress = True
        except Exception:
            pass   # bad date string — don't suppress, degrade gracefully
    if _suppress:
        return {
            "available":   False,
            "front_iv":    round(front_atm_iv, 2) if front_atm_iv else 0.0,
            "back_iv":     None,
            "slope":       None,
            "regime":      "EXPIRY_DAY",
            "ts_label":    "Expiry day — term structure suppressed (front IV unreliable)",
            "ts_color":    "#6B7280",
            "ts_score":    0.0,
        }
    if not back_atm_iv or back_atm_iv <= 0 or front_atm_iv <= 0:
        return {
            "available":   False,
            "front_iv":    round(front_atm_iv, 2),
            "back_iv":     None,
            "slope":       None,
            "regime":      "UNAVAILABLE",
            "ts_label":    "Term structure data unavailable",
            "ts_color":    "#6B7280",
            "ts_score":    0.0,
        }
    slope = front_atm_iv - back_atm_iv   # positive = backwardation
    if slope > 2.0:
        regime    = "BACKWARDATION"
        ts_label  = f"Backwardation: front IV {front_atm_iv:.1f}% > back {back_atm_iv:.1f}% (+{slope:.1f}pts) — near-term event risk"
        ts_color  = "#DC2626"
        ts_score  = -8.0   # near-term fear → reduce range/condor confidence
    elif slope > 0.5:
        regime    = "MILD_BACK"
        ts_label  = f"Mild backwardation: front {front_atm_iv:.1f}% > back {back_atm_iv:.1f}% (+{slope:.1f}pts) — elevated near-term demand"
        ts_color  = "#F59E0B"
        ts_score  = -4.0
    elif slope < -2.0:
        regime    = "STEEP_CONTANGO"
        ts_label  = f"Steep contango: front {front_atm_iv:.1f}% < back {back_atm_iv:.1f}% ({slope:.1f}pts) — market calm, range trades favoured"
        ts_color  = "#059669"
        ts_score  = +5.0
    elif slope < -0.5:
        regime    = "CONTANGO"
        ts_label  = f"Contango: front {front_atm_iv:.1f}% < back {back_atm_iv:.1f}% ({slope:.1f}pts) — normal structure"
        ts_color  = "#10B981"
        ts_score  = +3.0
    else:
        regime    = "FLAT"
        ts_label  = f"Flat term structure: front {front_atm_iv:.1f}% ≈ back {back_atm_iv:.1f}% ({slope:+.1f}pts) — transitional"
        ts_color  = "#6B7280"
        ts_score  = 0.0
    return {
        "available":  True,
        "front_iv":   round(front_atm_iv, 2),
        "back_iv":    round(back_atm_iv, 2),
        "slope":      round(slope, 2),
        "regime":     regime,
        "ts_label":   ts_label,
        "ts_color":   ts_color,
        "ts_score":   ts_score,
    }


# ── Module C: India VIX ───────────────────────────────────────────────────────
_vix_id_cache = {}

def _resolve_india_vix_id():
    """Look up India VIX security ID from the Dhan instruments master CSV."""
    if "VIX" in _vix_id_cache:
        return _vix_id_cache["VIX"]
    master = _load_dhan_instrument_master()
    if master is None:
        return None
    try:
        cols = set(master.columns)
        tsym_col  = next((c for c in ["SEM_TRADING_SYMBOL", "SM_SYMBOL_NAME"] if c in cols), None)
        secid_col = next((c for c in ["SEM_SMST_SECURITY_ID", "SEM_SECURITY_ID"] if c in cols), None)
        seg_col   = next((c for c in ["SEM_EXM_EXCH_ID"] if c in cols), None)
        if not tsym_col or not secid_col:
            return None
        df = master.copy()
        if seg_col:
            df = df[df[seg_col].astype(str).str.strip().str.upper() == "NSE"]
        mask = df[tsym_col].astype(str).str.strip().str.upper().str.contains("VIX", na=False)
        vix_rows = df[mask]
        if vix_rows.empty:
            return None
        sec_id = str(int(float(vix_rows.iloc[0][secid_col])))
        _vix_id_cache["VIX"] = sec_id
        return sec_id
    except Exception:
        return None


_vix_ltp_cache_v7 = {"ltp": 0.0, "ts": 0.0}
_VIX_CACHE_SEC = 58

def fetch_india_vix_ltp():
    """
    Fetch India VIX LTP via Dhan /v2/marketfeed/ltp using the VIX security ID.
    Falls back to 0.0 if unavailable (graceful — VIX not a required signal).
    """
    if not USE_DHAN:
        return 0.0
    now = time.time()
    if now - _vix_ltp_cache_v7["ts"] < _VIX_CACHE_SEC and _vix_ltp_cache_v7["ltp"] > 0:
        return _vix_ltp_cache_v7["ltp"]
    vix_id = _resolve_india_vix_id()
    if not vix_id:
        return 0.0
    try:
        # H1+H2+H3 fix: shared helper.
        rjson = _dhan_post(
            "https://api.dhan.co/v2/marketfeed/ltp",
            {"NSE_INDEX": [int(vix_id)]},
            timeout=8,
        )
        seg = (rjson.get("data") or {}).get("NSE_INDEX") or {}
        for _, info in seg.items():
            ltp = float(info.get("last_price") or info.get("ltp") or 0)
            if ltp > 0:
                _vix_ltp_cache_v7["ltp"] = ltp
                _vix_ltp_cache_v7["ts"]  = now
                return ltp
    except (DhanAPIError, Exception) as e:
        try:
            print(f"[fetch_india_vix_ltp] error: {e}", flush=True)
        except Exception:
            pass
    return _vix_ltp_cache_v7.get("ltp", 0.0)


def classify_vix_signal(vix_val: float, vix_history: list):
    """
    Classify India VIX level and intraday change into a bias signal.
    vix_history: list of float (recent VIX readings in session)

    VIX < 13   : Very low — complacency / range-bound — bearish contrarian risk
    13–15      : Low-normal — calm trending environment
    15–18      : Normal — balanced
    18–22      : Elevated — caution, directional but volatile
    > 22       : High fear — mean-reversion / protective bias
    """
    if vix_val <= 0:
        return {
            "available": False,
            "vix":       0.0,
            "regime":    "UNAVAILABLE",
            "vix_label": "India VIX unavailable",
            "vix_color": "#6B7280",
            "vix_score": 0.0,
            "vix_change":None,
        }
    # Intraday change
    vix_change = None
    if len(vix_history) >= 2:
        vix_change = round(vix_val - vix_history[-2], 2)
    # Classify level
    if vix_val > 22:
        regime = "HIGH_FEAR"; vix_color = "#DC2626"; vix_score = -8.0
        vix_label = f"VIX {vix_val:.1f} — HIGH FEAR: elevated put demand, mean-reversion risk"
    elif vix_val > 18:
        regime = "ELEVATED"; vix_color = "#F59E0B"; vix_score = -4.0
        vix_label = f"VIX {vix_val:.1f} — Elevated: volatile conditions, reduce size"
    elif vix_val > 15:
        regime = "NORMAL"; vix_color = "#6B7280"; vix_score = 0.0
        vix_label = f"VIX {vix_val:.1f} — Normal range: no regime distortion"
    elif vix_val > 13:
        regime = "LOW_CALM"; vix_color = "#10B981"; vix_score = +3.0
        vix_label = f"VIX {vix_val:.1f} — Low/calm: range trades and premium selling favoured"
    else:
        regime = "COMPLACENCY"; vix_color = "#F59E0B"; vix_score = -3.0
        vix_label = f"VIX {vix_val:.1f} — Very low: complacency alert, contrarian reversal risk"
    # Spike modifier
    if vix_change is not None and vix_change >= 1.5:
        vix_label += f"  ⚡ Spike +{vix_change:.2f} pts this tick"
        vix_score  = min(vix_score - 3.0, -3.0)
        vix_color  = "#DC2626"
    elif vix_change is not None and vix_change <= -1.5:
        vix_label += f"  ↓ Deflating {vix_change:.2f} pts"
        vix_score  = max(vix_score + 2.0, 2.0)
    return {
        "available":  True,
        "vix":        round(vix_val, 2),
        "regime":     regime,
        "vix_label":  vix_label,
        "vix_color":  vix_color,
        "vix_score":  vix_score,
        "vix_change": vix_change,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED BIAS DECISION ENGINE  (Chapter 17 & 18)  —  merged inline in v4
# Previously: from combined_bias_engine import generate_combined_decision
# ═══════════════════════════════════════════════════════════════════════════════

def get_s34_band(score: float) -> str:
    """Map S3/4 bias score to a band letter A-E."""
    if   score >= 51:   return "A"   # Strong Bull
    elif score >= 16:   return "B"   # Mild Bull
    elif score >= -15:  return "C"   # Neutral
    elif score >= -50:  return "D"   # Mild Bear
    else:               return "E"   # Strong Bear


# IV Smile composite buckets
_BEARISH_FEAR_IDS    = {1, 2, 3, 9}   # Put Skew, Crash Fear, Bearish Drift, Two-Sided Fork
_BULLISH_NEUTRAL_IDS = {4, 5, 6}     # Call Skew, Melt-Up, Post-Crash Relief

def get_smile_bucket(scenario_id: int) -> str:
    if scenario_id in _BEARISH_FEAR_IDS:
        return "BEARISH_FEAR"
    elif scenario_id in _BULLISH_NEUTRAL_IDS:
        return "BULLISH_NEUTRAL"
    else:
        return "NEUTRAL"


# ── 4-Quadrant metadata ─────────────────────────────────────────────────────
_QUADRANT_META = {
    "Q1": {
        "name":   "Full Bull Alignment",
        "short":  "Q1 — FULL BULL",
        "color":  "#059669",
        "badge_bg": "#ECFDF5",
        "description": (
            "Structure is bullish (S3/4 positive) AND the IV Smile confirms "
            "that sentiment is NOT overly fearful. Both engines agree — highest "
            "probability bullish environment. Favour long delta, call spreads, "
            "and systematic put-selling when IV rank is elevated."
        ),
        "action": "LEAN BULLISH  —  long delta, call debit spreads, theta-selling below spot",
    },
    "Q2": {
        "name":   "Structural Bull / Sentiment Cautious",
        "short":  "Q2 — STRUCTURAL BULL / CAUTIOUS",
        "color":  "#D97706",
        "badge_bg": "#FFFBEB",
        "description": (
            "OI structure and momentum are bullish BUT the IV Smile is showing "
            "put-skew or fear. Institutions are hedging into the rally. The move "
            "can continue but at risk of a sharp reversal if the put-wing keeps "
            "building. Reduce position size; favour protective collars or limited-risk "
            "long structures. Watch S4 IV Skew component for further divergence."
        ),
        "action": "BULLISH WITH CAUTION  —  reduced size, hedge with put spreads, trail stops",
    },
    "Q3": {
        "name":   "Structural Bear / Sentiment Recovering",
        "short":  "Q3 — STRUCTURAL BEAR / RECOVERING",
        "color":  "#2563EB",
        "badge_bg": "#EFF6FF",
        "description": (
            "OI structure and momentum are bearish BUT the IV Smile is showing "
            "relief or call buying — fear is deflating. This is the most nuanced "
            "quadrant: it can precede a bottom or be a dead-cat bounce. Do NOT "
            "short aggressively into a deflating put-wing. Wait for S3/4 score to "
            "turn less negative before adding long exposure."
        ),
        "action": "CAUTIOUSLY BEARISH  —  cover shorts on IV crush, wait for structure to improve",
    },
    "Q4": {
        "name":   "Full Bear Alignment",
        "short":  "Q4 — FULL BEAR",
        "color":  "#DC2626",
        "badge_bg": "#FEF2F2",
        "description": (
            "Structure is bearish (S3/4 negative) AND the IV Smile confirms "
            "fear / put-skew. Both engines agree — highest probability bearish "
            "environment. Favour short delta, put spreads, and call-selling above "
            "resistance. Highest confidence short signal when S3/4 <= -40 and "
            "smile is Crash Fear (Sc02)."
        ),
        "action": "LEAN BEARISH  —  short delta, put debit spreads, bear call spreads above resistance",
    },
    "CN": {
        "name":   "Neutral / No Edge",
        "short":  "CN — NEUTRAL",
        "color":  "#6B7280",
        "badge_bg": "#F9FAFB",
        "description": (
            "Either the S3/4 score is in the neutral band (-15 to +15) or the IV "
            "Smile is indeterminate / pre-event. Neither engine provides a reliable "
            "directional edge. Avoid directional positions; favour premium-selling "
            "structures (iron condors) only if IV rank supports it."
        ),
        "action": "NO DIRECTIONAL EDGE  —  avoid naked direction, favour non-directional structures",
    },
}


def classify_quadrant(s34_score: float, scenario_id: int) -> dict:
    """4-Quadrant classification with optional enhanced-price-layer override."""
    band         = get_s34_band(s34_score)
    smile_bucket = get_smile_bucket(scenario_id)

    if band == "C":
        q = "CN"
    elif band in ("A", "B"):
        if   smile_bucket == "BULLISH_NEUTRAL": q = "Q1"
        elif smile_bucket == "BEARISH_FEAR":    q = "Q2"
        else:                                    q = "Q1"
    else:
        if   smile_bucket == "BULLISH_NEUTRAL": q = "Q3"
        elif smile_bucket == "BEARISH_FEAR":    q = "Q4"
        else:                                    q = "Q4"

    meta = _QUADRANT_META[q]
    return {
        "quadrant":     q,
        "name":         meta["name"],
        "short":        meta["short"],
        "color":        meta["color"],
        "badge_bg":     meta["badge_bg"],
        "description":  meta["description"],
        "action":       meta["action"],
        "smile_bucket": smile_bucket,
        "s34_band":     band,
    }


# ── Divergence metadata ────────────────────────────────────────────────────
_DIVERGENCE_META = {
    1: {
        "type":   "Type 1 — Capitulation Bottom",
        "color":  "#059669",
        "badge_bg": "#ECFDF5",
        "detail": (
            "S3/4 is at extreme bear levels (<=-51) AND the IV Smile is showing "
            "Crash Fear (Sc02, put wing >12pts + IV rank >=65). When both engines "
            "simultaneously hit maximum bearish readings, the risk/reward for new "
            "short positions deteriorates sharply. Hedgers are already in — they "
            "are the sellers of the rally that follows. Watch for put-wing deflation "
            "as the first sign of capitulation completing."
        ),
        "warning": "POTENTIAL CAPITULATION BOTTOM — do not initiate new shorts at these levels",
    },
    2: {
        "type":   "Type 2 — Structural Ceiling",
        "color":  "#D97706",
        "badge_bg": "#FFFBEB",
        "detail": (
            "S3/4 is at strong bull levels (>=+51) BUT the IV Smile is building "
            "a put-skew (Sc01/Sc03). This indicates institutions are hedging long "
            "positions even as the structural score peaks. Smart money is reducing "
            "risk at the top — a classic ceiling pattern. Reduce longs and avoid "
            "adding fresh long exposure at these combined readings."
        ),
        "warning": "STRUCTURAL CEILING SIGNAL — smart money hedging long positions; consider reducing longs",
    },
    3: {
        "type":   "Type 3 — Squeeze Warning",
        "color":  "#9333EA",
        "badge_bg": "#FAF5FF",
        "detail": (
            "S3/4 is neutral (-15 to +15) AND the IV Smile is showing maximum "
            "compression (Sc08 — Coiled Spring / IV rank <=20). The market is "
            "building energy for a move of unknown direction. Directional biases "
            "are unreliable here. Favour long-vol structures (straddles/strangles) "
            "but be mindful of theta decay — time the entry carefully."
        ),
        "warning": "SQUEEZE WARNING — breakout imminent; direction unknown; favour long vol",
    },
    4: {
        "type":   "Type 4 — Bear Trap",
        "color":  "#2563EB",
        "badge_bg": "#EFF6FF",
        "detail": (
            "S3/4 is negative (D or E band, score <=-16) BUT the IV Smile is "
            "showing call buying or relief (Sc04, Sc05, Sc06). Sentiment is "
            "recovering even as the structure looks bearish — the hallmark of a "
            "bear trap. Shorts caught in this divergence face a rapid squeeze. "
            "Cover or reduce short exposure when this signal activates."
        ),
        "warning": "BEAR TRAP RISK — call buying despite bearish OI structure; cover shorts",
    },
    5: {
        "type":   "Type 5 — Pre-Move Setup",
        "color":  "#B45309",
        "badge_bg": "#FEF3C7",
        "detail": (
            "PCR is at an extreme level AND S3/4 is moderately biased AND the "
            "IV Smile is directionally aligned. This combination suggests smart "
            "money positioning ahead of a move. Watch for confirmation via "
            "intraday OI velocity and gamma flip proximity before acting."
        ),
        "warning": "PRE-MOVE SETUP — smart money positioning; wait for OI velocity / GEX confirmation",
    },
}


def detect_divergence_type(
    s34_score: float,
    scenario_id: int,
    pcr: float = 1.0,
    div_proximity: float = 0.0,
) -> dict | None:
    """
    Returns a divergence dict or None if no divergence is active.

    v4 enhancement: when div_proximity >= 60 but the hard thresholds are not
    yet met, return a SOFT / APPROACHING divergence with reduced urgency so
    the dashboard can show early warning.
    """
    band         = get_s34_band(s34_score)
    smile_bucket = get_smile_bucket(scenario_id)

    # ── Hard divergence triggers (unchanged from v3) ──────────────────────
    # Type 1: Capitulation Bottom
    if band == "E" and scenario_id == 2:
        d = _DIVERGENCE_META[1].copy()
        d["divergence_id"] = 1
        d["strength"] = "HARD"
        return d

    # Type 2: Structural Ceiling
    if band == "A" and scenario_id in (1, 3):
        d = _DIVERGENCE_META[2].copy()
        d["divergence_id"] = 2
        d["strength"] = "HARD"
        return d

    # Type 4: Bear Trap
    if band in ("D", "E") and smile_bucket == "BULLISH_NEUTRAL":
        d = _DIVERGENCE_META[4].copy()
        d["divergence_id"] = 4
        d["strength"] = "HARD"
        return d

    # Type 3: Squeeze Warning
    if band == "C" and scenario_id == 8:
        d = _DIVERGENCE_META[3].copy()
        d["divergence_id"] = 3
        d["strength"] = "HARD"
        return d

    # Type 5: Pre-Move Setup
    if (pcr < 0.70 or pcr > 1.55) and abs(s34_score) >= 20 and scenario_id in (1, 2, 3, 4, 5, 6):
        d = _DIVERGENCE_META[5].copy()
        d["divergence_id"] = 5
        d["strength"] = "HARD"
        return d

    # ── v4 SOFT / APPROACHING divergence (proximity score >= 60) ─────────
    if div_proximity >= 60:
        # Determine which divergence type is closest to triggering
        soft_id = None
        if s34_score < -30 and scenario_id in (1, 2, 3, 9):
            soft_id = 1   # approaching Capitulation Bottom
        elif s34_score > 30 and scenario_id in (1, 3, 9):
            soft_id = 2   # approaching Structural Ceiling
        elif abs(s34_score) < 20 and scenario_id == 8:
            soft_id = 3   # approaching Squeeze
        elif s34_score < -10 and smile_bucket == "BULLISH_NEUTRAL":
            soft_id = 4   # approaching Bear Trap
        elif (pcr < 0.85 or pcr > 1.40) and abs(s34_score) >= 12 and scenario_id in (1, 2, 3, 4, 5, 6):
            soft_id = 5   # approaching Pre-Move

        if soft_id is not None:
            d = _DIVERGENCE_META[soft_id].copy()
            d["divergence_id"] = soft_id
            d["strength"] = "APPROACHING"
            d["proximity_score"] = div_proximity
            d["warning"] = (
                f"APPROACHING {d['type'].split(' — ')[0].upper()} "
                f"(proximity {div_proximity:.0f}/100) — monitor closely, not yet confirmed"
            )
            d["detail"] = (
                d["detail"]
                + f"\n\n[Early Warning] Proximity score is {div_proximity:.0f}/100, "
                f"indicating this divergence type is approaching trigger thresholds. "
                f"Watch for hard trigger confirmation in the next 1-3 ticks."
            )
            return d

    return None


def _confidence_label(s34_score: float, smile_confidence: float, quadrant: str,
                      enhanced_bias: dict | None = None) -> tuple:
    """
    Returns (label, color) for overall combined confidence.
    v4: when enhanced_bias is provided, its enhanced_conf can boost the label.
    """
    magnitude = abs(s34_score)

    if quadrant in ("Q1", "Q4"):
        if magnitude >= 51 and smile_confidence >= 75:
            label, color = "HIGH", "#059669"
        elif magnitude >= 20 and smile_confidence >= 60:
            label, color = "MODERATE-HIGH", "#16A34A"
        else:
            label, color = "MODERATE", "#D97706"
    elif quadrant in ("Q2", "Q3"):
        label, color = "MODERATE", "#D97706"
    else:
        label, color = "LOW", "#6B7280"

    # v4: Enhanced price layer can upgrade MODERATE → MODERATE-HIGH
    # when all 4 sub-signals agree (agreement_pct == 100) and enhanced_conf >= 65
    if enhanced_bias and label == "MODERATE":
        _econf = safe_num(enhanced_bias.get("enhanced_conf", 0))
        _agree = safe_num(enhanced_bias.get("agreement_pct", 0))
        if _econf >= 65 and _agree >= 100:
            label, color = "MODERATE-HIGH", "#16A34A"

    return label, color


def generate_combined_decision(
    s34: dict,
    smile: dict | None,
    m: dict,
    enhanced_bias: dict | None = None,
) -> dict:
    """
    Master combined decision function (v4 — merged inline + enhanced price layer).

    Parameters
    ----------
    s34           : dict from compute_section34_bias()
    smile         : dict from classify_iv_smile_scenario() or None
    m             : dict from compute_metrics()
    enhanced_bias : dict from compute_enhanced_price_bias() or None  [v4 NEW]

    Returns a verdict dict consumed by the dashboard panel.
    """
    # Fallbacks for smile
    fallback_smile_id   = 0
    fallback_smile_name = "Indeterminate"
    fallback_smile_conf = 0.0

    s34_score     = float(s34.get("bias_score", 0))
    s34_direction = s34.get("direction", "NEUTRAL")
    s34_breakdown = s34.get("signal_breakdown", {})

    pcr = float(m.get("pcr", 1.0)) if m else 1.0
    iv_rank = safe_num(m.get("iv_rank", 50)) if m else 50.0

    if smile is None:
        scenario_id   = fallback_smile_id
        scenario_name = fallback_smile_name
        smile_conf    = fallback_smile_conf
    else:
        scenario_id   = int(smile.get("scenario_id", 0))
        scenario_name = smile.get("scenario_name", "Unknown")
        smile_conf    = float(smile.get("confidence", 0))

    smile_bucket = get_smile_bucket(scenario_id)
    quad_info    = classify_quadrant(s34_score, scenario_id)

    # ── v4: Pass divergence proximity into divergence detector ───────────
    # Compute a quick inline proximity if not already available from caller
    _div_prox = 0.0
    if enhanced_bias:
        # Use the same proximity logic as the Leading Signals panel
        _scores = []
        _t1_s34 = max(0, min(100, (abs(s34_score) - 40) / 11 * 100)) if s34_score < -40 else 0
        _t1_smile = max(0, min(100, (iv_rank - 50) / 15 * 100)) if iv_rank > 50 else 0
        if scenario_id in (1, 2, 3): _t1_smile = max(_t1_smile, 60)
        _scores.append((_t1_s34 + _t1_smile) / 2)

        _t2_s34 = max(0, min(100, (s34_score - 40) / 11 * 100)) if s34_score > 40 else 0
        _t2_smile = 70 if scenario_id in (1, 3) else (30 if scenario_id in (2, 9) else 0)
        _scores.append((_t2_s34 + _t2_smile) / 2)

        _t3_neutral = max(0, min(100, (15 - abs(s34_score)) / 15 * 100)) if abs(s34_score) < 15 else 0
        _t3_compress = max(0, min(100, (25 - iv_rank) / 5 * 100)) if iv_rank < 25 else 0
        _scores.append((_t3_neutral + _t3_compress) / 2)

        _t5_pcr = 0
        if pcr < 0.70: _t5_pcr = max(0, min(100, (0.70 - pcr) / 0.20 * 100))
        elif pcr > 1.55: _t5_pcr = max(0, min(100, (pcr - 1.55) / 0.25 * 100))
        _t5_s34 = max(0, min(100, (abs(s34_score) - 15) / 15 * 100)) if abs(s34_score) > 15 else 0
        _scores.append((_t5_pcr + _t5_s34) / 2)

        _div_prox = round(max(_scores), 1) if _scores else 0.0

    divergence = detect_divergence_type(s34_score, scenario_id, pcr, div_proximity=_div_prox)
    conf_label, conf_color = _confidence_label(s34_score, smile_conf, quad_info["quadrant"],
                                                enhanced_bias=enhanced_bias)

    # ── v4: Enhanced price layer can shift quadrant in borderline cases ───
    # When enhanced_bias strongly disagrees with the quadrant (enhanced_score
    # opposite sign, |enhanced_score| >= 35, agreement_pct == 0), and the
    # original quadrant was a weak Q2/Q3, we can flip to CN.
    _quadrant_overridden = False
    if enhanced_bias:
        _e_score = safe_num(enhanced_bias.get("enhanced_score", 0))
        _e_agree = safe_num(enhanced_bias.get("agreement_pct", 0))
        _orig_q  = quad_info["quadrant"]
        # Q2 = structural bull / sentiment cautious → if enhanced says BEARISH, downgrade to CN
        if _orig_q == "Q2" and _e_score <= -35 and _e_agree == 0:
            quad_info = _QUADRANT_META["CN"].copy()
            quad_info["quadrant"] = "CN"
            quad_info["smile_bucket"] = smile_bucket
            quad_info["s34_band"] = get_s34_band(s34_score)
            quad_info["description"] += (
                " [v4 OVERRIDDEN to CN: Enhanced Price Layer (VWAP/TermStructure/VIX) "
                "strongly disagrees with the structural bull reading. "
                "Price action, term structure, and VIX all point bearish — "
                "the structural OI bull signal is likely a head-fake.]"
            )
            _quadrant_overridden = True
            conf_label, conf_color = "LOW", "#6B7280"
        # Q3 = structural bear / sentiment recovering → if enhanced says BULLISH, upgrade to CN
        elif _orig_q == "Q3" and _e_score >= 35 and _e_agree == 0:
            quad_info = _QUADRANT_META["CN"].copy()
            quad_info["quadrant"] = "CN"
            quad_info["smile_bucket"] = smile_bucket
            quad_info["s34_band"] = get_s34_band(s34_score)
            quad_info["description"] += (
                " [v4 OVERRIDDEN to CN: Enhanced Price Layer (VWAP/TermStructure/VIX) "
                "strongly disagrees with the structural bear reading. "
                "Price action, term structure, and VIX all point bullish — "
                "the structural OI bear signal is likely a bear trap.]"
            )
            _quadrant_overridden = True
            conf_label, conf_color = "LOW", "#6B7280"

    # ── Build explanation lines ───────────────────────────────────────────
    lines: list = []

    # Line 1: S3/4 summary
    band = quad_info.get("s34_band", get_s34_band(s34_score))
    band_desc = {
        "A": "strong bull (+51 to +100)",
        "B": "mild bull (+16 to +50)",
        "C": "neutral (-15 to +15)",
        "D": "mild bear (-50 to -16)",
        "E": "strong bear (-100 to -51)",
    }
    lines.append(
        f"S3/4 Engine: score {s34_score:+.0f} -> Band {band} ({band_desc.get(band, '')}) "
        f"- {s34_direction} structural bias."
    )

    # Line 2: IV Smile summary
    if smile is None:
        lines.append("IV Smile Engine: insufficient data - classification pending.")
    else:
        lines.append(
            f"IV Smile Engine: {scenario_name} (Sc{scenario_id:02d}) "
            f"-> {smile_bucket.replace('_', ' ')} bucket "
            f"[confidence {smile_conf:.0f}%]."
        )

    # Line 3: Quadrant result
    _quad_short = quad_info.get("short", quad_info.get("quadrant", "CN"))
    _quad_desc  = quad_info.get("description", "")
    lines.append(f"Combined Quadrant: {_quad_short}. {_quad_desc}")

    # Line 4: Signal bridge (S4 IV Skew)
    s4_val = s34_breakdown.get("S4 IV Skew", None)
    if s4_val is not None:
        s4_sign = "put-skew detected" if s4_val < 0 else ("call-skew / flat" if s4_val > 0 else "neutral")
        lines.append(
            f"S4 IV Skew (bridge signal): {s4_val:+.0f}/12 -> {s4_sign}. "
            "This shared signal links both engines - use it to validate alignment."
        )

    # Line 5: v4 Enhanced Price Layer contribution
    if enhanced_bias:
        _e_score = safe_num(enhanced_bias.get("enhanced_score", 0))
        _e_conf  = safe_num(enhanced_bias.get("enhanced_conf", 0))
        _n_avail = enhanced_bias.get("new_signals_available", 0)
        _price_s = safe_num(enhanced_bias.get("price_score", 0))
        _ts_s    = safe_num(enhanced_bias.get("ts_score", 0))
        _vix_s   = safe_num(enhanced_bias.get("vix_score", 0))
        lines.append(
            f"Enhanced Price Layer (v4): VWAP/OR {_price_s:+.1f}, "
            f"TermStructure {_ts_s:+.1f}, VIX {_vix_s:+.1f} "
            f"({_n_avail}/3 live) -> enhanced score {_e_score:+.1f} "
            f"[conf {_e_conf:.0f}%]."
        )
        if _quadrant_overridden:
            lines.append(
                "QUADRANT OVERRIDDEN by Enhanced Price Layer: Q2/Q3 -> CN. "
                "Price-action signals strongly disagree with OI structure."
            )

    # Line 6: Divergence note (if active)
    if divergence:
        _strength = divergence.get("strength", "HARD")
        if _strength == "APPROACHING":
            lines.append(
                f"EARLY WARNING - {divergence['type']}: {divergence['warning']}"
            )
        else:
            lines.append(
                f"DIVERGENCE ACTIVE: {divergence['type']} - {divergence['warning']}"
            )

    available = True
    if smile is None:
        available = False

    return {
        "quadrant":          quad_info.get("quadrant", "CN"),
        "quadrant_name":     quad_info.get("name", "Neutral / No Edge"),
        "quadrant_short":    quad_info.get("short", "CN — NEUTRAL"),
        "quadrant_color":    quad_info.get("color", "#6B7280"),
        "badge_bg":          quad_info.get("badge_bg", "#F9FAFB"),
        "action":            quad_info.get("action", ""),
        "explanation_lines": lines,
        "divergence":        divergence,
        "confidence_label":  conf_label,
        "confidence_color":  conf_color,
        "s34_score":         s34_score,
        "s34_direction":     s34_direction,
        "s34_band":          band,
        "smile_scenario":    scenario_name,
        "smile_bucket":      smile_bucket,
        "smile_confidence":  smile_conf,
        "pcr":               pcr,
        "available":         available,
        # v4 new keys
        "enhanced_score":    safe_num(enhanced_bias.get("enhanced_score", 0)) if enhanced_bias else 0.0,
        "enhanced_conf":     safe_num(enhanced_bias.get("enhanced_conf", 0))  if enhanced_bias else 0.0,
        "quadrant_overridden": _quadrant_overridden,
        "div_proximity":     _div_prox,
    }


# ══ END COMBINED BIAS DECISION ENGINE (inline) ═════════════════════════════


# ── Enhanced combined price bias aggregator ───────────────────────────────────
def compute_enhanced_price_bias(vwap_or, ts_signal, vix_signal, s34_score: float, spot: float):
    """
    Combine the three new signals with the existing S3/4 score into one
    Enhanced Bias Score (-100 to +100) and confidence rating.

    Base weights (total 100 pts):
      S3/4 options flow  : 70 pts  (existing engine, unchanged)
      VWAP + OR          : 10 pts  (new — price confirmation)
      Term Structure      : 10 pts  (new — IV slope confirmation)
      India VIX           : 10 pts  (new — volatility regime)

    DYNAMIC REDISTRIBUTION: when a new-signal module is unavailable its 10 pts
    are redistributed to S3/4 so the enhanced score never deflates below the
    raw S3/4 score when the new layers have no data.

    The final score is the weighted sum scaled to [-100, +100].
    Confidence is boosted when all four signals agree.
    """
    price_s = safe_num(vwap_or["price_score"])  if vwap_or  else 0.0
    ts_s    = safe_num(ts_signal["ts_score"])   if ts_signal else 0.0
    vix_s   = safe_num(vix_signal["vix_score"]) if vix_signal else 0.0

    # Determine which new modules are live
    price_live = vwap_or  is not None and vwap_or.get("n_candles", 0) > 5
    ts_live    = ts_signal  is not None and ts_signal.get("available", False)
    vix_live   = vix_signal is not None and vix_signal.get("available", False)

    # Dynamic weight allocation: unavailable modules cede their 10 pts to S3/4
    w_price = 10 if price_live else 0
    w_ts    = 10 if ts_live    else 0
    w_vix   = 10 if vix_live   else 0
    w_s34   = 100 - w_price - w_ts - w_vix   # absorbs freed weight; min 70, max 100

    # Normalise each signal to its weight bucket
    s34_contrib   = (s34_score / 100.0) * w_s34
    price_contrib = (price_s   / 10.0)  * w_price
    ts_contrib    = (ts_s      / 8.0)   * w_ts   # ts_score max ±8
    vix_contrib   = (vix_s     / 8.0)   * w_vix  # vix_score max ±8

    raw_score = s34_contrib + price_contrib + ts_contrib + vix_contrib
    enhanced_score = round(max(-100.0, min(100.0, raw_score)), 1)

    # Signal agreement for confidence
    # H14 fix: agreement_pct was previously measured against the SIGN of
    # `enhanced_score` itself — which is dominated by s34 (weight w_s34=50).
    # The score and its confidence were circularly derived from the same inputs:
    # a strongly-weighted s34 component would dominate the score sign, then
    # agreement_pct would mechanically read ~1.0 (since s34's sign matches the
    # score sign). Confidence became a function of s34 weight, not of genuine
    # cross-signal corroboration.
    # Fix: measure agreement against s34's sign (the dominant input), not the
    # output score. Non-s34 signals (price/ts/vix) that agree with s34 boost
    # confidence; non-s34 signals that disagree pull it down. s34 itself is
    # excluded from the agreement count so the metric reflects corroboration
    # FROM the other signals, not self-agreement.
    signs_non_s34 = []
    if price_s   != 0:   signs_non_s34.append(1 if price_s   > 0 else -1)
    if ts_s      != 0:   signs_non_s34.append(1 if ts_s      > 0 else -1)
    if vix_s     != 0:   signs_non_s34.append(1 if vix_s     > 0 else -1)
    s34_sign = 1 if s34_score > 0 else (-1 if s34_score < 0 else 0)
    if s34_sign != 0 and signs_non_s34:
        agree_count = sum(1 for s in signs_non_s34 if s == s34_sign)
        agreement_pct = agree_count / len(signs_non_s34)
    else:
        # s34 is zero or no other signals available — fall back to old formula
        signs = []
        if s34_score != 0:   signs.append(1 if s34_score > 0 else -1)
        if price_s   != 0:   signs.append(1 if price_s   > 0 else -1)
        if ts_s      != 0:   signs.append(1 if ts_s      > 0 else -1)
        if vix_s     != 0:   signs.append(1 if vix_s     > 0 else -1)
        agree_count = sum(1 for s in signs if s == (1 if enhanced_score >= 0 else -1))
        agreement_pct = agree_count / max(len(signs), 1)

    base_confidence = min(abs(enhanced_score), 100)
    conf_bonus      = agreement_pct * 20.0   # up to +20 when all agree
    enhanced_conf   = round(min(100.0, base_confidence * (0.4 + 0.6 * agreement_pct)), 1)

    if   enhanced_score >= 30:  direction = "BULLISH";    color = "#059669"
    elif enhanced_score >= 10:  direction = "MILDLY BULLISH"; color = "#10B981"
    elif enhanced_score <= -30: direction = "BEARISH";    color = "#DC2626"
    elif enhanced_score <= -10: direction = "MILDLY BEARISH"; color = "#F59E0B"
    else:                       direction = "NEUTRAL";    color = "#6B7280"

    # Determine how many new signals are live (reuse booleans from weight calc)
    new_signals_available = sum([price_live, ts_live, vix_live])

    return {
        "enhanced_score":          enhanced_score,
        "direction":               direction,
        "color":                   color,
        "enhanced_conf":           enhanced_conf,
        "agreement_pct":           round(agreement_pct * 100, 0),
        "s34_score":               s34_score,
        "price_score":             price_s,
        "ts_score":                ts_s,
        "vix_score":               vix_s,
        "new_signals_available":   new_signals_available,
    }

# ══ END ENHANCED PRICE CONFIRMATION LAYER ════════════════════════════════════


# ─── History helpers ──────────────────────────────────────────────────────────
def build_history_entry(m, spot, call_oi_total, put_oi_total, expiry, synth_excess=None, basis_gap=None, traded_basis=None):
    return {
        "ts":              now_ist().strftime("%Y-%m-%dT%H:%M:%S"),
        "spot":            spot,
        "atm_iv":          m.get("atm_iv", 0),
        # Zero-vega guard: store None (not 0.0) when both sides are zero so the
        # Vega Diff chart filter (`if _cv is not None and _pv is not None`) drops
        # the tick cleanly rather than plotting a meaningless zero flatline.
        "atm_call_vega":   m.get("atm_call_vega") if safe_num(m.get("atm_call_vega", 0)) > 0 else None,
        "atm_put_vega":    m.get("atm_put_vega")  if safe_num(m.get("atm_put_vega",  0)) > 0 else None,
        "net_delta":    m.get("net_delta", 0),
        "oi_net_delta": m.get("momentum", 0),
        "momentum":     m.get("momentum", 0),
        "max_pain":     m.get("max_pain", 0),
        "support":      m.get("support", 0),
        "resistance":   m.get("resistance", 0),
        "gex":          m.get("gex", 0),
        "pcr":          m.get("pcr", 0),
        "atm_pressure": m.get("atm_pressure", 0),
        "wall_width":   m.get("wall_width", 0),
        "gamma_flip":   m.get("gamma_flip", None),
        "iv_rank":      m.get("iv_rank", 50),
        "gt_ratio":     m.get("gt_ratio", 0),
        "atm":          m.get("atm", 0),
        "call_oi_total":call_oi_total,
        "put_oi_total": put_oi_total,
        "synth_excess": synth_excess,
        "basis_gap":    basis_gap,
        "traded_basis": traded_basis,
        "call_dw_flow": float(sum(r.get("call_oi_chg", 0) * abs(r.get("call_delta", 0.3))
                               for r in m.get("_df_band_records", []))),
        "put_dw_flow":  float(sum(r.get("put_oi_chg", 0) * abs(r.get("put_delta", 0.3))
                               for r in m.get("_df_band_records", []))),
    }


# ─── Fetch + compute ──────────────────────────────────────────────────────────
# Server-side cache: shared across ALL visitor sessions in the same process.
# Visitors only read from this cache; only the timed expiry triggers a new API call.
#
# CI #7 fix: separate "fetch in progress" flag from the cache itself so concurrent
# visitors don't serialize on the network call. Pattern:
#   - Acquire lock briefly. If cache fresh → return cached payload.
#   - If cache stale AND no fetch in progress → set fetch_in_progress=True,
#     release lock, do the fetch, re-acquire lock, store payload, clear flag.
#   - If cache stale AND fetch already in progress → return stale payload
#     (stale-while-revalidate).
_srv_cache_lock      = threading.Lock()
_srv_cache           = {"payload": None, "source": None, "last_fetch_ts": 0.0}
_srv_fetch_in_progress = False     # CI #7 fix: single-flight flag

def _raw_fetch_and_compute(expiry_override=None, history=None):
    """Actual Dhan API fetch — never called directly by visitors."""
    df, spot, expiry, source = get_option_chain(expiry_override)
    if df.empty:
        return None, source

    m = compute_metrics(df, spot, expiry, history=history)
    if not m:
        return None, source

    df_band  = m.pop("df_band", df)
    df_sig   = m.pop("df_signal", df)

    traded_fut   = fetch_futures_ltp(expiry)
    _atm         = safe_num(m.get("atm", 0))
    _df_band_lst = df_band.fillna(0).to_dict("records")
    m["_df_band_records"] = _df_band_lst

    sf           = compute_synthetic_future(_df_band_lst, spot, _atm, expiry)
    synth_excess = round(sf["synth_excess"], 2) if sf else None
    traded_basis = round(safe_num(traded_fut) - spot, 2) if traded_fut and safe_num(traded_fut) > 0 else None
    basis_gap    = round(safe_num(traded_fut) - sf["synthetic"], 2) if sf and traded_fut and safe_num(traded_fut) > 0 else None

    payload = {
        "symbol":  SYMBOL,
        "source":  source,
        "spot":    spot,
        "expiry":  expiry,
        "metrics": m,
        "df_band": _df_band_lst,
        "df_signal": df_sig.fillna(0).to_dict("records"),
        "traded_future": traded_fut if traded_fut and safe_num(traded_fut) > 0 else None,
        "synth_excess":  synth_excess,
        "basis_gap":     basis_gap,
        "traded_basis":  traded_basis,
        "ts_ist":        ist_str(),
        "is_live":       is_market_hours() and USE_DHAN,
    }
    return payload, source


def get_server_data(expiry_override=None):
    """
    Returns (payload, source, last_fetch_ts) for ALL visitors.
    Only calls Dhan API when the owner-configured refresh interval has elapsed.
    Visitor page-refreshes (every 60 s) NEVER trigger a new API call.
    last_fetch_ts is the Unix timestamp of the most recent Dhan fetch — used by
    callers to dedup history entries so one data snapshot → exactly one entry.

    CI #7 fix: single-flight refresh — the lock is held only for the cache
    check, NOT during the network call. Concurrent visitors return the stale
    payload while one thread refreshes (stale-while-revalidate).
    """
    global _srv_cache, _srv_fetch_in_progress
    settings = _load_owner_settings()
    interval = settings.get("refresh_interval", REFRESH_SECONDS)
    expiry_to_use = expiry_override or settings.get("selected_expiry")

    # ── Phase 1: brief lock to check cache freshness ──
    with _srv_cache_lock:
        now = time.time()
        cache_expired = (now - _srv_cache["last_fetch_ts"]) >= interval
        if not cache_expired and _srv_cache["payload"] is not None:
            # Cache fresh — return immediately
            return (
                _srv_cache.get("payload"),
                _srv_cache.get("source", "N/A"),
                _srv_cache.get("last_fetch_ts", 0.0),
            )
        # Cache stale — check if another thread is already fetching
        if _srv_fetch_in_progress:
            # Stale-while-revalidate: return what we have, let the other thread finish
            return (
                _srv_cache.get("payload"),
                _srv_cache.get("source", "N/A"),
                _srv_cache.get("last_fetch_ts", 0.0),
            )
        # Claim the fetch slot
        _srv_fetch_in_progress = True

    # ── Phase 2: do the network call WITHOUT holding the lock ──
    try:
        _hist_for_ivr = _load_history()
        payload, source = _raw_fetch_and_compute(expiry_to_use, history=_hist_for_ivr)
    except Exception:
        payload, source = None, "API ERROR"

    # ── Phase 3: re-acquire lock to update cache + clear flag ──
    with _srv_cache_lock:
        _srv_fetch_in_progress = False
        if payload is not None:
            _srv_cache["payload"] = payload
            _srv_cache["source"]  = source
            _srv_cache["last_fetch_ts"] = time.time()
        # If payload is None (fetch failed), keep the previous stale payload
        # but bump last_fetch_ts by a short cooldown to prevent retry storms.
        # CI #4 / H7 fix: was 0.0 → every subsequent visitor retried the failed
        # fetch with no backoff.
        elif _srv_cache["payload"] is not None:
            _srv_cache["last_fetch_ts"] = time.time() - max(0, interval - 30)  # 30s cooldown

    return (
        _srv_cache.get("payload"),
        _srv_cache.get("source", "N/A"),
        _srv_cache.get("last_fetch_ts", 0.0),
    )


def _force_server_refresh(expiry_override=None):
    """Owner-only: clear ALL caches so the next get_server_data() fetches fresh data.

    CI #4 fix: previously only zeroed `_srv_cache["last_fetch_ts"]`. That left
    three Streamlit-cached fetchers (intraday candles, back-expiry OI band,
    back-expiry ATM IV) and two module-level LTP caches serving stale data for
    their full TTLs. The owner clicked "Refresh Now" and saw a partially stale
    dashboard. Now we clear everything explicitly.
    """
    global _srv_cache
    with _srv_cache_lock:
        _srv_cache["last_fetch_ts"] = 0.0   # expire the cache immediately

    # CI #4 fix: clear Streamlit's @st.cache_data layer
    try:
        st.cache_data.clear()
    except Exception:
        pass

    # CI #4 fix: clear module-level LTP caches (futures + VIX)
    # These are read by `fetch_futures_ltp` / `fetch_india_vix_ltp` and were
    # previously left serving stale values for ~60s after a manual refresh.
    try:
        _fut_ltp_cache["ts"] = 0.0
    except Exception:
        pass
    try:
        _vix_ltp_cache_v7["ts"] = 0.0
    except Exception:
        pass


# ─── OI Velocity bucket helpers ───────────────────────────────────────────────
def _parse_ts_to_bucket(ts_str):
    try:
        t_part = ts_str.split("T")[-1] if "T" in ts_str else ts_str
        parts  = t_part.split(":")
        hh, mm = int(parts[0]), int(parts[1])
        return f"{hh:02d}:{(mm // 15) * 15:02d}"
    except Exception:
        return None


def compute_dw_flow_buckets(sym_history):
    """
    v23-4 — Decay-weighted OI flow + rolling-window normaliser.

    Improvement 1 — Decay-weighted running sum (replaces flat np.cumsum):
      running[i] = DW_FLOW_DECAY × running[i-1] + bucket_net[i]
      Older session data fades; the line stays reactive to the current regime.

    Improvement 2 — Rolling-window normalised net flow:
      net_flow_norm[i] = net_flow[i] / max(|net_flow[i-W+1..i]|)
      Bounded [−1, +1]. Window = NORM_WINDOW_BUCKETS (default 12 × 15min ≈ 3 hr).
    """
    if len(sym_history) < 2:
        return {}

    _DECAY = DW_FLOW_DECAY

    bucket_data = {}
    for tick in sym_history:
        bkt = _parse_ts_to_bucket(tick.get("ts", ""))
        if bkt is None:
            continue
        if bkt not in bucket_data:
            bucket_data[bkt] = {
                "call_dw": [], "put_dw": [], "spot": [],
                "gex": [], "gamma_flip": [], "max_pain": [],
                "support": [], "resistance": [],
            }
        bucket_data[bkt]["call_dw"].append(safe_num(tick.get("call_dw_flow", 0)))
        bucket_data[bkt]["put_dw"].append(safe_num(tick.get("put_dw_flow", 0)))
        bucket_data[bkt]["spot"].append(safe_num(tick.get("spot", 0)))
        bucket_data[bkt]["gex"].append(safe_num(tick.get("gex", 0)))
        gf = tick.get("gamma_flip")
        if gf is not None:
            bucket_data[bkt]["gamma_flip"].append(safe_num(gf))
        bucket_data[bkt]["max_pain"].append(safe_num(tick.get("max_pain", 0)))
        bucket_data[bkt]["support"].append(safe_num(tick.get("support", 0)))
        bucket_data[bkt]["resistance"].append(safe_num(tick.get("resistance", 0)))

    labels = sorted(bucket_data.keys())

    call_flow_raw = [float(np.sum(bucket_data[b]["call_dw"])) for b in labels]
    put_flow_raw  = [float(np.sum(bucket_data[b]["put_dw"]))  for b in labels]
    net_flow_raw  = [p - c for c, p in zip(call_flow_raw, put_flow_raw)]

    # Improvement 1: Decay-weighted running totals
    def _decay_cumsum(raw_series, decay):
        result = []
        running = 0.0
        for v in raw_series:
            running = decay * running + v
            result.append(running)
        return result

    call_flow = _decay_cumsum(call_flow_raw, _DECAY)
    put_flow  = _decay_cumsum(put_flow_raw,  _DECAY)
    net_flow  = _decay_cumsum(net_flow_raw,  _DECAY)

    # Improvement 2: Rolling-window normaliser
    _net_flow_norm_list = []
    for _ni, _nv in enumerate(net_flow):
        if NORM_WINDOW_BUCKETS is None:
            _win = net_flow[:_ni + 1]
        else:
            _win = net_flow[max(0, _ni - NORM_WINDOW_BUCKETS + 1): _ni + 1]
        _denom = max((abs(v) for v in _win), default=1.0)
        _denom = max(_denom, 1.0)
        _net_flow_norm_list.append(float(np.clip(_nv / _denom, -1.0, 1.0)))
    net_flow_norm = _net_flow_norm_list

    _session_max = max(
        (abs(v) for v in net_flow[max(0, len(net_flow) - (NORM_WINDOW_BUCKETS or len(net_flow))):]),
        default=1.0,
    )
    _session_max = max(_session_max, 1.0)

    spot_close  = [float(np.mean(bucket_data[b]["spot"]))   for b in labels]
    gex_mean    = [float(np.mean(bucket_data[b]["gex"]))    for b in labels]

    def _last_valid(lst): return lst[-1] if lst else None
    gamma_flip_v  = [_last_valid(bucket_data[b]["gamma_flip"])  for b in labels]
    max_pain_v    = [float(np.mean(bucket_data[b]["max_pain"])) for b in labels]
    support_v     = [float(np.mean(bucket_data[b]["support"]))  for b in labels]
    resistance_v  = [float(np.mean(bucket_data[b]["resistance"])) for b in labels]

    all_dw = [abs(v) for v in call_flow_raw + put_flow_raw]
    delta_active = any(v > 1000 for v in all_dw)

    return {
        "labels"        : labels,
        "call_flow"     : call_flow,
        "put_flow"      : put_flow,
        "net_flow"      : net_flow,
        "net_flow_norm" : net_flow_norm,
        "net_flow_raw"  : net_flow_raw,
        "spot"          : spot_close,
        "gex"           : gex_mean,
        "gamma_flip"    : gamma_flip_v,
        "max_pain"      : max_pain_v,
        "support"       : support_v,
        "resistance"    : resistance_v,
        "delta_active"  : delta_active,
        "decay"         : _DECAY,
        "session_max"   : _session_max,
    }


def compute_raw_oi_buckets(sym_history):
    """
    v23-4 — Raw (non-delta-weighted) OI change into 15-min buckets.
    PCR-of-Flow = put_oi_added[bucket] / call_oi_added[bucket] − 1 per bucket.
    Volume guard: total < PCR_MIN_BUCKET_OI → signal clamped to 0.
    """
    if len(sym_history) < 2:
        return {}

    _MIN_BUCKET_OI = PCR_MIN_BUCKET_OI

    bucket_data = {}
    for i, tick in enumerate(sym_history):
        bkt = _parse_ts_to_bucket(tick.get("ts", ""))
        if bkt is None:
            continue
        if bkt not in bucket_data:
            bucket_data[bkt] = {
                "call_chg": [], "put_chg": [],
                "spot": [], "gex": [],
                "gamma_flip": [], "max_pain": [],
                "support": [], "resistance": [],
            }
        if i == 0:
            c_chg = p_chg = 0.0
        else:
            prev  = sym_history[i - 1]
            c_chg = safe_num(tick.get("call_oi_total", 0)) - safe_num(prev.get("call_oi_total", 0))
            p_chg = safe_num(tick.get("put_oi_total",  0)) - safe_num(prev.get("put_oi_total",  0))
        bucket_data[bkt]["call_chg"].append(c_chg)
        bucket_data[bkt]["put_chg"].append(p_chg)
        bucket_data[bkt]["spot"].append(safe_num(tick.get("spot", 0)))
        bucket_data[bkt]["gex"].append(safe_num(tick.get("gex", 0)))
        gf = tick.get("gamma_flip")
        if gf is not None:
            bucket_data[bkt]["gamma_flip"].append(safe_num(gf))
        bucket_data[bkt]["max_pain"].append(safe_num(tick.get("max_pain", 0)))
        bucket_data[bkt]["support"].append(safe_num(tick.get("support", 0)))
        bucket_data[bkt]["resistance"].append(safe_num(tick.get("resistance", 0)))

    labels = sorted(bucket_data.keys())

    call_chg_raw = [float(np.sum(bucket_data[b]["call_chg"])) for b in labels]
    put_chg_raw  = [float(np.sum(bucket_data[b]["put_chg"]))  for b in labels]

    call_chg = list(float(v) for v in np.cumsum(call_chg_raw))
    put_chg  = list(float(v) for v in np.cumsum(put_chg_raw))
    net_raw  = [p - c for c, p in zip(call_chg, put_chg)]

    # PCR-of-Flow per bucket
    pcr_flow_signal = []
    pcr_flow_raw    = []
    for c, p in zip(call_chg_raw, put_chg_raw):
        total = abs(c) + abs(p)
        if total < _MIN_BUCKET_OI:
            pcr_flow_signal.append(0.0)
            pcr_flow_raw.append(1.0)
        else:
            c_abs = max(abs(c), 1.0)
            p_abs = max(abs(p), 1.0)
            ratio = p_abs / c_abs
            pcr_flow_signal.append(float(np.clip(ratio - 1.0, -5.0, 5.0)))
            pcr_flow_raw.append(float(ratio))

    spot_close = [float(np.mean(bucket_data[b]["spot"])) for b in labels]

    def _last_valid(lst): return lst[-1] if lst else None
    gamma_flip_v = [_last_valid(bucket_data[b]["gamma_flip"])    for b in labels]
    max_pain_v   = [float(np.mean(bucket_data[b]["max_pain"]))   for b in labels]
    support_v    = [float(np.mean(bucket_data[b]["support"]))    for b in labels]
    resistance_v = [float(np.mean(bucket_data[b]["resistance"])) for b in labels]

    return {
        "labels"          : labels,
        "call_chg"        : call_chg,
        "put_chg"         : put_chg,
        "net_raw"         : net_raw,
        "pcr_flow_signal" : pcr_flow_signal,
        "pcr_flow_raw"    : pcr_flow_raw,
        "spot"            : spot_close,
        "gamma_flip"      : gamma_flip_v,
        "max_pain"        : max_pain_v,
        "support"         : support_v,
        "resistance"      : resistance_v,
        "min_bucket_oi"   : _MIN_BUCKET_OI,
    }


def compute_gamma_blast_monitor(bkt: dict, m: dict, alert: dict, spot_px: float = 0.0) -> dict:
    """
    Compute a gamma blast risk score (0-100) and stage from existing Section 9 signals.
    Identical logic to the Dash app version — grounded in already-computed values only.

    Stage 1 STRUCTURAL SETUP  : score  1–24  (preconditions present)
    Stage 2 PRESSURE BUILDING : score 25–54  (G/T rising, persistent flow)
    Stage 3 IMMINENT BLAST    : score 55–79  (Section 8 fires active)
    Stage 4 BLAST IN MOTION   : score 80–100 (all conditions met)
    CLEAR                     : score 0
    """
    score   = 0
    signals = []   # list of (label, value_str, pts, hex_color)

    gex      = safe_num(m.get("gex", 0))
    gflip    = safe_num(m.get("gamma_flip", 0))
    wall_w   = safe_num(m.get("wall_width", 400))
    gt_ratio = safe_num(m.get("gt_ratio", 0))
    # Fix #1: m dict from compute_metrics() never has a "spot" key.
    # Use the explicitly passed spot_px (module-level `spot` from payload).
    # Fall back to ATM strike (≈ spot within ±25 pts) so the score is always live.
    spot     = safe_num(spot_px) if spot_px else safe_num(m.get("atm", 0))

    # ── Stage 1 (max 30 pts) ──────────────────────────────────────────────────
    if gex < 0:
        score += 15
        signals.append(("GEX Sign", f"{gex:,.0f} (negative)", 15, "#DC2626"))
    elif 0 <= gex < 1000:
        score += 10
        signals.append(("GEX Sign", f"{gex:,.0f} (near-zero)", 10, "#D97706"))
    else:
        signals.append(("GEX Sign", f"{gex:,.0f} (positive)", 0, "#059669"))

    if gflip > 0 and spot > 0:
        _thresh = max(2.0 * wall_w / 20, 100)
        flip_dist = abs(spot - gflip)
        if spot < gflip:
            score += 15
            signals.append(("Flip Position", f"Spot {flip_dist:.0f}pts BELOW flip", 15, "#DC2626"))
        elif flip_dist <= _thresh:
            score += 10
            signals.append(("Flip Position", f"Spot {flip_dist:.0f}pts from flip (≤{_thresh:.0f})", 10, "#D97706"))
        else:
            signals.append(("Flip Position", f"Spot {flip_dist:.0f}pts above flip", 0, "#059669"))
    else:
        signals.append(("Flip Position", "N/A", 0, "#6B7280"))

    # ── Stage 2 (max 30 pts) ──────────────────────────────────────────────────
    if gt_ratio >= 0.08 and gex < 0:
        score += 15
        signals.append(("G/T Ratio", f"{gt_ratio:.4f} — HIGH G/T fire", 15, "#DC2626"))
    elif gt_ratio >= 0.05:
        score += 8
        signals.append(("G/T Ratio", f"{gt_ratio:.4f} — watch zone", 8, "#D97706"))
    else:
        signals.append(("G/T Ratio", f"{gt_ratio:.4f} — normal", 0, "#059669"))

    net_flow = bkt.get("net_flow", [])
    if len(net_flow) >= 3:
        last3     = net_flow[-3:]
        neg_count = sum(1 for v in last3 if v < 0)
        pos_count = sum(1 for v in last3 if v > 0)
        if neg_count >= 2:
            score += 10
            signals.append(("Flow Direction", f"{neg_count}/3 buckets bearish", 10, "#DC2626"))
        elif pos_count >= 2:
            signals.append(("Flow Direction", f"{pos_count}/3 buckets bullish", 0, "#059669"))
        else:
            signals.append(("Flow Direction", "Mixed", 0, "#6B7280"))
        if len(net_flow) >= 2:
            accel = net_flow[-1] - net_flow[-2]
            if accel < 0 and neg_count >= 2:
                score += 5
                signals.append(("Flow Accel", "Bearish & accelerating ↓", 5, "#DC2626"))
            elif accel > 0 and pos_count >= 2:
                signals.append(("Flow Accel", "Bullish & accelerating ↑", 0, "#059669"))
            else:
                signals.append(("Flow Accel", f"Δ {accel:+.1f} — decelerating/mixed", 0, "#6B7280"))
    else:
        signals.append(("Flow Direction", "< 3 buckets — building", 0, "#6B7280"))

    # ── Stage 3 (from Pre-Move Alert fires, max 45 pts) ───────────────────────
    fires = alert.get("fires", [])
    pre_move_score = alert.get("pre_move_score", 0)
    _fire_pts = {"IV_SPIKE": 8, "OI_VELOCITY": 12, "GEX_FLIP_RISK": 15, "HIGH_GT": 10}
    for fk, pts in _fire_pts.items():
        fired = any(fk.replace("_", " ").lower() in str(f).lower() or
                    fk.lower() in str(f).lower() for f in fires)
        if fired:
            score += pts
            signals.append((f"🔥 {fk.replace('_',' ')}", "S8 fire", pts, "#DC2626"))

    score = min(100, score)

    if score == 0:
        stage, s_col, s_bg = "CLEAR",                "#059669", "#ECFDF5"
    elif score < 25:
        stage, s_col, s_bg = "STAGE 1 — STRUCTURAL", "#0F766E", "#F0FDFA"
    elif score < 55:
        stage, s_col, s_bg = "STAGE 2 — PRESSURE",   "#B45309", "#FFFBEB"
    elif score < 80:
        stage, s_col, s_bg = "STAGE 3 — IMMINENT",   "#DC2626", "#FEF2F2"
    else:
        stage, s_col, s_bg = "STAGE 4 — BLAST",      "#7C3AED", "#F5F3FF"

    if score == 0:
        note = "No blast preconditions. GEX positive, spot above flip — dealers absorbing."
    elif score < 25:
        note = "Structural preconditions present. Monitor G/T ratio and flow direction."
    elif score < 55:
        note = "Pressure building. G/T rising and/or persistent bearish flow. Watch Section 8 fires."
    elif score < 80:
        note = "⚠️ Imminent blast. Section 8 fires active. Spot near or through flip. Act or protect now."
    else:
        note = "🚨 Blast in motion. All conditions met. IV spike likely imminent."

    return {
        "score": score, "stage": stage, "s_col": s_col, "s_bg": s_bg,
        "signals": signals, "note": note, "pre_move_score": pre_move_score,
    }


def compute_dw_composite_bias(bkt, expiry_str=None):
    """
    Composite bias score −100 to +100 from:
      35% — Net delta flow direction + momentum
      25% — Delta flow acceleration (last bucket vs prev)
      20% — GEX sign & magnitude
      20% — Gamma flip side (spot above/below flip)
      Max pain gravity: only weighted on expiry week (Thurs/Fri)
    """
    if not bkt or len(bkt.get("labels", [])) < 2:
        return {"score": 0, "direction": "NEUTRAL", "confidence": 0,
                "components": {}, "narrative": "Insufficient data — need 2+ buckets."}

    labels   = bkt["labels"]
    net_flow = bkt["net_flow"]
    gex_arr  = bkt["gex"]
    gf_arr   = bkt["gamma_flip"]
    spot_arr = bkt["spot"]
    # CHANGE 4 (audit fix): use the RAW (un-decayed) net flow for normalization.
    # `compute_dw_flow_buckets` exposes `net_flow_raw` alongside the decay-weighted
    # `net_flow`. With DW_FLOW_DECAY=0.85, the decayed series grows monotonically
    # under sustained flow (steady state ≈ X / (1-0.85) = 6.67X). Normalizing
    # decayed flow_3 against decayed session_max therefore saturates near ±1.0
    # after a few consistent buckets — destroying magnitude differentiation
    # between "mild sustained" and "strong sustained" sessions.
    # Using raw on both sides preserves the magnitude signal while keeping the
    # decayed series available for the display label.
    net_flow_raw = bkt.get("net_flow_raw", net_flow)  # fallback to decayed if missing

    recent_flow     = net_flow[-1]                                # decayed (for label)
    flow_3          = float(np.mean(net_flow[-3:])) if len(net_flow) >= 3 else recent_flow   # decayed (for label)
    recent_flow_raw = net_flow_raw[-1]
    flow_3_raw      = float(np.mean(net_flow_raw[-3:])) if len(net_flow_raw) >= 3 else recent_flow_raw
    session_range_raw = max(abs(f) for f in net_flow_raw) if any(f != 0 for f in net_flow_raw) else 1.0
    session_range   = session_range_raw                           # for label consistency

    flow_norm = max(-1.0, min(1.0, flow_3_raw / max(session_range_raw, 1.0)))
    c1_score  = round(flow_norm * 35, 1)
    c1_label  = (f"Net Δ-flow (PUT−CALL): {flow_3:+,.0f}  "
                 f"({'PUT dominant — bullish' if flow_3 > 0 else 'CALL dominant — bearish'})  "
                 f"[session max (raw): {session_range:,.0f}]")

    if len(net_flow_raw) >= 2:
        # Acceleration: use RAW flow change so the signal reflects genuine
        # per-bucket delta, not the smoothed decayed difference.
        accel      = net_flow_raw[-1] - net_flow_raw[-2]
        accel_norm = max(-1.0, min(1.0, accel / max(session_range_raw, 1.0)))
        c2_score   = round(accel_norm * 25, 1)
        c2_label   = (f"Flow accel (PUT−CALL): {accel:+,.0f}  "
                      f"({'PUT accelerating ↑' if accel_norm > 0.1 else 'CALL accelerating ↓' if accel_norm < -0.1 else 'steady →'})")
    else:
        c2_score, c2_label = 0, "Acceleration: need 2+ buckets"

    gex_now = gex_arr[-1] if gex_arr else 0
    if gex_now > 0:
        # FIX (Issue 1): was +10 bull / -20 bear — undocumented structural tilt.
        # Now symmetric ±10: both regimes carry equal directional weight.
        # Short-gamma danger is captured by the Fake Breakout engine (Section 8).
        c3_score = +10
        c3_label = f"GEX +{gex_now:,.0f} (long-gamma regime — pinning tendency)"
    elif gex_now < 0:
        c3_score = -10
        c3_label = f"GEX {gex_now:,.0f} (short-gamma — trending/amplifying regime)"
    else:
        c3_score, c3_label = 0, "GEX near zero — unstable transition"

    spot_now = spot_arr[-1] if spot_arr else 0
    gf_now   = next((g for g in reversed(gf_arr) if g is not None), None)
    if gf_now and spot_now > 0:
        dist      = spot_now - gf_now
        dist_norm = max(-1.0, min(1.0, dist / max(abs(dist) + 1e-9, 500)))
        c4_score  = round(dist_norm * 20, 1)
        side_lbl  = "above flip (stable)" if dist > 0 else "BELOW flip (short-gamma danger)"
        c4_label  = f"Spot {dist:+.0f}pts from flip @ {gf_now:,.0f} — {side_lbl}"
    else:
        c4_score, c4_label = 0, "Gamma flip not available"

    c5_score, c5_label = 0, "Max pain: not expiry week (no weight)"
    if expiry_str:
        try:
            exp_date    = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            days_to_exp = (exp_date - date.today()).days
            if days_to_exp <= 2:
                mp_now = bkt["max_pain"][-1] if bkt["max_pain"] else 0
                if mp_now > 0 and spot_now > 0:
                    pull      = mp_now - spot_now
                    pull_norm = max(-1.0, min(1.0, pull / 500))
                    c5_score  = round(pull_norm * 10, 1)
                    c5_label  = f"Max pain gravity: {pull:+.0f}pts toward {mp_now:,.0f} ({days_to_exp}d to exp)"
        except Exception:
            pass

    total_score = max(-100, min(100, c1_score + c2_score + c3_score + c4_score + c5_score))

    if total_score >= 45:
        direction = "BULLISH"
    elif total_score >= 15:
        direction = "MILD BULLISH"
    elif total_score <= -45:
        direction = "BEARISH"
    elif total_score <= -15:
        direction = "MILD BEARISH"
    else:
        direction = "NEUTRAL"

    # FIX (Issue 2): confidence was purely abs(total_score)*multiplier — a monotonic
    # transform of score magnitude that ignores whether components actually agree.
    # Now: count how many scored components agree in sign with the direction,
    # blend 60% agreement ratio + 40% score magnitude for a richer signal.
    _all_comp_scores = [c1_score, c2_score, c3_score, c4_score]
    if c5_score != 0:
        _all_comp_scores.append(c5_score)
    _total_sign = 1 if total_score > 0 else (-1 if total_score < 0 else 0)
    if _total_sign == 0:
        confidence = max(0, 100 - int(abs(total_score) * 3))
    else:
        _agree = sum(1 for s in _all_comp_scores if s * _total_sign > 0)
        _n     = len(_all_comp_scores)
        _agree_pct = _agree / _n if _n > 0 else 0.0
        _mag_norm  = min(1.0, abs(total_score) / 100.0)
        confidence = min(100, max(0, int((_agree_pct * 0.60 + _mag_norm * 0.40) * 100)))

    _delta_ok  = bkt.get("delta_active", False)
    _flow_note = (
        f"Net Δ-flow (PUT−CALL) {'positive — PUT dominant (bullish)' if flow_3 > 0 else 'negative — CALL dominant (bearish)'}"
        if _delta_ok
        else "⚠️ Δ-flow zero or proxy only — bias reflects GEX + flip, NOT delta flow"
    )
    narrative = (
        f"{_flow_note} | "
        f"GEX {'long-gamma (range-bound)' if gex_now > 0 else 'short-gamma (amplifying)'} | "
        f"{'Above' if (gf_now and spot_now > gf_now) else 'Below'} gamma flip"
        + (f" | Max pain pull {c5_label}" if c5_score != 0 else "")
    )

    return {
        "score"      : round(total_score, 1),
        "direction"  : direction,
        "confidence" : confidence,
        "components" : {
            "net_flow_dir" : (c1_score, c1_label),
            "flow_accel"   : (c2_score, c2_label),
            "gex_regime"   : (c3_score, c3_label),
            "flip_side"    : (c4_score, c4_label),
            "max_pain"     : (c5_score, c5_label),
        },
        "narrative"  : narrative,
        "delta_active": bkt.get("delta_active", False),
    }


# ─── EOD Erosion Model  IDENTICAL to Dash app ────────────────────────────────
_SCENARIOS = {"Bull +1%":+0.010,"Bull +0.5%":+0.005,"Base  0%":0.000,"Bear -0.5%":-0.005,"Bear -1%":-0.010}

def _iv_crush_assumption(cal_days):
    if cal_days == 0: return -0.35
    if cal_days == 1: return -0.15
    if cal_days <= 3: return -0.08
    if cal_days <= 7: return -0.03
    return -0.01

def compute_eod_erosion(df, spot, atm_iv, expiry_str):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    now   = now_ist()
    open_ = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    SESSION_HRS = 6.25
    if now >= close:    frac = 0.0
    elif now <= open_:  frac = 1.0
    else:
        elapsed = (now - open_).total_seconds() / 3600.0
        frac = max(0.0, (SESSION_HRS - elapsed) / SESSION_HRS)
    # H12 fix: `_bs_greeks` returns theta per CALENDAR day (/365). The previous
    # code multiplied theta by `frac` = (session_hours_remaining / 6.25), treating
    # the trading session as a calendar day and overestimating intraday decay by
    # ~24/6.25 ≈ 3.84×. The correct scaling is calendar-day-correct:
    #   theta_per_session_remaining = theta * (session_hours_remaining / 24)
    # We keep `frac` for the "fraction of session remaining" label/anchor but use
    # `frac_calendar` for the actual theta scaling. (Note: this makes EOD erosion
    # projections less aggressive — for expiry-day acceleration, see the
    # iv_crush_frac component, which is separately tuned.)
    frac_calendar = frac * (SESSION_HRS / 24.0)
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        cal_days = max((exp_date - datetime.now().date()).days, 0)
    except Exception:
        cal_days = 1
    iv_crush_frac = _iv_crush_assumption(cal_days)
    call_oi_med = float(df["call_oi"].replace(0, np.nan).median() or 1)
    put_oi_med  = float(df["put_oi"].replace(0, np.nan).median()  or 1)
    rows_call, rows_put = [], []
    for _, r in df.iterrows():
        K = float(r["strike"])
        for side in ("call", "put"):
            ltp   = safe_num(r.get(f"{side}_ltp", 0))
            theta = safe_num(r.get(f"{side}_theta", 0))
            delta = safe_num(r.get(f"{side}_delta", 0))
            gamma = safe_num(r.get(f"{side}_gamma", 0))
            vega  = safe_num(r.get(f"{side}_vega", 0))
            iv    = safe_num(r.get(f"{side}_iv", 0))
            oi    = safe_num(r.get(f"{side}_oi", 0))
            oi_chg= safe_num(r.get(f"{side}_oi_chg", 0))
            bid   = safe_num(r.get(f"{side}_bid", 0))
            ask   = safe_num(r.get(f"{side}_ask", 0))
            if ltp <= 0.50:
                continue
            delta_iv_pp   = iv * iv_crush_frac
            theta_component = theta * frac_calendar   # H12 fix: was `theta * frac` (3.84× over-estimated)
            scenario_erosions = {}
            for scen_name, pct_move in _SCENARIOS.items():
                dS = spot * pct_move
                new_spot = spot + dS
                intrinsic = max(0.0, new_spot - K) if side == "call" else max(0.0, K - new_spot)
                delta_p = delta*dS + 0.5*gamma*dS**2 + theta_component + vega*delta_iv_pp
                proj = max(intrinsic, ltp + delta_p)
                erosion = round((ltp - proj) / ltp * 100, 1)
                scenario_erosions[scen_name] = erosion
            base_erosion = scenario_erosions.get("Base  0%", 0.0)
            if base_erosion < 30.0:
                continue
            scen_values = list(scenario_erosions.values())
            erosion_min  = round(min(scen_values), 1)
            erosion_max  = round(max(scen_values), 1)
            robust_count = sum(1 for v in scen_values if v >= 30.0)
            abs_delta = abs(delta)
            if abs_delta <= 0.05: f1=30
            elif abs_delta <= 0.10: f1=25
            elif abs_delta <= 0.20: f1=18
            elif abs_delta <= 0.30: f1=10
            elif abs_delta <= 0.40: f1=3
            else: f1=-15
            if cal_days==0: f2=20
            elif cal_days==1: f2=15
            elif cal_days<=3: f2=10
            elif cal_days<=7: f2=5
            else: f2=0
            f3 = {5:20,4:15,3:10,2:5,1:0}.get(robust_count,0)
            med_oi = call_oi_med if side=="call" else put_oi_med
            if oi>med_oi*2.0 and oi_chg>0: f4=15
            elif oi>med_oi*2.0: f4=10
            elif oi>med_oi and oi_chg>0: f4=8
            elif oi>med_oi: f4=5
            elif oi_chg>0: f4=3
            else: f4=0
            if 0<iv<=atm_iv*0.80: f5=10
            elif 0<iv<=atm_iv*0.90: f5=6
            elif 0<iv<atm_iv: f5=3
            elif iv>atm_iv*1.20: f5=-5
            else: f5=0
            if ask>bid>0:
                spread_ratio = (ask-bid)/ltp
                if spread_ratio<0.10: f6=5
                elif spread_ratio<0.20: f6=3
                elif spread_ratio>0.35: f6=-3
                else: f6=1
            else: f6=0
            score = max(0, min(100, f1+f2+f3+f4+f5+f6))
            if score>=72: conf_label="HIGH"
            elif score>=50: conf_label="MEDIUM"
            else: conf_label="LOW"
            row = {"Strike":int(K),"LTP (₹)":round(ltp,2),
                   "Proj.EOD (₹)":round(max(0,ltp*(1-base_erosion/100)),2),
                   "Base Erosion %":base_erosion,
                   "Erosion Range":f"{erosion_min}%  {erosion_max}%",
                   "Robust (of 5)":robust_count,
                   "Θ Decay (₹)":round(abs(theta*frac_calendar),2),  # H12 fix: was theta*frac
                   "ν·ΔIV (₹)":round(vega*delta_iv_pp,2),
                   "Delta":round(delta,3),"IV %":round(iv,1),
                   "OI":int(oi),"OI Chg":int(oi_chg),
                   "Score":score,"Confidence":conf_label}
            if side=="call": rows_call.append(row)
            else:            rows_put.append(row)
    calls_df = pd.DataFrame(rows_call).sort_values("Score",ascending=False) if rows_call else pd.DataFrame()
    puts_df  = pd.DataFrame(rows_put).sort_values("Score",ascending=False)  if rows_put  else pd.DataFrame()
    return calls_df, puts_df


# ─────────────────────────────────────────────────────────────────────────────
# SERVER-SIDE PERSISTENT STATE: History + Settings + Bias/Smile history
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
_HISTORY_FILE       = os.path.join(_BASE_DIR, "nifty_history.json")
_SETTINGS_FILE      = os.path.join(_BASE_DIR, "nifty_settings.json")
_BIAS_HISTORY_FILE  = os.path.join(_BASE_DIR, "nifty_bias_history.json")
_SMILE_HISTORY_FILE = os.path.join(_BASE_DIR, "nifty_smile_history.json")

# CI #6 fix: single process-wide lock for ALL persistent-state writes.
# Prevents concurrent visitor sessions from corrupting the JSON files via
# interleaved read-modify-write cycles (which the previous bare `open(w)` calls
# allowed — last writer wins, file can be truncated if process is killed mid-write).
_persist_lock = threading.Lock()

# H25 fix: TTL cache for _load_owner_settings. The function was being called
# 2-3× per rerun from multiple sites (sidebar, get_server_data, banner) and
# each call did an `open + json.load`. With N visitors refreshing every 60s,
# that's 2-3N disk reads/minute for a file that changes only when the owner
# toggles a setting. 5s TTL coalesces reads within a single rerun.
_owner_settings_cache = {"data": None, "ts": 0.0}
_OWNER_SETTINGS_TTL = 5.0   # seconds

# H26 fix: mtime check for _load_bias_history. The function reads the entire
# JSON file from disk every rerun (60s) per visitor for cross-visitor dedup.
# With mtime check, we only re-read when the file actually changed.
_bias_history_cache = {"data": None, "mtime": -1.0}


def _atomic_json_write(path, obj):
    """CI #6 fix: atomic JSON write via write-temp-then-os.replace.

    - Writes to `<path>.tmp` first, then atomically renames to `<path>` via
      os.replace (POSIX atomicity guarantee).
    - Refuses to write NaN/Infinity (allow_nan=False) so the file stays
      RFC-8259 compliant and parseable by non-Python tools.
    - Caller must already hold _persist_lock if cross-file atomicity matters.
    """
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, allow_nan=False)
        os.replace(tmp, path)   # atomic on POSIX
    except (TypeError, ValueError) as e:
        # NaN/Infinity in obj — refuse to corrupt the file
        try:
            os.remove(tmp)
        except OSError:
            pass
        # Re-raise so the caller's `except Exception: pass` logs nothing,
        # but the existing file on disk is preserved.
        raise


# ── Main tick history ─────────────────────────────────────────────────────────
def _load_history():
    """Load history from disk on fresh session start."""
    with _persist_lock:
        try:
            with open(_HISTORY_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data[-500:]
        except Exception:
            pass
    return []

def _save_history(history):
    """Persist history to disk after every new tick."""
    with _persist_lock:
        try:
            _atomic_json_write(_HISTORY_FILE, history[-500:])
        except Exception:
            pass

# ── Owner settings (refresh interval, selected expiry) ───────────────────────
def _load_owner_settings():
    # H25 fix: TTL cache — coalesce multiple calls within 5 seconds.
    now = time.time()
    if (_owner_settings_cache["data"] is not None and
        now - _owner_settings_cache["ts"] < _OWNER_SETTINGS_TTL):
        return _owner_settings_cache["data"]
    with _persist_lock:
        try:
            with open(_SETTINGS_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {"refresh_interval": REFRESH_SECONDS, "selected_expiry": None}
    _owner_settings_cache["data"] = data
    _owner_settings_cache["ts"]   = now
    return data

def _save_owner_settings(settings):
    with _persist_lock:
        try:
            _atomic_json_write(_SETTINGS_FILE, settings)
        except Exception:
            pass
    # H25 fix: invalidate the TTL cache so the next read picks up the new value
    _owner_settings_cache["data"] = None
    _owner_settings_cache["ts"]   = 0.0

# ── S3/4 Bias chart history (persisted so mid-session joiners see full chart) ─
def _load_bias_history():
    # H26 fix: mtime check — only re-read when the file actually changed.
    # Was reading the entire JSON every rerun per visitor for cross-visitor dedup.
    try:
        current_mtime = os.path.getmtime(_BIAS_HISTORY_FILE)
    except OSError:
        current_mtime = -1.0
    # Cache hit if mtime is unchanged AND we have data (or both are -1 = file missing)
    if current_mtime == _bias_history_cache["mtime"]:
        return _bias_history_cache["data"] if _bias_history_cache["data"] is not None else []
    with _persist_lock:
        try:
            with open(_BIAS_HISTORY_FILE, "r") as f:
                data = json.load(f)
                data = data if isinstance(data, list) else []
        except Exception:
            data = []
    _bias_history_cache["data"]  = data
    _bias_history_cache["mtime"] = current_mtime
    return data

def _save_bias_history(bh):
    with _persist_lock:
        try:
            _atomic_json_write(_BIAS_HISTORY_FILE, bh[-60:])
        except Exception:
            pass
    # H26 fix: invalidate mtime cache so next read picks up the new file
    _bias_history_cache["data"]  = None
    _bias_history_cache["mtime"] = -1.0

# ── IV Smile history (persisted so mid-session joiners get trend context) ─────
def _load_smile_history():
    with _persist_lock:
        try:
            with open(_SMILE_HISTORY_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

def _save_smile_history(sh):
    with _persist_lock:
        try:
            _atomic_json_write(_SMILE_HISTORY_FILE, sh[-20:])
        except Exception:
            pass

# ── Session state initialisation (loads server-side data for every new visitor) ─
if "history" not in st.session_state:
    st.session_state.history = _load_history()          # seed from disk on new session

if "bias_history" not in st.session_state:
    _bh_disk = _load_bias_history()
    st.session_state.bias_history = _bh_disk
    # Set last_ts so we don't immediately duplicate the last entry; fall back to 0 if empty
    st.session_state.bias_history_last_ts = (
        float(_bh_disk[-1].get("_ts_unix", 0)) if _bh_disk else 0.0
    )

if "iv_smile_history" not in st.session_state:
    st.session_state.iv_smile_history = _load_smile_history()

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] { background: #F5F6FA; }
[data-testid="stHeader"] { background: transparent; }
div[data-testid="stMetric"] { background:#fff; border-radius:10px; padding:10px 14px;
    border:1px solid #E5E7EB; box-shadow:0 1px 4px rgba(0,0,0,0.06); }
.section-header { background:#EEF2FF; border-left:4px solid #5C35CC; border-radius:8px;
    padding:8px 16px; font-weight:700; font-size:13px; color:#5C35CC;
    margin-bottom:12px; letter-spacing:0.3px; }
.card { background:#fff; border-radius:10px; padding:14px 16px; border:1px solid #E5E7EB;
    box-shadow:0 1px 4px rgba(0,0,0,0.06); margin-bottom:8px; }
.badge { border-radius:6px; padding:2px 9px; font-size:12px; font-weight:700;
    display:inline-block; white-space:nowrap; }
.alert-danger { background:#FEF2F2; border:1px solid #DC2626; border-radius:10px;
    padding:12px 16px; margin-bottom:12px; }
.alert-watch  { background:#FFFBEB; border:1px solid #D97706; border-radius:10px;
    padding:12px 16px; margin-bottom:12px; }
.alert-none   { background:#ECFDF5; border:1px solid #059669; border-radius:10px;
    padding:12px 16px; margin-bottom:12px; }
.data-live  { color:#059669; font-weight:700; font-size:12px; }
.data-demo  { color:#D97706; font-weight:700; font-size:12px; }

/* ── Mobile & iPad Responsive ─────────────────────────────────────────── */
@media (max-width: 900px) {
  .section-header { font-size:12px; padding:7px 12px; }
  .card { padding:10px 12px; }
  div[data-testid="stMetric"] { padding:8px 10px; }
}
@media (max-width: 600px) {
  .section-header { font-size:11px; padding:6px 10px; letter-spacing:0; }
  .card { padding:8px 10px; border-radius:8px; }
  div[data-testid="stMetric"] { padding:6px 8px; border-radius:8px; }
  .badge { font-size:10px; padding:2px 7px; }
  .js-plotly-plot .plotly .gtitle { font-size:10px !important; }
  /* H22 fix: GRF + Combined Decision grids had no media query — 3-column / 2-column
     layouts were squishing to ~120px each on 360px phones. Force single column. */
  .grf-grid-3, .grf-grid-2 { grid-template-columns: 1fr !important; }
  /* Inline div-based grids (style attribute) can't be targeted via CSS, but
     st.columns-based layouts already collapse to single column automatically. */
}
[data-testid="stAppViewContainer"],
[data-testid="block-container"] {
  max-width: 100% !important;
  overflow-x: hidden !important;
}
/* Section 9 sentiment indicator cards ─ wrap gracefully on narrow screens */
.s9-ind-row { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }
.s9-ind-card {
  flex: 1 1 140px; min-width:120px;
  background:#fff; border-radius:10px; padding:10px 12px;
  text-align:center; box-sizing:border-box;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"""
    <div style="background:#5C35CC;border-radius:12px;padding:14px 22px;margin-bottom:12px;">
      <div style="color:#fff;font-size:22px;font-weight:800;margin:0;">
         Shantanu's Options Analysis Dashboard
      </div>
      <div style="color:rgba(255,255,255,0.75);font-size:11px;margin-top:3px;">
        NIFTY 50 · NIFTY Futures · Live F&O Intelligence · Bias Engine · Strategy Engine
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    mh = is_market_hours()
    data_status = " LIVE DATA (Dhan API)" if (USE_DHAN and mh) else (" DEMO MODE" if USE_DEMO_MODE else " After-Hours (Dhan Connected)")
    st.markdown(f"""
    <div style="background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:10px 14px;text-align:right;margin-top:4px;">
      <div style="font-size:12px;font-weight:700;color:{'#059669' if USE_DHAN and mh else '#D97706'};">{data_status}</div>
      <div style="font-size:11px;color:#6B7280;margin-top:2px;">{ist_str()}</div>
      <div style="font-size:11px;color:#6B7280;">Auto-refresh: {_effective_refresh}s</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA STATUS BANNER
# ─────────────────────────────────────────────────────────────────────────────
def _data_status_banner(use_dhan, market_open, use_demo, refresh_secs, data_source=None):
    """Colour-coded full-width banner showing exactly what data state the app is in.

    CI #10 fix: added `data_source` param so the banner can distinguish a real
    API ERROR (live mode that failed and fell back to demo) from a genuine
    demo-mode (no credentials). Previously both showed the same DEMO banner,
    masking transient API failures.
    """
    _now = ist_str("%H:%M:%S IST  |  %a %d %b %Y")

    # CI #10 fix: detect API-error fallback (live mode that failed over to demo)
    _is_api_error = bool(data_source and "API ERROR" in data_source)

    if _is_api_error:
        # ── RED — live mode but API call failed; serving stale demo data ──────
        bg, border = "#FEF2F2", "#DC2626"
        dot = ""
        headline = "API ERROR — showing fallback demo data"
        detail   = ("Dhan API call failed (timeout, rate-limit, or auth issue) · "
                    "Values shown are <strong>simulated</strong>, not real market data · "
                    "Will auto-retry on next refresh")
        label_bg, label_fg = "#DC2626", "#ffffff"
        label_text = "● API ERROR"

    elif use_dhan and market_open:
        # ── GREEN — live, everything working ──────────────────────────────────
        bg, border, dot_color = "#ECFDF5", "#059669", "#059669"
        dot = ""
        headline = "LIVE DATA — Calculations are LIVE"
        detail   = (f"Sourced from <strong>Dhan API</strong> in real time · "
                    f"Auto-refresh every <strong>{refresh_secs}s</strong> · "
                    f"Market hours: MonFri 09:1515:30 IST")
        label_bg, label_fg = "#059669", "#ffffff"
        label_text = "● LIVE"

    elif use_dhan and not market_open:
        # ── AMBER — Dhan connected but market closed; data is stale ──────────
        n = now_ist()
        is_weekend = n.weekday() >= 5
        reason = "Weekend" if is_weekend else "Market closed (09:1515:30 IST)"
        bg, border = "#FFFBEB", "#D97706"
        dot = ""
        headline = f"STALE DATA — {reason}"
        detail   = (f"Last snapshot from <strong>Dhan API</strong> — no new ticks until market reopens · "
                    f"Auto-refresh every <strong>{refresh_secs}s</strong> to keep app alive")
        label_bg, label_fg = "#D97706", "#ffffff"
        label_text = "● STALE"

    else:
        # ── YELLOW — demo / no credentials ────────────────────────────────────
        bg, border = "#FEFCE8", "#CA8A04"
        dot = ""
        headline = "DEMO MODE — Synthetic data only"
        detail   = ("No Dhan API credentials configured · All values are <strong>simulated</strong> · "
                    "Add <code>DHAN_CLIENT_ID</code> and <code>DHAN_ACCESS_TOKEN</code> to Streamlit secrets for live data")
        label_bg, label_fg = "#CA8A04", "#ffffff"
        label_text = "● DEMO"

    st.markdown(f"""
    <div style="
        background:{bg};
        border:1.5px solid {border};
        border-radius:10px;
        padding:10px 18px;
        margin-bottom:14px;
        display:flex;
        align-items:center;
        gap:14px;
        flex-wrap:wrap;
    ">
      <span style="
          background:{label_bg};color:{label_fg};
          font-size:11px;font-weight:800;letter-spacing:0.8px;
          border-radius:5px;padding:3px 10px;white-space:nowrap;
      ">{label_text}</span>
      <span style="font-size:13px;font-weight:800;color:#1A1A2E;white-space:nowrap;">
        {dot} {headline}
      </span>
      <span style="font-size:11px;color:#374151;flex:1;min-width:180px;">
        {detail}
      </span>
      <span style="font-size:11px;color:#6B7280;white-space:nowrap;margin-left:auto;">
         {_now}
      </span>
    </div>
    """, unsafe_allow_html=True)

# CI #10 fix: the banner used to be rendered here, BEFORE `payload` was fetched.
# That meant on the very first run (no payload yet) the API-ERROR branch could
# never fire. The banner has been MOVED to after the fetch (L~4520) so it can
# inspect `data_source` from the just-fetched payload.

# ─────────────────────────────────────────────────────────────────────────────
# OWNER SIDEBAR + CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
# Fetch expiry list first so sidebar can show it
_expiry_list = fetch_dhan_expiry_list() if USE_DHAN else []
sel_expiry, _manual_refresh = _render_owner_sidebar(_expiry_list)

# If owner selected an expiry use it; otherwise auto (nearest)
if not sel_expiry and _expiry_list:
    sel_expiry = _expiry_list[0]

# Handle manual refresh triggered from owner sidebar only
if _manual_refresh:
    _force_server_refresh()   # expire server cache; next get_server_data() re-fetches from Dhan
    st.rerun()

# Show a compact read-only expiry label on the main page for all visitors
with st.container():
    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        st.markdown("**Instrument:** NIFTY 50 &nbsp;·&nbsp; "
                    f"**Expiry:** {sel_expiry if sel_expiry else 'Auto (nearest)'}")
        st.caption("Read-only live view · Open ⚙️ sidebar to change expiry, refresh rate, or force reload.")
    with ctrl_col2:
        st.markdown(f"<div style='text-align:right;font-size:11px;color:#6B7280;padding-top:6px;'>"
                    f" Auto-refresh: <strong>{_effective_refresh}s</strong></div>",
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────────────────────────────────────
# H24 fix: previously `st.spinner(...)` wrapped get_server_data unconditionally,
# causing a visible flicker every 60s even when the server cache was fresh
# (returning in <1ms). Pre-check cache age first; only show spinner when a real
# network fetch is about to happen.
_settings_h24  = _load_owner_settings()
_interval_h24  = _settings_h24.get("refresh_interval", REFRESH_SECONDS)
_cache_age_h24 = time.time() - _srv_cache.get("last_fetch_ts", 0.0)
_needs_fetch_h24 = (_cache_age_h24 >= _interval_h24) or (_srv_cache.get("payload") is None)

if _needs_fetch_h24:
    with st.spinner("Fetching NIFTY option chain"):
        payload, data_source, _payload_fetch_ts = get_server_data(sel_expiry)
else:
    payload, data_source, _payload_fetch_ts = get_server_data(sel_expiry)

# CI #10 fix: render the data-status banner HERE (after fetch) so it can inspect
# `data_source` and distinguish a real API ERROR from genuine DEMO MODE.
_data_status_banner(USE_DHAN, mh, USE_DEMO_MODE, _effective_refresh, data_source=data_source)

if payload is None:
    st.error("❌ Could not fetch option chain data. Please check your API credentials or try again.")
    st.stop()

m        = payload["metrics"]
spot     = payload["spot"]
expiry   = payload["expiry"]
# CHANGE 1 (audit fix): legacy bias is retained ONLY for its regime / vol_regime /
# near_flip fields (which `strategy_recommendation` consumes for the FLIP / RANGE
# / PINNED branches). Direction + confidence now come from the Combined Decision
# panel (writer-positioning convention) — see the adapter below, set after
# `_combined_decision` is computed.
bias     = compute_nifty_bias(m, st.session_state.history)
history  = st.session_state.history

# Append history entry
df_band_records = payload["df_band"]
call_oi_total = sum(r.get("call_oi", 0) for r in df_band_records)
put_oi_total  = sum(r.get("put_oi", 0)  for r in df_band_records)
hist_entry    = build_history_entry(
    m, spot, call_oi_total, put_oi_total, expiry,
    synth_excess=payload.get("synth_excess"),
    basis_gap=payload.get("basis_gap"),
    traded_basis=payload.get("traded_basis"),
)
# Tag with server fetch timestamp for dedup.
# Multiple visitors refreshing against the same cached server snapshot must
# produce exactly ONE history entry — not one per visitor per refresh.
hist_entry["_fetch_ts"] = _payload_fetch_ts
hist_entry["_df_band_records"] = df_band_records  # Enhanced NDM: store per-strike LTP for prev-tick comparison
# Only append if this is a genuinely new Dhan data fetch (fetch_ts changed)
if not history or history[-1].get("_fetch_ts", 0) != _payload_fetch_ts:
    history.append(hist_entry)
    _save_history(history)   # persist new tick to disk
history = history[-500:]
st.session_state.history = history

# ── Fix #2: today-only history for session-sensitive intraday functions ───────
# Full `history` (up to 500 ticks, multi-day) is kept for compute_temporal_iv_rank
# which needs cross-day context to compute a meaningful IV percentile.
# All session-bucketed functions (DW flow, raw OI, sentiments, Section 5 baseline)
# only make sense within the current trading day — yesterday's 09:30 bucket must
# not merge with today's 09:30.
_today_str    = date.today().isoformat()                           # e.g. "2026-06-27"
today_history = [h for h in history if h.get("ts", "").startswith(_today_str)]
# ── end Fix #2 setup ──────────────────────────────────────────────────────────

# ── Fix #4: Adaptive NORMAL_SKEW_BASELINE ─────────────────────────────────────
# Compute a trailing median of norm_skew over recent neutral-regime sessions
# (|s34_score| < 20) from persisted bias history, rather than using the hardcoded
# value of 30.0.  Requires norm_skew to be stored in bias_history entries
# (added below in the bias-history append block).  Falls back to 30.0 until
# ≥5 neutral-session data points accumulate.
_bh_for_baseline = _load_bias_history()   # mtime-cached — cheap second call
_neutral_skews = [
    float(x["norm_skew"])
    for x in _bh_for_baseline
    if abs(safe_num(x.get("score", 999))) < 20   # neutral session
    and isinstance(x.get("norm_skew"), (int, float))
]
if len(_neutral_skews) >= 5:
    _adaptive_skew_baseline = round(
        float(sorted(_neutral_skews)[len(_neutral_skews) // 2]), 1   # median
    )
else:
    _adaptive_skew_baseline = 30.0   # not enough history yet — use default
# ── end Fix #4 setup ──────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# MARKET BIAS SUMMARY (top-level)
# NOTE (CHANGE 2 audit fix): `bs`/`regime` here come from the legacy
# `compute_nifty_bias` engine, which uses SIGNED delta × OI (net_delta). That
# measures dealer hedge-flow pressure, NOT writer positioning. The number is
# still useful as a hedge-flow read, but should not be interpreted as the
# authoritative directional bias — that role belongs to the S3/4 engine below.
# ─────────────────────────────────────────────────────────────────────────────
bs              = bias["bias_score"]
bc              = GREEN if bs > 15 else (RED if bs < -15 else AMBER)
direction_label = bias["direction"]
regime          = bias["regime"]
# ── Section 3 & 4 bias (5-signal engine) — recorded every 5 minutes ──────────
# Step 0: Roll detection — fetch back-expiry full OI band to assess whether
# near-ATM OI changes are mechanical roll vs genuine directional activity.
# This runs BEFORE S34 bias so the roll_discount reaches S2 at score time.
_roll_back_exp     = _expiry_list[1] if len(_expiry_list) > 1 else None
_back_oi_band_df   = fetch_back_expiry_oi_band(_roll_back_exp) if _roll_back_exp else None
_front_df_for_roll = pd.DataFrame(payload["df_band"]) if payload.get("df_band") else None
_roll_data         = detect_roll_activity(_front_df_for_roll, _back_oi_band_df, spot)
_s34_bias = compute_section34_bias(
    payload["df_band"], m, spot,
    roll_discount=_roll_data["momentum_discount"],
    front_expiry=_expiry_list[0] if _expiry_list else None,   # Fix #3: dynamic roll weekday
    skew_baseline=_adaptive_skew_baseline,                     # Fix #4: adaptive S4 baseline
)
_s34_score = _s34_bias["bias_score"]
_s34_breakdown = _s34_bias.get("signal_breakdown", {})

# ── v4.1 S4 FIX: Option 2 — Intraday Skew Momentum Anchor (±3 pts) ──────────
# The ATM-normalised skew (Option 1, inside compute_section34_bias) removes the
# chronic bearish baseline. This block adds the intraday DIRECTIONAL component:
# has fear been building or easing SINCE THE SESSION OPEN?
#
# session_state.opening_norm_skew is set on the first refresh each session and
# held fixed. Each subsequent refresh measures the change from that anchor.
# A rising norm_skew intraday = fear building = bearish contribution (up to −3 pts).
# A falling norm_skew intraday = fear easing  = bullish contribution (up to +3 pts).
#
# This naturally resets at the start of each new session (Streamlit restarts) and
# requires zero stored history beyond the current session.
_current_norm_skew = _s34_bias.get("norm_skew", 30.0)
if "opening_norm_skew" not in st.session_state or st.session_state.opening_norm_skew is None:
    st.session_state.opening_norm_skew = _current_norm_skew
_delta_skew = _current_norm_skew - st.session_state.opening_norm_skew
# 15 pct-point intraday change → full ±3 pts; sign: rising skew = bearish (negative)
_s4_intra = max(-3.0, min(3.0, -(_delta_skew / 15.0) * 3.0))
# Merge into bias score and breakdown (S4 now reflects level + intraday momentum)
_s34_bias["bias_score"] = max(-100.0, min(100.0, _s34_bias["bias_score"] + _s4_intra))
_s34_bias["signal_breakdown"]["S4 IV Skew"] = round(
    _s34_bias["signal_breakdown"].get("S4 IV Skew", 0.0) + _s4_intra, 1
)
# Refresh local aliases so all downstream code sees the updated values
_s34_score     = _s34_bias["bias_score"]
_s34_breakdown = _s34_bias.get("signal_breakdown", {})
# ── end S4 intraday anchor ────────────────────────────────────────────────────

# ── Early IV smile call for the Combined Bias panel (Chapter 17 & 18) ────
# Uses df_band already loaded above; relies on smile session history if available.
_early_smile_hist = st.session_state.get("iv_smile_history", [])
_early_df_band    = pd.DataFrame(payload["df_band"]) if payload.get("df_band") else None
_early_smile      = classify_iv_smile_scenario(
    _early_df_band, m, spot, _early_smile_hist
) if _early_df_band is not None else None
# ── end early IV smile call ───────────────────────────────────────────────

# ── Enhanced Price Confirmation Layer  (v7 — surgical addition) ──────────────
# Module A: VWAP + Opening Range
_intraday_candles  = fetch_nifty_intraday_candles()
_vwap_or_data      = compute_vwap_opening_range(_intraday_candles)

# Module B: Term Structure (front vs back expiry ATM IV)
_back_expiry       = _expiry_list[1] if len(_expiry_list) > 1 else None
_back_atm_iv       = fetch_back_expiry_atm_iv(_back_expiry) if _back_expiry else None
_ts_data           = compute_term_structure_signal(
    safe_num(m.get("atm_iv", 0)), _back_atm_iv,
    front_expiry=_expiry_list[0] if _expiry_list else None,   # v4.1: dynamic expiry-day suppression
)

# ── v4.1 S5 REPLACEMENT: IV Term Structure Injection (±8 pts) ────────────────
# compute_section34_bias() leaves s5 = 0.0 (placeholder). We inject the real
# value here because _ts_data is not available inside that function.
#
# DTE confidence weight: the term structure comparison becomes noise-dominated
# when the front expiry has very few days left (IV collapses erratically).
# Weight tapers from 1.0 (≥4 days, Monday) → 0.75 (3 days) → 0.50 (2 days, Wed)
# → 0.0 (1 day). Expiry day itself is already suppressed to ts_score=0
# inside compute_term_structure_signal() itself — no additional guard needed.
#
# This replaces PCR Contrarian (S5) which was correlated with S1 (both use
# put_OI / call_OI). IV term structure uses a completely different data source.
from datetime import date as _dte_date_cls
try:
    _dte_front = max(1, (_dte_date_cls.fromisoformat(_expiry_list[0]) - _dte_date_cls.today()).days) \
                 if _expiry_list else 4
except Exception:
    _dte_front = 4   # safe fallback: assume normal Monday-like DTE

if   _dte_front >= 4: _s5_dte_weight = 1.00   # Mon / any day with ≥4 days left
elif _dte_front == 3: _s5_dte_weight = 0.75   # Tuesday — mild suppression
elif _dte_front == 2: _s5_dte_weight = 0.50   # Wednesday — moderate suppression
else:                 _s5_dte_weight = 0.00   # 1 day left: fully suppress

_s5_ts = round(_ts_data.get("ts_score", 0.0) * _s5_dte_weight, 1)
# Inject into the running bias score and breakdown
_s34_bias["bias_score"] = max(-100.0, min(100.0, _s34_bias["bias_score"] + _s5_ts))
_s34_bias["signal_breakdown"]["S5 Term Str"] = _s5_ts
_s34_score     = _s34_bias["bias_score"]
_s34_breakdown = _s34_bias.get("signal_breakdown", {})
# ── end S5 term structure injection ──────────────────────────────────────────

# Module C: India VIX
_vix_raw           = fetch_india_vix_ltp()
# Maintain a lightweight intraday VIX history in session_state for spike detection
if "vix_history" not in st.session_state:
    st.session_state.vix_history = []
if _vix_raw > 0:
    if (not st.session_state.vix_history or
            st.session_state.vix_history[-1] != _vix_raw):
        st.session_state.vix_history.append(_vix_raw)
        st.session_state.vix_history = st.session_state.vix_history[-30:]
_vix_data          = classify_vix_signal(_vix_raw, st.session_state.vix_history)

# Aggregate into Enhanced Price Bias
_enhanced_bias     = compute_enhanced_price_bias(
    _vwap_or_data, _ts_data, _vix_data, _s34_score, spot
)
# ── end Enhanced Price Confirmation Layer ─────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════
# v4 IMPROVEMENTS — All new computations inserted here
# ═══════════════════════════════════════════════════════════════════

# ── v4 #5: Feed Enhanced Price Layer into Combined Decision ──────────
_combined_decision = generate_combined_decision(_s34_bias, _early_smile, m, _enhanced_bias)

# ── v4 #1: Bias Velocity (Signal 6) — computed here where bias_history is available ──
_bias_hist = st.session_state.get("bias_history", [])
if len(_bias_hist) >= 3:
    _recent_scores = [safe_num(x.get("score", 0)) for x in _bias_hist[-4:]]
    _recent_scores.append(float(_s34_score))
    _velocity = _recent_scores[-1] - _recent_scores[-2]
    _accel = (_recent_scores[-1] - _recent_scores[-2]) - (_recent_scores[-2] - _recent_scores[-3]) if len(_recent_scores) >= 3 else 0.0
    # Scale: 20-pt change over 1 tick = full ±10 score
    _s6 = max(-10.0, min(10.0, (_velocity / 20.0) * 10.0))
    # Update the bias score with velocity
    _s34_score_v4 = max(-100.0, min(100.0, _s34_score + _s6))
    _s34_bias["bias_score"] = _s34_score_v4
    if _s34_score_v4 >= 15: _s34_bias["direction"] = "BULLISH"
    elif _s34_score_v4 <= -15: _s34_bias["direction"] = "BEARISH"
    else: _s34_bias["direction"] = "NEUTRAL"
    _s34_bias["signal_breakdown"]["S6 Velocity"] = round(_s6, 1)
else:
    _velocity = 0.0; _accel = 0.0; _s6 = 0.0; _s34_score_v4 = _s34_score

# ── CHANGE 1 (audit fix): Rewire strategy_recommendation to Combined Decision ──
# The legacy `bias` dict (compute_nifty_bias) uses signed-delta net_delta as a
# directional signal — that's a hedge-flow metric, not a writer-positioning
# metric. The Combined Decision panel's `_s34_bias["direction"]` is the correct
# writer-positioning read (|Δ|-weighted, put-writing = bullish convention).
#
# We preserve `regime / vol_regime / near_flip` from the legacy bias because
# `strategy_recommendation` consumes them for the FLIP / RANGE / PINNED
# branches (and those regime classifications come from `classify_gamma_regime`,
# which is conceptually independent of the directional convention).
#
# We also preserve the legacy `confidence` (not `enhanced_conf`) because the
# WAIT threshold `BIAS_WEIGHTS["confidence_min_strategy"]=35` was tuned against
# the legacy ~73-max scale. Switching to enhanced_conf (0-100) would shift the
# effective threshold — leave that for a follow-up calibration pass.
_strat_bias = {
    "direction":  _s34_bias.get("direction", "NEUTRAL"),
    "confidence": bias.get("confidence", 0),
    "regime":     bias.get("regime", "TRANSITION"),
    "vol_regime": bias.get("vol_regime", "MID_VOL"),
    "near_flip":  bias.get("near_flip", False),
}
strat = strategy_recommendation(_strat_bias, m, st.session_state.history)

# ── v4 #4: Divergence Proximity Score (0-100) ─────────────────────────
def _compute_divergence_proximity(s34_score, scenario_id, pcr, iv_rank):
    """How close are we to triggering any of the 5 divergence types?
    Returns 0-100. >= 60 triggers an APPROACHING DIVERGENCE alert."""
    scores = []
    # Type 1 proximity: S3/4 approaching strong bear AND smile approaching crash fear
    t1_s34 = max(0, min(100, (abs(s34_score) - 40) / 11 * 100)) if s34_score < -40 else 0
    t1_smile = max(0, min(100, (iv_rank - 50) / 15 * 100)) if iv_rank > 50 else 0
    if scenario_id in (1, 2, 3): t1_smile = max(t1_smile, 60)
    scores.append((t1_s34 + t1_smile) / 2)
    # Type 2 proximity: S3/4 approaching strong bull AND smile has any put skew
    t2_s34 = max(0, min(100, (s34_score - 40) / 11 * 100)) if s34_score > 40 else 0
    t2_smile = 70 if scenario_id in (1, 3) else (30 if scenario_id in (2, 9) else 0)
    scores.append((t2_s34 + t2_smile) / 2)
    # Type 3 proximity: S3/4 narrowing to neutral AND IV rank compressing
    t3_neutral = max(0, min(100, (15 - abs(s34_score)) / 15 * 100)) if abs(s34_score) < 15 else 0
    t3_compress = max(0, min(100, (25 - iv_rank) / 5 * 100)) if iv_rank < 25 else 0
    scores.append((t3_neutral + t3_compress) / 2)
    # Type 5 proximity: PCR approaching extreme AND S3/4 moderately biased
    t5_pcr = 0
    if pcr < 0.70: t5_pcr = max(0, min(100, (0.70 - pcr) / 0.20 * 100))
    elif pcr > 1.55: t5_pcr = max(0, min(100, (pcr - 1.55) / 0.25 * 100))
    t5_s34 = max(0, min(100, (abs(s34_score) - 15) / 15 * 100)) if abs(s34_score) > 15 else 0
    scores.append((t5_pcr + t5_s34) / 2)
    return round(max(scores), 1) if scores else 0.0

_div_proximity = _compute_divergence_proximity(
    _s34_score, _early_smile.get("scenario_id", 0) if _early_smile else 0,
    m.get("pcr", 1.0), m.get("iv_rank", 50)
)

# ── v4 #8: Gamma Flip Proximity ──────────────────────────────────────
_gamma_flip_val = m.get("gamma_flip")
_gamma_flip_proximity = None
if _gamma_flip_val and _gamma_flip_val > 0 and spot > 0:
    _flip_dist = abs(spot - _gamma_flip_val)
    _wall_w = safe_num(m.get("wall_width", 400))
    _step = max(_wall_w / 20, 50)
    _proximity_threshold = max(2.0 * _step, 100)
    _gamma_flip_proximity = {
        "flip_strike": round(_gamma_flip_val, 0),
        "distance_pts": round(_flip_dist, 1),
        "threshold_pts": round(_proximity_threshold, 1),
        "pct_of_threshold": round(_flip_dist / _proximity_threshold * 100, 1) if _proximity_threshold > 0 else 0,
        "side": "ABOVE" if spot > _gamma_flip_val else "BELOW",
        "zone": "FLIP_ZONE" if _flip_dist < _proximity_threshold else "SAFE",
        "regime_risk": "HIGH" if _flip_dist < _step else ("ELEVATED" if _flip_dist < _proximity_threshold else "LOW"),
    }

# ── v4 #7: OI Momentum Exhaustion ────────────────────────────────────
_oi_exhaustion = None
if len(_bias_hist) >= 5:
    _mom_hist = [safe_num(x.get("s2", 0)) for x in _bias_hist[-6:]]
    _mom_hist.append(_s34_bias["signal_breakdown"].get("S2 Momentum", 0))
    if len(_mom_hist) >= 5:
        _sign = 1 if _mom_hist[-1] > 0 else (-1 if _mom_hist[-1] < 0 else 0)
        if _sign != 0:
            _signed = [v * _sign for v in _mom_hist]
            _all_positive = all(v > 0 for v in _signed)
            if _all_positive:
                _magnitudes = [abs(v) for v in _mom_hist]
                _recent_mag = np.mean(_magnitudes[-2:])
                _earlier_mag = np.mean(_magnitudes[:3]) if len(_magnitudes) >= 3 else _recent_mag
                _exhaust_ratio = _recent_mag / _earlier_mag if _earlier_mag > 0.5 else 1.0
                _oi_exhaustion = {
                    "direction": "BULL" if _sign > 0 else "BEAR",
                    "exhaust_ratio": round(_exhaust_ratio, 2),
                    "exhausting": _exhaust_ratio < 0.50,
                    "label": ("EXHAUSTING" if _exhaust_ratio < 0.50 else
                             "FADING" if _exhaust_ratio < 0.75 else "STRONG"),
                    "color": ("#DC2626" if _exhaust_ratio < 0.50 else
                             "#F59E0B" if _exhaust_ratio < 0.75 else "#059669"),
                }

# ── v4 #6: Inter-Expiry OI Flow / Roll Signal ─────────────────────────
_inter_expiry_signal = None
if len(_expiry_list) > 1 and USE_DHAN:
    try:
        _back_exp = _expiry_list[1]
        # CI #8 fix: use the cached wrapper so concurrent visitors don't each
        # fire a fresh Dhan POST for the back expiry (was violating ~1-req/3s
        # rate limit). 5-min TTL on the cached wrapper is sufficient because
        # roll detection doesn't need per-minute granularity.
        _back_chain, _, _ = fetch_dhan_option_chain_cached(_back_exp)
        if not _back_chain.empty:
            _back_m = compute_metrics(_back_chain, spot, _back_exp)
            if _back_m:
                _front_pcr = m.get("pcr", 1.0)
                _back_pcr = _back_m.get("pcr", 1.0)
                _pcr_diff = _front_pcr - _back_pcr
                _front_mom = safe_num(m.get("momentum", 0))
                _back_mom = safe_num(_back_m.get("momentum", 0))
                # Roll detection: front momentum fading while back building
                _roll_signal = None
                if abs(_front_mom) < abs(_back_mom) * 0.3 and abs(_back_mom) > 500:
                    _roll_dir = "BEAR" if _back_mom < 0 else "BULL"
                    _roll_signal = f"Rolling {_roll_dir} to next expiry"
                _inter_expiry_signal = {
                    "front_pcr": round(_front_pcr, 2),
                    "back_pcr": round(_back_pcr, 2),
                    "pcr_diff": round(_pcr_diff, 2),
                    "front_momentum": round(_front_mom, 0),
                    "back_momentum": round(_back_mom, 0),
                    "roll_signal": _roll_signal,
                    "available": True,
                }
    except Exception:
        pass

# ── v4 #3: Smart Money OI Quality Filter (applied to history entry) ────
# Store smart money filtered metrics for next tick's use
if "_smart_money_stats" not in st.session_state:
    st.session_state._smart_money_stats = {}


_now_bias = time.time()
# Dedup on server fetch timestamp, not wall-clock time.
# Reading from disk catches entries written by OTHER visitor sessions, preventing
# duplicate bias points from multiple simultaneous visitors.
_bh_disk_cur = _load_bias_history()
_last_bh_fetch_ts = float(_bh_disk_cur[-1].get("_fetch_ts", 0)) if _bh_disk_cur else 0.0

if _payload_fetch_ts != _last_bh_fetch_ts:
    # Start from the freshest disk state so no visitor session goes out of sync
    _bh_tmp = _bh_disk_cur
    _bh_tmp.append({
        "ts":        datetime.fromtimestamp(_payload_fetch_ts, tz=IST).strftime("%H:%M"),
        "_ts_unix":  _payload_fetch_ts,
        "_fetch_ts": _payload_fetch_ts,   # server fetch id — used for cross-session dedup
        "spot":      spot,
        "score":     float(_s34_score),
        "direction": _s34_bias["direction"],
        "s1":        _s34_breakdown.get("S1 Net OI",     0.0),
        "s2":        _s34_breakdown.get("S2 Momentum",   0.0),
        "s3":        _s34_breakdown.get("S3 Key Levels", 0.0),
        "s4":        _s34_breakdown.get("S4 IV Skew",    0.0),
        "s5":        _s34_breakdown.get("S5 Term Str",   0.0),
        "s6":        _s34_breakdown.get("S6 Velocity",   0.0),
        # Fix #4: store norm_skew so adaptive NORMAL_SKEW_BASELINE can self-calibrate
        "norm_skew": float(_s34_bias.get("norm_skew", 30.0)),
    })
    st.session_state.bias_history = _bh_tmp[-60:]   # ~5 hrs at data-refresh cadence
    st.session_state.bias_history_last_ts = _now_bias
    _save_bias_history(st.session_state.bias_history)   # persist for mid-session joiners
# ─────────────────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════════
# GREEK RISK FRAMEWORK — Intraday Bias & Confidence Score           (v1.0)
# Derived from live metrics already computed above:
#   Net Delta · OI Momentum · GEX · Gamma Walls · PCR · Max Pain · IV Rank
# Scoring: Gamma (0-3) + Delta (0-3) + Momentum (0-4) = 0-10
# ═════════════════════════════════════════════════════════════════════════════

def _compute_grf(m_dict, spot_px):
    """Greek Risk Framework scorer — all inputs from compute_metrics() dict."""
    nd      = safe_num(m_dict.get("net_delta",           0))
    mom     = safe_num(m_dict.get("momentum",            0))
    gex     = safe_num(m_dict.get("gex",                 0))
    d_res   = safe_num(m_dict.get("dist_to_resistance",  0))   # resistance - spot  (>0 = spot below wall)
    d_sup   = safe_num(m_dict.get("dist_to_support",     0))   # spot - support      (>0 = spot above wall)
    mp_val  = safe_num(m_dict.get("max_pain",       spot_px))
    pcr     = safe_num(m_dict.get("pcr",                1.0))
    iv_r    = safe_num(m_dict.get("iv_rank",             50))
    gflip   = m_dict.get("gamma_flip")
    sup_w   = safe_num(m_dict.get("support",             0))
    res_w   = safe_num(m_dict.get("resistance",          0))
    fac     = []

    # BUG 3 FIX — adaptive noise floor: scale thresholds to actual OI size.
    # Quiet days have low absolute net_delta; active days have high values.
    # Using 0.1% / 0.05% of total wide-band OI as the meaningful-signal floor.
    _oi_scale = max(1.0, safe_num(m_dict.get("call_oi_total", 0)) + safe_num(m_dict.get("put_oi_total", 0)))
    nd_sig    = abs(nd)  > max(100, _oi_scale * 0.001)
    mom_sig   = abs(mom) > max(50,  _oi_scale * 0.0005)

    # 1. Gamma Score (0-3): range quality from GEX + wall distances ──────────
    g = 0
    # BUG 1 FIX — only award buffer points when spot is INSIDE the S/R band.
    # When d_res ≤ 0 spot has broken above resistance; when d_sup ≤ 0 spot has
    # broken below support. Using abs() on a negative distance made a breakout
    # falsely look like a safe buffer. Instead: breakout = 0 gamma score.
    inside_band = (d_res > 0) and (d_sup > 0)
    if inside_band:
        min_buf = (min(d_res, d_sup) / spot_px * 100) if spot_px > 0 else 0
        if   min_buf > 1.0: g += 2; fac.append(f"Walls {min_buf:.1f}% from spot — safe sell range")
        elif min_buf > 0.5: g += 1; fac.append(f"Moderate wall buffer ({min_buf:.1f}%)")
        else:                        fac.append(f"Walls very close ({min_buf:.1f}%) — elevated gamma risk")
        if gex > 0: g += 1
    else:
        broke_dir = "above resistance" if d_res <= 0 else "below support"
        fac.append(f"⚠ Spot {broke_dir} — gamma range breached, avoid selling")
    g = min(g, 3)

    # 2. Delta Score (0-3): net delta direction + confirming anchors ──────────
    d = 0
    nd_bull = nd > 0
    if nd_sig:
        d += 1
        fac.append(f"Net delta {'bullish' if nd_bull else 'bearish'} ({nd:+,.0f})")
    if nd_sig and mp_val > 0 and spot_px > 0:
        mp_bull = mp_val > spot_px
        if nd_bull == mp_bull and abs(mp_val - spot_px) > 20:
            d += 1
            fac.append(f"Max pain ({int(mp_val)}) confirms {'upside' if mp_bull else 'downside'} pull")
    if nd_sig and gflip is not None:
        gf = safe_num(gflip)
        if gf > 0 and (spot_px > gf) == nd_bull:
            d += 1
            fac.append(f"Spot {'above' if spot_px > gf else 'below'} gamma flip ({int(gf)}) — regime aligned")
    d = min(d, 3)

    # 3. Momentum Score (0-4): OI flow direction + PCR + IV rank ─────────────
    ms       = 0
    mom_bull = mom > 0
    if not mom_sig:
        ms = 1   # flat/negligible flow — neutral
    elif nd_sig and mom_bull == nd_bull:
        ms = 3
        fac.append(f"OI momentum confirms {'bullish' if mom_bull else 'bearish'} flow ({mom:+,.0f})")
    elif nd_sig and mom_bull != nd_bull:
        ms = 0
        fac.append(f"⚠ Momentum contradicts net delta — divergence, cut size")
    else:
        ms = 2
    if nd_sig:
        if nd_bull and pcr >= 1.2:
            ms = min(ms + 1, 4); fac.append(f"PCR {pcr:.2f} confirms bullish support")
        elif not nd_bull and pcr <= 0.8:
            ms = min(ms + 1, 4); fac.append(f"PCR {pcr:.2f} confirms bearish pressure")
    if   iv_r <= 35: ms = min(ms + 1, 4)   # calm IV = ideal sell environment
    elif iv_r >= 70: ms = max(ms - 1, 0)   # high IV = elevated risk
    ms = min(ms, 4)

    total = g + d + ms

    # BUG 2 FIX — exhaustive label logic so momentum-only signals surface correctly.
    # Original had mom<=0 / mom>=0 in the mixed branches, swallowing the flat-momentum
    # and momentum-only cases into a silent "NEUTRAL" that hid real directional flow.
    if   nd > 0 and mom > 0:   bias_s = "BULLISH"
    elif nd < 0 and mom < 0:   bias_s = "BEARISH"
    elif nd > 0 and mom < 0:   bias_s = "MIXED — delta bull / momentum fading"
    elif nd < 0 and mom > 0:   bias_s = "MIXED — delta bear / momentum recovering"
    elif nd > 0:               bias_s = "BULLISH (flat momentum)"
    elif nd < 0:               bias_s = "BEARISH (flat momentum)"
    elif mom > 0:              bias_s = "NEUTRAL — flow tilting bullish"
    elif mom < 0:              bias_s = "NEUTRAL — flow tilting bearish"
    else:                      bias_s = "NEUTRAL"

    # Conviction label + recommendation
    if   total >= 8: conv, cc, sl, rtxt = "HIGH CONVICTION", "#059669", "Full size",  "All Greeks aligned. Deploy full planned size within the gamma range."
    elif total >= 6: conv, cc, sl, rtxt = "GOOD SETUP",      "#10B981", "Standard",   "Most signals confirm. Trade standard size; monitor the weakest Greek."
    elif total >= 4: conv, cc, sl, rtxt = "MODERATE",        "#D97706", "Half size",  "Mixed signals. Half size only, or wait 30–60 min for clarity."
    elif total >= 2: conv, cc, sl, rtxt = "LOW",             "#F59E0B", "Avoid",      "Greeks not aligned. Watch only — do not deploy capital now."
    else:            conv, cc, sl, rtxt = "NO TRADE",        "#DC2626", "Stay out",   "Conflicting signals. Protect capital and wait for a cleaner setup."

    iv_env = "Low IV — ideal" if iv_r <= 35 else ("High IV — caution" if iv_r >= 70 else "Mid IV — ok")
    return dict(
        total=total, g=g, d=d, ms=ms,
        bias_s=bias_s, conv=conv, cc=cc, sl=sl, rtxt=rtxt,
        fac=fac[:4],
        gamma_range=f"{int(sup_w)}–{int(res_w)}" if sup_w and res_w else "—",
        iv_env=iv_env, iv_r=iv_r,
    )

_grf        = _compute_grf(m, spot)
_grf_dc     = GREEN if _grf["bias_s"] == "BULLISH" else (RED if _grf["bias_s"] == "BEARISH" else AMBER)
_grf_fac_html = "".join(
    f'<div style="font-size:11px;color:#374151;padding:2px 0;line-height:1.4;">&#9656; {f}</div>'
    for f in _grf["fac"]
) or '<div style="font-size:11px;color:#9CA3AF;">Collecting signals…</div>'

def _gbar(v, mx, clr):
    pct = int(v / mx * 100)
    return (f'<div style="background:#F3F4F6;border-radius:4px;height:7px;margin-top:4px;">'
            f'<div style="width:{pct}%;background:{clr};height:7px;border-radius:4px;"></div></div>')

# ═════════════════════════════════════════════════════════════════════════════
# ⚡ ENHANCED BIAS PANEL — TOP OF DASHBOARD  (v7 addition)
# Combines existing S3/4 options flow with three new live layers:
#   VWAP + Opening Range · Term Structure (front/back IV) · India VIX
# Renders BEFORE all other sections. No existing code touched below this block.
# ═════════════════════════════════════════════════════════════════════════════
def _render_enhanced_bias_panel(eb, vwap_or, ts, vix, cd, spot_px, metrics):
    """
    Top-of-dashboard panel surfacing the four-layer Enhanced Price Bias.
    All arguments are pre-computed above; this function is purely presentational.
    """
    esc   = eb["enhanced_score"]
    ecol  = eb["color"]
    edir  = eb["direction"]
    econf = eb["enhanced_conf"]
    eagr  = eb["agreement_pct"]
    nsig  = eb["new_signals_available"]

    # ── Helper: small info chip ───────────────────────────────────────────────
    def _chip(label, value, color, bg=None):
        bg = bg or f"{color}18"
        return (f'<span style="background:{bg};color:{color};border:1px solid {color};'
                f'border-radius:5px;padding:2px 9px;font-size:11px;font-weight:700;'
                f'white-space:nowrap;">{label}: {value}</span>')

    # ── Row 1: main badge + score bar ─────────────────────────────────────────
    bar_pct = int(abs(esc))
    bar_color = ecol
    score_bar = (
        f'<div style="background:#F3F4F6;border-radius:4px;height:8px;margin:6px 0 4px 0;">'
        f'<div style="width:{bar_pct}%;background:{bar_color};height:8px;border-radius:4px;'
        f'transition:width 0.4s;"></div></div>'
    )

    # ── Row 2: four signal chips ──────────────────────────────────────────────
    s34_col  = "#059669" if eb["s34_score"] > 10 else ("#DC2626" if eb["s34_score"] < -10 else "#6B7280")
    chips_html = " ".join([
        _chip("S3/4", f"{eb['s34_score']:+.0f}", s34_col),
        _chip("VWAP/OR",
              f"{eb['price_score']:+.0f}" if vwap_or else "—",
              vwap_or["price_color"] if vwap_or else "#6B7280"),
        _chip("Term Struct",
              ts["regime"].replace("_"," ") if ts and ts["available"] else "—",
              ts["ts_color"] if ts and ts["available"] else "#6B7280"),
        _chip("VIX",
              f"{vix['vix']:.1f}" if vix and vix["available"] else "—",
              vix["vix_color"] if vix and vix["available"] else "#6B7280"),
    ])

    # ── Row 3: detail lines for each new signal ───────────────────────────────
    detail_lines = []
    if vwap_or and vwap_or.get("n_candles", 0) > 5:
        detail_lines.append(
            f'<div style="font-size:11px;color:#374151;padding:2px 0;">&#9642; '
            f'<strong>VWAP</strong> {vwap_or["vwap"]:,.1f} &nbsp;·&nbsp; '
            f'OR {vwap_or["or_low"]:,.0f}–{vwap_or["or_high"]:,.0f} &nbsp;·&nbsp; '
            f'<span style="color:{vwap_or["price_color"]};font-weight:700;">{vwap_or["price_label"]}</span>'
            f'</div>'
        )
    elif not USE_DHAN:
        detail_lines.append(
            '<div style="font-size:11px;color:#9CA3AF;padding:2px 0;">'
            '&#9642; VWAP/OR: unavailable in demo mode</div>'
        )
    else:
        detail_lines.append(
            '<div style="font-size:11px;color:#9CA3AF;padding:2px 0;">'
            '&#9642; VWAP/OR: building (need 5+ candles — session starting)</div>'
        )

    if ts and ts["available"]:
        detail_lines.append(
            f'<div style="font-size:11px;color:#374151;padding:2px 0;">&#9642; '
            f'<strong>Term Structure</strong> — <span style="color:{ts["ts_color"]};font-weight:700;">'
            f'{ts["ts_label"]}</span></div>'
        )
    else:
        detail_lines.append(
            '<div style="font-size:11px;color:#9CA3AF;padding:2px 0;">'
            '&#9642; Term Structure: single expiry only (back-month data unavailable)</div>'
        )

    if vix and vix["available"]:
        chg_str = (f' &nbsp;·&nbsp; Δ {vix["vix_change"]:+.2f} pts this tick'
                   if vix["vix_change"] is not None else "")
        detail_lines.append(
            f'<div style="font-size:11px;color:#374151;padding:2px 0;">&#9642; '
            f'<strong>India VIX</strong> — <span style="color:{vix["vix_color"]};font-weight:700;">'
            f'{vix["vix_label"]}</span>{chg_str}</div>'
        )
    else:
        detail_lines.append(
            '<div style="font-size:11px;color:#9CA3AF;padding:2px 0;">'
            '&#9642; India VIX: not available via Dhan API on this account</div>'
        )

    details_html = "\n".join(detail_lines)

    # ── Agreement indicator ───────────────────────────────────────────────────
    agr_color = "#059669" if eagr >= 75 else ("#F59E0B" if eagr >= 50 else "#DC2626")
    agr_label = "High agreement" if eagr >= 75 else ("Partial agreement" if eagr >= 50 else "Mixed signals")

    # ── Confidence bar ────────────────────────────────────────────────────────
    conf_bar = (
        f'<div style="background:#F3F4F6;border-radius:4px;height:5px;margin-top:4px;">'
        f'<div style="width:{int(econf)}%;background:{ecol};height:5px;border-radius:4px;"></div></div>'
    )

    st.markdown(
        '<div class="section-header">⚡ Enhanced Market Bias &mdash; '
        'Options Flow + Price Confirmation + Term Structure + VIX</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
<div style="
    background:#fff;
    border:2px solid {ecol};
    border-radius:12px;
    padding:14px 20px 12px 24px;
    margin-bottom:14px;
    position:relative;
    box-shadow:0 2px 8px rgba(0,0,0,0.07);
">
  <!-- left accent bar -->
  <div style="position:absolute;left:0;top:0;bottom:0;width:6px;
       background:{ecol};border-radius:12px 0 0 12px;"></div>

  <!-- Row 1: Direction badge + score + confidence chips -->
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px;">
    <span style="font-size:16px;font-weight:900;color:#1A1A2E;">⚡ Enhanced Bias</span>
    <span style="background:{ecol};color:#fff;border-radius:6px;
          padding:4px 14px;font-size:14px;font-weight:800;letter-spacing:0.5px;">
      {edir}
    </span>
    <span style="background:{ecol}22;color:{ecol};border:1px solid {ecol};
          border-radius:6px;padding:2px 10px;font-size:13px;font-weight:800;">
      {esc:+.0f} / 100
    </span>
    <span style="background:{agr_color}22;color:{agr_color};border:1px solid {agr_color};
          border-radius:6px;padding:2px 9px;font-size:11px;font-weight:700;">
      {agr_label} ({int(eagr)}%)
    </span>
    <span style="margin-left:auto;font-size:10px;color:#9CA3AF;">
      {nsig}/3 new signals live
    </span>
  </div>

  <!-- Score bar -->
  {score_bar}

  <!-- Row 2: signal chips -->
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 10px 0;">
    {chips_html}
  </div>

  <!-- Row 3: confidence sub-bar -->
  <div style="font-size:10px;color:#9CA3AF;margin-bottom:2px;">
    Composite confidence: {econf:.0f}%
  </div>
  {conf_bar}

  <!-- Row 4: detail lines -->
  <div style="margin-top:10px;padding-top:10px;border-top:1px solid #F3F4F6;">
    {details_html}
  </div>

  <!-- Footer note -->
  <div style="font-size:10px;color:#9CA3AF;margin-top:8px;">
    Weight: S3/4 flow 70% · VWAP+OR 10% · Term structure 10% · India VIX 10%
    &nbsp;·&nbsp; v7 enhanced layer · existing engines unchanged
  </div>
</div>
""", unsafe_allow_html=True)

# ── Render the Enhanced Bias Panel at the top of the dashboard ───────────────
_render_enhanced_bias_panel(
    _enhanced_bias, _vwap_or_data, _ts_data, _vix_data, _combined_decision, spot, m
)
# ══ END ENHANCED BIAS PANEL ═══════════════════════════════════════════════════

st.markdown(
    '<div class="section-header">&#128300; Greek Risk Framework &mdash; Hedge-Flow Pressure &amp; Confidence</div>',
    unsafe_allow_html=True)
st.markdown(f"""
<div style="background:#fff;border:1.5px solid {_grf['cc']};border-radius:10px;
     padding:14px 18px 12px 22px;margin-bottom:14px;position:relative;">
  <div style="position:absolute;left:0;top:0;bottom:0;width:5px;
       background:{_grf['cc']};border-radius:10px 0 0 10px;"></div>

  <!-- Row 1: bias badge + conviction + range/size -->
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px;">
    <span style="background:{_grf_dc}22;color:{_grf_dc};border:1px solid {_grf_dc};
          border-radius:6px;padding:3px 12px;font-size:13px;font-weight:800;">
      {_grf['bias_s']}
    </span>
    <span style="background:{_grf['cc']}22;color:{_grf['cc']};border:1px solid {_grf['cc']};
          border-radius:6px;padding:2px 10px;font-size:12px;font-weight:700;">
      {_grf['conv']} &nbsp;·&nbsp; {_grf['total']}/10
    </span>
    <span style="font-size:11px;color:#6B7280;margin-left:auto;">
      Gamma range: <strong>{_grf['gamma_range']}</strong>
      &nbsp;·&nbsp; Position size: <strong style="color:{_grf['cc']};">{_grf['sl']}</strong>
    </span>
  </div>

  <!-- Row 2: sub-score progress bars -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:10px;">
    <div>
      <div style="font-size:11px;font-weight:600;color:#6B7280;">
        Gamma · Range quality &nbsp;<strong style="color:#1A1A2E;">{_grf['g']}/3</strong>
      </div>
      {_gbar(_grf['g'], 3, '#5DCAA5')}
    </div>
    <div>
      <div style="font-size:11px;font-weight:600;color:#6B7280;">
        Delta · Equilibrium &nbsp;<strong style="color:#1A1A2E;">{_grf['d']}/3</strong>
      </div>
      {_gbar(_grf['d'], 3, '#378ADD')}
    </div>
    <div>
      <div style="font-size:11px;font-weight:600;color:#6B7280;">
        Momentum · Flow &nbsp;<strong style="color:#1A1A2E;">{_grf['ms']}/4</strong>
      </div>
      {_gbar(_grf['ms'], 4, '#7F77DD')}
    </div>
  </div>

  <!-- Row 3: key signals + recommendation -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;
       padding-top:10px;border-top:1px solid #F3F4F6;">
    <div>
      <div style="font-size:11px;font-weight:700;color:#374151;margin-bottom:4px;">Key signals</div>
      {_grf_fac_html}
    </div>
    <div style="background:{_grf['cc']}22;border-radius:8px;padding:10px 12px;">
      <div style="font-size:11px;font-weight:700;color:{_grf['cc']};margin-bottom:4px;">
        Recommendation
      </div>
      <div style="font-size:12px;color:#374151;line-height:1.55;">{_grf['rtxt']}</div>
      <div style="font-size:10px;color:#9CA3AF;margin-top:6px;">
        {_grf['iv_env']} (IV rank {_grf['iv_r']:.0f}) &nbsp;·&nbsp;
        Sources: net delta · OI momentum · GEX · PCR · max pain
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
# ══ END Greek Risk Framework ══════════════════════════════════════════════════

# ── SECTION 3 & 4 MARKET BIAS vs NIFTY SPOT ─────────────────────────────────
_bh_data = st.session_state.get("bias_history", [])

# ── Current snapshot badge (always visible, even before 2 data points) ───────
_s34_dir   = _s34_bias["direction"]
_s34_bc    = "#22C55E" if _s34_score > 15 else ("#EF4444" if _s34_score < -15 else "#F59E0B")
_s34_bd    = _s34_breakdown
_s34_parts = " | ".join(f"{k}: {v:+.0f}" for k, v in _s34_bd.items())
st.markdown(
    f"""<div style="background:#fff;border:1.5px solid {_s34_bc};border-radius:10px;
         padding:10px 16px;margin-bottom:10px;display:flex;align-items:center;
         gap:12px;flex-wrap:wrap;">
      <span style="font-size:13px;font-weight:800;color:#1A1A2E;">
        &#x1F4CA; Section 3&4 Bias</span>
      <span style="background:{_s34_bc}22;color:{_s34_bc};border:1px solid {_s34_bc};
            border-radius:6px;padding:2px 10px;font-size:13px;font-weight:700;">
        {_s34_dir} &nbsp; {_s34_score:+.0f}/100</span>
      <span style="font-size:11px;color:#6B7280;">{_s34_parts}</span>
      <span style="font-size:11px;color:#9CA3AF;margin-left:auto;">
        5-min chart updates every 5 min · {len(_bh_data)} pts</span>
    </div>""",
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# 🎯 COMBINED MARKET BIAS DECISION  (Chapter 17 & 18)
# Surgical addition — do not modify any code below this block
# ═════════════════════════════════════════════════════════════════════════════
def _render_combined_bias_panel(cd: dict) -> None:
    """
    Renders the top-of-dashboard Combined Bias Decision panel.
    cd = output of generate_combined_decision()
    """
    q        = cd["quadrant"]
    qcolor   = cd["quadrant_color"]
    qbg      = cd["badge_bg"]
    qshort   = cd["quadrant_short"]
    action   = cd["action"]
    conf_l   = cd["confidence_label"]
    conf_c   = cd["confidence_color"]
    lines    = cd["explanation_lines"]
    div      = cd["divergence"]
    s34_sc   = cd["s34_score"]
    s34_dir  = cd["s34_direction"]
    smile_sc = cd["smile_scenario"]
    pcr_val  = cd["pcr"]

    # Confidence chip colour variant
    conf_alpha = "33"   # semi-transparent background

    # Divergence section (only if active)
    div_html = ""
    if div:
        _div_strength = div.get("strength", "HARD")
        _div_icon = "⚠" if _div_strength == "HARD" else "🔮"
        _div_border_style = "solid" if _div_strength == "HARD" else "dashed"
        _div_opacity = "1.0" if _div_strength == "HARD" else "0.75"
        div_html = (
            f'<div style="background:{div["badge_bg"]};border:1.5px { _div_border_style} {div["color"]};'
            f'border-radius:8px;padding:8px 14px;margin-top:10px;opacity:{_div_opacity};">'
            f'<span style="font-size:12px;font-weight:800;color:{div["color"]};">'
            f'{_div_icon} {_div_strength}: {div["type"]}</span>'
            f'<div style="font-size:11.5px;color:#374151;margin-top:4px;">{div["warning"]}</div>'
            f'<div style="font-size:11px;color:#6B7280;margin-top:3px;">{div["detail"]}</div>'
            f'</div>'
        )

    # Explanation lines HTML
    lines_html = "".join(
        f'<div style="font-size:11.5px;color:#374151;padding:2px 0;line-height:1.5;">'
        f'&#9656; {ln}</div>'
        for ln in lines
    )

    # Colour bar strip at left edge (mimics the manual's colour-coded quadrant strips)
    colour_bar = f"""<div style="
        position:absolute;left:0;top:0;bottom:0;width:5px;
        background:{qcolor};border-radius:10px 0 0 10px;
    "></div>"""

    # Python ≤3.11 fix: backslashes inside f-string {} expressions are a SyntaxError
    # before PEP 701 (3.12). Extract conditional HTML snippets into plain variables first.
    _override_badge = (
        "<span style='background:#FEF3C7;color:#B45309;"
        "border:1px dashed #B45309;border-radius:6px;"
        "padding:2px 9px;font-size:10px;font-weight:700;'>"
        "OVERRIDDEN by Price Layer</span>"
        if cd.get("quadrant_overridden") else ""
    )
    _enhanced_score_html = (
        "<span>Enhanced Score: <strong style='color:#1A1A2E;'>{:+.0f}</strong></span>".format(
            cd.get("enhanced_score", 0)
        ) if cd.get("enhanced_score", 0) != 0 else ""
    )
    st.markdown(f"""
<div style="
    background:{qbg};
    border:1.5px solid {qcolor};
    border-radius:10px;
    padding:12px 18px 12px 22px;
    margin-bottom:12px;
    position:relative;
">
  {colour_bar}
  <!-- Row 1: title + badges -->
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px;">
    <span style="font-size:15px;font-weight:900;color:#1A1A2E;">
      🎯 Combined Bias Decision
    </span>
    <span style="
        background:{qcolor};color:#fff;
        border-radius:6px;padding:3px 12px;
        font-size:13px;font-weight:800;letter-spacing:0.5px;
    ">{qshort}</span>
    <span style="
        background:{conf_c}{conf_alpha};color:{conf_c};
        border:1px solid {conf_c};
        border-radius:6px;padding:2px 9px;
        font-size:11px;font-weight:700;
    ">Confidence: {conf_l}</span>
    {_override_badge}
  </div>
  <!-- Row 2: action line -->
  <div style="
      font-size:12.5px;font-weight:700;color:{qcolor};
      margin-bottom:8px;letter-spacing:0.2px;
  ">{action}</div>
  <!-- Row 3: explanation lines -->
  {lines_html}
  <!-- Row 4: mini metrics strip -->
  <div style="
      display:flex;gap:18px;flex-wrap:wrap;
      margin-top:8px;padding-top:8px;
      border-top:1px solid #E5E7EB;
      font-size:11px;color:#6B7280;
  ">
    <span>S3/4 Score: <strong style="color:{qcolor};">{s34_sc:+.0f}</strong> ({s34_dir})</span>
    <span>IV Smile: <strong style="color:#374151;">{smile_sc}</strong></span>
    <span>PCR: <strong style="color:#374151;">{pcr_val:.2f}</strong></span>
    {_enhanced_score_html}
    <span style="margin-left:auto;font-size:10px;color:#9CA3AF;">
      Chapters 17 &amp; 18 · Combined Bias Engine (v4 inline)
    </span>
  </div>
  {div_html}
</div>
""", unsafe_allow_html=True)

# ── Call the panel ────────────────────────────────────────────────────────────
_render_combined_bias_panel(_combined_decision)
# ═════════════════════════════════════════════════════════════════════════════
# END COMBINED MARKET BIAS DECISION PANEL
# ═════════════════════════════════════════════════════════════════════════════

if len(_bh_data) >= 2:
    import plotly.graph_objs as _go2
    _B_GREEN = "#22C55E"; _B_RED = "#EF4444"; _B_CYAN = "#22D3EE"
    _B_ZONE  = "#F3F4F6"  # neutral zone fill

    _bh_ts    = [r["ts"]    for r in _bh_data]
    _bh_spot  = [r["spot"]  for r in _bh_data]
    _bh_score = [r["score"] for r in _bh_data]

    # Build custom hover for each point (signal breakdown if available)
    _bh_hover = []
    for r in _bh_data:
        _hlines = [
            f"<b>{r['ts']}</b>",
            f"Bias: <b>{r['score']:+.0f}</b> ({r.get('direction','—')})",
            f"Spot: {r['spot']:,.0f}",
        ]
        for _sk in ("s1","s2","s3","s4","s5","s6"):
            if _sk in r:
                _label = {"s1":"S1 Net OI","s2":"S2 Mom","s3":"S3 Levels",
                          "s4":"S4 Skew","s5":"S5 TS","s6":"S6 Vel"}.get(_sk, _sk)
                _hlines.append(f"{_label}: {r[_sk]:+.0f}")
        _bh_hover.append("<br>".join(_hlines))

    _bf = _go2.Figure()

    # ── Zone bands ────────────────────────────────────────────────────────────
    # Strong bullish zone (40–100): light green tint
    _bf.add_hrect(y0=40,  y1=100, fillcolor="#DCFCE7", opacity=0.25, layer="below", line_width=0)
    # Neutral zone (±15): grey
    _bf.add_hrect(y0=-15, y1=15,  fillcolor="#E5E7EB", opacity=0.35, layer="below", line_width=0)
    # Strong bearish zone (-40 to -100): light red tint
    _bf.add_hrect(y0=-100, y1=-40, fillcolor="#FEE2E2", opacity=0.25, layer="below", line_width=0)

    # ── Reference lines ───────────────────────────────────────────────────────
    for _rl, _rd, _rc in [
        (0,   "dot",   "#9CA3AF"),   # zero line
        (40,  "dash",  "#86EFAC"),   # bull conviction
        (-40, "dash",  "#FCA5A5"),   # bear conviction
        (15,  "dot",   "#D1D5DB"),   # neutral upper edge
        (-15, "dot",   "#D1D5DB"),   # neutral lower edge
    ]:
        _bf.add_hline(y=_rl, line_width=1.2, line_dash=_rd, line_color=_rc)

    # ── Nifty Spot on right axis (cyan) ───────────────────────────────────────
    _bf.add_trace(_go2.Scatter(
        x=_bh_ts, y=_bh_spot, name="Nifty Spot",
        mode="lines+markers",
        line=dict(color=_B_CYAN, width=2.2),
        marker=dict(size=4, color=_B_CYAN),
        yaxis="y2",
        hovertemplate="%{x}<br>Spot: %{y:,.0f}<extra>Spot</extra>",
    ))

    # ── Bias line: colour-coded segments (green above 0, red below) ───────────
    for _bi in range(len(_bh_score)):
        _bc = _B_GREEN if _bh_score[_bi] >= 0 else _B_RED
        if _bi < len(_bh_score) - 1:
            # Colour each segment by the current point's sign
            _bf.add_trace(_go2.Scatter(
                x=[_bh_ts[_bi], _bh_ts[_bi + 1]],
                y=[_bh_score[_bi], _bh_score[_bi + 1]],
                mode="lines",
                line=dict(color=_bc, width=2.8),
                showlegend=False,
                yaxis="y1",
                hoverinfo="skip",
            ))
        # Marker with full breakdown tooltip
        _bf.add_trace(_go2.Scatter(
            x=[_bh_ts[_bi]], y=[_bh_score[_bi]],
            mode="markers",
            marker=dict(
                color=_bc, size=8,
                line=dict(color="#fff", width=1.5),
                symbol="circle",
            ),
            name="S3/4 Bias" if _bi == 0 else None,
            showlegend=(_bi == 0),
            customdata=[_bh_hover[_bi]],
            hovertemplate="%{customdata}<extra></extra>",
            yaxis="y1",
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    _spot_vals  = [v for v in _bh_spot if v > 0]
    _spot_pad   = (max(_spot_vals) - min(_spot_vals)) * 0.15 if len(_spot_vals) > 1 else 50
    _spot_range = [min(_spot_vals) - _spot_pad, max(_spot_vals) + _spot_pad] if _spot_vals else None

    _bf.update_layout(
        title=dict(
            text=(
                f"S3/4 Market Bias vs Nifty Spot — "
                f"<span style='color:{_s34_bc}'>{_s34_dir} {_s34_score:+.0f}</span>"
                f"  <span style='font-size:11px;color:#9CA3AF'>· 5-min snapshots</span>"
            ),
            font=dict(size=13, color="#1A1A2E"),
        ),
        height=270,
        paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
        margin=dict(l=52, r=66, t=44, b=30),
        font=dict(color="#1A1A2E", size=11),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)),
        hovermode="closest",
        yaxis=dict(
            title=dict(text="Bias Score", font=dict(size=10, color="#1A1A2E")),
            range=[-105, 105],
            zeroline=False,
            gridcolor="#F3F4F6",
            tickvals=[-100, -60, -40, -15, 0, 15, 40, 60, 100],
            tickfont=dict(size=9),
        ),
        yaxis2=dict(
            title=dict(text="Nifty Spot", font=dict(size=10, color=_B_CYAN)),
            overlaying="y", side="right",
            showgrid=False, zeroline=False,
            range=_spot_range,
            tickfont=dict(size=9, color=_B_CYAN),
        ),
    )
    st.plotly_chart(_bf, width='stretch', config={"displayModeBar": False})  # H23 fix: was use_container_width=True
elif len(_bh_data) == 1:
    st.caption("⏳ Chart will appear after the second 5-minute snapshot is recorded.")

# ═══════════════════════════════════════════════════════════════════════════════
# GAMMA DATA SECTION — Option C: OI×Gamma Directional Balance
#                    — Sub-D:   Gamma Blast Proximity Detector
# All data sourced from the already-loaded option chain (_early_df_band).
# No new API calls. Gamma is computed during the main chain fetch (BS Greeks).
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header">⚡ Gamma Data — OI×Gamma Balance · Blast Proximity Detector</div>',
    unsafe_allow_html=True,
)

_gd_src = _early_df_band.copy() if _early_df_band is not None and not _early_df_band.empty else None

if _gd_src is not None:
    # ── Coerce numeric columns ────────────────────────────────────────────────
    for _gc in ["strike", "call_oi", "put_oi", "call_gamma", "put_gamma"]:
        if _gc in _gd_src.columns:
            _gd_src[_gc] = pd.to_numeric(_gd_src[_gc], errors="coerce").fillna(0.0)

    _gd_src = _gd_src.sort_values("strike").reset_index(drop=True)

    # ── Core Option C columns ────────────────────────────────────────────────
    # Standard GEX (industry formula): OI × Gamma × LotSize × Spot² × 0.01
    # Matches Perfiliev / SpotGamma / StockMojo convention.
    # Calls → +ve GEX (dealers buy spot to hedge → dampening/pinning force)
    # Puts  → subtracted (dealers sell spot → amplifying force)
    _spot2 = spot ** 2
    _gd_src["call_gex"] = _gd_src["call_oi"] * _gd_src["call_gamma"] * NIFTY_LOT_SIZE * _spot2 * 0.01
    _gd_src["put_gex"]  = _gd_src["put_oi"]  * _gd_src["put_gamma"]  * NIFTY_LOT_SIZE * _spot2 * 0.01
    _gd_src["net_gex"]  = _gd_src["call_gex"] - _gd_src["put_gex"]   # +ve = net long gamma (pinning), -ve = net short gamma (trending)

    # Chart-level gamma flip: cumsum of unweighted net_gex → zero-crossing.
    # This matches the bars shown on the chart (same formula), so the flip
    # annotation will visually align with where the purple line crosses zero.
    _chart_cumgex   = _gd_src.sort_values("strike")["net_gex"].cumsum().values
    _chart_strikes  = _gd_src.sort_values("strike")["strike"].values
    _chart_flip_cands = _chart_strikes[_chart_cumgex <= 0]
    _chart_gamma_flip = float(_chart_flip_cands[-1]) if len(_chart_flip_cands) > 0 else None

    _gd_atm_band = spot * 0.003

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 1 — Option C: Standard GEX per Strike
    # Formula: GEX = OI × Gamma × Spot² × 0.01  (notional-scaled, matches compute_true_gex)
    # Red bars   = Call GEX (dealers long gamma → buy dips/sell rallies → PINNING)
    # Green bars = Put GEX shown as negative (dealers short gamma → amplify moves → TRENDING)
    # Purple line = Net GEX: +ve = long-gamma/pinning regime, -ve = short-gamma/trending
    # Gamma Flip level = zero-crossing of cumulative Net GEX (computed in compute_true_gex).
    # ─────────────────────────────────────────────────────────────────────────
    # ── Net Vega per Strike ───────────────────────────────────────────────────
    # Net Vega = (Call OI × Call Vega) - (Put OI × Put Vega)
    # +ve = net long vega at that strike (buyers dominate → IV expansion expected)
    # -ve = net short vega (sellers dominate → IV suppressed / gravity well)
    for _vc in ["call_vega", "put_vega"]:
        if _vc in _gd_src.columns:
            _gd_src[_vc] = pd.to_numeric(_gd_src[_vc], errors="coerce").fillna(0.0)
    _gd_src["call_vega_exp"] = _gd_src["call_oi"] * _gd_src["call_vega"]   # call vega exposure
    _gd_src["put_vega_exp"]  = _gd_src["put_oi"]  * _gd_src["put_vega"]    # put vega exposure
    _gd_src["net_vega"]      = _gd_src["call_vega_exp"] - _gd_src["put_vega_exp"]  # +ve = net long vega

    # ── 2-column layout: GEX chart | Net Vega chart ──────────────────────────
    _gc_col1, _gc_col2 = st.columns(2)

    with _gc_col1:
        _gc1_fig = go.Figure()
        _gc1_fig.add_trace(go.Bar(
            x=_gd_src["strike"],
            y=_gd_src["call_gex"],
            name="Call GEX (Dealer Buy — Pinning)",
            marker_color="#EF4444",
            opacity=0.75,
            hovertemplate="Strike %{x:,.0f}<br>Call GEX: %{y:,.2f}<extra>Dealer Buy / Pinning</extra>",
        ))
        _gc1_fig.add_trace(go.Bar(
            x=_gd_src["strike"],
            y=-_gd_src["put_gex"],
            name="Put GEX (Dealer Sell — Amplifying)",
            marker_color="#22C55E",
            opacity=0.75,
            hovertemplate="Strike %{x:,.0f}<br>Put GEX: %{y:,.2f}<extra>Dealer Sell / Amplifying</extra>",
        ))
        _gc1_fig.add_trace(go.Scatter(
            x=_gd_src["strike"],
            y=_gd_src["net_gex"],
            name="Net GEX",
            mode="lines+markers",
            line=dict(color="#7C3AED", width=2.2),
            marker=dict(size=5, color="#7C3AED"),
            hovertemplate="Strike %{x:,.0f}<br>Net GEX: %{y:,.2f}<extra>Net GEX</extra>",
        ))
        _gc1_fig.add_vline(
            x=spot, line_dash="dash", line_color="#F59E0B", line_width=2,
            annotation_text=f"Spot {spot:,.0f}",
            annotation_font=dict(size=10, color="#F59E0B"),
            annotation_position="top right",
        )
        # Chart-level gamma flip — computed from the same unweighted bars shown here,
        # so the annotation always aligns with where the purple Net GEX line crosses zero.
        if _chart_gamma_flip is not None:
            _gc1_fig.add_vline(
                x=_chart_gamma_flip, line_dash="dot", line_color="#10B981", line_width=1.8,
                annotation_text=f"Flip {int(_chart_gamma_flip):,}",
                annotation_font=dict(size=9, color="#10B981"),
                annotation_position="top left",
            )
        _gc1_fig.update_layout(
            title=dict(
                text="Option C — Standard GEX per Strike  "
                     "<span style='font-size:11px;color:#6B7280'>"
                     "Red=Call GEX (Pinning) · Green=Put GEX (Amplifying) · Purple=Net GEX · "
                     "Green dot=Gamma Flip</span>",
                font=dict(size=13),
            ),
            barmode="overlay",
            height=310,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
            margin=dict(l=55, r=20, t=50, b=30),
            legend=dict(orientation="h", y=1.18, font=dict(size=10)),
            yaxis=dict(
                title="GEX  (OI × Γ × LotSize × Spot² × 0.01)",
                gridcolor="#F3F4F6",
                zeroline=True, zerolinecolor="#9CA3AF", zerolinewidth=1.2,
                tickfont=dict(size=9),
            ),
            xaxis=dict(title="Strike", tickfont=dict(size=9)),
            font=dict(color="#1A1A2E", size=11),
        )
        st.plotly_chart(_gc1_fig, use_container_width=True, config={"displayModeBar": False})

    with _gc_col2:
        # ─────────────────────────────────────────────────────────────────────
        # Net Vega per Strike chart
        # Call bars = Call OI × Call Vega  (call-side vega exposure)
        # Put bars  = Put OI × Put Vega shown negative (put-side vega exposure)
        # OI includes both buyers and sellers, so these bars show NET notional
        # vega exposure per side — not strictly "long/short" positional data.
        # Orange line = Net Vega: +ve = call-side dominant, -ve = put-side dominant
        # Gravity wells (large -ve net vega) = IV suppression / ceiling zones
        # Large +ve net vega strikes = IV expansion / breakout kindling zones
        # ─────────────────────────────────────────────────────────────────────
        _gv_fig = go.Figure()
        _gv_fig.add_trace(go.Bar(
            x=_gd_src["strike"],
            y=_gd_src["call_vega_exp"],
            name="Call Vega Exposure (ΣOI×Vega)",
            marker_color="#2563EB",
            opacity=0.70,
            hovertemplate="Strike %{x:,.0f}<br>Call Vega Exp: %{y:,.2f}<extra>Call-side Vega Exposure</extra>",
        ))
        _gv_fig.add_trace(go.Bar(
            x=_gd_src["strike"],
            y=-_gd_src["put_vega_exp"],
            name="Put Vega Exposure (ΣOI×Vega)",
            marker_color="#D97706",
            opacity=0.70,
            hovertemplate="Strike %{x:,.0f}<br>Put Vega Exp: %{y:,.2f}<extra>Put-side Vega Exposure</extra>",
        ))
        _gv_fig.add_trace(go.Scatter(
            x=_gd_src["strike"],
            y=_gd_src["net_vega"],
            name="Net Vega",
            mode="lines+markers",
            line=dict(color="#F97316", width=2.2),
            marker=dict(size=5, color="#F97316"),
            hovertemplate="Strike %{x:,.0f}<br>Net Vega: %{y:,.2f}<extra>Net Vega</extra>",
        ))
        _gv_fig.add_vline(
            x=spot, line_dash="dash", line_color="#F59E0B", line_width=2,
            annotation_text=f"Spot {spot:,.0f}",
            annotation_font=dict(size=10, color="#F59E0B"),
            annotation_position="top right",
        )
        _gv_fig.update_layout(
            title=dict(
                text="Net Vega Exposure per Strike  "
                     "<span style='font-size:11px;color:#6B7280'>"
                     "Blue=Call Vega Exp (ΣOI×Vega) · Amber=Put Vega Exp · Orange=Net · "
                     "+ve=Call-side dominant · −ve=Put-side dominant</span>",
                font=dict(size=13),
            ),
            barmode="overlay",
            height=310,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
            margin=dict(l=55, r=20, t=50, b=30),
            legend=dict(orientation="h", y=1.18, font=dict(size=10)),
            yaxis=dict(
                title="Vega Exposure  (ΣOI × Vega per Strike)",
                gridcolor="#F3F4F6",
                zeroline=True, zerolinecolor="#9CA3AF", zerolinewidth=1.2,
                tickfont=dict(size=9),
            ),
            xaxis=dict(title="Strike", tickfont=dict(size=9)),
            font=dict(color="#1A1A2E", size=11),
        )
        st.plotly_chart(_gv_fig, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────────
    # ATM BAND VEGA EXPOSURE DIFF vs SPOT — dual-axis time-series
    # X-axis : time ticks (today_history)
    # Left Y : Nifty Spot (amber line)
    # Right Y: ΣCall(OI×Vega) − ΣPut(OI×Vega) across ATM ± N strikes (purple)
    #          N is set by owner (default ±2 = ±100 pts)
    # +ve diff = call-side vega exposure > put-side = call buyers building near ATM
    # −ve diff = put-side dominant = downside hedge demand / protective buying
    # Zero-cross = vega exposure parity = transitional / balanced IV regime
    # Smoother than single-strike: ATM can drift ±N strikes before any jump occurs
    # ─────────────────────────────────────────────────────────────────────────
    # Infer the band width used from the first available history entry today
    _vd_band_n = 2   # default display label
    for _hh in today_history:
        if _hh.get("vega_band_strikes") is not None:
            _vd_band_n = int(_hh["vega_band_strikes"])
            break
    _vd_times, _vd_spot, _vd_vdiff, _vd_atm_k = [], [], [], []
    for _h in today_history:
        _cv = _h.get("atm_call_vega")
        _pv = _h.get("atm_put_vega")
        if _cv is not None and _pv is not None and _h.get("spot"):
            _vd_times.append(_h["ts"][11:19])        # HH:MM:SS from ISO timestamp
            _vd_spot.append(float(_h["spot"]))
            _vd_vdiff.append(round(float(_cv) - float(_pv), 6))
            _vd_atm_k.append(int(_h.get("atm", 0)))

    if len(_vd_times) >= 2:
        _vd_fig = go.Figure()
        # Left axis — Nifty Spot
        _vd_fig.add_trace(go.Scatter(
            x=_vd_times, y=_vd_spot,
            name="Nifty Spot",
            mode="lines",
            line=dict(color="#F59E0B", width=2.5),
            yaxis="y1",
            hovertemplate="%{x}<br>Spot: <b>%{y:,.0f}</b><extra>Spot</extra>",
        ))
        # Right axis — Band Vega Exposure Diff (Call − Put)
        _vd_fig.add_trace(go.Scatter(
            x=_vd_times, y=_vd_vdiff,
            name=f"Band Vega Diff (Call−Put, ±{_vd_band_n} strikes)",
            mode="lines+markers",
            line=dict(color="#7C3AED", width=2.0),
            marker=dict(size=4, color="#7C3AED"),
            yaxis="y2",
            hovertemplate="%{x}<br>Vega Diff: <b>%{y:,.2f}</b><extra>Band Vega Diff</extra>",
        ))
        # Zero line on right axis (vega parity)
        _vd_fig.add_hline(
            y=0, yref="y2",
            line_dash="dot", line_color="#C4B5FD", line_width=1.5,
            annotation_text="Vega Parity",
            annotation_font=dict(size=9, color="#7C3AED"),
            annotation_position="right",
        )
        # Mark ATM strike changes as vertical lines
        _prev_atm = None
        for _ti, _ak in zip(_vd_times, _vd_atm_k):
            if _ak and _ak != _prev_atm and _prev_atm is not None:
                _vd_fig.add_vline(
                    x=_ti, line_dash="dash", line_color="#6B7280",
                    line_width=1, opacity=0.5,
                    annotation_text=f"ATM→{_ak:,}",
                    annotation_font=dict(size=8, color="#6B7280"),
                    annotation_position="top left",
                )
            _prev_atm = _ak
        _vd_fig.update_layout(
            title=dict(
                text=f"ATM Band Vega Diff (±{_vd_band_n} strikes, ΣOI×Vega: Call−Put) vs Nifty Spot  "
                     "<span style='font-size:11px;color:#6B7280'>"
                     "Amber=Spot (left) · Purple=ΣCall Vega Exp − ΣPut Vega Exp (right) · "
                     "+ve=Call-side dominant · −ve=Put-side / hedge demand · "
                     "Grey dash=ATM strike change</span>",
                font=dict(size=13),
            ),
            height=250,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
            margin=dict(l=65, r=65, t=50, b=30),
            legend=dict(orientation="h", y=1.20, font=dict(size=10)),
            yaxis=dict(
                title="Nifty Spot",
                titlefont=dict(color="#F59E0B"),
                tickfont=dict(color="#F59E0B", size=9),
                gridcolor="#F3F4F6",
                autorange=True,
                showgrid=True,
            ),
            yaxis2=dict(
                title=f"Band Vega Diff  (±{_vd_band_n}×50 pts)",
                titlefont=dict(color="#7C3AED"),
                tickfont=dict(color="#7C3AED", size=9),
                overlaying="y", side="right",
                zeroline=True, zerolinecolor="#C4B5FD", zerolinewidth=1.2,
                autorange=True,
                showgrid=False,
            ),
            xaxis=dict(
                tickfont=dict(size=9),
                title="Time (IST)",
                showgrid=True, gridcolor="#F3F4F6",
            ),
            hovermode="x unified",
            font=dict(color="#1A1A2E", size=11),
        )
        st.plotly_chart(_vd_fig, use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("⏳ ATM Band Vega Diff chart — accumulating ticks (needs ≥2 data refreshes to plot)", icon="📊")

    # ═════════════════════════════════════════════════════════════════════════
    # LIVE GEX + VEGA INTERPRETATION ENGINE  (v3 — lot-size corrected GEX,
    # unweighted flip, magnitude-checked regime flags)
    # Matrix 1: High+veGEX+suppressed-vega / Low−veGEX+expanding-vega /
    #           FlipZone+VegaSpike / ATMmaxGEX+neutralVega
    # Matrix 2: Zone-level below/above spot breakdown + confluence + extended wall
    # "Vega" here means Net OI-weighted Vega Exposure (ΣOI×Vega), not raw unit vega.
    # Refreshes with every data tick — no extra API calls needed.
    # ═════════════════════════════════════════════════════════════════════════
    try:
        _atm_mask    = (_gd_src["strike"] - spot).abs() <= 3 * NIFTY_STEP
        _above_mask  = _gd_src["strike"] > spot
        _below_mask  = _gd_src["strike"] < spot

        # ── Key GEX levels ────────────────────────────────────────────────────
        _net_gex_total   = float(_gd_src["net_gex"].sum())
        _cw_idx          = _gd_src.loc[_above_mask, "call_gex"].idxmax() if _above_mask.any() else _gd_src["call_gex"].idxmax()
        _call_wall_k     = int(_gd_src.loc[_cw_idx, "strike"])
        _pw_idx          = _gd_src.loc[_below_mask, "put_gex"].idxmax() if _below_mask.any() else _gd_src["put_gex"].idxmax()
        _put_wall_k      = int(_gd_src.loc[_pw_idx, "strike"])
        _gamma_flip_now  = m.get("gamma_flip")
        _dist_to_call    = _call_wall_k - spot if _call_wall_k else 0
        _dist_to_put     = spot - _put_wall_k  if _put_wall_k  else 0
        _range_width     = _dist_to_call + _dist_to_put
        _flip_str        = f"{int(_gamma_flip_now):,}" if _gamma_flip_now else "N/A"

        # ── GAP 3: ATM-max GEX strike (for expiry magnet detection) ──────────
        _max_net_gex_idx = _gd_src["net_gex"].idxmax()
        _max_gex_strike  = int(_gd_src.loc[_max_net_gex_idx, "strike"])
        _atm_strike      = int(round(spot / NIFTY_STEP) * NIFTY_STEP)
        _max_gex_is_atm  = abs(_max_gex_strike - _atm_strike) <= NIFTY_STEP   # peak GEX at ATM?

        # ── Key Vega levels ───────────────────────────────────────────────────
        _net_vega_atm    = float(_gd_src.loc[_atm_mask, "net_vega"].sum())
        _net_vega_total  = float(_gd_src["net_vega"].sum())
        _vega_abs_max    = abs(_net_vega_total) if _net_vega_total != 0 else 1.0

        # GAP 2: Net vega specifically AT the gamma flip strike
        _flip_vega_atm   = 0.0
        if _gamma_flip_now is not None:
            _flip_mask   = (_gd_src["strike"] - _gamma_flip_now).abs() <= NIFTY_STEP
            _flip_vega_atm = float(_gd_src.loc[_flip_mask, "net_vega"].sum())
        _flip_vega_spiking = abs(_flip_vega_atm) > 0.15 * _vega_abs_max   # vega spike at flip strike

        _max_neg_vega_k  = int(_gd_src.loc[(_below_mask | _atm_mask), "net_vega"].idxmin()
                               if (_below_mask | _atm_mask).any() else _gd_src["net_vega"].idxmin())
        _max_neg_vega_k  = int(_gd_src.loc[_max_neg_vega_k, "strike"])
        _max_pos_vega_k  = int(_gd_src.loc[(_above_mask | _atm_mask), "net_vega"].idxmax()
                               if (_above_mask | _atm_mask).any() else _gd_src["net_vega"].idxmax())
        _max_pos_vega_k  = int(_gd_src.loc[_max_pos_vega_k, "strike"])

        # GAP 3: Net vega at ATM-max-GEX strike (for expiry magnet: should be near-zero)
        _atm_gex_vega_mask  = (_gd_src["strike"] - _max_gex_strike).abs() <= NIFTY_STEP
        _atm_max_gex_vega   = float(_gd_src.loc[_atm_gex_vega_mask, "net_vega"].sum())
        _vega_neutral_atm   = abs(_atm_max_gex_vega) < 0.08 * _vega_abs_max   # near-zero = neutral

        # ── GEX regime flags ─────────────────────────────────────────────────
        _is_pos_gex      = _net_gex_total > 0
        # Magnitude check: net GEX must exceed 30% of the largest single-strike GEX
        # magnitude to qualify as "high". Replaces the previous tautological check
        # (_net_gex_total > 0.3 * abs(_net_gex_total)) which was always True when positive.
        _gex_peak        = float(_gd_src["net_gex"].abs().max()) if not _gd_src.empty else 1.0
        _is_high_pos_gex = _is_pos_gex and (_net_gex_total > 0.3 * max(_gex_peak, 1.0))
        _above_flip      = (_gamma_flip_now is not None and spot > _gamma_flip_now)
        _near_flip       = (_gamma_flip_now is not None and abs(spot - _gamma_flip_now) <= 2 * NIFTY_STEP)

        # ── Vega regime flags ─────────────────────────────────────────────────
        _iv_suppressed   = _net_vega_atm < 0
        _iv_expanding    = _net_vega_atm > 0
        # GAP 1: magnitude — is long vega HIGH or just marginal?
        _high_long_vega  = _net_vega_atm > 0.25 * _vega_abs_max

        # ── GAP 4: Zone-level below/above spot GEX+Vega breakdown ────────────
        _below_net_gex   = float(_gd_src.loc[_below_mask, "net_gex"].sum())
        _below_net_vega  = float(_gd_src.loc[_below_mask, "net_vega"].sum())
        _above_net_gex   = float(_gd_src.loc[_above_mask, "net_gex"].sum())
        _above_net_vega  = float(_gd_src.loc[_above_mask, "net_vega"].sum())

        # Below-spot zone classification
        if _below_net_gex > 0 and _below_net_vega < 0:
            _below_zone_lbl  = "🟢 Soft floor — Put writers defending, IV suppressed below spot"
        elif _below_net_gex <= 0 and _below_net_vega < 0:
            _below_zone_lbl  = "🟡 Weak floor — Low GEX + IV sellers; floor can break quickly"
        elif _below_net_vega > 0:
            _below_zone_lbl  = "🔴 No floor — IV buyers below spot = market pricing downside move"
        else:
            _below_zone_lbl  = "⬜ Neutral below spot"

        # Above-spot zone classification
        if _above_net_gex > 0 and _above_net_vega > 0:
            _above_zone_lbl  = "🔴 Hard ceiling — Dealer sell wall + IV kindling; breakout triggers vol spike"
        elif _above_net_gex > 0 and _above_net_vega <= 0:
            _above_zone_lbl  = "🟡 Soft ceiling — GEX resistance but IV sellers above; muted breakout"
        elif _above_net_gex <= 0 and _above_net_vega > 0:
            _above_zone_lbl  = "🟠 No ceiling — Low GEX + IV buyers above = market pricing upside move"
        else:
            _above_zone_lbl  = "⬜ Neutral above spot"

        # GAP 4: Confluence check — do call_gex peak & call_vega_exp peak share the same strike?
        _cw_vega_idx     = _gd_src.loc[_above_mask, "call_vega_exp"].idxmax() if _above_mask.any() else None
        _call_vega_wall_k = int(_gd_src.loc[_cw_vega_idx, "strike"]) if _cw_vega_idx is not None else 0
        _confluence      = (_call_wall_k == _call_vega_wall_k)   # GEX peak == Vega peak at same strike

        # GAP 4: Secondary/extended call wall — 2nd highest call_gex above primary wall
        _ext_mask        = _gd_src["strike"] > _call_wall_k
        _ext_call_wall_k = 0
        _ext_call_gex    = 0.0
        if _ext_mask.any():
            _ext_cw_idx      = _gd_src.loc[_ext_mask, "call_gex"].idxmax()
            _ext_call_wall_k = int(_gd_src.loc[_ext_cw_idx, "strike"])
            _ext_call_gex    = float(_gd_src.loc[_ext_cw_idx, "call_gex"])
            _ext_vega_here   = float(_gd_src.loc[(_gd_src["strike"] == _ext_call_wall_k), "net_vega"].sum())
            _has_ext_wall    = _ext_call_gex > 0.25 * float(_gd_src.loc[_cw_idx, "call_gex"])
        else:
            _has_ext_wall    = False
            _ext_vega_here   = 0.0

        # ════════════════════════════════════════════════════════════════════
        # REGIME CLASSIFICATION (priority order: most specific first)
        # ════════════════════════════════════════════════════════════════════

        # GAP 3: Expiry Magnet — ATM has max GEX AND vega is neutral there
        if _max_gex_is_atm and _vega_neutral_atm and _is_pos_gex:
            _regime_tag   = "🧲 EXPIRY MAGNET  ·  CLASSIC PINNING"
            _regime_color = "#0891B2"
            _regime_bg    = "#ECFEFF"
            _regime_text  = (
                f"The <b>maximum GEX in the entire chain is at the ATM strike ({_max_gex_strike:,})</b> — "
                f"exactly where spot ({spot:,.0f}) is trading — and Net Vega there is near-zero (no strong IV bias). "
                f"This is the textbook <b>expiry magnet pattern</b>: dealers have maximum hedging concentration at ATM, "
                f"creating a gravitational pull that keeps spot pinned. "
                f"Expect the market to oscillate tightly around <b>{_max_gex_strike:,}</b> into expiry."
            )
            _action_lines = [
                f"✅ <b>Short ATM straddle at {_max_gex_strike:,}</b> — classic expiry magnet, theta is maximum here",
                f"✅ <b>Iron condor</b> with strikes at {_put_wall_k:,} / {_call_wall_k:,} — both walls defined",
                f"⚠️ <b>Avoid directional bets</b> — the magnet will frustrate both bulls and bears",
                f"⚠️ <b>Stop-loss</b> if spot moves more than {2*NIFTY_STEP:.0f} pts from {_max_gex_strike:,} and holds — magnet broken",
            ]

        # GAP 2: Flip zone WITH vega spike at flip strike = high-risk / avoid short straddle
        elif _near_flip and _flip_vega_spiking:
            _regime_tag   = "⚡ FLIP ZONE  ·  VEGA SPIKE  ·  HIGH RISK"
            _regime_color = "#9333EA"
            _regime_bg    = "#FAF5FF"
            _flip_vega_dir = "long" if _flip_vega_atm > 0 else "short"
            _regime_text  = (
                f"Spot ({spot:,.0f}) is <b>within {abs(spot - _gamma_flip_now):.0f} pts of the Gamma Flip ({_flip_str})</b> "
                f"AND Net Vega is <b>spiking at that exact strike</b> (IV {'buyers' if _flip_vega_atm > 0 else 'sellers'} concentrated at the flip). "
                f"This is the most dangerous combination: a regime change is imminent AND IV is set to "
                f"{'expand sharply if the flip is crossed' if _flip_vega_atm > 0 else 'compress suddenly — a trap for IV buyers'}. "
                f"<b>Avoid short straddles here</b> — the flip + vega spike = asymmetric risk."
            )
            _action_lines = [
                f"🚨 <b>DO NOT sell straddles or strangles</b> — flip zone + vega spike = maximum regime uncertainty",
                f"🚨 <b>Reduce ALL short option exposure</b> immediately until spot resolves clear of {_flip_str}",
                f"✅ <b>Long straddle / strangle</b> is the only structurally safe position here",
                f"✅ <b>Binary trigger</b>: close above {_flip_str} = buy calls/bull spreads; close below = buy puts/bear spreads",
            ]

        # Flip zone WITHOUT vega spike = straddle is ok (cheaper, no IV spike risk)
        elif _near_flip and not _flip_vega_spiking:
            _regime_tag   = "⚡ FLIP ZONE  ·  LOW VEGA  ·  REGIME UNSTABLE"
            _regime_color = "#7C3AED"
            _regime_bg    = "#F5F3FF"
            _regime_text  = (
                f"Spot ({spot:,.0f}) is <b>within {abs(spot - _gamma_flip_now):.0f} pts of the Gamma Flip ({_flip_str})</b>. "
                f"IV at the flip strike is <b>not spiking yet</b> — options are relatively cheap here. "
                f"A small move can flip dealers from long gamma (pinning) to short gamma (amplifying), "
                f"but the lack of vega spike means the move may be slower to develop. "
                f"{'IV buyers active at ATM — directional move being priced in.' if _iv_expanding else 'IV sellers still present — flip may not trigger immediately.'}"
            )
            _action_lines = [
                f"⚠️ <b>Reduce position size</b> — regime change is close, direction unclear",
                f"⚠️ <b>No new short premium</b> until spot resolves above or below {_flip_str}",
                f"✅ <b>Long straddle / strangles are relatively cheap here</b> — IV not yet spiking at flip",
                f"✅ <b>Watch {_flip_str}</b>: close above = pinning regime; close below = trending/amplifying begins",
            ]

        # GAP 1 (enhanced): Low/negative GEX + HIGH long vega = EXPLOSIVE BREAKOUT RISK
        elif not _is_pos_gex and _high_long_vega:
            _regime_tag   = "🔴 EXPLOSIVE BREAKOUT RISK  ·  SHORT-GAMMA + HIGH LONG VEGA"
            _regime_color = "#B91C1C"
            _regime_bg    = "#FEF2F2"
            _regime_text  = (
                f"<b>Explosive breakout risk is elevated.</b> "
                f"Spot ({spot:,.0f}) is in a short-gamma regime (Net GEX negative"
                + (f", below Gamma Flip at {_flip_str}" if _gamma_flip_now else "")
                + f") — dealers amplify every move. "
                f"Simultaneously, IV buyers have <b>high long vega exposure near ATM</b>, "
                f"meaning the market is actively pricing in a large swing. "
                f"Both forces point the same direction: <b>a big move is coming and it will accelerate</b>."
            )
            _action_lines = [
                f"🚨 <b>DO NOT sell naked options</b> — short-gamma + high long vega = explosive move risk",
                f"🚨 <b>Close any short strangles / short straddles immediately</b>",
                f"✅ <b>Buy directional debit spreads</b> in the direction momentum is pointing",
                f"✅ <b>Long straddle / strangle</b> if direction unclear — IV expansion will pay for it",
                f"📌 <b>IV kindling zone at {_max_pos_vega_k:,}</b> — crossing here triggers maximum vol expansion",
            ]

        # Low/negative GEX + marginal/low long vega = trending but not explosive
        elif not _is_pos_gex and _iv_expanding:
            _regime_tag   = "🟠 TRENDING  ·  IV EXPANDING"
            _regime_color = "#DC2626"
            _regime_bg    = "#FEF2F2"
            _regime_text  = (
                f"Spot ({spot:,.0f}) is in a short-gamma regime (Net GEX negative"
                + (f", below Gamma Flip at {_flip_str}" if _gamma_flip_now else "")
                + f"). Dealers amplify moves — they buy rallies and sell drops. "
                f"IV buyers are active at ATM (positive Net Vega), though not at explosive levels yet. "
                f"This is a <b>directional session</b> — moves are likely to extend."
            )
            _action_lines = [
                f"🚨 <b>Do NOT sell naked options</b> — short gamma + expanding IV = compounding risk",
                f"✅ <b>Buy directional debit spreads</b> in the direction of the trend",
                f"✅ <b>Long straddle</b> near ATM if direction unclear — vega gains likely",
                f"📌 <b>IV gravity well at {_max_neg_vega_k:,}</b> — expect temporary IV compression if spot reaches there",
            ]

        # Positive GEX + IV suppressed = strong pinning (Matrix 1 Row 1 — fully covered)
        elif _is_pos_gex and _above_flip and _iv_suppressed:
            _regime_tag   = "🟢 STRONG PINNING  ·  IV SUPPRESSED"
            _regime_color = "#059669"
            _regime_bg    = "#ECFDF5"
            _regime_text  = (
                f"<b>Strong pinning regime.</b> "
                f"Spot ({spot:,.0f}) is above the Gamma Flip ({_flip_str}) — dealers are net long gamma "
                f"(they dampen all moves). "
                f"IV sellers dominate at ATM (negative Net Vega), compressing premiums further. "
                f"The Call Wall at <b>{_call_wall_k:,}</b> ({_dist_to_call:.0f} pts away) acts as a hard ceiling "
                + (f"<b>with vega confluence</b> (GEX peak = Vega peak at same strike — double confirmation). " if _confluence else ". ")
                + f"This is a classic low-range session."
            )
            _action_lines = [
                f"✅ <b>Sell OTM strangles / short straddle</b> — IV suppressed, theta decay is maximised",
                f"✅ <b>Range to trade: {_put_wall_k:,} – {_call_wall_k:,}</b> ({_range_width:.0f} pts wide) — fade moves to extremes",
                f"⚠️ <b>Stop-loss</b> if spot closes above {_call_wall_k:,} — pinning regime breaks",
                f"⚠️ <b>Stop-loss</b> if spot closes below Gamma Flip ({_flip_str}) — regime shifts to trending",
            ]

        elif _is_pos_gex and _above_flip and _iv_expanding:
            _regime_tag   = "🟡 RANGE-BOUND  ·  IV BUILDING"
            _regime_color = "#D97706"
            _regime_bg    = "#FFFBEB"
            _regime_text  = (
                f"Spot ({spot:,.0f}) is above the Gamma Flip ({_flip_str}) — dealers still long gamma (pinning). "
                f"However, IV buyers are becoming active near ATM (positive Net Vega), suggesting "
                f"the market is <b>pricing in a potential breakout</b>. "
                f"The Call Wall at <b>{_call_wall_k:,}</b> is the key trigger level. "
                f"This is a <b>transitional state</b> — pinning may break if IV keeps building."
            )
            _action_lines = [
                f"⚠️ <b>Avoid naked short premium</b> — IV is rising, vega losses could offset theta gains",
                f"✅ <b>Consider debit spreads</b> toward {_call_wall_k:,} if IV momentum continues",
                f"✅ <b>Watch {_call_wall_k:,}</b> — a break with volume = short-gamma trending regime",
                f"📌 <b>Range still valid</b>: {_put_wall_k:,} – {_call_wall_k:,}, but risk is elevated",
            ]

        elif not _is_pos_gex and _iv_suppressed:
            _regime_tag   = "🟠 TRENDING  ·  IV SUPPRESSED (CAUTION)"
            _regime_color = "#EA580C"
            _regime_bg    = "#FFF7ED"
            _regime_text  = (
                f"Spot ({spot:,.0f}) is in a short-gamma regime (Net GEX negative) but IV sellers are still active near ATM. "
                f"<b>Unstable combination</b> — the directional move is underway but IV hasn't caught up yet. "
                f"A sudden IV spike is possible if spot approaches the IV kindling zone at <b>{_max_pos_vega_k:,}</b>."
            )
            _action_lines = [
                f"⚠️ <b>Caution with short premium</b> — short-gamma amplifies moves quickly",
                f"✅ <b>Tight stop losses</b> on any short options position",
                f"📌 <b>Watch {_max_pos_vega_k:,}</b> — if spot reaches IV kindling zone, expect vol spike",
                f"✅ <b>Debit spreads</b> are safer than naked options in this regime",
            ]

        else:
            _regime_tag   = "⬜ NEUTRAL  ·  REGIME UNCLEAR"
            _regime_color = "#6B7280"
            _regime_bg    = "#F9FAFB"
            _regime_text  = (
                f"GEX and Vega signals are mixed near spot ({spot:,.0f}). "
                f"Insufficient signal strength for a high-confidence regime call. Collect more ticks."
            )
            _action_lines = [
                "📌 Wait for clearer GEX/Vega alignment before entering directional or premium trades",
            ]

        # ── GAP 4: Zone-level breakdown row ──────────────────────────────────
        _zone_html = (
            f"<div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px'>"
            f"<div style='flex:1;min-width:200px;background:rgba(0,0,0,0.04);border-radius:8px;"
            f"padding:8px 12px;font-size:12px'>"
            f"<span style='font-weight:700;color:#6B7280;font-size:10px;text-transform:uppercase'>"
            f"Below Spot Zone</span><br>{_below_zone_lbl}</div>"
            f"<div style='flex:1;min-width:200px;background:rgba(0,0,0,0.04);border-radius:8px;"
            f"padding:8px 12px;font-size:12px'>"
            f"<span style='font-weight:700;color:#6B7280;font-size:10px;text-transform:uppercase'>"
            f"Above Spot Zone</span><br>{_above_zone_lbl}</div>"
            + (f"<div style='flex:1;min-width:200px;background:#FEF9C3;border-radius:8px;"
               f"padding:8px 12px;font-size:12px'>"
               f"<span style='font-weight:700;color:#92400E;font-size:10px;text-transform:uppercase'>"
               f"Extended Resistance</span><br>"
               f"🟡 Secondary wall at <b>{_ext_call_wall_k:,}</b> — "
               f"{'+ IV expansion risk on breakout past primary wall' if _ext_vega_here > 0 else 'muted vega above primary wall'}"
               f"</div>" if _has_ext_wall else "")
            + (f"<div style='flex:1;min-width:200px;background:#F0FDF4;border-radius:8px;"
               f"padding:8px 12px;font-size:12px'>"
               f"<span style='font-weight:700;color:#166534;font-size:10px;text-transform:uppercase'>"
               f"Confluence ✅</span><br>"
               f"GEX peak = Vega peak at <b>{_call_wall_k:,}</b> — double-confirmed resistance"
               f"</div>" if _confluence else "")
            + f"</div>"
        )

        # ── Key levels strip ──────────────────────────────────────────────────
        _levels_html = (
            f"<span style='color:#DC2626;font-weight:700'>Call Wall: {_call_wall_k:,}</span>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"<span style='color:#059669;font-weight:700'>Put Wall: {_put_wall_k:,}</span>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"<span style='color:#7C3AED;font-weight:700'>Gamma Flip: {_flip_str}</span>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"<span style='color:#F97316;font-weight:700'>IV Gravity: {_max_neg_vega_k:,}</span>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"<span style='color:#2563EB;font-weight:700'>IV Kindling: {_max_pos_vega_k:,}</span>"
            + (f"&nbsp;&nbsp;|&nbsp;&nbsp;<span style='color:#0891B2;font-weight:700'>Ext. Wall: {_ext_call_wall_k:,}</span>"
               if _has_ext_wall else "")
        )

        _actions_html = "".join(
            f"<div style='margin:4px 0;font-size:13px;color:#1A1A2E'>{a}</div>" for a in _action_lines
        )

        st.markdown(
            f"""<div style='background:{_regime_bg};border:2px solid {_regime_color};
                border-radius:12px;padding:16px 20px;margin:14px 0'>
              <div style='font-size:11px;font-weight:700;color:#6B7280;
                text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px'>
                ⚡ GEX + Vega Live Interpretation</div>
              <div style='font-size:15px;font-weight:800;color:{_regime_color};
                margin-bottom:10px'>{_regime_tag}</div>
              <div style='font-size:13px;color:#374151;line-height:1.7;
                margin-bottom:12px'>{_regime_text}</div>
              {_zone_html}
              <div style='background:rgba(0,0,0,0.04);border-radius:8px;
                padding:10px 14px;margin-bottom:12px;font-size:11px;
                color:#6B7280;line-height:2.0'>{_levels_html}</div>
              <div style='font-size:11px;font-weight:700;color:#6B7280;
                text-transform:uppercase;margin-bottom:6px'>Actionable Guidance</div>
              {_actions_html}
            </div>""",
            unsafe_allow_html=True,
        )
    except Exception as _interp_err:
        st.caption(f"GEX+Vega interpretation unavailable: {_interp_err}")

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 2 + ALERT CARD — Sub-D: Gamma Blast Proximity Detector
    # Identifies the nearest high-gamma OI concentration above and below spot.
    # A large OI×Gamma strike within ±2 strikes of spot = gamma spring loaded.
    # When spot moves through such a strike, the dealer hedging cascade begins.
    # ─────────────────────────────────────────────────────────────────────────
    # Partition: OTM calls above spot, OTM puts below spot (pure directional walls)
    _gd_above = _gd_src[_gd_src["strike"] > (spot + _gd_atm_band)].copy()
    _gd_below = _gd_src[_gd_src["strike"] < (spot - _gd_atm_band)].copy()

    # Focus on ±5 strikes for proximity assessment
    _gd_near_above = _gd_above[_gd_above["strike"] <= spot + 5 * NIFTY_STEP]
    _gd_near_below = _gd_below[_gd_below["strike"] >= spot - 5 * NIFTY_STEP]

    # Nearest significant call wall (highest call_gex within 5 strikes above)
    _gd_call_wall_k = None; _gd_call_wall_gex = 0.0; _gd_call_wall_dist = None
    _gd_put_wall_k  = None; _gd_put_wall_gex  = 0.0; _gd_put_wall_dist  = None

    if not _gd_near_above.empty:
        _cw = _gd_near_above.loc[_gd_near_above["call_gex"].idxmax()]
        _gd_call_wall_k   = float(_cw["strike"])
        _gd_call_wall_gex = float(_cw["call_gex"])
        _gd_call_wall_dist = _gd_call_wall_k - spot

    if not _gd_near_below.empty:
        _pw = _gd_near_below.loc[_gd_near_below["put_gex"].idxmax()]
        _gd_put_wall_k   = float(_pw["strike"])
        _gd_put_wall_gex = float(_pw["put_gex"])
        _gd_put_wall_dist = spot - _gd_put_wall_k

    # Blast readiness threshold: top-50th percentile of band OI×Gamma
    _gd_all_gex = pd.concat([_gd_src["call_gex"], _gd_src["put_gex"]])
    _gd_gex_p50 = float(_gd_all_gex.quantile(0.50)) if len(_gd_all_gex) > 4 else 0.0

    _gd_call_blast = (
        _gd_call_wall_dist is not None
        and _gd_call_wall_dist <= 2 * NIFTY_STEP
        and _gd_call_wall_gex >= _gd_gex_p50
    )
    _gd_put_blast = (
        _gd_put_wall_dist is not None
        and _gd_put_wall_dist <= 2 * NIFTY_STEP
        and _gd_put_wall_gex >= _gd_gex_p50
    )

    # Alert label
    if _gd_call_blast and _gd_put_blast:
        _gd_blast_color = "#7C3AED"
        _gd_blast_icon  = "⚡⚡"
        _gd_blast_label = "BILATERAL GAMMA SPRING"
        _gd_blast_sub   = "Large walls ≤ 2 strikes on both sides"
    elif _gd_call_blast:
        _gd_blast_color = "#DC2626"
        _gd_blast_icon  = "⚡"
        _gd_blast_label = "UPSIDE BLAST RISK"
        _gd_blast_sub   = f"Call wall at {_gd_call_wall_k:,.0f} — {_gd_call_wall_dist:.0f} pts away"
    elif _gd_put_blast:
        _gd_blast_color = "#16A34A"
        _gd_blast_icon  = "⚡"
        _gd_blast_label = "DOWNSIDE BLAST RISK"
        _gd_blast_sub   = f"Put wall at {_gd_put_wall_k:,.0f} — {_gd_put_wall_dist:.0f} pts away"
    else:
        _gd_blast_color = "#6B7280"
        _gd_blast_icon  = "—"
        _gd_blast_label = "CALM"
        _gd_blast_sub   = "No high-gamma wall within 2 strikes"

    _gd_col1, _gd_col2 = st.columns([3, 1])

    with _gd_col1:
        # Horizontal proximity bar chart — shows gamma walls around spot
        # Call walls plotted rightward (+x), put walls leftward (-x), spot at centre
        _gc2_fig = go.Figure()

        if not _gd_above.empty:
            _gc2_fig.add_trace(go.Bar(
                x=_gd_above["call_gex"],
                y=[f"{int(k):,}" for k in _gd_above["strike"]],
                orientation="h",
                name="Call OI×Γ (above spot)",
                marker_color=[
                    "#DC2626" if (_gd_call_wall_k and abs(k - _gd_call_wall_k) < 1) else "#FCA5A5"
                    for k in _gd_above["strike"]
                ],
                opacity=0.9,
                hovertemplate="Strike %{y}<br>Call OI×Γ: %{x:.5f}<extra>Call Wall</extra>",
            ))

        if not _gd_below.empty:
            _gc2_fig.add_trace(go.Bar(
                x=-_gd_below["put_gex"],
                y=[f"{int(k):,}" for k in _gd_below["strike"]],
                orientation="h",
                name="Put OI×Γ (below spot)",
                marker_color=[
                    "#15803D" if (_gd_put_wall_k and abs(k - _gd_put_wall_k) < 1) else "#86EFAC"
                    for k in _gd_below["strike"]
                ],
                opacity=0.9,
                hovertemplate="Strike %{y}<br>Put OI×Γ: %{x:.5f}<extra>Put Wall</extra>",
            ))

        _gc2_fig.add_vline(x=0, line_color="#9CA3AF", line_width=1.2)
        _gc2_fig.update_layout(
            title=dict(
                text="Sub-D — Blast Proximity  "
                     "<span style='font-size:11px;color:#6B7280'>"
                     "← Put Gamma Pressure | Call Gamma Pressure →  "
                     f"Highlighted = nearest wall | Spot {spot:,.0f}</span>",
                font=dict(size=13),
            ),
            barmode="overlay",
            height=310,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
            margin=dict(l=75, r=20, t=50, b=30),
            legend=dict(orientation="h", y=1.22, font=dict(size=10)),
            yaxis=dict(
                title="Strike",
                gridcolor="#F3F4F6",
                tickfont=dict(size=9),
                autorange="reversed",   # highest strike at top = natural orientation
            ),
            xaxis=dict(
                title="← Put OI×Γ  |  Call OI×Γ →",
                gridcolor="#F3F4F6",
                zeroline=True, zerolinecolor="#9CA3AF",
                tickfont=dict(size=9),
            ),
            font=dict(color="#1A1A2E", size=11),
        )
        st.plotly_chart(_gc2_fig, width='stretch', config={"displayModeBar": False})  # H23 fix: was use_container_width=True

    with _gd_col2:
        # ── Blast alert card ─────────────────────────────────────────────────
        _gd_cw_line1 = f"Strike: <b>{_gd_call_wall_k:,.0f}</b>" if _gd_call_wall_k else "No data"
        _gd_cw_line2 = f"Dist: <b>+{_gd_call_wall_dist:.0f} pts</b>" if _gd_call_wall_dist else ""
        _gd_cw_line3 = f"OI×Γ: {_gd_call_wall_gex:.5f}" if _gd_call_wall_gex else ""
        _gd_pw_line1 = f"Strike: <b>{_gd_put_wall_k:,.0f}</b>"  if _gd_put_wall_k  else "No data"
        _gd_pw_line2 = f"Dist: <b>-{_gd_put_wall_dist:.0f} pts</b>"  if _gd_put_wall_dist  else ""
        _gd_pw_line3 = f"OI×Γ: {_gd_put_wall_gex:.5f}"  if _gd_put_wall_gex  else ""

        st.markdown(f"""
        <div style="background:#fff;border:2.5px solid {_gd_blast_color};border-radius:12px;
                    padding:14px 16px;margin-top:10px;">
          <div style="text-align:center;font-size:22px;margin-bottom:2px;">{_gd_blast_icon}</div>
          <div style="text-align:center;font-size:13px;font-weight:700;
                      color:{_gd_blast_color};margin-bottom:4px;">{_gd_blast_label}</div>
          <div style="text-align:center;font-size:10px;color:#6B7280;
                      margin-bottom:12px;">{_gd_blast_sub}</div>
          <hr style="border:none;border-top:1px solid #F3F4F6;margin:10px 0;">
          <div style="font-size:11px;color:#374151;line-height:2.0;">
            <span style="color:#EF4444;font-weight:600;">▲ Nearest Call Wall</span><br>
            {_gd_cw_line1}<br>
            {_gd_cw_line2}<br>
            {_gd_cw_line3}
          </div>
          <hr style="border:none;border-top:1px solid #F3F4F6;margin:10px 0;">
          <div style="font-size:11px;color:#374151;line-height:2.0;">
            <span style="color:#22C55E;font-weight:600;">▼ Nearest Put Wall</span><br>
            {_gd_pw_line1}<br>
            {_gd_pw_line2}<br>
            {_gd_pw_line3}
          </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("⏳ Gamma data unavailable — waiting for option chain.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: KEY PRICE LEVELS
# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════
# v4 #9: LEADING SIGNALS / EARLY WARNING PANEL
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔮 Leading Signals / Early Warning</div>', unsafe_allow_html=True)

_ls_col1, _ls_col2, _ls_col3 = st.columns(3)

with _ls_col1:
    # Divergence Proximity Gauge
    _dp_color = "#DC2626" if _div_proximity >= 60 else ("#F59E0B" if _div_proximity >= 35 else "#059669")
    _dp_label = "APPROACHING DIVERGENCE" if _div_proximity >= 60 else ("WATCHING" if _div_proximity >= 35 else "CLEAR")
    st.markdown(f"""
    <div class="card">
      <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Divergence Proximity</div>
      <div style="font-size:28px;font-weight:900;color:{_dp_color};margin:6px 0;">{_div_proximity:.0f} / 100</div>
      <div style="background:#E5E7EB;border-radius:4px;height:6px;margin:4px 0;">
        <div style="background:{_dp_color};height:6px;border-radius:4px;width:{_div_proximity}%;"></div>
      </div>
      <div style="font-size:12px;font-weight:700;color:{_dp_color};">{_dp_label}</div>
      <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">Fires at 60+ — early warning before actual divergences trigger</div>
    </div>""", unsafe_allow_html=True)

    # Bias Velocity
    _vel_color = "#059669" if _velocity > 2 else ("#DC2626" if _velocity < -2 else "#6B7280")
    _vel_label = "ACCELERATING BULL" if _velocity > 5 else ("ACCELERATING BEAR" if _velocity < -5 else "STEADY")
    st.markdown(f"""
    <div class="card">
      <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Bias Velocity (Signal 6)</div>
      <div style="font-size:28px;font-weight:900;color:{_vel_color};margin:6px 0;">{_velocity:+.1f} pts/tick</div>
      <div style="font-size:12px;font-weight:700;color:{_vel_color};">{_vel_label}</div>
      <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">Acceleration: {_accel:+.1f} pts/tick²</div>
    </div>""", unsafe_allow_html=True)

with _ls_col2:
    # Gamma Flip Proximity
    if _gamma_flip_proximity:
        _gfp = _gamma_flip_proximity
        _gfp_color = "#DC2626" if _gfp["regime_risk"] == "HIGH" else ("#F59E0B" if _gfp["regime_risk"] == "ELEVATED" else "#059669")
        st.markdown(f"""
        <div class="card">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Gamma Flip Proximity</div>
          <div style="font-size:16px;font-weight:800;color:#1A1A2E;">Flip @ {_gfp["flip_strike"]:,.0f}</div>
          <div style="font-size:13px;color:{_gfp_color};font-weight:700;">Spot {_gfp["side"]} by {_gfp["distance_pts"]:,.0f}pts ({_gfp["pct_of_threshold"]:.0f}% of threshold)</div>
          <div style="background:#E5E7EB;border-radius:4px;height:6px;margin:4px 0;">
            <div style="background:{_gfp_color};height:6px;border-radius:4px;width:{min(100, _gfp["pct_of_threshold"])}%;"></div>
          </div>
          <div style="font-size:12px;font-weight:700;color:{_gfp_color};">Risk: {_gfp["regime_risk"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="card"><div style="font-size:11px;font-weight:700;color:#6B7280;">Gamma Flip Proximity</div><div style="font-size:12px;color:#9CA3AF;margin-top:6px;">No gamma flip detected</div></div>', unsafe_allow_html=True)

    # OI Momentum Exhaustion
    if _oi_exhaustion:
        _oe = _oi_exhaustion
        st.markdown(f"""
        <div class="card">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">OI Momentum Exhaustion</div>
          <div style="font-size:16px;font-weight:800;color:{_oe["color"]};">{_oe["label"]}</div>
          <div style="font-size:12px;color:#374151;">{_oe["direction"]} flow exhaust ratio: {_oe["exhaust_ratio"]:.2f}</div>
          <div style="background:#E5E7EB;border-radius:4px;height:6px;margin:4px 0;">
            <div style="background:{_oe["color"]};height:6px;border-radius:4px;width:{min(100, _oe["exhaust_ratio"] * 100)}%;"></div>
          </div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">Ratio <0.50 = exhaustion (reversal risk)</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="card"><div style="font-size:11px;font-weight:700;color:#6B7280;">OI Momentum Exhaustion</div><div style="font-size:12px;color:#9CA3AF;margin-top:6px;">Need 5+ ticks of history</div></div>', unsafe_allow_html=True)

with _ls_col3:
    # Inter-Expiry OI Flow
    # ── Inter-Expiry OI Flow: enhanced with roll detection output ────────────
    _rd = _roll_data   # from detect_roll_activity() computed above
    _roll_frac  = _rd.get("roll_fraction", 0.0)
    _roll_det   = _rd.get("roll_detected", False)
    _mom_disc   = _rd.get("momentum_discount", 1.0)
    _roll_win   = _s34_bias.get("roll_window_active", False)
    _roll_disc_applied = _s34_bias.get("roll_discount_applied", 1.0)
    _rd_detail  = (_rd.get("details") or ["Roll detection inactive"])[0]
    _card_color = "#DC2626" if _roll_det else ("#D97706" if _roll_win else "#059669")
    _disc_label = f"{_mom_disc*100:.0f}% S2 strength" if _mom_disc < 1.0 else "Full S2 strength"
    if _inter_expiry_signal and _inter_expiry_signal["available"]:
        _ie = _inter_expiry_signal
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {_card_color};">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Inter-Expiry OI Flow</div>
          <div style="font-size:13px;font-weight:700;color:#1A1A2E;">Front PCR: {_ie["front_pcr"]:.2f} · Back PCR: {_ie["back_pcr"]:.2f}</div>
          <div style="font-size:12px;color:#374151;">PCR diff (front–back): {_ie["pcr_diff"]:+.2f} · Front mom: {_ie["front_momentum"]:+,.0f}</div>
          <div style="font-size:11px;color:{_card_color};font-weight:700;margin-top:5px;">
            {"⚠ ROLL WINDOW ACTIVE" if _roll_win else "✓ No roll window"} · Roll fraction: {_roll_frac*100:.0f}% · {_disc_label}
          </div>
          <div style="font-size:11px;color:#6B7280;margin-top:3px;">{_rd_detail}</div>
        </div>""", unsafe_allow_html=True)
    else:
        # Even without cross-PCR data, show roll detection status
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {_card_color};">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Inter-Expiry OI Flow</div>
          <div style="font-size:11px;color:{_card_color};font-weight:700;margin-top:5px;">
            {"⚠ ROLL WINDOW ACTIVE" if _roll_win else "✓ No roll window"} · Roll fraction: {_roll_frac*100:.0f}% · {_disc_label}
          </div>
          <div style="font-size:11px;color:#6B7280;margin-top:3px;">{_rd_detail}</div>
        </div>""", unsafe_allow_html=True)

    # Smart Money OI Filter Stats
    _sm_quality_count = _s34_bias.get("quality_strikes_count", 0)
    _sm_label = "SMART MONEY" if _sm_quality_count > 5 else "MIXED"
    _sm_color = "#059669" if _sm_quality_count > 5 else "#F59E0B"
    st.markdown(f"""
    <div class="card">
      <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Smart Money OI Filter</div>
      <div style="font-size:16px;font-weight:800;color:{_sm_color};">{_sm_label}</div>
      <div style="font-size:12px;color:#374151;">Quality strikes passing filter</div>
      <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">Filters noise: requires significance floor + proximity to ATM</div>
    </div>""", unsafe_allow_html=True)

# ══ END LEADING SIGNALS PANEL ════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# SHANTANU'S VIEW — ND/NDM Decision Matrix (Institutional Framework)
# Based on: Options Hedging Pressure & Market Movement Analysis PDF
# Data sources: Section 4 (df_band), Section 2 (momentum), Greek Risk Framework
# ═══════════════════════════════════════════════════════════════════

st.markdown(
    '<div style="font-size:20px;font-weight:900;color:#7C3AED;letter-spacing:0.5px;'    'padding:14px 0 6px 0;border-bottom:2px solid #7C3AED;margin-bottom:12px;">'    '🎯 Shantanu\'s View</div>',
    unsafe_allow_html=True
)

# ── Compute per-strike ND and NDM from df_band_records ──────────────────────
# ND  per strike = (Call OI × |Call Δ|) − (Put OI × |Put Δ|)
# NDM per strike = (Call OI Chg × |Call Δ|) − (Put OI Chg × |Put Δ|)
_sv_df = pd.DataFrame(df_band_records).copy() if df_band_records else pd.DataFrame()

if not _sv_df.empty:
    _sv_cd  = _sv_df["call_delta"].abs()
    _sv_pd  = _sv_df["put_delta"].abs()
    _sv_df["_nd"]  = (_sv_df["call_oi"]     * _sv_cd) - (_sv_df["put_oi"]     * _sv_pd)
    _sv_df["_ndm"] = (_sv_df["call_oi_chg"] * _sv_cd) - (_sv_df["put_oi_chg"] * _sv_pd)

    _sv_atm   = safe_num(m.get("atm", spot))
    _sv_calls = _sv_df[_sv_df["strike"] > _sv_atm]
    _sv_puts  = _sv_df[_sv_df["strike"] < _sv_atm]
    _sv_atm_r = _sv_df[(_sv_df["strike"] >= _sv_atm - 25) & (_sv_df["strike"] <= _sv_atm + 25)]

    otm_call_nd  = float(_sv_calls["_nd"].sum())
    otm_call_ndm = float(_sv_calls["_ndm"].sum())
    otm_put_nd   = float(_sv_puts["_nd"].sum())
    otm_put_ndm  = float(_sv_puts["_ndm"].sum())
    atm_nd       = float(_sv_atm_r["_nd"].sum())
    atm_ndm      = float(_sv_atm_r["_ndm"].sum())
    total_nd     = float(_sv_df["_nd"].sum())
    total_ndm    = float(_sv_df["_ndm"].sum())

    _sv_df["_gex"] = (
        (_sv_df["call_oi"] * _sv_df["call_gamma"]) -
        (_sv_df["put_oi"]  * _sv_df["put_gamma"])
    ) * (spot ** 2) * 0.01
    _sv_gex_idx    = _sv_df["_gex"].abs().idxmax()
    _sv_gex_strike = int(_sv_df.loc[_sv_gex_idx, "strike"])
    _sv_ndm_at_gex = float(_sv_df.loc[_sv_gex_idx, "_ndm"])

    try:
        _sv_dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
    except Exception:
        _sv_dte = 5
    _sv_near_expiry = _sv_dte <= 2

    _sv_vix     = safe_num(_vix_raw)
    _sv_vix_chg = _vix_data.get("vix_change") if _vix_data else None
    _sv_pcr     = safe_num(m.get("pcr", 1.0))

    _sv_bull_pts = 0.0
    _sv_bear_pts = 0.0
    _sv_criteria = []

    # Criterion 1: OTM Call zone
    if otm_call_nd > 0 and otm_call_ndm > 0:
        _sv_bull_pts += 3
        _sv_criteria.append(("✅", "OTM Call: ND ⊕  NDM ⊕",
            "Strong bullish delta cascade — fresh call buying; dealers forced to buy futures",
            "bull", 3))
    elif otm_call_nd > 0 and otm_call_ndm < 0:
        _sv_bear_pts += 2
        _sv_criteria.append(("⚠️", "OTM Call: ND ⊕  NDM ⊖",
            "Bullish structure cracking — call unwind in progress; EXIT LONGS signal",
            "bear", 2))
    elif otm_call_nd < 0 and otm_call_ndm < 0:
        _sv_bear_pts += 1
        _sv_criteria.append(("❌", "OTM Call: ND ⊖  NDM ⊖",
            "Put dominance at upside strikes — no bullish call accumulation",
            "bear", 1))
    else:
        _sv_criteria.append(("➖", "OTM Call: Mixed / Flat",
            "No clear directional signal from OTM call zone", "neutral", 0))

    # Criterion 2: OTM Put zone
    if otm_put_nd < 0 and otm_put_ndm < 0:
        _sv_bear_pts += 3
        _sv_criteria.append(("✅", "OTM Put: ND ⊖  NDM ⊖",
            "Strong bearish delta cascade — fresh put buying; dealers forced to sell futures",
            "bear", 3))
    elif otm_put_nd < 0 and otm_put_ndm > 0:
        _sv_bull_pts += 2
        _sv_criteria.append(("⚠️", "OTM Put: ND ⊖  NDM ⊕",
            "Bearish structure cracking — put unwind; COVER SHORTS / potential squeeze up",
            "bull", 2))
    elif otm_put_nd > 0 and otm_put_ndm > 0:
        _sv_bull_pts += 1
        _sv_criteria.append(("🟢", "OTM Put: ND ⊕  NDM ⊕",
            "Call dominance below spot — put writers stepping back; support building",
            "bull", 1))
    else:
        _sv_criteria.append(("➖", "OTM Put: Mixed / Flat",
            "No clear directional signal from OTM put zone", "neutral", 0))

    # Criterion 3: ATM NDM (Golden Rule)
    if atm_ndm > 500:
        _sv_bull_pts += 1.5
        _sv_criteria.append(("✅", f"ATM NDM ⊕  ({atm_ndm:+,.0f})",
            "Fresh bullish flow at ATM — maximum gamma zone; most aggressive dealer buying",
            "bull", 1.5))
    elif atm_ndm < -500:
        _sv_bear_pts += 1.5
        _sv_criteria.append(("❌", f"ATM NDM ⊖  ({atm_ndm:+,.0f})",
            "Fresh bearish flow at ATM — maximum gamma zone; most aggressive dealer selling",
            "bear", 1.5))
    else:
        _sv_criteria.append(("➖", f"ATM NDM Flat  ({atm_ndm:+,.0f})",
            "No fresh conviction at ATM — await confirmation candle", "neutral", 0))

    # Criterion 4: NDM at highest-GEX strike
    if _sv_ndm_at_gex > 0:
        _sv_bull_pts += 1
        _sv_criteria.append(("✅", f"NDM ⊕ at Highest GEX ({_sv_gex_strike:,})",
            "Expect velocity UP — NDM firing where dealer hedging is most explosive",
            "bull", 1))
    elif _sv_ndm_at_gex < 0:
        _sv_bear_pts += 1
        _sv_criteria.append(("❌", f"NDM ⊖ at Highest GEX ({_sv_gex_strike:,})",
            "Expect velocity DOWN — NDM firing where dealer hedging is most explosive",
            "bear", 1))
    else:
        _sv_criteria.append(("➖", f"NDM Flat at Highest GEX ({_sv_gex_strike:,})",
            "No velocity signal at most explosive GEX concentration", "neutral", 0))

    # Criterion 5: VIX + NDM
    _sv_vix_up   = _sv_vix_chg is not None and _sv_vix_chg >= 0.3
    _sv_vix_down = _sv_vix_chg is not None and _sv_vix_chg <= -0.3
    if _sv_vix > 0:
        if _sv_vix_up and total_ndm < 0:
            _sv_bear_pts += 2
            _sv_criteria.append(("❌", f"VIX Rising ({_sv_vix:.1f}, +{_sv_vix_chg:.2f}) + NDM ⊖",
                "Institutional fear confirmed — REAL breakdown, not noise", "bear", 2))
        elif _sv_vix_down and total_ndm > 0:
            _sv_bull_pts += 1.5
            _sv_criteria.append(("✅", f"VIX Falling ({_sv_vix:.1f}, {_sv_vix_chg:.2f}) + NDM ⊕",
                "VIX deflating + bullish NDM — calm institutional accumulation", "bull", 1.5))
        elif _sv_vix > 18 and total_ndm < 0:
            _sv_bear_pts += 1
            _sv_criteria.append(("⚠️", f"VIX Elevated ({_sv_vix:.1f}) + NDM ⊖",
                "Elevated fear + bearish momentum — defensive posture warranted", "bear", 1))
        elif _sv_vix < 13 and total_ndm > 0:
            _sv_criteria.append(("⚠️", f"VIX Very Low ({_sv_vix:.1f}) + NDM ⊕",
                "Complacency alert — do not chase bull signal blindly", "caution", 0))
        else:
            _sv_criteria.append(("➖", f"VIX {_sv_vix:.1f} — Neutral Context",
                "No VIX amplification of NDM signal this session", "neutral", 0))
    else:
        _sv_criteria.append(("➖", "VIX Unavailable",
            "India VIX feed not connected — cannot cross-confirm NDM", "neutral", 0))

    # Criterion 6: Near expiry + NDM spike
    if _sv_near_expiry and abs(total_ndm) > 500:
        _sv_criteria.append(("⚡", f"Near Expiry ({_sv_dte}d) + NDM Spike ({total_ndm:+,.0f})",
            "MAXIMUM IMPACT — gamma at peak; treat all signals with urgency", "amplify", 0))

    # Criterion 7: PCR extreme + divergence
    _sv_div = (total_nd > 0 and total_ndm < 0) or (total_nd < 0 and total_ndm > 0)
    if (_sv_pcr > 1.5 or _sv_pcr < 0.7) and _sv_div:
        _sv_criteria.append(("⚠️", f"PCR Extreme ({_sv_pcr:.2f}) + ND/NDM Divergence",
            f"Classic mean-reversion setup — Max Pain gravity dominant ({int(m.get('max_pain', spot)):,})",
            "reversal", 0))

    # Final decision
    _sv_net = _sv_bull_pts - _sv_bear_pts
    if _sv_net >= 5:
        _sv_dir="STRONG BULL";    _sv_dc="#059669"; _sv_dbg="#D1FAE5"
        _sv_act="BUY / HOLD LONGS — Dealer hedge mechanical bid building"
    elif _sv_net >= 2.5:
        _sv_dir="MODERATE BULL";  _sv_dc="#10B981"; _sv_dbg="#ECFDF5"
        _sv_act="Lean Long — Bullish bias with partial confirmation"
    elif _sv_net <= -5:
        _sv_dir="STRONG BEAR";    _sv_dc="#DC2626"; _sv_dbg="#FEE2E2"
        _sv_act="SELL / HOLD SHORTS — Dealer hedge mechanical offer active"
    elif _sv_net <= -2.5:
        _sv_dir="MODERATE BEAR";  _sv_dc="#EF4444"; _sv_dbg="#FEF2F2"
        _sv_act="Lean Short — Bearish bias with partial confirmation"
    else:
        _sv_dir="NEUTRAL / WAIT"; _sv_dc="#D97706"; _sv_dbg="#FFFBEB"
        _sv_act="No clear edge — reduce size; await one more confirming criterion"

    _sv_agree = (total_nd > 0 and total_ndm > 0) or (total_nd < 0 and total_ndm < 0)
    _sv_nact  = sum(1 for c in _sv_criteria if c[3] not in ("neutral",))
    if _sv_agree and abs(_sv_net) >= 5:
        _sv_cf="HIGH";   _sv_cc="#059669"; _sv_cp=min(92, 70 + _sv_nact * 4)
    elif _sv_agree and abs(_sv_net) >= 2.5:
        _sv_cf="MEDIUM"; _sv_cc="#D97706"; _sv_cp=min(72, 52 + _sv_nact * 4)
    elif not _sv_agree:
        _sv_cf="LOW  (ND/NDM Diverge — Trust NDM)"; _sv_cc="#DC2626"; _sv_cp=max(22, 38 - _sv_nact * 3)
    else:
        _sv_cf="LOW";    _sv_cc="#DC2626"; _sv_cp=28

    # Render criteria cards (4 per row)
    for _sv_r0 in range(0, len(_sv_criteria), 4):
        _sv_row  = _sv_criteria[_sv_r0: _sv_r0 + 4]
        _sv_rcols = st.columns(len(_sv_row))
        for _sv_col, (icon, label, expl, side, pts) in zip(_sv_rcols, _sv_row):
            if side == "bull":
                _cb = "#059669"; _cbg = "#F0FDF4"
            elif side == "bear":
                _cb = "#DC2626"; _cbg = "#FFF5F5"
            elif side in ("caution", "reversal", "amplify"):
                _cb = "#D97706"; _cbg = "#FFFBEB"
            else:
                _cb = "#D1D5DB"; _cbg = "#F9FAFB"
            _ptxt = (f"<span style='float:right;font-size:11px;font-weight:700;color:{_cb};'>"
                     f"+{pts:.0f}pt{'s' if pts != 1 else ''}</span>") if pts > 0 else ""
            _sv_col.markdown(f"""
            <div style="background:{_cbg};border-left:4px solid {_cb};border-radius:6px;
                        padding:10px 12px;min-height:84px;">
              <div style="font-size:12.5px;font-weight:800;color:#1A1A2E;line-height:1.3;">
                {icon} {label} {_ptxt}</div>
              <div style="font-size:10.5px;color:#4B5563;margin-top:5px;line-height:1.4;">{expl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Verdict banner
    _sv_bc   = sum(1 for c in _sv_criteria if c[3] == "bull")
    _sv_berc = sum(1 for c in _sv_criteria if c[3] == "bear")
    _sv_nlbl = "⊕ BULLISH" if total_ndm > 0 else ("⊖ BEARISH" if total_ndm < 0 else "FLAT")
    _sv_dlbl = "⊕ BULLISH" if total_nd  > 0 else ("⊖ BEARISH" if total_nd  < 0 else "FLAT")
    _sv_atxt = "✅ Aligned" if _sv_agree else "⚠️ Diverging — Trust NDM"
    _sv_acol = "#059669"   if _sv_agree else "#DC2626"

    _vc1, _vc2, _vc3 = st.columns([3, 2, 2])
    with _vc1:
        st.markdown(f"""
        <div style="background:{_sv_dbg};border:2px solid {_sv_dc};border-radius:10px;
                    padding:16px 18px;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:{_sv_dc};text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:4px;">Final Bias</div>
          <div style="font-size:26px;font-weight:900;color:{_sv_dc};line-height:1.1;">{_sv_dir}</div>
          <div style="font-size:12px;color:#374151;margin-top:8px;font-weight:600;">{_sv_act}</div>
        </div>""", unsafe_allow_html=True)
    with _vc2:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:14px;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Confidence</div>
          <div style="font-size:17px;font-weight:900;color:{_sv_cc};margin:6px 0;">{_sv_cf}</div>
          <div style="background:#E5E7EB;border-radius:4px;height:6px;margin:4px 0;">
            <div style="background:{_sv_cc};height:6px;border-radius:4px;width:{_sv_cp}%;"></div></div>
          <div style="font-size:11px;color:#6B7280;margin-top:4px;">{_sv_cp}% signal quality</div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:3px;">
            Bull criteria: {_sv_bc} · Bear criteria: {_sv_berc}</div>
        </div>""", unsafe_allow_html=True)
    with _vc3:
        st.markdown(f"""
        <div class="card" style="padding:14px;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;
                      margin-bottom:6px;">ND / NDM Dual Filter</div>
          <div style="font-size:12px;font-weight:700;color:#1A1A2E;">Total ND:&nbsp;
            <span style="color:{'#059669' if total_nd>0 else '#DC2626'};">
              {_sv_dlbl} ({total_nd:+,.0f})</span></div>
          <div style="font-size:12px;font-weight:700;color:#1A1A2E;margin-top:4px;">Total NDM:&nbsp;
            <span style="color:{'#059669' if total_ndm>0 else '#DC2626'};">
              {_sv_nlbl} ({total_ndm:+,.0f})</span></div>
          <div style="font-size:11px;font-weight:800;color:{_sv_acol};margin-top:8px;">{_sv_atxt}</div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:3px;">
            Golden Rule: when divergent, NDM overrides ND</div>
        </div>""", unsafe_allow_html=True)


    # ── Enhanced NDM — Buyer/Writer Adjusted (Shantanu Framework Upgrade) ──────
    st.markdown(
        '<div style="font-size:16px;font-weight:900;color:#7C3AED;letter-spacing:0.4px;'
        'padding:14px 0 6px 0;border-top:2px solid #E5E7EB;margin-top:16px;margin-bottom:8px;">'
        '🔬 Enhanced NDM — Buyer / Writer Adjusted</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "Raw NDM assumes ALL OI addition is buyer-driven. "
        "Enhanced NDM corrects this: when premium FALLS as OI rises, a WRITER is adding — "
        "the MM takes the opposite side, reversing the hedge direction. "
        "If Enhanced NDM diverges from Raw NDM, the raw signal is unreliable."
    )

    # Build Enhanced NDM per strike using call_ltp / put_ltp direction
    # Prev-tick LTP: look up history[-2] if available, else fall back to current (no change)
    _prev_band_map = {}
    if len(history) >= 2:
        _prev_band_records = history[-2].get("_df_band_records", [])
        for _pb in _prev_band_records:
            _prev_band_map[_pb.get("strike")] = _pb

    _endm_rows = []
    for _r in df_band_records:
        _strike     = _r.get("strike", 0)
        _c_oi_chg   = float(_r.get("call_oi_chg", 0) or 0)
        _p_oi_chg   = float(_r.get("put_oi_chg",  0) or 0)
        _c_delta    = abs(float(_r.get("call_delta", 0) or 0))
        _p_delta    = abs(float(_r.get("put_delta",  0) or 0))
        _c_ltp      = float(_r.get("call_ltp", 0) or 0)
        _p_ltp      = float(_r.get("put_ltp",  0) or 0)

        # Prev LTP — fallback to current if no history (= no change, prem_dir = +1)
        _prev       = _prev_band_map.get(_strike, {})
        _c_ltp_prev = float(_prev.get("call_ltp", _c_ltp) or _c_ltp)
        _p_ltp_prev = float(_prev.get("put_ltp",  _p_ltp) or _p_ltp)

        # +1 = premium rising (buyer aggressor), -1 = premium falling (writer aggressor)
        _c_prem_dir = 1 if _c_ltp >= _c_ltp_prev else -1
        _p_prem_dir = 1 if _p_ltp >= _p_ltp_prev else -1

        # Call: buyer aggressor → MM short call → buys futures → +delta
        # Call: writer aggressor → MM long call  → sells futures → -delta
        _c_contrib  = _c_oi_chg * _c_delta * _c_prem_dir

        # Put: buyer aggressor → MM short put → sells futures → -delta  (prem_dir=+1 → negative)
        # Put: writer aggressor → MM long put  → buys futures → +delta  (prem_dir=-1 → positive)
        _p_contrib  = _p_oi_chg * _p_delta * (-_p_prem_dir)

        _endm_val   = _c_contrib + _p_contrib
        _raw_ndm_v  = (_c_oi_chg * _c_delta) - (_p_oi_chg * _p_delta)

        _endm_rows.append({
            "Strike":        int(_strike),
            "C OI Chg":      int(_c_oi_chg),
            "C Prem Dir":    "↑ Buyer" if _c_prem_dir == 1 else "↓ Writer",
            "P OI Chg":      int(_p_oi_chg),
            "P Prem Dir":    "↑ Buyer" if _p_prem_dir == 1 else "↓ Writer",
            "Enhanced NDM":  round(_endm_val),
            "Raw NDM":       round(_raw_ndm_v),
        })

    _endm_df        = pd.DataFrame(_endm_rows).sort_values("Strike", ascending=False)
    _endm_total_e   = int(_endm_df["Enhanced NDM"].sum())
    _endm_total_r   = int(_endm_df["Raw NDM"].sum())

    # Signal classification
    if _endm_total_e > 0 and _endm_total_r > 0:
        _endm_signal = "✅ CONFIRMED BULLISH — Buyer-driven call pressure. MM hedge = buy futures."
        _endm_sc     = "#059669"; _endm_sbg = "#D1FAE5"
    elif _endm_total_e < 0 and _endm_total_r < 0:
        _endm_signal = "✅ CONFIRMED BEARISH — Buyer-driven put pressure. MM hedge = sell futures."
        _endm_sc     = "#DC2626"; _endm_sbg = "#FEE2E2"
    elif _endm_total_e > 0 and _endm_total_r < 0:
        _endm_signal = "⚠️ DIVERGENCE — Writer puts reversing raw signal → Lean BULLISH. Verify VIX + PCR."
        _endm_sc     = "#D97706"; _endm_sbg = "#FFFBEB"
    elif _endm_total_e < 0 and _endm_total_r > 0:
        _endm_signal = "⚠️ DIVERGENCE — Writer calls reversing raw signal → Lean BEARISH. Verify VIX + PCR."
        _endm_sc     = "#D97706"; _endm_sbg = "#FFFBEB"
    else:
        _endm_signal = "➖ NEUTRAL / MIXED — No dominant aggressor side."
        _endm_sc     = "#6B7280"; _endm_sbg = "#F9FAFB"

    _endm_rc = "#059669" if _endm_total_r > 0 else "#DC2626" if _endm_total_r < 0 else "#6B7280"

    # Suppress note during first 15 min of session
    _now_ist_sv = now_ist()
    _endm_suppress = (_now_ist_sv.hour == 9 and _now_ist_sv.minute < 30)
    if _endm_suppress:
        st.warning(
            "⚠️ Enhanced NDM suppressed during 09:15–09:30: gap-open premium spikes make "
            "buyer/writer classification unreliable. Signal activates after 09:30."
        )
    else:
        _ec1, _ec2, _ec3 = st.columns([1, 1, 2])
        with _ec1:
            st.markdown(f"""
            <div style="background:#F8F7FF;border:1.5px solid #7C3AED;border-radius:10px;
                        padding:14px 16px;text-align:center;">
              <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;
                          letter-spacing:0.5px;margin-bottom:4px;">Enhanced NDM</div>
              <div style="font-size:24px;font-weight:900;color:{_endm_sc};">{_endm_total_e:+,}</div>
              <div style="font-size:10px;color:#9CA3AF;margin-top:3px;">Buyer/Writer Adjusted</div>
            </div>""", unsafe_allow_html=True)
        with _ec2:
            st.markdown(f"""
            <div style="background:#F9FAFB;border:1.5px solid #E5E7EB;border-radius:10px;
                        padding:14px 16px;text-align:center;">
              <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;
                          letter-spacing:0.5px;margin-bottom:4px;">Raw NDM</div>
              <div style="font-size:24px;font-weight:900;color:{_endm_rc};">{_endm_total_r:+,}</div>
              <div style="font-size:10px;color:#9CA3AF;margin-top:3px;">Standard Formula</div>
            </div>""", unsafe_allow_html=True)
        with _ec3:
            st.markdown(f"""
            <div style="background:{_endm_sbg};border:1.5px solid {_endm_sc};border-radius:10px;
                        padding:14px 16px;">
              <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;
                          letter-spacing:0.5px;margin-bottom:6px;">Signal Interpretation</div>
              <div style="font-size:13px;font-weight:800;color:{_endm_sc};line-height:1.5;">
                {_endm_signal}</div>
              <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">
                Divergence = Raw NDM unreliable. Trust Enhanced NDM + cross-check VIX &amp; PCR.</div>
            </div>""", unsafe_allow_html=True)

        with st.expander("📊 Strike-by-Strike Enhanced NDM Breakdown", expanded=False):
            st.caption(
                "↑ Buyer = OI added with rising premium (MM hedges WITH the move). "
                "↓ Writer = OI added with falling premium (MM hedges AGAINST the move, flipping sign). "
                "Enhanced NDM corrects raw NDM for writer-dominated strikes."
            )

            def _endm_style(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return "color:#059669;font-weight:700"
                    elif val < 0:
                        return "color:#DC2626;font-weight:700"
                return ""

            st.dataframe(
                _endm_df.style.map(_endm_style, subset=["Enhanced NDM", "Raw NDM"]),
                use_container_width=True,
                hide_index=True
            )
    # ── End Enhanced NDM ──────────────────────────────────────────────────────

else:
    st.info("⏳ Shantanu's View: Waiting for option chain data to initialise.")

# ══ END SHANTANU'S VIEW ═══════════════════════════════════════════════

st.markdown('<div class="section-header"> Section 3  Key Price Levels</div>', unsafe_allow_html=True)

gex_val   = m.get("gex", 0)
gflip     = m.get("gamma_flip")
gflip_str = str(int(gflip)) if gflip else "N/A"
spot_vs_atm = spot - safe_num(m.get("atm", spot))
chg_col = GREEN if spot_vs_atm >= 0 else RED
level_items = [
    ("Spot",       f"{spot:,.2f}",             GREEN if spot_vs_atm>=0 else RED, f"vs ATM {spot_vs_atm:+.1f}"),
    ("Max Pain",   int(m["max_pain"]),          PINK,    "Writer equilibrium"),
    ("Support",    int(m["support"]),           GREEN,   f"Dist: {m['dist_to_support']:.0f}"),
    ("Resistance", int(m["resistance"]),        RED,     f"Dist: {m['dist_to_resistance']:.0f}"),
    ("ATM Strike", int(m["atm"]),               BLUE,    "Nearest strike"),
    ("Wall Width", int(m["wall_width"]),        CYAN,    "Put→Call wall"),
    ("Gamma Flip", gflip_str,                   RED if (gflip and spot<gflip) else GREEN,
                   ("Short-γ zone" if gflip and spot<gflip else "Above flip")),
    ("Net GEX",    f"{gex_val:,.0f}",           GREEN if gex_val>0 else RED,
                   "+ve=pin / -ve=trend"),
]
lv_cols = st.columns(8)
for col, (label, val, color, tip) in zip(lv_cols, level_items):
    col.markdown(f"""
    <div class="card" style="text-align:center;border-bottom:3px solid {color};">
      <div style="font-size:11px;font-weight:600;color:#6B7280;">{label}</div>
      <div style="font-size:20px;font-weight:800;color:{color};">{val}</div>
      <div style="font-size:11px;color:#374151;margin-top:2px;">{tip}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: STRIKE-WISE CHARTS (Structural Band ±10)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Section 4  Strike-wise Charts (Structural Band ±10)</div>', unsafe_allow_html=True)

df_band = pd.DataFrame(payload["df_band"])
if not df_band.empty:
    x = df_band["strike"]

    call_delta_abs = df_band["call_delta"].abs()
    put_delta_abs  = df_band["put_delta"].abs()
    dv = (df_band["call_oi"] * call_delta_abs) - (df_band["put_oi"] * put_delta_abs)
    dc = [RED if v>0 else GREEN for v in dv]
    mv = (df_band["call_oi_chg"] * call_delta_abs) - (df_band["put_oi_chg"] * put_delta_abs)
    mc = [RED if v>0 else GREEN for v in mv]

    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        f1 = go.Figure(go.Bar(x=x, y=dv, marker_color=dc))
        f1.add_vline(x=spot, line_width=2, line_dash="dash", line_color=CYAN)
        f1.update_layout(title="★ Δ-Weighted Net OI", height=275,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB", margin=dict(l=40,r=18,t=50,b=40), font=dict(color="#1A1A2E",size=11),
            hoverlabel=dict(bgcolor="#fff",font_color="#1A1A2E",font_size=11),
            legend=dict(font=dict(color="#1A1A2E",size=11)))
        st.plotly_chart(f1, width='stretch', config={"displayModeBar":False})
    with ch2:
        f2 = go.Figure(go.Bar(x=x, y=mv, marker_color=mc))
        f2.add_vline(x=spot, line_width=2, line_dash="dash", line_color=CYAN)
        f2.update_layout(title="★ Δ-Weighted OI Momentum", height=275,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB", margin=dict(l=40,r=18,t=50,b=40), font=dict(color="#1A1A2E",size=11),
            hoverlabel=dict(bgcolor="#fff",font_color="#1A1A2E",font_size=11),
            legend=dict(font=dict(color="#1A1A2E",size=11)))
        st.plotly_chart(f2, width='stretch', config={"displayModeBar":False})
    with ch3:
        gv = (df_band["call_oi"]*df_band["call_gamma"] - df_band["put_oi"]*df_band["put_gamma"]) * spot**2 * 0.01
        gc2 = [RED if v>0 else GREEN for v in gv]
        f3 = go.Figure(go.Bar(x=x, y=gv, marker_color=gc2))
        f3.add_vline(x=spot, line_width=2, line_dash="dash", line_color=CYAN)
        if m.get("gamma_flip"):
            f3.add_vline(x=m["gamma_flip"], line_width=2, line_dash="dot", line_color=RED)
            f3.add_annotation(x=m["gamma_flip"], y=0, text=f"Flip@{int(m['gamma_flip'])}",
                showarrow=True, arrowhead=2, font=dict(color=RED,size=10), ax=0, ay=-30)
        f3.update_layout(title="★ GEX per Strike (Distance-Weighted)", height=275,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB", margin=dict(l=40,r=18,t=50,b=40), font=dict(color="#1A1A2E",size=11),
            hoverlabel=dict(bgcolor="#fff",font_color="#1A1A2E",font_size=11),
            legend=dict(font=dict(color="#1A1A2E",size=11)))
        st.plotly_chart(f3, width='stretch', config={"displayModeBar":False})

    ch4, ch5 = st.columns(2)
    with ch4:
        f4 = go.Figure([
            go.Bar(x=x, y=df_band["call_oi"], name="Call OI", marker_color="#38BDF8"),
            go.Bar(x=x, y=df_band["put_oi"],  name="Put OI",  marker_color="#FB7185"),
        ])
        f4.add_vline(x=spot, line_width=2, line_dash="dash", line_color=CYAN)
        f4.update_layout(title="Call vs Put OI", height=275, barmode="group",
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB", margin=dict(l=40,r=18,t=50,b=40), font=dict(color="#1A1A2E",size=11),
            hoverlabel=dict(bgcolor="#fff",font_color="#1A1A2E",font_size=11),
            legend=dict(orientation="h",y=1.08,x=0,font=dict(color="#1A1A2E",size=11)))
        st.plotly_chart(f4, width='stretch', config={"displayModeBar":False})
    with ch5:
        f5 = go.Figure([
            go.Scatter(x=x, y=df_band["call_iv"], mode="lines+markers", name="Call IV", line=dict(color="#38BDF8",width=2.5)),
            go.Scatter(x=x, y=df_band["put_iv"],  mode="lines+markers", name="Put IV",  line=dict(color="#FB7185",width=2.5)),
        ])
        f5.add_vline(x=spot, line_width=2, line_dash="dash", line_color=CYAN)
        f5.update_layout(title="IV Smile", height=275,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB", margin=dict(l=40,r=18,t=50,b=40), font=dict(color="#1A1A2E",size=11),
            hoverlabel=dict(bgcolor="#fff",font_color="#1A1A2E",font_size=11),
            legend=dict(orientation="h",y=1.08,x=0,font=dict(color="#1A1A2E",size=11)), yaxis_title="IV %")
        st.plotly_chart(f5, width='stretch', config={"displayModeBar":False})

    # ── IV Smile Live Interpretation (full-width, powered by session history) ─
    # Maintain intraday rolling history for trend-aware classification
    if "iv_smile_history" not in st.session_state:
        st.session_state["iv_smile_history"] = []
    _iv_hist = st.session_state["iv_smile_history"]

    # Compute OTM wing excesses for this tick and append to session state
    _iv_atm      = safe_num(m.get("atm", spot))
    _iv_atm_iv   = safe_num(m.get("atm_iv", 0))
    _iv_step     = NIFTY_STEP
    _iv_otm_p    = df_band.loc[
        df_band["strike"].between(_iv_atm - 6*_iv_step, _iv_atm - 2*_iv_step) & (df_band["put_iv"]  > 0.5), "put_iv"
    ]
    _iv_otm_c    = df_band.loc[
        df_band["strike"].between(_iv_atm + 2*_iv_step, _iv_atm + 6*_iv_step) & (df_band["call_iv"] > 0.5), "call_iv"
    ]
    if len(_iv_otm_p) >= 2 and len(_iv_otm_c) >= 2 and _iv_atm_iv > 0:
        _iv_pwe  = float(_iv_otm_p.mean()) - _iv_atm_iv
        _iv_cwe  = float(_iv_otm_c.mean()) - _iv_atm_iv
        _iv_tick = {
            "ts":              payload.get("ts_ist", ""),
            "atm_iv":          _iv_atm_iv,
            "put_wing_excess": _iv_pwe,
            "call_wing_excess":_iv_cwe,
            "skew_asymmetry":  _iv_pwe - _iv_cwe,
        }
        # Deduplicate same-timestamp reruns; keep rolling 20-tick window (~5 hrs)
        if not _iv_hist or _iv_hist[-1]["ts"] != _iv_tick["ts"]:
            _iv_hist.append(_iv_tick)
            if len(_iv_hist) > 20:
                _iv_hist.pop(0)
            _save_smile_history(_iv_hist)   # persist for mid-session joiners

    _iv_sc = classify_iv_smile_scenario(df_band, m, spot, _iv_hist)
    if _iv_sc:
        _badge_map = {
            "BEARISH":        ("#FEF2F2", "#EF4444"),
            "MILD BEAR":      ("#FFFBEB", "#D97706"),
            "EXTREME BEAR":   ("#FEF2F2", "#DC2626"),
            "BULLISH":        ("#ECFDF5", "#22C55E"),
            "EXTREME BULL":   ("#ECFDF5", "#059669"),
            "RECOVERING":     ("#EFF6FF", "#2563EB"),
            "NEUTRAL":        ("#F9FAFB", "#6B7280"),
            "EVENT RISK":     ("#FFFBEB", "#D97706"),
            "VOL COLLAPSE":   ("#F9FAFB", "#6B7280"),
            "BREAKOUT ALERT": ("#FEF3C7", "#D97706"),
            "ANOMALY":        ("#FEF3C7", "#F59E0B"),
        }
        _bg, _bc = _badge_map.get(_iv_sc["badge"], ("#F9FAFB", "#6B7280"))
        _strat_html = " ".join(
            '<span style="display:inline-block;background:rgba(92,53,204,0.08);'
            'border:1px solid rgba(92,53,204,0.25);color:#5C35CC;font-size:10px;'
            'font-family:monospace;padding:2px 8px;border-radius:4px;margin:2px 2px 0 0;">'
            + s + '</span>'
            for s in _iv_sc["strategies"]
        )
        _sig_html = "".join(
            '<div style="font-size:11.5px;color:#374151;padding:2px 0;">&#9656; ' + sig + '</div>'
            for sig in _iv_sc["signals"]
        )
        _conf     = _iv_sc["confidence"]
        _conf_col = "#059669" if _conf >= 75 else ("#D97706" if _conf >= 50 else "#6B7280")
        _tr       = _iv_sc.get("trend", {})

        # 30-min trend bar (shown only when history has 3+ ticks)
        _trend_html = ""
        if _tr.get("has_trend"):
            _td_iv   = _tr["d_atm_iv"]
            _td_put  = _tr["d_put_wing"]
            _td_cal  = _tr["d_call_wing"]
            _c_iv    = "#DC2626" if _td_iv  > 0.5  else ("#22C55E" if _td_iv  < -0.5  else "#6B7280")
            _c_put   = "#DC2626" if _td_put > 0.5  else ("#22C55E" if _td_put < -0.5  else "#6B7280")
            _c_cal   = "#22C55E" if _td_cal > 0.5  else ("#DC2626" if _td_cal < -0.5  else "#6B7280")
            _trend_html = (
                '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;'
                'padding-top:8px;border-top:1px solid #E5E7EB;align-items:center;">'
                '<span style="font-size:10px;font-weight:700;color:#6B7280;">30-MIN TREND &#916;:</span>'
                '<span style="font-size:11px;font-weight:700;color:' + _c_iv + ';">ATM IV {:+.1f} pts</span>'.format(_td_iv) +
                '<span style="color:#D1D5DB;">|</span>'
                '<span style="font-size:11px;font-weight:700;color:' + _c_put + ';">Put Wing {:+.1f} pts</span>'.format(_td_put) +
                '<span style="color:#D1D5DB;">|</span>'
                '<span style="font-size:11px;font-weight:700;color:' + _c_cal + ';">Call Wing {:+.1f} pts</span>'.format(_td_cal) +
                '<span style="font-size:10px;color:#9CA3AF;margin-left:auto;">{} ticks this session</span>'.format(_tr["ticks"]) +
                '</div>'
            )

        _put_col  = "#DC2626" if _iv_sc["put_wing_excess"]  > 4  else ("#D97706" if _iv_sc["put_wing_excess"]  > 0 else "#22C55E")
        _call_col = "#22C55E" if _iv_sc["call_wing_excess"] > 4  else ("#D97706" if _iv_sc["call_wing_excess"] > 0 else "#DC2626")
        _skew_col = "#DC2626" if _iv_sc["skew_asymmetry"]   > 2  else ("#22C55E" if _iv_sc["skew_asymmetry"]  < -2 else "#D97706")

        st.markdown(
            '<div style="background:' + _bg + ';border:1.5px solid ' + _bc + ';border-radius:10px;'
            'padding:14px 18px;margin-top:10px;margin-bottom:4px;">'

            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap;">'
            '<span style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:0.08em;">'
            '&#128208; IV Smile Live Interpretation'
            '</span>'
            '<span style="background:' + _bc + '22;color:' + _bc + ';border:1.5px solid ' + _bc + ';border-radius:6px;'
            'padding:3px 10px;font-size:12px;font-weight:800;">'
            '#{} &middot; {}'.format(_iv_sc["scenario_id"], _iv_sc["scenario_name"]) +
            '</span>'
            '<span style="background:' + _bc + ';color:#fff;border-radius:5px;'
            'padding:2px 9px;font-size:11px;font-weight:700;">' + _iv_sc["badge"] + '</span>'
            '</div>'

            '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;">'

            '<div style="background:#fff;border-radius:7px;padding:8px 12px;border:1px solid #E5E7EB;flex:1;min-width:90px;text-align:center;">'
            '<div style="font-size:9px;font-weight:700;color:#6B7280;text-transform:uppercase;">ATM IV</div>'
            '<div style="font-size:16px;font-weight:800;color:#1A1A2E;">{:.1f}%</div>'.format(_iv_sc["atm_iv"]) +
            '<div style="font-size:9px;color:#6B7280;">Rank {:.0f}%ile</div>'.format(_iv_sc["iv_rank"]) +
            '</div>'

            '<div style="background:#fff;border-radius:7px;padding:8px 12px;border:1px solid #E5E7EB;flex:1;min-width:90px;text-align:center;">'
            '<div style="font-size:9px;font-weight:700;color:#6B7280;text-transform:uppercase;">Put Wing &#916;</div>'
            '<div style="font-size:16px;font-weight:800;color:' + _put_col + ';">{:+.1f} pts</div>'.format(_iv_sc["put_wing_excess"]) +
            '<div style="font-size:9px;color:#6B7280;">vs ATM</div>'
            '</div>'

            '<div style="background:#fff;border-radius:7px;padding:8px 12px;border:1px solid #E5E7EB;flex:1;min-width:90px;text-align:center;">'
            '<div style="font-size:9px;font-weight:700;color:#6B7280;text-transform:uppercase;">Call Wing &#916;</div>'
            '<div style="font-size:16px;font-weight:800;color:' + _call_col + ';">{:+.1f} pts</div>'.format(_iv_sc["call_wing_excess"]) +
            '<div style="font-size:9px;color:#6B7280;">vs ATM</div>'
            '</div>'

            '<div style="background:#fff;border-radius:7px;padding:8px 12px;border:1px solid #E5E7EB;flex:1;min-width:90px;text-align:center;">'
            '<div style="font-size:9px;font-weight:700;color:#6B7280;text-transform:uppercase;">Skew Asym</div>'
            '<div style="font-size:16px;font-weight:800;color:' + _skew_col + ';">{:+.1f}</div>'.format(_iv_sc["skew_asymmetry"]) +
            '<div style="font-size:9px;color:#6B7280;">put &#8722; call excess</div>'
            '</div>'

            '<div style="background:#fff;border-radius:7px;padding:8px 12px;border:1px solid #E5E7EB;flex:1;min-width:100px;">'
            '<div style="font-size:9px;font-weight:700;color:#6B7280;text-transform:uppercase;margin-bottom:3px;">Confidence</div>'
            '<div style="background:#E5E7EB;border-radius:3px;height:5px;margin-bottom:3px;">'
            '<div style="background:' + _conf_col + ';width:{}%;height:5px;border-radius:3px;"></div>'.format(_conf) +
            '</div>'
            '<div style="font-size:13px;font-weight:800;color:' + _conf_col + ';">{}%</div>'.format(_conf) +
            '</div>'

            '</div>'

            '<div style="display:flex;gap:14px;flex-wrap:wrap;">'
            '<div style="flex:1;min-width:200px;">'
            '<div style="font-size:9px;font-weight:700;color:#6B7280;text-transform:uppercase;margin-bottom:4px;">Signals</div>'
            + _sig_html +
            '</div>'
            '<div style="flex:1;min-width:200px;">'
            '<div style="font-size:9px;font-weight:700;color:#6B7280;text-transform:uppercase;margin-bottom:5px;">Strategies</div>'
            + _strat_html +
            '<div style="font-size:11px;color:#6B7280;margin-top:7px;line-height:1.5;">' + _iv_sc["description"] + '</div>'
            '</div>'
            '</div>'
            + _trend_html +
            '</div>',
            unsafe_allow_html=True,
        )



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MARKET SENTIMENTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Section 1  Market Sentiments</div>', unsafe_allow_html=True)

s = compute_market_sentiments(today_history)  # Fix #2: intraday only
if s:
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    for col, label, val, color in [
        (sc1, "VEGA",     s["vega_label"],     s["vega_color"]),
        (sc2, "THETA",    s["theta_label"],    s["theta_color"]),
        (sc3, "OI",       s["oi_label"],       s["oi_color"]),
        (sc4, "STRENGTH", s["strength_label"], s["strength_color"]),
        (sc5, "POS",      f"{s['pos_score']:+.2f}", s["pos_dot"]),
        (sc6, "SENTIMENT",s["overall"],        s["overall_color"]),
    ]:
        col.markdown(f"""
        <div class="card" style="text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">{label}</div>
          <div style="font-size:20px;font-weight:800;color:{color};">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    if s.get("warming"):
        st.warning(f"⏳ Warming up  {s['n_ticks']} tick(s). Absolute thresholds active until 5 ticks.")
    st.caption(f"{s['pos_caption']} | Based on last {s['n_ticks']} ticks | Vega=IV z-score · Theta=GEX z-score · OI=PCR z-score · Strength=Net-Delta z-score")
else:
    st.info("Collecting data need at least 3 ticks for Market Sentiments.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: BIAS ENGINE + STRATEGY + KEY METRICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Section 2  Bias Engine · Strategy · Key Metrics</div>', unsafe_allow_html=True)

# Gauge + header metrics
# NOTE (CHANGE 2 audit fix): the "Hedge-Flow Bias" metric below is the legacy
# signed-delta net_delta score. Treat it as a dealer hedge-pressure read, not
# the authoritative directional call — that lives in the S3/4 / Combined
# Decision panels further down.
header_cols = st.columns(8)
metric_defs = [
    ("Symbol",      SYMBOL,                      ACCENT),
    ("Spot",        f"{spot:,.2f}",              "#1A1A2E"),
    ("Expiry",      expiry,                       MUTED),
    ("ATM IV",      f"{m['atm_iv']:.2f}%",       CYAN),
    ("ATM Strike",  int(m['atm']),                BLUE),
    ("Hedge-Flow Bias", f"{bs:+.1f}",            bc),
    ("Confidence",  f"{bias['confidence']:.0f}%", BLUE),
    ("Regime",      regime[:12],                  AMBER),
]
for col, (label, val, color) in zip(header_cols, metric_defs):
    col.markdown(f"""
    <div class="card" style="text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#6B7280;text-transform:uppercase;">{label}</div>
      <div style="font-size:16px;font-weight:800;color:{color};">{val}</div>
    </div>
    """, unsafe_allow_html=True)

# Gauge chart
bias_col, strat_col, metrics_col = st.columns([1, 1.2, 2.2])

with bias_col:
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bs,
        domain={"x":[0,1],"y":[0,1]},
        title={"text":"Hedge-Flow Bias (signed Δ×OI)","font":{"color":TEXT,"size":12}},
        number={"font":{"color":bc,"size":34}},
        gauge={
            "axis":{"range":[-100,100],"tickcolor":"#444"},
            "bar":{"color":bc},
            "bgcolor":"#fff",
            "steps":[
                {"range":[-100,-40],"color":"#fde8e8"},
                {"range":[-40,-15],"color":"#fef3c7"},
                {"range":[-15,15],"color":"#f0fdf4"},
                {"range":[15,40],"color":"#dcfce7"},
                {"range":[40,100],"color":"#bbf7d0"},
            ],
            "threshold":{"line":{"color":bc,"width":3},"thickness":0.8,"value":bs},
        },
    ))
    gauge_fig.update_layout(paper_bgcolor="#fff",plot_bgcolor="#fff",
                            margin=dict(l=20,r=20,t=30,b=5),height=200)
    st.plotly_chart(gauge_fig, width='stretch', config={"displayModeBar":False})

with strat_col:
    st.markdown(f"""
    <div class="card">
      <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Market Mode</div>
      <div style="font-size:16px;font-weight:700;color:{strat['mode_color']};margin-bottom:8px;">{strat['market_mode']}</div>
      <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Direction</div>
      <div style="font-size:14px;font-weight:700;color:{bc};margin-bottom:8px;">{direction_label}</div>
      <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Strategy</div>
      <div style="font-size:14px;font-weight:700;color:{strat['color']};margin-bottom:6px;">{strat['name']}</div>
      <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Execution</div>
      <div style="font-size:13px;font-weight:700;color:#111;margin-bottom:6px;line-height:1.5;">{strat['legs']}</div>
      <div style="font-size:11px;color:#0891B2;font-weight:600;">{strat.get('iv_context','')}</div>
      <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;margin-top:8px;">Why</div>
      {''.join(f'<div style="font-size:12px;color:#111;font-weight:600;margin-top:3px;"> {f}</div>' for f in bias.get('factors',[]))}
    </div>
    """, unsafe_allow_html=True)

with metrics_col:
    mc = st.columns(4)
    metric_items = [
        ("EV Ratio", m["ev_ratio"], GREEN if m["ev_ratio"]>=1.05 else (AMBER if m["ev_ratio"]>=0.95 else RED)),
        ("Net Delta", f"{int(m['net_delta']):,}", GREEN if m["net_delta"]>0 else RED),
        ("Momentum",  f"{int(m['momentum']):,}",  GREEN if m["momentum"]>0 else RED),
        ("GEX",       f"{m['gex']:,.0f}",          GREEN if m["gex"]>0 else RED),
        ("Vanna",     f"{m['vanna']:.2f}",          GREEN if m["vanna"]>0 else RED),
        ("Vega Skew", m["vega_skew"],               GREEN if m["vega_skew"]>=1.05 else RED),
        ("PCR",       m["pcr"],                     GREEN if 0.6<=m["pcr"]<=1.2 else (BLUE if m["pcr"]>1.2 else RED)),
        ("G/T Ratio", m["gt_ratio"],                BLUE),
        ("ATM Pressure", f"{int(m['atm_pressure']):,}", GREEN if m["atm_pressure"]>0 else RED),
        ("Skew Slope", m["skew_slope"],              RED if m["skew_slope"]>0 else GREEN),
        ("Near OI %", f"{m['near_oi_concentration']*100:.1f}%", CYAN),
        ("Near OI Chg%", f"{m['near_oichg_concentration']*100:.1f}%", CYAN),
        ("IV Rank",   f"{m.get('iv_rank',0):.0f}",  RED if m.get("iv_rank",0)>=70 else (GREEN if m.get("iv_rank",0)<=30 else AMBER)),
        ("Gamma Flip", int(m["gamma_flip"]) if m.get("gamma_flip") else "N/A",
                       RED if m.get("gamma_flip") and m.get("atm",0)<m["gamma_flip"] else GREEN),
        ("Support",   int(m["support"]),    GREEN),
        ("Resistance",int(m["resistance"]), RED),
    ]
    for i, (label, val, color) in enumerate(metric_items):
        with mc[i % 4]:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:8px 10px;">
              <div style="font-size:10px;font-weight:700;color:#6B7280;text-transform:uppercase;">{label}</div>
              <div style="font-size:16px;font-weight:800;color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: PRE-MOVE ALERT + FAKE BREAKOUT + OI VELOCITY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚡ Section 8 — Pre-Move Alert | Fake Breakout Detector | OI Velocity</div>', unsafe_allow_html=True)

alert   = compute_pre_move_alert(m, history)
fbo     = compute_fake_breakout_score(m, history)
oi_vel  = compute_oi_velocity(history)

# Pre-move alert banner
level_map = {"NONE":("alert-none","✅ Pre-Move Monitor: All Clear","#059669"),
             "MONITOR":("alert-none"," Pre-Move Monitor: 1 Signal  Monitor","#2563EB"),
             "WATCH":("alert-watch","⚠️ Pre-Move Monitor: Stay Alert","#D97706"),
             "DANGER":("alert-danger"," Pre-Move Monitor: Elevated Move Risk","#DC2626")}
css_cls, title_txt, txt_col = level_map.get(alert["alert_level"], level_map["NONE"])
fires_str = "    ".join(alert["fires"]) if alert["fires"] else ""
details_str = " | ".join(alert["details"][:4])
st.markdown(f"""
<div class="{css_cls}">
  <div style="font-weight:700;font-size:14px;color:{txt_col};">{title_txt}  (Score: {alert['pre_move_score']}/4)</div>
  {f'<div style="font-size:12px;font-weight:600;color:{txt_col};margin-top:4px;">Active: {fires_str}</div>' if fires_str else ''}
  <div style="font-size:12px;color:{txt_col};margin-top:4px;line-height:1.6;">{details_str}</div>
</div>
""", unsafe_allow_html=True)

# FBO + WSI + OI Velocity cards
fbo_col, vel_col = st.columns([2, 2])
with fbo_col:
    fbo_colors = {"NONE":BLUE,"WATCH":AMBER,"DANGER":RED}
    fc = fbo_colors.get(fbo["alert_level"], BLUE)
    st.markdown(f"""
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-size:11px;font-weight:700;color:#1A1A2E;text-transform:uppercase;">Fake Breakout Score</span>
        <span style="font-size:22px;font-weight:800;color:{fc};">{fbo['score']} / 100</span>
      </div>
      <div style="font-size:12px;font-weight:600;color:{fc};padding:6px 8px;background:#F9FAFB;border-radius:6px;margin-bottom:8px;">{fbo['alert_text']}</div>
      {''.join(f'<div style="display:flex;justify-content:space-between;border-bottom:1px solid #E5E7EB;padding:4px 0;"><span style="font-size:11px;font-weight:600;color:#374151;">{k}</span><span style="font-size:11px;color:#6B7280;flex:1;margin:0 8px;">{v[1]}</span><span style="font-size:12px;font-weight:700;color:{"#DC2626" if v[0]>=20 else "#D97706" if v[0]>=10 else "#6B7280"};">+{v[0]}</span></div>' for k,v in fbo.get("factor_breakdown",{}).items())}
    </div>
    """, unsafe_allow_html=True)

with vel_col:
    vel_c_col = RED if oi_vel["call_vel_zscore"]>=2 else (AMBER if oi_vel["call_vel_zscore"]>=1.2 else GREEN if oi_vel["call_vel_zscore"]<=-1.2 else MUTED)
    vel_p_col = RED if oi_vel["put_vel_zscore"]>=2  else (AMBER if oi_vel["put_vel_zscore"]>=1.2  else GREEN if oi_vel["put_vel_zscore"]<=-1.2  else MUTED)
    st.markdown(f"""
    <div class="card">
      <div style="display:flex;gap:12px;margin-bottom:10px;">
        <div style="flex:1;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Call OI Vel/tick</div>
          <div style="font-size:18px;font-weight:700;color:{vel_c_col};">{oi_vel['call_oi_velocity']:+,.0f}</div>
          <div style="font-size:12px;font-weight:600;color:{vel_c_col};">z={oi_vel['call_vel_zscore']:+.2f}</div>
        </div>
        <div style="flex:1;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">Put OI Vel/tick</div>
          <div style="font-size:18px;font-weight:700;color:{vel_p_col};">{oi_vel['put_oi_velocity']:+,.0f}</div>
          <div style="font-size:12px;font-weight:600;color:{vel_p_col};">z={oi_vel['put_vel_zscore']:+.2f}</div>
        </div>
      </div>
      <div style="font-size:12px;font-weight:600;color:{'#DC2626' if oi_vel['alert_level']=='DANGER' else '#D97706' if oi_vel['alert_level']=='WATCH' else '#059669'};padding:6px 8px;background:#F9FAFB;border-radius:6px;">
        {oi_vel['alert_text']}
      </div>
    </div>
    """, unsafe_allow_html=True)

# IV History chart + OI Velocity charts
if len(today_history) >= 3:  # Fix #2: intraday only
    # Build 15-min bucketed history charts
    hist_df = pd.DataFrame(today_history)  # Fix #2: intraday only
    def bucket_series(key):
        try:
            t = pd.to_datetime(hist_df["ts"], errors="coerce")
            hist_df2 = hist_df.assign(t=t)
            hist_df2["bucket"] = hist_df2["t"].dt.floor("15min")
            hist_df2[key] = pd.to_numeric(hist_df2[key], errors="coerce").fillna(0)
            base = hist_df2[key].iloc[0]
            grp = hist_df2.groupby("bucket")[key].last().reset_index()
            grp["cum"] = grp[key] - base
            grp["lbl"] = grp["bucket"].dt.strftime("%H:%M")
            return grp["lbl"].tolist(), grp["cum"].tolist()
        except Exception:
            return [], []

    iv_labels, iv_vals = bucket_series("atm_iv")
    iv_color = "#059669" if (iv_vals[-1]<-0.5 if iv_vals else False) else ("#DC2626" if (iv_vals[-1]>0.5 if iv_vals else False) else CYAN)

    # Call/Put OI velocity z-score 15-min
    def compute_vel_buckets(side_key, total_key):
        try:
            arr = np.array([safe_num(x.get(total_key,0)) for x in today_history], dtype=float)  # Fix #2
            ts  = [x.get("ts","") for x in today_history]  # Fix #2
            vel = np.diff(arr); ts_v = ts[1:]
            buckets = {}
            for i, t in enumerate(ts_v):
                lbl = _parse_ts_to_bucket(t) or t
                buckets[lbl] = buckets.get(lbl, 0.0) + float(vel[i])
            labels = sorted(buckets.keys())
            arr_c  = np.array([buckets[l] for l in labels], dtype=float)
            if arr_c.std() < 1e-9:
                return labels, [0.0]*len(labels)
            z_arr = (arr_c - arr_c.mean()) / arr_c.std()
            return labels, list(z_arr)
        except Exception:
            return [], []

    c_vel_labels, c_vel_zs = compute_vel_buckets("call", "call_oi_total")
    p_vel_labels, p_vel_zs = compute_vel_buckets("put",  "put_oi_total")

    iv_col, cv_col, pv_col = st.columns(3)
    with iv_col:
        if iv_labels:
            iv_fig = go.Figure()
            iv_fig.add_hline(y=0, line_dash="dash", line_color=MUTED, opacity=0.5)
            iv_fig.add_trace(go.Scatter(x=iv_labels, y=iv_vals, mode="lines+markers",
                line=dict(color=iv_color, width=2.5), marker=dict(size=6),
                name="Cumul Δ ATM IV",
                hovertemplate="%{x}<br>Δ IV: %{y:+.2f}pp<extra></extra>"))
            iv_fig.update_layout(title="Cumulative Δ ATM IV (15-min)", height=260,
                paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
                margin=dict(l=40,r=18,t=50,b=40),
                font=dict(color="#1A1A2E",size=11),
            hoverlabel=dict(bgcolor="#fff",font_color="#1A1A2E",font_size=11),
            legend=dict(font=dict(color="#1A1A2E",size=11)))
            st.plotly_chart(iv_fig, width='stretch', config={"displayModeBar":False})
    with cv_col:
        if c_vel_labels:
            cf = go.Figure()
            cf.add_hline(y=2.0,  line_dash="dot", line_color=RED,   opacity=0.65, annotation_text="+2σ",  annotation_font_color=RED)
            cf.add_hline(y=-2.0, line_dash="dot", line_color=GREEN, opacity=0.65, annotation_text="-2σ",  annotation_font_color=GREEN)
            cf.add_hline(y=0,    line_dash="dash",line_color=MUTED, opacity=0.4)
            cf.add_trace(go.Scatter(x=c_vel_labels, y=c_vel_zs, mode="lines+markers",
                line=dict(color="#38BDF8",width=2.5), marker=dict(size=6),
                hovertemplate="%{x}<br>Z: %{y:+.2f}σ<extra></extra>", name="Call OI Vel Z"))
            cf.update_layout(title="Δ Call OI Velocity Z-Score (15-min)", height=260,
                paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
                margin=dict(l=40,r=18,t=50,b=40), font=dict(color="#1A1A2E",size=11),
            hoverlabel=dict(bgcolor="#fff",font_color="#1A1A2E",font_size=11),
            legend=dict(font=dict(color="#1A1A2E",size=11)))
            st.plotly_chart(cf, width='stretch', config={"displayModeBar":False})
    with pv_col:
        if p_vel_labels:
            pf = go.Figure()
            pf.add_hline(y=2.0,  line_dash="dot", line_color=RED,   opacity=0.65, annotation_text="+2σ",  annotation_font_color=RED)
            pf.add_hline(y=-2.0, line_dash="dot", line_color=GREEN, opacity=0.65, annotation_text="-2σ",  annotation_font_color=GREEN)
            pf.add_hline(y=0,    line_dash="dash",line_color=MUTED, opacity=0.4)
            pf.add_trace(go.Scatter(x=p_vel_labels, y=p_vel_zs, mode="lines+markers",
                line=dict(color="#FB7185",width=2.5), marker=dict(size=6),
                hovertemplate="%{x}<br>Z: %{y:+.2f}σ<extra></extra>", name="Put OI Vel Z"))
            pf.update_layout(title="Δ Put OI Velocity Z-Score (15-min)", height=260,
                paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
                margin=dict(l=40,r=18,t=50,b=40), font=dict(color="#1A1A2E",size=11),
            hoverlabel=dict(bgcolor="#fff",font_color="#1A1A2E",font_size=11),
            legend=dict(font=dict(color="#1A1A2E",size=11)))
            st.plotly_chart(pf, width='stretch', config={"displayModeBar":False})
else:
    st.info(f"Collecting history data for Section 8 charts {len(history)}/3 ticks received.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: DELTA-WEIGHTED OI FLOW ENGINE  (v23-4)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🧲 Section 9 — Δ-Weighted OI Flow · Raw OI Change · Sentiment · Composite Bias</div>',
    unsafe_allow_html=True,
)
# Legend note
st.markdown(
    '''<div style="font-size:12px;color:#6B7280;background:#F9FAFB;padding:8px 12px;
               border-radius:6px;line-height:1.8;margin-bottom:10px;">
    <strong>Graph 1 — Δ-Weighted OI Flow:</strong>
    running[i] = decay×running[i-1] + bucket_Σ(OI_chg×|δ|)  |
    Green line=PUT floor building (bullish)  Red line=CALL ceiling building (bearish)  
    Coloured segments=net sentiment vs ZERO  Purple dashed=rolling-normalised net ∈[−1,+1].<br>
    <strong>Graph 2 — Raw OI Change + PCR-of-Flow:</strong>
    put_added/call_added − 1 per bucket  |  Green=PUT-heavy(bullish)  Red=CALL-heavy(bearish)  |
    Vol guard ≥{} lots.<br>
    <strong>Graph 3 — NIFTY Spot:</strong>
    15-min closes with Gamma Flip (amber), Support/Resistance walls (green/red), Max Pain (grey).<br>
    <em>Click a legend item to toggle that line on/off.  Double-click to isolate it.</em>
    </div>'''.format(PCR_MIN_BUCKET_OI),
    unsafe_allow_html=True,
)

if len(history) >= 2:
    bkt  = compute_dw_flow_buckets(today_history)   # Fix #2: intraday only
    rbkt = compute_raw_oi_buckets(today_history)    # Fix #2: intraday only

    if bkt and len(bkt.get("labels", [])) >= 1:
        labels        = bkt["labels"]
        call_flow     = bkt["call_flow"]
        put_flow      = bkt["put_flow"]
        net_flow      = bkt["net_flow"]
        net_flow_norm = bkt.get("net_flow_norm", [])
        spot_arr      = bkt["spot"]
        decay         = bkt.get("decay", DW_FLOW_DECAY)
        session_max   = bkt.get("session_max", 1.0)
        norm_w        = NORM_WINDOW_BUCKETS if NORM_WINDOW_BUCKETS else "session"

        # ── Confluence Summary Panel (v23-4) ─────────────────────────────────
        pcr_signal = rbkt.get("pcr_flow_signal", []) if rbkt else []
        call_chg   = rbkt.get("call_chg", [])        if rbkt else []
        put_chg_rb = rbkt.get("put_chg", [])         if rbkt else []

        def _last2(lst, n=2): return lst[-n:] if len(lst) >= n else lst[:]

        # NET Sentiment direction
        recent_net = _last2(net_flow, 3)
        if len(recent_net) >= 2:
            signs      = [1 if v > 0 else (-1 if v < 0 else 0) for v in recent_net]
            bull_count = signs.count(1);  bear_count = signs.count(-1)
            if bull_count >= 2:   net_dir, net_consec = "BULLISH", bull_count
            elif bear_count >= 2: net_dir, net_consec = "BEARISH", bear_count
            else:                 net_dir, net_consec = "MIXED", 0
        elif len(recent_net) == 1:
            net_dir = "BULLISH" if recent_net[0] > 0 else ("BEARISH" if recent_net[0] < 0 else "MIXED")
            net_consec = 1
        else:
            net_dir, net_consec = "MIXED", 0

        # NET Normalised strength
        norm_val = net_flow_norm[-1] if net_flow_norm else 0.0
        abs_norm = abs(norm_val)
        if abs_norm >= 0.65:   norm_label = "STRONG"
        elif abs_norm >= 0.30: norm_label = "MODERATE"
        else:                  norm_label = "WEAK"

        # PCR-of-Flow confluence
        recent_pcr = _last2(pcr_signal, 3)
        if recent_pcr:
            pcr_signs = [1 if v > 0.05 else (-1 if v < -0.05 else 0) for v in recent_pcr]
            pcr_bull  = pcr_signs.count(1);  pcr_bear = pcr_signs.count(-1)
            if pcr_bull >= 2:   pcr_dir = "BULLISH"
            elif pcr_bear >= 2: pcr_dir = "BEARISH"
            else:               pcr_dir = "MIXED"
            pcr_latest = recent_pcr[-1]
        else:
            pcr_dir, pcr_latest = "MIXED", 0.0

        # Cumulative OI slope
        if len(call_chg) >= 2 and len(put_chg_rb) >= 2:
            call_slope = call_chg[-1] - call_chg[-min(3, len(call_chg))]
            put_slope  = put_chg_rb[-1] - put_chg_rb[-min(3, len(put_chg_rb))]
            if   put_slope > call_slope * 1.1:  cum_dir = "BULLISH"
            elif call_slope > put_slope * 1.1:  cum_dir = "BEARISH"
            else:                               cum_dir = "MIXED"
        else:
            cum_dir = "MIXED"

        # FIX (Issue 4): norm_val vote was a direct transform of net_flow (same
        # source as net_dir), letting net_flow cast 2 of 4 votes alone.
        # Replaced with flow acceleration: net_flow[-1] − net_flow[-2].
        # Acceleration is the second derivative — can disagree with direction
        # (e.g. flow bullish but decelerating) and is a genuinely independent signal.
        if len(net_flow) >= 2:
            _accel    = net_flow[-1] - net_flow[-2]
            accel_dir = "BULLISH" if _accel > 0 else ("BEARISH" if _accel < 0 else "MIXED")
            accel_val = _accel
        else:
            accel_dir, accel_val = "MIXED", 0.0

        # Confluence tally — 4 independent signals
        bull_votes = sum([
            1 if net_dir   == "BULLISH" else 0,   # delta-weighted flow direction (G1)
            1 if pcr_dir   == "BULLISH" else 0,   # PCR-of-Flow (G2, raw OI)
            1 if cum_dir   == "BULLISH" else 0,   # cumulative OI slope (G2, raw OI)
            1 if accel_dir == "BULLISH" else 0,   # flow acceleration 2nd derivative (G1)
        ])
        bear_votes = sum([
            1 if net_dir   == "BEARISH" else 0,
            1 if pcr_dir   == "BEARISH" else 0,
            1 if cum_dir   == "BEARISH" else 0,
            1 if accel_dir == "BEARISH" else 0,
        ])

        delta_active = bkt.get("delta_active", False)

        if bull_votes >= 3:
            overall_dir, overall_col, overall_bg, overall_emoji = "BULLISH",      "#059669", "#ECFDF5", "🟢"
        elif bear_votes >= 3:
            overall_dir, overall_col, overall_bg, overall_emoji = "BEARISH",      "#DC2626", "#FEF2F2", "🔴"
        elif bull_votes == 2 and bear_votes <= 1:
            overall_dir, overall_col, overall_bg, overall_emoji = "MILD BULLISH", "#10B981", "#F0FDF4", "🟡"
        elif bear_votes == 2 and bull_votes <= 1:
            overall_dir, overall_col, overall_bg, overall_emoji = "MILD BEARISH", "#F87171", "#FFF5F5", "🟡"
        else:
            overall_dir, overall_col, overall_bg, overall_emoji = "NEUTRAL / MIXED", "#D97706", "#FFFBEB", "⚪"

        strength_col = overall_col if abs_norm >= 0.65 else ("#D97706" if abs_norm >= 0.30 else "#6B7280")
        strength_label = "HIGH" if abs_norm >= 0.65 else ("MEDIUM" if abs_norm >= 0.30 else "LOW")

        consec_txt = f"{net_consec} of last 3 buckets confirm" if net_consec > 0 else "buckets mixed"
        if overall_dir in ("BULLISH", "MILD BULLISH"):
            explanation = (
                f"Put writers are active — floor is building. "
                f"NET sentiment green ({consec_txt}), "
                f"PCR-of-Flow {'also bullish — strong confluence.' if pcr_dir == 'BULLISH' else 'mixed — wait for confirmation.'} "
                f"Flow {'accelerating ↑' if accel_dir == 'BULLISH' else 'decelerating ↓' if accel_dir == 'BEARISH' else 'steady →'}. "
                f"Normalised flow context: {norm_val:+.2f}. "
                f"{'✅ Delta-weighted active — signals reliable.' if delta_active else '⚠️ Proxy mode — treat with caution.'}"
            )
        elif overall_dir in ("BEARISH", "MILD BEARISH"):
            explanation = (
                f"Call writers dominating — ceiling pressure building. "
                f"NET sentiment red ({consec_txt}), "
                f"PCR-of-Flow {'also bearish — strong confluence.' if pcr_dir == 'BEARISH' else 'mixed — wait for confirmation.'} "
                f"Flow {'accelerating ↓' if accel_dir == 'BEARISH' else 'decelerating ↑' if accel_dir == 'BULLISH' else 'steady →'}. "
                f"Normalised flow context: {norm_val:+.2f}. "
                f"{'✅ Delta-weighted active — signals reliable.' if delta_active else '⚠️ Proxy mode — treat with caution.'}"
            )
        else:
            explanation = (
                f"Flow is balanced — no dominant side yet. "
                f"NET: {net_dir.lower()}, PCR-of-Flow: {pcr_dir.lower()}, CumOI: {cum_dir.lower()}, Accel: {accel_dir.lower()}. "
                f"Normalised flow near zero ({norm_val:+.2f}). "
                f"Wait for 2+ consecutive aligned buckets before acting. "
                f"{'✅ Delta-weighted active.' if delta_active else '⚠️ Proxy mode active.'}"
            )

        net_col   = "#059669" if net_dir   == "BULLISH" else ("#DC2626" if net_dir   == "BEARISH" else "#D97706")
        accel_col = "#059669" if accel_dir == "BULLISH" else ("#DC2626" if accel_dir == "BEARISH" else "#D97706")
        pcr_col   = "#059669" if pcr_dir   == "BULLISH" else ("#DC2626" if pcr_dir   == "BEARISH" else "#D97706")
        cum_col   = "#059669" if cum_dir   == "BULLISH" else ("#DC2626" if cum_dir   == "BEARISH" else "#D97706")

        # ── Proxy mode warning ───────────────────────────────────────────────
        if not delta_active:
            st.warning(
                "⚠️ **PROXY MODE ACTIVE — Delta flow unreliable**  |  "
                "Dhan returned IV = 0 for all strikes this tick — BS backfill had no input. "
                "Charts use a flat 0.3 delta proxy (= scaled raw OI change). "
                "Sentiment line and composite bias score are not meaningful until real IV arrives.",
                icon=None,
            )

        # ── Confluence header panel ──────────────────────────────────────────
        st.markdown(
            f'''<div style="background:{overall_bg};border:1.5px solid {overall_col};border-radius:10px;
                        padding:12px 16px;margin-bottom:12px;">
              <div style="display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:10px;">
                <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
                  <span style="font-size:10px;font-weight:700;color:#6B7280;letter-spacing:0.05em;">DIRECTION</span>
                  <span style="background:{overall_col}22;color:{overall_col};border:1.5px solid {overall_col};
                               border-radius:6px;padding:3px 10px;font-size:13px;font-weight:800;">
                    {overall_emoji} {overall_dir}
                  </span>
                </div>
                <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
                  <span style="font-size:10px;font-weight:700;color:#6B7280;letter-spacing:0.05em;">STRENGTH</span>
                  <span style="background:{strength_col}22;color:{strength_col};border:1.5px solid {strength_col};
                               border-radius:6px;padding:3px 10px;font-size:13px;font-weight:800;">
                    {strength_label}  {norm_val:+.2f}
                  </span>
                </div>
                <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
                  <span style="font-size:10px;font-weight:700;color:#6B7280;letter-spacing:0.05em;">CONFLUENCE</span>
                  <span style="background:#F9FAFB;color:#1A1A2E;border:1px solid #E5E7EB;
                               border-radius:6px;padding:3px 10px;font-size:12px;font-weight:700;">
                    {max(bull_votes, bear_votes)}/4 signals agree
                  </span>
                </div>
                <div style="flex:1;min-width:220px;font-size:12px;color:#1A1A2E;line-height:1.6;
                            background:rgba(255,255,255,0.55);border-radius:6px;padding:4px 8px;">
                  {explanation}
                </div>
              </div>
              <div style="display:flex;gap:8px;flex-wrap:wrap;">
                <div style="flex:1;min-width:120px;background:#F9FAFB;border-radius:8px;padding:8px 10px;
                            border-left:3px solid {net_col};">
                  <div style="font-size:10px;color:#6B7280;font-weight:700;text-transform:uppercase;">NET Sentiment (G1)</div>
                  <div style="font-size:13px;font-weight:800;color:{net_col};">{net_dir}</div>
                  <div style="font-size:10px;color:#6B7280;margin-top:2px;">{net_consec}/3 buckets {net_dir.lower()}</div>
                </div>
                <div style="flex:1;min-width:120px;background:#F9FAFB;border-radius:8px;padding:8px 10px;
                            border-left:3px solid {accel_col};">
                  <div style="font-size:10px;color:#6B7280;font-weight:700;text-transform:uppercase;">Flow Accel (G1)</div>
                  <div style="font-size:13px;font-weight:800;color:{accel_col};">{accel_dir}</div>
                  <div style="font-size:10px;color:#6B7280;margin-top:2px;">Δ last bucket | norm: {norm_val:+.2f}</div>
                </div>
                <div style="flex:1;min-width:120px;background:#F9FAFB;border-radius:8px;padding:8px 10px;
                            border-left:3px solid {pcr_col};">
                  <div style="font-size:10px;color:#6B7280;font-weight:700;text-transform:uppercase;">PCR-of-Flow (G2)</div>
                  <div style="font-size:13px;font-weight:800;color:{pcr_col};">{pcr_dir}</div>
                  <div style="font-size:10px;color:#6B7280;margin-top:2px;">latest bucket: {pcr_latest:+.2f} | ratio−1</div>
                </div>
                <div style="flex:1;min-width:120px;background:#F9FAFB;border-radius:8px;padding:8px 10px;
                            border-left:3px solid {cum_col};">
                  <div style="font-size:10px;color:#6B7280;font-weight:700;text-transform:uppercase;">Cumul. OI Slope (G2)</div>
                  <div style="font-size:13px;font-weight:800;color:{cum_col};">{cum_dir}</div>
                  <div style="font-size:10px;color:#6B7280;margin-top:2px;">put slope vs call slope (last 3 bkts)</div>
                </div>
              </div>
            </div>''',
            unsafe_allow_html=True,
        )

        # ── Graph 1: Δ-Weighted OI Flow (full-width, v23-4) ─────────────────
        import math as _math
        dw_fig = go.Figure()

        # Call Δ-Flow
        dw_fig.add_trace(go.Scatter(
            x=labels, y=call_flow,
            name=f"CALL Δ-Flow  [Σ(OI_chg × |δ|), decay={decay}]  → ceiling building (bearish)",
            legendgroup="call_flow", mode="lines+markers",
            line=dict(color=RED, width=3), marker=dict(size=6, color=RED),
            hovertemplate="<b>%{x}</b><br>Call Δ-flow (decay-wtd): %{y:,.0f}<extra>CALL Δ-Flow</extra>",
            yaxis="y1",
        ))

        # Put Δ-Flow
        dw_fig.add_trace(go.Scatter(
            x=labels, y=put_flow,
            name=f"PUT  Δ-Flow  [Σ(OI_chg × |δ|), decay={decay}]  → floor building (bullish)",
            legendgroup="put_flow", mode="lines+markers",
            line=dict(color=GREEN, width=3), marker=dict(size=6, color=GREEN),
            hovertemplate="<b>%{x}</b><br>Put Δ-flow (decay-wtd): %{y:,.0f}<extra>PUT Δ-Flow</extra>",
            yaxis="y1",
        ))

        # Net Sentiment segments — coloured vs ZERO
        _COL_BULL = "#16A34A";  _COL_BEAR = "#DC2626";  _COL_NEUT = AMBER
        _SYMLOG_LIN = max(1.0, max(abs(v) for v in net_flow) * 0.01) if net_flow else 1.0
        def _symlog(v): return _math.copysign(_math.log10(1.0 + abs(v) / _SYMLOG_LIN), v)
        net_flow_sl = [_symlog(v) for v in net_flow]

        if len(net_flow) >= 2:
            for _i in range(1, len(net_flow)):
                _curr_real = net_flow[_i]
                _is_live   = (_i == len(net_flow) - 1)
                _seg_col   = _COL_BULL if _curr_real > 0 else (_COL_BEAR if _curr_real < 0 else _COL_NEUT)
                _bias_lbl  = ("BULLISH — put-writing dominates" if _curr_real > 0
                               else ("BEARISH — call-writing dominates" if _curr_real < 0 else "NEUTRAL"))
                dw_fig.add_trace(go.Scatter(
                    x=[labels[_i - 1], labels[_i]], y=[net_flow_sl[_i - 1], net_flow_sl[_i]],
                    customdata=[[net_flow[_i - 1]], [_curr_real]],
                    name="NET Sentiment  [PUT_dw − CALL_dw, sign vs zero]",
                    legendgroup="net_flow", showlegend=(_i == 1),
                    mode="lines+markers",
                    line=dict(color=_seg_col, width=3.5 if _is_live else 2.5, dash="solid"),
                    marker=dict(size=9 if _is_live else 6, color=_seg_col),
                    hovertemplate=(
                        "<b>%{x}</b>" + ("  🔴 LIVE" if _is_live else "") + "<br>"
                        "Net Δ-flow: %{customdata[0]:,.0f}<br>Signal: " + _bias_lbl +
                        "<extra>NET Sentiment</extra>"
                    ),
                    yaxis="y1",
                ))

        # Normalised Net Flow (y3, dashed — colour-coded by previous bucket close)
        if net_flow_norm:
            if len(net_flow_norm) == 1:
                _seg_col = _COL_BULL if net_flow_norm[0] > 0 else (_COL_BEAR if net_flow_norm[0] < 0 else _COL_NEUT)
                dw_fig.add_trace(go.Scatter(
                    x=labels[:1], y=net_flow_norm[:1],
                    name=f"NET Normalised  [rolling-max window={norm_w} bkts]  ∈ [−1, +1]  ÷ rolling_max≈{session_max:,.0f}",
                    legendgroup="net_norm", showlegend=True,
                    mode="lines+markers",
                    line=dict(color=_seg_col, width=2.5, dash="dash"),
                    marker=dict(size=5, color=_seg_col,
                                symbol="circle" if net_flow_norm[0] >= 0 else "circle-open"),
                    hovertemplate=(
                        "<b>%{x}</b><br>Net norm: %{y:.3f}<br>"
                        f"+1=max bullish  −1=max bearish  |  rolling max≈{session_max:,.0f}"
                        "<extra>NET Normalised</extra>"
                    ),
                    yaxis="y3",
                ))
            else:
                for _ni in range(1, len(net_flow_norm)):
                    # colour: current value vs previous bucket close
                    _prev = net_flow_norm[_ni - 1]
                    _curr = net_flow_norm[_ni]
                    _seg_col = _COL_BULL if _curr > _prev else (_COL_BEAR if _curr < _prev else _COL_NEUT)
                    _is_live = (_ni == len(net_flow_norm) - 1)
                    dw_fig.add_trace(go.Scatter(
                        x=[labels[_ni - 1], labels[_ni]],
                        y=[net_flow_norm[_ni - 1], net_flow_norm[_ni]],
                        name=f"NET Normalised  [rolling-max window={norm_w} bkts]  ∈ [−1, +1]  ÷ rolling_max≈{session_max:,.0f}",
                        legendgroup="net_norm", showlegend=(_ni == 1),
                        mode="lines+markers",
                        line=dict(color=_seg_col, width=2.5 if _is_live else 2.0, dash="dash"),
                        marker=dict(size=7 if _is_live else 5, color=_seg_col,
                                    symbol=["circle" if v >= 0 else "circle-open"
                                            for v in [net_flow_norm[_ni - 1], net_flow_norm[_ni]]]),
                        hovertemplate=(
                            "<b>%{x}</b>" + ("  🔴 LIVE" if _is_live else "") + "<br>"
                            "Net norm: %{y:.3f}<br>"
                            f"+1=max bullish  −1=max bearish  |  rolling max≈{session_max:,.0f}"
                            "<extra>NET Normalised</extra>"
                        ),
                        yaxis="y3",
                    ))
            dw_fig.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=0.8, opacity=0.35)

        dw_fig.add_hline(y=0, line_dash="dash", line_color=MUTED, line_width=1)

        # NIFTY Spot on y2
        if spot_arr:
            dw_fig.add_trace(go.Scatter(
                x=labels, y=spot_arr, name="NIFTY Spot", legendgroup="spot",
                mode="lines+markers", line=dict(color=BLUE, width=2.5), marker=dict(size=5, color=BLUE),
                hovertemplate="<b>%{x}</b><br>Spot: %{y:,.2f}<extra>NIFTY Spot</extra>",
                yaxis="y2",
            ))

        delta_note = "" if delta_active else "  ⚠️ PROXY MODE — delta≈0, greeks sparse"
        dw_fig.update_layout(
            title=(
                f"Graph 1 — Δ-Weighted OI Flow  |  "
                f"running[i] = {decay}×running[i-1] + bucket_Σ(OI_chg×|δ|)  |  "
                f"Net normalised over rolling {norm_w}-bucket window{delta_note}"
            ),
            height=420, paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
            margin=dict(l=52, r=110, t=68, b=44),
            font=dict(color="#1A1A2E", size=11),
            hoverlabel=dict(bgcolor="#fff", font_color="#1A1A2E", font_size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color="#1A1A2E", size=10),
                        itemclick="toggle", itemdoubleclick="toggleothers"),
            yaxis=dict(
                title="Decay-Weighted OI Flow (symlog)<br><sup>Green=PUT floor  Red=CALL ceiling  Segments=net sentiment</sup>",
                side="left", showgrid=True, gridcolor="#F3F4F6",
                zeroline=True, zerolinecolor=MUTED, zerolinewidth=1,
            ),
            yaxis2=dict(title="NIFTY Spot", overlaying="y", side="right",
                        showgrid=False, zeroline=False, position=0.88),
            yaxis3=dict(
                title="Net Norm [−1 → +1]<br><sup>rolling max, purple dashed</sup>",
                overlaying="y", side="right", showgrid=False, zeroline=True,
                zerolinecolor="#7C3AED", zerolinewidth=0.8,
                anchor="free", position=1.0, range=[-1.25, 1.25],
                tickfont=dict(size=9, color="#7C3AED"),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-½", "0", "+½", "+1"],
            ),
        )
        st.plotly_chart(dw_fig, width='stretch', config={"displayModeBar": False})

        # ── Graph 2: Raw OI Change + PCR-of-Flow (full-width, v23-4) ────────
        if rbkt and rbkt.get("labels"):
            rl        = rbkt["labels"]
            call_chg  = rbkt["call_chg"]
            put_chg   = rbkt["put_chg"]
            pcr_sig   = rbkt["pcr_flow_signal"]
            pcr_raw_v = rbkt.get("pcr_flow_raw", [])
            raw_spot  = rbkt["spot"]
            min_bkt   = rbkt.get("min_bucket_oi", PCR_MIN_BUCKET_OI)

            rf = go.Figure()
            # Call OI cumulative
            rf.add_trace(go.Scatter(
                x=rl, y=call_chg,
                name="CALL OI  [cumulative raw change — ceiling building (bearish)]",
                legendgroup="call_chg", mode="lines+markers",
                line=dict(color=RED, width=3), marker=dict(size=6, color=RED),
                hovertemplate="<b>%{x}</b><br>Call OI Δ cumulative: %{y:+,.0f}<extra>CALL OI cumulative</extra>",
                yaxis="y1",
            ))
            # Put OI cumulative
            rf.add_trace(go.Scatter(
                x=rl, y=put_chg,
                name="PUT  OI  [cumulative raw change — floor building (bullish)]",
                legendgroup="put_chg", mode="lines+markers",
                line=dict(color=GREEN, width=3), marker=dict(size=6, color=GREEN),
                hovertemplate="<b>%{x}</b><br>Put OI Δ cumulative: %{y:+,.0f}<extra>PUT OI cumulative</extra>",
                yaxis="y1",
            ))
            # PCR-of-Flow segments (y3) — coloured vs ZERO
            if len(pcr_sig) >= 2:
                for _ri in range(1, len(pcr_sig)):
                    _cs = pcr_sig[_ri];  _ps = pcr_sig[_ri - 1]
                    _cr = pcr_raw_v[_ri] if pcr_raw_v else 1.0
                    _pr = pcr_raw_v[_ri - 1] if pcr_raw_v else 1.0
                    _live = (_ri == len(pcr_sig) - 1)
                    _scol = _COL_BULL if _cs > 0 else (_COL_BEAR if _cs < 0 else _COL_NEUT)
                    _blbl = ("PUT-heavy ↑ — bullish floor building" if _cs > 0
                             else ("CALL-heavy ↓ — bearish ceiling building" if _cs < 0
                                   else "Balanced / thin bucket (vol guard)"))
                    rf.add_trace(go.Scatter(
                        x=[rl[_ri - 1], rl[_ri]], y=[_ps, _cs],
                        customdata=[[_pr, _ps], [_cr, _cs]],
                        name="PCR-of-Flow  [put_added/call_added − 1, sign vs zero]",
                        legendgroup="pcr_flow", showlegend=(_ri == 1),
                        mode="lines+markers",
                        line=dict(color=_scol, width=3.5 if _live else 2.5),
                        marker=dict(size=9 if _live else 6, color=_scol),
                        hovertemplate=(
                            "<b>%{x}</b>" + ("  🔴 LIVE" if _live else "") + "<br>"
                            "PCR-Flow ratio: %{customdata[0]:.3f}<br>"
                            "Signal (ratio−1): %{customdata[1]:+.3f}<br>"
                            "Signal: " + _blbl + f"<br>Vol guard ≥{min_bkt:,} lots"
                            "<extra>PCR-of-Flow</extra>"
                        ),
                        yaxis="y3",
                    ))
            elif len(pcr_sig) == 1:
                _c = _COL_BULL if pcr_sig[0] > 0 else (_COL_BEAR if pcr_sig[0] < 0 else _COL_NEUT)
                rf.add_trace(go.Scatter(
                    x=rl, y=pcr_sig, name="PCR-of-Flow  [put_added/call_added − 1, sign vs zero]",
                    legendgroup="pcr_flow", mode="markers", marker=dict(size=9, color=_c),
                    hovertemplate="<b>%{x}</b><br>PCR signal: %{y:+.3f}<extra>PCR-of-Flow</extra>",
                    yaxis="y3",
                ))
            rf.add_hline(y=0, line_dash="dash", line_color=MUTED, line_width=1)
            # Spot on y2
            if raw_spot:
                rf.add_trace(go.Scatter(
                    x=rl, y=raw_spot, name="NIFTY Spot", legendgroup="spot",
                    mode="lines+markers", line=dict(color=BLUE, width=2.5), marker=dict(size=5, color=BLUE),
                    hovertemplate="<b>%{x}</b><br>Spot: %{y:,.2f}<extra>NIFTY Spot</extra>",
                    yaxis="y2",
                ))
            # Key levels on y2
            gf2  = next((g for g in reversed(rbkt.get("gamma_flip", [])) if g is not None), None)
            sup2 = rbkt["support"][-1]    if rbkt.get("support")    else None
            res2 = rbkt["resistance"][-1] if rbkt.get("resistance") else None
            if gf2:
                rf.add_hline(y=gf2, line_dash="solid", line_color=AMBER, line_width=2,
                             annotation_text=f"Gamma Flip {gf2:,.0f}", annotation_font=dict(color=AMBER, size=10),
                             yref="y2")
            if sup2 and sup2 > 0:
                rf.add_hline(y=sup2, line_dash="dash", line_color=GREEN, line_width=1.5,
                             annotation_text=f"Support {sup2:,.0f}", annotation_font=dict(color=GREEN, size=10),
                             yref="y2")
            if res2 and res2 > 0:
                rf.add_hline(y=res2, line_dash="dash", line_color=RED, line_width=1.5,
                             annotation_text=f"Resistance {res2:,.0f}", annotation_font=dict(color=RED, size=10),
                             yref="y2")
            rf.update_layout(
                title=(
                    f"Graph 2 — Raw OI Change + PCR-of-Flow Sentiment  |  "
                    f"Formula: put_added / call_added − 1  |  "
                    f"Vol guard ≥{min_bkt:,} lots  |  "
                    "GREEN=PUT-heavy(bullish)  RED=CALL-heavy(bearish)  coloured vs ZERO"
                ),
                height=420, paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
                margin=dict(l=52, r=110, t=68, b=44),
                font=dict(color="#1A1A2E", size=11),
                hoverlabel=dict(bgcolor="#fff", font_color="#1A1A2E", font_size=11),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                            font=dict(color="#1A1A2E", size=10),
                            itemclick="toggle", itemdoubleclick="toggleothers"),
                yaxis=dict(
                    title="Cumulative Raw OI Change (lots)<br><sup>Green=PUT cumulative  Red=CALL cumulative</sup>",
                    side="left", showgrid=True, gridcolor="#F3F4F6",
                    zeroline=True, zerolinecolor=MUTED, zerolinewidth=1,
                ),
                yaxis2=dict(title="NIFTY Spot", overlaying="y", side="right",
                            showgrid=False, zeroline=False, position=0.88),
                yaxis3=dict(
                    title="PCR-Flow (put/call − 1)<br><sup>+ve=bullish  −ve=bearish</sup>",
                    overlaying="y", side="right", showgrid=False, zeroline=True,
                    zerolinecolor="#6B7280", zerolinewidth=1,
                    anchor="free", position=1.0,
                    tickfont=dict(size=9, color="#6B7280"),
                    tickvals=[-2, -1, -0.5, 0, 0.5, 1, 2],
                    ticktext=["-2", "-1", "-½", "0", "+½", "+1", "+2"],
                ),
            )
            st.plotly_chart(rf, width='stretch', config={"displayModeBar": False})

        # ── Graph 3: NIFTY Spot with Key Levels (full-width, v23-4) ─────────
        if spot_arr:
            sf2 = go.Figure()
            sf2.add_trace(go.Scatter(
                x=labels, y=spot_arr,
                name="NIFTY Spot  [15-min bucket close]",
                legendgroup="spot", mode="lines+markers",
                line=dict(color=BLUE, width=3), marker=dict(size=6, color=BLUE),
                hovertemplate="<b>%{x}</b><br>Spot: %{y:,.2f}<extra>NIFTY Spot</extra>",
            ))
            gf_val = next((g for g in reversed(bkt.get("gamma_flip", [])) if g is not None), None)
            mp_val = bkt["max_pain"][-1] if bkt.get("max_pain") else None
            sup_v  = bkt["support"][-1]    if bkt.get("support")    else None
            res_v  = bkt["resistance"][-1] if bkt.get("resistance") else None
            if gf_val:
                sf2.add_hline(y=gf_val, line_dash="solid", line_color=AMBER, line_width=2.5,
                              annotation_text=f"Gamma Flip  {gf_val:,.0f}  [GEX=0 crossover]",
                              annotation_position="top right",
                              annotation_font=dict(color=AMBER, size=10))
            if mp_val and mp_val > 0:
                sf2.add_hline(y=mp_val, line_dash="dot", line_color=MUTED, line_width=1.5,
                              annotation_text=f"Max Pain  {mp_val:,.0f}  [aggregate loss minimum]",
                              annotation_position="bottom right",
                              annotation_font=dict(color=MUTED, size=10))
            if sup_v and sup_v > 0:
                sf2.add_hline(y=sup_v, line_dash="dash", line_color=GREEN, line_width=2,
                              annotation_text=f"Support  {sup_v:,.0f}  [max put OI wall]",
                              annotation_position="top left",
                              annotation_font=dict(color=GREEN, size=10))
            if res_v and res_v > 0:
                sf2.add_hline(y=res_v, line_dash="dash", line_color=RED, line_width=2,
                              annotation_text=f"Resistance  {res_v:,.0f}  [max call OI wall]",
                              annotation_position="bottom left",
                              annotation_font=dict(color=RED, size=10))
            sf2.update_layout(
                title=(
                    "Graph 3 — NIFTY Spot (15-min)  |  "
                    "Amber=Gamma Flip  Green=Put Wall(Support)  Red=Call Wall(Resistance)  Grey=Max Pain"
                ),
                height=420, paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
                margin=dict(l=52, r=18, t=68, b=44),
                font=dict(color="#1A1A2E", size=11),
                hoverlabel=dict(bgcolor="#fff", font_color="#1A1A2E", font_size=11),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                            font=dict(color="#1A1A2E", size=10),
                            itemclick="toggle", itemdoubleclick="toggleothers"),
                yaxis=dict(title="NIFTY Spot (points)", showgrid=True, gridcolor="#F3F4F6", zeroline=False),
            )
            st.plotly_chart(sf2, width='stretch', config={"displayModeBar": False})

        # ── Composite Δ-Flow Bias + Gamma Blast Monitor side by side ────────
        # H19 fix: was `bias = ...` which shadowed the module-level `bias`
        # (compute_nifty_bias) used elsewhere. The two dicts have different
        # schemas (legacy has factors/regime/confidence; DW has components/
        # narrative/delta_active). Renamed to `dw_bias` so any code after this
        # block that reads `bias["bias_score"]` etc. correctly gets the legacy
        # nifty-bias values, not the DW composite.
        dw_bias = compute_dw_composite_bias(bkt, expiry)

        # Compute gamma blast monitor
        _alert_for_gbm = compute_pre_move_alert(m, history)
        gbm = compute_gamma_blast_monitor(bkt, m, _alert_for_gbm, spot_px=spot)  # Fix #1

        bias_col, blast_col = st.columns(2)

        with bias_col:
            b_score  = dw_bias.get("score", 0)
            b_dir    = dw_bias.get("direction", "NEUTRAL")
            b_conf   = dw_bias.get("confidence", 0)
            b_comps  = dw_bias.get("components", {})
            b_narr   = dw_bias.get("narrative", "")
            b_dact   = dw_bias.get("delta_active", False)

            if b_score >= 45:    b_col = "#059669"
            elif b_score >= 15:  b_col = "#10B981"
            elif b_score <= -45: b_col = "#DC2626"
            elif b_score <= -15: b_col = "#F87171"
            else:                b_col = AMBER

            pct_bar = int((b_score + 100) / 2)
            delta_badge = (
                '<span style="font-size:11px;font-weight:700;background:#ECFDF5;color:#065F46;padding:2px 8px;border-radius:5px;">✅ Δ-weighted active</span>'
                if b_dact else
                '<span style="font-size:11px;font-weight:700;background:#FEF3C7;color:#92400E;padding:2px 8px;border-radius:5px;">⚠️ Δ≈0 — raw proxy</span>'
            )
            comp_labels = {
                "net_flow_dir": "Net Δ-Flow Direction",
                "flow_accel"  : "Flow Acceleration",
                "gex_regime"  : "GEX Regime",
                "flip_side"   : "vs Gamma Flip",
                "max_pain"    : "Max Pain Gravity",
            }
            comp_rows_html = ""
            for key, (pts, lbl) in b_comps.items():
                col = "#059669" if pts > 0 else ("#DC2626" if pts < 0 else MUTED)
                comp_rows_html += (
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;align-items:flex-start;">'
                    f'<span style="font-size:12px;color:#1A1A2E;font-weight:600;min-width:160px;">{comp_labels.get(key, key)}</span>'
                    f'<span style="font-size:12px;font-weight:700;color:{col};margin-right:8px;white-space:nowrap;">{pts:+.1f} pts</span>'
                    f'<span style="font-size:11px;color:#6B7280;font-weight:500;">{lbl}</span></div>'
                )

            st.markdown(
                f'''<div style="background:#fff;border:1px solid #E5E7EB;border-radius:10px;
                            padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.07);margin-top:6px;">
                  <div style="display:flex;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px;">
                    <span style="font-weight:800;font-size:14px;color:#5C35CC;">Composite Δ-Flow Bias</span>
                    <span style="background:{b_col}22;color:{b_col};border:1px solid {b_col};
                                 border-radius:6px;padding:2px 10px;font-size:13px;font-weight:800;">
                      {b_dir}  {b_score:+.1f}/100
                    </span>
                    <span style="font-size:12px;color:#6B7280;font-weight:600;">Conf {b_conf}%</span>
                    {delta_badge}
                  </div>
                  <div style="background:#E5E7EB;border-radius:6px;height:10px;margin-bottom:10px;position:relative;">
                    <div style="width:{pct_bar}%;height:100%;background:{b_col};border-radius:6px;transition:width 0.4s ease;"></div>
                    <div style="position:absolute;top:-1px;left:50%;width:2px;height:12px;background:#6B7280;"></div>
                  </div>
                  <div style="margin-bottom:8px;">{comp_rows_html}</div>
                  <div style="font-size:12px;color:#1A1A2E;font-weight:500;background:#F9FAFB;
                              padding:7px 10px;border-radius:6px;line-height:1.6;">{b_narr}</div>
                </div>''',
                unsafe_allow_html=True,
            )

        with blast_col:
            g_score  = gbm["score"]
            g_stage  = gbm["stage"]
            g_col    = gbm["s_col"]
            g_bg     = gbm["s_bg"]
            g_sigs   = gbm["signals"]
            g_note   = gbm["note"]
            g_pms    = gbm["pre_move_score"]

            g_pct    = g_score  # 0-100 bar
            pms_col  = "#DC2626" if g_pms >= 2 else "#6B7280"
            pms_bg   = "#FEF2F2" if g_pms >= 2 else "#F9FAFB"

            sig_rows_html = ""
            for sig_label, sig_val, sig_pts, sig_color in g_sigs:
                pts_str = f"+{sig_pts}pt" if sig_pts > 0 else "—"
                sig_rows_html += (
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;align-items:flex-start;">'
                    f'<span style="font-size:11.5px;color:#1A1A2E;font-weight:600;min-width:120px;">{sig_label}</span>'
                    f'<span style="font-size:11px;font-weight:700;color:{sig_color};margin-right:6px;white-space:nowrap;">{pts_str}</span>'
                    f'<span style="font-size:11px;color:#6B7280;font-weight:500;">{sig_val}</span></div>'
                )

            st.markdown(
                f'''<div style="background:#fff;border:2px solid {g_col};border-radius:10px;
                            padding:14px 16px;box-shadow:0 1px 6px rgba(0,0,0,0.10);margin-top:6px;height:100%;">
                  <div style="display:flex;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
                    <span style="font-weight:800;font-size:13.5px;color:{g_col};">💥 Gamma Blast Monitor</span>
                    <span style="background:{g_bg};color:{g_col};border:1px solid {g_col};
                                 border-radius:6px;padding:2px 9px;font-size:12px;font-weight:800;">{g_stage}</span>
                    <span style="font-size:12px;color:#6B7280;font-weight:600;">Risk: {g_score}/100</span>
                    <span style="font-size:11px;color:{pms_col};font-weight:700;
                                 background:{pms_bg};padding:2px 7px;border-radius:5px;">S8 Fires: {g_pms}/4</span>
                  </div>
                  <div style="background:#E5E7EB;border-radius:6px;height:10px;margin-bottom:10px;">
                    <div style="width:{g_pct}%;height:100%;background:{g_col};border-radius:6px;transition:width 0.4s ease;"></div>
                  </div>
                  <div style="margin-bottom:8px;">{sig_rows_html}</div>
                  <div style="font-size:11.5px;color:#1A1A2E;font-weight:500;background:{g_bg};
                              padding:6px 10px;border-radius:6px;line-height:1.55;
                              border-left:3px solid {g_col};">
                    <strong style="color:{g_col};">What this means: </strong>{g_note}
                  </div>
                  <div style="font-size:10px;color:#6B7280;margin-top:6px;font-style:italic;">
                    Stage 1=structural setup · Stage 2=pressure building · Stage 3=imminent · Stage 4=blast in motion
                  </div>
                </div>''',
                unsafe_allow_html=True,
            )

    else:
        st.info(f"⏳ Collecting data for Section 9 — need 2+ buckets (≈30 min after open)...")
else:
    st.info(f"Collecting history for Section 9 charts — {len(history)}/2 ticks.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: BASIS TRIANGULATION (NIFTY Futures)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Section 10  Basis Triangulation · Spot · Synthetic Future · NIFTY Futures</div>', unsafe_allow_html=True)
st.caption("Put-call parity synthetic future vs live traded NIFTY futures vs fair carry. Futures − Synthetic (★) is the primary leading signal.")

_atm_s10 = safe_num(m.get("atm", 0))
sf_res   = compute_synthetic_future(payload["df_band"], spot, _atm_s10, expiry)
sig      = compute_basis_signals(sf_res, payload.get("traded_future"))

if sig:
    b10c1, b10c2, b10c3, b10c4 = st.columns(4)
    for col, label, val, color, sub in [
        (b10c1, "SPOT",             sig["spot"],         "#1E40AF", "Reference price"),
        (b10c2, "SYNTHETIC FUTURE", sig["synthetic"],    "#7C3AED", f"C-P parity @ {sig['atm']}"),
        (b10c3, "TRADED FUTURE",    sig["traded_future"] if sig["has_traded"] else "NO FEED",
                 GREEN if sig["has_traded"] else MUTED, "Live NIFTY Fut LTP"),
        (b10c4, "FAIR CARRY",       sig["fair_future"],  "#9CA3AF", f"r=6.5% · {sig['T_days']:.0f}d"),
    ]:
        col.markdown(f"""
        <div class="card" style="text-align:center;border:1.5px solid {color};">
          <div style="font-size:10px;font-weight:700;color:#6B7280;text-transform:uppercase;">{label}</div>
          <div style="font-size:20px;font-weight:800;color:{color};">{'N/A' if isinstance(val, type(None)) else (f'{val:,.2f}' if isinstance(val, float) else val)}</div>
          <div style="font-size:11px;color:#6B7280;margin-top:2px;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # Gap chips
    gc = st.columns(5)
    for col, label, val in [
        (gc[0], "Synth − Spot",    sig["synth_basis"]),
        (gc[1], "Synth − Fair",    sig["synth_excess"]),
        (gc[2], "Futures − Fair",  sig.get("traded_excess")),
        (gc[3], "Futures − Synth ★", sig.get("basis_gap")),
        (gc[4], "Fair Basis",      sig["fair_basis"]),
    ]:
        if val is None:
            col.metric(label, "N/A")
        else:
            color_str = "normal" if abs(val) <= 2 else ("inverse" if val < 0 else "normal")
            col.metric(label, f"{val:+.1f} pts", delta=None)

    sc_color = sig["summary_color"]
    st.markdown(f"""
    <div style="background:{sc_color}12;border:1.5px solid {sc_color};border-radius:8px;padding:8px 14px;margin:10px 0;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <span style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;">BASIS BIAS:</span>
      <span style="font-size:15px;font-weight:800;color:{sc_color};">{sig['summary_label']}</span>
      <span style="font-size:11px;color:#6B7280;font-style:italic;">independent of main bias engine  use as confirmation</span>
    </div>
    """, unsafe_allow_html=True)

    for txt, color in sig["signals"]:
        st.markdown(f'<div style="font-size:12px;color:{TEXT};margin-bottom:4px;"><span style="color:{color};font-size:13px;">● </span>{txt}</div>', unsafe_allow_html=True)
else:
    st.info("⏳ Basis Triangulation  collecting data (need valid ATM call & put LTP).")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: INTRADAY CUMULATIVE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Section 5  Intraday Cumulative Metrics (15-min buckets, IST)</div>', unsafe_allow_html=True)

if len(today_history) >= 3:  # Fix #2: intraday only — prevents yesterday's first tick becoming the baseline
    hist_df2 = pd.DataFrame(today_history)  # Fix #2
    try:
        t2 = pd.to_datetime(hist_df2["ts"], errors="coerce")
        hist_df2 = hist_df2.assign(t=t2)
        hist_df2["bucket"] = hist_df2["t"].dt.floor("15min")
        cols_5 = ["atm_iv","net_delta","oi_net_delta","max_pain"]
        for c in cols_5:
            hist_df2[c] = pd.to_numeric(hist_df2[c], errors="coerce").fillna(0)
        base5 = {c: hist_df2[c].iloc[0] for c in cols_5}
        grp5  = hist_df2.groupby("bucket")[cols_5].last().reset_index()
        for c in cols_5:
            grp5[c] = grp5[c] - base5[c]
        grp5["lbl"] = grp5["bucket"].dt.strftime("%H:%M")

        def make_cum_chart(col, title, color):
            fig = go.Figure(go.Scatter(x=grp5["lbl"], y=grp5[col], mode="lines+markers",
                line=dict(color=color, width=2.5), marker=dict(size=5)))
            fig.update_layout(title=title, height=260, paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
                margin=dict(l=40,r=18,t=50,b=40), font=dict(color="#1A1A2E",size=11),
            hoverlabel=dict(bgcolor="#fff",font_color="#1A1A2E",font_size=11),
            legend=dict(font=dict(color="#1A1A2E",size=11)))
            return fig

        c5_1, c5_2, c5_3, c5_4 = st.columns(4)
        with c5_1: st.plotly_chart(make_cum_chart("atm_iv","Cumul Δ ATM IV (15-min)",CYAN),        width='stretch', config={"displayModeBar":False})
        with c5_2: st.plotly_chart(make_cum_chart("net_delta","Cumul Δ Net Delta (15-min)",BLUE),  width='stretch', config={"displayModeBar":False})
        with c5_3: st.plotly_chart(make_cum_chart("oi_net_delta","Cumul Δ OI Momentum (15-min)",AMBER), width='stretch', config={"displayModeBar":False})
        with c5_4: st.plotly_chart(make_cum_chart("max_pain","Cumul Δ Max Pain (15-min)",PINK),   width='stretch', config={"displayModeBar":False})
    except Exception as e:
        st.warning(f"Section 5 chart error: {e}")
else:
    st.info(f"Collecting history for Section 5 {len(history)}/3 ticks.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: EOD THETA EROSION SCANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⏳ Section 6  EOD Theta Erosion Scanner (>30% Premium Decay)</div>', unsafe_allow_html=True)
st.caption("Strikes projected to lose >30% of current premium by EOD · Model: ΔP = δ·ΔS + ½Γ·ΔS² + Θ·frac + ν·Δσ · 5 scenarios (0%, ±0.5%, ±1%) · All times in IST")

df_for_erosion = pd.DataFrame(payload["df_band"])
atm_iv_val = safe_num(m.get("atm_iv", 16), 16)
if not df_for_erosion.empty and spot > 0:
    calls_df, puts_df = compute_eod_erosion(df_for_erosion, spot, atm_iv_val, expiry)
    display_cols = ["Strike","LTP (₹)","Proj.EOD (₹)","Base Erosion %","Erosion Range",
                    "Robust (of 5)","Θ Decay (₹)","ν·ΔIV (₹)","Delta","IV %","OI","OI Chg","Score","Confidence"]
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("** CALL Side  Premium Erosion >30% by EOD**")
        if not calls_df.empty:
            st.dataframe(calls_df[display_cols].style.apply(
                lambda x: ["color:#00E676;font-weight:bold" if v=="HIGH" else
                           "color:#FFD740;font-weight:bold" if v=="MEDIUM" else
                           "color:#FF5252;font-weight:bold" for v in x], subset=["Confidence"]
            ), width='stretch', hide_index=True)
        else:
            st.info("No CALL strikes meet the >30% EOD erosion threshold.")
    with ec2:
        st.markdown("** PUT Side  Premium Erosion >30% by EOD**")
        if not puts_df.empty:
            st.dataframe(puts_df[display_cols], width='stretch', hide_index=True)
        else:
            st.info("No PUT strikes meet the >30% EOD erosion threshold.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: RAW OPTION CHAIN TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Section 7  Raw Option Chain (ATM Band) · Delta Health Check</div>', unsafe_allow_html=True)

df_band_disp = pd.DataFrame(payload["df_band"])
if not df_band_disp.empty:
    needed = ["strike","call_oi","call_oi_chg","call_delta","call_iv","put_iv","put_delta","put_oi_chg","put_oi"]
    for c in needed:
        if c not in df_band_disp.columns:
            df_band_disp[c] = 0

    df_show = df_band_disp[needed].sort_values("strike", ascending=False).copy()

    # Delta health
    raw_delta = df_show["call_delta"].abs()
    delta_zeros = (raw_delta < 0.001).sum()
    delta_total = len(raw_delta)
    if delta_zeros > delta_total * 0.5:
        st.warning(f"⚠️ {delta_zeros}/{delta_total} strikes have delta≈0  Dhan greeks may be sparse.")
    else:
        st.success(f"✅ Delta populated for {delta_total - delta_zeros}/{delta_total} strikes  delta-weighted flow is active.")

    df_show.columns = ["Strike","C OI","C OI Chg","C Δ","C IV","P IV","P Δ","P OI Chg","P OI"]
    st.dataframe(df_show.style.format({
        "C OI":"{:,}","P OI":"{:,}","C OI Chg":"{:+,}","P OI Chg":"{:+,}",
        "C Δ":"{:.3f}","P Δ":"{:.3f}","C IV":"{:.1f}%","P IV":"{:.1f}%","Strike":"{:,}"
    }), width='stretch', hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER + AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style="text-align:center;font-size:12px;color:#6B7280;padding:8px 0;">
  <strong>Shantanu's Options Analysis  NIFTY 50 · NIFTY Futures</strong><br>
  Structural Band ±10 · Signal Band ±5 · Bias Engine · Strategy Engine<br>
  All timestamps in <strong>IST (Asia/Kolkata)</strong> ·
  {' LIVE data (Dhan API, refreshed every ' + str(_effective_refresh) + 's)' if USE_DHAN and mh else ' Demo mode' if USE_DEMO_MODE else ' After-hours cached data (refresh: ' + str(_effective_refresh) + 's)'}<br>
  Last updated: <strong>{payload['ts_ist']}</strong>
</div>
""", unsafe_allow_html=True)

# ─── Auto-refresh note ─────────────────────────────────────────────────────────
# CI #5 fix: st_autorefresh was previously registered here at the end of the
# script. It has been moved to the top (right after set_page_config) so it
# survives any st.stop() triggered by API failures. See L75 for the actual call.
# Page always refreshes every 60 s for ALL visitors.
# Data refresh cadence is governed by the server-side interval set in owner mode —
# independent of this.
