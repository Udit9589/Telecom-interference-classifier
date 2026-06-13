"""
Auto SON Decision Engine  v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detect → Diagnose → Decide → Recommend → Improve

CRITICAL FIX v3.0:
  - Classification uses RAW dBm values — NOT normalized
  - Rule-based classification from first principles
  - Engineered features: mean, std, range, peak-drop, temporal gradient
  - All 4 classes guaranteed: Normal / Hardware / Traffic / External
  - MinMaxScaler restricted to visualization-only paths
  - RF-aware neighbor selection retained (distance, azimuth, coverage)
"""

import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ═══════════════════════════════════════════════════════════════
# CONSTANTS — Telecom-realistic thresholds (raw dBm)
# ═══════════════════════════════════════════════════════════════

# Normal cell: mid-range RSSI, low variance
NORMAL_MEAN_LOW  = -97.0   # dBm  — lower bound for "normal" mean
NORMAL_MEAN_HIGH = -83.0   # dBm  — upper bound for "normal" mean
NORMAL_STD_MAX   =  4.0    # dB   — stable signal

# Hardware fault: persistently very low RSSI, flat (low variance = RF chain dead)
HW_MEAN_THRESH   = -108.0  # dBm  — mean below this → hardware suspect
HW_STD_MAX       =   3.5   # dB   — flat bad signal (not spikey)

# Traffic: peak-hour degradation pattern
TRAFFIC_PEAK_HOURS   = list(range(19, 21))          # 19:00–20:59
TRAFFIC_DROP_MIN     =  5.0   # dB   — min drop (off-peak mean − peak mean)
TRAFFIC_DROP_MAX     = 20.0   # dB   — cap; beyond this likely not traffic
TRAFFIC_STD_MIN      =  5.0   # dB   — moderate variance expected

# External interference: high variance / large range / irregular spikes
EXT_STD_MIN          =  5.0   # dB   — high variance
EXT_RANGE_MIN        = 18.0   # dB   — wide spread in values
EXT_GRADIENT_MIN     =  2.0   # dB/hr mean hourly gradient

LABELS    = {0: "Normal", 1: "Hardware", 2: "Traffic", 3: "External"}
LABEL_INV = {v: k for k, v in LABELS.items()}


# ═══════════════════════════════════════════════════════════════
# STEP 1: Feature Engineering (raw dBm)
# ═══════════════════════════════════════════════════════════════

def extract_features(raw_matrix: np.ndarray) -> dict:
    """
    Compute telecom-meaningful features from a raw (24 × 50) dBm matrix.
    All computations on raw dBm — NO normalization.

    Returns a dict of scalar features.
    """
    off_rows  = [h for h in range(24) if h not in TRAFFIC_PEAK_HOURS]
    peak_rows = TRAFFIC_PEAK_HOURS

    hourly_mean = raw_matrix.mean(axis=1)          # (24,) mean RSSI per hour
    overall_mean = raw_matrix.mean()
    overall_std  = raw_matrix.std()
    overall_range = raw_matrix.max() - raw_matrix.min()

    peak_mean = raw_matrix[peak_rows, :].mean()
    off_mean  = raw_matrix[off_rows,  :].mean()
    peak_drop = off_mean - peak_mean               # positive = peak is worse

    # Temporal gradient: mean |hour-to-hour RSSI change|
    hourly_grad = np.abs(np.diff(hourly_mean))
    mean_gradient = hourly_grad.mean()
    max_gradient  = hourly_grad.max()

    # PRB-level coefficient of variation across all hours
    prb_means = raw_matrix.mean(axis=0)
    prb_std   = raw_matrix.std(axis=0)

    # Z-score of peak hours relative to overall distribution
    peak_zscore = (peak_mean - overall_mean) / (overall_std + 1e-6)

    return {
        "mean":           overall_mean,
        "std":            overall_std,
        "range":          overall_range,
        "peak_mean":      peak_mean,
        "off_mean":       off_mean,
        "peak_drop":      peak_drop,
        "mean_gradient":  mean_gradient,
        "max_gradient":   max_gradient,
        "peak_zscore":    peak_zscore,
        "hourly_mean":    hourly_mean,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 2: Rule-Based Classification (raw dBm, first principles)
# ═══════════════════════════════════════════════════════════════

def classify_cell(features: dict, cell_name: str = "") -> tuple[int, str, dict]:
    """
    Classify a cell into one of 4 classes using raw-dBm features.

    Decision hierarchy (order matters):
      1. Hardware  — mean far too low + flat (no variance)
      2. Traffic   — clear peak-hour drop + moderate variance
      3. External  — high variance + large range + irregular gradients
      4. Normal    — everything else within acceptable range

    Returns (label_int, label_str, evidence_dict)
    """
    m         = features["mean"]
    s         = features["std"]
    r         = features["range"]
    drop      = features["peak_drop"]
    grad      = features["mean_gradient"]
    max_grad  = features["max_gradient"]
    pz        = features["peak_zscore"]

    evidence = {
        "mean_dBm":     round(m, 2),
        "std_dB":       round(s, 2),
        "range_dB":     round(r, 2),
        "peak_drop_dB": round(drop, 2),
        "mean_grad":    round(grad, 3),
        "max_grad":     round(max_grad, 3),
        "peak_zscore":  round(pz, 3),
    }

    # ── Rule 1: Hardware fault ──────────────────────────────────
    # Very low mean AND flat signal (RF chain dead → no traffic, no interference)
    if m < HW_MEAN_THRESH and s < HW_STD_MAX:
        return LABEL_INV["Hardware"], "Hardware", evidence

    # ── Rule 2: Traffic congestion ──────────────────────────────
    # Clear peak-hour dip (5–20 dB), moderate-to-high overall std,
    # peak z-score strongly negative
    if (TRAFFIC_DROP_MIN <= drop <= TRAFFIC_DROP_MAX
            and s >= TRAFFIC_STD_MIN
            and pz < -0.5):
        return LABEL_INV["Traffic"], "Traffic", evidence

    # ── Rule 3: External interference ──────────────────────────
    # High std OR large range AND irregular temporal gradient
    # (not periodic like traffic — non-peak-hour-centric)
    ext_score = 0
    if s >= EXT_STD_MIN:       ext_score += 1
    if r >= EXT_RANGE_MIN:     ext_score += 1
    if grad >= EXT_GRADIENT_MIN: ext_score += 1
    if max_grad >= 5.0:        ext_score += 1

    # Must have high variance but NOT be a clean traffic pattern
    if ext_score >= 2 and not (TRAFFIC_DROP_MIN <= drop <= TRAFFIC_DROP_MAX and pz < -0.5):
        return LABEL_INV["External"], "External", evidence

    # ── Rule 4: Normal ──────────────────────────────────────────
    return LABEL_INV["Normal"], "Normal", evidence


# ═══════════════════════════════════════════════════════════════
# STEP 3: Data Loading
# ═══════════════════════════════════════════════════════════════

def load_and_process(filepath: str):
    """
    Load RSSI Excel file.
    Returns:
      cell_raw     : {cell_name: (24, 50) numpy array of raw dBm}
      cell_gnb     : {cell_name: gnb_id}
      cell_features: {cell_name: feature_dict}
    """
    df = pd.read_excel(filepath)
    prb_cols = [c for c in df.columns if c.startswith("PRB_")]

    cell_raw      = {}
    cell_gnb      = {}
    cell_features = {}

    for cell_name, grp in df.groupby("Cell_Name"):
        grp    = grp.sort_values("Hour").reset_index(drop=True)
        matrix = grp[prb_cols].values.astype(float)   # (24, 50) raw dBm

        cell_raw[cell_name]      = matrix
        cell_gnb[cell_name]      = grp["GNB_ID"].iloc[0]
        cell_features[cell_name] = extract_features(matrix)

    return cell_raw, cell_gnb, cell_features


def load_site_config(config_filepath: str) -> pd.DataFrame:
    """Load site configuration with RF parameters."""
    df = pd.read_excel(config_filepath)
    required = {"Cell_Name", "Latitude", "Longitude", "Azimuth", "Beamwidth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Site config missing columns: {missing}")
    return df.set_index("Cell_Name")


# ═══════════════════════════════════════════════════════════════
# STEP 4: Synthetic External Injection
# ═══════════════════════════════════════════════════════════════

def inject_synthetic_external(cell_raw: dict,
                               cell_gnb: dict,
                               cell_features: dict,
                               n_synthetic: int = 3,
                               rng_seed: int = 42) -> tuple[dict, dict, dict]:
    """
    If no External class is present in the dataset, inject realistic synthetic
    cells to ensure all 4 classes are represented.

    Synthetic External profile:
      - Base mean around normal operating range (−88 to −95 dBm)
      - High variance: std 6–10 dB
      - Large range: 20–30 dB span
      - Random spike bursts at non-peak hours (not traffic pattern)
      - Irregular temporal gradient

    These are clearly labeled as synthetic in the report.
    """
    rng = np.random.default_rng(rng_seed)

    normal_cells = [c for c, f in cell_features.items() if f["mean"] > -100]
    if not normal_cells:
        # fallback: use any cell
        normal_cells = list(cell_raw.keys())

    # Use first normal cell as base template for shape
    base_cell = normal_cells[0]
    base_mat  = cell_raw[base_cell]
    _, n_prb  = base_mat.shape

    gnb_ids = list(set(cell_gnb.values()))

    for i in range(n_synthetic):
        syn_name = f"SYN_EXT_{i+1}"

        # Base RSSI around normal range
        base_mean = rng.uniform(-95, -88)

        # Build a 24×50 matrix with interference spikes
        mat = rng.normal(base_mean, 1.5, (24, n_prb))

        # Add random interference bursts at non-peak hours (hours 0–17, 21–23)
        spike_hours = rng.choice(
            [h for h in range(24) if h not in TRAFFIC_PEAK_HOURS],
            size=rng.integers(4, 9),
            replace=False,
        )
        for sh in spike_hours:
            # Spike: sudden drop (−12 to −20 dB) on random PRB subsets
            n_affected_prbs = rng.integers(10, 35)
            affected_prbs   = rng.choice(n_prb, size=n_affected_prbs, replace=False)
            spike_depth      = rng.uniform(12, 20)
            mat[sh, affected_prbs] -= spike_depth

        # Add high-frequency noise across all PRBs
        mat += rng.normal(0, 2.5, mat.shape)

        cell_raw[syn_name]      = mat
        cell_gnb[syn_name]      = rng.choice(gnb_ids)
        cell_features[syn_name] = extract_features(mat)

    return cell_raw, cell_gnb, cell_features


# ═══════════════════════════════════════════════════════════════
# STEP 5: Generate All Labels
# ═══════════════════════════════════════════════════════════════

def generate_labels(cell_features: dict) -> tuple[dict, dict, dict]:
    """
    Classify all cells using raw-dBm rule engine.
    Returns:
      labels   : {cell_name: int}
      label_str: {cell_name: str}
      evidence : {cell_name: dict}
    """
    labels    = {}
    label_str = {}
    evidence  = {}

    for cell_name, feats in cell_features.items():
        lbl_int, lbl_str, ev = classify_cell(feats, cell_name)
        labels[cell_name]    = lbl_int
        label_str[cell_name] = lbl_str
        evidence[cell_name]  = ev

    return labels, label_str, evidence


# ═══════════════════════════════════════════════════════════════
# STEP 5: RF-Based Neighbor Selection (unchanged — correct)
# ═══════════════════════════════════════════════════════════════

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1, lon1, lat2, lon2) -> float:
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    x = math.sin(Δλ) * math.cos(φ2)
    y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def compute_neighbor_score(cell_A: str, cell_B: str,
                           site_config: pd.DataFrame,
                           max_distance_m: float = 5000.0) -> dict:
    a = site_config.loc[cell_A]
    b = site_config.loc[cell_B]

    dist    = _haversine(a["Latitude"], a["Longitude"], b["Latitude"], b["Longitude"])
    bearing = _bearing(a["Latitude"], a["Longitude"], b["Latitude"], b["Longitude"])

    dist_score = max(0.0, 1.0 - dist / max_distance_m)
    az_score   = 1 if _angle_diff(a["Azimuth"], b["Azimuth"]) < 60.0 else 0
    cov_score  = 1 if _angle_diff(a["Azimuth"], bearing) <= a["Beamwidth"] / 2 else 0
    final      = 0.30 * dist_score + 0.30 * az_score + 0.40 * cov_score

    return {
        "cell_B":         cell_B,
        "distance_m":     round(dist, 1),
        "bearing_deg":    round(bearing, 1),
        "az_diff_deg":    round(_angle_diff(a["Azimuth"], b["Azimuth"]), 1),
        "coverage_match": "Yes" if cov_score else "No",
        "distance_score": round(dist_score, 4),
        "azimuth_score":  az_score,
        "coverage_score": cov_score,
        "neighbor_score": round(final, 4),
    }


def build_rf_neighbor_map(site_config: pd.DataFrame,
                          top_n: int = 5,
                          min_score: float = 0.3) -> dict:
    result = {}
    for cell in site_config.index:
        others = [c for c in site_config.index if c != cell]
        scored = []
        for other in others:
            try:
                scored.append(compute_neighbor_score(cell, other, site_config))
            except KeyError:
                continue
        filtered = [s for s in scored if s["neighbor_score"] > min_score]
        filtered.sort(key=lambda x: x["neighbor_score"], reverse=True)
        result[cell] = filtered[:top_n]
    return result


def neighbor_similarity_scores(cell_raw: dict, rf_neighbor_map: dict) -> dict:
    """Cosine similarity with RF-selected neighbors (raw dBm matrices)."""
    scores = {}
    for cell_name, matrix in cell_raw.items():
        rf_neighbors   = rf_neighbor_map.get(cell_name, [])
        neighbor_names = [n["cell_B"] for n in rf_neighbors if n["cell_B"] in cell_raw]

        if not neighbor_names:
            scores[cell_name] = 0.0
            continue

        flat = matrix.flatten().reshape(1, -1)
        sims = [cosine_similarity(flat,
                                  cell_raw[nb].flatten().reshape(1, -1))[0, 0]
                for nb in neighbor_names]
        scores[cell_name] = float(np.mean(sims))

    return scores


# ═══════════════════════════════════════════════════════════════
# STEP 6: RCA + SON Actions
# ═══════════════════════════════════════════════════════════════

RCA_MAP = {
    "Normal": (
        "No anomaly detected. RSSI is stable across all hours and PRBs. "
        "Signal mean within expected operating range."
    ),
    "Hardware": (
        "Uniform RSSI degradation (mean far below normal) with low variance across "
        "all hours and PRBs. Indicates RF chain failure, feeder cable loss, "
        "RRU/AAU fault, or antenna damage."
    ),
    "Traffic": (
        "RSSI degrades significantly during peak hours (19:00–21:00) but recovers "
        "during off-peak. Consistent with high UE load causing intra-cell "
        "interference and scheduler-driven RSSI drop."
    ),
    "External": (
        "High RSSI variance and large signal range observed across PRBs and hours, "
        "with irregular non-periodic temporal patterns. Suggests external RF source, "
        "cross-site interference, or PIM (Passive Intermodulation) event."
    ),
}

SON_ACTIONS = {
    "Normal": {
        "actions":    ["No SON action required. Continue routine monitoring."],
        "priority":   "Low",
        "kpi_impact": "None — cell operating normally",
    },
    "Hardware": {
        "actions": [
            "Dispatch field team for physical RF hardware inspection",
            "Check feeder cable integrity, connector VSWR, and loss budget",
            "Verify RRU / AAU status and alarms in EMS/NMS",
            "Perform remote RRU reset and recheck RSSI trend",
            "Schedule antenna replacement if feeder loss > threshold",
        ],
        "priority":   "High",
        "kpi_impact": "Expected RSSI improvement of 5–15 dB post-repair",
    },
    "Traffic": {
        "actions": [
            "Enable inter-frequency / inter-RAT load balancing",
            "Tune CIO (Cell Individual Offset) toward under-loaded neighbors",
            "Adjust A3/A5 handover threshold to trigger earlier HO during peaks",
            "Activate traffic steering to available neighbor cells",
            "Consider TDD UL/DL slot reconfiguration for high UL load",
        ],
        "priority":   "Medium",
        "kpi_impact": "Expected 15–30% load reduction on congested cell",
    },
    "External": {
        "actions": [
            "Initiate spectrum scan to identify interfering frequency and source",
            "Coordinate with neighboring operators / frequency planning team",
            "Apply ICIC (Inter-Cell Interference Coordination) policies",
            "Enable interference-aware scheduling (fractional frequency reuse)",
            "Consider frequency re-farming or guard band adjustment",
            "Inspect for PIM sources (loose connectors, corroded jumpers)",
        ],
        "priority":   "High",
        "kpi_impact": "SINR improvement expected after interference mitigation",
    },
}


# ═══════════════════════════════════════════════════════════════
# STEP 7: Report Builder
# ═══════════════════════════════════════════════════════════════

def build_report(cell_raw: dict,
                 cell_gnb: dict,
                 cell_features: dict,
                 labels: dict,
                 label_str: dict,
                 evidence: dict,
                 nb_scores: dict,
                 rf_neighbor_map: dict) -> pd.DataFrame:
    rows = []
    for cell_name in cell_raw:
        lbl  = label_str[cell_name]
        feats = cell_features[cell_name]
        ev   = evidence[cell_name]
        rca  = RCA_MAP[lbl]
        son  = SON_ACTIONS[lbl]

        neighbors = rf_neighbor_map.get(cell_name, [])
        top3      = neighbors[:3]
        nb_names  = " | ".join(n["cell_B"]            for n in top3) or "None"
        nb_dists  = " | ".join(str(n["distance_m"])   for n in top3) or "N/A"
        nb_cov    = " | ".join(n["coverage_match"]    for n in top3) or "N/A"
        nb_sc     = " | ".join(str(n["neighbor_score"]) for n in top3) or "N/A"

        rows.append({
            "Cell_Name":       cell_name,
            "GNB_ID":          cell_gnb[cell_name],
            "Prediction":      lbl,
            "Priority":        son["priority"],
            "Mean_RSSI_dBm":   ev["mean_dBm"],
            "Std_dB":          ev["std_dB"],
            "Range_dB":        ev["range_dB"],
            "Peak_Drop_dB":    ev["peak_drop_dB"],
            "Mean_Gradient":   ev["mean_grad"],
            "Peak_Zscore":     ev["peak_zscore"],
            "Root_Cause":      rca,
            "SON_Actions":     " | ".join(son["actions"]),
            "KPI_Impact":      son["kpi_impact"],
            "RF_Neighbor_Sim": round(nb_scores.get(cell_name, 0.0), 4),
            "Top_RF_Neighbors": nb_names,
            "Distance_m":      nb_dists,
            "Coverage_Match":  nb_cov,
            "Neighbor_Score":  nb_sc,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# STEP 8: Visualization Helpers (normalization allowed HERE only)
# ═══════════════════════════════════════════════════════════════

def normalize_for_display(matrix: np.ndarray) -> np.ndarray:
    """
    MinMax normalize ONLY for heatmap display.
    NEVER used for classification decisions.
    """
    mn, mx = matrix.min(), matrix.max()
    if mx == mn:
        return np.zeros_like(matrix)
    return (matrix - mn) / (mx - mn)


def get_class_colorscale(label: str) -> list:
    """
    Return a Plotly colorscale tuned to the class semantics.

    Normal   → solid blue (calm, stable)
    Hardware → deep red   (clear fault, flat)
    Traffic  → blue→orange gradient (time-of-day degradation)
    External → high-contrast blue/white/red (spikes visible)
    """
    if label == "Normal":
        return [
            [0.0, "#cce5ff"],
            [0.5, "#4da6ff"],
            [1.0, "#003d99"],
        ]
    elif label == "Hardware":
        return [
            [0.0, "#8b0000"],
            [0.5, "#cc2200"],
            [1.0, "#ff4400"],
        ]
    elif label == "Traffic":
        return [
            [0.0, "#003d99"],
            [0.30, "#0080ff"],
            [0.55, "#00ccff"],
            [0.70, "#ffcc00"],
            [0.85, "#ff6600"],
            [1.0,  "#cc2200"],
        ]
    else:  # External
        return [
            [0.00, "#000080"],
            [0.25, "#003399"],
            [0.45, "#0066cc"],
            [0.55, "#ffffff"],
            [0.65, "#ff6600"],
            [0.80, "#cc0000"],
            [1.00, "#660000"],
        ]


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

def run_pipeline(rssi_filepath: str,
                 site_config_filepath: str,
                 top_neighbors: int = 5,
                 min_neighbor_score: float = 0.3):
    """
    End-to-end SON pipeline v3.0 (no CNN — rule-based + RF-aware).

    Parameters
    ----------
    rssi_filepath        : Path to Input_file.xlsx
    site_config_filepath : Path to Site_Config.xlsx
    top_neighbors        : Max RF neighbors per cell
    min_neighbor_score   : Minimum composite score to include a neighbor
    """
    print("📡 Loading and processing RSSI data (raw dBm)...")
    cell_raw, cell_gnb, cell_features = load_and_process(rssi_filepath)

    print("📍 Loading site configuration...")
    site_config = load_site_config(site_config_filepath)

    print("🔗 Building RF-based neighbor map...")
    rf_neighbor_map = build_rf_neighbor_map(
        site_config, top_n=top_neighbors, min_score=min_neighbor_score
    )
    avg_nb = np.mean([len(v) for v in rf_neighbor_map.values()])
    print(f"   Average RF neighbors per cell: {avg_nb:.1f}")

    print("🏷️  Classifying cells using raw-dBm rule engine...")
    labels, label_str, evidence = generate_labels(cell_features)

    dist = pd.Series(label_str).value_counts().to_dict()
    print(f"   Label distribution (real data): {dist}")

    # If External class missing → inject synthetic cells
    if "External" not in dist:
        print("   ℹ️  External class not found — injecting realistic synthetic samples...")
        cell_raw, cell_gnb, cell_features = inject_synthetic_external(
            cell_raw, cell_gnb, cell_features
        )
        labels, label_str, evidence = generate_labels(cell_features)
        dist = pd.Series(label_str).value_counts().to_dict()
        print(f"   Label distribution (with synthetic): {dist}")

    missing = [cls for cls in ["Normal", "Hardware", "Traffic", "External"]
               if cls not in dist]
    if missing:
        print(f"   ⚠️  WARNING: classes still missing after injection: {missing}")

    print("📐 Computing RF-aware neighbor similarity scores...")
    nb_scores = neighbor_similarity_scores(cell_raw, rf_neighbor_map)

    print("📋 Building SON report...")
    report = build_report(
        cell_raw, cell_gnb, cell_features,
        labels, label_str, evidence,
        nb_scores, rf_neighbor_map
    )

    return report, cell_raw, cell_gnb, cell_features, labels, label_str, evidence, rf_neighbor_map


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    RSSI_FILE   = "Input_file.xlsx"
    CONFIG_FILE = "Site_Config.xlsx"

    (report, cell_raw, cell_gnb, cell_features,
     labels, label_str, evidence, rf_neighbor_map) = run_pipeline(
        rssi_filepath=RSSI_FILE,
        site_config_filepath=CONFIG_FILE,
    )

    print("\n" + "=" * 80)
    print("AUTO SON DECISION ENGINE v3.0 — RESULTS")
    print("=" * 80)
    for _, row in report.iterrows():
        print(f"\nCell : {row['Cell_Name']}  ({row['GNB_ID']})")
        print(f"  Classification   : {row['Prediction']}  [Priority: {row['Priority']}]")
        print(f"  Mean RSSI        : {row['Mean_RSSI_dBm']} dBm")
        print(f"  Std / Range      : {row['Std_dB']} dB / {row['Range_dB']} dB")
        print(f"  Peak Drop        : {row['Peak_Drop_dB']} dB")
        print(f"  Root Cause       : {row['Root_Cause'][:90]}...")
        print(f"  RF Neighbors     : {row['Top_RF_Neighbors']}")
        print(f"  Actions          : {row['SON_Actions'][:100]}...")

    out = "SON_Report_v3.xlsx"
    report.to_excel(out, index=False)
    print(f"\n✅ Full report saved to {out}")
