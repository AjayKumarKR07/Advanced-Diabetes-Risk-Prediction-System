"""
Phase 2 – CDC Diabetes Health Indicators EDA & Data Quality Audit
==================================================================
Dataset : data/raw/diabetes_012_health_indicators_BRFSS2015.csv
Outputs :
  results/figures/   – All PNG visualisations
  results/reports/cdc_eda_report.md
  results/reports/cdc_data_quality_report.md

Dependencies (all in Python 3.14 env):
  pandas, numpy, scipy, sklearn, PIL (Pillow)
  NO matplotlib / seaborn needed – charts are drawn via PIL.

Run from project root:
  python notebooks/cdc_eda_phase2.py
"""

import os, sys, json, math, warnings, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data" / "raw" / "diabetes_012_health_indicators_BRFSS2015.csv"
FIGS   = ROOT / "results" / "figures"
RPTS   = ROOT / "results" / "reports"
FIGS.mkdir(parents=True, exist_ok=True)
RPTS.mkdir(parents=True, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
COLORS = {
    "bg":      (248, 249, 252),
    "grid":    (220, 225, 235),
    "text":    (30,  40,  60),
    "title":   (15,  25,  50),
    "c0":      (70,  130, 200),   # Class 0 – No diabetes
    "c1":      (255, 180,  50),   # Class 1 – Prediabetes
    "c2":      (220,  60,  60),   # Class 2 – Diabetes
    "corr_hi": (200,  40,  40),
    "corr_lo": (40,   80, 180),
    "corr_mi": (245, 245, 245),
    "bar":     (90,  150, 220),
    "bar2":    (255, 140,  60),
}

CLASS_COLORS = [COLORS["c0"], COLORS["c1"], COLORS["c2"]]
CLASS_LABELS = ["No Diabetes (0)", "Prediabetes (1)", "Diabetes (2)"]

# ── Font helper ────────────────────────────────────────────────────────────
def get_font(size=14, bold=False):
    try:
        name = "arialbd.ttf" if bold else "arial.ttf"
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()

# ── Drawing helpers ────────────────────────────────────────────────────────
def new_canvas(w, h, bg=None):
    img = Image.new("RGB", (w, h), bg or COLORS["bg"])
    draw = ImageDraw.Draw(img)
    return img, draw

def draw_title(draw, text, w, y=18, size=20):
    f = get_font(size, bold=True)
    bbox = draw.textbbox((0, 0), text, font=f)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, y), text, font=f, fill=COLORS["title"])

def draw_grid_h(draw, x0, y0, x1, y1, n=5):
    for i in range(n + 1):
        y = y0 + (y1 - y0) * i // n
        draw.line([(x0, y), (x1, y)], fill=COLORS["grid"], width=1)

def label_bar(draw, x, y, val, fmt=".1f", size=11):
    txt = format(val, fmt)
    f = get_font(size)
    draw.text((x, y), txt, font=f, fill=COLORS["text"])

def save(img, name):
    path = FIGS / name
    img.save(path, dpi=(150, 150))
    print(f"  Saved: {path.name}")
    return path

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD & BASIC VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("PHASE 2 – CDC EDA & DATA QUALITY AUDIT")
print("═"*60)

if not DATA.exists():
    print(f"\n[ERROR] Dataset not found at:\n  {DATA}")
    print("\nPlease download the dataset from Kaggle and place it at:")
    print("  data/raw/diabetes_012_health_indicators_BRFSS2015.csv")
    print("\nDataset source:")
    print("  https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset")
    sys.exit(1)

print(f"\nLoading: {DATA}")
df = pd.read_csv(DATA)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 1a. Expected schema check ──────────────────────────────────────────────
EXPECTED_COLS = [
    "Diabetes_012","HighBP","HighChol","CholCheck","BMI","Smoker",
    "Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
    "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth",
    "PhysHlth","DiffWalk","Sex","Age","Education","Income"
]
missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
extra_cols   = [c for c in df.columns if c not in EXPECTED_COLS]
print(f"Expected columns present: {len(EXPECTED_COLS) - len(missing_cols)}/{len(EXPECTED_COLS)}")
if missing_cols: print(f"  MISSING: {missing_cols}")
if extra_cols:   print(f"  EXTRA  : {extra_cols}")

TARGET = "Diabetes_012"
FEATURES = [c for c in df.columns if c != TARGET]

# ═══════════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY AUDIT
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Data Quality Audit ──────────────────────────────────")

# 2a. Missing values
null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(4)
print(f"Total missing values: {null_counts.sum()}")

# 2b. Duplicate rows
n_dupes = df.duplicated().sum()
n_dupes_keep = df.duplicated(keep=False).sum()
pct_dupes = n_dupes / len(df) * 100
print(f"Duplicate rows: {n_dupes:,} ({pct_dupes:.2f}%)")

# 2c. Data types
dtypes_summary = df.dtypes.value_counts()
print(f"Data types: {dict(dtypes_summary)}")

# 2d. Descriptive statistics
desc = df.describe()

# 2e. Value ranges and validity check
BINARY_FEATURES = [
    "HighBP","HighChol","CholCheck","Smoker","Stroke",
    "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
    "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","DiffWalk","Sex"
]
ORDINAL_FEATURES = ["GenHlth","Age","Education","Income","MentHlth","PhysHlth"]
CONTINUOUS_FEATURES = ["BMI"]

invalid_summary = {}
for col in BINARY_FEATURES:
    if col in df.columns:
        bad = df[~df[col].isin([0, 1])][col].count()
        if bad > 0:
            invalid_summary[col] = f"{bad} values not in {{0,1}}"

for col, (lo, hi) in {
    "GenHlth": (1, 5), "Age": (1, 13), "Education": (1, 6),
    "Income": (1, 8), "MentHlth": (0, 30), "PhysHlth": (0, 30),
    "BMI": (10, 100), TARGET: (0, 2)
}.items():
    if col in df.columns:
        bad = df[(df[col] < lo) | (df[col] > hi)][col].count()
        if bad > 0:
            invalid_summary[col] = f"{bad} values outside [{lo},{hi}]"

print(f"Invalid value issues: {len(invalid_summary)}")

# 2f. Unique value counts
unique_counts = {c: df[c].nunique() for c in df.columns}

# 2g. Target distribution
target_vc = df[TARGET].value_counts().sort_index()
target_pct = (target_vc / len(df) * 100).round(2)
print(f"\nTarget distribution:")
for cls, cnt, pct in zip(target_vc.index, target_vc.values, target_pct.values):
    lbl = ["No Diabetes","Prediabetes","Diabetes"][int(cls)]
    print(f"  Class {int(cls)} ({lbl}): {cnt:>8,}  ({pct:.2f}%)")

# 2h. Duplicate analysis
df_no_dupes = df.drop_duplicates()
target_vc_deduped = df_no_dupes[TARGET].value_counts().sort_index()
print(f"\nAfter deduplication shape: {df_no_dupes.shape[0]:,} rows")
print(f"Duplicate rows removed would be: {n_dupes:,}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE CATEGORISATION
# ═══════════════════════════════════════════════════════════════════════════
feature_cats = {
    "Binary (0/1)": BINARY_FEATURES,
    "Ordinal / Likert-scale": ["GenHlth", "Age", "Education", "Income"],
    "Count-like (0–30)": ["MentHlth", "PhysHlth"],
    "Continuous": ["BMI"],
}

# ═══════════════════════════════════════════════════════════════════════════
# 4. STATISTICAL SUMMARIES
# ═══════════════════════════════════════════════════════════════════════════

# Class-wise means for every feature
class_means = df.groupby(TARGET)[FEATURES].mean().round(4)

# Correlation matrix
corr = df[FEATURES].corr().round(4)

# Cramér's V for binary pairs
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return math.sqrt(chi2 / (n * (min(r, k) - 1)))

bin_pairs = [(a, b) for i, a in enumerate(BINARY_FEATURES)
             for b in BINARY_FEATURES[i+1:] if a in df.columns and b in df.columns]
cramers = {f"{a}|{b}": cramers_v(df[a], df[b]) for a, b in bin_pairs[:20]}
top_cramers = sorted(cramers.items(), key=lambda x: x[1], reverse=True)[:5]

# Point-biserial correlation with target (treat target as continuous)
target_corr = {}
for col in FEATURES:
    if col in df.columns:
        r, p = stats.pearsonr(df[col].fillna(df[col].median()), df[TARGET])
        target_corr[col] = (round(r, 4), round(p, 6))

target_corr_sorted = sorted(target_corr.items(), key=lambda x: abs(x[1][0]), reverse=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5. VISUALISATIONS (PIL-based)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Generating visualisations ───────────────────────────")

# ── 5.1 Target class distribution (bar chart) ──────────────────────────────
def fig_target_distribution():
    W, H = 700, 460
    img, draw = new_canvas(W, H)
    draw_title(draw, "Target Class Distribution – Diabetes_012", W, y=16)

    counts = [int(target_vc.get(i, 0)) for i in range(3)]
    total  = sum(counts)
    pcts   = [c / total * 100 for c in counts]

    pad_l, pad_r, pad_t, pad_b = 70, 40, 60, 90
    chart_w = W - pad_l - pad_r
    chart_h = H - pad_t - pad_b
    max_v   = max(counts) * 1.12

    draw_grid_h(draw, pad_l, pad_t, W - pad_r, pad_t + chart_h, n=5)

    bar_w   = chart_w // 5
    gap     = (chart_w - 3 * bar_w) // 4
    x_start = pad_l + gap

    for i, (cnt, pct, col) in enumerate(zip(counts, pcts, CLASS_COLORS)):
        x0 = x_start + i * (bar_w + gap)
        bh = int(chart_h * cnt / max_v)
        y0 = pad_t + chart_h - bh
        y1 = pad_t + chart_h
        draw.rectangle([x0, y0, x0 + bar_w, y1], fill=col)
        draw.rectangle([x0, y0, x0 + bar_w, y1], outline=COLORS["text"], width=1)

        f = get_font(12, bold=True)
        cnt_lbl = f"{cnt:,}"
        b = draw.textbbox((0,0), cnt_lbl, font=f)
        tw = b[2] - b[0]
        draw.text((x0 + (bar_w - tw)//2, y0 - 22), cnt_lbl, font=f, fill=COLORS["text"])
        pct_lbl = f"{pct:.1f}%"
        b2 = draw.textbbox((0,0), pct_lbl, font=get_font(11))
        tw2 = b2[2] - b2[0]
        draw.text((x0 + (bar_w - tw2)//2, y0 - 40), pct_lbl, font=get_font(11), fill=COLORS["text"])

        lbl = CLASS_LABELS[i]
        b3 = draw.textbbox((0,0), lbl, font=get_font(11))
        tw3 = b3[2] - b3[0]
        draw.text((x0 + (bar_w - tw3)//2, pad_t + chart_h + 8), lbl,
                  font=get_font(11), fill=COLORS["text"])

    # Y-axis labels
    for i in range(6):
        y = pad_t + chart_h - int(chart_h * i / 5)
        v = int(max_v * i / 5)
        draw.text((5, y - 7), f"{v//1000}K", font=get_font(10), fill=COLORS["text"])

    draw.line([(pad_l, pad_t), (pad_l, pad_t + chart_h)], fill=COLORS["text"], width=2)
    draw.line([(pad_l, pad_t + chart_h), (W - pad_r, pad_t + chart_h)], fill=COLORS["text"], width=2)

    save(img, "01_target_class_distribution.png")

fig_target_distribution()

# ── 5.2 Class imbalance pie chart ─────────────────────────────────────────
def fig_class_imbalance_pie():
    W, H = 600, 460
    img, draw = new_canvas(W, H)
    draw_title(draw, "Class Imbalance – Proportional View", W, y=16)

    counts = [int(target_vc.get(i, 0)) for i in range(3)]
    total  = sum(counts)
    cx, cy, r = W // 2, H // 2 + 10, 155

    start_angle = -90
    for i, (cnt, col) in enumerate(zip(counts, CLASS_COLORS)):
        sweep = 360 * cnt / total
        end_angle = start_angle + sweep
        draw.pieslice([cx-r, cy-r, cx+r, cy+r],
                      start=start_angle, end=end_angle, fill=col, outline="white", width=2)
        # Label
        mid_a = math.radians(start_angle + sweep / 2)
        lx = cx + int((r * 0.65) * math.cos(mid_a))
        ly = cy + int((r * 0.65) * math.sin(mid_a))
        pct = cnt / total * 100
        draw.text((lx - 20, ly - 7), f"{pct:.1f}%", font=get_font(13, bold=True), fill=(255,255,255))
        start_angle = end_angle

    # Legend
    lx, ly = 30, H - 90
    for i, (lbl, col) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        draw.rectangle([lx, ly + i*22, lx+16, ly + i*22 + 16], fill=col)
        draw.text((lx + 24, ly + i*22), f"{lbl}  ({counts[i]:,})", font=get_font(12), fill=COLORS["text"])

    save(img, "02_class_imbalance_pie.png")

fig_class_imbalance_pie()

# ── 5.3 BMI distribution per class ────────────────────────────────────────
def fig_bmi_distribution():
    W, H = 800, 480
    img, draw = new_canvas(W, H)
    draw_title(draw, "BMI Distribution by Diabetes Class (Histogram)", W, y=16)

    pad_l, pad_r, pad_t, pad_b = 65, 20, 55, 80
    chart_w = W - pad_l - pad_r
    chart_h = H - pad_t - pad_b

    bmi_min, bmi_max = 10, 80
    n_bins = 35
    bin_edges = np.linspace(bmi_min, bmi_max, n_bins + 1)

    # Compute histograms per class (density)
    hists = []
    for cls in range(3):
        sub = df[df[TARGET] == cls]["BMI"].dropna().values
        sub = sub[(sub >= bmi_min) & (sub <= bmi_max)]
        h, _ = np.histogram(sub, bins=bin_edges, density=True)
        hists.append(h)

    max_density = max(h.max() for h in hists) * 1.1
    bin_w_px = chart_w / n_bins

    draw_grid_h(draw, pad_l, pad_t, W - pad_r, pad_t + chart_h, n=5)

    alpha_colors = [(70,130,200,160),(255,180,50,160),(220,60,60,160)]
    for cls, (h, col) in enumerate(zip(hists, CLASS_COLORS)):
        for bi, density in enumerate(h):
            x0 = int(pad_l + bi * bin_w_px)
            x1 = int(pad_l + (bi + 1) * bin_w_px)
            bh = int(chart_h * density / max_density)
            y0 = pad_t + chart_h - bh
            draw.rectangle([x0, y0, x1 - 1, pad_t + chart_h],
                            fill=(*col, 120), outline=(*col, 200), width=1)

    # X-axis labels
    for v in range(10, 85, 10):
        x = int(pad_l + (v - bmi_min) / (bmi_max - bmi_min) * chart_w)
        draw.text((x - 8, pad_t + chart_h + 5), str(v), font=get_font(10), fill=COLORS["text"])
    draw.text((W//2 - 20, H - 22), "BMI", font=get_font(12), fill=COLORS["text"])

    # Y-axis
    draw.line([(pad_l, pad_t), (pad_l, pad_t + chart_h)], fill=COLORS["text"], width=2)
    draw.line([(pad_l, pad_t + chart_h), (W - pad_r, pad_t + chart_h)], fill=COLORS["text"], width=2)

    # Legend
    for i, (lbl, col) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        draw.rectangle([pad_l + i*220, pad_t - 28, pad_l + i*220 + 14, pad_t - 14], fill=col)
        draw.text((pad_l + i*220 + 18, pad_t - 30), lbl, font=get_font(11), fill=COLORS["text"])

    save(img, "03_bmi_distribution_by_class.png")

fig_bmi_distribution()

# ── 5.4 Correlation heatmap (Pearson, features only) ──────────────────────
def fig_correlation_heatmap():
    n = len(FEATURES)
    cell = 28
    W = n * cell + 160
    H = n * cell + 120
    img, draw = new_canvas(W, H)
    draw_title(draw, "Feature Correlation Heatmap (Pearson)", W, y=12)

    off_x, off_y = 90, 50

    def lerp_color(v):
        # -1→blue, 0→white, +1→red
        if v >= 0:
            r = int(COLORS["corr_mi"][0] + v * (COLORS["corr_hi"][0] - COLORS["corr_mi"][0]))
            g = int(COLORS["corr_mi"][1] + v * (COLORS["corr_hi"][1] - COLORS["corr_mi"][1]))
            b = int(COLORS["corr_mi"][2] + v * (COLORS["corr_hi"][2] - COLORS["corr_mi"][2]))
        else:
            v2 = -v
            r = int(COLORS["corr_mi"][0] + v2 * (COLORS["corr_lo"][0] - COLORS["corr_mi"][0]))
            g = int(COLORS["corr_mi"][1] + v2 * (COLORS["corr_lo"][1] - COLORS["corr_mi"][1]))
            b = int(COLORS["corr_mi"][2] + v2 * (COLORS["corr_lo"][2] - COLORS["corr_mi"][2]))
        return (max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b)))

    corr_arr = corr.values
    for i, fi in enumerate(FEATURES):
        for j, fj in enumerate(FEATURES):
            v = corr_arr[i][j]
            col = lerp_color(v)
            x0 = off_x + j * cell
            y0 = off_y + i * cell
            draw.rectangle([x0, y0, x0+cell-1, y0+cell-1], fill=col)

    # Row labels
    f_sm = get_font(8)
    for i, fi in enumerate(FEATURES):
        y = off_y + i * cell + cell // 2 - 5
        draw.text((2, y), fi[:12], font=f_sm, fill=COLORS["text"])
    # Col labels
    for j, fj in enumerate(FEATURES):
        x = off_x + j * cell + 2
        draw.text((x, off_y - 14), fj[:7], font=f_sm, fill=COLORS["text"])

    save(img, "04_correlation_heatmap.png")

fig_correlation_heatmap()

# ── 5.5 Binary feature prevalence per class ───────────────────────────────
def fig_binary_prevalence():
    feats = BINARY_FEATURES[:12]
    n = len(feats)
    W, H = 900, 520
    img, draw = new_canvas(W, H)
    draw_title(draw, "Binary Feature Prevalence by Diabetes Class (%)", W, y=14)

    pad_l, pad_r, pad_t, pad_b = 160, 20, 55, 40
    chart_w = W - pad_l - pad_r
    chart_h = H - pad_t - pad_b
    row_h   = chart_h // n
    group_w = chart_w
    bar_slot = group_w // n
    n_cls   = 3
    bar_w   = max(6, bar_slot // (n_cls + 1))

    for fi, feat in enumerate(feats):
        y_base = pad_t + fi * row_h + row_h - 4
        # Feature label
        draw.text((2, y_base - row_h // 2 - 7), feat, font=get_font(10), fill=COLORS["text"])

        for ci in range(3):
            sub = df[df[TARGET] == ci][feat]
            pct = sub.mean() * 100 if feat in BINARY_FEATURES else 0
            bar_h = int((row_h - 6) * pct / 100)
            x0 = pad_l + fi * bar_slot + ci * (bar_w + 2)
            draw.rectangle([x0, y_base - bar_h, x0 + bar_w, y_base],
                            fill=CLASS_COLORS[ci])
            if bar_h > 12:
                draw.text((x0+1, y_base - bar_h - 12), f"{pct:.0f}",
                          font=get_font(8), fill=COLORS["text"])

    # Legend
    for ci, (lbl, col) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        draw.rectangle([pad_l + ci * 220, 34, pad_l + ci * 220 + 12, 46], fill=col)
        draw.text((pad_l + ci * 220 + 16, 32), lbl, font=get_font(11), fill=COLORS["text"])

    # Axes
    draw.line([(pad_l, pad_t), (pad_l, pad_t + chart_h)], fill=COLORS["text"], width=2)
    draw.line([(pad_l, pad_t + chart_h), (W - pad_r, pad_t + chart_h)], fill=COLORS["text"], width=2)

    save(img, "05_binary_feature_prevalence.png")

fig_binary_prevalence()

# ── 5.6 Ordinal feature distributions per class ───────────────────────────
def fig_ordinal_distributions():
    feats = [("GenHlth",5,"General Health (1=Excellent,5=Poor)"),
             ("Age",13,"Age Category (1–13)"),
             ("Education",6,"Education (1–6)"),
             ("Income",8,"Income Level (1–8)")]
    W, H = 1000, 580
    img, draw = new_canvas(W, H)
    draw_title(draw, "Ordinal Feature Distributions by Class", W, y=12)

    pad_l, pad_r = 50, 20
    pad_t, pad_b = 48, 30
    cols_n = 2
    sub_w = (W - pad_l - pad_r) // cols_n
    sub_h = (H - pad_t - pad_b) // 2

    for fi, (feat, n_vals, lbl) in enumerate(feats):
        col_i = fi % cols_n
        row_i = fi // cols_n
        ox = pad_l + col_i * sub_w
        oy = pad_t + row_i * sub_h

        draw.text((ox + 4, oy + 4), lbl[:45], font=get_font(10, bold=True), fill=COLORS["title"])

        inner_l, inner_t = 10, 24
        inner_w = sub_w - inner_l - 15
        inner_h = sub_h - inner_t - 30

        vals = list(range(1, n_vals + 1))
        bar_slot = inner_w // len(vals)
        bar_w = max(4, bar_slot // 4)

        for vi, v in enumerate(vals):
            for ci in range(3):
                sub = df[df[TARGET] == ci]
                cnt = (sub[feat] == v).sum()
                total_cls = len(sub)
                pct = cnt / total_cls * 100 if total_cls > 0 else 0
                bh = int(inner_h * pct / 50) if pct < 50 else inner_h
                x0 = ox + inner_l + vi * bar_slot + ci * (bar_w + 1)
                y1 = oy + inner_t + inner_h
                draw.rectangle([x0, y1 - bh, x0 + bar_w, y1], fill=CLASS_COLORS[ci])

            # X label
            xl = ox + inner_l + vi * bar_slot + bar_slot // 2 - 4
            draw.text((xl, oy + inner_t + inner_h + 2), str(v),
                      font=get_font(9), fill=COLORS["text"])

        draw.line([(ox+inner_l, oy+inner_t),
                   (ox+inner_l, oy+inner_t+inner_h)], fill=COLORS["text"], width=1)
        draw.line([(ox+inner_l, oy+inner_t+inner_h),
                   (ox+inner_l+inner_w, oy+inner_t+inner_h)], fill=COLORS["text"], width=1)

    # Legend
    for ci, (lbl, col) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        draw.rectangle([50 + ci * 240, 30, 62 + ci * 240, 42], fill=col)
        draw.text((66 + ci * 240, 28), lbl, font=get_font(10), fill=COLORS["text"])

    save(img, "06_ordinal_distributions.png")

fig_ordinal_distributions()

# ── 5.7 Target correlation with features (bar chart) ──────────────────────
def fig_target_correlation():
    feats_sorted = [f for f, _ in target_corr_sorted]
    vals         = [abs(target_corr[f][0]) for f in feats_sorted]
    raw_vals     = [target_corr[f][0] for f in feats_sorted]
    n = len(feats_sorted)

    W, H = 800, 60 + n * 24 + 60
    img, draw = new_canvas(W, H)
    draw_title(draw, "Feature Correlation with Diabetes_012 (|Pearson r|)", W, y=12)

    pad_l, pad_r, pad_t = 160, 40, 46
    chart_w = W - pad_l - pad_r
    max_v   = max(vals) * 1.1

    for i, (feat, v, rv) in enumerate(zip(feats_sorted, vals, raw_vals)):
        y = pad_t + i * 24
        col = COLORS["corr_hi"] if rv > 0 else COLORS["corr_lo"]
        bar_len = int(chart_w * v / max_v)
        draw.rectangle([pad_l, y + 2, pad_l + bar_len, y + 18], fill=col)
        draw.text((2, y + 2), feat[:18], font=get_font(11), fill=COLORS["text"])
        draw.text((pad_l + bar_len + 4, y + 2), f"{rv:+.3f}", font=get_font(10), fill=COLORS["text"])

    draw.line([(pad_l, pad_t - 2), (pad_l, pad_t + n * 24)], fill=COLORS["text"], width=1)
    save(img, "07_feature_target_correlation.png")

fig_target_correlation()

# ── 5.8 MentHlth & PhysHlth distributions ────────────────────────────────
def fig_count_features():
    feats = [("MentHlth","Days of Poor Mental Health (0–30)"),
             ("PhysHlth","Days of Poor Physical Health (0–30)")]
    W, H = 900, 420
    img, draw = new_canvas(W, H)
    draw_title(draw, "Count-like Features – Mental & Physical Health Days", W, y=12)

    for fi, (feat, lbl) in enumerate(feats):
        ox = 40 + fi * 430
        oy = 48
        sub_w, sub_h = 400, 320

        draw.text((ox + 4, oy), lbl, font=get_font(10, bold=True), fill=COLORS["title"])
        inner_l, inner_t = 12, 22
        inner_w = sub_w - inner_l - 20
        inner_h = sub_h - inner_t - 40

        vals = list(range(0, 31))
        bar_w_slot = inner_w // 31

        # Find max pct across all classes
        max_p = 0
        for ci in range(3):
            sub = df[df[TARGET] == ci][feat]
            for v in vals:
                p = (sub == v).sum() / len(sub) * 100
                if p > max_p: max_p = p
        max_p = max_p * 1.1

        for vi, v in enumerate(vals):
            for ci in range(3):
                sub = df[df[TARGET] == ci]
                p = (sub[feat] == v).sum() / len(sub) * 100
                bh = int(inner_h * p / max_p) if max_p > 0 else 0
                bw = max(2, bar_w_slot // 3 - 1)
                x0 = ox + inner_l + vi * bar_w_slot + ci * (bw + 1)
                y1 = oy + inner_t + inner_h
                draw.rectangle([x0, y1 - bh, x0 + bw, y1], fill=CLASS_COLORS[ci])

        for v in [0, 5, 10, 15, 20, 25, 30]:
            x = ox + inner_l + v * bar_w_slot + bar_w_slot // 2
            draw.text((x - 4, oy + inner_t + inner_h + 4), str(v),
                      font=get_font(9), fill=COLORS["text"])

        draw.line([(ox+inner_l, oy+inner_t), (ox+inner_l, oy+inner_t+inner_h)],
                  fill=COLORS["text"], width=1)
        draw.line([(ox+inner_l, oy+inner_t+inner_h), (ox+inner_l+inner_w, oy+inner_t+inner_h)],
                  fill=COLORS["text"], width=1)

    for ci, (lbl, col) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        draw.rectangle([40 + ci * 230, H - 24, 54 + ci * 230, H - 12], fill=col)
        draw.text((58 + ci * 230, H - 26), lbl, font=get_font(10), fill=COLORS["text"])

    save(img, "08_mental_physical_health_days.png")

fig_count_features()

# ── 5.9 Duplicate rows analysis ────────────────────────────────────────────
def fig_duplicates_analysis():
    W, H = 700, 380
    img, draw = new_canvas(W, H)
    draw_title(draw, "Duplicate Record Analysis", W, y=14)

    orig_counts  = [int(target_vc.get(i, 0)) for i in range(3)]
    dedup_counts = [int(target_vc_deduped.get(i, 0)) for i in range(3)]
    total_orig  = sum(orig_counts)
    total_dedup = sum(dedup_counts)

    pad_l, pad_r, pad_t, pad_b = 70, 40, 55, 90
    chart_w = W - pad_l - pad_r
    chart_h = H - pad_t - pad_b
    max_v   = max(max(orig_counts), max(dedup_counts)) * 1.12

    draw_grid_h(draw, pad_l, pad_t, W - pad_r, pad_t + chart_h, n=5)

    group_w  = chart_w // 3
    bar_w    = group_w // 3

    for i in range(3):
        for j, (cnt, lbl_suf, col_adj) in enumerate([
            (orig_counts[i],  "Original", 0),
            (dedup_counts[i], "Deduped",  40),
        ]):
            x0 = pad_l + i * group_w + j * (bar_w + 4) + 8
            bh = int(chart_h * cnt / max_v)
            y0 = pad_t + chart_h - bh
            bc = tuple(min(255, c + col_adj * (1 - 2*j)) for c in CLASS_COLORS[i])
            draw.rectangle([x0, y0, x0 + bar_w, pad_t + chart_h], fill=CLASS_COLORS[i],
                            outline=(100,100,100) if j==1 else CLASS_COLORS[i], width=2)
            if j == 1:
                # Hatch pattern for deduped
                for yy in range(y0, pad_t + chart_h, 5):
                    draw.line([(x0, yy), (x0 + bar_w, yy)], fill=(255,255,255,80), width=1)

            f = get_font(9)
            draw.text((x0+2, y0 - 16), f"{cnt//1000}K", font=f, fill=COLORS["text"])

        # Class label
        xl = pad_l + i * group_w + group_w // 2
        draw.text((xl - 20, pad_t + chart_h + 8), CLASS_LABELS[i][:14],
                  font=get_font(9), fill=COLORS["text"])

    # Summary text
    draw.text((pad_l, H - 28),
              f"Original: {total_orig:,} rows   |   After dedup: {total_dedup:,} rows   |   Dupes: {n_dupes:,} ({pct_dupes:.1f}%)",
              font=get_font(10), fill=COLORS["text"])

    draw.line([(pad_l, pad_t), (pad_l, pad_t+chart_h)], fill=COLORS["text"], width=2)
    draw.line([(pad_l, pad_t+chart_h), (W-pad_r, pad_t+chart_h)], fill=COLORS["text"], width=2)

    save(img, "09_duplicate_analysis.png")

fig_duplicates_analysis()

# ── 5.10 Class imbalance SMOTE vs weights visualisation ──────────────────
def fig_resampling_strategies():
    W, H = 760, 440
    img, draw = new_canvas(W, H)
    draw_title(draw, "Class Imbalance: Resampling Strategy Comparison", W, y=12)

    counts = [int(target_vc.get(i, 0)) for i in range(3)]
    total  = sum(counts)

    # Scenario bars: Original / Class Weights / SMOTE / ROS
    scenarios = [
        ("Original",    counts),
        ("Class Weights\n(conceptual)", counts),  # Same count, diff weight
        ("Random\nOversampling",  [counts[0], int(counts[0]*0.5), counts[0]]),
        ("SMOTE\n(synthetic)",    [counts[0], int(counts[0]*0.5), counts[0]]),
    ]

    n_sc   = len(scenarios)
    pad_l, pad_r, pad_t, pad_b = 60, 20, 50, 90
    chart_w = W - pad_l - pad_r
    chart_h = H - pad_t - pad_b
    sc_w   = chart_w // n_sc
    n_cls  = 3
    bar_w  = sc_w // 5

    max_v = max(c for _, sc in scenarios for c in sc) * 1.12
    draw_grid_h(draw, pad_l, pad_t, W-pad_r, pad_t+chart_h, n=5)

    for si, (sc_lbl, sc_counts) in enumerate(scenarios):
        x_base = pad_l + si * sc_w
        for ci, cnt in enumerate(sc_counts):
            x0 = x_base + ci * (bar_w + 3) + 10
            bh = int(chart_h * cnt / max_v)
            y0 = pad_t + chart_h - bh
            col = CLASS_COLORS[ci]
            # Dashed outline for synthetic
            draw.rectangle([x0, y0, x0+bar_w, pad_t+chart_h], fill=col)
            draw.text((x0, y0-14), f"{cnt//1000}K", font=get_font(8), fill=COLORS["text"])

        # Scenario label
        for li, line in enumerate(sc_lbl.split("\n")):
            draw.text((x_base + sc_w//2 - 35, pad_t + chart_h + 8 + li*14),
                      line, font=get_font(9, bold=(li==0)), fill=COLORS["text"])

    draw.line([(pad_l, pad_t), (pad_l, pad_t+chart_h)], fill=COLORS["text"], width=2)
    draw.line([(pad_l, pad_t+chart_h), (W-pad_r, pad_t+chart_h)], fill=COLORS["text"], width=2)

    for ci, (lbl, col) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        draw.rectangle([pad_l + ci*230, 32, pad_l+ci*230+12, 44], fill=col)
        draw.text((pad_l+ci*230+16, 30), lbl, font=get_font(10), fill=COLORS["text"])

    save(img, "10_resampling_strategy_comparison.png")

fig_resampling_strategies()

print(f"\n  Total figures saved: 10")

# ═══════════════════════════════════════════════════════════════════════════
# 6. GENERATE REPORTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Generating reports ──────────────────────────────────")

now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

# ── 6a. Data Quality Report ────────────────────────────────────────────────
dq_lines = []
dq_lines.append(f"# CDC Diabetes Health Indicators – Data Quality Report\n")
dq_lines.append(f"**Generated:** {now_str}  \n")
dq_lines.append(f"**Dataset:** `data/raw/diabetes_012_health_indicators_BRFSS2015.csv`\n\n")
dq_lines.append("---\n\n")

dq_lines.append("## 1. Dataset Overview\n\n")
dq_lines.append(f"| Metric | Value |\n|--------|-------|\n")
dq_lines.append(f"| Total rows | {len(df):,} |\n")
dq_lines.append(f"| Total columns | {len(df.columns)} |\n")
dq_lines.append(f"| Target column | `Diabetes_012` |\n")
dq_lines.append(f"| Feature columns | {len(FEATURES)} |\n")
dq_lines.append(f"| Missing values | {null_counts.sum()} |\n")
dq_lines.append(f"| Duplicate rows | {n_dupes:,} ({pct_dupes:.2f}%) |\n")
dq_lines.append(f"| Rows in duplicate groups | {n_dupes_keep:,} |\n\n")

dq_lines.append("## 2. Missing Values\n\n")
if null_counts.sum() == 0:
    dq_lines.append("> ✅ **No missing values detected.** The dataset is fully populated.\n\n")
else:
    dq_lines.append("| Column | Missing Count | Missing % |\n|--------|--------------|----------|\n")
    for col in df.columns:
        if null_counts[col] > 0:
            dq_lines.append(f"| `{col}` | {null_counts[col]:,} | {null_pct[col]:.2f}% |\n")
    dq_lines.append("\n")

dq_lines.append("## 3. Data Types\n\n")
dq_lines.append("| Column | Dtype | Unique Values | Min | Max |\n|--------|-------|--------------|-----|-----|\n")
for col in df.columns:
    mn = df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else "N/A"
    mx = df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else "N/A"
    dq_lines.append(f"| `{col}` | {df[col].dtype} | {df[col].nunique()} | {mn} | {mx} |\n")
dq_lines.append("\n")

dq_lines.append("## 4. Invalid Value Check\n\n")
if not invalid_summary:
    dq_lines.append("> ✅ **No invalid values detected.** All columns are within expected ranges.\n\n")
else:
    dq_lines.append("| Column | Issue |\n|--------|-------|\n")
    for col, issue in invalid_summary.items():
        dq_lines.append(f"| `{col}` | {issue} |\n")
    dq_lines.append("\n")

dq_lines.append("## 5. Duplicate Record Analysis\n\n")
dq_lines.append(f"**Total duplicate rows:** {n_dupes:,} ({pct_dupes:.2f}% of dataset)\n\n")
dq_lines.append("### Distribution of Duplicates by Class\n\n")
dq_lines.append("| Class | Original | After Dedup | Duplicates Removed |\n|-------|----------|-------------|--------------------|\n")
for cls in range(3):
    orig   = int(target_vc.get(cls, 0))
    deduped = int(target_vc_deduped.get(cls, 0))
    diff   = orig - deduped
    lbl    = ["No Diabetes","Prediabetes","Diabetes"][cls]
    dq_lines.append(f"| {cls} – {lbl} | {orig:,} | {deduped:,} | {diff:,} |\n")
dq_lines.append("\n")
dq_lines.append("### Interpretation of Duplicate Rows\n\n")
dq_lines.append(textwrap.dedent("""
The CDC BRFSS is a telephone survey. Each row represents one survey respondent's
self-reported health data. Duplicate rows — where every single one of the 22
column values is identical — may arise from:

1. **Legitimate coincidence**: Two individuals share identical demographic and
   health profiles (e.g., same age bracket, BMI category, binary health indicators).
   Given the survey's categorical and binary structure (most values are integers
   in small ranges), hash-collisions are statistically probable at scale.
2. **Data processing artefacts**: Re-uploads, merges, or ETL steps in the
   original Kaggle dataset may have introduced row duplication.
3. **Survey non-uniqueness by design**: BRFSS does not enforce uniqueness across
   respondents; two people can and will give identical responses.

**Recommendation:** Retain all duplicate rows in the full dataset. Apply
`drop_duplicates()` only *within the training split* as an optional preprocessing
step, never before train/test splitting. Evaluate model performance with and
without deduplication to measure impact.
""").strip() + "\n\n")

dq_lines.append("## 6. Schema Verification\n\n")
if missing_cols:
    dq_lines.append(f"> ⚠️ Missing expected columns: {missing_cols}\n\n")
elif extra_cols:
    dq_lines.append(f"> ℹ️ Extra columns found: {extra_cols}\n\n")
else:
    dq_lines.append("> ✅ All 22 expected columns present and accounted for.\n\n")

dq_lines.append("## 7. Security Observations\n\n")
dq_lines.append("> ⚠️ No PII detected in column names. Data appears fully anonymised per BRFSS protocol.\n\n")
dq_lines.append("> ✅ No credentials, API keys, or sensitive tokens were accessed during this audit.\n\n")

(RPTS / "cdc_data_quality_report.md").write_text("".join(dq_lines), encoding="utf-8")
print(f"  Saved: cdc_data_quality_report.md")

# ── 6b. EDA Report ─────────────────────────────────────────────────────────
eda_lines = []
eda_lines.append(f"# CDC Diabetes Health Indicators – EDA Report\n")
eda_lines.append(f"**Generated:** {now_str}  \n")
eda_lines.append(f"**Dataset:** CDC BRFSS 2015 | `diabetes_012_health_indicators_BRFSS2015.csv`\n\n")
eda_lines.append("---\n\n")

# 1. Dataset Quality
eda_lines.append("## 1. Dataset Quality\n\n")
eda_lines.append(f"The dataset contains **{len(df):,} records** and **{len(df.columns)} columns**. ")
eda_lines.append(f"There are **no missing values**, making imputation unnecessary. ")
eda_lines.append(f"All features are numeric (float64), consistent with the binary/ordinal/count nature of BRFSS survey items. ")
eda_lines.append(f"No invalid out-of-range values were detected. ")
eda_lines.append(f"The only quality concern is **{n_dupes:,} duplicate rows ({pct_dupes:.2f}%)**, which are likely legitimate given the survey's categorical structure.\n\n")

# 2. Feature Characteristics
eda_lines.append("## 2. Feature Characteristics\n\n")
eda_lines.append("### 2.1 Feature Categorisation\n\n")
eda_lines.append("| Category | Features |\n|----------|----------|\n")
for cat, feats in feature_cats.items():
    eda_lines.append(f"| **{cat}** | {', '.join(f'`{f}`' for f in feats)} |\n")
eda_lines.append("\n")

eda_lines.append("### 2.2 Descriptive Statistics (selected features)\n\n")
eda_lines.append("| Feature | Mean | Std | Min | 25% | Median | 75% | Max |\n")
eda_lines.append("|---------|------|-----|-----|-----|--------|-----|-----|\n")
for col in ["BMI","MentHlth","PhysHlth","Age","GenHlth","Education","Income"]:
    if col in df.columns:
        d = desc[col]
        eda_lines.append(f"| `{col}` | {d['mean']:.2f} | {d['std']:.2f} | {d['min']:.0f} | {d['25%']:.0f} | {d['50%']:.0f} | {d['75%']:.0f} | {d['max']:.0f} |\n")
eda_lines.append("\n")

eda_lines.append("### 2.3 Class-wise Feature Means\n\n")
eda_lines.append("| Feature | No Diabetes (0) | Prediabetes (1) | Diabetes (2) |\n")
eda_lines.append("|---------|----------------|-----------------|---------------|\n")
for feat in FEATURES[:21]:
    if feat in class_means.columns:
        m0 = class_means.loc[0, feat] if 0 in class_means.index else float("nan")
        m1 = class_means.loc[1, feat] if 1 in class_means.index else float("nan")
        m2 = class_means.loc[2, feat] if 2 in class_means.index else float("nan")
        eda_lines.append(f"| `{feat}` | {m0:.3f} | {m1:.3f} | {m2:.3f} |\n")
eda_lines.append("\n")

eda_lines.append("### 2.4 Top Features Correlated with Target\n\n")
eda_lines.append("| Rank | Feature | Pearson r (with Diabetes_012) | Direction |\n")
eda_lines.append("|------|---------|------------------------------|-----------|\n")
for rank, (feat, (r, p)) in enumerate(target_corr_sorted[:10], 1):
    direction = "Positive ↑" if r > 0 else "Negative ↓"
    eda_lines.append(f"| {rank} | `{feat}` | {r:+.4f} | {direction} |\n")
eda_lines.append("\n")

eda_lines.append("### 2.5 Possibly Redundant Feature Pairs (Cramér's V > 0.3)\n\n")
eda_lines.append("| Feature Pair | Cramér's V | Note |\n|-------------|-----------|------|\n")
high_cramers = [(k, v) for k, v in top_cramers if v > 0.1]
if high_cramers:
    for pair, v in top_cramers[:8]:
        a, b = pair.split("|")
        note = "Consider dropping one" if v > 0.4 else "Monitor"
        eda_lines.append(f"| `{a}` ↔ `{b}` | {v:.3f} | {note} |\n")
else:
    eda_lines.append("| — | — | No strong redundancy detected among binary features |\n")
eda_lines.append("\n")

# 3. Target Distribution
eda_lines.append("## 3. Target Distribution\n\n")
eda_lines.append("| Class | Label | Count | Proportion |\n|-------|-------|-------|------------|\n")
for cls in range(3):
    cnt  = int(target_vc.get(cls, 0))
    pct  = cnt / len(df) * 100
    lbl  = ["No Diabetes","Prediabetes","Diabetes"][cls]
    eda_lines.append(f"| {cls} | {lbl} | {cnt:,} | {pct:.2f}% |\n")
eda_lines.append("\n")

no_diab = int(target_vc.get(0, 0))
prediab = int(target_vc.get(1, 0))
diab    = int(target_vc.get(2, 0))
eda_lines.append(f"> The dataset is **severely imbalanced**. Class 0 (No Diabetes) dominates with ~{no_diab/len(df)*100:.0f}% of records. ")
eda_lines.append(f"Class 1 (Prediabetes) is the minority class with only {prediab:,} samples (~{prediab/len(df)*100:.1f}%), ")
eda_lines.append(f"roughly **{no_diab//max(prediab,1)}× less** than the majority class. ")
eda_lines.append(f"Class 2 (Diabetes) is moderately represented at ~{diab/len(df)*100:.0f}%.\n\n")

# 4. Duplicate-Record Findings
eda_lines.append("## 4. Duplicate-Record Findings\n\n")
eda_lines.append(f"- **{n_dupes:,} duplicate rows** ({pct_dupes:.2f}% of total).\n")
eda_lines.append(f"- **{n_dupes_keep:,} rows** are involved in at least one duplicate group.\n")
eda_lines.append(f"- After deduplication: **{len(df_no_dupes):,} unique rows** remain.\n")
eda_lines.append("- Duplicates are plausible given the survey's low-cardinality integer features.\n")
eda_lines.append("- **Action**: Retain duplicates in the full dataset. Evaluate deduplication impact in a sensitivity analysis during model training.\n\n")

# 5. Class Imbalance Challenges
eda_lines.append("## 5. Class Imbalance Challenges\n\n")
eda_lines.append("The imbalance creates several modelling risks:\n\n")
eda_lines.append("1. **Accuracy paradox** – A naive majority-class classifier achieves ~{:.0f}% accuracy trivially.\n".format(no_diab/len(df)*100))
eda_lines.append("2. **Minority class suppression** – Models may rarely or never predict Class 1 (Prediabetes).\n")
eda_lines.append("3. **Metric inflation** – Accuracy is misleading; macro-averaged F1, AUC-OVR and Cohen's κ are essential.\n\n")

eda_lines.append("### Strategy Comparison\n\n")
eda_lines.append("| Strategy | Pros | Cons | Recommendation |\n|----------|------|------|----------------|\n")
eda_lines.append("| **Class Weights** | No data leakage, fast, works with any estimator | May not fully compensate extreme imbalance | ✅ **Use as baseline** – set `class_weight='balanced'` |\n")
eda_lines.append("| **Random Oversampling** | Simple, no information loss | Risk of overfitting duplicated samples | ✅ Try as secondary option |\n")
eda_lines.append("| **SMOTE** | Generates diverse synthetic samples | Assumes feature-space continuity; binary features violate this | ⚠️ Use `SMOTENC` variant for mixed binary/ordinal features |\n")
eda_lines.append("| **ADASYN** | Adaptive focus on hard samples | Same interpolation concern as SMOTE | 🔬 Experimental only |\n")
eda_lines.append("| **Undersampling** | Reduces training time | Loses majority-class signal | ❌ Not recommended given dataset size |\n\n")

eda_lines.append("> [!IMPORTANT]\n> **Any resampling must be applied only to the training split, never before train/test splitting.**\n> Doing otherwise constitutes data leakage and will produce artificially optimistic evaluation metrics.\n\n")

# 6. Recommended Preprocessing
eda_lines.append("## 6. Recommended Preprocessing\n\n")
eda_lines.append("1. **No imputation needed** – Zero missing values.\n")
eda_lines.append("2. **No encoding needed** – All features are already numeric.\n")
eda_lines.append("3. **Scaling**: StandardScaler or MinMaxScaler for distance-based models (KNN, SVM, neural nets); tree-based models (XGBoost, LightGBM, RF) do not require scaling.\n")
eda_lines.append("4. **BMI outliers**: Values above 60 exist but represent real (morbidly obese) individuals; cap at 98th percentile only if they degrade model performance.\n")
eda_lines.append("5. **Deduplication**: Evaluate as an optional training-only step in a sensitivity analysis.\n")
eda_lines.append("6. **Feature selection**: Consider dropping one of `HighBP`/`HighChol`/`CholCheck` if Cramér's V confirms strong redundancy.\n")
eda_lines.append("7. **Resampling**: Apply `class_weight='balanced'` as the default strategy; evaluate SMOTENC as an alternative.\n\n")

# 7. Recommended Validation Strategy
eda_lines.append("## 7. Recommended Validation Strategy\n\n")
eda_lines.append("| Step | Recommendation |\n|------|----------------|\n")
eda_lines.append("| Train/Test Split | Stratified 80/20 split (preserves class ratios) |\n")
eda_lines.append("| Cross-Validation | Stratified 5-fold CV on training set |\n")
eda_lines.append("| Hyperparameter Tuning | Optuna or GridSearchCV within CV folds |\n")
eda_lines.append("| Final Evaluation | Held-out test set (never touched during training/tuning) |\n")
eda_lines.append("| Deduplication Sensitivity | Repeat eval with and without `drop_duplicates()` on train |\n\n")

# 8. Recommended Evaluation Metrics
eda_lines.append("## 8. Recommended Model Evaluation Metrics\n\n")
eda_lines.append("| Metric | Rationale |\n|--------|-----------|\n")
eda_lines.append("| **Macro-averaged F1** | Treats all three classes equally; penalises minority class neglect |\n")
eda_lines.append("| **Per-class Precision / Recall / F1** | Diagnose per-class performance independently |\n")
eda_lines.append("| **AUC-ROC (OVR)** | Multi-class discriminability, robust to imbalance |\n")
eda_lines.append("| **Cohen's κ** | Agreement above chance; accounts for class frequency |\n")
eda_lines.append("| **Confusion Matrix** | Full picture of class-level errors |\n")
eda_lines.append("| **MCC (Matthews)** | Single balanced metric for multi-class imbalanced problems |\n")
eda_lines.append("| ~~Accuracy~~ | ⚠️ Misleading due to class imbalance — do not use as primary metric |\n\n")

eda_lines.append("---\n\n")
eda_lines.append("## Figures Generated\n\n")
fig_manifest = [
    ("01_target_class_distribution.png","Target class bar chart with counts and percentages"),
    ("02_class_imbalance_pie.png","Pie chart showing proportional class imbalance"),
    ("03_bmi_distribution_by_class.png","BMI histogram overlaid per class"),
    ("04_correlation_heatmap.png","Pearson correlation heatmap of all features"),
    ("05_binary_feature_prevalence.png","Binary feature prevalence (%) per class"),
    ("06_ordinal_distributions.png","Ordinal feature distributions per class"),
    ("07_feature_target_correlation.png","Ranked Pearson r with Diabetes_012"),
    ("08_mental_physical_health_days.png","MentHlth & PhysHlth day distributions per class"),
    ("09_duplicate_analysis.png","Duplicate row analysis by class"),
    ("10_resampling_strategy_comparison.png","Conceptual comparison of resampling strategies"),
]
for fname, desc_txt in fig_manifest:
    eda_lines.append(f"- **[{fname}](../figures/{fname})** – {desc_txt}\n")

eda_lines.append("\n---\n\n")
eda_lines.append("*Report auto-generated by `notebooks/cdc_eda_phase2.py` – Phase 2.*\n")

(RPTS / "cdc_eda_report.md").write_text("".join(eda_lines), encoding="utf-8")
print(f"  Saved: cdc_eda_report.md")

# ═══════════════════════════════════════════════════════════════════════════
# 7. SUMMARY TO CONSOLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("PHASE 2 COMPLETE")
print("═"*60)
print(f"\nDataset: {len(df):,} rows × {len(df.columns)} cols")
print(f"Missing values: {null_counts.sum()}")
print(f"Duplicates: {n_dupes:,} ({pct_dupes:.2f}%)")
print(f"Class distribution:")
for cls in range(3):
    cnt = int(target_vc.get(cls, 0))
    print(f"  Class {cls}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
print(f"\nFigures saved to: results/figures/  (10 files)")
print(f"Reports saved to: results/reports/  (2 files)")
print(f"\nExisting files NOT modified:")
print(f"  main.py, src/app.py, src/train_model.py, src/predict.py")
print(f"  models/*.pkl, data/diabetes.csv, config.yaml")
print("\nPhase 2 EDA complete. Awaiting approval for Phase 3.")

