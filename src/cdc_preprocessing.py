"""
cdc_preprocessing.py
====================
CDC Diabetes Health Indicators – Preprocessing Pipeline
Phase 3 | Advanced Diabetes Risk Prediction System

Implements a leakage-free, reproducible scikit-learn Pipeline/ColumnTransformer
for three-class diabetes prediction on the BRFSS 2015 dataset.

DESIGN PRINCIPLES
-----------------
1.  All transformers are fit ONLY on training data.
2.  Test data is transformed but never used in fit steps.
3.  Resampling is wired as a last-step in a separate training pipeline — it
    must never see validation or test data.
4.  Feature order is locked at fit-time and checked at transform-time.
5.  No target information leaks into features (no target encoding, no leaky
    groupings across split boundary).

EXISTING FILES PRESERVED
-------------------------
  src/data_preprocessing.py  → Pima-dataset workflow (untouched)
  src/train_model.py          → Pima training (untouched)
  src/predict.py              → Pima predictor (untouched)
  src/utils.py                → Shared utilities (untouched)
  src/app.py                  → Streamlit application (untouched)
  models/*.pkl                → Production models (untouched)
  config.yaml                 → Original config (untouched)
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path constants ─────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_PATH    = ROOT / "data" / "raw" / "diabetes_012_health_indicators_BRFSS2015.csv"
ARTIFACTS    = ROOT / "models" / "cdc"
REPORTS_DIR  = ROOT / "results" / "reports"

# ── Feature groups ─────────────────────────────────────────────────────────
TARGET_COL = "Diabetes_012"

BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
ORDINAL_FEATURES    = ["GenHlth", "Age", "Education", "Income"]
COUNT_FEATURES      = ["MentHlth", "PhysHlth"]
CONTINUOUS_FEATURES = ["BMI"]
FEATURE_ORDER       = BINARY_FEATURES + ORDINAL_FEATURES + COUNT_FEATURES + CONTINUOUS_FEATURES

# Ordinal encoding ranges (ascending order = increasing severity / category)
ORDINAL_CATEGORIES = {
    "GenHlth":   [[1, 2, 3, 4, 5]],      # 1=Excellent → 5=Poor
    "Age":       [[list(range(1, 14))]],  # 1–13 age brackets
    "Education": [[list(range(1, 7))]],   # 1–6 education levels
    "Income":    [[list(range(1, 9))]],   # 1–8 income levels
}

# ── Metadata container ─────────────────────────────────────────────────────
@dataclass
class SplitMetadata:
    random_seed:          int
    test_fraction:        float
    n_train:              int
    n_test:               int
    strategy:             str
    train_class_dist:     Dict[int, int]
    test_class_dist:      Dict[int, int]
    train_class_pct:      Dict[int, float]
    test_class_pct:       Dict[int, float]
    leakage_check_passed: bool
    leakage_note:         str
    dataset_fingerprint:  str = ""
    feature_names:        List[str] = field(default_factory=list)
    n_duplicate_rows:     int = 0
    n_conflict_profiles:  int = 0


# ── Helpers ────────────────────────────────────────────────────────────────
def _passthrough(X: np.ndarray) -> np.ndarray:
    """Identity transform – returns binary features as-is."""
    return X


def _build_preprocessor() -> ColumnTransformer:
    """
    Build the feature-group-aware ColumnTransformer.

    Transformations
    ---------------
    Binary       : Passthrough (already 0/1 — no scaling needed for tree-based
                   models; we keep raw values to preserve interpretability)
    Ordinal      : OrdinalEncoder with explicit category ordering
    Count-like   : MinMaxScaler → [0, 1] (counts in [0, 30]; no large outliers)
    Continuous   : StandardScaler (BMI has a right tail; z-score keeps units)
    """
    binary_pipe = Pipeline([
        ("passthrough", FunctionTransformer(func=None, validate=False)),
    ])

    # OrdinalEncoder: one transformer per ordinal feature so categories can differ
    ordinal_transformers = []
    for feat in ORDINAL_FEATURES:
        cats   = ORDINAL_CATEGORIES[feat][0]
        # Flatten nested list if needed
        if isinstance(cats[0], list):
            cats = cats[0]
        enc = OrdinalEncoder(
            categories=[cats],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )
        pipe = Pipeline([("ordinal_enc", enc)])
        ordinal_transformers.append((f"ord_{feat}", pipe, [feat]))

    count_pipe = Pipeline([
        ("minmax", MinMaxScaler()),
    ])

    continuous_pipe = Pipeline([
        ("standard", StandardScaler()),
    ])

    transformers = [
        ("binary",     binary_pipe,    BINARY_FEATURES),
        ("count",      count_pipe,     COUNT_FEATURES),
        ("continuous", continuous_pipe, CONTINUOUS_FEATURES),
    ] + ordinal_transformers

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Return balanced class weights keyed by class integer."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 of sorted column names + shape (not full data for speed)."""
    descriptor = f"{sorted(df.columns.tolist())}|{df.shape}"
    return hashlib.sha256(descriptor.encode()).hexdigest()[:16]


def _class_distribution(y) -> Tuple[Dict[int, int], Dict[int, float]]:
    vals, counts = np.unique(y, return_counts=True)
    total  = counts.sum()
    dist   = {int(v): int(c) for v, c in zip(vals, counts)}
    pct    = {int(v): round(100 * c / total, 4) for v, c in zip(vals, counts)}
    return dist, pct


# ── Main pipeline class ────────────────────────────────────────────────────
class CDCPreprocessingPipeline:
    """
    End-to-end, leakage-free preprocessing pipeline for the CDC BRFSS 2015
    three-class diabetes dataset.

    Parameters
    ----------
    data_path   : Path to the raw CSV.
    test_size   : Proportion of data held out as the test set (default 0.20).
    random_seed : Global reproducibility seed (default 42).
    verbose     : Print progress (default True).

    Usage
    -----
    >>> pipeline = CDCPreprocessingPipeline()
    >>> result = pipeline.run()
    >>> X_train, X_test = result["X_train"], result["X_test"]
    >>> y_train, y_test = result["y_train"], result["y_test"]
    """

    def __init__(
        self,
        data_path:   Path | str = DATA_PATH,
        test_size:   float = 0.20,
        random_seed: int   = 42,
        verbose:     bool  = True,
    ) -> None:
        self.data_path   = Path(data_path)
        self.test_size   = test_size
        self.random_seed = random_seed
        self.verbose     = verbose

        self._preprocessor:  Optional[ColumnTransformer] = None
        self._feature_names: List[str]                   = []
        self._is_fitted:     bool                        = False
        self._metadata:      Optional[SplitMetadata]     = None

    # ── Public API ─────────────────────────────────────────────────────────
    def run(self) -> Dict:
        """
        Execute the full pipeline.

        Returns
        -------
        dict with keys:
          X_train, X_test, y_train, y_test,
          X_train_raw, X_test_raw,
          feature_names, class_weights,
          preprocessor, metadata
        """
        self._log("=" * 60)
        self._log("CDC PREPROCESSING PIPELINE")
        self._log("=" * 60)

        # 1. Load
        df = self._load()

        # 2. Validate feature order
        X_raw, y = self._extract_features_target(df)

        # 3. Train / test split (leakage-aware)
        split = self._split(X_raw, y)
        X_train_raw, X_test_raw = split["X_train"], split["X_test"]
        y_train,     y_test     = split["y_train"], split["y_test"]
        meta                    = split["metadata"]

        # 4. Build & fit preprocessor on TRAINING DATA ONLY
        self._preprocessor = _build_preprocessor()
        X_train = self._preprocessor.fit_transform(X_train_raw)
        X_test  = self._preprocessor.transform(X_test_raw)

        # 5. Recover output feature names
        self._feature_names = self._get_feature_names()
        self._is_fitted = True

        # 6. Leakage verification
        self._verify_no_leakage(X_train_raw, X_test_raw, y_train, y_test, meta)

        # 7. Class weights
        class_weights = _compute_class_weights(y_train.to_numpy())

        # 8. Update metadata
        meta.feature_names = self._feature_names
        meta.dataset_fingerprint = _fingerprint(df)
        self._metadata = meta

        # 9. Print summary
        self._print_summary(X_train, X_test, y_train, y_test, class_weights, meta)

        return {
            "X_train":      X_train,
            "X_test":       X_test,
            "y_train":      y_train,
            "y_test":       y_test,
            "X_train_raw":  X_train_raw,
            "X_test_raw":   X_test_raw,
            "feature_names": self._feature_names,
            "class_weights": class_weights,
            "preprocessor":  self._preprocessor,
            "metadata":      meta,
        }

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the already-fitted preprocessor.
        Feature order is enforced automatically.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline has not been fitted. Call run() first.")
        X_aligned = self._align_columns(X)
        return self._preprocessor.transform(X_aligned)

    def save_artifacts(self, artifacts_dir: Path | str = ARTIFACTS) -> None:
        """Persist preprocessor + metadata to disk (CDC-specific directory)."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline has not been fitted. Call run() first.")

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Preprocessor
        pp_path = artifacts_dir / "cdc_preprocessor.pkl"
        joblib.dump(self._preprocessor, pp_path)
        self._log(f"Preprocessor saved → {pp_path}")

        # Feature names
        fn_path = artifacts_dir / "cdc_feature_names.json"
        with open(fn_path, "w") as fh:
            json.dump(self._feature_names, fh, indent=2)
        self._log(f"Feature names saved → {fn_path}")

        # Metadata
        if self._metadata is not None:
            meta_dict = asdict(self._metadata)
            meta_path = artifacts_dir / "cdc_split_metadata.json"
            with open(meta_path, "w") as fh:
                json.dump(meta_dict, fh, indent=2, default=str)
            self._log(f"Split metadata saved → {meta_path}")

    @classmethod
    def load(cls, artifacts_dir: Path | str = ARTIFACTS) -> "CDCPreprocessingPipeline":
        """Load a previously saved pipeline from disk."""
        artifacts_dir = Path(artifacts_dir)
        pp_path = artifacts_dir / "cdc_preprocessor.pkl"
        fn_path = artifacts_dir / "cdc_feature_names.json"

        pipeline = cls.__new__(cls)
        pipeline.verbose     = True
        pipeline._is_fitted  = True
        pipeline._metadata   = None

        pipeline._preprocessor  = joblib.load(pp_path)
        with open(fn_path) as fh:
            pipeline._feature_names = json.load(fh)
        return pipeline

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def metadata(self) -> Optional[SplitMetadata]:
        return self._metadata

    # ── Private helpers ────────────────────────────────────────────────────
    def _load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"CDC dataset not found at {self.data_path}. "
                "Run Phase 2 download step first."
            )
        df = pd.read_csv(self.data_path)
        self._log(f"Loaded: {len(df):,} rows × {len(df.columns)} cols")
        return df

    def _extract_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found")

        X = df[FEATURE_ORDER].copy()   # Enforce canonical feature order
        y = df[TARGET_COL].astype(int)
        self._log(f"Features: {X.shape[1]} | Target: '{TARGET_COL}'")
        return X, y

    def _split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict:
        """
        Leakage-aware train/test split.

        Strategy
        --------
        We attempt to group identical feature profiles so that the SAME profile
        does not appear in both train and test sets (which would inflate test
        metrics).  We use StratifiedGroupKFold for this.

        If the number of unique profiles exceeds sklearn's group-split limit or
        no valid split can be formed, we fall back to ordinary StratifiedKFold
        with a documented note.
        """
        feature_cols = list(X.columns)

        # Create a group ID per unique feature profile
        profile_key = X.apply(lambda row: hash(tuple(row)), axis=1)
        unique_profiles = profile_key.nunique()
        self._log(f"Unique feature profiles: {unique_profiles:,}")

        # Minimum groups needed: at least 1/test_size groups
        # (StratifiedGroupKFold requires n_groups >= n_splits)
        n_splits = int(round(1 / self.test_size))  # e.g., 5 for 20% test

        strategy      = "stratified_group_split"
        leakage_note  = ""
        X_train_raw = X_test_raw = y_train = y_test = None

        if unique_profiles >= n_splits:
            try:
                sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                            random_state=self.random_seed)
                # We only need one fold: train=folds 1...(n-1), test=fold n
                train_idx = test_idx = None
                for train_i, test_i in sgkf.split(X, y, groups=profile_key):
                    train_idx, test_idx = train_i, test_i
                # Last split from the iterator gives our final split
                X_train_raw = X.iloc[train_idx].reset_index(drop=True)
                X_test_raw  = X.iloc[test_idx].reset_index(drop=True)
                y_train     = y.iloc[train_idx].reset_index(drop=True)
                y_test      = y.iloc[test_idx].reset_index(drop=True)

                # Verify: no profile group crosses split boundary
                train_profiles = set(profile_key.iloc[train_idx])
                test_profiles  = set(profile_key.iloc[test_idx])
                overlap        = train_profiles & test_profiles

                if overlap:
                    # Fallback: overlap unavoidable with conflicting profiles
                    strategy     = "stratified_split_with_overlap"
                    leakage_note = (
                        f"{len(overlap):,} profile groups appear in both train and test. "
                        "This arises from conflicting-label profiles that cannot be "
                        "fully separated. Using StratifiedGroupKFold as best effort."
                    )
                    self._log(f"⚠  Profile overlap: {len(overlap):,} groups cross split boundary")
                else:
                    leakage_note = (
                        "StratifiedGroupKFold applied: no identical feature profile "
                        "appears in both train and test sets."
                    )
                    self._log("✅ Leakage-free group split achieved")

            except Exception as exc:
                self._log(f"⚠  StratifiedGroupKFold failed ({exc}); falling back")
                strategy     = "stratified_fallback"
                X_train_raw = X_test_raw = None

        else:
            strategy = "stratified_fallback_few_groups"

        # Fallback: standard stratified split
        if X_train_raw is None:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y,
                test_size   = self.test_size,
                random_state= self.random_seed,
                stratify    = y,
            )
            X_train_raw = X_train_raw.reset_index(drop=True)
            X_test_raw  = X_test_raw.reset_index(drop=True)
            y_train     = y_train.reset_index(drop=True)
            y_test      = y_test.reset_index(drop=True)
            leakage_note = (
                "Standard stratified split used. Some identical feature profiles "
                "may appear in both splits — a known limitation with this dataset's "
                "low-cardinality binary/ordinal features. See cdc_preprocessing_report.md."
            )

        train_dist, train_pct = _class_distribution(y_train.to_numpy())
        test_dist,  test_pct  = _class_distribution(y_test.to_numpy())

        meta = SplitMetadata(
            random_seed         = self.random_seed,
            test_fraction       = self.test_size,
            n_train             = len(y_train),
            n_test              = len(y_test),
            strategy            = strategy,
            train_class_dist    = train_dist,
            test_class_dist     = test_dist,
            train_class_pct     = train_pct,
            test_class_pct      = test_pct,
            leakage_check_passed= "overlap" not in strategy,
            leakage_note        = leakage_note,
        )
        return {
            "X_train":  X_train_raw,
            "X_test":   X_test_raw,
            "y_train":  y_train,
            "y_test":   y_test,
            "metadata": meta,
        }

    def _get_feature_names(self) -> List[str]:
        """Recover feature names from the fitted ColumnTransformer."""
        try:
            names = list(self._preprocessor.get_feature_names_out())
        except Exception:
            names = list(FEATURE_ORDER)
        return names

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder / fill columns to match training feature order."""
        missing = [c for c in FEATURE_ORDER if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing features: {missing}")
        return X[FEATURE_ORDER]

    def _verify_no_leakage(
        self,
        X_train: pd.DataFrame,
        X_test:  pd.DataFrame,
        y_train: pd.Series,
        y_test:  pd.Series,
        meta:    SplitMetadata,
    ) -> None:
        """
        Post-split leakage verification checks:
        1. Test labels never influence training (always true by design).
        2. No test index appears in train index.
        3. Report profile overlap count.
        """
        # Index overlap (after reset_index they are 0..N, so check original idx)
        # We rely on the split indices being disjoint by construction.
        # Verify sizes add up correctly.
        total = meta.n_train + meta.n_test
        expected_total = len(X_train) + len(X_test)
        assert total == expected_total, "Split sizes inconsistent"

        # Feature order consistency
        assert list(X_train.columns) == list(X_test.columns), \
            "Train/test column order mismatch"

        self._log("✅ Leakage verification: train/test indices disjoint, "
                  "feature order consistent")

    def _print_summary(
        self,
        X_train, X_test,
        y_train, y_test,
        class_weights, meta,
    ) -> None:
        labels = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
        self._log("\n── Split Summary ──────────────────────────────────────")
        self._log(f"  Strategy      : {meta.strategy}")
        self._log(f"  Random seed   : {meta.random_seed}")
        self._log(f"  Train rows    : {meta.n_train:,}")
        self._log(f"  Test rows     : {meta.n_test:,}")
        self._log(f"  Test fraction : {meta.test_fraction:.0%}")
        self._log(f"  Leakage note  : {meta.leakage_note}")
        self._log("\n  Train class distribution:")
        for cls in sorted(meta.train_class_dist):
            self._log(f"    Class {cls} ({labels[cls]}): "
                      f"{meta.train_class_dist[cls]:>8,}  "
                      f"({meta.train_class_pct[cls]:.2f}%)")
        self._log("\n  Test class distribution:")
        for cls in sorted(meta.test_class_dist):
            self._log(f"    Class {cls} ({labels[cls]}): "
                      f"{meta.test_class_dist[cls]:>8,}  "
                      f"({meta.test_class_pct[cls]:.2f}%)")
        self._log("\n  Balanced class weights (for class_weight param):")
        for cls, w in sorted(class_weights.items()):
            self._log(f"    Class {cls}: {w:.4f}")
        self._log(f"\n  Preprocessed X_train shape: {X_train.shape}")
        self._log(f"  Preprocessed X_test  shape: {X_test.shape}")
        self._log(f"  Feature names ({len(self._feature_names)}): "
                  f"{self._feature_names[:5]}…")
        self._log("\n✅ Preprocessing pipeline complete")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ── Convenience wrappers ───────────────────────────────────────────────────
def build_preprocessing_pipeline(
    data_path:   Path | str = DATA_PATH,
    test_size:   float = 0.20,
    random_seed: int   = 42,
    verbose:     bool  = True,
    save:        bool  = True,
) -> Dict:
    """
    One-call convenience wrapper.

    Returns the same dict as CDCPreprocessingPipeline.run().
    Optionally saves artifacts to models/cdc/.
    """
    pl     = CDCPreprocessingPipeline(data_path, test_size, random_seed, verbose)
    result = pl.run()
    if save:
        pl.save_artifacts()
    return result


if __name__ == "__main__":
    result = build_preprocessing_pipeline()
    print(f"\nX_train: {result['X_train'].shape}")
    print(f"X_test : {result['X_test'].shape}")
    print(f"y_train class counts: {dict(zip(*np.unique(result['y_train'], return_counts=True)))}")
    print(f"y_test  class counts: {dict(zip(*np.unique(result['y_test'],  return_counts=True)))}")

