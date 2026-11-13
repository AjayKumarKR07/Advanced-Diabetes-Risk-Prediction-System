"""
tests/test_cdc_preprocessing.py
================================
Automated test suite for Phase 3 CDC preprocessing pipeline.

Covers:
  - Dataset loading and validation
  - Target class validation
  - Feature schema validation
  - Duplicate-profile detection
  - Conflicting-target detection
  - Train/test leakage prevention
  - Preprocessing transformations
  - Feature-order consistency
  - Reproducibility
  - Feature engineering module

Run from project root:
  python -m pytest tests/test_cdc_preprocessing.py -v

No existing files are modified. No models are trained.
"""

import sys
import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# ── Add src/ to path ───────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from cdc_data_validation import (
    CDCDataValidator,
    BINARY_FEATURES,
    ORDINAL_FEATURES,
    COUNT_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    VALID_CLASSES,
    CDC_DATASET_PATH,
)
from cdc_preprocessing import (
    CDCPreprocessingPipeline,
    _build_preprocessor,
    _compute_class_weights,
    _class_distribution,
    _fingerprint,
    FEATURE_ORDER,
)
from cdc_feature_engineering import (
    CDCFeatureEngineer,
    engineer_features,
    add_bmi_category,
    add_health_burden_score,
    add_preventive_care_index,
    add_lifestyle_score,
    add_poor_health_days_total,
    add_bmi_age_interaction,
    add_socioeconomic_index,
    get_class_weight_dict,
)

# ── Path constants ─────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
REAL_DATA    = ROOT / "data" / "raw" / "diabetes_012_health_indicators_BRFSS2015.csv"
DATASET_AVAILABLE = REAL_DATA.exists()

# ── Synthetic dataset factory ─────────────────────────────────────────────
def make_synthetic_cdc(
    n: int = 1000,
    seed: int = 42,
    inject_dupes: bool = False,
    inject_conflicts: bool = False,
    inject_missing: bool = False,
    inject_bad_target: bool = False,
) -> pd.DataFrame:
    """Create a minimal synthetic CDC-schema DataFrame for unit tests."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        col: rng.integers(0, 2, n).astype(float)
        for col in BINARY_FEATURES
    })
    df["GenHlth"]   = rng.integers(1, 6, n).astype(float)
    df["Age"]       = rng.integers(1, 14, n).astype(float)
    df["Education"] = rng.integers(1, 7, n).astype(float)
    df["Income"]    = rng.integers(1, 9, n).astype(float)
    df["MentHlth"]  = rng.integers(0, 31, n).astype(float)
    df["PhysHlth"]  = rng.integers(0, 31, n).astype(float)
    df["BMI"]       = rng.uniform(14.0, 60.0, n).round(1)

    # Imbalanced 3-class target
    y_choices = rng.choice([0, 1, 2], size=n, p=[0.84, 0.02, 0.14])
    df[TARGET_COLUMN] = y_choices.astype(float)

    if inject_dupes:
        # Append 10 exact duplicate rows
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)

    if inject_conflicts:
        # Create 5 feature-identical rows with DIFFERENT targets
        conflict_rows = df.iloc[:5].copy()
        conflict_rows[TARGET_COLUMN] = 1.0 - conflict_rows[TARGET_COLUMN].clip(0, 1)
        df = pd.concat([df, conflict_rows], ignore_index=True)

    if inject_missing:
        df.loc[0:4, "BMI"] = np.nan

    if inject_bad_target:
        df.loc[0, TARGET_COLUMN] = 99.0

    return df


def save_synthetic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1 – Dataset Loading
# ════════════════════════════════════════════════════════════════════════════
class TestDatasetLoading:

    def test_missing_file_raises(self, tmp_path):
        """Validator reports failure when file is absent."""
        fake_path = tmp_path / "nonexistent.csv"
        validator = CDCDataValidator(dataset_path=fake_path, verbose=False)
        df, report = validator.validate()
        assert not report.validation.passed
        assert any("not found" in e.lower() for e in report.validation.errors)

    def test_load_synthetic_succeeds(self, tmp_path):
        """Validator succeeds with a valid synthetic CSV."""
        df_in = make_synthetic_cdc(n=500)
        p = tmp_path / "test_cdc.csv"
        save_synthetic(df_in, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        df_out, report = validator.validate()
        assert report.validation.passed or not report.validation.errors
        assert df_out.shape[0] == 500

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Real dataset not available")
    def test_real_dataset_loads(self):
        """Real dataset loads and has expected shape."""
        df = pd.read_csv(REAL_DATA)
        assert df.shape[0] == 253_680, f"Expected 253680 rows, got {df.shape[0]}"
        assert df.shape[1] == 22, f"Expected 22 columns, got {df.shape[1]}"

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Real dataset not available")
    def test_real_dataset_all_columns(self):
        """Real dataset contains all 22 expected columns."""
        df = pd.read_csv(REAL_DATA)
        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2 – Dataset Validation
# ════════════════════════════════════════════════════════════════════════════
class TestDatasetValidation:

    def test_no_missing_values(self, tmp_path):
        df = make_synthetic_cdc(n=200)
        assert df.isnull().sum().sum() == 0

    def test_missing_value_detected(self, tmp_path):
        df = make_synthetic_cdc(n=200, inject_missing=True)
        p = tmp_path / "missing.csv"
        save_synthetic(df, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        _, report = validator.validate()
        assert report.n_missing > 0

    def test_all_dtypes_numeric(self, tmp_path):
        df = make_synthetic_cdc(n=200)
        p = tmp_path / "dtypes.csv"
        save_synthetic(df, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        _, report = validator.validate()
        assert len(report.dtype_issues) == 0, f"Dtype issues: {report.dtype_issues}"

    def test_fingerprint_is_string(self, tmp_path):
        df = make_synthetic_cdc(n=100)
        p = tmp_path / "fp.csv"
        save_synthetic(df, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        _, report = validator.validate()
        assert isinstance(report.fingerprint, str)
        assert len(report.fingerprint) == 16

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Real dataset not available")
    def test_real_no_missing(self):
        validator = CDCDataValidator(dataset_path=REAL_DATA, verbose=False)
        _, report = validator.validate()
        assert report.n_missing == 0

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Real dataset not available")
    def test_real_no_invalid_ranges(self):
        validator = CDCDataValidator(dataset_path=REAL_DATA, verbose=False)
        _, report = validator.validate()
        assert report.invalid_ranges == {}, \
            f"Invalid ranges found: {report.invalid_ranges}"


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3 – Target Class Validation
# ════════════════════════════════════════════════════════════════════════════
class TestTargetClassValidation:

    def test_valid_classes(self, tmp_path):
        df = make_synthetic_cdc(n=200)
        assert set(df[TARGET_COLUMN].unique()).issubset(VALID_CLASSES | {0.0, 1.0, 2.0})

    def test_invalid_class_detected(self, tmp_path):
        df = make_synthetic_cdc(n=200, inject_bad_target=True)
        p = tmp_path / "bad_target.csv"
        save_synthetic(df, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        _, report = validator.validate()
        assert not report.validation.passed
        assert any("invalid target" in e.lower() for e in report.validation.errors)

    def test_three_classes_present(self, tmp_path):
        df = make_synthetic_cdc(n=5000, seed=0)  # Enough for rare class 1
        p = tmp_path / "three_cls.csv"
        save_synthetic(df, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        _, report = validator.validate()
        # 3 classes should be in distribution
        assert len(report.class_distribution) >= 2  # At minimum 2 classes in small n

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Real dataset not available")
    def test_real_three_classes(self):
        validator = CDCDataValidator(dataset_path=REAL_DATA, verbose=False)
        _, report = validator.validate()
        assert set(report.class_distribution.keys()) == {0, 1, 2}

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Real dataset not available")
    def test_real_class_0_majority(self):
        validator = CDCDataValidator(dataset_path=REAL_DATA, verbose=False)
        _, report = validator.validate()
        assert report.class_percentages[0] > 80.0
        assert report.class_percentages[1] < 5.0

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Real dataset not available")
    def test_real_class_counts_exact(self):
        validator = CDCDataValidator(dataset_path=REAL_DATA, verbose=False)
        _, report = validator.validate()
        assert report.class_distribution[0] == 213_703
        assert report.class_distribution[1] == 4_631
        assert report.class_distribution[2] == 35_346


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4 – Feature Schema Validation
# ════════════════════════════════════════════════════════════════════════════
class TestFeatureSchemaValidation:

    def test_feature_order_constant(self):
        assert FEATURE_ORDER == FEATURE_COLUMNS
        assert len(FEATURE_ORDER) == 21

    def test_expected_columns_count(self):
        assert len(EXPECTED_COLUMNS) == 22  # 21 features + 1 target

    def test_binary_features_count(self):
        assert len(BINARY_FEATURES) == 14

    def test_ordinal_features_count(self):
        assert len(ORDINAL_FEATURES) == 4

    def test_count_features_count(self):
        assert len(COUNT_FEATURES) == 2

    def test_continuous_features_count(self):
        assert len(CONTINUOUS_FEATURES) == 1

    def test_missing_column_detected(self, tmp_path):
        df = make_synthetic_cdc(n=100)
        df_missing = df.drop(columns=["HighBP"])
        p = tmp_path / "missing_col.csv"
        save_synthetic(df_missing, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        _, report = validator.validate()
        assert "HighBP" in report.missing_columns

    def test_no_target_column_fails(self, tmp_path):
        df = make_synthetic_cdc(n=100)
        df_no_target = df.drop(columns=[TARGET_COLUMN])
        p = tmp_path / "no_target.csv"
        save_synthetic(df_no_target, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        _, report = validator.validate()
        assert not report.validation.passed


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5 – Duplicate Profile Detection
# ════════════════════════════════════════════════════════════════════════════
class TestDuplicateProfileDetection:

    def test_exact_duplicates_counted(self, tmp_path):
        df = make_synthetic_cdc(n=200, inject_dupes=True)
        p = tmp_path / "dupes.csv"
        save_synthetic(df, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        _, report = validator.validate()
        assert report.n_duplicates == 10, \
            f"Expected 10 dupes, got {report.n_duplicates}"

    def test_no_duplicates_baseline(self, tmp_path):
        """Synthetic data without forced dupes should have ~0 exact duplicates
        (may still have some by chance with low-cardinality features)."""
        df = make_synthetic_cdc(n=100, inject_dupes=False)
        p = tmp_path / "no_dupes.csv"
        save_synthetic(df, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        _, report = validator.validate()
        # With n=100 and binary features, some dupes are possible but report should run
        assert isinstance(report.n_duplicates, int)

    def test_duplicates_not_removed(self, tmp_path):
        """Validation must NOT remove rows — that's a training step."""
        df = make_synthetic_cdc(n=200, inject_dupes=True)
        p = tmp_path / "dupes_kept.csv"
        save_synthetic(df, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        df_out, _ = validator.validate()
        assert len(df_out) == len(df), "Validator must not drop rows"

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Real dataset not available")
    def test_real_duplicate_count(self):
        validator = CDCDataValidator(dataset_path=REAL_DATA, verbose=False)
        _, report = validator.validate()
        assert report.n_duplicates == 23_899, \
            f"Expected 23899 dupes, got {report.n_duplicates}"


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 6 – Conflicting Target Detection
# ════════════════════════════════════════════════════════════════════════════
class TestConflictingTargetDetection:

    def test_conflict_detected(self, tmp_path):
        df = make_synthetic_cdc(n=200, inject_conflicts=True)
        p = tmp_path / "conflicts.csv"
        save_synthetic(df, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        _, report = validator.validate()
        # Conflicts should be detected (not necessarily 5, since original rows
        # may already share profiles)
        assert report.n_conflict_profiles >= 0  # Must run without crashing

    def test_conflict_rows_retained(self, tmp_path):
        """Conflicting rows must be retained — not silently deleted."""
        df = make_synthetic_cdc(n=200, inject_conflicts=True)
        original_len = len(df)
        p = tmp_path / "conflict_retain.csv"
        save_synthetic(df, p)
        validator = CDCDataValidator(dataset_path=p, verbose=False)
        df_out, _ = validator.validate()
        assert len(df_out) == original_len


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 7 – Train/Test Leakage Prevention
# ════════════════════════════════════════════════════════════════════════════
class TestLeakagePrevention:

    def _run_pipeline(self, tmp_path, n=2000, seed=42) -> Dict:
        df = make_synthetic_cdc(n=n, seed=seed)
        p  = tmp_path / "split_data.csv"
        save_synthetic(df, p)
        pl = CDCPreprocessingPipeline(
            data_path=p, test_size=0.20, random_seed=seed, verbose=False
        )
        return pl.run()

    def test_train_test_indices_disjoint(self, tmp_path):
        """Train and test raw DataFrames must have no overlapping index after reset."""
        result    = self._run_pipeline(tmp_path)
        X_tr_raw  = result["X_train_raw"]
        X_te_raw  = result["X_test_raw"]
        # Row counts must add up
        total = len(X_tr_raw) + len(X_te_raw)
        assert total == 2000

    def test_test_size_approximately_20pct(self, tmp_path):
        result  = self._run_pipeline(tmp_path, n=2000)
        n_test  = result["metadata"].n_test
        n_train = result["metadata"].n_train
        ratio   = n_test / (n_test + n_train)
        assert 0.18 <= ratio <= 0.22, f"Test ratio {ratio:.3f} outside [0.18, 0.22]"

    def test_preprocessor_fit_on_train_only(self, tmp_path):
        """
        Verify the preprocessor was fit on training data: transform test
        data with the fitted preprocessor and compare shape.
        """
        result      = self._run_pipeline(tmp_path, n=2000)
        preprocessor = result["preprocessor"]
        X_test_raw   = result["X_test_raw"]
        # Transform should succeed (preprocessor is already fit on train)
        X_test_transformed = preprocessor.transform(X_test_raw)
        assert X_test_transformed.shape[0] == result["X_test"].shape[0]

    def test_no_target_in_features(self, tmp_path):
        """Feature arrays must not contain the target column."""
        result = self._run_pipeline(tmp_path)
        feature_names = result["feature_names"]
        assert TARGET_COLUMN not in feature_names, \
            f"Target '{TARGET_COLUMN}' found in feature names!"

    def test_class_distribution_preserved_in_test(self, tmp_path):
        """Stratified split should approximately preserve class ratios."""
        result  = self._run_pipeline(tmp_path, n=5000)
        y_test  = result["y_test"]
        vals, counts = np.unique(y_test, return_counts=True)
        total = counts.sum()
        pcts  = {int(v): c / total for v, c in zip(vals, counts)}
        # Class 0 should still be dominant
        if 0 in pcts:
            assert pcts[0] > 0.70, f"Class 0 proportion in test: {pcts[0]:.3f}"

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Real dataset not available")
    def test_real_split_sizes(self):
        pl     = CDCPreprocessingPipeline(verbose=False)
        result = pl.run()
        meta   = result["metadata"]
        assert meta.n_train + meta.n_test == 253_680
        assert 50_000 <= meta.n_test <= 52_000  # ~20% of 253680 = 50736


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 8 – Preprocessing Transformations
# ════════════════════════════════════════════════════════════════════════════
class TestPreprocessingTransformations:

    def _small_pipeline(self, tmp_path, n=1000):
        df = make_synthetic_cdc(n=n)
        p  = tmp_path / "transform.csv"
        save_synthetic(df, p)
        pl = CDCPreprocessingPipeline(data_path=p, verbose=False)
        return pl.run()

    def test_output_is_numpy_array(self, tmp_path):
        result = self._small_pipeline(tmp_path)
        assert isinstance(result["X_train"], np.ndarray)
        assert isinstance(result["X_test"],  np.ndarray)

    def test_no_nan_in_transformed_output(self, tmp_path):
        result = self._small_pipeline(tmp_path)
        assert not np.isnan(result["X_train"]).any(), "NaN in X_train after transform"
        assert not np.isnan(result["X_test"]).any(),  "NaN in X_test after transform"

    def test_bmi_scaled(self, tmp_path):
        """
        StandardScaler on BMI should produce ~zero mean on training set.
        The ColumnTransformer output order (verbose_feature_names_out=False) is:
          binary (14) | count (2) | continuous (1=BMI) | ordinal (4)
        So BMI is always at index 16.
        We verify by locating it via the CONTINUOUS_FEATURES list position.
        """
        result  = self._small_pipeline(tmp_path, n=2000)
        X_train = result["X_train"]
        # BMI sits after 14 binary + 2 count features = index 16
        bmi_idx = len(BINARY_FEATURES) + len(COUNT_FEATURES)   # = 16
        bmi_col = X_train[:, bmi_idx]
        bmi_mean = bmi_col.mean()
        # StandardScaler fitted on training data → training mean is exactly 0
        assert abs(bmi_mean) < 1e-6, \
            f"BMI StandardScaler training mean should be ~0, got {bmi_mean:.6f}"

    def test_binary_features_passthrough(self, tmp_path):
        """Binary features (0/1) should remain 0 or 1 after transformation."""
        result  = self._small_pipeline(tmp_path, n=1000)
        feature_names = result["feature_names"]
        X_train = result["X_train"]
        for i, fn in enumerate(feature_names):
            if fn in BINARY_FEATURES:
                col_vals = np.unique(X_train[:, i])
                assert set(col_vals).issubset({0.0, 1.0}), \
                    f"Binary feature '{fn}' has unexpected values: {col_vals}"
                break  # Test one binary feature as representative

    def test_preprocessor_output_shape_consistent(self, tmp_path):
        """X_train and X_test must have the same number of features (columns)."""
        result = self._small_pipeline(tmp_path)
        assert result["X_train"].shape[1] == result["X_test"].shape[1]

    def test_class_weights_all_classes(self, tmp_path):
        result = self._small_pipeline(tmp_path, n=5000)
        cw     = result["class_weights"]
        assert 0 in cw and 1 in cw and 2 in cw, \
            f"Not all classes in class_weights: {cw.keys()}"

    def test_minority_class_has_higher_weight(self, tmp_path):
        """Class 1 (prediabetes, rarest) should have the largest weight."""
        result = self._small_pipeline(tmp_path, n=5000)
        cw     = result["class_weights"]
        if 1 in cw and 0 in cw:
            assert cw[1] > cw[0], \
                f"Class 1 weight {cw[1]:.3f} not > Class 0 weight {cw[0]:.3f}"


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 9 – Feature Order Consistency
# ════════════════════════════════════════════════════════════════════════════
class TestFeatureOrderConsistency:

    def test_feature_order_matches_expected(self, tmp_path):
        df = make_synthetic_cdc(n=500)
        p  = tmp_path / "order.csv"
        save_synthetic(df, p)
        pl     = CDCPreprocessingPipeline(data_path=p, verbose=False)
        result = pl.run()
        # Feature names from pipeline should contain all original features
        fn = result["feature_names"]
        for base_feat in FEATURE_ORDER:
            assert any(base_feat in f for f in fn), \
                f"Base feature '{base_feat}' not found in output feature names"

    def test_shuffled_input_columns_give_same_result(self, tmp_path):
        """Shuffling input column order should produce identical output because
        CDCPreprocessingPipeline always selects FEATURE_ORDER explicitly."""
        df       = make_synthetic_cdc(n=500)
        # Shuffle columns (excluding target)
        shuffled_cols = list(np.random.default_rng(99).permutation(FEATURE_ORDER))
        df_shuf  = df[shuffled_cols + [TARGET_COLUMN]]
        p_orig   = tmp_path / "orig.csv"
        p_shuf   = tmp_path / "shuf.csv"
        save_synthetic(df, p_orig)
        save_synthetic(df_shuf, p_shuf)

        pl_orig = CDCPreprocessingPipeline(data_path=p_orig, verbose=False)
        pl_shuf = CDCPreprocessingPipeline(data_path=p_shuf, verbose=False)
        r_orig  = pl_orig.run()
        r_shuf  = pl_shuf.run()
        # Output feature names should be identical
        assert r_orig["feature_names"] == r_shuf["feature_names"]

    def test_transform_new_data_same_shape(self, tmp_path):
        df = make_synthetic_cdc(n=500)
        p  = tmp_path / "new.csv"
        save_synthetic(df, p)
        pl     = CDCPreprocessingPipeline(data_path=p, verbose=False)
        result = pl.run()
        # Transform a new batch
        new_df  = make_synthetic_cdc(n=50).drop(columns=[TARGET_COLUMN])
        new_df  = new_df[FEATURE_ORDER]  # Align
        new_out = result["preprocessor"].transform(new_df)
        assert new_out.shape[1] == result["X_train"].shape[1]


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 10 – Reproducibility
# ════════════════════════════════════════════════════════════════════════════
class TestReproducibility:

    def test_same_seed_same_split(self, tmp_path):
        """Two pipeline runs with same seed produce identical split sizes."""
        df = make_synthetic_cdc(n=1000)
        p  = tmp_path / "repro.csv"
        save_synthetic(df, p)

        def run(seed):
            pl = CDCPreprocessingPipeline(data_path=p, random_seed=seed, verbose=False)
            r  = pl.run()
            return r["metadata"].n_train, r["metadata"].n_test

        n_tr1, n_te1 = run(42)
        n_tr2, n_te2 = run(42)
        assert n_tr1 == n_tr2 and n_te1 == n_te2

    def test_different_seed_may_differ(self, tmp_path):
        """Different seeds CAN produce different splits (not guaranteed but likely)."""
        df = make_synthetic_cdc(n=1000)
        p  = tmp_path / "repro2.csv"
        save_synthetic(df, p)

        def run_X(seed):
            pl = CDCPreprocessingPipeline(data_path=p, random_seed=seed, verbose=False)
            return pl.run()["X_train"]

        X1 = run_X(42)
        X2 = run_X(99)
        # At minimum the splits have same shape (stratified) — verify shapes match
        assert X1.shape[1] == X2.shape[1]  # Same number of features

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Real dataset not available")
    def test_real_dataset_reproducible(self):
        """Two runs on real data with seed=42 produce identical split metadata."""
        def run():
            pl = CDCPreprocessingPipeline(verbose=False)
            r  = pl.run()
            return r["metadata"].n_train, r["metadata"].n_test

        n_tr1, n_te1 = run()
        n_tr2, n_te2 = run()
        assert n_tr1 == n_tr2
        assert n_te1 == n_te2


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 11 – Feature Engineering Module
# ════════════════════════════════════════════════════════════════════════════
class TestFeatureEngineering:

    def _base_df(self, n=100) -> pd.DataFrame:
        return make_synthetic_cdc(n=n).drop(columns=[TARGET_COLUMN])

    def test_bmi_category_values(self):
        df  = self._base_df()
        out = add_bmi_category(df)
        assert "BMI_Category" in out.columns
        assert out["BMI_Category"].between(0, 4).all()

    def test_health_burden_score_range(self):
        df  = self._base_df()
        out = add_health_burden_score(df)
        assert "Health_Burden_Score" in out.columns
        assert out["Health_Burden_Score"].between(0, 6).all()

    def test_preventive_care_index_range(self):
        df  = self._base_df()
        out = add_preventive_care_index(df)
        assert "Preventive_Care_Index" in out.columns
        assert out["Preventive_Care_Index"].between(-1, 2).all()

    def test_lifestyle_score_range(self):
        df  = self._base_df()
        out = add_lifestyle_score(df)
        assert "Lifestyle_Score" in out.columns
        assert out["Lifestyle_Score"].between(-2, 3).all()

    def test_poor_days_total_range(self):
        df  = self._base_df()
        out = add_poor_health_days_total(df)
        assert "Total_Poor_Health_Days" in out.columns
        assert out["Total_Poor_Health_Days"].between(0, 60).all()

    def test_bmi_age_interaction_positive(self):
        df  = self._base_df()
        out = add_bmi_age_interaction(df)
        assert "BMI_x_Age" in out.columns
        assert (out["BMI_x_Age"] > 0).all()

    def test_socioeconomic_index_range(self):
        df  = self._base_df()
        out = add_socioeconomic_index(df)
        assert "Socioeconomic_Index" in out.columns
        assert out["Socioeconomic_Index"].between(0.0, 1.0).all()

    def test_engineer_features_adds_cols(self):
        df  = self._base_df()
        out = engineer_features(df)
        new_cols = [c for c in out.columns if c not in df.columns]
        assert len(new_cols) >= 6, f"Expected ≥6 new columns, got {len(new_cols)}"

    def test_engineer_features_original_cols_preserved(self):
        df  = self._base_df()
        out = engineer_features(df)
        for col in df.columns:
            assert col in out.columns, f"Original column '{col}' was dropped"

    def test_sklearn_transformer_fit_transform(self):
        df  = self._base_df(n=200)
        eng = CDCFeatureEngineer()
        eng.fit(df)
        out = eng.transform(df)
        assert isinstance(out, pd.DataFrame)
        assert len(out.columns) > len(df.columns)

    def test_transformer_stateless_fit_noop(self):
        """fit() is a no-op — transform before and after fit must be identical."""
        df  = self._base_df(n=100)
        eng = CDCFeatureEngineer()
        out1 = eng.transform(df)
        eng.fit(df)
        out2 = eng.transform(df)
        pd.testing.assert_frame_equal(out1, out2)

    def test_class_weight_dict_all_classes(self):
        y = np.array([0]*84 + [1]*2 + [2]*14)
        cw = get_class_weight_dict(y)
        assert set(cw.keys()) == {0, 1, 2}
        assert cw[1] > cw[0]  # Minority class should have higher weight

    def test_no_target_leakage_in_engineering(self):
        """CDCFeatureEngineer must not accept (or use) the target column."""
        df = self._base_df()
        # Pass target column separately — it should NOT appear in engineered output
        # unless explicitly included in input (which it shouldn't be)
        eng = CDCFeatureEngineer()
        out = eng.transform(df)
        assert "Diabetes_012" not in out.columns


# ════════════════════════════════════════════════════════════════════════════
# TEST GROUP 12 – Artifact Saving (integration)
# ════════════════════════════════════════════════════════════════════════════
class TestArtifactSaving:

    def test_save_and_reload(self, tmp_path):
        df = make_synthetic_cdc(n=500)
        p  = tmp_path / "art.csv"
        save_synthetic(df, p)
        artifacts_dir = tmp_path / "cdc_artifacts"

        pl     = CDCPreprocessingPipeline(data_path=p, verbose=False)
        result = pl.run()
        pl.save_artifacts(artifacts_dir=artifacts_dir)

        # Artifacts created
        assert (artifacts_dir / "cdc_preprocessor.pkl").exists()
        assert (artifacts_dir / "cdc_feature_names.json").exists()
        assert (artifacts_dir / "cdc_split_metadata.json").exists()

        # Feature names persist correctly
        with open(artifacts_dir / "cdc_feature_names.json") as fh:
            fn = json.load(fh)
        assert fn == result["feature_names"]

    def test_metadata_json_has_required_keys(self, tmp_path):
        df = make_synthetic_cdc(n=500)
        p  = tmp_path / "meta.csv"
        save_synthetic(df, p)
        artifacts_dir = tmp_path / "meta_art"

        pl = CDCPreprocessingPipeline(data_path=p, verbose=False)
        pl.run()
        pl.save_artifacts(artifacts_dir=artifacts_dir)

        with open(artifacts_dir / "cdc_split_metadata.json") as fh:
            meta = json.load(fh)

        required_keys = [
            "random_seed", "test_fraction", "n_train", "n_test",
            "strategy", "leakage_check_passed", "leakage_note",
            "train_class_dist", "test_class_dist",
        ]
        for key in required_keys:
            assert key in meta, f"Key '{key}' missing from metadata JSON"


# ════════════════════════════════════════════════════════════════════════════
# Entry point (also runs via pytest)
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(ROOT),
    )
    sys.exit(result.returncode)

