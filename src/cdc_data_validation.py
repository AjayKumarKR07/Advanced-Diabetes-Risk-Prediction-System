"""
cdc_data_validation.py
======================
CDC Diabetes Health Indicators – Dataset Validation Module
Phase 3 | Advanced Diabetes Risk Prediction System

Validates the BRFSS 2015 three-class diabetes dataset without touching or
modifying any of the original project's Pima/Kaggle workflow:
  - src/data_preprocessing.py   (untouched)
  - src/train_model.py          (untouched)
  - src/predict.py              (untouched)
  - src/utils.py                (untouched)
  - src/app.py                  (untouched)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────
CDC_DATASET_PATH = Path("data/raw/diabetes_012_health_indicators_BRFSS2015.csv")
TARGET_COLUMN    = "Diabetes_012"
VALID_CLASSES    = {0.0, 1.0, 2.0}
EXPECTED_ROWS    = 253_680

# Feature definitions (Phase 2 findings)
BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
ORDINAL_FEATURES  = ["GenHlth", "Age", "Education", "Income"]
COUNT_FEATURES    = ["MentHlth", "PhysHlth"]
CONTINUOUS_FEATURES = ["BMI"]
FEATURE_COLUMNS   = BINARY_FEATURES + ORDINAL_FEATURES + COUNT_FEATURES + CONTINUOUS_FEATURES
EXPECTED_COLUMNS  = FEATURE_COLUMNS + [TARGET_COLUMN]

# Valid value ranges per feature
VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "HighBP":               (0, 1),
    "HighChol":             (0, 1),
    "CholCheck":            (0, 1),
    "BMI":                  (10, 100),
    "Smoker":               (0, 1),
    "Stroke":               (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity":         (0, 1),
    "Fruits":               (0, 1),
    "Veggies":              (0, 1),
    "HvyAlcoholConsump":    (0, 1),
    "AnyHealthcare":        (0, 1),
    "NoDocbcCost":          (0, 1),
    "GenHlth":              (1, 5),
    "MentHlth":             (0, 30),
    "PhysHlth":             (0, 30),
    "DiffWalk":             (0, 1),
    "Sex":                  (0, 1),
    "Age":                  (1, 13),
    "Education":            (1, 6),
    "Income":               (1, 8),
    TARGET_COLUMN:          (0, 2),
}


# ── Validation result container ────────────────────────────────────────────
@dataclass
class ValidationResult:
    """Stores all validation findings for a single dataset check."""
    passed:   bool  = True
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info:     List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.info.append(msg)


@dataclass
class DatasetReport:
    """Aggregated report produced by CDCDataValidator."""
    fingerprint:          str  = ""
    n_rows:               int  = 0
    n_cols:               int  = 0
    n_missing:            int  = 0
    n_duplicates:         int  = 0
    n_duplicate_profiles: int  = 0
    n_conflict_profiles:  int  = 0
    conflict_rows:        int  = 0
    class_distribution:   Dict[int, int]   = field(default_factory=dict)
    class_percentages:    Dict[int, float] = field(default_factory=dict)
    invalid_ranges:       Dict[str, int]   = field(default_factory=dict)
    dtype_issues:         List[str]        = field(default_factory=list)
    missing_columns:      List[str]        = field(default_factory=list)
    extra_columns:        List[str]        = field(default_factory=list)
    validation:           ValidationResult = field(default_factory=ValidationResult)
    duplicate_profile_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    conflict_profile_df:  Optional[pd.DataFrame] = field(default=None, repr=False)


# ── Validator class ────────────────────────────────────────────────────────
class CDCDataValidator:
    """
    Validates the CDC BRFSS 2015 diabetes dataset.

    Usage
    -----
    >>> validator = CDCDataValidator()
    >>> df, report = validator.validate()
    >>> print(report.validation.passed)
    """

    def __init__(
        self,
        dataset_path: Path | str = CDC_DATASET_PATH,
        verbose: bool = True,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.verbose      = verbose

    # ── Public entry point ─────────────────────────────────────────────────
    def validate(self) -> Tuple[pd.DataFrame, DatasetReport]:
        """
        Run all validation checks.

        Returns
        -------
        df     : Raw loaded DataFrame (unmodified).
        report : DatasetReport with all findings.
        """
        report = DatasetReport()
        result = report.validation

        self._log("=" * 60)
        self._log("CDC DATASET VALIDATION")
        self._log("=" * 60)

        # ── 1. File existence ──────────────────────────────────────────────
        if not self._check_file_exists(result):
            return pd.DataFrame(), report

        # ── 2. Load ────────────────────────────────────────────────────────
        df = self._load(result, report)
        if df is None:
            return pd.DataFrame(), report

        # ── 3. Schema ──────────────────────────────────────────────────────
        self._check_schema(df, result, report)

        # ── 4. Target ──────────────────────────────────────────────────────
        self._check_target(df, result, report)

        # ── 5. Data types ─────────────────────────────────────────────────
        self._check_dtypes(df, result, report)

        # ── 6. Missing values ─────────────────────────────────────────────
        self._check_missing(df, result, report)

        # ── 7. Value ranges ───────────────────────────────────────────────
        self._check_ranges(df, result, report)

        # ── 8. Exact duplicates ───────────────────────────────────────────
        self._check_duplicates(df, result, report)

        # ── 9. Duplicate feature profiles (input-only) ────────────────────
        self._check_duplicate_profiles(df, result, report)

        # ── 10. Conflicting profiles ──────────────────────────────────────
        self._check_conflicting_profiles(df, result, report)

        # ── 11. Summary ───────────────────────────────────────────────────
        self._summarise(result)

        return df, report

    # ── Private checks ─────────────────────────────────────────────────────
    def _check_file_exists(self, result: ValidationResult) -> bool:
        if not self.dataset_path.exists():
            result.fail(
                f"Dataset not found: {self.dataset_path}\n"
                "  → Place the file at data/raw/diabetes_012_health_indicators_BRFSS2015.csv"
            )
            return False
        self._log(f"✅ File found: {self.dataset_path}")
        return True

    def _load(
        self,
        result: ValidationResult,
        report: DatasetReport,
    ) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(self.dataset_path)
        except Exception as exc:
            result.fail(f"Failed to read CSV: {exc}")
            return None

        report.n_rows = len(df)
        report.n_cols = len(df.columns)

        # Fingerprint (SHA-256 of raw bytes, truncated)
        with open(self.dataset_path, "rb") as fh:
            report.fingerprint = hashlib.sha256(fh.read()).hexdigest()[:16]

        self._log(f"✅ Loaded: {report.n_rows:,} rows × {report.n_cols} cols")
        self._log(f"   Fingerprint (SHA-256 prefix): {report.fingerprint}")

        if report.n_rows != EXPECTED_ROWS:
            result.warn(
                f"Row count {report.n_rows:,} ≠ expected {EXPECTED_ROWS:,}. "
                "Validation continues."
            )
        return df

    def _check_schema(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        report: DatasetReport,
    ) -> None:
        present        = set(df.columns)
        expected       = set(EXPECTED_COLUMNS)
        missing        = sorted(expected - present)
        extra          = sorted(present - expected)
        report.missing_columns = missing
        report.extra_columns   = extra

        if missing:
            result.fail(f"Missing columns: {missing}")
        else:
            self._log(f"✅ Schema: all {len(EXPECTED_COLUMNS)} expected columns present")

        if extra:
            result.warn(f"Unexpected extra columns (ignored): {extra}")

    def _check_target(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        report: DatasetReport,
    ) -> None:
        if TARGET_COLUMN not in df.columns:
            result.fail(f"Target column '{TARGET_COLUMN}' not found")
            return

        unique_vals = set(df[TARGET_COLUMN].unique())
        invalid     = unique_vals - VALID_CLASSES
        if invalid:
            result.fail(f"Invalid target values: {invalid}. Expected subset of {VALID_CLASSES}")
            # Still compute distribution for the rows that are valid
        else:
            self._log(f"✅ Target classes: {sorted(unique_vals)} (all valid)")

        # Class distribution (computed even when invalid values exist)
        try:
            vc = df[TARGET_COLUMN].value_counts().sort_index()
            for cls, cnt in vc.items():
                report.class_distribution[int(cls)] = int(cnt)
                report.class_percentages[int(cls)]  = round(cnt / len(df) * 100, 4)
        except Exception:
            pass  # Non-numeric target — distribution not computable

        self._log("   Class distribution:")
        labels = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
        for cls in sorted(report.class_distribution):
            cnt = report.class_distribution[cls]
            pct = report.class_percentages[cls]
            lbl = labels.get(cls, f"Unknown({cls})")
            self._log(f"     Class {cls} ({lbl}): {cnt:>8,}  ({pct:.2f}%)")

    def _check_dtypes(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        report: DatasetReport,
    ) -> None:
        issues = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' has non-numeric dtype: {df[col].dtype}")
        report.dtype_issues = issues
        if issues:
            for issue in issues:
                result.warn(issue)
        else:
            self._log("✅ All columns are numeric dtype (float64)")

    def _check_missing(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        report: DatasetReport,
    ) -> None:
        total_null = int(df.isnull().sum().sum())
        report.n_missing = total_null
        if total_null > 0:
            null_per_col = df.isnull().sum()
            null_per_col = null_per_col[null_per_col > 0]
            result.warn(
                f"Missing values detected ({total_null:,} total):\n"
                + null_per_col.to_string()
            )
        else:
            self._log("✅ Missing values: none")

    def _check_ranges(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        report: DatasetReport,
    ) -> None:
        for col, (lo, hi) in VALID_RANGES.items():
            if col not in df.columns:
                continue
            n_bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_bad > 0:
                report.invalid_ranges[col] = n_bad
                result.warn(f"Column '{col}': {n_bad:,} values outside [{lo}, {hi}]")

        if not report.invalid_ranges:
            self._log("✅ All feature values within expected ranges")

    def _check_duplicates(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        report: DatasetReport,
    ) -> None:
        n_dupes = int(df.duplicated().sum())
        report.n_duplicates = n_dupes
        pct = n_dupes / len(df) * 100
        self._log(f"ℹ️  Exact duplicate rows: {n_dupes:,} ({pct:.2f}%)")
        if n_dupes > 0:
            result.note(
                f"{n_dupes:,} exact duplicate rows ({pct:.2f}%). "
                "Likely legitimate (see Phase 2 analysis). "
                "Records are RETAINED — use drop_duplicates() inside training folds only."
            )

    def _check_duplicate_profiles(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        report: DatasetReport,
    ) -> None:
        """
        Duplicate input profiles = rows where all FEATURE columns are identical
        but we ignore the target.  These tell us about survey response collisions.
        """
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        profile_counts = df.groupby(feature_cols).size().reset_index(name="_count")
        dup_profiles   = profile_counts[profile_counts["_count"] > 1]
        report.n_duplicate_profiles = len(dup_profiles)

        self._log(
            f"ℹ️  Distinct feature profiles that appear >1 time: "
            f"{report.n_duplicate_profiles:,}"
        )
        if report.n_duplicate_profiles > 0:
            result.note(
                f"{report.n_duplicate_profiles:,} unique feature profiles appear in "
                "multiple rows. These are accounted for in the leakage-aware split strategy."
            )
            report.duplicate_profile_df = dup_profiles

    def _check_conflicting_profiles(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        report: DatasetReport,
    ) -> None:
        """
        Conflicting profiles = same feature vector, different target labels.
        These cannot be disambiguated by any feature-based model.
        """
        if TARGET_COLUMN not in df.columns:
            result.note("Cannot check conflicting profiles — target column absent")
            return

        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        grouped = df.groupby(feature_cols)[TARGET_COLUMN].nunique().reset_index(
            name="_n_targets"
        )
        conflicts = grouped[grouped["_n_targets"] > 1]
        report.n_conflict_profiles = len(conflicts)

        # Count how many raw rows are involved
        if len(conflicts) > 0:
            merged = df.merge(
                conflicts[feature_cols], on=feature_cols, how="inner"
            )
            report.conflict_rows = len(merged)
            report.conflict_profile_df = conflicts

            result.warn(
                f"{report.n_conflict_profiles:,} feature profiles have conflicting "
                f"target labels (irreducible noise). "
                f"Involves {report.conflict_rows:,} rows. "
                "These rows are RETAINED but represent Bayes-error noise."
            )
        else:
            self._log("✅ No conflicting feature profiles (no label noise detected)")

    def _summarise(self, result: ValidationResult) -> None:
        self._log("\n" + "─" * 60)
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        self._log(f"Validation: {status}")
        if result.errors:
            self._log(f"  Errors   ({len(result.errors)}): ")
            for e in result.errors:
                self._log(f"    ✗ {e}")
        if result.warnings:
            self._log(f"  Warnings ({len(result.warnings)}): ")
            for w in result.warnings:
                self._log(f"    ⚠ {w}")
        if result.info:
            self._log(f"  Notes    ({len(result.info)}): ")
            for n in result.info:
                self._log(f"    ℹ {n}")

    # ── Helpers ────────────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── Report serialisation ───────────────────────────────────────────────
    def save_report(
        self,
        report: DatasetReport,
        path: Path | str = "results/reports/cdc_validation_detail.json",
    ) -> None:
        """Save a JSON-serialisable snapshot of the report."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "fingerprint":          report.fingerprint,
            "n_rows":               report.n_rows,
            "n_cols":               report.n_cols,
            "n_missing":            report.n_missing,
            "n_duplicates":         report.n_duplicates,
            "n_duplicate_profiles": report.n_duplicate_profiles,
            "n_conflict_profiles":  report.n_conflict_profiles,
            "conflict_rows":        report.conflict_rows,
            "class_distribution":   report.class_distribution,
            "class_percentages":    report.class_percentages,
            "invalid_ranges":       report.invalid_ranges,
            "dtype_issues":         report.dtype_issues,
            "missing_columns":      report.missing_columns,
            "extra_columns":        report.extra_columns,
            "validation_passed":    report.validation.passed,
            "errors":               report.validation.errors,
            "warnings":             report.validation.warnings,
            "notes":                report.validation.info,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, default=str)
        print(f"Validation report saved → {path}")


# ── Convenience function ───────────────────────────────────────────────────
def validate_cdc_dataset(
    dataset_path: Path | str = CDC_DATASET_PATH,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, DatasetReport]:
    """One-liner convenience wrapper."""
    validator = CDCDataValidator(dataset_path=dataset_path, verbose=verbose)
    return validator.validate()


if __name__ == "__main__":
    df, report = validate_cdc_dataset()
    if not report.validation.passed:
        raise SystemExit("Dataset validation failed — see errors above.")
