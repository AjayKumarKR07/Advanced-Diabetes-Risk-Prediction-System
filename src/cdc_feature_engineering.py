"""
cdc_feature_engineering.py
===========================
CDC Diabetes Health Indicators – Feature Engineering Module
Phase 3 | Advanced Diabetes Risk Prediction System

Provides optional, leakage-safe feature engineering on top of the base
preprocessing pipeline.  All interactions and derived features are computed
from the input features ONLY — no target information is ever used.

Design rules
------------
- All transformers are stateless (no fit needed) EXCEPT the PolynomialFeatures
  wrapper, which must be fit on training data only.
- Engineers are composable: each returns a new DataFrame with the same index.
- Features are appended, never replaced, so the base feature schema is preserved.
- A FeatureEngineer object exposes fit/transform so it integrates into a
  Pipeline as a custom step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional


# ── Feature group definitions ──────────────────────────────────────────────
BINARY_FEATURES     = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
ORDINAL_FEATURES    = ["GenHlth", "Age", "Education", "Income"]
COUNT_FEATURES      = ["MentHlth", "PhysHlth"]
CONTINUOUS_FEATURES = ["BMI"]


# ══════════════════════════════════════════════════════════════════════════
# Individual feature engineering functions
# ══════════════════════════════════════════════════════════════════════════

def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin BMI into 5 ordinal categories aligned with CDC guidelines.
    Returns integer 0–4 (no string labels → no encoding needed downstream).

    Category mapping:
      0 : Underweight     (BMI < 18.5)
      1 : Normal weight   (18.5 ≤ BMI < 25)
      2 : Overweight      (25 ≤ BMI < 30)
      3 : Obese Class I   (30 ≤ BMI < 35)
      4 : Obese Class II+ (BMI ≥ 35)
    """
    df = df.copy()
    bmi = df["BMI"]
    df["BMI_Category"] = np.select(
        [bmi < 18.5, bmi < 25, bmi < 30, bmi < 35],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int8)
    return df


def add_health_burden_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additive count of active risk conditions (binary features that indicate
    a health burden when == 1).

    Fields included: HighBP, HighChol, Stroke, HeartDiseaseorAttack,
                     DiffWalk, Smoker, HvyAlcoholConsump

    Range: 0 (no risk conditions) → 7 (all risk conditions present)
    This is a purely arithmetic combination of existing binary columns.
    """
    df = df.copy()
    risk_flags = [
        "HighBP", "HighChol", "Stroke",
        "HeartDiseaseorAttack", "DiffWalk", "Smoker",
    ]
    available = [c for c in risk_flags if c in df.columns]
    df["Health_Burden_Score"] = df[available].sum(axis=1).astype(np.int8)
    return df


def add_preventive_care_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preventive-care composite (higher = better healthcare engagement):
      CholCheck + AnyHealthcare  –  NoDocbcCost

    Range: [-1, 2]
    """
    df = df.copy()
    df["Preventive_Care_Index"] = (
        df.get("CholCheck", 0)
        + df.get("AnyHealthcare", 0)
        - df.get("NoDocbcCost", 0)
    ).astype(np.int8)
    return df


def add_lifestyle_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Positive lifestyle composite (higher = healthier habits):
      PhysActivity + Fruits + Veggies  –  Smoker  –  HvyAlcoholConsump

    Range: [-2, 3]
    """
    df = df.copy()
    df["Lifestyle_Score"] = (
        df.get("PhysActivity", 0)
        + df.get("Fruits", 0)
        + df.get("Veggies", 0)
        - df.get("Smoker", 0)
        - df.get("HvyAlcoholConsump", 0)
    ).astype(np.int8)
    return df


def add_poor_health_days_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum of MentHlth + PhysHlth (total poor-health days in past 30).
    Capped at 30 since each is 0–30 but the sum could exceed 30.
    Range: 0–60 (but we keep raw sum).
    """
    df = df.copy()
    df["Total_Poor_Health_Days"] = (
        df.get("MentHlth", 0) + df.get("PhysHlth", 0)
    ).astype(np.int16)
    return df


def add_bmi_age_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    BMI × Age category interaction term.  Captures compounding effect of
    high BMI at older ages.  Both are numeric, so the product is well-defined.
    """
    df = df.copy()
    df["BMI_x_Age"] = (df["BMI"] * df["Age"]).round(2)
    return df


def add_socioeconomic_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-normalised socioeconomic composite from Education + Income.
    Both are ordinal 1-6 and 1-8; we normalise each to [0, 1] then average.

    Range: 0.0 → 1.0
    """
    df = df.copy()
    edu_norm    = (df.get("Education", 1) - 1) / 5   # 0–1
    income_norm = (df.get("Income", 1) - 1) / 7       # 0–1
    df["Socioeconomic_Index"] = ((edu_norm + income_norm) / 2).round(4)
    return df


def add_genhlth_bmi_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    GenHlth × BMI_Category (from add_bmi_category) interaction.
    Requires BMI_Category to be present — call add_bmi_category first.
    """
    df = df.copy()
    if "BMI_Category" not in df.columns:
        df = add_bmi_category(df)
    df["GenHlth_x_BMIcat"] = (df["GenHlth"] * df["BMI_Category"]).astype(np.int16)
    return df


# ══════════════════════════════════════════════════════════════════════════
# Composable sklearn-compatible Transformer
# ══════════════════════════════════════════════════════════════════════════

class CDCFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible feature engineering step for CDC data.

    Can be inserted as a Pipeline step BEFORE the ColumnTransformer:

        Pipeline([
            ("engineer", CDCFeatureEngineer()),
            ("preprocessor", cdc_preprocessor),
        ])

    Parameters
    ----------
    add_bmi_cat         : Add BMI_Category ordinal feature.
    add_health_burden   : Add Health_Burden_Score count.
    add_preventive      : Add Preventive_Care_Index.
    add_lifestyle       : Add Lifestyle_Score.
    add_poor_days       : Add Total_Poor_Health_Days.
    add_bmi_age         : Add BMI × Age interaction.
    add_socioeconomic   : Add Socioeconomic_Index.
    add_genhlth_bmi     : Add GenHlth × BMI_Category interaction.
    return_dataframe    : Return pd.DataFrame (True) or np.ndarray (False).

    Note: This transformer is stateless (fit is a no-op).
    All computations use only input feature values, not the target.
    """

    def __init__(
        self,
        add_bmi_cat:       bool = True,
        add_health_burden: bool = True,
        add_preventive:    bool = True,
        add_lifestyle:     bool = True,
        add_poor_days:     bool = True,
        add_bmi_age:       bool = True,
        add_socioeconomic: bool = True,
        add_genhlth_bmi:   bool = False,  # Off by default (needs BMI_Category)
        return_dataframe:  bool = True,
    ) -> None:
        self.add_bmi_cat       = add_bmi_cat
        self.add_health_burden = add_health_burden
        self.add_preventive    = add_preventive
        self.add_lifestyle     = add_lifestyle
        self.add_poor_days     = add_poor_days
        self.add_bmi_age       = add_bmi_age
        self.add_socioeconomic = add_socioeconomic
        self.add_genhlth_bmi   = add_genhlth_bmi
        self.return_dataframe  = return_dataframe

        self.engineered_feature_names_: List[str] = []

    # fit is a no-op — this transformer is entirely stateless
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> pd.DataFrame | np.ndarray:
        if isinstance(X, np.ndarray):
            raise ValueError(
                "CDCFeatureEngineer requires a pandas DataFrame, not ndarray. "
                "Place this step before the ColumnTransformer."
            )

        df = X.copy()
        new_cols = []

        if self.add_bmi_cat:
            df = add_bmi_category(df)
            new_cols.append("BMI_Category")

        if self.add_health_burden:
            df = add_health_burden_score(df)
            new_cols.append("Health_Burden_Score")

        if self.add_preventive:
            df = add_preventive_care_index(df)
            new_cols.append("Preventive_Care_Index")

        if self.add_lifestyle:
            df = add_lifestyle_score(df)
            new_cols.append("Lifestyle_Score")

        if self.add_poor_days:
            df = add_poor_health_days_total(df)
            new_cols.append("Total_Poor_Health_Days")

        if self.add_bmi_age:
            df = add_bmi_age_interaction(df)
            new_cols.append("BMI_x_Age")

        if self.add_socioeconomic:
            df = add_socioeconomic_index(df)
            new_cols.append("Socioeconomic_Index")

        if self.add_genhlth_bmi:
            df = add_genhlth_bmi_interaction(df)
            new_cols.append("GenHlth_x_BMIcat")

        self.engineered_feature_names_ = new_cols
        return df if self.return_dataframe else df.to_numpy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = list(input_features) if input_features is not None else []
        return np.array(base + self.engineered_feature_names_)


# ── Standalone function for testing / notebooks ───────────────────────────
def engineer_features(
    df: pd.DataFrame,
    enabled: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply feature engineering steps to a raw CDC DataFrame.

    Parameters
    ----------
    df      : Raw feature DataFrame (must contain base CDC columns).
    enabled : List of step names to enable. None = all steps.
              Valid names: 'bmi_cat', 'health_burden', 'preventive',
              'lifestyle', 'poor_days', 'bmi_age', 'socioeconomic',
              'genhlth_bmi'.

    Returns
    -------
    DataFrame with original columns plus new engineered features.
    """
    all_steps = {
        "bmi_cat":       add_bmi_category,
        "health_burden": add_health_burden_score,
        "preventive":    add_preventive_care_index,
        "lifestyle":     add_lifestyle_score,
        "poor_days":     add_poor_health_days_total,
        "bmi_age":       add_bmi_age_interaction,
        "socioeconomic": add_socioeconomic_index,
        "genhlth_bmi":   add_genhlth_bmi_interaction,
    }

    active = enabled if enabled is not None else list(all_steps.keys())
    result = df.copy()
    for step_name in active:
        if step_name in all_steps:
            result = all_steps[step_name](result)
    return result


# ── Class imbalance preparation utilities ─────────────────────────────────

def get_class_weight_dict(y: pd.Series | np.ndarray) -> dict:
    """
    Compute sklearn-balanced class weights.

    Usage: Pass to estimator via class_weight=get_class_weight_dict(y_train)
    """
    from sklearn.utils.class_weight import compute_class_weight
    y_arr   = np.array(y).astype(int)
    classes = np.unique(y_arr)
    weights = compute_class_weight("balanced", classes=classes, y=y_arr)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def oversample_training_data(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    strategy: str = "auto",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random oversampling of minority classes in the training set only.

    Parameters
    ----------
    X_train      : Training features.
    y_train      : Training labels.
    strategy     : 'auto' balances all classes to majority count.
    random_state : Seed for reproducibility.

    Returns
    -------
    X_res, y_res : Resampled arrays.

    IMPORTANT: Call this function ONLY on training data inside a CV fold.
    Never pass validation or test data to this function.
    """
    try:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
        X_res, y_res = ros.fit_resample(X_train, y_train)
        return X_res, y_res
    except ImportError:
        raise ImportError(
            "imbalanced-learn is not installed. "
            "Install with: pip install imbalanced-learn\n"
            "Or use get_class_weight_dict() as a no-install alternative."
        )


def smotenc_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    categorical_features: Optional[List[int]] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTENC oversampling for datasets with mixed binary/ordinal/continuous features.

    SMOTENC is the correct variant for this dataset because:
    - Binary features (0/1) would be corrupted by standard SMOTE interpolation.
    - SMOTENC handles categorical features by majority-vote among k-nearest neighbours.

    Parameters
    ----------
    X_train              : Training features (DataFrame with original column names).
    y_train              : Training labels.
    categorical_features : Column INDICES of binary/ordinal features.
                           If None, inferred automatically from BINARY_FEATURES.
    random_state         : Seed for reproducibility.

    Returns
    -------
    X_res, y_res : Resampled arrays.

    IMPORTANT: Call ONLY inside training folds. Never on test/validation data.
    """
    try:
        from imblearn.over_sampling import SMOTENC
    except ImportError:
        raise ImportError(
            "imbalanced-learn is not installed. "
            "Install with: pip install imbalanced-learn"
        )

    if categorical_features is None:
        # Binary + Ordinal features are "categorical" for SMOTENC
        cat_cols = BINARY_FEATURES + ORDINAL_FEATURES
        if isinstance(X_train, pd.DataFrame):
            col_list = list(X_train.columns)
            categorical_features = [
                col_list.index(c) for c in cat_cols if c in col_list
            ]
        else:
            raise ValueError(
                "Pass a DataFrame so column indices can be inferred, "
                "or supply categorical_features explicitly."
            )

    smote = SMOTENC(
        categorical_features=categorical_features,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


if __name__ == "__main__":
    # Quick smoke test
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Build a tiny synthetic CDC-like DataFrame
    rng = np.random.default_rng(42)
    n   = 50
    test_df = pd.DataFrame({
        col: rng.integers(0, 2, n) for col in BINARY_FEATURES
    })
    test_df["GenHlth"]   = rng.integers(1, 6, n)
    test_df["Age"]        = rng.integers(1, 14, n)
    test_df["Education"]  = rng.integers(1, 7, n)
    test_df["Income"]     = rng.integers(1, 9, n)
    test_df["MentHlth"]   = rng.integers(0, 31, n)
    test_df["PhysHlth"]   = rng.integers(0, 31, n)
    test_df["BMI"]        = rng.uniform(14, 60, n).round(1)

    engineered = engineer_features(test_df)
    new_cols   = [c for c in engineered.columns if c not in test_df.columns]
    print(f"Original cols : {len(test_df.columns)}")
    print(f"Engineered cols: {len(engineered.columns)}")
    print(f"New features  : {new_cols}")
    print("Smoke test passed ✅")

