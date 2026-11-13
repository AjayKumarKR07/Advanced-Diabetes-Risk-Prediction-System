"""
backend/app/schemas/prediction.py
=================================
Pydantic data schemas for request validation and structured API responses.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field, field_validator


class DiabetesFeatureInput(BaseModel):
    """
    Input schema strictly validating the 21 CDC BRFSS 2015 health indicator features.
    Enforces clinical value boundaries and categorical ranges.
    """
    HighBP: int = Field(
        ..., ge=0, le=1, description="High Blood Pressure: 0 = No, 1 = Yes", json_schema_extra={"example": 1}
    )
    HighChol: int = Field(
        ..., ge=0, le=1, description="High Cholesterol: 0 = No, 1 = Yes", json_schema_extra={"example": 1}
    )
    CholCheck: int = Field(
        ..., ge=0, le=1, description="Cholesterol checked within past 5 years: 0 = No, 1 = Yes", json_schema_extra={"example": 1}
    )
    BMI: float = Field(
        ..., ge=10.0, le=100.0, description="Body Mass Index (BMI): 10.0 to 100.0 kg/m²", json_schema_extra={"example": 28.5}
    )
    Smoker: int = Field(
        ..., ge=0, le=1, description="Smoked at least 100 cigarettes in lifetime: 0 = No, 1 = Yes", json_schema_extra={"example": 0}
    )
    Stroke: int = Field(
        ..., ge=0, le=1, description="Ever diagnosed with a stroke: 0 = No, 1 = Yes", json_schema_extra={"example": 0}
    )
    HeartDiseaseorAttack: int = Field(
        ..., ge=0, le=1, description="Coronary heart disease (CHD) or myocardial infarction (MI): 0 = No, 1 = Yes", json_schema_extra={"example": 0}
    )
    PhysActivity: int = Field(
        ..., ge=0, le=1, description="Physical activity in past 30 days (excluding job): 0 = No, 1 = Yes", json_schema_extra={"example": 1}
    )
    Fruits: int = Field(
        ..., ge=0, le=1, description="Consume fruit 1 or more times per day: 0 = No, 1 = Yes", json_schema_extra={"example": 1}
    )
    Veggies: int = Field(
        ..., ge=0, le=1, description="Consume vegetables 1 or more times per day: 0 = No, 1 = Yes", json_schema_extra={"example": 1}
    )
    HvyAlcoholConsump: int = Field(
        ..., ge=0, le=1, description="Heavy alcohol consumption (men >14 drinks/wk, women >7 drinks/wk): 0 = No, 1 = Yes", json_schema_extra={"example": 0}
    )
    AnyHealthcare: int = Field(
        ..., ge=0, le=1, description="Have any health care coverage: 0 = No, 1 = Yes", json_schema_extra={"example": 1}
    )
    NoDocbcCost: int = Field(
        ..., ge=0, le=1, description="Could not see doctor because of cost in past 12 months: 0 = No, 1 = Yes", json_schema_extra={"example": 0}
    )
    GenHlth: int = Field(
        ..., ge=1, le=5, description="General health self-rating: 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor", json_schema_extra={"example": 3}
    )
    MentHlth: int = Field(
        ..., ge=0, le=30, description="Days of poor mental health in past 30 days: 0 to 30", json_schema_extra={"example": 2}
    )
    PhysHlth: int = Field(
        ..., ge=0, le=30, description="Days of physical illness or injury in past 30 days: 0 to 30", json_schema_extra={"example": 4}
    )
    DiffWalk: int = Field(
        ..., ge=0, le=1, description="Serious difficulty walking or climbing stairs: 0 = No, 1 = Yes", json_schema_extra={"example": 0}
    )
    Sex: int = Field(
        ..., ge=0, le=1, description="Sex: 0 = Female, 1 = Male", json_schema_extra={"example": 1}
    )
    Age: int = Field(
        ..., ge=1, le=13, description="Age category: 1=18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44, 6=45-49, 7=50-54, 8=55-59, 9=60-64, 10=65-69, 11=70-74, 12=75-79, 13=80+", json_schema_extra={"example": 8}
    )
    Education: int = Field(
        ..., ge=1, le=6, description="Education level: 1=Never attended/kindergarten to 6=College 4+ years", json_schema_extra={"example": 5}
    )
    Income: int = Field(
        ..., ge=1, le=8, description="Income level: 1=<$10k to 8=>$75k", json_schema_extra={"example": 6}
    )


class ProbabilityDistribution(BaseModel):
    """Three-class probability distribution summing to 1.0."""
    no_diabetes: float = Field(..., description="Estimated probability of No Diabetes (Class 0)")
    prediabetes: float = Field(..., description="Estimated probability of Prediabetes (Class 1)")
    diabetes: float = Field(..., description="Estimated probability of Diabetes (Class 2)")


class PredictionResponse(BaseModel):
    """Complete prediction payload returned by POST /api/predict."""
    predicted_class: int = Field(..., description="Predicted class integer: 0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes")
    predicted_label: str = Field(..., description="Human-readable classification label")
    probabilities: ProbabilityDistribution = Field(..., description="Model-estimated posterior class probabilities")
    risk_tier: str = Field(..., description="Qualitative risk classification based on prediabetes/diabetes probability")
    confidence: float = Field(..., description="Probability associated with the predicted class")
    disclaimer: str = Field(..., description="Mandatory academic research prototype disclaimer")
    model_version: str = Field(..., description="Evaluated model pipeline identifier")


class ModelInfoResponse(BaseModel):
    """Payload returned by GET /api/model-info."""
    model_name: str
    model_algorithm: str
    dataset: str
    feature_count: int
    features: List[str]
    class_mapping: Dict[int, str]
    evaluation_performance: Dict[str, Any]
    disclaimer: str


class HealthResponse(BaseModel):
    """Payload returned by GET /api/health."""
    status: str
    timestamp: str
    model_loaded: bool
    version: str

