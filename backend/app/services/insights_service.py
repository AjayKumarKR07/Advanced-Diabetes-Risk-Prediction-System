"""
backend/app/services/insights_service.py
=======================================
Transparent, rule-based educational health insights and recommendation engine (Phase 6.16 & Phase 6.16.1).
Analyzes user submitted assessment indicators (BMI, HighBP, HighChol, PhysActivity,
Smoker, GenHlth, PhysHlth, MentHlth, DiffWalk) and provides explainable, non-medical
educational suggestions, evidence citations, indicator limitations, and neutral longitudinal change tracking.
All queries strictly enforce user ownership.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.assessment import Assessment
from backend.app.schemas.insights_schemas import (
    EducationalRecommendation,
    HealthInsightsResponse,
    HistoricalComparisonItem,
    InsightsHistoryResponse,
    InsightsSummary,
)

GEN_HLTH_LABELS: Dict[int, str] = {
    1: "1 — Excellent",
    2: "2 — Very Good",
    3: "3 — Good",
    4: "4 — Fair",
    5: "5 — Poor",
}

# Authoritative evidence references
REF_CDC_BMI = (
    "Centers for Disease Control and Prevention (CDC). About Adult BMI. "
    "https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html"
)
REF_AHA_ACC_HYPERTENSION = (
    "Whelton PK, et al. 2017 ACC/AHA Guideline for the Prevention, Detection, Evaluation, and "
    "Management of High Blood Pressure in Adults. J Am Coll Cardiol. 2018;71(19):e127-e248."
)
REF_CDC_BRFSS = (
    "Centers for Disease Control and Prevention (CDC). Behavioral Risk Factor Surveillance System "
    "(BRFSS) Survey Data and Documentation."
)
REF_AHA_ACC_CHOLESTEROL = (
    "Grundy SM, et al. 2018 AHA/ACC Guideline on the Management of Blood Cholesterol. "
    "Circulation. 2019;139(25):e1082-e1143."
)
REF_WHO_PHYS_ACTIVITY = (
    "World Health Organization (WHO). WHO Guidelines on Physical Activity and Sedentary Behaviour. "
    "Geneva: World Health Organization; 2020."
)
REF_CDC_TOBACCO = (
    "Centers for Disease Control and Prevention (CDC). Tobacco Use Surveillance: Definition of "
    "Cigarette Smoking. BRFSS Survey Documentation."
)
REF_CDC_HRQOL = (
    "Centers for Disease Control and Prevention (CDC). Measuring Healthy Days: Population Assessment "
    "of Health-Related Quality of Life (HRQOL). CDC; 2000."
)
REF_CDC_DISABILITY = (
    "Centers for Disease Control and Prevention (CDC). BRFSS Functional Disability & Mobility Protocol."
)


def _generate_recommendations_for_assessment(
    assessment: Assessment,
) -> List[EducationalRecommendation]:
    """
    Generate transparent, rule-based educational recommendations from a single assessment's indicators.
    Enforces strict scientific accuracy, neutral terminology, authoritative evidence citations,
    and explicit disclosures of survey indicator limitations.
    """
    recs: List[EducationalRecommendation] = []
    eval_date = (
        assessment.created_at.strftime("%b %d, %Y")
        if assessment.created_at
        else datetime.now().strftime("%b %d, %Y")
    )
    a_id = assessment.id

    # 1. BMI Indicator (Screening classification, not medical diagnosis)
    if assessment.BMI is not None:
        bmi_val = round(float(assessment.BMI), 1)
        if bmi_val < 18.5:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_bmi_underweight_{a_id}",
                    category="Metabolic & Body",
                    priority="medium",
                    observation=f"Your submitted BMI is {bmi_val}, which falls into the underweight screening classification (<18.5 kg/m²).",
                    indicator_name="Body Mass Index (BMI)",
                    indicator_value=f"{bmi_val} kg/m²",
                    explanation=(
                        "BMI is an epidemiological screening classification based on weight and height. An adult BMI "
                        "below 18.5 may indicate lower lean body mass or nutritional intake, which can affect physiological "
                        "reserves and overall energy balance."
                    ),
                    suggestion=(
                        "Consider discussing your nutritional intake and energy requirements with a healthcare provider "
                        "or registered dietitian to support balanced nourishment."
                    ),
                    reason=f"Triggered because submitted BMI ({bmi_val}) is below 18.5 kg/m² in assessment #{a_id}.",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_BMI,
                    limitations=(
                        "BMI is a screening metric, not a diagnostic measure of individual health or nutritional status. "
                        "It does not evaluate body composition, muscle mass, bone density, or metabolic health directly."
                    ),
                )
            )
        elif bmi_val < 25.0:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_bmi_normal_{a_id}",
                    category="Metabolic & Body",
                    priority="positive",
                    observation=f"Your submitted BMI is {bmi_val}, which falls within the conventional normal weight screening classification (18.5–24.9 kg/m²).",
                    indicator_name="Body Mass Index (BMI)",
                    indicator_value=f"{bmi_val} kg/m²",
                    explanation=(
                        "A screening BMI in the 18.5–24.9 range is statistically associated with favorable baseline "
                        "metabolic and cardiovascular profiles in population-level epidemiological studies."
                    ),
                    suggestion=(
                        "Continue maintaining balanced nutritional habits, adequate hydration, and consistent physical activity "
                        "to sustain your current physical vitality."
                    ),
                    reason=f"Triggered because submitted BMI ({bmi_val}) is in the normal screening range (18.5–24.9 kg/m²).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_BMI,
                    limitations=(
                        "BMI does not measure body fat percentage or visceral adiposity directly. Individuals with higher "
                        "muscle mass or differing fat distribution may have varied metabolic profiles."
                    ),
                )
            )
        elif bmi_val < 30.0:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_bmi_overweight_{a_id}",
                    category="Metabolic & Body",
                    priority="medium",
                    observation=f"Your submitted BMI is {bmi_val}, categorized within the overweight adult screening classification (25.0–29.9 kg/m²).",
                    indicator_name="Body Mass Index (BMI)",
                    indicator_value=f"{bmi_val} kg/m²",
                    explanation=(
                        "BMI serves as a screening classification rather than a direct measure of body fat or metabolic health. "
                        "In population studies, this range is correlated with higher statistical likelihood of elevated adiposity, "
                        "though individual metabolic health varies widely."
                    ),
                    suggestion=(
                        "Focus on holistic lifestyle behaviors—such as dietary fiber intake, balanced meals, and regular physical movement—"
                        "and discuss personal health markers with your clinician during routine checkups."
                    ),
                    reason=f"Triggered because submitted BMI ({bmi_val}) is between 25.0 and 29.9 kg/m² in assessment #{a_id}.",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_BMI,
                    limitations=(
                        "BMI does not differentiate between lean muscle mass and adipose tissue, nor does it measure visceral fat distribution, "
                        "which is more metabolically active than subcutaneous fat."
                    ),
                )
            )
        else:  # >= 30.0
            recs.append(
                EducationalRecommendation(
                    id=f"rec_bmi_obesity_{a_id}",
                    category="Metabolic & Body",
                    priority="high",
                    observation=f"Your submitted BMI is {bmi_val}, which falls into the obesity adult screening classification (≥30.0 kg/m²).",
                    indicator_name="Body Mass Index (BMI)",
                    indicator_value=f"{bmi_val} kg/m²",
                    explanation=(
                        "In large surveillance cohorts, a screening BMI of 30.0 or greater is statistically associated with increased "
                        "workload on metabolic and cardiovascular systems. However, BMI remains a screening tool, not a clinical diagnosis."
                    ),
                    suggestion=(
                        "Discuss your overall health, lifestyle routines, and routine preventive evaluations with your healthcare provider "
                        "to establish personalized wellness goals."
                    ),
                    reason=f"Triggered because submitted BMI ({bmi_val}) is 30.0 kg/m² or greater in assessment #{a_id}.",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_BMI,
                    limitations=(
                        "BMI cannot diagnose metabolic dysfunction, insulin resistance, or diabetes on its own. Comprehensive evaluation "
                        "requires clinical context, laboratory testing, and physical examination by a healthcare provider."
                    ),
                )
            )

    # 2. High Blood Pressure Indicator (HighBP) — Self-reported diagnosis notification
    if assessment.HighBP is not None:
        if assessment.HighBP == 1:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_highbp_active_{a_id}",
                    category="Cardiovascular Health",
                    priority="high",
                    observation="You reported having been told by a doctor, nurse, or health professional that you have high blood pressure.",
                    indicator_name="High Blood Pressure (HighBP)",
                    indicator_value="Yes (1) — Reported diagnosis",
                    explanation=(
                        "Self-reported diagnosis reflects previous notification from a healthcare provider. Chronic elevated blood pressure "
                        "is a major independent cardiovascular risk factor and frequently co-occurs with insulin resistance and metabolic changes."
                    ),
                    suggestion=(
                        "Follow your clinician's guidance for blood pressure monitoring, discuss lifestyle strategies such as sodium moderation "
                        "and dietary approaches (like DASH principles), and adhere to any recommended medical plan."
                    ),
                    reason="Triggered because HighBP was submitted as 1 (Yes).",
                    date_generated=eval_date,
                    evidence_reference=REF_AHA_ACC_HYPERTENSION,
                    limitations=(
                        "Self-reported survey response reflects recall of previous provider advice; does not provide verified current "
                        "systolic/diastolic blood pressure measurements and cannot assess current hemodynamic control."
                    ),
                )
            )
        else:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_highbp_normal_{a_id}",
                    category="Cardiovascular Health",
                    priority="info",
                    observation="You did not report having been told by a health professional that you have high blood pressure.",
                    indicator_name="High Blood Pressure (HighBP)",
                    indicator_value="No (0) — Not told high",
                    explanation=(
                        "Not reporting a prior diagnosis indicates you have not been informed of having high blood pressure. However, "
                        "survey responses do not confirm normal clinical blood pressure, as hypertension is frequently asymptomatic until measured."
                    ),
                    suggestion=(
                        "Continue scheduling routine blood pressure checks during regular wellness visits, as clinical measurement is the "
                        "only definitive way to evaluate blood pressure."
                    ),
                    reason="Triggered because HighBP was submitted as 0 (No).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_BRFSS,
                    limitations=(
                        "Does not constitute a verified clinical measurement; undiagnosed or asymptomatic elevated blood pressure cannot "
                        "be excluded without a physical reading."
                    ),
                )
            )

    # 3. High Cholesterol Indicator (HighChol) — Self-reported diagnosis notification
    if assessment.HighChol is not None:
        if assessment.HighChol == 1:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_highchol_active_{a_id}",
                    category="Cardiovascular Health",
                    priority="high",
                    observation="You reported having been told by a doctor, nurse, or health professional that your blood cholesterol is high.",
                    indicator_name="High Cholesterol (HighChol)",
                    indicator_value="Yes (1) — Reported diagnosis",
                    explanation=(
                        "Self-reported elevated cholesterol indicates past provider notification of hyperlipidemia. Serum lipid imbalances "
                        "(elevated LDL, low HDL, elevated triglycerides) contribute to cardiovascular risk and commonly parallel metabolic dysregulation."
                    ),
                    suggestion=(
                        "Discuss your lipid profile with your doctor and explore heart-healthy nutritional patterns emphasizing soluble fiber, "
                        "unsaturated fats, and regular physical activity."
                    ),
                    reason="Triggered because HighChol was submitted as 1 (Yes).",
                    date_generated=eval_date,
                    evidence_reference=REF_AHA_ACC_CHOLESTEROL,
                    limitations=(
                        "Self-reported survey indicator reflects recall of provider advice; does not provide quantitative lipid concentrations "
                        "(total cholesterol, LDL, HDL, triglycerides)."
                    ),
                )
            )
        else:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_highchol_normal_{a_id}",
                    category="Cardiovascular Health",
                    priority="info",
                    observation="You did not report having been told by a health professional that your blood cholesterol is high.",
                    indicator_name="High Cholesterol (HighChol)",
                    indicator_value="No (0) — Not told high",
                    explanation=(
                        "Not reporting high cholesterol means you have not been diagnosed or notified of high cholesterol by a provider. "
                        "This does not represent a confirmed normal laboratory reading, which requires clinical lipid panel testing."
                    ),
                    suggestion=(
                        "Follow standard preventive guidelines for periodic fasting or non-fasting lipid screening (typically every 4 to 6 years "
                        "for low-risk adults, or as advised by your physician)."
                    ),
                    reason="Triggered because HighChol was submitted as 0 (No).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_BRFSS,
                    limitations=(
                        "Survey response depends on whether screening was performed and accurately recalled; does not guarantee normal lipid biochemistry."
                    ),
                )
            )

    # 4. Physical Activity Indicator (PhysActivity) — Any leisure-time movement
    if assessment.PhysActivity is not None:
        if assessment.PhysActivity == 0:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_physactivity_sedentary_{a_id}",
                    category="Physical Lifestyle",
                    priority="high",
                    observation="You reported no leisure-time physical activity or exercise outside of regular employment in the past 30 days.",
                    indicator_name="Physical Activity (PhysActivity)",
                    indicator_value="Inactive (0)",
                    explanation=(
                        "Sedentary leisure time is associated with lower insulin sensitivity and reduced muscular glucose uptake. "
                        "Consistent physical activity is a primary lifestyle pillar for metabolic and cardiovascular well-being."
                    ),
                    suggestion=(
                        "Consider gradually integrating moderate physical activity (such as brisk walking for 15–30 minutes daily toward "
                        "the recommended 150 minutes weekly) after consulting your medical provider."
                    ),
                    reason="Triggered because PhysActivity was submitted as 0 (No leisure physical activity).",
                    date_generated=eval_date,
                    evidence_reference=REF_WHO_PHYS_ACTIVITY,
                    limitations=(
                        "Single binary question assesses any leisure activity in the past 30 days; does not capture occupational activity, "
                        "duration, intensity, or resistance exercise."
                    ),
                )
            )
        else:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_physactivity_active_{a_id}",
                    category="Physical Lifestyle",
                    priority="positive",
                    observation="You reported participating in physical activity or exercise outside of regular employment in the past 30 days.",
                    indicator_name="Physical Activity (PhysActivity)",
                    indicator_value="Active (1)",
                    explanation=(
                        "Regular physical activity stimulates muscular glucose transport and supports metabolic flexibility, "
                        "cardiorespiratory fitness, and vascular function."
                    ),
                    suggestion=(
                        "Maintain your active routine and consider incorporating varied activities such as aerobic movement and "
                        "muscle-strengthening exercises on two or more days per week."
                    ),
                    reason="Triggered because PhysActivity was submitted as 1 (Active).",
                    date_generated=eval_date,
                    evidence_reference=REF_WHO_PHYS_ACTIVITY,
                    limitations=(
                        "Self-reported leisure activity does not measure precise metabolic equivalents (MET-hours) or exercise intensity."
                    ),
                )
            )

    # 5. Smoking Status (Smoker) — Lifetime threshold of 100 cigarettes
    if assessment.Smoker is not None:
        if assessment.Smoker == 1:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_smoker_history_{a_id}",
                    category="General Well-being",
                    priority="medium",
                    observation="You reported having smoked at least 100 cigarettes in your lifetime.",
                    indicator_name="Tobacco Exposure (Smoker)",
                    indicator_value="Yes (1) — ≥100 cigarettes",
                    explanation=(
                        "Under CDC BRFSS definitions, smoking at least 100 cigarettes in a lifetime marks a recognized threshold of cumulative "
                        "tobacco exposure. Historical or ongoing tobacco smoke exposure promotes oxidative stress and endothelial inflammation."
                    ),
                    suggestion=(
                        "If you currently smoke, speak with a healthcare provider about evidence-based cessation strategies. If you have "
                        "already quit, maintaining tobacco abstinence continues to reduce cardiovascular and metabolic risk."
                    ),
                    reason="Triggered because Smoker was reported as 1 (≥100 lifetime cigarettes).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_TOBACCO,
                    limitations=(
                        "Measures lifetime cumulative threshold (≥100 cigarettes); does not capture current daily smoking status, years since "
                        "quitting, pack-years, or alternative nicotine products."
                    ),
                )
            )
        else:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_smoker_nonsmoker_{a_id}",
                    category="General Well-being",
                    priority="positive",
                    observation="You did not report having smoked at least 100 cigarettes in your lifetime.",
                    indicator_name="Tobacco Exposure (Smoker)",
                    indicator_value="No (0) — <100 cigarettes",
                    explanation=(
                        "This response indicates you have not reached the CDC lifetime threshold of 100 cigarettes. It does not prove that "
                        "you have never smoked or are completely tobacco-free, but indicates lower cumulative exposure relative to the survey threshold."
                    ),
                    suggestion=(
                        "Continue to avoid initiating tobacco or nicotine product use, and minimize exposure to secondhand smoke environments."
                    ),
                    reason="Triggered because Smoker was reported as 0 (<100 lifetime cigarettes).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_TOBACCO,
                    limitations=(
                        "Does not distinguish between complete lifetime abstinence, light/occasional smoking (<100 cigarettes), or "
                        "non-cigarette tobacco/nicotine use."
                    ),
                )
            )

    # 6. General Health Rating (GenHlth) — Subjective self-rating (1–5)
    if assessment.GenHlth is not None:
        gh_label = GEN_HLTH_LABELS.get(assessment.GenHlth, f"Rating {assessment.GenHlth}")
        if assessment.GenHlth >= 4:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_genhlth_poor_{a_id}",
                    category="General Well-being",
                    priority="high",
                    observation=f"You self-rated your general health as '{gh_label}' on a 5-point survey scale.",
                    indicator_name="Self-Reported General Health (GenHlth)",
                    indicator_value=gh_label,
                    explanation=(
                        "Self-rated general health is a validated subjective measure reflecting perceived daily vitality and physical well-being. "
                        "It captures personal perception rather than a specific clinical diagnosis."
                    ),
                    suggestion=(
                        "If persistent fatigue, pain, or functional limitations are affecting your quality of life, consider discussing these "
                        "symptoms with your primary care provider."
                    ),
                    reason=f"Triggered because GenHlth was reported as {assessment.GenHlth} (Fair or Poor).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_HRQOL,
                    limitations=(
                        "Subjective self-rating reflecting overall perception; does not identify specific medical conditions or objective clinical pathology."
                    ),
                )
            )
        elif assessment.GenHlth == 3:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_genhlth_good_{a_id}",
                    category="General Well-being",
                    priority="info",
                    observation=f"You self-rated your general health as '{gh_label}' on a 5-point survey scale.",
                    indicator_name="Self-Reported General Health (GenHlth)",
                    indicator_value=gh_label,
                    explanation=(
                        "A 'Good' rating represents moderate self-perceived health with opportunities to enhance daily energy and well-being "
                        "through balanced lifestyle habits."
                    ),
                    suggestion=(
                        "Explore daily lifestyle routines—such as sleep consistency, hydration, and stress moderation—to support your overall "
                        "sense of wellness."
                    ),
                    reason="Triggered because GenHlth was reported as 3 (Good).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_HRQOL,
                    limitations="Subjective self-rating; does not correlate with specific diagnostic laboratory markers.",
                )
            )
        else:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_genhlth_excellent_{a_id}",
                    category="General Well-being",
                    priority="positive",
                    observation=f"You self-rated your general health as '{gh_label}' on a 5-point survey scale.",
                    indicator_name="Self-Reported General Health (GenHlth)",
                    indicator_value=gh_label,
                    explanation=(
                        "High self-rated health is strongly correlated in population studies with perceived vitality, functional independence, "
                        "and overall physical resilience."
                    ),
                    suggestion=(
                        "Continue the balanced lifestyle habits and preventive care routines that support your positive sense of well-being."
                    ),
                    reason=f"Triggered because GenHlth was reported as {assessment.GenHlth} (Excellent or Very Good).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_HRQOL,
                    limitations="Subjective self-rating; subjective perception does not guarantee absence of asymptomatic conditions.",
                )
            )

    # 7. Physical Health Days (PhysHlth) — Past 30-day recall
    if assessment.PhysHlth is not None:
        if assessment.PhysHlth >= 14:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_physhlth_frequent_{a_id}",
                    category="General Well-being",
                    priority="high",
                    observation=f"You reported {assessment.PhysHlth} days in the past 30 days where physical illness or injury hindered your health.",
                    indicator_name="Physical Health Days (PhysHlth)",
                    indicator_value=f"{assessment.PhysHlth} of 30 days",
                    explanation=(
                        "Reporting 14 or more days of poor physical health per month is a CDC surveillance benchmark for frequent physical distress. "
                        "It reflects perceived impairment but does not specify a medical diagnosis."
                    ),
                    suggestion=(
                        "Consider discussing recurring physical symptoms, pain, or illness with your healthcare provider to evaluate underlying causes."
                    ),
                    reason=f"Triggered because PhysHlth was reported as {assessment.PhysHlth} days (≥14 days).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_HRQOL,
                    limitations=(
                        "Self-reported 30-day recall; does not identify underlying physiological etiology, acute injury versus chronic disease, or clinical severity."
                    ),
                )
            )
        elif assessment.PhysHlth > 0:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_physhlth_moderate_{a_id}",
                    category="General Well-being",
                    priority="info",
                    observation=f"You reported {assessment.PhysHlth} days of poor physical health in the past 30 days.",
                    indicator_name="Physical Health Days (PhysHlth)",
                    indicator_value=f"{assessment.PhysHlth} of 30 days",
                    explanation=(
                        "Occasional days of physical illness, fatigue, or strain are common; self-monitoring helps identify whether episodes are isolated "
                        "or becoming more frequent."
                    ),
                    suggestion="Pay attention to rest, muscular recovery, and ergonomic habits on days when physical fatigue occurs.",
                    reason=f"Triggered because PhysHlth was reported as {assessment.PhysHlth} days (1–13 days).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_HRQOL,
                    limitations="Subjective self-report based on 30-day recall.",
                )
            )
        else:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_physhlth_zero_{a_id}",
                    category="General Well-being",
                    priority="positive",
                    observation="You reported 0 days of poor physical health in the past 30 days.",
                    indicator_name="Physical Health Days (PhysHlth)",
                    indicator_value="0 of 30 days",
                    explanation="Reporting zero days of physical illness or injury reflects strong self-perceived daily physical function during the past month.",
                    suggestion="Continue preventive health practices, balanced physical activity, and adequate recovery to sustain physical well-being.",
                    reason="Triggered because PhysHlth was reported as 0 days.",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_HRQOL,
                    limitations="Subjective 30-day recall; does not rule out subclinical or asymptomatic conditions.",
                )
            )

    # 8. Mental Health Days (MentHlth) — Past 30-day recall
    if assessment.MentHlth is not None:
        if assessment.MentHlth >= 14:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_menthlth_frequent_{a_id}",
                    category="General Well-being",
                    priority="high",
                    observation=f"You reported {assessment.MentHlth} days in the past 30 days where stress, depression, or emotional issues hindered your health.",
                    indicator_name="Mental Health Days (MentHlth)",
                    indicator_value=f"{assessment.MentHlth} of 30 days",
                    explanation=(
                        "Reporting 14 or more days of poor mental health is a CDC threshold for frequent mental distress. It serves as a screening "
                        "indicator of emotional burden, not a psychiatric diagnosis."
                    ),
                    suggestion=(
                        "If emotional strain or stress is interfering with your daily activities, consider reaching out to a healthcare professional, "
                        "counselor, or trusted support resource."
                    ),
                    reason=f"Triggered because MentHlth was reported as {assessment.MentHlth} days (≥14 days).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_HRQOL,
                    limitations=(
                        "Survey screening measure of recent emotional well-being; is not a clinical diagnostic instrument for depression, anxiety, "
                        "or other mental health disorders."
                    ),
                )
            )
        elif assessment.MentHlth > 0:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_menthlth_moderate_{a_id}",
                    category="General Well-being",
                    priority="info",
                    observation=f"You reported {assessment.MentHlth} days of emotional distress or stress in the past 30 days.",
                    indicator_name="Mental Health Days (MentHlth)",
                    indicator_value=f"{assessment.MentHlth} of 30 days",
                    explanation="Periodic stress or emotional fatigue is a common experience; tracking helps recognize trends and opportunities for restorative practices.",
                    suggestion="Explore stress-reduction techniques such as mindfulness, outdoor walks, social connection, and adequate sleep to support emotional balance.",
                    reason=f"Triggered because MentHlth was reported as {assessment.MentHlth} days (1–13 days).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_HRQOL,
                    limitations="Subjective self-reported recall over 30 days.",
                )
            )
        else:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_menthlth_zero_{a_id}",
                    category="General Well-being",
                    priority="positive",
                    observation="You reported 0 days of mental or emotional distress in the past 30 days.",
                    indicator_name="Mental Health Days (MentHlth)",
                    indicator_value="0 of 30 days",
                    explanation="Reporting zero days of emotional distress reflects positive self-perceived psychological well-being over the past month.",
                    suggestion="Continue healthy stress management habits, restful sleep patterns, and supportive social connections.",
                    reason="Triggered because MentHlth was reported as 0 days.",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_HRQOL,
                    limitations="Subjective 30-day recall; reflects personal perception over a defined time window.",
                )
            )

    # 9. Difficulty Walking (DiffWalk) — Functional mobility screening
    if assessment.DiffWalk is not None:
        if assessment.DiffWalk == 1:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_diffwalk_present_{a_id}",
                    category="Physical Lifestyle",
                    priority="high",
                    observation="You reported serious difficulty walking or climbing stairs.",
                    indicator_name="Mobility Limitation (DiffWalk)",
                    indicator_value="Yes (1)",
                    explanation=(
                        "This survey item screens for self-reported ambulatory difficulty. It indicates perceived mobility limitation "
                        "but does not diagnose a specific orthopedic, neuromuscular, or cardiopulmonary condition."
                    ),
                    suggestion=(
                        "Discuss mobility limitations with a physician or physical therapist to explore safe, joint-preserving physical activity "
                        "options (such as aquatic movement or seated exercises) and supportive aids."
                    ),
                    reason="Triggered because DiffWalk was reported as 1 (Yes).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_DISABILITY,
                    limitations=(
                        "Self-reported screening question; does not establish etiology, clinical diagnosis, or objective functional impairment grading."
                    ),
                )
            )
        else:
            recs.append(
                EducationalRecommendation(
                    id=f"rec_diffwalk_none_{a_id}",
                    category="Physical Lifestyle",
                    priority="positive",
                    observation="You did not report serious difficulty walking or climbing stairs.",
                    indicator_name="Mobility Limitation (DiffWalk)",
                    indicator_value="No (0)",
                    explanation="Reporting no severe ambulatory difficulty indicates functional mobility for routine daily activities and walking.",
                    suggestion="Take advantage of functional mobility through regular daily walking and staying physically active as tolerated.",
                    reason="Triggered because DiffWalk was reported as 0 (No).",
                    date_generated=eval_date,
                    evidence_reference=REF_CDC_DISABILITY,
                    limitations="Self-reported screening question; does not assess specific joint pathology or endurance.",
                )
            )

    return recs


def _generate_historical_comparisons(
    curr: Assessment, prev: Assessment
) -> List[HistoricalComparisonItem]:
    """
    Generate longitudinal comparison items between the user's current and previous assessments.
    Enforces neutral descriptions, discloses conflicting responses, handles missing values safely,
    and never presents changes as clinical improvement/worsening or disease progression.
    """
    items: List[HistoricalComparisonItem] = []

    # 1. BMI Comparison (Neutral descriptions: increased, decreased, relatively stable)
    if curr.BMI is not None and prev.BMI is not None:
        curr_bmi = round(float(curr.BMI), 1)
        prev_bmi = round(float(prev.BMI), 1)
        delta_bmi = round(curr_bmi - prev_bmi, 1)

        if delta_bmi < -0.5:
            bmi_type = "decreased"
            bmi_obs = (
                f"BMI decreased by {abs(delta_bmi)} kg/m² ({prev_bmi} → {curr_bmi}). Weight changes can occur from "
                "intentional lifestyle adjustments, appetite shifts, or other factors; discuss sustained changes with your clinician."
            )
        elif delta_bmi > 0.5:
            bmi_type = "increased"
            bmi_obs = (
                f"BMI increased by {delta_bmi} kg/m² ({prev_bmi} → {curr_bmi}). Discuss body composition goals and overall "
                "health with your healthcare provider."
            )
        else:
            bmi_type = "stable"
            bmi_obs = f"BMI remained relatively stable ({prev_bmi} → {curr_bmi}, Δ {delta_bmi:+} kg/m²)."

        items.append(
            HistoricalComparisonItem(
                indicator="Body Mass Index (BMI)",
                previous_value=f"{prev_bmi} kg/m²",
                current_value=f"{curr_bmi} kg/m²",
                change_type=bmi_type,
                observation=bmi_obs,
            )
        )

    # 2. High Blood Pressure Comparison
    if curr.HighBP is not None and prev.HighBP is not None:
        if prev.HighBP == 0 and curr.HighBP == 1:
            bp_type = "increased"
            bp_obs = "Self-reported status shifted to reporting a high blood pressure diagnosis. Schedule clinical follow-up for blood pressure monitoring."
        elif prev.HighBP == 1 and curr.HighBP == 0:
            bp_type = "conflicting"
            bp_obs = (
                "Response variation noted: High blood pressure diagnosis was reported previously but not reported currently. "
                "Note that while blood pressure may be managed with treatment, diagnostic history usually persists; verify status with your provider."
            )
        else:
            bp_type = "stable"
            bp_obs = f"Reported blood pressure status remained consistent ({'Reported high BP' if curr.HighBP == 1 else 'Did not report high BP'})."

        items.append(
            HistoricalComparisonItem(
                indicator="High Blood Pressure",
                previous_value="Reported (1)" if prev.HighBP == 1 else "Not Reported (0)",
                current_value="Reported (1)" if curr.HighBP == 1 else "Not Reported (0)",
                change_type=bp_type,
                observation=bp_obs,
            )
        )

    # 3. High Cholesterol Comparison
    if curr.HighChol is not None and prev.HighChol is not None:
        if prev.HighChol == 0 and curr.HighChol == 1:
            chol_type = "increased"
            chol_obs = "Self-reported status shifted to reporting a high cholesterol diagnosis. Consult your clinician for lipid assessment."
        elif prev.HighChol == 1 and curr.HighChol == 0:
            chol_type = "conflicting"
            chol_obs = (
                "Response variation noted: High cholesterol diagnosis was reported previously but not reported currently. "
                "Confirm current serum lipid levels with fasting laboratory tests."
            )
        else:
            chol_type = "stable"
            chol_obs = f"Reported cholesterol status remained consistent ({'Reported high cholesterol' if curr.HighChol == 1 else 'Did not report high cholesterol'})."

        items.append(
            HistoricalComparisonItem(
                indicator="High Cholesterol",
                previous_value="Reported (1)" if prev.HighChol == 1 else "Not Reported (0)",
                current_value="Reported (1)" if curr.HighChol == 1 else "Not Reported (0)",
                change_type=chol_type,
                observation=chol_obs,
            )
        )

    # 4. Smoking Comparison (Disclose conflicts where lifetime threshold reverts)
    if curr.Smoker is not None and prev.Smoker is not None:
        if prev.Smoker == 0 and curr.Smoker == 1:
            smk_type = "increased"
            smk_obs = "Self-reported lifetime smoking status shifted to reporting ≥100 cigarettes."
        elif prev.Smoker == 1 and curr.Smoker == 0:
            smk_type = "conflicting"
            smk_obs = (
                f"Survey response conflict: Lifetime smoking of ≥100 cigarettes was reported in assessment #{prev.id}, "
                f"whereas <100 cigarettes was reported in assessment #{curr.id}. Lifetime thresholds cannot decrease; "
                "this may reflect recall variation or differing survey interpretation."
            )
        else:
            smk_type = "stable"
            smk_obs = f"Reported lifetime smoking threshold status remained consistent ({'≥100 cigarettes' if curr.Smoker == 1 else '<100 cigarettes'})."

        items.append(
            HistoricalComparisonItem(
                indicator="Tobacco Exposure (Smoker)",
                previous_value="≥100 cigarettes (1)" if prev.Smoker == 1 else "<100 cigarettes (0)",
                current_value="≥100 cigarettes (1)" if curr.Smoker == 1 else "<100 cigarettes (0)",
                change_type=smk_type,
                observation=smk_obs,
            )
        )

    # 5. Physical Activity Comparison
    if curr.PhysActivity is not None and prev.PhysActivity is not None:
        if prev.PhysActivity == 0 and curr.PhysActivity == 1:
            act_type = "increased"
            act_obs = "Self-reported physical activity shifted from inactive to active leisure-time exercise in the past 30 days."
        elif prev.PhysActivity == 1 and curr.PhysActivity == 0:
            act_type = "decreased"
            act_obs = "Self-reported physical activity shifted to inactive in the past 30 days. Consider gradual re-introduction of regular movement."
        else:
            act_type = "stable"
            act_obs = f"Physical activity level remained steady ({'Active' if curr.PhysActivity == 1 else 'Inactive'})."

        items.append(
            HistoricalComparisonItem(
                indicator="Physical Activity",
                previous_value="Active (1)" if prev.PhysActivity == 1 else "Inactive (0)",
                current_value="Active (1)" if curr.PhysActivity == 1 else "Inactive (0)",
                change_type=act_type,
                observation=act_obs,
            )
        )

    # 6. General Health Rating Comparison
    if curr.GenHlth is not None and prev.GenHlth is not None:
        prev_gh_label = GEN_HLTH_LABELS.get(prev.GenHlth, str(prev.GenHlth))
        curr_gh_label = GEN_HLTH_LABELS.get(curr.GenHlth, str(curr.GenHlth))
        gh_delta = curr.GenHlth - prev.GenHlth

        if gh_delta < 0:
            gh_type = "increased"
            gh_obs = f"Self-rated general health shifted favorably from '{prev_gh_label}' to '{curr_gh_label}'. Self-reported perception reflects personal vitality, not clinical cure."
        elif gh_delta > 0:
            gh_type = "decreased"
            gh_obs = f"Self-rated general health shifted from '{prev_gh_label}' to '{curr_gh_label}'. Discuss persistent health challenges with your clinician."
        else:
            gh_type = "stable"
            gh_obs = f"Self-rated general health remained consistent at '{curr_gh_label}'."

        items.append(
            HistoricalComparisonItem(
                indicator="General Health",
                previous_value=prev_gh_label,
                current_value=curr_gh_label,
                change_type=gh_type,
                observation=gh_obs,
            )
        )

    # 7. Physical Distress Days Comparison
    if curr.PhysHlth is not None and prev.PhysHlth is not None:
        p_delta = curr.PhysHlth - prev.PhysHlth
        if p_delta < 0:
            p_type = "decreased"
            p_obs = f"Reported physical distress days decreased by {abs(p_delta)} days ({prev.PhysHlth} → {curr.PhysHlth} days/month)."
        elif p_delta > 0:
            p_type = "increased"
            p_obs = f"Reported physical distress days increased by {p_delta} days ({prev.PhysHlth} → {curr.PhysHlth} days/month)."
        else:
            p_type = "stable"
            p_obs = f"Reported physical distress days remained constant at {curr.PhysHlth} days/month."

        items.append(
            HistoricalComparisonItem(
                indicator="Physical Health Days",
                previous_value=f"{prev.PhysHlth} days",
                current_value=f"{curr.PhysHlth} days",
                change_type=p_type,
                observation=p_obs,
            )
        )

    # 8. Mental Distress Days Comparison
    if curr.MentHlth is not None and prev.MentHlth is not None:
        m_delta = curr.MentHlth - prev.MentHlth
        if m_delta < 0:
            m_type = "decreased"
            m_obs = f"Reported mental distress days decreased by {abs(m_delta)} days ({prev.MentHlth} → {curr.MentHlth} days/month)."
        elif m_delta > 0:
            m_type = "increased"
            m_obs = f"Reported mental distress days increased by {m_delta} days ({prev.MentHlth} → {curr.MentHlth} days/month)."
        else:
            m_type = "stable"
            m_obs = f"Reported mental distress days remained constant at {curr.MentHlth} days/month."

        items.append(
            HistoricalComparisonItem(
                indicator="Mental Health Days",
                previous_value=f"{prev.MentHlth} days",
                current_value=f"{curr.MentHlth} days",
                change_type=m_type,
                observation=m_obs,
            )
        )

    return items


def get_health_insights(
    db: Session,
    user_id: int,
    assessment_id: Optional[int] = None,
) -> HealthInsightsResponse:
    """
    Retrieve personalized educational health recommendations and longitudinal comparisons
    scoped strictly to the authenticated user.
    """
    stmt = (
        select(Assessment)
        .where(Assessment.user_id == user_id)
        .order_by(Assessment.created_at.desc())
    )
    assessments = list(db.scalars(stmt))
    total_assessments = len(assessments)

    if total_assessments == 0:
        return HealthInsightsResponse(
            total_assessments=0,
            summary=InsightsSummary(total_insights=0),
            recommendations=[],
            historical_comparisons=[],
        )

    target_assessment: Assessment
    if assessment_id is not None:
        matched = next((a for a in assessments if a.id == assessment_id), None)
        if matched is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assessment #{assessment_id} not found for this user account.",
            )
        target_assessment = matched
    else:
        target_assessment = assessments[0]

    recommendations = _generate_recommendations_for_assessment(target_assessment)

    historical_comparisons: List[HistoricalComparisonItem] = []
    if total_assessments >= 2:
        idx = next((i for i, a in enumerate(assessments) if a.id == target_assessment.id), 0)
        if idx + 1 < total_assessments:
            prev_assessment = assessments[idx + 1]
            historical_comparisons = _generate_historical_comparisons(
                target_assessment, prev_assessment
            )

    high_count = sum(1 for r in recommendations if r.priority == "high")
    med_count = sum(1 for r in recommendations if r.priority == "medium")
    pos_count = sum(1 for r in recommendations if r.priority == "positive")

    cat_breakdown: Dict[str, int] = {}
    for r in recommendations:
        cat_breakdown[r.category] = cat_breakdown.get(r.category, 0) + 1

    summary = InsightsSummary(
        total_insights=len(recommendations),
        high_priority_count=high_count,
        medium_priority_count=med_count,
        positive_habits_count=pos_count,
        categories_breakdown=cat_breakdown,
        latest_assessment_id=target_assessment.id,
        latest_assessment_date=(
            target_assessment.created_at.strftime("%b %d, %Y")
            if target_assessment.created_at
            else None
        ),
    )

    return HealthInsightsResponse(
        assessment_id=target_assessment.id,
        assessment_date=summary.latest_assessment_date,
        total_assessments=total_assessments,
        summary=summary,
        recommendations=recommendations,
        historical_comparisons=historical_comparisons,
    )


def get_insights_summary(db: Session, user_id: int) -> InsightsSummary:
    """Retrieve quick KPI summary of insights for authenticated user."""
    full = get_health_insights(db, user_id=user_id)
    return full.summary


def get_insights_history(db: Session, user_id: int) -> InsightsHistoryResponse:
    """Retrieve longitudinal indicator comparison history for authenticated user."""
    stmt = (
        select(Assessment)
        .where(Assessment.user_id == user_id)
        .order_by(Assessment.created_at.desc())
        .limit(10)
    )
    assessments = list(db.scalars(stmt))
    total = len(assessments)

    comparisons: List[HistoricalComparisonItem] = []
    if total >= 2:
        comparisons = _generate_historical_comparisons(assessments[0], assessments[1])

    return InsightsHistoryResponse(
        total_assessments=total,
        assessments_analyzed=min(total, 2),
        comparisons=comparisons,
    )
