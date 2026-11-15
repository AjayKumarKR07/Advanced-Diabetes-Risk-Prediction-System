"""
backend/app/services/insights_service.py
========================================
Transparent, rule-based educational health insights and recommendation engine (Phase 6.16).
Analyzes user submitted assessment indicators (BMI, HighBP, HighChol, PhysActivity,
Smoker, GenHlth, PhysHlth, MentHlth, DiffWalk) and provides explainable, non-medical
educational suggestions and longitudinal change tracking.
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


def _generate_recommendations_for_assessment(
    assessment: Assessment,
) -> List[EducationalRecommendation]:
    """
    Generate transparent, rule-based educational recommendations from a single assessment's indicators.
    """
    recs: List[EducationalRecommendation] = []
    eval_date = (
        assessment.created_at.strftime("%b %d, %Y")
        if assessment.created_at
        else datetime.now().strftime("%b %d, %Y")
    )
    a_id = assessment.id

    # 1. BMI Indicator
    bmi_val = round(float(assessment.BMI), 1)
    if bmi_val < 18.5:
        recs.append(
            EducationalRecommendation(
                id=f"rec_bmi_underweight_{a_id}",
                category="Metabolic & Body",
                priority="medium",
                observation=f"Your submitted BMI is {bmi_val}, which falls into the Underweight screening category (<18.5).",
                indicator_name="Body Mass Index (BMI)",
                indicator_value=f"{bmi_val} kg/m²",
                explanation=(
                    "In public health literature, a BMI below 18.5 can indicate low lean body mass or nutritional "
                    "insufficiencies, which may affect baseline metabolic reserves and energy vitality."
                ),
                suggestion=(
                    "Consider discussing your energy balance and nutrient density with a registered dietitian or "
                    "primary care physician to ensure adequate nourishment."
                ),
                reason=f"Triggered because submitted BMI ({bmi_val}) is below 18.5 in assessment #{a_id}.",
                date_generated=eval_date,
            )
        )
    elif bmi_val < 25.0:
        recs.append(
            EducationalRecommendation(
                id=f"rec_bmi_normal_{a_id}",
                category="Metabolic & Body",
                priority="positive",
                observation=f"Your submitted BMI is {bmi_val}, which falls within the Normal Weight adult screening range (18.5–24.9).",
                indicator_name="Body Mass Index (BMI)",
                indicator_value=f"{bmi_val} kg/m²",
                explanation=(
                    "Maintaining a BMI in the conventional normal range is associated with favorable baseline insulin sensitivity "
                    "and reduced incidence of chronic metabolic complications across population cohorts."
                ),
                suggestion=(
                    "Continue maintaining balanced nutrition, adequate hydration, and consistent physical activity to sustain this healthy baseline."
                ),
                reason=f"Triggered because submitted BMI ({bmi_val}) is in the normal screening range (18.5–24.9).",
                date_generated=eval_date,
            )
        )
    elif bmi_val < 30.0:
        recs.append(
            EducationalRecommendation(
                id=f"rec_bmi_overweight_{a_id}",
                category="Metabolic & Body",
                priority="medium",
                observation=f"Your submitted BMI is {bmi_val}, categorized within the Overweight adult screening range (25.0–29.9).",
                indicator_name="Body Mass Index (BMI)",
                indicator_value=f"{bmi_val} kg/m²",
                explanation=(
                    "Epidemiological data from the CDC BRFSS demonstrates that higher adipose mass can increase systemic "
                    "free fatty acid circulation, which is a recognized population-level correlate with altered glucose tolerance."
                ),
                suggestion=(
                    "Consider reviewing dietary fiber intake, portion sizes, and weekly physical activity with your healthcare "
                    "provider as part of proactive routine wellness planning."
                ),
                reason=f"Triggered because submitted BMI ({bmi_val}) is between 25.0 and 29.9 in assessment #{a_id}.",
                date_generated=eval_date,
            )
        )
    else:  # >= 30.0
        recs.append(
            EducationalRecommendation(
                id=f"rec_bmi_obesity_{a_id}",
                category="Metabolic & Body",
                priority="high",
                observation=f"Your submitted BMI is {bmi_val}, which falls into the Obesity adult screening range (≥30.0).",
                indicator_name="Body Mass Index (BMI)",
                indicator_value=f"{bmi_val} kg/m²",
                explanation=(
                    "In large surveillance cohorts, obesity is strongly associated with elevated metabolic workload, "
                    "subclinical inflammation, and increased vulnerability to glycemic dysregulation."
                ),
                suggestion=(
                    "We recommend consulting a licensed physician for comprehensive clinical laboratory testing "
                    "(e.g. Fasting Plasma Glucose, HbA1c, lipid panel) and discussing structured lifestyle interventions."
                ),
                reason=f"Triggered because submitted BMI ({bmi_val}) is 30.0 or greater in assessment #{a_id}.",
                date_generated=eval_date,
            )
        )

    # 2. High Blood Pressure Indicator (HighBP)
    if assessment.HighBP == 1:
        recs.append(
            EducationalRecommendation(
                id=f"rec_highbp_active_{a_id}",
                category="Cardiovascular Health",
                priority="high",
                observation="You self-reported a diagnosed history of high blood pressure (hypertension).",
                indicator_name="High Blood Pressure (HighBP)",
                indicator_value="Yes (1)",
                explanation=(
                    "Elevated blood pressure frequently clusters with insulin resistance and metabolic syndrome due to shared "
                    "endothelial stiffness, sympathetic nervous activation, and renal sodium handling pathways."
                ),
                suggestion=(
                    "Regularly monitor your blood pressure, practice dietary sodium moderation (such as DASH diet principles), "
                    "and maintain ongoing checkups with your physician."
                ),
                reason="Triggered because HighBP was submitted as 1 (Yes).",
                date_generated=eval_date,
            )
        )
    else:
        recs.append(
            EducationalRecommendation(
                id=f"rec_highbp_normal_{a_id}",
                category="Cardiovascular Health",
                priority="positive",
                observation="You reported no diagnosed history of high blood pressure.",
                indicator_name="High Blood Pressure (HighBP)",
                indicator_value="No (0)",
                explanation=(
                    "Absence of diagnosed hypertension is a positive cardiovascular health marker that supports normal microvascular "
                    "perfusion across vital metabolic tissues."
                ),
                suggestion="Continue routine blood pressure checks during annual wellness visits to support long-term vascular health.",
                reason="Triggered because HighBP was submitted as 0 (No).",
                date_generated=eval_date,
            )
        )

    # 3. High Cholesterol Indicator (HighChol)
    if assessment.HighChol == 1:
        recs.append(
            EducationalRecommendation(
                id=f"rec_highchol_active_{a_id}",
                category="Cardiovascular Health",
                priority="high",
                observation="You self-reported having been told by a doctor or health professional that your cholesterol is high.",
                indicator_name="High Cholesterol (HighChol)",
                indicator_value="Yes (1)",
                explanation=(
                    "Dyslipidemia (elevated triglycerides, low HDL, or elevated LDL) commonly parallels glycemic dysregulation "
                    "and increases long-term cardiovascular burden."
                ),
                suggestion=(
                    "Discuss periodic fasting lipid profiling with your doctor, and explore heart-healthy meal planning rich in "
                    "soluble dietary fiber, nuts, seeds, and unsaturated fats."
                ),
                reason="Triggered because HighChol was submitted as 1 (Yes).",
                date_generated=eval_date,
            )
        )
    else:
        recs.append(
            EducationalRecommendation(
                id=f"rec_highchol_normal_{a_id}",
                category="Cardiovascular Health",
                priority="positive",
                observation="You reported no diagnosed history of elevated cholesterol.",
                indicator_name="High Cholesterol (HighChol)",
                indicator_value="No (0)",
                explanation=(
                    "Normal lipid levels contribute to vascular patency and reduce atherosclerotic cardiovascular vulnerability."
                ),
                suggestion="Maintain preventive screening every 4 to 6 years (or as recommended by your physician) to ensure healthy lipid balance.",
                reason="Triggered because HighChol was submitted as 0 (No).",
                date_generated=eval_date,
            )
        )

    # 4. Physical Activity Indicator (PhysActivity)
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
                    "Sedentary lifestyle reduces skeletal muscle glucose transporter (GLUT4) translocation, impairing "
                    "insulin-stimulated glucose uptake and contributing to metabolic stiffness."
                ),
                suggestion=(
                    "Aim to introduce gradual, moderate-intensity movement (e.g. brisk walking for 15–30 minutes daily, "
                    "totaling 150 minutes weekly) after consulting your medical provider."
                ),
                reason="Triggered because PhysActivity was submitted as 0 (No leisure physical activity).",
                date_generated=eval_date,
            )
        )
    else:
        recs.append(
            EducationalRecommendation(
                id=f"rec_physactivity_active_{a_id}",
                category="Physical Lifestyle",
                priority="positive",
                observation="You reported participating in physical activity or exercise in the past 30 days.",
                indicator_name="Physical Activity (PhysActivity)",
                indicator_value="Active (1)",
                explanation=(
                    "Routine physical activity acutely enhances muscle insulin sensitivity and stimulates glucose clearance independent "
                    "of insulin pathways, offering sustained metabolic and cardiovascular benefits."
                ),
                suggestion=(
                    "Maintain your active routine and consider adding strength training twice weekly to further support skeletal muscle health."
                ),
                reason="Triggered because PhysActivity was submitted as 1 (Active).",
                date_generated=eval_date,
            )
        )

    # 5. Smoking Status (Smoker)
    if assessment.Smoker == 1:
        recs.append(
            EducationalRecommendation(
                id=f"rec_smoker_history_{a_id}",
                category="General Well-being",
                priority="medium",
                observation="You reported a history of smoking at least 100 cigarettes in your lifetime.",
                indicator_name="Tobacco Exposure (Smoker)",
                indicator_value="Yes (1)",
                explanation=(
                    "Tobacco smoke compounds promote systemic oxidative stress, increase central adiposity deposition, "
                    "and impair cellular insulin signaling."
                ),
                suggestion=(
                    "If currently smoking, explore evidence-based cessation programs, nicotine replacement strategies, "
                    "or physician-guided support. If you have already quit, celebrate and maintain that vital health milestone."
                ),
                reason="Triggered because Smoker was reported as 1 (≥100 lifetime cigarettes).",
                date_generated=eval_date,
            )
        )
    else:
        recs.append(
            EducationalRecommendation(
                id=f"rec_smoker_nonsmoker_{a_id}",
                category="General Well-being",
                priority="positive",
                observation="You reported no history of smoking at least 100 cigarettes.",
                indicator_name="Tobacco Exposure (Smoker)",
                indicator_value="No (0)",
                explanation="Non-smoking status avoids critical endothelial toxin exposure and preserves microvascular integrity.",
                suggestion="Continue tobacco-free living and minimize exposure to secondhand smoke environments.",
                reason="Triggered because Smoker was reported as 0 (No).",
                date_generated=eval_date,
            )
        )

    # 6. General Health Rating (GenHlth)
    gh_label = GEN_HLTH_LABELS.get(assessment.GenHlth, f"Rating {assessment.GenHlth}")
    if assessment.GenHlth >= 4:
        recs.append(
            EducationalRecommendation(
                id=f"rec_genhlth_poor_{a_id}",
                category="General Well-being",
                priority="high",
                observation=f"You self-rated your general health as '{gh_label}'.",
                indicator_name="Self-Reported General Health (GenHlth)",
                indicator_value=gh_label,
                explanation=(
                    "Self-reported health is a robust, well-established public health indicator that correlates with systemic symptom "
                    "burden, chronic condition management challenges, and perceived functional limits."
                ),
                suggestion=(
                    "Discuss persistent fatigue or health challenges with your primary care provider for holistic evaluation and support."
                ),
                reason=f"Triggered because GenHlth was reported as {assessment.GenHlth} (Fair or Poor).",
                date_generated=eval_date,
            )
        )
    elif assessment.GenHlth == 3:
        recs.append(
            EducationalRecommendation(
                id=f"rec_genhlth_good_{a_id}",
                category="General Well-being",
                priority="info",
                observation=f"You self-rated your general health as '{gh_label}'.",
                indicator_name="Self-Reported General Health (GenHlth)",
                indicator_value=gh_label,
                explanation=(
                    "A 'Good' rating represents moderate vitality with potential opportunities to enhance daily well-being "
                    "through lifestyle and stress management adjustments."
                ),
                suggestion=(
                    "Identify specific wellness areas—such as sleep hygiene, hydration, or stress reduction—to elevate your day-to-day energy."
                ),
                reason="Triggered because GenHlth was reported as 3 (Good).",
                date_generated=eval_date,
            )
        )
    else:
        recs.append(
            EducationalRecommendation(
                id=f"rec_genhlth_excellent_{a_id}",
                category="General Well-being",
                priority="positive",
                observation=f"You self-rated your general health as '{gh_label}'.",
                indicator_name="Self-Reported General Health (GenHlth)",
                indicator_value=gh_label,
                explanation=(
                    "High self-rated health correlates strongly with physical independence, positive functional status, and physiological resilience."
                ),
                suggestion="Continue the balanced lifestyle practices and preventive health measures that support your strong sense of well-being.",
                reason=f"Triggered because GenHlth was reported as {assessment.GenHlth} (Excellent or Very Good).",
                date_generated=eval_date,
            )
        )

    # 7. Physical Health Days (PhysHlth)
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
                    "Frequent physical distress (≥14 days/month) is a recognized CDC threshold indicating potential chronic impairment, "
                    "persistent pain, or recurring illness."
                ),
                suggestion=(
                    "We recommend reviewing these recurrent symptoms with your physician to formulate a targeted diagnostic or rehabilitative plan."
                ),
                reason=f"Triggered because PhysHlth was reported as {assessment.PhysHlth} days (≥14 days).",
                date_generated=eval_date,
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
                explanation="Occasional physical strain or minor illness is common; tracking helps discern if episodes become more frequent.",
                suggestion="Ensure sufficient muscular recovery, ergonomic adjustments, and hydration on days of physical fatigue.",
                reason=f"Triggered because PhysHlth was reported as {assessment.PhysHlth} days (1–13 days).",
                date_generated=eval_date,
            )
        )

    # 8. Mental Health Days (MentHlth)
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
                    "Chronic emotional stress triggers neuroendocrine hypothalamic-pituitary-adrenal (HPA) axis activation, "
                    "elevating cortisol and potentially disrupting sleep and glucose regulation."
                ),
                suggestion=(
                    "Consider speaking with a licensed mental health professional, counselor, or doctor for compassionate, confidential guidance."
                ),
                reason=f"Triggered because MentHlth was reported as {assessment.MentHlth} days (≥14 days).",
                date_generated=eval_date,
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
                explanation="Periodic stress is normal, but establishing restorative coping rituals supports long-term mental clarity and metabolic health.",
                suggestion="Explore stress management techniques such as mindfulness meditation, outdoor walks, or creative leisure activities.",
                reason=f"Triggered because MentHlth was reported as {assessment.MentHlth} days (1–13 days).",
                date_generated=eval_date,
            )
        )

    # 9. Difficulty Walking (DiffWalk)
    if assessment.DiffWalk == 1:
        recs.append(
            EducationalRecommendation(
                id=f"rec_diffwalk_present_{a_id}",
                category="Physical Lifestyle",
                priority="high",
                observation="You self-reported having serious difficulty walking or climbing stairs.",
                indicator_name="Mobility Limitation (DiffWalk)",
                indicator_value="Yes (1)",
                explanation=(
                    "Mobility limitations can elevate sedentary time and reduce baseline caloric expenditure, making low-impact "
                    "alternatives essential for metabolic maintenance."
                ),
                suggestion=(
                    "Consult a physical therapist or physician regarding safe, joint-preserving exercises like water aerobics, "
                    "seated resistance bands, or recumbent cycling."
                ),
                reason="Triggered because DiffWalk was reported as 1 (Yes).",
                date_generated=eval_date,
            )
        )
    else:
        recs.append(
            EducationalRecommendation(
                id=f"rec_diffwalk_none_{a_id}",
                category="Physical Lifestyle",
                priority="positive",
                observation="You reported no difficulty walking or climbing stairs.",
                indicator_name="Mobility Limitation (DiffWalk)",
                indicator_value="No (0)",
                explanation="Unrestricted mobility preserves physical independence and facilitates regular daily physical movement.",
                suggestion="Take advantage of functional mobility with daily walking and stair climbing when feasible.",
                reason="Triggered because DiffWalk was reported as 0 (No).",
                date_generated=eval_date,
            )
        )

    return recs


def _generate_historical_comparisons(
    curr: Assessment, prev: Assessment
) -> List[HistoricalComparisonItem]:
    """
    Generate longitudinal comparison items between the user's current and previous assessments.
    """
    items: List[HistoricalComparisonItem] = []

    # 1. BMI Comparison
    curr_bmi = round(float(curr.BMI), 1)
    prev_bmi = round(float(prev.BMI), 1)
    delta_bmi = round(curr_bmi - prev_bmi, 1)

    if delta_bmi < -0.5:
        bmi_type = "improved"
        bmi_obs = f"BMI decreased by {abs(delta_bmi)} kg/m² ({prev_bmi} → {curr_bmi}). If intentional, this reflects favorable body composition progress."
    elif delta_bmi > 0.5:
        bmi_type = "declined"
        bmi_obs = f"BMI increased by {delta_bmi} kg/m² ({prev_bmi} → {curr_bmi}). Discuss body composition goals with your provider."
    else:
        bmi_type = "stable"
        bmi_obs = f"BMI remained stable ({prev_bmi} → {curr_bmi}, Δ {delta_bmi:+} kg/m²)."

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
    if prev.HighBP != curr.HighBP:
        bp_type = "declined" if curr.HighBP == 1 else "improved"
        bp_obs = (
            "Status shifted to reported high blood pressure. Ensure regular clinical monitoring."
            if curr.HighBP == 1
            else "Status shifted from reported high blood pressure to none. Confirm with your doctor."
        )
    else:
        bp_type = "stable"
        bp_obs = f"Reported blood pressure status remained consistent ({'Hypertension' if curr.HighBP == 1 else 'Normal'})."

    items.append(
        HistoricalComparisonItem(
            indicator="High Blood Pressure",
            previous_value="Yes" if prev.HighBP == 1 else "No",
            current_value="Yes" if curr.HighBP == 1 else "No",
            change_type=bp_type,
            observation=bp_obs,
        )
    )

    # 3. High Cholesterol Comparison
    if prev.HighChol != curr.HighChol:
        chol_type = "declined" if curr.HighChol == 1 else "improved"
        chol_obs = (
            "Status shifted to reported high cholesterol. Discuss lipid testing with your physician."
            if curr.HighChol == 1
            else "Status shifted from reported high cholesterol to none. Verify with fasting lab panels."
        )
    else:
        chol_type = "stable"
        chol_obs = f"Reported cholesterol status remained consistent ({'Elevated' if curr.HighChol == 1 else 'Normal'})."

    items.append(
        HistoricalComparisonItem(
            indicator="High Cholesterol",
            previous_value="Yes" if prev.HighChol == 1 else "No",
            current_value="Yes" if curr.HighChol == 1 else "No",
            change_type=chol_type,
            observation=chol_obs,
        )
    )

    # 4. Physical Activity Comparison
    if prev.PhysActivity != curr.PhysActivity:
        act_type = "improved" if curr.PhysActivity == 1 else "declined"
        act_obs = (
            "Positive lifestyle change: You reported active exercise after previously reporting none."
            if curr.PhysActivity == 1
            else "Reported physical activity decreased to inactive. Consider re-introducing regular movement."
        )
    else:
        act_type = "stable"
        act_obs = f"Physical activity level remained steady ({'Active' if curr.PhysActivity == 1 else 'Inactive'})."

    items.append(
        HistoricalComparisonItem(
            indicator="Physical Activity",
            previous_value="Active" if prev.PhysActivity == 1 else "Inactive",
            current_value="Active" if curr.PhysActivity == 1 else "Inactive",
            change_type=act_type,
            observation=act_obs,
        )
    )

    # 5. General Health Rating Comparison
    prev_gh_label = GEN_HLTH_LABELS.get(prev.GenHlth, str(prev.GenHlth))
    curr_gh_label = GEN_HLTH_LABELS.get(curr.GenHlth, str(curr.GenHlth))
    gh_delta = curr.GenHlth - prev.GenHlth

    if gh_delta < 0:
        gh_type = "improved"
        gh_obs = f"Self-rated general health improved favorably from '{prev_gh_label}' to '{curr_gh_label}'."
    elif gh_delta > 0:
        gh_type = "declined"
        gh_obs = f"Self-rated general health shifted from '{prev_gh_label}' to '{curr_gh_label}'."
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

    # 6. Physical Distress Days Comparison
    p_delta = curr.PhysHlth - prev.PhysHlth
    if p_delta < 0:
        p_type = "improved"
        p_obs = f"Physical distress days decreased by {abs(p_delta)} days ({prev.PhysHlth} → {curr.PhysHlth} days/month)."
    elif p_delta > 0:
        p_type = "declined"
        p_obs = f"Physical distress days increased by {p_delta} days ({prev.PhysHlth} → {curr.PhysHlth} days/month)."
    else:
        p_type = "stable"
        p_obs = f"Physical distress days remained constant at {curr.PhysHlth} days/month."

    items.append(
        HistoricalComparisonItem(
            indicator="Physical Health Days",
            previous_value=f"{prev.PhysHlth} days",
            current_value=f"{curr.PhysHlth} days",
            change_type=p_type,
            observation=p_obs,
        )
    )

    # 7. Mental Distress Days Comparison
    m_delta = curr.MentHlth - prev.MentHlth
    if m_delta < 0:
        m_type = "improved"
        m_obs = f"Mental distress days decreased by {abs(m_delta)} days ({prev.MentHlth} → {curr.MentHlth} days/month)."
    elif m_delta > 0:
        m_type = "declined"
        m_obs = f"Mental distress days increased by {m_delta} days ({prev.MentHlth} → {curr.MentHlth} days/month)."
    else:
        m_type = "stable"
        m_obs = f"Mental distress days remained constant at {curr.MentHlth} days/month."

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
    # Fetch user's assessments ordered newest to oldest
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

    # If specific assessment requested, verify ownership
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

    # Generate recommendations for target assessment
    recommendations = _generate_recommendations_for_assessment(target_assessment)

    # Generate historical comparisons if previous assessment exists
    historical_comparisons: List[HistoricalComparisonItem] = []
    if total_assessments >= 2:
        # Find index of target_assessment
        idx = next((i for i, a in enumerate(assessments) if a.id == target_assessment.id), 0)
        if idx + 1 < total_assessments:
            prev_assessment = assessments[idx + 1]
            historical_comparisons = _generate_historical_comparisons(
                target_assessment, prev_assessment
            )

    # Summary statistics
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

