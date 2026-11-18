"""
backend/app/api/chat_routes.py
===============================
AI Health Assistant chat endpoint using Groq API (OpenAI-compatible, ultra-fast).
Model: llama-3.3-70b-versatile (or mixtral-8x7b-32768 as fallback)
Endpoint: https://api.groq.com/openai/v1
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.security import get_current_user
from backend.app.models.user import User

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["AI Health Assistant"])

SYSTEM_PROMPT = """You are HealthBot, a friendly AI health education assistant for DiabetesRisk AI — \
an academic research platform for diabetes risk awareness using the CDC BRFSS 2015 dataset.

RULES YOU MUST FOLLOW:
- Provide GENERAL health education and information ONLY.
- NEVER give medical diagnoses, clinical advice, or treatment recommendations.
- Always remind users to consult qualified healthcare professionals for medical decisions.
- Keep responses concise, well-structured, and empathetic.
- Use simple language — avoid overly technical jargon unless the user asks for details.
- For any direct medical question, include a brief safety disclaimer.

PLATFORM CONTEXT (share when relevant):
- This app uses a Logistic Regression model trained on 253,680 CDC BRFSS 2015 self-reported survey records.
- 3-class prediction: Class 0 (No Diabetes), Class 1 (Prediabetes), Class 2 (Diabetes).
- ROC-AUC: 0.7707 (macro-averaged). Model probability is NOT a clinical diagnosis.
- 21 health indicators analyzed per assessment: HighBP, HighChol, CholCheck, BMI, Smoker, \
Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, \
NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income.
- The app is strictly a research prototype and NOT a certified clinical diagnostic device.

You can help users:
- Understand what their assessment results mean in general terms.
- Learn about risk factors for type 2 diabetes.
- Understand CDC BRFSS health indicators (BMI, blood pressure, cholesterol, etc.).
- Understand healthy lifestyle habits for metabolic health.
- Navigate the app's features (Dashboard, History, New Assessment, etc.)."""


class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    disclaimer: str = (
        "HealthBot provides general health education only. "
        "This is NOT medical advice or a clinical diagnosis. "
        "Always consult a qualified healthcare professional for medical decisions."
    )


@chat_router.post("", response_model=ChatResponse)
async def chat_with_healthbot(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """AI health education assistant powered by Groq (ultra-fast LLM inference)."""
    if not settings.GROQ_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI assistant is not configured. Please add GROQ_API_KEY to backend/.env.",
        )

    try:
        from openai import OpenAI  # Lazy import — only loaded when endpoint is called

        client = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )

        # Build messages list: system prompt + last 10 conversation turns
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in payload.messages[-10:]:
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",   # Groq — confirmed working model
            messages=messages,
            max_tokens=600,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        logger.info("HealthBot replied to user %s (%d chars)", current_user.id, len(reply))
        return ChatResponse(reply=reply)

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="openai package is not installed. Run: pip install openai>=1.0.0",
        )
    except Exception as exc:
        logger.error("HealthBot error for user %s: %s", current_user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI assistant error: {str(exc)}",
        )
