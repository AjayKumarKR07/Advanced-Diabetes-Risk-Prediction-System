/**
 * src/pages/HelpAbout.jsx
 * =========================
 * Phase 6.17.2 — Help & About page.
 * Getting started guide, feature walkthroughs, FAQ, privacy info,
 * and academic project context. Not a medical reference.
 */

import { useState } from 'react'
import {
  HelpCircle, ChevronDown, ChevronRight, PlusCircle,
  LayoutDashboard, FileText, BarChart2, Lightbulb,
  Shield, BookOpen, MessageCircle, Download, Bell,
  Activity, Info, ExternalLink,
} from 'lucide-react'

function Accordion({ question, children }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-slate-50 transition-colors text-left"
      >
        <span className="text-sm font-semibold text-slate-700">{question}</span>
        {open
          ? <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
          : <ChevronRight className="w-4 h-4 text-slate-400 shrink-0" />}
      </button>
      {open && (
        <div className="px-5 pb-4 pt-1 border-t border-slate-100 text-sm text-slate-600 leading-relaxed space-y-2">
          {children}
        </div>
      )}
    </div>
  )
}

function FeatureCard({ icon: Icon, title, path, desc }) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl px-4 py-3 flex gap-3">
      <div className="w-8 h-8 bg-blue-50 rounded-lg flex items-center justify-center shrink-0">
        <Icon className="w-4 h-4 text-blue-600" />
      </div>
      <div>
        <div className="flex items-center gap-2">
          <p className="text-sm font-semibold text-slate-800">{title}</p>
          {path && <code className="text-[10px] text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">{path}</code>}
        </div>
        <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{desc}</p>
      </div>
    </div>
  )
}

export default function HelpAbout() {
  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center shrink-0">
          <HelpCircle className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-slate-900">Help & About</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Guide to using DiabetesRisk AI — an academic research prototype, not a medical diagnostic tool.
          </p>
        </div>
      </div>

      {/* Important disclaimer banner */}
      <div className="bg-amber-50 border border-amber-300 rounded-2xl px-5 py-4 flex gap-3">
        <Shield className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div className="text-sm text-amber-800">
          <p className="font-bold mb-1">Educational Research Prototype — Not a Medical Tool</p>
          <p className="text-xs leading-relaxed">
            DiabetesRisk AI is an academic machine learning research project built on CDC BRFSS 2015
            survey data. It provides <strong>statistical risk estimates based on population
            survey patterns</strong>, not clinical diagnoses. Never use results to make healthcare
            decisions. Always consult a qualified physician or endocrinologist for medical evaluation.
          </p>
        </div>
      </div>

      {/* Project overview */}
      <div>
        <h2 className="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-600" /> Project Overview
        </h2>
        <div className="bg-white border border-slate-200 rounded-2xl p-5 text-sm text-slate-600 space-y-2 leading-relaxed">
          <p>
            DiabetesRisk AI applies a frozen Phase 5 Multinomial Logistic Regression model to 21 CDC
            health indicators collected from US adults via the BRFSS 2015 telephone survey. The system
            classifies responses into three categories: <strong>No Diabetes (Class 0)</strong>,{' '}
            <strong>Prediabetes (Class 1)</strong>, and <strong>Diabetes (Class 2)</strong>.
          </p>
          <p>
            The model achieves 64.3% test accuracy with a balanced accuracy of 51.7%. Due to heavy class
            imbalance (~2.5% prediabetes), Class 1 performance is limited (Precision: 0.03, F1: 0.06).
            These limitations are disclosed throughout the application.
          </p>
          <p>
            All data is processed and stored securely per user account. No health data is shared with
            third parties. This is a demonstration project — results carry no clinical weight.
          </p>
        </div>
      </div>

      {/* Getting started */}
      <div>
        <h2 className="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
          <PlusCircle className="w-4 h-4 text-blue-600" /> Getting Started
        </h2>
        <div className="space-y-3">
          {[
            {
              step: '1',
              title: 'Create an Account',
              body: 'Register with your name and email at /register. Passwords are hashed with bcrypt — your credentials are never stored in plaintext.',
            },
            {
              step: '2',
              title: 'Complete a Health Assessment',
              body: 'Navigate to New Assessment (/assess) and answer 21 CDC BRFSS health indicator questions. The form validates all inputs against CDC encoding rules before submission.',
            },
            {
              step: '3',
              title: 'Review Your Results',
              body: 'After submission, view your risk class, probability breakdown, and AI explanation (XAI) showing which factors influenced the prediction most.',
            },
            {
              step: '4',
              title: 'Track Over Time',
              body: 'Complete additional assessments to unlock Analytics and trend tracking. The Dashboard shows your risk history and key health metrics.',
            },
          ].map(({ step, title, body }) => (
            <div key={step} className="bg-white border border-slate-200 rounded-xl px-5 py-4 flex gap-4">
              <div className="w-7 h-7 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold shrink-0">
                {step}
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">{title}</p>
                <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature guide */}
      <div>
        <h2 className="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
          <LayoutDashboard className="w-4 h-4 text-blue-600" /> Application Features
        </h2>
        <div className="space-y-2">
          <FeatureCard icon={LayoutDashboard} title="Dashboard" path="/dashboard"
            desc="Overview of your latest risk class, probability score, BMI, and recent assessment history. Quick links to new assessments and analytics." />
          <FeatureCard icon={BarChart2} title="Personal Analytics" path="/analytics"
            desc="Charts showing BMI trend, risk class distribution, and health score progression across all your assessments. Requires at least 1 completed assessment." />
          <FeatureCard icon={Lightbulb} title="Health Insights" path="/insights"
            desc="Personalised evidence-based health guidance based on your risk profile. Covers nutrition, exercise, clinical monitoring, and lifestyle recommendations." />
          <FeatureCard icon={FileText} title="Health Reports" path="/reports"
            desc="Generate and download multi-page PDF reports for any assessment. Reports include risk probability, feature contributions (XAI), and health education notes." />
          <FeatureCard icon={Activity} title="Model Performance" path="/model-performance"
            desc="Frozen Phase 5 evaluation metrics: accuracy, F1-scores, ROC-AUC, and per-class performance charts. Transparently shows model limitations." />
          <FeatureCard icon={BookOpen} title="Research & Methodology" path="/research"
            desc="Detailed documentation of the CDC BRFSS 2015 dataset, 21-feature definitions, preprocessing steps, group-aware validation, and scientific limitations." />
          <FeatureCard icon={Bell} title="Notifications" path="/notifications"
            desc="Health reminders and alerts based on your assessment history — 30-day reassessment prompts, risk alerts, and feature announcements." />
          <FeatureCard icon={MessageCircle} title="HealthBot AI Chat" path="(chat bubble)"
            desc="Floating AI chat assistant powered by Groq (openai/gpt-oss-120b). Ask questions about your results, diabetes risk factors, or CDC health indicators." />
        </div>
      </div>

      {/* Understanding Results */}
      <div>
        <h2 className="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
          <BarChart2 className="w-4 h-4 text-blue-600" /> Understanding Your Prediction Results
        </h2>
        <div className="bg-white border border-slate-200 rounded-2xl p-5 space-y-3 text-sm text-slate-600">
          <div>
            <p className="font-semibold text-slate-800 mb-1">Risk Class</p>
            <p className="text-xs leading-relaxed">
              The model outputs one of three classes: <span className="text-emerald-700 font-semibold">No Diabetes (0)</span>,{' '}
              <span className="text-amber-700 font-semibold">Prediabetes (1)</span>, or{' '}
              <span className="text-red-700 font-semibold">Diabetes (2)</span>. This is the class with the highest
              predicted probability — it reflects population-level patterns, not individual clinical status.
            </p>
          </div>
          <div>
            <p className="font-semibold text-slate-800 mb-1">Probability Score</p>
            <p className="text-xs leading-relaxed">
              The probability percentage shows the model's confidence in the predicted class using the
              softmax function. A higher percentage indicates stronger statistical resemblance to survey
              respondents in that group — it is <strong>not</strong> a medical risk percentage.
            </p>
          </div>
          <div>
            <p className="font-semibold text-slate-800 mb-1">AI Explanation (XAI)</p>
            <p className="text-xs leading-relaxed">
              The feature contribution chart shows which of your 21 health indicators pushed the
              prediction toward (red) or away from (green) the predicted class. Contributions are
              computed from logistic regression coefficients × standardised feature values. This is for
              educational understanding only — correlation, not causation.
            </p>
          </div>
          <div className="bg-amber-50 border border-amber-200 rounded-xl px-3 py-2">
            <p className="text-xs text-amber-700 font-semibold">
              ⚠️ Prediabetes predictions carry high uncertainty (Precision: 3.1%). If classified as Class 1,
              consult a physician for a formal HbA1c or Fasting Plasma Glucose test.
            </p>
          </div>
        </div>
      </div>

      {/* PDF Reports */}
      <div>
        <h2 className="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
          <Download className="w-4 h-4 text-blue-600" /> Generating PDF Reports
        </h2>
        <div className="bg-white border border-slate-200 rounded-2xl p-5 text-sm text-slate-600 space-y-2 leading-relaxed">
          <p>Navigate to <strong>Health Reports</strong> (/reports) to generate PDF downloads.</p>
          <ul className="space-y-1.5 list-none">
            {[
              'Click "Download Latest Report" to generate a PDF for your most recent assessment.',
              'Each row in the Assessment Records table has an individual PDF download button.',
              'Reports include: patient info, risk class, probability scores, XAI breakdown, and health education notes.',
              'PDF generation happens entirely in your browser — no data is transmitted externally.',
              'jsPDF is loaded on-demand only when you click download (not in the initial page load).',
            ].map((pt, i) => (
              <li key={i} className="flex items-start gap-2 text-xs">
                <span className="text-blue-500 font-bold shrink-0">•</span> {pt}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Privacy & Security */}
      <div>
        <h2 className="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
          <Shield className="w-4 h-4 text-blue-600" /> Privacy & Security
        </h2>
        <div className="space-y-2">
          {[
            { title: 'Authentication', body: 'JWT Bearer tokens with configurable expiry. Passwords hashed with bcrypt (12 rounds). Tokens stored in memory — not localStorage.' },
            { title: 'Data Isolation', body: 'All assessment, prediction, and notification data is scoped to your authenticated user ID. No cross-user data access is possible through the API.' },
            { title: 'No Third-Party Sharing', body: 'Your health assessment data is not shared with any third party. HealthBot messages are sent to Groq API without personally identifiable information.' },
            { title: 'Avatar Storage', body: 'Profile avatars are stored server-side per user ID. File format and size are validated (JPEG/PNG/WebP, max 2 MB).' },
            { title: 'No Clinical Storage', body: 'This system does not store data in a HIPAA-compliant manner. Do not enter real personal health information into a research prototype.' },
          ].map(({ title, body }) => (
            <div key={title} className="bg-white border border-slate-200 rounded-xl px-4 py-3">
              <p className="text-xs font-bold text-slate-700 mb-0.5 flex items-center gap-1.5">
                <Shield className="w-3 h-3 text-blue-500" /> {title}
              </p>
              <p className="text-xs text-slate-500 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </div>

      {/* FAQ */}
      <div>
        <h2 className="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
          <HelpCircle className="w-4 h-4 text-blue-600" /> Frequently Asked Questions
        </h2>
        <div className="space-y-2">
          <Accordion question="Is this a real medical test?">
            <p>No. DiabetesRisk AI is an academic research engineering project. It uses a machine learning model
            trained on a 2015 public survey dataset. Results are population-level statistical estimates and
            cannot replace clinical laboratory tests (HbA1c, Fasting Plasma Glucose) or a physician's evaluation.</p>
          </Accordion>
          <Accordion question="Why is my Prediabetes result uncertain?">
            <p>Class 1 (Prediabetes) represents only ~2.5% of the training data. Despite using balanced class
            weights, the model achieves a precision of only 3.1% and an F1 of 5.6% for this class. If you
            receive a Prediabetes result, we strongly recommend requesting a formal HbA1c test from your doctor.</p>
          </Accordion>
          <Accordion question="Can I delete my account and data?">
            <p>Yes. Go to Settings → Account → Delete Account. This permanently deletes your account and
            all associated assessments, predictions, and profile data from the database.</p>
          </Accordion>
          <Accordion question="Why does the same assessment give different results after re-submission?">
            <p>The model is deterministic — identical inputs always produce identical outputs. If you observe
            different results, your inputs may have changed (even slightly), particularly BMI, Age, or categorical
            fields that use ordinal encoding.</p>
          </Accordion>
          <Accordion question="How do notifications work?">
            <p>Notifications are generated from your assessment history using rule-based logic on the server.
            They include 30-day reassessment reminders, risk class alerts, and feature tips. Read state is
            tracked in your browser (localStorage) — notifications reset if you clear browser data or switch browsers.</p>
          </Accordion>
          <Accordion question="What is HealthBot?">
            <p>HealthBot is an AI chat assistant powered by Groq's inference API (model: openai/gpt-oss-120b).
            It can answer questions about your results, explain CDC health indicators, and provide general
            diabetes awareness information. It is not a medical professional and its responses are not medical advice.</p>
          </Accordion>
          <Accordion question="How do I generate a PDF report?">
            <p>Navigate to Health Reports (/reports). Click "Download Latest Report" for your most recent
            assessment, or click the PDF button next to any individual assessment row. The PDF is generated
            entirely in your browser using jsPDF — nothing is uploaded externally.</p>
          </Accordion>
          <Accordion question="Is my data encrypted?">
            <p>Data in transit is protected by HTTPS (TLS) when deployed. Passwords are hashed with bcrypt.
            The PostgreSQL database is not encrypted at rest in the development configuration — this is a
            research prototype, not a production healthcare system.</p>
          </Accordion>
        </div>
      </div>

      {/* Academic info */}
      <div className="bg-slate-50 border border-slate-200 rounded-2xl px-5 py-4 flex items-start gap-3">
        <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-500 space-y-1">
          <p className="font-semibold text-slate-600">Academic Project Information</p>
          <p>
            DiabetesRisk AI is a full-stack machine learning engineering project demonstrating end-to-end
            deployment of a research-grade diabetes risk prediction model. The project spans data
            preprocessing (CDC BRFSS 2015), group-aware model benchmarking, Phase 5 frozen evaluation,
            REST API backend (FastAPI + PostgreSQL), and a React + Vite frontend.
          </p>
          <p>
            All evaluation metrics, model artifacts, and methodology details are real and unmodified.
            No results have been fabricated or cherry-picked.
          </p>
        </div>
      </div>
    </div>
  )
}
