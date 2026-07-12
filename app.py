import os
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from fpdf import FPDF

# ==========================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="CrediShield AI Co-Pilot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "decision_history" not in st.session_state:
    st.session_state.decision_history = []


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")



@st.cache_resource
def train_risk_model():
    """
    Trains a small logistic regression on synthetic-but-realistic data so the
    dashboard has a genuine model to explain with SHAP, rather than a
    hand-coded if/else score. This is a stand-in for a real underwriting
    model trained on historical loan performance data.
    """
    rng = np.random.default_rng(42)
    n = 4000

    credit_hist = rng.integers(0, 2, n)
    income = rng.normal(5000, 2200, n).clip(500, None)
    loan_amt = rng.normal(150000, 65000, n).clip(5000, None)
    employment_years = rng.normal(5, 3, n).clip(0, None)

    risk_logit = (
        -1.0
        + 2.6 * (1 - credit_hist)
        - 0.00028 * (income - 3000)
        + 0.0000035 * (loan_amt - 100000)
        - 0.07 * employment_years
    )
    prob_default = 1 / (1 + np.exp(-risk_logit))
    y = rng.binomial(1, prob_default)

    X = np.column_stack([credit_hist, income, loan_amt, employment_years])
    model = LogisticRegression()
    model.fit(X, y)

    explainer = shap.LinearExplainer(model, X)
    return model, explainer


FEATURE_NAMES = ["Credit History OK", "Monthly Income", "Loan Amount", "Employment (yrs)"]


def run_risk_prediction(model, explainer, tabular_data):
    X = np.array([[
        tabular_data["Credit_History"],
        tabular_data["ApplicantIncome"],
        tabular_data["LoanAmount"],
        tabular_data["EmploymentYears"],
    ]])
    prob_default = model.predict_proba(X)[0][1]
    shap_values = explainer.shap_values(X)[0]
    return float(prob_default), shap_values


def run_cnn_verification(image_file):
    """Placeholder for a document/collateral authenticity check.
    Wire this up to a real vision model or ID-verification API for production use."""
    time.sleep(0.3)
    if image_file is not None:
        return 0.92, "Match Verified (simulated): Document authenticity high."
    return 0.0, "No image uploaded."


# ==========================================
# 3. LLM SUMMARIZATION (real Groq API call)
# ==========================================
GROQ_MODEL = "llama-3.3-70b-versatile"  # fast + free-tier friendly; swap for e.g. "llama-3.1-8b-instant" if you want it even faster/cheaper


def run_llm_summarization(applicant_notes: str, tabular_data: dict, api_key: str) -> str:
    if not applicant_notes.strip():
        return "No applicant notes provided."

    if not api_key:
        return "⚠️ GROQ_API_KEY is not configured."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        prompt = f"""You are an underwriting assistant helping a human loan officer review an applicant's letter of intent.

Applicant context:
- Monthly income: ${tabular_data['ApplicantIncome']:,}
- Requested loan amount: ${tabular_data['LoanAmount']:,}
- Credit history meets guidelines: {"Yes" if tabular_data['Credit_History'] == 1 else "No"}

Applicant's letter of intent / background notes:
\"\"\"{applicant_notes}\"\"\"

Respond with three short sections:
1. Summary (1-2 sentences) of the applicant's stated purpose for the loan.
2. Risk flags: any inconsistencies, red flags, or missing context worth the underwriter's attention (or state "None apparent").
3. Suggested follow-up question(s) for the underwriter to ask, if any.

Keep the whole response under 150 words. Do not make a final approve/deny decision — that stays with the human underwriter."""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ LLM call failed ({e.__class__.__name__}): {e}\n\nFalling back — please review notes manually."


# ==========================================
# 4. PDF REPORT GENERATOR
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'CrediShield AI Co-Pilot - Underwriting Report', 0, 1, 'C')
        self.ln(10)


def generate_pdf(data, risk_score, llm_summary, decision, comments):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    def sanitize(text):
        if text is None:
            return ""
        return str(text).encode('latin-1', errors='replace').decode('latin-1')

    pdf.cell(0, 10, sanitize(f"Applicant Name: {data['Name']}"), 0, 1)
    pdf.cell(0, 10, sanitize(f"Requested Loan Amount: ${data['LoanAmount']:,}"), 0, 1)
    pdf.cell(0, 10, sanitize(f"Credit History Flag: {data['Credit_History']}"), 0, 1)
    pdf.cell(0, 10, sanitize(f"Primary AI Risk Score: {risk_score * 100:.1f}%"), 0, 1)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "AI Co-Pilot Text Summarization:", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, sanitize(llm_summary))
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Human-in-the-Loop Final Decision:", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 10, sanitize(f"Decision Status: {decision}"), 0, 1)
    pdf.multi_cell(0, 7, sanitize(f"Underwriter Comments: {comments}"))

    try:
        return pdf.output(dest='S').encode('latin-1', errors='replace')
    except AttributeError:
        return bytes(pdf.output())
    except TypeError:
        return pdf.output()


# Automatically load the API key
active_api_key = get_api_key()


# Initialize the ML model
model, explainer = train_risk_model()

# ==========================================
# 6. USER INTERFACE
# ==========================================
st.title("🏦 CrediShield: AI Co-Pilot for Bank Loan Underwriting")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📋 Underwriting Dashboard", "📊 Decision History Logs", "💼 Business Model & Strategy"])

# ------------------------------------------
# TAB 1: UNDERWRITING DASHBOARD
# ------------------------------------------
with tab1:
    st.header("Applicant Multimodal Evaluation Panel")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Modality 1: Tabular Financial Data")
        app_name = st.text_input("Applicant Full Name", "Jane Doe")
        income = st.number_input("Monthly Applicant Income ($)", min_value=0, value=5500, step=500)
        loan_amt = st.number_input("Requested Loan Amount ($)", min_value=1000, value=150000, step=5000)
        employment_years = st.number_input("Employment Length (years)", min_value=0.0, value=4.0, step=0.5)
        credit_hist = st.selectbox("Credit History Meets Guidelines?", options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

        st.subheader("Modality 2: Unstructured Text Data")
        notes = st.text_area(
            "Applicant Letter of Intent / Background Notes",
            value="I am applying for this loan to consolidate my high-interest credit card debt accumulated during a medical transition last year. I have maintained steady employment for 4 years."
        )

    with col2:
        st.subheader("Modality 3: Image Data (KYC / Collateral)")
        uploaded_image = st.file_uploader("Upload Scanned ID or Property Collateral Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Modality Asset", width=300)
        else:
            st.info("Please upload an image asset to fulfill the multimodal pipeline validation.")

    st.markdown("---")
    if st.button("⚡ Run Co-Pilot Multimodal Risk Assessment", type="primary"):
        if not app_name.strip():
            st.error("Applicant name is required.")
        else:
            with st.spinner("Processing risk model, document check, and note analysis..."):
                tab_payload = {
                    'ApplicantIncome': income,
                    'LoanAmount': loan_amt,
                    'Credit_History': credit_hist,
                    'EmploymentYears': employment_years,
                    'Name': app_name,
                }

                risk_pct, shap_values = run_risk_prediction(model, explainer, tab_payload)
                cnn_conf, cnn_msg = run_cnn_verification(uploaded_image)
                llm_explanation = run_llm_summarization(notes, tab_payload, active_api_key)

                st.session_state.current_analysis = {
                    "name": app_name,
                    "data": tab_payload,
                    "risk_score": risk_pct,
                    "shap_values": shap_values,
                    "cnn_status": cnn_msg,
                    "llm_summary": llm_explanation,
                }

    if "current_analysis" in st.session_state:
        res = st.session_state.current_analysis

        st.subheader("🤖 Co-Pilot Assessment Matrix")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(label="Modeled Default Risk", value=f"{res['risk_score']*100:.1f}%", delta="- Low Risk" if res['risk_score'] < 0.4 else "+ High Risk")
            st.caption("Engine: Logistic regression risk model")

        with res_col2:
            st.markdown("**Image Verification Metric:**")
            st.info(res['cnn_status'])

        with res_col3:
            st.markdown("**LLM Text Insights & Summarization:**")
            st.write(res['llm_summary'])

        st.markdown("---")
        st.subheader("🔍 Explainable AI (XAI) Dashboard")

        xai_col1, xai_col2 = st.columns(2)
        with xai_col1:
            st.markdown("**Feature Contribution (real SHAP values)**")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            colors = ['#d62728' if v > 0 else '#2ca02c' for v in res['shap_values']]
            ax.barh(FEATURE_NAMES, res['shap_values'], color=colors)
            ax.set_xlabel('SHAP contribution to default-risk logit')
            ax.axvline(0, color='black', linewidth=0.8)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Red = pushes risk up, green = pushes risk down.")

        with xai_col2:
            st.markdown("**Confidence Interval Distribution**")
            mu, sigma = res['risk_score'], 0.1
            rng = np.random.default_rng()
            s = rng.normal(mu, sigma, 1000).clip(0, 1)
            fig2, ax2 = plt.subplots(figsize=(5, 2.5))
            ax2.hist(s, 30, density=True, alpha=0.6)
            ax2.axvline(mu, color='r', linestyle='dashed', linewidth=2)
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption("Illustrative uncertainty band around the point estimate, not a calibrated interval.")

        st.markdown("---")
        st.subheader("🧑‍✈️ Human-in-the-Loop Override Panel")

        underwriter_comments = st.text_area("Underwriter Justification & Audit Log Notes")

        final_col1, final_col2, final_col3 = st.columns(3)

        with final_col1:
            if st.button("✅ Approve Application", use_container_width=True):
                st.session_state.decision_history.append({
                    "name": res['name'], "decision": "APPROVED",
                    "score": res['risk_score'], "comments": underwriter_comments
                })
                st.success(f"Application for {res['name']} marked as APPROVED.")

        with final_col2:
            if st.button("❌ Reject Application", use_container_width=True):
                st.session_state.decision_history.append({
                    "name": res['name'], "decision": "REJECTED",
                    "score": res['risk_score'], "comments": underwriter_comments
                })
                st.error(f"Application for {res['name']} marked as REJECTED.")

        with final_col3:
            pdf_bytes = generate_pdf(res['data'], res['risk_score'], res['llm_summary'], "Processed", underwriter_comments)
            st.download_button(
                label="📄 Export Standard Audit Report PDF",
                data=pdf_bytes,
                file_name=f"CrediShield_Report_{res['name'].replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# ------------------------------------------
# TAB 2: DECISION HISTORY LOGS
# ------------------------------------------
with tab2:
    st.header("Operational Human-in-the-Loop Audit Registry")
    if len(st.session_state.decision_history) == 0:
        st.info("No applications processed during this session yet.")
    else:
        df_logs = pd.DataFrame(st.session_state.decision_history)
        st.dataframe(df_logs, use_container_width=True)
        st.download_button(
            "⬇️ Download Logs as CSV",
            data=df_logs.to_csv(index=False).encode("utf-8"),
            file_name="credishield_decision_logs.csv",
            mime="text/csv",
        )

# ------------------------------------------
# TAB 3: BUSINESS MODEL & STRATEGY
# ------------------------------------------
with tab3:
    st.header("Commercialization Strategy & Value Proposition")

    st.markdown("""
    ### **The Problem Opportunity**
    Traditional mortgage and commercial loan underwriting pipelines take anywhere between **5 to 15 business days** to reconcile financial history datasets, verify identity claims, and manually skim applicant intake rationales.

    ### **Our Solution: CrediShield Co-Pilot**
    CrediShield decreases verification cycle times down to **less than 10 minutes** by integrating a scalable multimodal architecture that automates deep technical vetting while keeping the human credit specialist firmly in control.
    """)

    biz_col1, biz_col2, biz_col3 = st.columns(3)

    with biz_col1:
        st.metric(label="B2B SaaS License / Seat Fee", value="$150 / mo")
        st.caption("Targeted at regional community credit unions and regional commercial banks.")

    with biz_col2:
        st.metric(label="API Consumption Cost", value="$0.45 / application")
        st.caption("Volume processing pricing scaling past 10,000 monthly application volumes.")

    with biz_col3:
        st.metric(label="Projected Customer ROI Increase", value="34%")
        st.caption("Calculated via optimized employee hours spent reviewing physical artifacts manually.")
