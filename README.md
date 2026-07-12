# AI-Co-Pilot-For-Bank-Loan
An AI co-pilot for bank loan repossession is a specialized digital assistant designed to streamline the recovery process, improve decision-making, and ensure regulatory compliance.


# 🏦 CrediShield AI Co-Pilot for Bank Loan Underwriting

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![SHAP](https://img.shields.io/badge/Explainable-AI-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 📌 Overview

CrediShield is an AI-powered Bank Loan Underwriting Co-Pilot that assists loan officers in evaluating loan applications using **Multimodal Artificial Intelligence**, **Explainable AI (XAI)**, and a **Human-in-the-Loop** workflow.

Instead of replacing human underwriters, the system provides intelligent recommendations, risk predictions, document verification, and AI-generated summaries to accelerate and improve the loan approval process.

---

## 🎯 Features

- 📊 AI-powered Loan Risk Prediction
- 📝 LLM-based Applicant Note Analysis using Groq (Llama 3.3)
- 🖼️ KYC / Collateral Image Verification
- 🔍 Explainable AI (SHAP Feature Importance)
- 👨‍💼 Human-in-the-Loop Decision Making
- 📄 Automatic PDF Report Generation
- 📈 Interactive Streamlit Dashboard
- 📂 Decision History Logging
- 📥 CSV Export
- ⚡ Fast and User-Friendly Interface

---

# 🏗️ System Architecture

```
                    Applicant Information
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
      ▼                    ▼                    ▼
 Financial Data      Applicant Notes      KYC Image
(Tabular Data)         (Text Data)      (Image Data)
      │                    │                    │
      ▼                    ▼                    ▼
 Logistic Regression     Groq LLM      Image Verification
      │                    │                    │
      └──────────────┬─────┴────────────────────┘
                     ▼
            AI Risk Assessment Engine
                     │
                     ▼
           Explainable AI (SHAP)
                     │
                     ▼
      Human Underwriter Decision Panel
                     │
                     ▼
           PDF Report & Audit Logs
```

---

# 🧠 AI Components

### 📊 Machine Learning

- Logistic Regression
- Probability-based Risk Prediction
- Synthetic Loan Dataset

### 🤖 Large Language Model

- Groq API
- Llama 3.3 70B Versatile

Used for:

- Applicant Note Summarization
- Risk Flag Detection
- Follow-up Question Suggestions

### 🔍 Explainable AI

- SHAP
- Feature Contribution Visualization
- Model Transparency

---

# 📦 Project Structure

```
CrediShield-AI-CoPilot/
│
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   ├── dashboard.png
│   ├── workflow.png
│   └── report.png
│
├── reports/
│
└── .streamlit/
    └── secrets.toml
```

---

# 📊 Dashboard Modules

### 🏦 Underwriting Dashboard

- Applicant Information
- Income
- Loan Amount
- Employment History
- Credit History

---

### 📝 AI Text Analysis

- Applicant Letter Analysis
- Summary Generation
- Risk Flags
- Suggested Questions

---

### 🖼️ Document Verification

- KYC Upload
- Property Collateral Upload
- Verification Status

---

### 🔍 Explainable AI

- SHAP Bar Charts
- Feature Contributions
- Confidence Distribution

---

### 👨‍💼 Human-in-the-Loop

- Approve Application
- Reject Application
- Add Comments
- Audit Logging

---

### 📄 Report Generation

Generate professional underwriting reports including:

- Applicant Details
- AI Risk Score
- LLM Summary
- Final Decision
- Underwriter Notes

---

# 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Frontend | Streamlit |
| Machine Learning | Scikit-Learn |
| Explainability | SHAP |
| Visualization | Matplotlib |
| Data Processing | NumPy, Pandas |
| LLM | Groq API |
| PDF Reports | FPDF |

---

# 📥 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/CrediShield-AI-CoPilot.git
```

Navigate into the project

```bash
cd CrediShield-AI-CoPilot
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Configure Groq API

Create

```
.streamlit/secrets.toml
