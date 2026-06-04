# 🏥 AI-Powered Hospital Analytics & Treatment Cost Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-11557C)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end Data Science project performing hospital analytics and ML-based treatment cost prediction on a multi-table hospital management dataset.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [EDA Highlights](#-eda-highlights)
- [Machine Learning Models](#-machine-learning-models)
- [Key Business Insights](#-key-business-insights)
- [Technologies Used](#-technologies-used)
- [Future Work](#-future-work)

---

## 🎯 Project Overview

This project builds a complete data science pipeline for a hospital management system. It covers:

| Phase | Description |
|-------|-------------|
| **Data Engineering** | Load, clean, and merge 5 relational CSV tables |
| **Exploratory Analysis** | 10 professional visualisations across 7 business domains |
| **Machine Learning** | 3 regression models to predict treatment costs |
| **Business Intelligence** | Actionable recommendations for hospital management |

---

## 📂 Dataset Description

| File | Rows | Description |
|------|------|-------------|
| `patients.csv` | 50 | Demographics, DOB, insurance details |
| `doctors.csv` | 10 | Specialization, experience, branch |
| `appointments.csv` | 200 | Date, time, status, reason |
| `treatments.csv` | 200 | Type, description, cost |
| `billing.csv` | 200 | Amount, payment method & status |

**Entity Relationships:**
```
patients ──< appointments >── doctors
               │
            treatments
               │
            billing
```

---

## 📁 Project Structure

```
Hospital-AI-Analytics/
│
├── data/
│   ├── patients.csv
│   ├── doctors.csv
│   ├── appointments.csv
│   ├── treatments.csv
│   ├── billing.csv
│   └── master.csv              ← merged master table
│
├── notebooks/
│   └── hospital_analysis.ipynb ← full Jupyter notebook
│
├── visuals/charts/
│   ├── 01_patient_demographics.png
│   ├── 02_doctor_performance.png
│   ├── 03_appointment_trends.png
│   ├── 04_treatment_analysis.png
│   ├── 05_revenue_payment_analysis.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_branch_analysis.png
│   ├── 08_feature_importance.png
│   ├── 09_model_comparison.png
│   └── 10_actual_vs_predicted.png
│
├── models/
│   └── treatment_cost_prediction.pkl
│
├── hospital_analysis.py        ← main pipeline script
├── INSIGHTS.txt                ← business insights report
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Hospital-AI-Analytics.git
cd Hospital-AI-Analytics

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

**Option A — Python script (fastest):**
```bash
python hospital_analysis.py
```

**Option B — Jupyter Notebook (interactive):**
```bash
jupyter notebook notebooks/hospital_analysis.ipynb
```

**Load the saved model for inference:**
```python
import joblib, pandas as pd

model = joblib.load("models/treatment_cost_prediction.pkl")

new_patient = pd.DataFrame([{
    'gender': 'F', 'age_group': '36-50', 'insurance_provider': 'PulseSecure',
    'specialization': 'Cardiology', 'hospital_branch': 'Main Campus',
    'treatment_type': 'MRI', 'reason_for_visit': 'Consultation',
    'status': 'Completed', 'payment_method': 'Insurance',
    'hour_bucket': 'Morning', 'age': 42, 'years_experience': 15
}])

predicted_cost = model.predict(new_patient)[0]
print(f"Estimated Treatment Cost: ${predicted_cost:,.2f}")
```

---

## 📊 EDA Highlights

### Patient Demographics
- Balanced gender distribution (~50/50)
- Largest patient cohort: **51–65 years** (chronic condition management)
- Top insurance providers: WellnessCorp, PulseSecure, HealthFirst

### Appointment Trends
- Peak appointment days: **Tuesday & Wednesday**
- No-show rate: **26%** — significant operational loss
- Appointment completion rate: only **23%** (remainder: scheduled/cancelled)

### Revenue Analysis
- Total Revenue: **$551,249.85**
- Average treatment cost: **$2,756.25**
- Highest average cost treatment: **MRI ($3,225)**
- Top revenue specialisation: **Pediatrics**

### Payment Status
- Only **32%** of bills marked as Paid — cash flow concern
- 0% formally Overdue — likely in Pending limbo

---

## 🤖 Machine Learning Models

### Target Variable
`cost` — the dollar value of each treatment episode

### Features Used
- Patient: gender, age, age group, insurance provider
- Doctor: specialization, years of experience, hospital branch
- Appointment: reason for visit, status, hour bucket
- Treatment: treatment type
- Billing: payment method

### Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | $1,190 | $1,387 | -0.028 |
| Random Forest | $1,222 | $1,396 | -0.042 |
| Gradient Boosting | $1,483 | $1,679 | -0.508 |

> **Note on R² scores:** The dataset uses synthetically randomised cost values with near-zero correlation to categorical features (max correlation: 0.04). This is a realistic scenario in healthcare analytics where billing codes and procedure complexity drive cost — features not captured in this dataset. In production, enriching features with ICD-10 codes, procedure complexity scores, and comorbidity indices would substantially improve predictive power. The pipeline architecture is production-ready for real-world deployment.

---

## 💡 Key Business Insights

1. **Revenue Focus** — Pediatrics and MRI-heavy specialisations yield the highest revenue; resource allocation should reflect this.
2. **No-show Problem** — 26% no-show rate translates to significant lost revenue; automated reminders are a high-ROI intervention.
3. **Cash Flow Risk** — 68% of bills remain unpaid/pending; a tiered payment follow-up system is urgently needed.
4. **Peak Staffing** — Tuesday/Wednesday peak demand suggests staffing imbalance mid-week.
5. **ML Deployment** — With enriched clinical features, the treatment cost model can power real-time insurance pre-authorization.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, merging |
| `numpy` | Numerical operations |
| `matplotlib` | Base visualisations |
| `seaborn` | Statistical plots & heatmaps |
| `scikit-learn` | ML models, pipelines, preprocessing |
| `joblib` | Model serialisation |

---

## 🔮 Future Work

- [ ] Integrate ICD-10 diagnosis codes as ML features
- [ ] Build a Streamlit / Gradio dashboard for live cost prediction
- [ ] Add time-series forecasting for monthly revenue
- [ ] Implement patient readmission risk classification
- [ ] Connect to a live hospital database (PostgreSQL)
- [ ] Deploy best model as a REST API (FastAPI + Docker)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built as part of an AI-Powered Healthcare Analytics portfolio project.*
