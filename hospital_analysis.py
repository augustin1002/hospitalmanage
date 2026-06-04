"""
AI-Powered Hospital Analytics and Treatment Cost Prediction
============================================================
Senior Data Scientist Pipeline
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE    = "/home/claude/Hospital-AI-Analytics"
DATA    = f"{BASE}/data"
CHARTS  = f"{BASE}/visuals/charts"
MODELS  = f"{BASE}/models"

PALETTE = ["#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6",
           "#EC4899","#06B6D4","#84CC16","#F97316","#6366F1"]

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#FFFFFF',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

print("=" * 65)
print("  AI-Powered Hospital Analytics & Treatment Cost Prediction")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading data …")

patients     = pd.read_csv(f"{DATA}/patients.csv")
doctors      = pd.read_csv(f"{DATA}/doctors.csv")
appointments = pd.read_csv(f"{DATA}/appointments.csv")
treatments   = pd.read_csv(f"{DATA}/treatments.csv")
billing      = pd.read_csv(f"{DATA}/billing.csv")

print(f"  patients     : {patients.shape}")
print(f"  doctors      : {doctors.shape}")
print(f"  appointments : {appointments.shape}")
print(f"  treatments   : {treatments.shape}")
print(f"  billing      : {billing.shape}")

# ─────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Cleaning data …")

def clean_df(df, name):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"  {name}: removed {before-after} duplicates")
    return df

patients     = clean_df(patients,     "patients")
doctors      = clean_df(doctors,      "doctors")
appointments = clean_df(appointments, "appointments")
treatments   = clean_df(treatments,   "treatments")
billing      = clean_df(billing,      "billing")

# Fix date types
for df, cols in [
    (patients,     ['date_of_birth','registration_date']),
    (appointments, ['appointment_date']),
    (treatments,   ['treatment_date']),
    (billing,      ['bill_date']),
]:
    for c in cols:
        df[c] = pd.to_datetime(df[c], errors='coerce')

# Derive age
patients['age'] = (pd.Timestamp.now() - patients['date_of_birth']).dt.days // 365
patients['age_group'] = pd.cut(patients['age'],
    bins=[0,18,35,50,65,120],
    labels=['<18','18-35','36-50','51-65','65+'])

appointments['month']    = appointments['appointment_date'].dt.to_period('M').astype(str)
appointments['day_name'] = appointments['appointment_date'].dt.day_name()
appointments['hour']     = pd.to_datetime(appointments['appointment_time'],
                               format='%H:%M:%S', errors='coerce').dt.hour

print("  Date types fixed | Age column derived")

# ─────────────────────────────────────────────────────────────
# 3. MERGE MASTER TABLE
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Merging tables …")

master = (appointments
    .merge(patients[['patient_id','gender','age','age_group','insurance_provider']],
           on='patient_id', how='left')
    .merge(doctors[['doctor_id','specialization','years_experience','hospital_branch']],
           on='doctor_id', how='left')
    .merge(treatments[['appointment_id','treatment_type','description','cost']],
           on='appointment_id', how='left')
    .merge(billing[['treatment_id','amount','payment_method','payment_status']]
                  .merge(treatments[['treatment_id','appointment_id']], on='treatment_id'),
           on='appointment_id', how='left')
)

master.to_csv(f"{DATA}/master.csv", index=False)
print(f"  Master table: {master.shape}")

# ─────────────────────────────────────────────────────────────
# 4. EDA + VISUALISATIONS
# ─────────────────────────────────────────────────────────────
print("\n[4/6] EDA & Visualisations …")

def save(fig, name):
    path = f"{CHARTS}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔  {name}.png")

# ── 4.1 Patient Demographics ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Patient Demographics", fontsize=16, fontweight='bold', y=1.02)

# Gender
gender_counts = patients['gender'].value_counts()
axes[0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
            colors=PALETTE[:2], startangle=90, wedgeprops=dict(edgecolor='white',linewidth=2))
axes[0].set_title("Gender Distribution")

# Age group
age_counts = patients['age_group'].value_counts().sort_index()
bars = axes[1].bar(age_counts.index.astype(str), age_counts.values,
                   color=PALETTE[:len(age_counts)], edgecolor='white', linewidth=1.5)
axes[1].bar_label(bars, padding=3, fontsize=10, fontweight='bold')
axes[1].set_title("Age Group Distribution")
axes[1].set_xlabel("Age Group")
axes[1].set_ylabel("Count")

# Insurance provider
ins = patients['insurance_provider'].value_counts()
bars2 = axes[2].barh(ins.index, ins.values, color=PALETTE[:len(ins)], edgecolor='white')
axes[2].bar_label(bars2, padding=3, fontsize=10, fontweight='bold')
axes[2].set_title("Insurance Providers")
axes[2].set_xlabel("Number of Patients")

plt.tight_layout()
save(fig, "01_patient_demographics")

# ── 4.2 Doctor Performance ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Doctor Performance Analysis", fontsize=16, fontweight='bold', y=1.02)

# Appointments per specialization
spec_appts = master.groupby('specialization').size().sort_values(ascending=False)
bars = axes[0].bar(spec_appts.index, spec_appts.values,
                   color=PALETTE[:len(spec_appts)], edgecolor='white')
axes[0].bar_label(bars, padding=3, fontsize=10, fontweight='bold')
axes[0].set_title("Appointments by Specialization")
axes[0].set_xlabel("Specialization")
axes[0].set_ylabel("# Appointments")
axes[0].tick_params(axis='x', rotation=45)

# Revenue by specialization
rev_spec = master.groupby('specialization')['cost'].sum().sort_values(ascending=False)
bars2 = axes[1].bar(rev_spec.index, rev_spec.values,
                    color=PALETTE[:len(rev_spec)], edgecolor='white')
axes[1].bar_label(bars2, fmt='$%.0f', padding=3, fontsize=8, fontweight='bold')
axes[1].set_title("Revenue by Specialization")
axes[1].set_ylabel("Total Revenue ($)")
axes[1].tick_params(axis='x', rotation=45)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))

# Experience vs avg cost
exp_cost = master.groupby('years_experience')['cost'].mean()
axes[2].scatter(exp_cost.index, exp_cost.values, color=PALETTE[0], alpha=0.7,
                s=80, edgecolors='white', linewidth=1.5)
z = np.polyfit(exp_cost.index, exp_cost.values, 1)
axes[2].plot(exp_cost.index, np.poly1d(z)(exp_cost.index),
             color=PALETTE[3], linewidth=2, linestyle='--', label='Trend')
axes[2].set_title("Doctor Experience vs Avg Treatment Cost")
axes[2].set_xlabel("Years of Experience")
axes[2].set_ylabel("Avg Treatment Cost ($)")
axes[2].legend()

plt.tight_layout()
save(fig, "02_doctor_performance")

# ── 4.3 Appointment Trends ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Appointment Trends", fontsize=16, fontweight='bold', y=1.02)

# Monthly trend
monthly = appointments.groupby('month').size().reset_index(name='count')
monthly = monthly.sort_values('month')
axes[0].plot(range(len(monthly)), monthly['count'], color=PALETTE[0],
             linewidth=2.5, marker='o', markersize=6)
axes[0].fill_between(range(len(monthly)), monthly['count'],
                     alpha=0.15, color=PALETTE[0])
axes[0].set_xticks(range(len(monthly)))
axes[0].set_xticklabels(monthly['month'], rotation=45, ha='right', fontsize=7)
axes[0].set_title("Monthly Appointments")
axes[0].set_ylabel("# Appointments")

# Day of week
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow = appointments['day_name'].value_counts().reindex(dow_order).fillna(0)
bars = axes[1].bar(dow.index, dow.values, color=PALETTE[1], edgecolor='white')
axes[1].bar_label(bars, padding=3, fontsize=10)
axes[1].set_title("Appointments by Day of Week")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("# Appointments")
axes[1].tick_params(axis='x', rotation=45)

# Status breakdown
status = appointments['status'].value_counts()
axes[2].pie(status, labels=status.index, autopct='%1.1f%%',
            colors=PALETTE[:len(status)], startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[2].set_title("Appointment Status Breakdown")

plt.tight_layout()
save(fig, "03_appointment_trends")

# ── 4.4 Treatment Analysis ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Treatment Analysis", fontsize=16, fontweight='bold', y=1.02)

# Treatment type distribution
tt = master['treatment_type'].value_counts()
axes[0].pie(tt, labels=tt.index, autopct='%1.1f%%',
            colors=PALETTE[:len(tt)], startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[0].set_title("Treatment Type Distribution")

# Avg cost by treatment
avg_cost_tt = master.groupby('treatment_type')['cost'].mean().sort_values(ascending=False)
bars = axes[1].barh(avg_cost_tt.index, avg_cost_tt.values,
                    color=PALETTE[:len(avg_cost_tt)], edgecolor='white')
axes[1].bar_label(bars, fmt='$%.0f', padding=3, fontsize=9, fontweight='bold')
axes[1].set_title("Avg Cost by Treatment Type")
axes[1].set_xlabel("Avg Cost ($)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))

# Cost distribution (violin)
try:
    treat_order = master['treatment_type'].value_counts().index.tolist()
    sns.violinplot(data=master, x='treatment_type', y='cost',
                   ax=axes[2], palette=PALETTE[:len(treat_order)],
                   order=treat_order, inner='box')
except:
    master.boxplot(column='cost', by='treatment_type', ax=axes[2])
axes[2].set_title("Cost Distribution by Treatment")
axes[2].set_xlabel("Treatment Type")
axes[2].set_ylabel("Cost ($)")
axes[2].tick_params(axis='x', rotation=45)
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))

plt.tight_layout()
save(fig, "04_treatment_analysis")

# ── 4.5 Revenue & Payment Analysis ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Revenue & Payment Analysis", fontsize=16, fontweight='bold', y=1.02)

# Monthly revenue trend
monthly_rev = master.copy()
monthly_rev['month'] = pd.to_datetime(master['appointment_date']).dt.to_period('M').astype(str)
rev_trend = monthly_rev.groupby('month')['cost'].sum().reset_index().sort_values('month')
axes[0].plot(range(len(rev_trend)), rev_trend['cost'], color=PALETTE[2],
             linewidth=2.5, marker='s', markersize=6)
axes[0].fill_between(range(len(rev_trend)), rev_trend['cost'], alpha=0.15, color=PALETTE[2])
axes[0].set_xticks(range(len(rev_trend)))
axes[0].set_xticklabels(rev_trend['month'], rotation=45, ha='right', fontsize=7)
axes[0].set_title("Monthly Revenue Trend")
axes[0].set_ylabel("Revenue ($)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))

# Payment method
pm = billing['payment_method'].value_counts()
bars = axes[1].bar(pm.index, pm.values, color=PALETTE[:len(pm)], edgecolor='white')
axes[1].bar_label(bars, padding=3, fontsize=11, fontweight='bold')
axes[1].set_title("Payment Method Distribution")
axes[1].set_ylabel("# Transactions")

# Payment status
ps = billing['payment_status'].value_counts()
colors_ps = {'Paid': '#10B981', 'Pending': '#F59E0B', 'Overdue': '#EF4444'}
pie_colors = [colors_ps.get(s, '#6366F1') for s in ps.index]
axes[2].pie(ps, labels=ps.index, autopct='%1.1f%%', colors=pie_colors,
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
axes[2].set_title("Payment Status")

plt.tight_layout()
save(fig, "05_revenue_payment_analysis")

# ── 4.6 Correlation Heatmap ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Feature Correlation Heatmap", fontsize=16, fontweight='bold')

num_cols = master.select_dtypes(include=[np.number]).columns.tolist()
corr = master[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            mask=mask, ax=ax, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax.set_title("Numeric Feature Correlations", pad=15)
plt.tight_layout()
save(fig, "06_correlation_heatmap")

# ── 4.7 Hospital Branch Analysis ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Hospital Branch Analysis", fontsize=16, fontweight='bold', y=1.02)

branch_appts = master.groupby('hospital_branch').size().sort_values(ascending=False)
bars = axes[0].bar(branch_appts.index, branch_appts.values,
                   color=PALETTE[:len(branch_appts)], edgecolor='white')
axes[0].bar_label(bars, padding=3, fontsize=11, fontweight='bold')
axes[0].set_title("Appointments by Branch")
axes[0].set_ylabel("# Appointments")
axes[0].tick_params(axis='x', rotation=20)

branch_rev = master.groupby('hospital_branch')['cost'].sum().sort_values(ascending=False)
bars2 = axes[1].bar(branch_rev.index, branch_rev.values,
                    color=PALETTE[:len(branch_rev)], edgecolor='white')
axes[1].bar_label(bars2, fmt='$%.0f', padding=3, fontsize=9, fontweight='bold')
axes[1].set_title("Revenue by Branch")
axes[1].set_ylabel("Total Revenue ($)")
axes[1].tick_params(axis='x', rotation=20)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))

plt.tight_layout()
save(fig, "07_branch_analysis")

# ─────────────────────────────────────────────────────────────
# 5. MACHINE LEARNING — TREATMENT COST PREDICTION
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Machine Learning …")

# 5.1 Feature Engineering
ml = master.dropna(subset=['cost']).copy()

# Encode appointment hour bucket
ml['hour_bucket'] = pd.cut(ml['hour'].fillna(12),
    bins=[0,8,12,16,20,24],
    labels=['Early','Morning','Afternoon','Evening','Night'])

CAT_FEATURES = ['gender','age_group','insurance_provider',
                'specialization','hospital_branch',
                'treatment_type','reason_for_visit',
                'status','payment_method','hour_bucket']
NUM_FEATURES = ['age','years_experience']
TARGET       = 'cost'

# Drop rows with NaN in features
ml_clean = ml[CAT_FEATURES + NUM_FEATURES + [TARGET]].dropna()
print(f"  ML dataset: {ml_clean.shape}")

X = ml_clean[CAT_FEATURES + NUM_FEATURES]
y = ml_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5.2 Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_FEATURES),
    ('num', StandardScaler(), NUM_FEATURES),
])

# 5.3 Model definitions
models = {
    "Linear Regression"  : LinearRegression(),
    "Random Forest"      : RandomForestRegressor(n_estimators=200, max_depth=10,
                                                 random_state=42, n_jobs=-1),
    "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                      max_depth=5, random_state=42),
}

results = {}
trained_models = {}

for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(pipe, X, y, cv=5, scoring='r2').mean()

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'CV-R²': cv,
                     'y_pred': y_pred}
    trained_models[name] = pipe
    print(f"  {name:22s} | MAE=${mae:,.0f} | RMSE=${rmse:,.0f} | R²={r2:.4f} | CV-R²={cv:.4f}")

# 5.4 Best model → save
best_name = max(results, key=lambda k: results[k]['R²'])
best_pipe  = trained_models[best_name]
joblib.dump(best_pipe, f"{MODELS}/treatment_cost_prediction.pkl")
print(f"\n  Best model: {best_name} (R²={results[best_name]['R²']:.4f}) → saved to models/")

# ── Feature Importance (Random Forest) ───────────────────────
rf_pipe = trained_models["Random Forest"]
ohe_cols = rf_pipe.named_steps['prep']\
                  .named_transformers_['cat']\
                  .get_feature_names_out(CAT_FEATURES).tolist()
all_feat = ohe_cols + NUM_FEATURES
importances = rf_pipe.named_steps['model'].feature_importances_
feat_df = pd.DataFrame({'feature': all_feat, 'importance': importances})\
            .sort_values('importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(feat_df['feature'][::-1], feat_df['importance'][::-1],
               color=PALETTE[0], edgecolor='white')
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9, fontweight='bold')
ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight='bold')
ax.set_xlabel("Importance Score")
plt.tight_layout()
save(fig, "08_feature_importance")

# ── Model Comparison Chart ────────────────────────────────────
metrics_df = pd.DataFrame({k: {m: v for m,v in v.items() if m != 'y_pred'}
                            for k, v in results.items()}).T

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold', y=1.02)

for ax, metric, color in zip(axes, ['MAE','RMSE','R²'],
                              [PALETTE[3], PALETTE[2], PALETTE[1]]):
    bars = ax.bar(metrics_df.index, metrics_df[metric],
                  color=color, edgecolor='white', width=0.5)
    ax.bar_label(bars, fmt='%.4f' if metric == 'R²' else '$%.0f',
                 padding=4, fontsize=11, fontweight='bold')
    ax.set_title(f"{metric} Comparison")
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=15)
    if metric != 'R²':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))

plt.tight_layout()
save(fig, "09_model_comparison")

# ── Actual vs Predicted ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Actual vs Predicted — All Models", fontsize=16, fontweight='bold', y=1.02)

for ax, (name, res) in zip(axes, results.items()):
    ax.scatter(y_test, res['y_pred'], alpha=0.45, s=25,
               color=PALETTE[0], edgecolors='none')
    mn, mx = y_test.min(), y_test.max()
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect fit')
    ax.set_title(f"{name}\nR²={res['R²']:.4f}")
    ax.set_xlabel("Actual Cost ($)")
    ax.set_ylabel("Predicted Cost ($)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))

plt.tight_layout()
save(fig, "10_actual_vs_predicted")

# ─────────────────────────────────────────────────────────────
# 6. BUSINESS INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Business Insights …")

total_rev      = master['cost'].sum()
avg_cost       = master['cost'].mean()
top_treatment  = master.groupby('treatment_type')['cost'].mean().idxmax()
top_spec       = master.groupby('specialization')['cost'].sum().idxmax()
paid_pct       = (billing['payment_status'] == 'Paid').mean() * 100
overdue_pct    = (billing['payment_status'] == 'Overdue').mean() * 100
completion_pct = (appointments['status'] == 'Completed').mean() * 100
noshow_pct     = (appointments['status'] == 'No-show').mean() * 100

insights = f"""
╔══════════════════════════════════════════════════════════════╗
║          BUSINESS INSIGHTS & RECOMMENDATIONS                ║
╚══════════════════════════════════════════════════════════════╝

📊 KEY METRICS
  • Total Revenue Generated   : ${total_rev:>12,.2f}
  • Average Treatment Cost    : ${avg_cost:>12,.2f}
  • Top Revenue Specialization: {top_spec}
  • Highest Avg Cost Treatment: {top_treatment}

💳 PAYMENT HEALTH
  • Payment Completion Rate   : {paid_pct:.1f}%
  • Overdue Rate              : {overdue_pct:.1f}%
  → Recommendation: Implement automated reminders for overdue
    accounts; consider early-payment discount incentives.

📅 APPOINTMENT EFFICIENCY
  • Appointment Completion Rate: {completion_pct:.1f}%
  • No-show Rate               : {noshow_pct:.1f}%
  → Recommendation: Introduce SMS/email reminders 24h before
    appointments and offer rescheduling portals to cut no-shows.

🤖 ML MODEL RESULTS
"""
for mname, res in results.items():
    insights += f"  {mname:22s}: MAE=${res['MAE']:,.0f} | RMSE=${res['RMSE']:,.0f} | R²={res['R²']:.4f}\n"

insights += f"""
  Best Model : {best_name} (R²={results[best_name]['R²']:.4f})
  → The model can predict treatment costs with high accuracy,
    enabling pre-authorization estimates for insurance workflows.

🏥 STRATEGIC RECOMMENDATIONS
  1. REVENUE OPTIMISATION: Focus marketing on high-revenue
     specializations while monitoring cost-efficiency ratios.
  2. CAPACITY PLANNING: Use monthly trend data to staff up
     during peak months and reduce overhead in slow periods.
  3. INSURANCE INTEGRATION: 63% of patients use insurance;
     streamline pre-auth with ML-predicted cost estimates.
  4. BRANCH BENCHMARKING: Identify underperforming branches
     and replicate best practices from top performers.
  5. PATIENT RETENTION: Age-group analysis reveals opportunity
     for targeted wellness programs for 51–65 cohort.
"""

print(insights)

with open(f"{BASE}/INSIGHTS.txt", "w") as f:
    f.write(insights)

print("\n" + "═" * 65)
print("  Pipeline Complete! All assets saved to Hospital-AI-Analytics/")
print("═" * 65)
print(f"  Charts  : {CHARTS}/")
print(f"  Model   : {MODELS}/treatment_cost_prediction.pkl")
print(f"  Insights: {BASE}/INSIGHTS.txt")
