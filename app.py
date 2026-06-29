import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Women's Literacy in India — NFHS-5",
    page_icon="📚",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Serif', serif;
    }
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .hero h1 {
        color: #e2e8f0;
        font-size: 2rem;
        margin: 0 0 0.4rem 0;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0f3460;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin: 0 0 0.3rem 0;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 600;
        color: #0f3460;
        font-family: 'IBM Plex Serif', serif;
    }
    .insight-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    .insight-box p { margin: 0; color: #0c4a6e; font-size: 0.92rem; }
    .section-label {
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Data loading & model ──────────────────────────────────────────────────────
FEATURES = [
    'Households using clean fuel for cooking3 (%)',
    'Population living in households that use an improved sanitation facility2 (%)',
    'Population living in households with electricity (%)',
    'Women age 20-24 years married before age 18 years (%)',
    'Institutional births (in the 5 years before the survey) (%)',
    'Children under 5 years who are stunted (height-for-age)18 (%)',
    'Households with any usual member covered under a health insurance/financing scheme (%)',
    'Women (age 15-49 years) whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2)21 (%)'
]
SHORT_NAMES = [
    'Clean Fuel', 'Sanitation', 'Electricity', 'Child Marriage',
    'Institutional Births', 'Child Stunting', 'Health Insurance', 'Low BMI'
]
TARGET_COL = 'Women (age 15-49) who are literate4 (%)'

@st.cache_data
def load_and_train(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['District Names'] = df['District Names'].astype(str).str.strip()
    df['State/UT'] = df['State/UT'].astype(str).str.strip()

    # Clean all numeric columns
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(r'[\(\)]', '', regex=True),
            errors='coerce'
        )

    df['high_literacy'] = (df[TARGET_COL] >= df[TARGET_COL].median()).astype(int)

    X = df[FEATURES].copy().fillna(df[FEATURES].apply(
        lambda c: pd.to_numeric(c.astype(str).str.strip(), errors='coerce')).median())
    y = df['high_literacy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_tr_sc, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_te_sc))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_report = classification_report(y_test, rf.predict(X_test),
                                      target_names=['Low Literacy', 'High Literacy'],
                                      output_dict=True)
    cm = confusion_matrix(y_test, rf.predict(X_test))
    importances = pd.Series(rf.feature_importances_, index=SHORT_NAMES).sort_values(ascending=False)

    return df, X, y, lr_acc, rf_acc, rf_report, cm, importances, scaler, rf

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Data")
    uploaded = st.file_uploader("Upload your NFHS-5 CSV file", type="csv")
    st.markdown("---")
    st.markdown("**About this project**")
    st.markdown(
        "Predicting district-level women's literacy using NFHS-5 socio-economic indicators. "
        "Built with Random Forest & Logistic Regression.\n\n"
        "**Data:** NFHS-5 (2019–21) | 706 districts"
    )
    st.markdown("---")
    st.markdown("*BSc Stats, BHU → MSc Eco, IIT Kanpur*")

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📚 Women's Literacy Across Indian Districts</h1>
  <p>A machine learning analysis of NFHS-5 data — 706 districts · 109 socio-economic indicators · 2019–21</p>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.info("👈 Upload your **datafile.csv** in the sidebar to load the dashboard.")
    st.stop()

df, X, y, lr_acc, rf_acc, rf_report, cm, importances, scaler, rf = load_and_train(uploaded)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 Model Results", "🔍 District Explorer", "💡 Key Insights"])

# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-label">Dataset Summary</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>Districts</h3><div class="value">{df["District Names"].nunique()}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>States / UTs</h3><div class="value">{df["State/UT"].nunique()}</div></div>', unsafe_allow_html=True)
    with c3:
        median_lit = round(df[TARGET_COL].median(), 1)
        st.markdown(f'<div class="metric-card"><h3>Median Literacy (%)</h3><div class="value">{median_lit}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>Indicators</h3><div class="value">109</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Distribution of Women's Literacy Rate**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df[TARGET_COL].dropna(), bins=30, color='#0f3460', edgecolor='white', alpha=0.85)
        ax.axvline(df[TARGET_COL].median(), color='#e94560', linestyle='--', linewidth=1.5, label=f'Median: {median_lit}%')
        ax.set_xlabel("Literacy Rate (%)", fontsize=10)
        ax.set_ylabel("Number of Districts", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("**Top 10 States by Average Women's Literacy**")
        state_lit = df.groupby('State/UT')[TARGET_COL].mean().sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.barh(state_lit.index, state_lit.values, color='#0f3460', alpha=0.85)
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
        ax.set_xlabel("Avg Literacy Rate (%)", fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Correlation Heatmap — Socio-economic Indicators vs Literacy**")
    corr_df = X.copy()
    corr_df.columns = SHORT_NAMES
    corr_df['Literacy'] = df[TARGET_COL]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 9})
    ax.set_title("Correlation between socio-economic indicators and women's literacy", fontsize=12, pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 2: Model Results ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>Logistic Regression</h3><div class="value">{lr_acc:.1%}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>Random Forest</h3><div class="value">{rf_acc:.1%}</div></div>', unsafe_allow_html=True)
    with c3:
        f1 = rf_report['weighted avg']['f1-score']
        st.markdown(f'<div class="metric-card"><h3>RF Weighted F1</h3><div class="value">{f1:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Feature Importance (Random Forest)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#e94560' if i == 0 else '#0f3460' for i in range(len(importances))]
        importances.sort_values().plot(kind='barh', ax=ax, color=colors[::-1], edgecolor='white')
        ax.axvline(importances.mean(), color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Average')
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("**Confusion Matrix (Random Forest)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low', 'High'],
                    yticklabels=['Low', 'High'],
                    ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title("Literacy Classification", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Detailed Classification Report**")
    report_df = pd.DataFrame(rf_report).T.round(3)
    st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
                 use_container_width=True)

# ── TAB 3: District Explorer ──────────────────────────────────────────────────
with tab3:
    st.markdown("**Explore any district's data and get a literacy prediction**")
    states = sorted(df['State/UT'].dropna().unique())
    selected_state = st.selectbox("Select State / UT", states)
    districts = sorted(df[df['State/UT'] == selected_state]['District Names'].unique())
    selected_district = st.selectbox("Select District", districts)

    row = df[(df['State/UT'] == selected_state) & (df['District Names'] == selected_district)]

    if not row.empty:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown(f"**{selected_district}, {selected_state}**")
            literacy_val = row[TARGET_COL].values[0]
            actual_class = "🟢 High Literacy" if literacy_val >= df[TARGET_COL].median() else "🔴 Low Literacy"
            st.markdown(f'<div class="metric-card"><h3>Women\'s Literacy Rate</h3><div class="value">{literacy_val:.1f}%</div></div>', unsafe_allow_html=True)
            st.markdown(f"**Actual class:** {actual_class}")

            # Predict
            x_row = row[FEATURES].copy()
            for col in FEATURES:
                x_row[col] = pd.to_numeric(x_row[col].astype(str).str.strip(), errors='coerce')
            x_row = x_row.fillna(X.median())
            pred = rf.predict(x_row)[0]
            prob = rf.predict_proba(x_row)[0]
            pred_label = "🟢 High Literacy" if pred == 1 else "🔴 Low Literacy"
            st.markdown(f"**Model prediction:** {pred_label} (confidence: {max(prob):.0%})")

        with col_right:
            st.markdown("**Key Indicators**")
            indicator_data = {}
            for feat, short in zip(FEATURES, SHORT_NAMES):
                val = row[feat].values[0]
                indicator_data[short] = f"{val:.1f}%" if pd.notna(val) else "N/A"
            ind_df = pd.DataFrame.from_dict(indicator_data, orient='index', columns=['Value'])
            st.dataframe(ind_df, use_container_width=True)

# ── TAB 4: Key Insights ───────────────────────────────────────────────────────
with tab4:
    st.markdown("### What the data tells us about women's literacy in India")

    insights = [
        ("🚽 Sanitation is the strongest infrastructure predictor",
         "Districts with higher sanitation access show a 0.67 correlation with women's literacy — "
         "the highest among all infrastructure indicators. Lack of toilets is a well-documented reason "
         "girls drop out of school in rural India."),
        ("💍 Child marriage collapses literacy (r = −0.61)",
         "Where girls marry before 18, literacy rates fall sharply. This is the second strongest predictor "
         "in the Random Forest model, confirming that early marriage ends schooling for millions of girls."),
        ("🍽️ Malnutrition is the #1 ML predictor (importance = 0.18)",
         "Low BMI — a proxy for chronic malnutrition — emerged as the single most important feature. "
         "Hungry children can't learn. This points to a deep poverty trap linking nutrition and education."),
        ("🏥 Health insurance schemes barely matter (r = 0.01)",
         "Coverage under government health schemes shows near-zero correlation with literacy. "
         "This suggests that insurance alone, without addressing nutrition and child marriage, "
         "is insufficient to drive educational outcomes."),
        ("🌳 Random Forest beats Logistic Regression (82.4% vs 72.5%)",
         "The non-linear interactions between indicators — captured by Random Forest — explain "
         "literacy better than a simple linear model. Poverty is multidimensional and so is its solution.")
    ]

    for title, body in insights:
        st.markdown(f"**{title}**")
        st.markdown(f'<div class="insight-box"><p>{body}</p></div>', unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("### Policy Implication")
    st.info(
        "Improving nutrition and ending child marriage may have stronger effects on women's literacy "
        "than expanding health insurance coverage alone. Development policy should target the "
        "**sanitation–nutrition–education nexus** at the district level."
    )
    st.markdown("*Data: NFHS-5 (2019–21) | Model: Random Forest Classifier | Built with Python + Streamlit*")
