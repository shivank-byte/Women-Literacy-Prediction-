import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                              mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler
import shap
import os

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
    .method-note {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
        font-size: 0.85rem;
        color: #78350f;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Features ──────────────────────────────────────────────────────────────────
# NOTE: "Female population who ever attended school" (r=0.93) and "10+ years
# schooling" (r=0.77) are excluded deliberately — they are near-duplicates of
# the target (data leakage), not independent predictors. The 12 features below
# are genuine socio-economic / infrastructure / health indicators.
FEATURES = [
    'Population living in households that use an improved sanitation facility2 (%)',
    'Households using clean fuel for cooking3 (%)',
    'Population living in households with electricity (%)',
    'Population living in households with an improved drinking-water source1 (%)',
    'Women age 20-24 years married before age 18 years (%)',
    'Population below age 15 years (%)',
    'Institutional births (in the 5 years before the survey) (%)',
    'Children under 5 years who are stunted (height-for-age)18 (%)',
    'Children under 5 years who are underweight (weight-for-age)18 (%)',
    'Households with any usual member covered under a health insurance/financing scheme (%)',
    'Women (age 15-49 years) whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2)21 (%)',
    'Births in the 5 years preceding the survey that are third or higher order (%)',
]
SHORT_NAMES = [
    'Sanitation', 'Clean Fuel', 'Electricity', 'Drinking Water',
    'Child Marriage', 'Pop. Below 15', 'Institutional Births',
    'Child Stunting', 'Child Underweight', 'Health Insurance',
    'Low BMI (Women)', 'High Birth Order'
]
TARGET_COL = 'Women (age 15-49) who are literate4 (%)'
DATA_PATH = os.path.join(os.path.dirname(__file__), "datafile.csv")

# ── Data loading & model ──────────────────────────────────────────────────────
@st.cache_data
def load_and_train(path):
    df = pd.read_csv(path)
    df['District Names'] = df['District Names'].astype(str).str.strip()
    df['State/UT'] = df['State/UT'].astype(str).str.strip()

    # Clean all numeric columns — RBI/NFHS style data has (xx.x), *, brackets
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(r'[\(\)\*]', '', regex=True),
            errors='coerce'
        )

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    threshold = df[TARGET_COL].median()
    df['high_literacy'] = (df[TARGET_COL] >= threshold).astype(int)

    X = df[FEATURES].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df['high_literacy']
    y_reg = df[TARGET_COL]

    # ── Baseline: majority-class guess ──
    majority_class = y.mode()[0]
    baseline_acc = (y == majority_class).mean()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_tr_sc, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_te_sc))
    lr_cv = cross_val_score(lr, scaler.fit_transform(X), y, cv=5).mean()

    # ── Random Forest: default vs tuned (GridSearchCV) ──
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default.fit(X_train, y_train)
    rf_default_acc = accuracy_score(y_test, rf_default.predict(X_test))

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 8, 12],
        'min_samples_leaf': [1, 3],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    rf = grid.best_estimator_
    best_params = grid.best_params_

    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_cv = cross_val_score(rf, X, y, cv=5).mean()
    rf_report = classification_report(y_test, rf.predict(X_test),
                                      target_names=['Low Literacy', 'High Literacy'],
                                      output_dict=True)
    cm = confusion_matrix(y_test, rf.predict(X_test))
    importances = pd.Series(rf.feature_importances_, index=SHORT_NAMES).sort_values(ascending=False)

    # ── Regression model: predict actual literacy % ──
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    reg.fit(Xr_train, yr_train)
    reg_pred = reg.predict(Xr_test)
    reg_mae = mean_absolute_error(yr_test, reg_pred)
    reg_r2 = r2_score(yr_test, reg_pred)

    # ── SHAP explainer ──
    explainer   = shap.TreeExplainer(rf)
    shap_output = explainer.shap_values(X)

    # Handle different SHAP output formats across versions:
    # Older SHAP: list of 2 arrays [class0, class1], each shape (n, features)
    # Newer SHAP: single 3D array shape (n, features, 2)
    if isinstance(shap_output, list):
        sv = shap_output[1]                  # class 1 (High Literacy)
    elif hasattr(shap_output, 'ndim') and shap_output.ndim == 3:
        sv = shap_output[:, :, 1]           # last axis = class index
    else:
        sv = shap_output                     # already 2D

    shap_df = pd.DataFrame(sv, columns=SHORT_NAMES)

    return (df, X, y, threshold, baseline_acc, lr_acc, lr_cv,
            rf_default_acc, rf_acc, rf_cv, best_params, rf_report, cm,
            importances, scaler, rf, reg, len(X_train), len(X_test),
            reg_mae, reg_r2, Xr_test, yr_test, reg_pred,
            explainer, shap_df)

# ── Load data automatically — no upload required ──────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error(
        "⚠️ `datafile.csv` not found in the app folder. "
        "Make sure it's uploaded to the same GitHub repository as `app.py`."
    )
    st.stop()

(df, X, y, THRESHOLD, baseline_acc, lr_acc, lr_cv,
 rf_default_acc, rf_acc, rf_cv, best_params, rf_report, cm,
 importances, scaler, rf, reg, n_train, n_test,
 reg_mae, reg_r2, Xr_test, yr_test, reg_pred,
 explainer, shap_df) = load_and_train(DATA_PATH)

median_lit = round(THRESHOLD, 1)

# ── GeoJSON for state choropleth ────────────────────────────────────────────
GEOJSON_URL = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"

# Map dataset state names -> geojson State_Name values
STATE_NAME_MAP = {
    "Andaman & Nicobar Islands": "Andaman & Nicobar Island",
    "Jammu & Kashmir": "Jammu & Kashmir",
    "Maharastra": "Maharashtra",
    "NCT of Delhi": "NCT of Delhi",
    "Dadra and Nagar Haveli & Daman and Diu": "Dadara & Nagar Havelli",
    "Telangana": "Telangana",
    "Odisha": "Odisha",
}

@st.cache_data
def load_geojson(url):
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None

geojson_data = load_geojson(GEOJSON_URL)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 About this project")
    st.markdown(
        "Predicting district-level women's literacy using NFHS-5 socio-economic, "
        "health, and infrastructure indicators.\n\n"
        f"**Data:** NFHS-5 (2019–21) · **{df.shape[0]} districts** · {df['State/UT'].nunique()} states/UTs\n\n"
        f"**Classification:** High vs Low literacy (split at median = {median_lit}%)\n\n"
        f"**Baseline (majority guess):** {baseline_acc:.1%}\n\n"
        f"**Random Forest (tuned):** {rf_acc:.1%} — beats baseline by {(rf_acc-baseline_acc)*100:.0f} pts\n\n"
        f"**Regression (literacy %):** MAE = {reg_mae:.2f} pts · R² = {reg_r2:.2f}\n\n"
        f"**Train/test split:** {n_train} / {n_test} districts (80/20)"
    )
    st.markdown("---")
    st.markdown("*BSc Earth Science, BHU → Intern, IIT Kanpur*")

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>📚 Women's Literacy Across Indian Districts</h1>
  <p>A machine learning analysis of NFHS-5 data — {df.shape[0]} districts · 12 socio-economic indicators · 2019–21</p>
</div>
""", unsafe_allow_html=True)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Overview", "🗺️ District Map", "🤖 Model Results",
     "🔍 District Explorer", "💡 Key Insights", "🔬 SHAP Analysis"]
)

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
        spread = round(df[TARGET_COL].max() - df[TARGET_COL].min(), 1)
        st.markdown(f'<div class="metric-card"><h3>Range (Max−Min)</h3><div class="value">{spread} pp</div></div>', unsafe_allow_html=True)

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

    st.markdown("**Bottom 10 States by Average Women's Literacy**")
    state_lit_low = df.groupby('State/UT')[TARGET_COL].mean().sort_values(ascending=True).head(10)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    bars = ax.barh(state_lit_low.index, state_lit_low.values, color='#e94560', alpha=0.85)
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
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation between socio-economic indicators and women's literacy", fontsize=12, pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 2: District Map ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-label">Geographic Distribution</p>', unsafe_allow_html=True)
    st.markdown("**State-wise average women's literacy rate**")

    state_avg = df.groupby('State/UT')[TARGET_COL].agg(['mean', 'count']).reset_index()
    state_avg.columns = ['State/UT', 'Avg Literacy (%)', 'Districts']
    state_avg = state_avg.sort_values('Avg Literacy (%)', ascending=False)
    state_avg['Avg Literacy (%)'] = state_avg['Avg Literacy (%)'].round(1)
    state_avg['geo_name'] = state_avg['State/UT'].replace(STATE_NAME_MAP)

    if geojson_data is not None:
        import plotly.graph_objects as go

        fig_map = go.Figure(go.Choropleth(
            geojson=geojson_data,
            locations=state_avg['geo_name'],
            z=state_avg['Avg Literacy (%)'],
            featureidkey="properties.State_Name",
            colorscale="RdYlGn",
            marker_line_color="white",
            marker_line_width=0.6,
            colorbar_title="Literacy (%)",
            hovertext=state_avg['State/UT'] + "<br>" + state_avg['Avg Literacy (%)'].astype(str) + "%",
            hoverinfo="text",
        ))
        fig_map.update_geos(
            visible=False, resolution=50,
            lonaxis_range=[68, 98], lataxis_range=[6, 38],
            projection_type="mercator"
        )
        fig_map.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown(
            '<div class="method-note">🗺️ Choropleth built from DataMeet-sourced state boundaries '
            '(<a href="https://github.com/Subhash9325/GeoJson-Data-of-Indian-States" target="_blank">'
            'Subhash9325/GeoJson-Data-of-Indian-States</a>), fetched live via GitHub raw URL. '
            'A few small UTs (Lakshadweep, Dadra & Nagar Haveli) may render faintly due to their size '
            'at this map scale.</div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("Map data could not be fetched (no internet access from this environment) — showing ranked bar chart instead.")
        fig, ax = plt.subplots(figsize=(11, 10))
        colors = plt.cm.RdYlGn((state_avg['Avg Literacy (%)'] - state_avg['Avg Literacy (%)'].min()) /
                                (state_avg['Avg Literacy (%)'].max() - state_avg['Avg Literacy (%)'].min()))
        bars = ax.barh(state_avg['State/UT'], state_avg['Avg Literacy (%)'], color=colors)
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=7)
        ax.set_xlabel("Average Women's Literacy Rate (%)", fontsize=10)
        ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Full state-wise table**")
    st.dataframe(
        state_avg[['State/UT', 'Avg Literacy (%)', 'Districts']]
            .style.background_gradient(cmap='RdYlGn', subset=['Avg Literacy (%)']),
        use_container_width=True, hide_index=True
    )

# ── TAB 3: Model Results ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="method-note">📐 <b>Methodology &amp; threshold:</b> Districts are split into '
        f'<b>High Literacy</b> (≥ {median_lit}%, the dataset median) and <b>Low Literacy</b> (below {median_lit}%) — '
        f'a binary classification task, chosen over predicting the exact percentage because it is more robust '
        f'on a {df.shape[0]}-row dataset and easier to act on for policy. Train/test split: {n_train} train / '
        f'{n_test} test districts (80/20, stratified).</div>',
        unsafe_allow_html=True
    )

    st.markdown("**Classification — beating the baseline**")
    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.markdown(f'<div class="metric-card"><h3>Baseline (majority guess)</h3><div class="value">{baseline_acc:.1%}</div></div>', unsafe_allow_html=True)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>Logistic Regression</h3><div class="value">{lr_acc:.1%}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>Random Forest (default)</h3><div class="value">{rf_default_acc:.1%}</div></div>', unsafe_allow_html=True)
    with c3:
        delta = (rf_acc - baseline_acc) * 100
        st.markdown(f'<div class="metric-card"><h3>Random Forest (tuned)</h3><div class="value">{rf_acc:.1%}</div></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="insight-box"><p>📊 Because the High/Low split is near-balanced '
        f'({y.value_counts()[1]} High vs {y.value_counts()[0]} Low districts), guessing the majority '
        f'class every time only gets <b>{baseline_acc:.1%}</b> accuracy. The tuned Random Forest reaches '
        f'<b>{rf_acc:.1%}</b> — <b>{delta:.0f} percentage points above baseline</b>, and {rf_cv:.1%} under '
        f'5-fold cross-validation, confirming the indicators genuinely carry predictive signal rather than '
        f'the model exploiting a lucky train/test split.</p></div>',
        unsafe_allow_html=True)
