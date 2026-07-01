# 📚 Women's Literacy Prediction — India (NFHS-5)

A machine learning project predicting district-level women's literacy across India, using
socio-economic, health, and infrastructure indicators from the National Family Health Survey
(NFHS-5, 2019–21).

**Live app:** *(add your Streamlit Cloud link here after deploying)*

---

## Why this project matters

India's literacy gap between men and women remains one of the most persistent development
challenges in the country. The 2011 Census put male literacy at ~82% and female literacy at
~65% — a 17-point gap that has narrowed only slowly since. NFHS-5 (2019–21) lets us look at
this problem at the **district level** (706 districts), which is the level at which most
state and central government interventions — Beti Bachao Beti Padhao, Samagra Shiksha, rural
sanitation drives — are actually designed and funded.

This project asks a specific, policy-relevant question:

> **Which socio-economic conditions predict whether a district will have high or low women's
> literacy — and how much does each one matter?**

Rather than just reporting literacy rates (which NFHS already does), this project builds
predictive models to (a) test which indicators carry genuine signal once data leakage is
removed, and (b) quantify how far those indicators alone can explain district-level outcomes.

---

## The economic and policy logic

Three mechanisms link the chosen indicators to women's literacy, drawn from the development
economics literature (Banerjee & Duflo, *Poor Economics*; UNESCO Global Education Monitoring
reports):

**1. Infrastructure removes barriers to schooling.**
Lack of household sanitation is a well-documented reason girls drop out of school after
puberty (no private facilities at school or nearby). Electricity and clean drinking water
free up girls' time — fetching water and fuel is disproportionately a girls'/women's task in
rural India, time that could otherwise go to school or study.

**2. Child marriage truncates education directly.**
Marriage before 18 — still practiced in a meaningful share of Indian districts — ends
schooling almost universally where it occurs. This is one of the most direct, mechanical
links in the dataset, not just a correlation: marriage typically ends a girl's enrollment.

**3. Malnutrition constrains learning capacity.**
Stunting and being underweight in early childhood are associated with cognitive
deficits that persist into adulthood (Victora et al., *The Lancet*, 2008). A malnourished
child entering school is starting from a disadvantage that compounds over years — this is
the "poverty trap" mechanism: poor nutrition → poor learning → low literacy → continued
poverty in the next generation.

**Policy implication:** these results suggest that literacy interventions which only target
"more schools" or "more teachers" may be missing the larger structural drivers. Sanitation,
nutrition, and ending child marriage may be **prerequisites** for literacy gains, not
separate development goals.

---

## Dataset

- **Source:** National Family Health Survey, Round 5 (NFHS-5), 2019–21, Government of India
- **Unit of observation:** District (706 districts, 36 states/UTs)
- **Target variable:** `Women (age 15-49) who are literate (%)`
- **Features used:** 12 socio-economic, infrastructure, and health indicators (see below)

### Why only 12 of 109 available columns?

The raw dataset has 109 NFHS-5 indicators. Two were deliberately **excluded** despite very
high raw correlation with the target:

| Excluded column | Correlation | Why excluded |
|---|---|---|
| Female population who ever attended school | r = 0.93 | Near-identical to the target — this is **data leakage**, not an independent predictor |
| Women with 10+ years of schooling | r = 0.77 | Same issue — too close to "literate" by definition |

Including either would make the model trivially accurate while explaining nothing useful.
The 12 features retained are genuine **independent** socio-economic conditions — they don't
contain the answer baked into the question.

### Features used

| Feature | Category |
|---|---|
| Improved sanitation facility access | Infrastructure |
| Clean cooking fuel access | Infrastructure |
| Household electricity access | Infrastructure |
| Improved drinking water access | Infrastructure |
| Women married before age 18 | Social |
| Population below age 15 | Demographic |
| Institutional births | Health |
| Child stunting (height-for-age) | Health |
| Child underweight (weight-for-age) | Health |
| Health insurance coverage | Health |
| Women with below-normal BMI | Health |
| Births of 3rd+ order | Demographic |

---

## Methodology

**1. Classification (High vs Low literacy)**
Districts are split at the **median literacy rate** into "High" and "Low" literacy classes —
a binary classification task, chosen over predicting the exact percentage because it is more
robust on a 706-row dataset and more directly actionable for policy targeting.

- **Baseline:** majority-class guess → ~50.3% accuracy (classes are near-balanced)
- **Logistic Regression** (standardized features)
- **Random Forest**, both default and **GridSearchCV-tuned** (5-fold CV over
  `n_estimators`, `max_depth`, `min_samples_leaf`)
- Train/test split: 80/20, stratified by class
- Validated with 5-fold cross-validation to confirm results aren't a lucky split

**2. Regression (exact literacy %)**
A Random Forest Regressor separately predicts the continuous literacy percentage, reporting
Mean Absolute Error and R² — useful when a single number is more informative than a
High/Low label.

**3. Geographic visualization**
A live choropleth map of India (state-level, via Plotly + GeoJSON boundaries fetched from
GitHub) shows the spatial pattern of literacy disparity directly.

---

## Results

| Model | Test Accuracy | 5-fold CV |
|---|---|---|
| Baseline (majority guess) | ~50% | — |
| Logistic Regression | ~80% | ~80% |
| Random Forest (default) | ~84% | — |
| **Random Forest (tuned)** | **~84%** | **~83%** |

| Regression metric | Value |
|---|---|
| Mean Absolute Error | ~4.6 percentage points |
| R² | ~0.77 |

**Top predictors (Random Forest feature importance):** women's BMI below normal, share of
population below age 15, sanitation access, child marriage rate, and high birth order —
together pointing to a nutrition–demographic–infrastructure cluster as the strongest signal.

---

## Limitations

This is a **cross-sectional, correlational** analysis — it identifies which indicators move
together with literacy, not proven causal mechanisms. Reverse causality is plausible (e.g.
literate women may also live in households with greater income, and therefore better
sanitation, rather than sanitation causing literacy). Findings should be read as suggestive
for further investigation, not definitive policy conclusions.

A true district-level choropleth would require district-boundary shapefiles; this project
uses state-level boundaries (verified, openly licensed) for the live map, with a full
district-level ranked table as a complement.

---

## Tech stack

- **Python** — pandas, NumPy
- **Scikit-learn** — LogisticRegression, RandomForestClassifier, RandomForestRegressor,
  GridSearchCV, StandardScaler
- **Visualization** — Matplotlib, Seaborn, Plotly (choropleth)
- **App framework** — Streamlit
- **Map data** — [Subhash9325/GeoJson-Data-of-Indian-States](https://github.com/Subhash9325/GeoJson-Data-of-Indian-States) (DataMeet-derived, openly licensed)
- **Explainability** — SHAP (SHapley Additive exPlanations) via `shap` library

---

| Tab | Content |
|---|---|
| 📊 Overview | Distribution, top/bottom states, correlation heatmap |
| 🗺️ District Map | Live India choropleth + state-wise table |
| 🤖 Model Results | Baseline vs LR vs RF (default + tuned), regression, confusion matrix |
| 🔍 District Explorer | Pick any district — see actual literacy, classifier prediction, regression prediction |
| 💡 Key Insights | Policy-facing findings with economic logic |
| 🔬 SHAP Analysis | Global mean SHAP, beeswarm plot, per-district waterfall explanation |

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `datafile.csv` automatically from the same folder — no manual upload needed.

---

## Project structure

```
├── app.py              # Streamlit app — all 5 tabs, model training, visualization
├── datafile.csv         # NFHS-5 district-level dataset (706 districts × 109 indicators)
├── requirements.txt     # Python dependencies
└── README.md             # This file
```

---

*BSc (Hons.) Earth Science, Banaras Hindu University · Intern, IIT Kanpur*
