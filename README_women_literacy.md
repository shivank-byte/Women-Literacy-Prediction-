# 📚 Women's Literacy Across Indian Districts — NFHS-5 Analysis

## Overview
A machine learning dashboard predicting whether an Indian district has high or low women's literacy, built on NFHS-5 (National Family Health Survey, 2019–21) data covering **706 districts** and **109 socio-economic indicators**. The project combines exploratory data analysis, statistical correlation, and classification modelling to identify which infrastructure and health factors most strongly predict women's literacy outcomes.

## Objectives
- Classify districts as high or low women's literacy based on socio-economic indicators
- Compare Logistic Regression and Random Forest for classification performance
- Identify the strongest predictors of women's literacy using feature importance
- Surface policy-relevant insights linking sanitation, nutrition, child marriage, and education

## Dataset
- **Source:** NFHS-5 (2019–21)
- **706 districts** across India
- **109 socio-economic indicators**, narrowed to 8 key features for modelling:
  Clean Fuel Access, Sanitation, Electricity, Child Marriage Rate, Institutional Births, Child Stunting, Health Insurance Coverage, Low BMI (malnutrition proxy)

## Methodology
- Data cleaning & numeric parsing across all indicator columns
- Median-split classification target (High vs. Low literacy)
- Feature scaling with StandardScaler
- **Logistic Regression** and **Random Forest Classifier** trained and compared
- Feature importance ranking, confusion matrix, and classification report
- Correlation heatmap across all socio-economic indicators

## Key Findings
- **Random Forest outperforms Logistic Regression** (82.4% vs. 72.5% accuracy) — literacy is driven by non-linear interactions between factors, not a simple linear relationship.
- **Malnutrition (Low BMI) is the single strongest ML predictor** (importance ≈ 0.18) — pointing to a poverty trap linking nutrition and education.
- **Sanitation access** shows the strongest infrastructure correlation with literacy (r ≈ 0.67).
- **Child marriage** is strongly negatively correlated with literacy (r ≈ −0.61) — early marriage cuts schooling short for millions of girls.
- **Health insurance coverage** shows almost no correlation with literacy (r ≈ 0.01), suggesting insurance access alone doesn't move educational outcomes without addressing nutrition and child marriage.

## Policy Implication
Improving nutrition and reducing child marriage may have a stronger effect on women's literacy than expanding health insurance coverage. Policy should target the **sanitation–nutrition–education nexus** at the district level.

## App Features
The Streamlit dashboard is organized into four tabs:
1. **📊 Overview** — dataset summary, literacy distribution, top states by literacy, correlation heatmap
2. **🤖 Model Results** — accuracy comparison, feature importance, confusion matrix, classification report
3. **🔍 District Explorer** — select any state/district to view its indicators and get a live model prediction
4. **💡 Key Insights** — narrative summary of findings and policy implications

## Tech Stack
```
Language        Python
Data            Pandas, NumPy
ML              Scikit-learn (Logistic Regression, Random Forest)
Visualization   Matplotlib, Seaborn
App Framework   Streamlit
```

## Skills Demonstrated
Statistical & ML Modelling · Feature Engineering · Feature Importance Analysis · Data Cleaning · Exploratory Data Analysis · Dashboard Development · Data Storytelling

## Run Locally
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
streamlit run app.py
```
Upload the NFHS-5 `datafile.csv` in the sidebar to load the dashboard.

## Academic Context
Built by **Shivank Thakur** — B.Sc. (Hons.) Statistics, Banaras Hindu University → M.Sc. Economics, IIT Kanpur.
