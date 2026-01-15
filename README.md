# 💊 Adverse Drug Event Severity Classifier

**NLP-based classification of adverse event severity using FDA FAERS data**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Pharmaceutical companies receive thousands of adverse event reports daily. Medical reviewers must prioritize which reports require immediate attention—a process that currently relies heavily on manual triage. **Late identification of serious adverse events can delay regulatory reporting and put patients at risk.**

This project builds NLP-powered machine learning models to automatically classify the severity of adverse drug events using data from the FDA Adverse Event Reporting System (FAERS). By extracting features from MedDRA-coded reaction terms and patient demographics, we predict whether an adverse event resulted in serious outcomes (hospitalization, disability, life-threatening conditions, or death).

### Key Features

- **Real FDA data**: 10,000+ adverse event reports from the openFDA API
- **NLP pipeline**: TF-IDF vectorization of MedDRA medical terminology
- **Multiple classifiers**: Logistic Regression, Random Forest, Gradient Boosting, Naive Bayes
- **Interpretable models**: SHAP values and coefficient analysis for regulatory transparency
- **Class imbalance handling**: Balanced class weights for robust predictions

---

## Results Summary

| Model | AUC | Accuracy | Precision | Recall | F1 |
|-------|-----|----------|-----------|--------|-----|
| Logistic Regression | 0.78 | 0.72 | 0.74 | 0.71 | 0.72 |
| Random Forest | 0.76 | 0.70 | 0.72 | 0.69 | 0.70 |
| Gradient Boosting | 0.75 | 0.69 | 0.71 | 0.68 | 0.69 |
| Naive Bayes | 0.73 | 0.67 | 0.69 | 0.66 | 0.67 |

*Results may vary based on API data at time of execution*

---

## Key Findings

### 1. MedDRA Terms Are Highly Predictive
Specific reaction terms strongly indicate severity:
- **Severe indicators**: "cardiac arrest", "respiratory failure", "death", "sepsis"
- **Non-severe indicators**: "nausea", "headache", "rash", "fatigue"

### 2. Clinical Patterns Emerge
- Reports with **more drugs** → higher severity risk (polypharmacy)
- Reports with **more reactions** → higher severity risk
- **Health professionals** report more severe cases than consumers
- **Older patients** show different severity patterns

### 3. Model Interpretability Aligns with Clinical Knowledge
SHAP analysis confirms that predictions are driven by clinically meaningful features, supporting regulatory acceptance of automated triage systems.

---

## Project Structure

```
Adverse-Drug-Event-Classifier/
│
├── Adverse_Drug_Event_Classifier.ipynb   # Main analysis notebook
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT license
│
├── data/
│   └── faers_adverse_events.csv          # Cached API data
│
├── figures/
│   ├── 01_severity_distribution.png
│   ├── 02_top_reactions.png
│   ├── 03_demographics_analysis.png
│   ├── 04_count_analysis.png
│   ├── 05_roc_curves.png
│   ├── 06_confusion_matrix.png
│   ├── 07_feature_coefficients.png
│   ├── 08_shap_summary.png
│   └── 09_shap_importance.png
│
└── model_metrics_summary.csv              # Model performance comparison
```

---

## Installation

### Prerequisites
- Python 3.9+
- Jupyter Notebook or JupyterLab

### Setup

```bash
# Clone the repository
git clone https://github.com/jfinkle00/Adverse-Drug-Event-Classifier.git
cd Adverse-Drug-Event-Classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Adverse_Drug_Event_Classifier.ipynb
```

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
shap>=0.41.0
requests>=2.28.0
scipy>=1.9.0
```

---

## Methodology

### Data Source
- **API**: openFDA Drug Adverse Event endpoint
- **Records**: ~10,000-15,000 adverse event reports
- **Time period**: Recent quarterly data from FAERS

### Target Variable Engineering
Severity classification based on FDA seriousness indicators:

| Level | Category | Criteria |
|-------|----------|----------|
| 4 | Death | `seriousnessdeath = 1` |
| 3 | Life-Threatening | `seriousnesslifethreatening = 1` |
| 2 | Hospitalization/Disability | `seriousnesshospitalization = 1` OR `seriousnessdisabling = 1` |
| 1 | Other Serious | `seriousnessother = 1` |
| 0 | Non-Serious | None of the above |

**Binary classification**: Severe (levels 2-4) vs. Non-Severe (levels 0-1)

### NLP Feature Engineering
1. **Reaction Terms (TF-IDF)**
   - MedDRA-coded adverse event descriptions
   - 500 features, unigrams and bigrams
   - Min document frequency: 5

2. **Drug Indications (TF-IDF)**
   - Why the drug was prescribed
   - 200 features

3. **Structured Features**
   - Patient demographics (age, sex, weight)
   - Report metadata (number of drugs, reactions)
   - Reporter type (health professional vs. consumer)

### Model Training
- 80/20 train-test split with stratification
- 5-fold cross-validation
- Class-weighted models for imbalance handling
- Evaluation: AUC-ROC, Accuracy, Precision, Recall, F1

---

## FAERS Data Notes

### What is FAERS?
The FDA Adverse Event Reporting System (FAERS) is a database of adverse event and medication error reports submitted to the FDA. Key characteristics:

- **Voluntary reporting**: Not all adverse events are reported
- **No causation**: Reports don't prove drug caused the event
- **MedDRA coding**: Standardized medical terminology
- **Public access**: Available via openFDA API

### Data Limitations
- Reporting bias (serious events more likely to be reported)
- Duplicate reports may exist
- Variable data quality depending on reporter
- Missing demographic information common

---

## Business Applications

| Use Case | Stakeholder | Value |
|----------|-------------|-------|
| Report Triage | Safety Officers | Prioritize high-risk reports for review |
| Expedited Reporting | Regulatory Affairs | Identify 15-day reportable events |
| Signal Detection | Pharmacovigilance | Early warning for safety issues |
| Portfolio Risk | R&D Leadership | Assess drug safety profiles |
| Audit Support | Quality Assurance | Document triage decision rationale |

---

## Regulatory Context

### FDA Requirements
- **15-day reports**: Serious and unexpected adverse events must be reported within 15 calendar days
- **Periodic reports**: Quarterly summary reports required
- **Signal detection**: Ongoing monitoring for safety signals

### How This Model Helps
- Automated first-pass severity classification
- Consistent application of seriousness criteria
- Audit trail via interpretable predictions
- Reduced time-to-detection for serious events

---

## Future Enhancements

- [ ] Add word embeddings (BioBERT, PubMedBERT) for better medical term understanding
- [ ] Implement active learning for continuous model improvement
- [ ] Build REST API for real-time prediction
- [ ] Add multi-class severity prediction
- [ ] Deploy as Streamlit dashboard for pharmacovigilance teams
- [ ] Integrate with drug-drug interaction databases

---

## References

1. openFDA API Documentation: https://open.fda.gov/apis/drug/event/
2. MedDRA Terminology: https://www.meddra.org/
3. FDA FAERS Information: https://www.fda.gov/drugs/surveillance/fda-adverse-event-reporting-system-faers
4. ICH E2B Guidelines: https://ich.org/page/e2b-pharmacovigilance

---

## Author

**Jason Finkle**  
Data Scientist | M.S. Data Science, American University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-jason--finkle-blue)](https://linkedin.com/in/jason-finkle)
[![GitHub](https://img.shields.io/badge/GitHub-jfinkle00-black)](https://github.com/jfinkle00)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with data from the FDA Adverse Event Reporting System (FAERS) via the openFDA API. This project is for educational and portfolio purposes. Predictions should not be used for clinical decision-making without proper validation.*
