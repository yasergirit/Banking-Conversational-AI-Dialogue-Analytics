# Banking Conversational AI & Dialogue Analytics

**A portfolio case study in banking intent classification, virtual assistant optimization, and customer dialogue analytics.**

This project explores two related questions:

1. **Can we recognize a customer's banking intent from a short written request?**
2. **How can analysts measure where a virtual assistant's customer journey succeeds or breaks down?**

It combines an NLP classification workflow on the public **BANKING77** dataset with a separate SQL analytics case study using **synthetic** conversation and banking-action events.

## Highlights

| Metric | Result |
|---|---:|
| Banking intents | 77 |
| Hold-out test utterances | 3,073 |
| **Test accuracy** | **89.36%** |
| **Test Macro F1** | **89.36%** |
| Coverage at a 0.40 confidence threshold | 72.05% |
| Accuracy among accepted predictions | 96.43% |

**Evaluation details:** The reported test metrics use a word + character TF-IDF / Logistic Regression classifier. Seven normalized train–test text overlaps were excluded from the original 3,080-row test split. The 0.40 threshold is an *offline routing simulation*, not a production-ready fallback policy; selective accuracy is **not** a customer issue-resolution rate.

## What I Built

- **Intent classification:** Compared word-level TF-IDF, class-weighted Logistic Regression, character n-grams, and combined word + character features.
- **Model diagnostics:** Investigated per-intent Precision, Recall, F1, confusion patterns, low-frequency classes, and ambiguous labels.
- **Confidence and fallback analysis:** Studied coverage vs. selective accuracy, high-confidence errors, and sigmoid probability calibration.
- **Dialogue analytics with SQL:** Designed a three-table DuckDB model for customer sessions, dialogue turns, and action events; calculated fallback, self-service resolution, and action success KPIs; explored handoffs and journey friction using synthetic records.

## Notebooks

| Notebook | Focus |
|---|---|
| [01 — Data Understanding & EDA](notebooks/01_data_understanding.ipynb) | Data quality, duplicates, class distribution, message length, and train–test overlap |
| [02 — Intent Classification](notebooks/02_baseline_classification.ipynb) | Baseline and optimization experiments, error analysis, confidence, calibration, final hold-out evaluation |
| [03 — Dialogue Analytics & SQL](notebooks/03_customer_dialogue_analytics.ipynb) | DuckDB schema, synthetic conversations, KPI definitions, data quality checks, customer journey analytics |

## Tech Stack

**Python · pandas · NumPy · scikit-learn · TF-IDF · Logistic Regression · SQL · DuckDB · Matplotlib · Seaborn · Jupyter**

## Run Locally

**1. Clone the repository and install the dependencies:**

```bash
git clone https://github.com/yasergirit/Banking-Conversational-AI-Dialogue-Analytics.git
cd Banking-Conversational-AI-Dialogue-Analytics
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

**2. Download the BANKING77 CSV files** (these are not committed to this repository):

```bash
mkdir -p data/raw
curl -L https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv -o data/raw/train.csv
curl -L https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv -o data/raw/test.csv
```

**3. Run the notebooks in order:**

```bash
python -m jupyter lab
```

Open the `notebooks/` folder and run **01 → 02 → 03**. Notebook 02 trains its models from the CSV files; Notebook 03 generates its own synthetic dataset and uses an isolated DuckDB demo database under `data/warehouse/`.

## Data & Scope

- **Classification data:** [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) — labeled standalone banking customer utterances, **not** full dialogue sessions.
- **Operational analytics data:** Fully **synthetic** conversations, model-like confidence values, handoffs, outcomes, and action events for demonstrating SQL-based KPI reporting.
- **Important distinction:** Notebook 03 does **not** run Notebook 02's trained classifier on its generated conversations; the two notebooks demonstrate complementary analytics capabilities rather than a deployed end-to-end banking assistant.
- **Limitations:** Some labels are ambiguous, some intents have few evaluation examples, and repeated experiments on the same validation split can bias model selection. The results do not establish production readiness, regulatory suitability, or actual customer-resolution performance.

## Author

**Yaser Girit** · [GitHub](https://github.com/yasergirit)

*Independent learning and portfolio project. Not affiliated with or endorsed by ING or any bank.*
