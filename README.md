# iPersona

> **Virtual Customer Simulation System for Product Launch Intelligence**

iPersona is a machine learning-powered desktop application that simulates how distinct customer personas respond to new product launches — built as a case study around Apple iPhone releases. It combines K-Means clustering, LLM-generated feedback, and rich visualizations to help marketers and product teams make data-driven decisions *before* launch.

---

## What It Does

1. **Clusters customers** from demographic/behavioral data into meaningful segments using K-Means
2. **Generates personas** — each cluster becomes a human-readable customer profile powered by GPT
3. **Simulates product feedback** — personas evaluate a new iPhone's specs (RAM, storage, battery, display, price) and produce match scores + natural language reviews
4. **Visualizes insights** — side-by-side comparisons of personas vs. product features via Matplotlib & Seaborn charts
5. **Enables interactive Q&A** — ask any persona follow-up questions through a built-in chatbot interface

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| UI | PyQt6 (desktop GUI) |
| ML / Clustering | scikit-learn, joblib |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| LLM Integration | OpenAI API (GPT-4.1-mini) |
| Export | openpyxl (Excel reports) |

---

## Project Structure

```
iPersona/
├── main.py                  # Entry point
├── config.py                # API keys, file paths, constants
├── app_gui.py               # PyQt6 desktop interface
├── pipeline.py              # End-to-end ML pipeline orchestration
├── clustering.py            # K-Means clustering logic
├── cluster_analysis.py      # Cluster profiling & persona extraction
├── data_preprocessing.py    # Data cleaning, normalization, outlier removal
├── review_generator.py      # LLM-based persona review generation
├── advanced_analysis.py     # GPT-powered deep-dive analysis
├── visualization.py         # Charts and cluster rating plots
├── report_generator.py      # PDF/Excel report export
├── menu_system.py           # CLI menu (legacy)
├── customer.csv             # Customer dataset (not included — see below)
└── Apple Product details.xlsx  # iPhone product dataset (not included — see below)
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)

**Key packages:**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
openai
```

### Configure API Key

Open `config.py` and set your OpenAI API key:

```python
OPENAI_API_KEY = "your-openai-api-key-here"
```

> **Never commit your API key to version control.** Use environment variables or a `.env` file in production.

### Add Your Data

Place the following files in the project root:
- `customer.csv` — customer demographic and behavioral data
- `Apple Product details.xlsx` — iPhone model specs

### Run the App

```bash
python main.py
```

---

## Pipeline Overview

```
customer.csv  ──►  Preprocessing  ──►  K-Means Clustering
                                              │
                                       Persona Generation (LLM)
                                              │
Apple Product details.xlsx  ──►  Product Loading
                                              │
                               Feedback Simulation & Scoring
                                              │
                          Visualization + Report Generation
```

---

## Output Examples

- **Cluster rating charts** comparing personas across Price, Performance, Features, Battery, and Screen dimensions
- **LLM-generated reviews** per persona for any new product
- **Match score analysis** — which persona is most/least likely to adopt the product
- **Excel export** of full analysis results
- **Interactive chatbot** for custom persona Q&A

---

## Limitations

- Customer dataset includes non-Apple brand users, so some clusters may reflect behaviors of budget-segment competitors (e.g., Google/Android users)
- The system does not model actual real-world product performance — only feature-based simulation
- Review quality depends on the richness of input data and LLM temperature settings
- Best results with a diverse, representative customer dataset

---

## Future Work

- Integrate real beta-tester feedback into the training loop (human-AI hybrid)
- Incorporate sales conversion data to refine cluster-to-purchase modeling
- Extend beyond Apple/iPhone to support any product category
- Add cloud deployment option (web interface via Streamlit or FastAPI)

---

## Team

Built for **CSCI323 — Modern Artificial Intelligence** at the University of Wollongong in Dubai.

| Name |
|---|
| Guzail Rafi |
| Luke Mason |
| Sadhguna Kumar |
| Saadiyah Asif |
| Kavya Vinod |
| Vyshnav Nair |
| Anika Raevskaia |

Supervised by **Dr. Patrick Mukala**

---

## License

This project was developed for academic purposes. All rights reserved by the authors.
