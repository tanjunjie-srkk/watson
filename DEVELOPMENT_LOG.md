# Watson Document Intelligence Platform — Development Log

> **Last updated:** 5 March 2026
> **Author:** Tan Jun Jie
> **Status:** Active development (pausing for another project)

---

## 1. What Is This Project?

**Watson** is an AI-powered document processing platform built for **Watson's Personal Care Stores SDN BHD** financial document workflows. It processes PDFs through a multi-stage pipeline:

```
PDF → OCR → Classification → Extraction → Structured JSON
```

The platform also includes a **bank statement reconciliation** module (Shopee seller settlement matching) as a separate workstream.

### Business Context

Watson's Personal Care Stores receives hundreds of financial documents from suppliers, landlords, utilities, and banks. This platform automates the extraction of structured data from those documents so it can feed into downstream accounting/ERP systems, replacing manual data entry.

---

## 2. Architecture Overview

### 2.1 Document Processing Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                  Streamlit UI  (app.py, ~2,600+ lines)       │
│   5 pages: Processing │ OCR Viewer │ Extraction │ Report │ Bank │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│                     OCR Layer (3 engines)                     │
│  ┌────────────────┐ ┌─────────────────┐ ┌──────────────────┐ │
│  │ Vision OCR     │ │ PyMuPDF         │ │ Azure Doc Intel  │ │
│  │ (ocr_agent.py) │ │ (extraction_    │ │ (extraction_     │ │
│  │ GPT-5.2 +     │ │  library.py)    │ │  doc_intel.py)   │ │
│  │ image input    │ │ Local, free     │ │ prebuilt-layout  │ │
│  └────────┬───────┘ └───────┬─────────┘ └────────┬─────────┘ │
│           └────────┬────────┴─────────────────────┘           │
│              Unified OCR JSON Schema                          │
└──────────────┬───────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Orchestrator (orchestrator.py)               │
│   classifier.py → AGENT_REGISTRY → extraction_*.py agents    │
└──────────────┬───────────────────────────────────────────────┘
               ▼
         Structured JSON Output (per document type)
```

### 2.2 Reconciliation Module (separate workstream)

```
src/reconciliation/
├── reconciliation_demo.py          # Shopee seller rule-based reconciliation
├── reconciliation_dashboard.py     # Streamlit dashboard for demo
├── fuzzy_matching_example.py       # Educational fuzzy matching demo
├── matchrule.py                    # (empty — placeholder)
└── ml/                             # ML-based reconciliation pipeline
    ├── run_pipeline.py             # End-to-end orchestrator
    ├── synthetic_dataset_generator.py  # Generates realistic Shopee data
    ├── data_ingestion.py           # Load & normalize CSVs
    ├── candidate_generation.py     # Blocking (1-to-1 + many-to-one)
    ├── feature_engineering.py      # Amount/date/text features
    ├── model_training.py           # LR / XGBoost / LightGBM
    ├── prediction.py               # Score + threshold logic
    ├── compare_ml_vs_rules.py      # ML vs rule-based comparison
    ├── pipeline_visualizer.py      # Streamlit step-by-step walkthrough
    ├── feedback.py                 # Human-in-the-loop feedback loop
    └── bank_reconciliaiton.py      # Legacy/initial sklearn prototype
```

---

## 3. Technology Stack

| Layer | Technology | Details |
|---|---|---|
| **LLM** | Azure OpenAI GPT-5.2 | `gpt-5.2-chat` deployment, API version `2024-12-01-preview` |
| **OCR (Vision)** | Azure OpenAI GPT-5.2 w/ image input | Base64-encoded page images → structured JSON |
| **OCR (Local)** | PyMuPDF (fitz) | Free, offline, digital PDFs only |
| **OCR (Cloud)** | Azure Document Intelligence | `prebuilt-layout` model + KEY_VALUE_PAIRS feature |
| **UI** | Streamlit | Custom Watsons-branded theme (teal `#006770`, `#00A0AF`) |
| **PDF Processing** | PyMuPDF (fitz) | PDF → PNG at 300 DPI |
| **ML Reconciliation** | XGBoost, LightGBM, Logistic Regression | Trained on synthetic Shopee settlement data |
| **Fuzzy Matching** | rapidfuzz | Token/partial ratio for text similarity |
| **Data** | pandas | All tabular processing |
| **Visualization** | Plotly | Charts in reconciliation dashboards |
| **Python** | 3.12+ (uses `str | None` union syntax) | |

### Key Dependencies (requirements.txt)

```
streamlit
pandas
openai
pymupdf
```

Additional (installed separately for specific pipelines):
- `azure-ai-documentintelligence`, `azure-core` — for Doc Intel pipeline
- `rapidfuzz` — for reconciliation fuzzy matching
- `xgboost`, `lightgbm`, `scikit-learn` — for ML reconciliation
- `plotly` — for reconciliation dashboards
- `Pillow` — for image preprocessing in Vision OCR
- `azure-ai-ml`, `azure-identity` — for token monitoring (Azure ML)

---

## 4. Supported Document Types (8 types)

| # | Type | Classifier Label | Extraction Agent | Key Fields |
|---|---|---|---|---|
| 1 | Commercial Invoice | `commercial_invoice` | `extraction_invoice.py` | vendor, invoice #, PO #, barcodes, line items, freight, grand total |
| 2 | Credit Note | `credit_note` | `extraction_invoice.py` (shared) | Same as invoice (reuses schema) |
| 3 | Travel Document | `travel` | `extraction_travel.py` | passengers, flights, ticket #, PNR, routing |
| 4 | Rental/Lease Invoice | `rental` | `extraction_rental.py` | property, unit #, tenancy period, base rent, service charges |
| 5 | Hotel Folio | `hotel` | `extraction_hotel.py` | guest, check-in/out, room #, room type, nights |
| 6 | Utility/Telecom Bill | `utility` | `extraction_utility.py` | account #, billing period, meter readings, surcharges |
| 7 | Statement of Account | `soa` | `extraction_soa.py` | transactions, aging (30/60/90), total outstanding |
| 8 | Bank Statement | `bank_statement` | `extraction_bank.py` | transactions (debit/credit/balance), opening/closing balance |

---

## 5. Key Design Decisions & Why

### 5.1 Three OCR Engines, One Schema

All three OCR backends (Vision, PyMuPDF, Document Intelligence) produce the **same unified JSON schema** so the downstream orchestrator/classifier/extraction agents work identically regardless of OCR source. This was a deliberate design choice to allow comparing accuracy across engines.

**Schema structure:**
```json
{
  "source": "vision_ocr | pymupdf | azure_document_intelligence",
  "file": "filename.pdf",
  "pages": [
    {
      "page_number": 1,
      "file_name": "...",
      "sections": [
        {"type": "table_header|table_row|key_value|paragraph|...", "content": "...", "confidence": 0.95}
      ]
    }
  ],
  "metadata": { "total_pages": N, "languages_detected": [...], "extraction_method": "..." }
}
```

**Section types:** `header`, `address`, `key_value`, `table_header`, `table_row`, `subtotal`, `paragraph`, `footer`, `signature`, `empty`

### 5.2 Two-Tier Classification (LLM + Keyword Fallback)

The classifier (`agents/classifier.py`) first tries GPT-5.2 classification on the first 8,000 chars of OCR JSON. If the LLM returns "unknown" or fails entirely (auth error, content filter, etc.), a **deterministic keyword matcher** scans the OCR text for known indicators. This ensures classification never completely fails even if the LLM is unavailable.

### 5.3 Per-Page DI Splitting Workaround

Azure Document Intelligence was observed to **silently drop pages** on some multi-page PDFs. The workaround in `extraction_doc_intel.py` splits multi-page PDFs into individual single-page PDFs using PyMuPDF, sends each separately, then merges results with correct page numbering.

### 5.4 Image Preprocessing for Vision OCR

The `ocr_agent.py` includes image preprocessing (sharpening, contrast enhancement, upscaling small images) before sending to GPT-5.2, to reduce digit confusion errors (8 vs 5, 0 vs 6, etc.).

### 5.5 Multi-Pass Consensus OCR

`ocr_agent.py` includes an experimental `ocr_image_multipass()` function that runs OCR N times (default 3) with `temperature=0.3` and uses **majority voting** on key-value pairs to reduce single-digit misreads. Higher accuracy but 3x the cost.

### 5.6 Hybrid OCR (Vision + Doc Intel Cross-Validation)

`hybrid_ocr.py` merges the strengths of both engines:
- Uses **GPT Vision** output as the base (better table structure)
- **Replaces key-value fields** with Doc Intel's more accurate numeric readings
- Handles double-colon artifacts from Doc Intel (`"Key : : Value"`)

### 5.7 Currency Handling

All extraction agents strip currency symbols from monetary fields and output number-only text. A separate `currency_note` field describes what currency the values are in (e.g., "All monetary values are in MYR").

### 5.8 Extraction Rules — "Never Calculate"

All agents follow strict rules:
- Extract values **exactly** as they appear — never reformat, recalculate, or "correct"
- Missing fields → `null`
- Low OCR confidence sections → `"low_confidence": true`
- Full multi-line descriptions preserved (joined with `\n`)

---

## 6. File-by-File Reference

### Core Pipeline Files

| File | Purpose | Lines (approx) |
|---|---|---|
| `app.py` | Streamlit web UI — full 5-page application with Watsons brand theme | ~2,600+ |
| `orchestrator.py` | Classifies OCR JSON → routes to correct extraction agent | ~100 |
| `ocr_agent.py` | Vision OCR engine (GPT-5.2 + image input), includes multi-pass consensus | ~400+ |
| `ocr_table_agent.py` | Experimental table-specific OCR with cross-page table continuation | ~100 |
| `extraction_library.py` | PyMuPDF text extraction → unified JSON → orchestrator | ~200 |
| `extraction_doc_intel.py` | Azure Document Intelligence → unified JSON → orchestrator | ~200+ |
| `hybrid_ocr.py` | Merges Vision OCR tables + Doc Intel key-values | ~200+ |
| `pdf_to_images.py` | PDF → PNG conversion at 300 DPI (for Vision OCR path) | ~50 |
| `token_monitoring.py` | Azure ML token usage monitoring setup (template, not connected) | ~70 |
| `pdfplumber_demo.py` | Quick demo of pdfplumber capabilities (exploratory) | ~100 |
| `extraction_agent.py` | Legacy standalone extraction (pre-orchestrator) | Legacy |

### Agent Modules (src/agents/)

| File | Purpose |
|---|---|
| `__init__.py` | Shared Azure OpenAI client, `call_extraction_agent()`, `maybe_parse_json()` |
| `classifier.py` | Two-tier classifier: LLM + keyword fallback. Returns one of 9 labels |
| `extraction_invoice.py` | Commercial invoice & credit note schema |
| `extraction_travel.py` | Travel document (flights, tickets, PNR) schema |
| `extraction_rental.py` | Rental/lease invoice schema |
| `extraction_hotel.py` | Hotel folio schema |
| `extraction_utility.py` | Utility/telecom bill schema (with surcharges array) |
| `extraction_soa.py` | Statement of account schema (with aging breakdown) |
| `extraction_bank.py` | Bank statement schema (with transaction listings) |

### Reconciliation Module (src/reconciliation/)

| File | Purpose |
|---|---|
| `reconciliation_demo.py` | Rule-based Shopee seller reconciliation with synthetic data |
| `reconciliation_dashboard.py` | Interactive Streamlit dashboard for the demo |
| `fuzzy_matching_example.py` | Educational script explaining fuzzy matching concepts |
| `matchrule.py` | Empty placeholder |
| `ml/run_pipeline.py` | **End-to-end ML pipeline**: load → candidates → features → train → predict |
| `ml/synthetic_dataset_generator.py` | Generates realistic Shopee orders, batches, bank transactions |
| `ml/data_ingestion.py` | Load & normalize CSV files (batches, bank, mapping) |
| `ml/candidate_generation.py` | Blocking: 1-to-1 and many-to-one candidate pair generation |
| `ml/feature_engineering.py` | Computes ~15 features: amount diff, date diff, text similarity, keyword flags |
| `ml/model_training.py` | Trains LR, XGBoost, LightGBM with CV + saves models |
| `ml/prediction.py` | Load model → score → threshold (≥0.90 auto, 0.60-0.90 review, <0.60 unmatched) |
| `ml/compare_ml_vs_rules.py` | Head-to-head comparison of ML vs 3 rule-based strategies |
| `ml/pipeline_visualizer.py` | Streamlit step-by-step pipeline walkthrough (for management presentations) |
| `ml/feedback.py` | Human-in-the-loop: JSONL feedback logging + merge into training data |
| `ml/bank_reconciliaiton.py` | Legacy/initial sklearn prototype (RandomForest on bank vs cash) |

---

## 7. Data & Output Directories

### Source Documents

| Directory | Contents |
|---|---|
| `src/docs/` | Source PDFs and converted page images |
| `src/docs/database/` | Uploaded documents via Streamlit UI (with team assignments in `doc_teams.json`) |
| `src/docs/Inv_*_images/` | Converted invoice page PNGs (7 invoices: Inv_1 through Inv_7) |
| `src/docs/LL_*_images/` | Converted rental invoice PNGs (8 variants) |
| `src/docs/SOA_*_images/` | Statement of account PNGs (5 docs) |
| `src/docs/Tel_*_images/` | Telecom bill PNGs |
| `src/docs/Utility_*_images/` | Utility bill PNGs |
| `Sample to CP/` | Sample PDFs for testing (15 files including invoices, utilities, rental, telecom) |

### OCR Outputs

| Directory | Engine | Contents |
|---|---|---|
| `src/ocr_output/` | Vision OCR (GPT-5.2) | JSON files + `token_usage_log.jsonl` |
| `src/ocr_output_pymupdf/` | PyMuPDF | JSON files |
| `src/ocr_output_doc_intel/` | Azure Document Intelligence | JSON files |
| `src/ocr_output_hybrid/` | Hybrid (Vision + DI merge) | JSON files |

### Extraction Outputs

| Directory | Pipeline | Contents |
|---|---|---|
| `src/extraction_output/` | Vision OCR pipeline | Extracted JSON + `token_usage_log.jsonl` |
| `src/extraction_output_pymupdf/` | PyMuPDF pipeline | Extracted JSON |
| `src/extraction_output_doc_intel/` | Doc Intel pipeline | Extracted JSON |
| `src/extraction_output_ori/` | Original/reference results | Early extraction results + bank matching reports |

### Reconciliation Data

| Directory | Contents |
|---|---|
| `src/reconciliation/demo_output/` | Generated CSVs from the rule-based demo (orders, releases, bank statements, reconciliation results) |
| `src/reconciliation/ml/` | Shopee_orders.csv, Shopee_batches.csv, Bank_transactions.csv, Match_mapping.csv |
| `src/reconciliation/ml/models/` | Trained models: `xgboost_20260303_*.pkl`, `lightgbm_20260303_*.pkl`, `logistic_regression_20260303_*.pkl` |
| `src/reconciliation/ml/output/` | `feature_matrix.csv`, `scored_candidates.csv`, `reconciliation_summary.csv`, `ml_vs_rules_comparison.csv` |

---

## 8. Environment Configuration

### Required Environment Variables

| Variable | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |

### Optional (with defaults)

| Variable | Default | Description |
|---|---|---|
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5.2-chat` | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | API version |

### Document Intelligence Only

| Variable | Description |
|---|---|
| `AZURE_DOC_INTEL_ENDPOINT` | Document Intelligence endpoint URL |
| `AZURE_DOC_INTEL_KEY` | Document Intelligence API key |

Config values are resolved: **env var first → Streamlit secrets fallback** (`st.secrets`).

---

## 9. How to Run

### Streamlit Web UI

```bash
cd src
streamlit run app.py
```

### CLI — Vision OCR Pipeline

```bash
python pdf_to_images.py --input docs/Inv_1.pdf      # Step 1: PDF → images
python ocr_agent.py                                   # Step 2: OCR (per-page)
python ocr_agent.py --batch                            # Step 2 (alt): batch mode
python orchestrator.py --input ocr_output/Inv_1.json  # Step 3: classify + extract
```

### CLI — PyMuPDF Pipeline (single command)

```bash
python extraction_library.py --input docs/Inv_1.pdf
python extraction_library.py --input docs/            # all PDFs in directory
python extraction_library.py --input docs/LL_1.pdf --type rental  # skip classifier
```

### CLI — Document Intelligence Pipeline

```bash
python extraction_doc_intel.py --input docs/Inv_1.pdf
python extraction_doc_intel.py --input docs/
```

### CLI — Hybrid OCR

```bash
python hybrid_ocr.py --input docs/Inv_1.pdf
python hybrid_ocr.py --vision-ocr ocr_output/Inv_1.json --di-ocr ocr_output_doc_intel/Inv_1.json
```

### Reconciliation Demo

```bash
cd src/reconciliation
streamlit run reconciliation_dashboard.py
```

### ML Reconciliation Pipeline

```bash
cd src/reconciliation/ml
python synthetic_dataset_generator.py   # generate data
python run_pipeline.py                  # train + evaluate
python compare_ml_vs_rules.py          # ML vs rules comparison
streamlit run pipeline_visualizer.py   # visual walkthrough
```

---

## 10. Streamlit UI Pages (app.py)

| Page | Access | Description |
|---|---|---|
| **📤 Document Processing** | All roles | Upload PDF → full pipeline (PDF→Images→OCR→Classify→Extract). Progress tracking, page thumbnails, tabbed results. |
| **🔍 OCR Viewer** | All roles | Browse saved OCR JSONs. Section-level confidence scoring (🟢 ≥0.95, 🟡 0.90–0.94, 🔴 <0.90). |
| **📊 Extraction Viewer** | All roles | Browse extraction results. Structured fields + line item DataFrames. Repository view with doc preview cards. |
| **📋 Report Format** | All roles | Consolidated tabular report of all extractions. Per-document status (verified/rejected/pending). |
| **🏦 Bank Matching** | Admin only | AP line-item reconciliation. Matches extracted invoices/utilities against bank statement transactions. Includes demo scenarios with sample data. |

### UI Features

- **Role-based access**: Admin, Sales, Rental — filters visible docs by assigned team
- **Upload & persist**: PDFs saved to `docs/database/` with team assignment
- **OCR mode toggle**: Batch (1 API call) vs Per-page (1 call per page)
- **Force document type**: Skip auto-classification
- **PDF inline preview**: Rendered via PyMuPDF
- **Download buttons**: Export OCR and extraction JSON
- **Watsons brand theme**: Teal gradient sidebar, custom CSS

---

## 11. Cost Estimates

### Per-Document (3-page invoice)

| Stage | Vision OCR | PyMuPDF | Doc Intel |
|---|---|---|---|
| OCR | ~6K–9K tokens ($0.02–0.05) | Free | ~$0.03 |
| Classifier | ~4K tokens (~$0.01) | Same | Same |
| Extraction | ~5K tokens (~$0.03) | Same | Same |
| **Total** | **~$0.06–$0.09** | **~$0.04** | **~$0.07** |

### Scalability (100 pages)

| Scenario | Tokens | API Calls | Latency |
|---|---|---|---|
| 1 doc × 100 pages | ~151K | 2 | ~30s |
| 100 docs × 1 page | ~260K | 200 | ~10 min serial |

One large doc = ~40% cheaper on tokens; many small docs = better accuracy + fault isolation + parallelizable.

---

## 12. Known Issues

1. **DI table parsing** — `prebuilt-layout` can merge/split table rows incorrectly on complex multi-row product layouts. Vision OCR handles these better.

2. **DI page dropping** — DI silently drops pages on some multi-page PDFs. **Mitigated** by per-page splitting in `extraction_doc_intel.py`.

3. **DI barcode digit errors** — Minor OCR misreads on barcodes (e.g., `04894819001315` → `04894819901315`). Vision OCR is more accurate for barcodes.

4. **Classifier edge cases** — Some commercial invoices classified as `credit_note` due to overlapping indicators. Both types share the same extraction schema as a workaround.

---

## 13. Backlog / TODO Items

### Document Processing Pipeline

- [ ] Optimize prompts to reduce token count
- [ ] Add async/parallel processing for batch document processing
- [ ] Evaluate `prebuilt-invoice` DI model for commercial invoices (native line-item extraction, skip LLM)
- [ ] Add confidence scoring to extraction output (not just OCR)
- [ ] Add document diff/comparison view across OCR engines
- [ ] Implement retry logic with exponential backoff on API failures
- [ ] Integrate PyMuPDF and Doc Intel pipelines into the Streamlit UI (currently only Vision OCR in UI)

### Reconciliation Module

- [ ] `matchrule.py` is empty — needs implementation of rule-based matching engine
- [ ] Connect ML reconciliation pipeline to Streamlit main app (`app.py`)
- [ ] ML pipeline currently trained on synthetic Shopee data — needs real Watson's data
- [ ] Human feedback loop (`feedback.py`) implemented but not yet wired into the UI
- [ ] Many-to-one candidate matching needs more testing
- [ ] `bank_reconciliaiton.py` (note typo in filename) is a legacy prototype — may be removed

---

## 14. What Was Done (Summary of Development Work)

### Phase 1 — OCR & Extraction Core

1. Built the **Vision OCR engine** (`ocr_agent.py`) with detailed system prompts for character-perfect transcription, table structure preservation, and confidence tagging.
2. Designed the **unified OCR JSON schema** (pages → sections → type/content/confidence) shared by all three OCR backends.
3. Built **8 extraction agents** (`agents/extraction_*.py`) with document-type-specific system prompts and JSON output schemas.
4. Built the **orchestrator** (`orchestrator.py`) with agent registry pattern for routing.
5. Built the **classifier** (`classifier.py`) with two-tier LLM + keyword fallback.

### Phase 2 — Alternative OCR Backends

6. Built the **PyMuPDF pipeline** (`extraction_library.py`) — free, local, works for digital PDFs.
7. Built the **Azure Document Intelligence pipeline** (`extraction_doc_intel.py`) — with per-page splitting workaround.
8. Built the **hybrid OCR pipeline** (`hybrid_ocr.py`) — merges Vision tables + DI key-values.

### Phase 3 — Streamlit UI

9. Built the full **Streamlit web application** (`app.py`, ~2,600+ lines):
   - 5 pages (Processing, OCR Viewer, Extraction Viewer, Report, Bank Matching)
   - Watsons brand theme with custom CSS
   - Role-based access (Admin, Sales, Rental)
   - Document upload with team assignment and persistence
   - PDF inline preview, progress tracking, download buttons

### Phase 4 — Image Preprocessing & Accuracy Improvements

10. Added **image preprocessing** (sharpening, contrast enhancement, upscaling) to Vision OCR.
11. Built **multi-pass consensus OCR** with majority voting to reduce digit misreads.
12. Added **token usage logging** (JSONL files tracking prompt/completion tokens).

### Phase 5 — Reconciliation Module

13. Built **rule-based Shopee seller reconciliation demo** (`reconciliation_demo.py`) with synthetic data generation.
14. Built **interactive reconciliation dashboard** (`reconciliation_dashboard.py`) in Streamlit.
15. Built **ML reconciliation pipeline** from scratch:
    - Synthetic dataset generator (realistic Shopee orders → batches → bank deposits)
    - Candidate pair generation (1-to-1 + many-to-one blocking)
    - Feature engineering (~15 features: amount/date/text similarity)
    - Model training (Logistic Regression, XGBoost, LightGBM)
    - Prediction with confidence thresholds (auto_match ≥0.90, review 0.60-0.90)
    - ML vs rule-based comparison (`compare_ml_vs_rules.py`)
    - Pipeline visualizer for management presentations
    - Human feedback loop for continuous model improvement

### Phase 6 — AP Bank Matching in Main UI

16. Built the **AP Line-Item Reconciliation page** in `app.py` with:
    - Demo scenarios with pre-built invoice/bank data
    - Visual matching with variance detection
    - Reconciliation summary with progress bars
    - CSV/TXT export functionality

### Exploratory Work

17. `pdfplumber_demo.py` — Explored pdfplumber for table detection (alternative approach).
18. `ocr_table_agent.py` — Experimental cross-page table continuation detection.
19. `token_monitoring.py` — Azure ML token monitoring template (not connected).
20. `fuzzy_matching_example.py` — Educational script for learning fuzzy matching concepts.

---

## 15. Test Documents Available

### In `Sample to CP/`

| File | Type |
|---|---|
| `1399IV25040805...WATSON'S PERSONAL CARE STORES SDN BHD.pdf` | Commercial Invoice |
| `1800045965-0325 RTL.pdf` | Rental |
| `2810003305-0325 ELEC.pdf` | Utility (Electricity) |
| `3264400169-1225 TOR.pdf` | Unknown |
| `600011068503-0226 RTL.pdf` | Rental |
| `772000046984 - 0226 SESCO.pdf` | Utility (SESCO) |
| `ERMIV25030013-0225 TOR.pdf` | Unknown |
| `ERRIV25030174-0325 ELEC.pdf` | Utility (Electricity) |
| `IV-08469-0226 RTL.pdf` | Rental |
| `KC102182-0126 RTL CN.pdf` | Credit Note (Rental) |
| `RD551511C3_EN_433737_26095_184.pdf` | Commercial Invoice |
| `Report format.xlsx` | Reference report format |
| `SESB.pdf` | Utility (SESB) |
| `TNB.pdf` | Utility (TNB) |
| `TS_TAXINV_t0003462-2810002827_419.pdf` | Tax Invoice |

### In `src/docs/database/` (uploaded via UI)

~30+ files including: Inv_1 through Inv_7, LL_1, SOA_2, Tel_3/4/5, Utility_2/5, TNB, and more — many with timestamp-suffixed duplicates from re-uploads.

---

## 16. Trained ML Models (Reconciliation)

Located in `src/reconciliation/ml/models/`:

| Model | File | Date |
|---|---|---|
| XGBoost | `xgboost_20260303_115937.pkl` | 3 Mar 2026 |
| LightGBM | `lightgbm_20260303_115939.pkl` | 3 Mar 2026 |
| Logistic Regression | `logistic_regression_20260303_115937.pkl` | 3 Mar 2026 |

All trained on synthetic Shopee settlement data. Features and scaler are bundled inside the pickle files. Prediction thresholds: auto_match ≥0.90, review 0.60-0.90, unmatched <0.60.

---

## 17. Quick Resume Checklist (When You Come Back)

1. **Activate venv**: `.venv\Scripts\Activate.ps1`
2. **Set environment variables**: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` (and optionally `AZURE_DOC_INTEL_ENDPOINT`, `AZURE_DOC_INTEL_KEY`)
3. **Run the UI**: `cd src && streamlit run app.py`
4. **Check the README**: `src/Readme.md` has detailed architecture docs
5. **Check backlog**: Section 13 above lists open items
6. **Reconciliation demo**: `cd src/reconciliation && streamlit run reconciliation_dashboard.py`
7. **ML pipeline**: `cd src/reconciliation/ml && python run_pipeline.py`

---

*End of development log.*
