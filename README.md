# Watson Document Intelligence Platform

> AI-powered document processing pipeline for OCR, classification, structured data extraction, and bank reconciliation — built for Watson's Personal Care Stores financial document workflows.

---

## Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [Tech Stack](#tech-stack)
3. [Requirements](#requirements)
4. [Repository Structure](#repository-structure)
5. [Important Files](#important-files)
6. [Working Concept](#working-concept)

---

## What Is This Project?

**Watson Document Intelligence Platform** is an AI-powered document processing system built for **Watson's Personal Care Stores SDN BHD**. The business receives hundreds of financial documents from suppliers, landlords, utilities, and banks. This platform automates the extraction of structured data from those documents — replacing manual data entry and feeding clean JSON into downstream accounting/ERP systems.

The platform processes PDFs through a multi-stage pipeline:

```
PDF → OCR → Classification → Extraction → Structured JSON
```

It supports **8 document types**:

| # | Document Type | Example |
|---|---|---|
| 1 | Commercial Invoice | Supplier invoices with barcodes & line items |
| 2 | Credit Note | Supplier credit notes (reuses invoice schema) |
| 3 | Travel Document | Flight bookings, ticket numbers, PNR |
| 4 | Rental/Lease Invoice | Tenancy charges by property unit |
| 5 | Hotel Folio | Guest stays, room charges |
| 6 | Utility/Telecom Bill | TNB electricity, broadband, telephone bills |
| 7 | Statement of Account (SOA) | Invoice aging by 30/60/90 days |
| 8 | Bank Statement | Transaction listing with debit/credit/balance |

A separate **bank reconciliation module** (Shopee seller settlement matching) is also included as a distinct workstream under `src/reconciliation/`.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM / Vision OCR** | Azure OpenAI GPT-5.2 | Document classification, data extraction, vision-based OCR |
| **OCR (Local)** | PyMuPDF (`fitz`) | Free, offline text extraction for digital PDFs |
| **OCR (Cloud)** | Azure Document Intelligence (`prebuilt-layout`) | Structured form & table extraction |
| **UI** | Streamlit | Full web application with Watsons brand theming |
| **PDF Processing** | PyMuPDF (`fitz`) | PDF → 300 DPI PNG page images |
| **ML Reconciliation** | XGBoost, LightGBM, Logistic Regression | Shopee settlement matching |
| **Fuzzy Matching** | rapidfuzz | Text similarity for reconciliation |
| **Data** | pandas | All tabular processing |
| **Visualization** | Plotly | Charts in reconciliation dashboards |
| **Language** | Python 3.12+ | Uses `str | None` union syntax |

---

## Requirements

The core `requirements.txt` lists the minimum dependencies:

```
streamlit
pandas
openai
pymupdf
```

Additional packages are installed separately depending on the pipeline used:

| Package(s) | Required For |
|---|---|
| `azure-ai-documentintelligence`, `azure-core` | Azure Document Intelligence OCR pipeline |
| `rapidfuzz` | Reconciliation fuzzy matching |
| `xgboost`, `lightgbm`, `scikit-learn` | ML reconciliation pipeline |
| `plotly` | Reconciliation dashboards |
| `Pillow` | Image preprocessing in Vision OCR |
| `azure-ai-ml`, `azure-identity` | Token usage monitoring (Azure ML) |

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI service endpoint URL |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | No (default: `gpt-5.2-chat`) | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | No (default: `2024-12-01-preview`) | API version |
| `AZURE_DOC_INTEL_ENDPOINT` | Doc Intel only | Document Intelligence endpoint URL |
| `AZURE_DOC_INTEL_KEY` | Doc Intel only | Document Intelligence API key |

> Config values are resolved: **env var first → Streamlit secrets fallback** (`st.secrets`).

---

## Repository Structure

```
watson/
├── README.md                         # This file
├── DEVELOPMENT_LOG.md                # Detailed dev log with architecture & decisions
├── requirements.txt                  # Core Python dependencies
├── Sample to CP/                     # Sample PDFs for testing (invoices, utilities, rental, telecom)
│
└── src/
    ├── app.py                        # Streamlit web UI (~2,600 lines) — main entry point
    ├── orchestrator.py               # Classify OCR JSON → route to extraction agent
    ├── ocr_agent.py                  # Vision OCR engine (GPT-5.2 + image input)
    ├── ocr_table_agent.py            # Experimental table-specific OCR
    ├── extraction_library.py         # PyMuPDF text extraction → unified JSON
    ├── extraction_doc_intel.py       # Azure Document Intelligence → unified JSON
    ├── hybrid_ocr.py                 # Merges Vision OCR tables + Doc Intel key-values
    ├── pdf_to_images.py              # PDF → PNG conversion at 300 DPI
    ├── extraction_agent.py           # Legacy standalone extraction (pre-orchestrator)
    ├── token_monitoring.py           # Azure ML token usage monitoring setup
    ├── pdfplumber_demo.py            # Exploratory pdfplumber demo
    │
    ├── agents/                       # Document-type extraction agents
    │   ├── __init__.py               # Shared Azure OpenAI client + call_extraction_agent()
    │   ├── classifier.py             # LLM + keyword document type classifier
    │   ├── extraction_invoice.py     # Commercial invoice & credit note
    │   ├── extraction_travel.py      # Travel documents
    │   ├── extraction_rental.py      # Rental/lease invoices
    │   ├── extraction_hotel.py       # Hotel folios
    │   ├── extraction_utility.py     # Utility/telecom bills
    │   ├── extraction_soa.py         # Statement of account
    │   └── extraction_bank.py        # Bank statements
    │
    ├── docs/                         # Source PDFs and converted page images
    │   ├── database/                 # PDFs uploaded via Streamlit UI
    │   ├── Inv_*_images/             # Invoice page PNGs
    │   ├── LL_*_images/              # Rental invoice page PNGs
    │   ├── SOA_*_images/             # SOA page PNGs
    │   ├── Tel_*_images/             # Telecom bill page PNGs
    │   └── Utility_*_images/         # Utility bill page PNGs
    │
    ├── ocr_output/                   # Vision OCR JSON output
    ├── ocr_output_pymupdf/           # PyMuPDF OCR JSON output
    ├── ocr_output_doc_intel/         # Document Intelligence OCR JSON output
    ├── ocr_output_hybrid/            # Hybrid (Vision + DI) OCR JSON output
    │
    ├── extraction_output/            # Vision OCR pipeline extracted JSON
    ├── extraction_output_pymupdf/    # PyMuPDF pipeline extracted JSON
    ├── extraction_output_doc_intel/  # Doc Intel pipeline extracted JSON
    ├── extraction_output_ori/        # Original/reference extraction results
    │
    └── reconciliation/               # Bank reconciliation module (separate workstream)
        ├── reconciliation_demo.py    # Rule-based Shopee reconciliation
        ├── reconciliation_dashboard.py # Streamlit dashboard for demo
        ├── fuzzy_matching_example.py # Fuzzy matching educational demo
        └── ml/                       # ML-based reconciliation pipeline
            ├── run_pipeline.py       # End-to-end orchestrator
            ├── synthetic_dataset_generator.py
            ├── data_ingestion.py
            ├── candidate_generation.py
            ├── feature_engineering.py
            ├── model_training.py
            ├── prediction.py
            ├── compare_ml_vs_rules.py
            ├── pipeline_visualizer.py
            └── feedback.py
```

---

## Important Files

| File | Why It Matters |
|---|---|
| [src/app.py](src/app.py) | **Main entry point.** Full Streamlit web application with 5 pages: Document Processing, OCR Viewer, Extraction Viewer, Report Format, and Bank Matching. Upload a PDF here and the entire pipeline runs automatically. |
| [src/orchestrator.py](src/orchestrator.py) | **Pipeline coordinator.** Takes an OCR JSON file, calls the classifier to determine document type, then routes to the correct extraction agent. The single function that ties classification + extraction together. |
| [src/agents/classifier.py](src/agents/classifier.py) | **Document type classifier.** Two-tier strategy: LLM-based (GPT-5.2) primary + deterministic keyword fallback. Returns one of 9 labels. Critical because downstream extraction schemas depend on correctly identifying the document type. |
| [src/agents/\_\_init\_\_.py](src/agents/__init__.py) | **Shared client & utilities.** Instantiates the Azure OpenAI client and exposes `call_extraction_agent()` and `maybe_parse_json()` used by every extraction agent. |
| [src/ocr_agent.py](src/ocr_agent.py) | **Vision OCR engine.** Converts page images to GPT-5.2 API calls with structured output. Includes image preprocessing (sharpening, contrast), per-page and batch modes, and an experimental multi-pass consensus OCR. |
| [src/extraction_library.py](src/extraction_library.py) | **PyMuPDF pipeline.** Free, offline OCR for digital PDFs. Produces the same unified JSON schema as Vision OCR, allowing a drop-in swap of the OCR source. |
| [src/extraction_doc_intel.py](src/extraction_doc_intel.py) | **Azure Document Intelligence pipeline.** Includes the per-page splitting workaround for multi-page PDFs (DI silently drops pages otherwise) and merges all page results back into the unified schema. |
| [src/hybrid_ocr.py](src/hybrid_ocr.py) | **Best-of-both engine.** Merges Vision OCR (better table structure) with Document Intelligence (more accurate key-value numerics), deduplicates artifacts, and produces a higher-quality combined output. |
| [src/agents/extraction_invoice.py](src/agents/extraction_invoice.py) | **Most-used extraction agent.** Handles both commercial invoices and credit notes. Contains the most complex prompt schema (barcodes, PO numbers, multi-line descriptions, freight charges). |
| [src/reconciliation/ml/run_pipeline.py](src/reconciliation/ml/run_pipeline.py) | **ML reconciliation orchestrator.** Runs the full Shopee seller settlement matching pipeline: load CSVs → generate candidates → engineer features → train models → score predictions. |

---

## Working Concept

### Document Processing Pipeline

The platform processes any financial PDF through four sequential stages:

```
┌──────────────────────────────────────────────────────────────┐
│                   Streamlit UI  (app.py)                     │
│  5 pages: Processing │ OCR Viewer │ Extraction │ Report │ Bank│
└─────────────────────────┬────────────────────────────────────┘
                          │ Upload PDF
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     OCR Layer (choose one)                    │
│                                                               │
│   Vision OCR          PyMuPDF            Azure Doc Intel      │
│   (ocr_agent.py)      (extraction_       (extraction_         │
│   GPT-5.2 +           library.py)        doc_intel.py)        │
│   image input         Local, free        prebuilt-layout      │
│                                                               │
│              ↓  Unified OCR JSON Schema  ↓                    │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  Orchestrator (orchestrator.py)               │
│                                                               │
│   classifier.py ──▶ document type label                      │
│                           ↓                                   │
│   AGENT_REGISTRY ──▶ extraction_*.py                         │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
                  Structured JSON Output
```

#### Stage 1 — PDF to Images *(Vision OCR path only)*

`pdf_to_images.py` renders each PDF page to a 300 DPI PNG using PyMuPDF. The PyMuPDF and Document Intelligence paths skip this step entirely.

#### Stage 2 — OCR

Three interchangeable engines all produce the **same unified JSON schema**:

```json
{
  "source": "vision_ocr | pymupdf | azure_document_intelligence",
  "pages": [
    {
      "page_number": 1,
      "sections": [
        { "type": "table_row", "content": "04894819001315 | COTTON PUFFS | 17760 | 0.970 | 17,227.20", "confidence": 0.95 },
        { "type": "key_value", "content": "CI Number : 26001099\nDate : 18 Mar 2025", "confidence": 0.98 }
      ]
    }
  ]
}
```

Section types include: `header`, `address`, `key_value`, `table_header`, `table_row`, `subtotal`, `paragraph`, `footer`, `signature`, `empty`.

| Engine | File | Cost | Best For |
|---|---|---|---|
| Vision OCR | `ocr_agent.py` | ~1,500 tokens/page | Scanned / image-heavy PDFs |
| PyMuPDF | `extraction_library.py` | Free (offline) | Digital/text-based PDFs |
| Document Intelligence | `extraction_doc_intel.py` | ~$0.01/page | Structured forms and tables |

The unified schema means you can swap OCR engines without changing anything downstream.

#### Stage 3 — Classification

`agents/classifier.py` uses a **two-tier strategy**:

1. **LLM** — sends the first 8,000 characters of OCR JSON to GPT-5.2 for classification.
2. **Keyword fallback** — if LLM returns `unknown` or fails, a deterministic scanner checks for known indicators (e.g., `kwh` → `utility`, `flight` → `travel`, `base rent` → `rental`).

This ensures classification never completely fails even when the LLM is unavailable.

#### Stage 4 — Extraction

`orchestrator.py` maps the classification label to a type-specific extraction agent:

| Label | Agent |
|---|---|
| `commercial_invoice`, `credit_note` | `extraction_invoice.py` |
| `travel` | `extraction_travel.py` |
| `rental` | `extraction_rental.py` |
| `hotel` | `extraction_hotel.py` |
| `utility` | `extraction_utility.py` |
| `soa` | `extraction_soa.py` |
| `bank_statement` | `extraction_bank.py` |

Each agent sends the full OCR JSON to GPT-5.2 with a tailored system prompt containing a document-type-specific output schema. The LLM returns a structured JSON object with all relevant fields. Core extraction rules applied by every agent:

- Extract values **exactly** as they appear — never reformat, recalculate, or correct
- Missing fields → `null`
- Monetary values → number-only (no currency symbols), with a `currency_note` field
- Low OCR confidence sections (< 0.90) are flagged with `"low_confidence": true`

---

### Bank Reconciliation Module

A separate workstream under `src/reconciliation/` that matches Shopee seller settlement payouts against bank transactions.

**Rule-based path:** `reconciliation_demo.py` applies deterministic matching rules on synthetic Shopee order/batch/bank data.

**ML path** (`ml/` subdirectory): A full supervised learning pipeline —
1. `synthetic_dataset_generator.py` — generates realistic Shopee orders, batches, and bank records
2. `candidate_generation.py` — creates candidate pairs via blocking (1-to-1 and many-to-one)
3. `feature_engineering.py` — computes ~15 features (amount diff, date diff, text similarity, keyword flags)
4. `model_training.py` — trains Logistic Regression, XGBoost, and LightGBM with cross-validation
5. `prediction.py` — scores candidates with threshold-based routing: ≥ 0.90 auto-match, 0.60–0.90 human review, < 0.60 unmatched

---

### How to Run

**Streamlit UI (recommended):**
```bash
cd src
streamlit run app.py
```

**CLI — Vision OCR pipeline:**
```bash
python pdf_to_images.py --input docs/Inv_1.pdf        # Step 1: PDF → images
python ocr_agent.py                                    # Step 2: OCR
python orchestrator.py --input ocr_output/Inv_1.json  # Step 3: classify + extract
```

**CLI — PyMuPDF pipeline (single command):**
```bash
python extraction_library.py --input docs/Inv_1.pdf
python extraction_library.py --input docs/            # all PDFs in directory
```

**CLI — Document Intelligence pipeline:**
```bash
python extraction_doc_intel.py --input docs/Inv_1.pdf
python extraction_doc_intel.py --input docs/
```
