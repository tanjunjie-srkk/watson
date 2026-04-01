# Watson Document Intelligence Platform

> AI-powered document processing pipeline for OCR, classification, structured data extraction, and bank statement reconciliation — built for Watson's Personal Care Stores financial document workflows.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pipeline Flow](#pipeline-flow)
4. [Project Structure](#project-structure)
5. [OCR Engines](#ocr-engines)
6. [Document Classification](#document-classification)
7. [Extraction Agents](#extraction-agents)
8. [Streamlit UI](#streamlit-ui)
9. [Output Formats](#output-formats)
10. [Configuration & Environment Variables](#configuration--environment-variables)
11. [Usage](#usage)
12. [Cost & Performance](#cost--performance)
13. [Known Issues & Backlog](#known-issues--backlog)

---

## Overview

The platform processes financial documents (PDFs) through a multi-stage AI pipeline:

```
PDF → OCR → Classification → Extraction → Structured JSON
```

It supports **8 document types**: commercial invoices, credit notes, travel documents, rental/lease invoices, hotel folios, utility/telecom bills, statements of account, and bank statements.

Three interchangeable OCR backends are available, all producing the same intermediate JSON schema so the downstream orchestrator works identically regardless of OCR source.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                       │
│   Document Processing │ OCR Viewer │ Extraction │ Report │ Bank │
└──────────────┬──────────────────────────────────────────────────┘
               │ Upload PDF
               ▼
┌──────────────────────────────────────────────────────────────┐
│                     OCR Layer (choose one)                    │
│                                                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Vision OCR   │  │ PyMuPDF Text     │  │ Azure Doc     │  │
│  │ (ocr_agent)  │  │ (extraction_     │  │ Intelligence  │  │
│  │              │  │  library)        │  │ (extraction_  │  │
│  │ GPT-5.2 +   │  │                  │  │  doc_intel)   │  │
│  │ image input  │  │ Local, no API    │  │               │  │
│  │              │  │ cost, digital    │  │ prebuilt-     │  │
│  │ Best for     │  │ PDFs only        │  │ layout model  │  │
│  │ scanned docs │  │                  │  │               │  │
│  └──────┬───────┘  └────────┬─────────┘  └──────┬────────┘  │
│         │                   │                    │           │
│         └───────────┬───────┴────────────────────┘           │
│                     │                                        │
│              Unified OCR JSON Schema                         │
└─────────────┬────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│                  Orchestrator (orchestrator.py)               │
│                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │ Classifier       │    │ Agent Registry                  │  │
│  │ (classifier.py)  │───▶│                                │  │
│  │                  │    │  commercial_invoice → invoice   │  │
│  │ LLM + keyword   │    │  credit_note → invoice          │  │
│  │ fallback         │    │  travel → travel                │  │
│  └──────────────────┘    │  rental → rental                │  │
│                          │  hotel → hotel                  │  │
│                          │  utility → utility              │  │
│                          │  soa → soa                      │  │
│                          │  bank_statement → bank          │  │
│                          └─────────────────────────────────┘  │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
        Structured JSON Output
        (line items, totals, dates, addresses, etc.)
```

---

## Pipeline Flow

### Stage 1 — PDF to Images (Vision OCR path only)

**File:** `pdf_to_images.py`

Converts each PDF page into a 300 DPI PNG image using PyMuPDF (`fitz`). The PyMuPDF and Document Intelligence pipelines skip this step since they work directly with PDF bytes.

### Stage 2 — OCR (Text Extraction)

Three interchangeable engines produce the **same unified JSON schema**:

| Engine | File | Input | API Calls | Cost | Best For |
|--------|------|-------|-----------|------|----------|
| **Vision OCR** | `ocr_agent.py` | PNG images | 1 per page (or 1 batch) | ~1,500 tokens/page | Scanned/image-heavy PDFs |
| **PyMuPDF** | `extraction_library.py` | PDF file | 0 (local) | Free | Digital/text-based PDFs |
| **Document Intelligence** | `extraction_doc_intel.py` | PDF file | 1 per page | ~$0.01/page | Structured forms, tables |

All three produce JSON like:
```json
{
  "source": "vision_ocr | pymupdf | azure_document_intelligence",
  "file": "Inv_1.pdf",
  "pages": [
    {
      "page_number": 1,
      "file_name": "Inv_1_page_1",
      "sections": [
        {"type": "table_header", "content": "Barcode | Description | Qty | Price | Amount", "confidence": 0.95},
        {"type": "table_row", "content": "04894819001315 | COTTON PUFFS | 17760 | 0.970 | 17,227.20", "confidence": 0.95},
        {"type": "key_value", "content": "CI Number : 26001099\nDate : 18 Mar 2025", "confidence": 0.98},
        {"type": "paragraph", "content": "Please direct all correspondence to...", "confidence": 0.98}
      ]
    }
  ],
  "metadata": {
    "total_pages": 3,
    "languages_detected": ["en"],
    "extraction_method": "..."
  }
}
```

**Section types:** `header`, `address`, `key_value`, `table_header`, `table_row`, `subtotal`, `paragraph`, `footer`, `signature`, `empty`

### Stage 3 — Classification

**File:** `agents/classifier.py`

Two-tier classification strategy:

1. **LLM Classification** — Sends first 8,000 chars of OCR JSON to GPT-5.2 with a classification prompt. Returns one of 9 labels.
2. **Keyword Fallback** — If LLM returns "unknown" or fails, a deterministic keyword matcher scans the OCR text for known indicators (e.g., "kwh" → utility, "flight" → travel, "base rent" → rental).

**Supported labels:** `commercial_invoice`, `credit_note`, `travel`, `rental`, `hotel`, `utility`, `soa`, `bank_statement`, `unknown`

### Stage 4 — Extraction

**File:** `orchestrator.py` → `agents/extraction_*.py`

The orchestrator routes the OCR JSON to a **type-specific extraction agent** based on the classification label. Each agent has a tailored system prompt with:
- A document-type-specific JSON output schema
- Field definitions matching that document type
- Rules for handling multi-line descriptions, monetary values, confidence tagging

The LLM reads the entire OCR JSON and returns a structured JSON object with all extracted fields.

---

## Project Structure

```
watson/
├── requirements.txt              # Python dependencies
├── src/
│   ├── app.py                    # Streamlit UI (2,600+ lines) — full web interface
│   ├── orchestrator.py           # Classify + route to extraction agent
│   ├── ocr_agent.py              # Vision OCR engine (GPT-5.2 + image input)
│   ├── ocr_table_agent.py        # Specialized table OCR (experimental)
│   ├── extraction_library.py     # PyMuPDF text extraction pipeline
│   ├── extraction_doc_intel.py   # Azure Document Intelligence pipeline
│   ├── extraction_agent.py       # Legacy standalone extraction (pre-orchestrator)
│   ├── pdf_to_images.py          # PDF → PNG conversion using PyMuPDF
│   ├── token_monitoring.py       # Azure ML token usage monitoring setup
│   │
│   ├── agents/                   # Extraction agent modules
│   │   ├── __init__.py           # Shared Azure OpenAI client, call_extraction_agent()
│   │   ├── classifier.py         # LLM + keyword document classifier
│   │   ├── extraction_invoice.py # Commercial invoice & credit note extraction
│   │   ├── extraction_travel.py  # Travel document extraction
│   │   ├── extraction_rental.py  # Rental/lease invoice extraction
│   │   ├── extraction_hotel.py   # Hotel folio extraction
│   │   ├── extraction_utility.py # Utility/telecom bill extraction
│   │   ├── extraction_soa.py     # Statement of account extraction
│   │   └── extraction_bank.py    # Bank statement extraction
│   │
│   ├── docs/                     # Source PDFs and converted images
│   │   ├── database/             # Uploaded documents (via Streamlit UI)
│   │   ├── Inv_*_images/         # Converted invoice page images
│   │   ├── LL_*_images/          # Converted rental invoice images
│   │   ├── SOA_*_images/         # Converted SOA images
│   │   ├── Tel_*_images/         # Converted telecom bill images
│   │   └── Utility_*_images/     # Converted utility bill images
│   │
│   ├── ocr_output/               # Vision OCR JSON results
│   ├── ocr_output_pymupdf/       # PyMuPDF OCR JSON results
│   ├── ocr_output_doc_intel/     # Document Intelligence OCR JSON results
│   │
│   ├── extraction_output/        # Vision OCR extraction results
│   ├── extraction_output_pymupdf/# PyMuPDF extraction results
│   ├── extraction_output_doc_intel/ # Doc Intel extraction results
│   └── extraction_output_ori/    # Original/reference extraction results
```

---

## OCR Engines

### 1. Vision OCR (`ocr_agent.py`)

Uses **Azure OpenAI GPT-5.2** with image input capability. Each PDF page is rendered to a 300 DPI PNG image and sent to the model with a detailed system prompt that instructs character-perfect transcription.

**Two modes:**
- **Per-page mode** (default): One API call per page image. Higher accuracy per page, but more calls.
- **Batch mode**: All page images in a single API call. Fewer calls / lower system-prompt overhead, but attention may degrade on many pages.

**Key prompt rules:**
- Transcribe characters exactly as printed — never correct spelling, recalculate sums, or infer missing data
- Classify every block of text into a section type (header, address, key_value, table_header, table_row, subtotal, paragraph, footer, signature, empty)
- Multi-line product descriptions must be joined into a single `table_row` section
- Assign confidence score (0.0–1.0) to each section

**Strengths:** Best accuracy for scanned documents, handwriting, complex layouts with merged cells.
**Weaknesses:** Highest cost (~1,500 tokens per page image + system prompt).

### 2. PyMuPDF (`extraction_library.py`)

Uses the **PyMuPDF (fitz)** library for direct text extraction from PDF objects. No API calls. Zero cost.

Each page's text is extracted via `page.get_text()` and wrapped in a `paragraph` section of the unified JSON schema.

**Strengths:** Free, fast, works offline, perfect for digitally-created PDFs.
**Weaknesses:** Cannot read scanned/image-only PDFs. No table structure recognition — all text comes as flat paragraphs.

### 3. Azure Document Intelligence (`extraction_doc_intel.py`)

Uses the **Azure Document Intelligence** service with the `prebuilt-layout` model and `KEY_VALUE_PAIRS` feature.

**Per-page splitting workaround:** Multi-page PDFs are split into individual single-page PDFs using PyMuPDF before sending to DI, because the DI service was observed to silently drop pages on some multi-page documents (e.g., page 3 of a 3-page invoice). Each page is analyzed separately, then results are merged with correct page numbering.

The DI result includes:
- **Tables** with cell-level structure → converted to `table_header` + `table_row` sections
- **Key-value pairs** with bounding regions → converted to `key_value` sections
- **Lines/paragraphs** → converted to `paragraph` sections

**Strengths:** Good at extracting structured forms, key-value pairs. Much cheaper than Vision OCR.
**Weaknesses:** Table parsing can merge/split rows incorrectly on complex layouts (e.g., multi-row product descriptions with sub-barcodes). Minor OCR digit errors observed (e.g., reading `0` as `9` in barcodes).

---

## Document Classification

**File:** `agents/classifier.py`

### Strategy

```
OCR JSON ──▶ LLM Classifier (GPT-5.2, first 8K chars)
                 │
                 ├──▶ Valid label? ──▶ Return label
                 │
                 └──▶ "unknown" or error
                          │
                          ▼
                 Keyword Fallback (regex/substring scan)
                          │
                          ▼
                 Return best-match label or "unknown"
```

### Classification Hints

| Indicators | → Label |
|---|---|
| Barcodes, PO numbers, trade terms (CFR, FOB), bill of lading | `commercial_invoice` |
| Flight numbers, routing, passenger names, ticket numbers | `travel` |
| Base rent, service charge, tenancy period, lot/unit number | `rental` |
| Room charge, check-in/check-out, folio, guest name | `hotel` |
| Meter reading, tariff, kWh, billing period, broadband, telephone | `utility` |
| List of invoices with aging (30/60/90 days), outstanding balance | `soa` |
| Bank name, account transactions, debit/credit columns, running balance | `bank_statement` |
| "Credit Note" or "CN" in title | `credit_note` |

---

## Extraction Agents

Each agent is a Python module under `agents/` containing a `SYSTEM_PROMPT` and `USER_PROMPT`. The orchestrator concatenates `USER_PROMPT + OCR_JSON` and sends it to GPT-5.2 with the `SYSTEM_PROMPT`.

### Common Extraction Rules (all agents)

1. Extract values **exactly** as they appear — no reformatting, recalculating, or correcting
2. `table_row` values are separated by `" | "` — map to columns by position using `table_header`
3. Missing fields → `null`
4. Low OCR confidence (`< 0.90`) → `"low_confidence": true`
5. Monetary values → number-only text (no currency symbols)
6. Full multi-line descriptions joined with `\n`

### Agent-Specific Schemas

| Agent | Key Fields |
|---|---|
| **Commercial Invoice** (`extraction_invoice.py`) | `vendor_name`, `invoice_number`, `invoice_date`, `po_number`, `bu_po_number`, `currency`, `bill_to`, `ship_to`, `payment_terms`, `trade_terms`, `line_items[]` (barcode, description, qty, unit_price, discount, tax, amount), `subtotal`, `freight_charges`, `grand_total` |
| **Travel** (`extraction_travel.py`) | `vendor_name`, `invoice_number`, `passengers[]`, `flights[]` (origin, destination, flight_number, departure/arrival dates, ticket_number, PNR), `line_items[]`, `grand_total` |
| **Rental** (`extraction_rental.py`) | `vendor_name`, `invoice_number`, `property_name`, `unit_number`, `trade_name`, `tenancy_period`, `line_items[]` (description, period_from, period_to, tax, amount), `payment_info` |
| **Hotel** (`extraction_hotel.py`) | `vendor_name`, `guest_name`, `check_in_date`, `check_out_date`, `room_number`, `room_type`, `nights`, `line_items[]`, `advances`, `grand_total` |
| **Utility** (`extraction_utility.py`) | `vendor_name`, `account_number`, `billing_period_from/to`, `service_address`, `line_items[]`, `subtotal`, `surcharges[]` (label, amount), `previous_balance`, `payment_received`, `grand_total` |
| **SOA** (`extraction_soa.py`) | `vendor_name`, `customer_name`, `statement_period`, `transactions[]` (date, document_number, type, debit, credit, balance), `aging` (current, 30/60/90/over_90), `total_outstanding` |
| **Bank Statement** (`extraction_bank.py`) | `bank_name`, `account_holder`, `account_number`, `opening_balance`, `closing_balance`, `transactions[]` (date, value_date, description, reference, debit, credit, balance), `total_debits`, `total_credits` |

---

## Streamlit UI

**File:** `app.py` (~2,600 lines)

A full-featured web application with Watsons brand theming (teal/red palette: `#006770`, `#00A0AF`, `#57e2c8`).

### Pages

| Page | Description |
|---|---|
| **📤 Document Processing** | Upload PDF → runs full pipeline (PDF→Images→OCR→Classify→Extract). Shows progress, page thumbnails, OCR and extraction results in tabs. Saves to `ocr_output/` and `extraction_output/`. |
| **🔍 OCR Viewer** | Browse saved OCR JSON results. Shows section breakdown with confidence scoring (🟢 ≥ 0.95, 🟡 0.90–0.94, 🔴 < 0.90). |
| **📊 Extraction Viewer** | Browse saved extraction results. Shows structured fields and line item DataFrames. Repository view with document preview cards. |
| **📋 Report Format** | Loads all extractions into a consolidated tabular report (DataFrame). Supports document-level status (verified / rejected / pending). |
| **🏦 Bank Matching** | (Admin only) Reconcile extracted invoices/utilities against bank statement transactions. Line-item matching with variance detection. |

### Theme Customization

- Sidebar: Teal gradient (`#006770` → `#00A0AF`) with Watsons branding
- Cards: White backgrounds with teal left-border accent
- Buttons: Primary teal, secondary red
- Status badges: Color-coded confidence indicators
- Responsive metric cards with stat displays

### Features

- **Role-based access**: Admin, Sales, Rental — filters visible documents by assigned team
- **Upload & persist**: PDFs saved to `docs/database/` with team and category assignment
- **OCR mode toggle**: Batch (all pages in 1 API call) or Per-page (1 call per page)
- **Force document type**: Users can skip auto-classification and select type manually
- **PDF inline preview**: Renders pages as images using PyMuPDF
- **Download buttons**: Export OCR and extraction JSON results

---

## Output Formats

### OCR Output
Unified JSON schema across all three engines. Stored in:
- `ocr_output/` — Vision OCR
- `ocr_output_pymupdf/` — PyMuPDF
- `ocr_output_doc_intel/` — Document Intelligence

### Extraction Output
Type-specific JSON (see [Extraction Agents](#extraction-agents)). Stored in:
- `extraction_output/` — from Vision OCR pipeline
- `extraction_output_pymupdf/` — from PyMuPDF pipeline
- `extraction_output_doc_intel/` — from Document Intelligence pipeline

### Token Usage Logs
JSONL files tracking prompt/completion token counts per API call:
- `ocr_output/token_usage_log.jsonl` — OCR token usage
- `extraction_output/token_usage_log.jsonl` — Extraction token usage

---

## Configuration & Environment Variables

### Required

| Variable | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |

### Optional (with defaults)

| Variable | Default | Description |
|---|---|---|
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5.2-chat` | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | API version |

### Document Intelligence Pipeline Only

| Variable | Description |
|---|---|
| `AZURE_DOC_INTEL_ENDPOINT` | Document Intelligence endpoint URL |
| `AZURE_DOC_INTEL_KEY` | Document Intelligence API key |

All config values are resolved first from environment variables, then from Streamlit secrets (`st.secrets`), enabling both CLI and Streamlit deployment.

---

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
# Core: streamlit, pandas, openai, pymupdf
# For Doc Intel pipeline: pip install azure-ai-documentintelligence azure-core
```

### Streamlit Web UI

```bash
cd src
streamlit run app.py
```

### CLI — Vision OCR Pipeline

```bash
# Step 1: Convert PDF to images
python pdf_to_images.py --input docs/Inv_1.pdf

# Step 2: OCR (per-page mode)
python ocr_agent.py

# Step 2 (alt): OCR (batch mode — all pages in one call)
python ocr_agent.py --batch

# Step 3: Classify + Extract
python orchestrator.py --input ocr_output/Inv_1.json

# Step 3 (with forced type):
python orchestrator.py --input ocr_output/Inv_1.json --type commercial_invoice
```

### CLI — PyMuPDF Pipeline

```bash
# Single PDF
python extraction_library.py --input docs/Inv_1.pdf

# All PDFs in a directory
python extraction_library.py --input docs/

# Force document type
python extraction_library.py --input docs/LL_1.pdf --type rental
```

### CLI — Document Intelligence Pipeline

```bash
# Single PDF
python extraction_doc_intel.py --input docs/Inv_1.pdf

# All PDFs in a directory
python extraction_doc_intel.py --input docs/

# Force document type
python extraction_doc_intel.py --input docs/LL_1.pdf --type rental
```

---

## Cost & Performance

### Per-Document Cost Estimate (3-page invoice)

| Stage | Vision OCR | PyMuPDF | Doc Intel |
|---|---|---|---|
| OCR | ~6K–9K tokens ($0.02–0.05) | Free | ~$0.03 (3 × $0.01) |
| Classifier | ~4K tokens (~$0.01) | Same | Same |
| Extraction | ~5K tokens (~$0.03) | Same | Same |
| **Total** | **~$0.06–$0.09** | **~$0.04** | **~$0.07** |

### Token Billing Model

Azure OpenAI charges **per token**, not per API call:
- **Input tokens**: ~$2.50–$5.00 per 1M tokens (varies by model/region)
- **Output tokens**: ~$10.00–$15.00 per 1M tokens
- **Image tokens**: ~1,500 tokens per page image (counted as input)
- System prompt is re-sent on every call — batch mode amortizes this overhead

### Scalability: 100 Pages — One Large Doc vs. Many Small Docs

| Scenario | LLM Input Tokens | LLM API Calls | Latency |
|---|---|---|---|
| 1 doc × 100 pages | ~151K | 2 (OCR + extract) | ~30s |
| 100 docs × 1 page | ~260K | 200 (100 OCR + 100 extract) | ~10 min serial |

**1 large document** is ~40% cheaper on tokens, but **many small documents** offers:
- Better accuracy (LLMs lose attention over very long inputs)
- Better fault isolation (1 failure loses only 1 document)
- Parallelizable (can run concurrently)

---

## Known Issues & Backlog

### Known Issues

1. **DI table parsing quality** — Document Intelligence `prebuilt-layout` can merge or split table rows incorrectly on complex multi-row product layouts (e.g., invoices with sub-barcode rows beneath main product rows). Vision OCR handles these more accurately.

2. **DI page dropping** — The DI service may silently drop pages on multi-page PDFs. **Mitigated** by splitting PDFs into single-page files before sending (implemented in `extraction_doc_intel.py`).

3. **DI barcode digit errors** — Minor OCR misreads observed on barcodes (e.g., `04894819001315` read as `04894819901315`). Vision OCR is more accurate for barcode digits.

4. **Classifier edge cases** — Some commercial invoices are classified as `credit_note` due to overlapping indicators. Both types share the same extraction schema as a workaround.

### Backlog

- [ ] Optimize prompts to reduce token count
- [ ] Add async/parallel processing for batch document processing
- [ ] Evaluate `prebuilt-invoice` DI model for commercial invoices (native line-item extraction, skip LLM)
- [ ] Add confidence scoring to extraction output (not just OCR)
- [ ] Add document diff/comparison view across OCR engines
- [ ] Implement retry logic with exponential backoff on API failures
- [ ] Integrate PyMuPDF and Document Intelligence pipelines into the Streamlit UI (currently Vision OCR only in UI)