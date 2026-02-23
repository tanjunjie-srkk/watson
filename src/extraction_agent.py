import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from openai import AzureOpenAI

def _get_config_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass

    return None


def _get_required_env(name: str) -> str:
    value = _get_config_value(name)
    if not value:
        raise RuntimeError(
            f"Missing required config value: {name}. Set it as an environment variable or Streamlit secret."
        )
    return value


def _extract_usage_dict(completion: object) -> dict | None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _append_token_usage_log(entry: dict) -> None:
    log_path = Path(__file__).resolve().parent / "extraction_output" / "token_usage_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_token_usage(completion: object, ocr_payload_chars: int) -> None:
    usage = _extract_usage_dict(completion)
    if not usage:
        return

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_mode": "extraction_from_ocr",
        "model": deployment,
        "ocr_payload_chars": ocr_payload_chars,
        **usage,
    }
    _append_token_usage_log(entry)


endpoint = _get_required_env("AZURE_OPENAI_ENDPOINT")
deployment = _get_config_value("AZURE_OPENAI_DEPLOYMENT") or "gpt-5.2-chat"
subscription_key = _get_required_env("AZURE_OPENAI_API_KEY")
api_version = _get_config_value("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

SYSTEM_PROMPT = """
You are a structured data extraction engine for financial documents.

You receive OCR output (JSON) from a prior OCR agent. Your task is to extract financially relevant information and return it in a structured JSON format.

The documents you process vary in type — rental bills, rental invoices, commercial invoices, flight tickets, hotel bills, utility bills, statements of account, etc. You must handle ALL types using the schema below.

═══ EXTRACTION RULES ═══

1. Extract values EXACTLY as they appear in the OCR output — do NOT reformat, recalculate, or correct them.
2. If a field appears on multiple pages with the same value (e.g. repeated headers), extract it ONCE.
3. For line items, collect ALL item rows from ALL pages into a single "line_items" array.
4. Use table_header sections to understand column order, then map table_row values by position.
   - table_row values are separated by " | ".
5. Ignore rows that contain only barcode/SKU/reference sub-information with no description and no amount.
6. If a core field is not found in the OCR output, set its value to null — do NOT guess or infer.
7. If the OCR confidence for a section is below 0.90, set "low_confidence": true on the relevant extracted item.
8. For ALL monetary fields, return amount-only text with NO currency symbols or codes.
    - Remove prefixes/suffixes like "MYR", "USD", "RM", "$", "SGD", etc.
    - Keep digits, commas, decimal points, minus signs, and parentheses exactly as shown.
    - Example: "MYR 1,234.56" → "1,234.56".

═══ CORE FIELDS (always extract — set null if not found) ═══

- document_type     : classify the document (e.g. "Rental or Rental Invoice ","Commercial Invoice", "Flight Ticket", "Hotel Invoice", "Utility Bill", "Statement of Account", "Credit Note", "SOA","Telephone Bill" etc.)
- vendor_name       : the company/entity issuing the document
- document_number   : invoice number, ticket number, reference number, or any primary document identifier
- document_date     : the main date on the document (preserve original format exactly)
- currency          : the currency used (e.g. "USD", "MYR") — extract from totals or amount fields
- total_amount      : the final total / grand total / net amount (as string, amount only, no currency symbols/codes)
- bill_to           : the recipient / customer name and address (as a single string, null if not found)

═══ LINE ITEMS (always extract into line_items array) ═══

For each item/service/charge row, extract:
- item_number       : row number or sequence number if shown (as string, null if not shown)
- description       : the FULL description text — include ALL lines (main name, variant, sub-description, campaign name, routing, passenger, etc.) joined with "\\n". Do NOT truncate or summarize.
- quantity          : as string, preserve original formatting (null if not shown)
- unit_price        : as string, preserve original numeric formatting, amount only (no currency symbols/codes) (null if not shown)
- tax               : as string, preserve original numeric formatting, amount only (no currency symbols/codes) (null if not shown)
- amount            : as string, preserve original numeric formatting, amount only (no currency symbols/codes) (null if not shown)
- low_confidence    : true if OCR confidence for this row was below 0.90, otherwise omit

═══ TOTALS (always extract — set null if not found) ═══

- subtotal          : as string, amount only (no currency symbols/codes), preserve numeric formatting
- tax_total         : total tax amount, as string, amount only (no currency symbols/codes) (null if not shown)
- discount          : discount amount, as string, amount only (no currency symbols/codes) (null if not shown)
- freight_charges   : shipping/freight charges, as string, amount only (no currency symbols/codes) (null if not shown)
- grand_total       : final total amount, as string, amount only (no currency symbols/codes), preserve numeric formatting
- amount_in_words   : the total written in words if present (null if not shown)

═══ ADDITIONAL FIELDS (dynamic — extract anything financially relevant not covered above) ═══

Place ALL other extracted key-value information into an "additional_fields" object.
The agent MUST scan the entire OCR output for any key-value pair, label-value pair, or named field that is financially or operationally relevant and include it here.

Examples of what belongs in additional_fields (non-exhaustive):
- PO numbers, reference numbers, booking numbers
- Payment terms, trade terms, bank details
- Dates (departure, arrival, check-in, check-out, due date)
- Addresses (business unit, ship-to, billing address)
- Routing, flight numbers, ticket numbers, passenger names
- Room numbers, hotel names, guest names
- Account numbers, contract numbers
- Country of origin, bill of lading, shipping details
- Any other labeled field visible in the document

Use the EXACT label from the document as the key (e.g. "PO Number", "Departure Date", "Ticket No.", "Room Reference").
Use the EXACT value from the OCR as the value — no reformatting.

═══ OUTPUT SCHEMA ═══

Return ONLY a single valid JSON object. No markdown fences. No explanations.

{
  "document_type": "<string>",
  "vendor_name": "<string or null>",
  "document_number": "<string or null>",
  "document_date": "<string or null>",
  "currency": "<string or null>",
  "total_amount": "<string or null>",
  "bill_to": "<string or null>",
  "line_items": [
    {
      "item_number": "<string or null>",
      "description": "<full multi-line description>",
      "quantity": "<string or null>",
      "unit_price": "<string or null>",
      "Reference / Supporting Information": "<string or null>",
      "tax": "<string or null>",
      "amount": "<string or null>",
      "low_confidence": true
    }
  ],
  "subtotal": "<string or null>",
  "tax_total": "<string or null>",
  "discount": "<string or null>",
  "freight_charges": "<string or null>",
  "grand_total": "<string or null>",
  "amount_in_words": "<string or null>",
  "additional_fields": {
    "<Label from document>": "<value>",
    "...": "..."
  }
}

═══ STRICT PROHIBITIONS ═══

- DO NOT calculate or verify totals / subtotals.
- DO NOT invent values for missing fields.
- DO NOT change number formatting (commas, decimals, leading zeros).
- DO NOT merge or split line items beyond what is in the OCR output.
- DO NOT truncate or summarize descriptions — include the FULL text.
- DO NOT add commentary or explanations.
"""

USER_PROMPT = (
    "Below is the OCR output (JSON) from a financial document. "
    "Extract all financially relevant information according to your instructions and return a single valid JSON object.\n\n"
    "OCR OUTPUT:\n"
)


def extract_from_ocr(ocr_json_str: str) -> str:
    """Send OCR output to the extraction agent and return the response."""
    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT + ocr_json_str},
            ],
            temperature=1.0,
        )
        _log_token_usage(completion=completion, ocr_payload_chars=len(ocr_json_str))
        return completion.choices[0].message.content or ""
    except Exception as e:
        return json.dumps(
            {"error": "extraction_failed", "message": str(e)},
            ensure_ascii=False,
        )


def _maybe_parse_json(text: str) -> object:
    try:
        return json.loads(text)
    except Exception:
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract invoice fields from OCR output JSON")
    parser.add_argument("--input", help="Path to the OCR output JSON file")
    parser.add_argument("--output", "-o", default=None, help="Path to save extraction result JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"OCR output file not found: {input_path}")

    ocr_json_str = input_path.read_text(encoding="utf-8")

    print(f"Sending {input_path.name} to extraction agent...")
    raw_result = extract_from_ocr(ocr_json_str)
    parsed = _maybe_parse_json(raw_result)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = input_path.parent.parent / "extraction_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_extracted.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print(f"Extraction saved to: {output_path}")


if __name__ == "__main__":
    main()
