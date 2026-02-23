"""LLM-based document classifier. Reads OCR JSON and returns a document type label."""

from agents import client, DEPLOYMENT

CLASSIFIER_PROMPT = """
You are a document classifier. You receive OCR output (JSON) from a scanned financial document.

Your ONLY task is to classify the document into exactly ONE of these categories:

- "commercial_invoice"   : Product/goods invoices with barcodes, PO numbers, shipping terms, product line items
- "travel"               : Flight tickets, travel agency invoices, itineraries with passenger/routing info
- "rental"               : Rental/lease invoices, mall rent, service charges, tenancy billing
- "hotel"                : Hotel folios/invoices, room charges, accommodation bills
- "utility"              : Utility bills (electricity, water, gas) and telecom bills (phone, internet)
- "soa"                  : Statements of account, outstanding balance summaries, aging reports
- "bank_statement"       : Bank account statements with transaction listings
- "credit_note"          : Credit notes / refund documents
- "unknown"              : Does not fit any category above

CLASSIFICATION HINTS:
- If you see barcodes, PO numbers, trade terms (CFR, FOB), bill of lading → "commercial_invoice"
- If you see flight numbers, routing, passenger names, ticket numbers → "travel"
- If you see base rent, service charge, tenancy period, lot/unit number → "rental"
- If you see room charge, check-in/check-out, folio, guest name → "hotel"
- If you see meter reading, tariff, kWh, billing period, subscriber → "utility"
- If you see list of invoices with aging (30/60/90 days), outstanding balance → "soa"
- If you see bank name, account transactions, debit/credit columns, running balance → "bank_statement"
- If you see "Credit Note" or "CN" in the title → "credit_note"
- Travel agency invoices for hotel bookings (with hotel name, check-in, room type but issued by a travel agent) → "travel"

Return ONLY the category label as a single word. No JSON. No explanation.
"""


def classify_document(ocr_json_str: str) -> str:
    """Classify the OCR output into a document type. Returns the category label string."""
    try:
        completion = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Classify this document. Return ONLY the category label.\n\n"
                        "OCR OUTPUT:\n" + ocr_json_str[:8000]  # Truncate to save tokens — headers are enough
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=20,
        )
        raw = (completion.choices[0].message.content or "").strip().lower().strip('"').strip("'")
        # Normalize
        valid = {
            "commercial_invoice", "travel", "rental", "hotel",
            "utility", "soa", "bank_statement", "credit_note", "unknown",
        }
        return raw if raw in valid else "unknown"
    except Exception:
        return "unknown"
