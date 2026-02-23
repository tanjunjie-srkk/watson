import base64
import argparse
import json
import mimetypes
import os
from pathlib import Path

from openai import AzureOpenAI, OpenAI

def _get_required_env(name: str) -> str:
  value = os.getenv(name)
  if not value:
    raise RuntimeError(f"Missing required environment variable: {name}")
  return value


endpoint = _get_required_env("AZURE_OPENAI_ENDPOINT")
model_name = "gpt-5.2-chat"
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")

subscription_key = _get_required_env("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


def _image_file_to_data_url(image_path: Path) -> str:
  mime_type, _ = mimetypes.guess_type(str(image_path))
  if not mime_type:
    mime_type = "application/octet-stream"
  b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
  return f"data:{mime_type};base64,{b64}"

SYSTEM_PROMPT = """
You are an OCR transcription engine. Your ONLY task is to read every visible character from the provided document image(s) and output a structured JSON transcription.

You are NOT an assistant, analyst, or calculator. You do NOT interpret, summarize, compute, or infer anything.

═══ TRANSCRIPTION RULES ═══

1. COPY EXACTLY what you see: every letter, digit, punctuation mark, symbol, and whitespace.
2. NEVER change, correct, round, recalculate, or reformat any value.
   - If the document says "30,752.88", output "30,752.88" — not "30752.88".
   - If a word is misspelled on the document, reproduce the misspelling.
3. PRESERVE the spatial/logical grouping of text:
   - Key-value pairs (e.g. "CI Number : 26001099") → keep label and value together.
   - Tables → preserve column headers and row alignment.
   - Addresses, notes, footers → keep as contiguous blocks.
4. READ EVERY PAGE. Do NOT skip repeated headers, footers, or boilerplate.
5. If multiple images are provided, treat them as pages of ONE document in the order given.

═══ CONFIDENCE TAGGING ═══

- Assign a confidence score (0.00–1.00) to each section.
- Set confidence = 1.00 ONLY when every character in that section is clearly legible.
- If a character or word is ambiguous, output your best reading and LOWER the confidence.
- If text is completely unreadable, output "[UNREADABLE]" with confidence 0.00.
- If content is visibly cut off at the page edge, output what is visible and append "[PARTIAL]".

═══ STRICT PROHIBITIONS ═══

- DO NOT add, remove, or alter any text that is not on the document.
- DO NOT calculate totals, subtotals, percentages, or derived values.
- DO NOT merge or reconcile data across pages or tables.
- DO NOT infer missing values or fill blanks.
- DO NOT rewrite table data into sentences or summaries.
- DO NOT add any explanation, commentary, or markdown.

═══ OUTPUT SCHEMA ═══

Return ONLY a single valid JSON object. No markdown fences. No text before or after.

{
  "pages": [
    {
      "page_number": <int>,
      "file_name": "<filename if provided>",
      "sections": [
        {
          "type": "<section_type>",
          "content": "<exact transcribed text>",
          "confidence": <float 0.00-1.00>
        }
      ]
    }
  ],
  "metadata": {
    "total_pages": <int>,
    "languages_detected": ["en"],
    "image_quality": "clear | noisy | blurry | low_resolution"
  }
}

Allowed section types:
- "header"       : document titles, company names, page labels (e.g. "Page 1 of 3")
- "address"      : address blocks (bill-to, ship-to, business unit)
- "key_value"    : label-value pairs (e.g. "PO Number : MY2501539")
- "table_header" : column header row of a table
- "table_row"    : one data row of a table, values separated by " | "
- "subtotal"     : subtotal / total / summary lines
- "paragraph"    : free-form text, notes, instructions
- "footer"       : page footers, disclaimers, correspondence addresses
- "signature"    : signature blocks, stamps, seals
- "empty"        : page has no extractable text (include reason in content)

═══ TABLE TRANSCRIPTION RULES ═══

- Output the column header row as one section with type "table_header", values separated by " | ".
- Output each data row as a separate section with type "table_row", values separated by " | ".
- Preserve the column order exactly as it appears on the document.
- If a cell is empty, output an empty string between the delimiters (e.g. "value1 |  | value3").
- DO NOT infer column meanings, merge rows, or reorder columns.

CRITICAL — MULTI-LINE PRODUCT DESCRIPTIONS:
- A single product row often has MULTIPLE lines of text in the description column:
    Line 1: Main product name (e.g. "WATSONS SIDE SEALED COTTON PUFFS")
    Line 2: Product variant/spec (e.g. "WATSONS SIDE SEALED COTTON PUFFS 189S SEA AW2022")
    Line 3: Campaign/collection name (e.g. "WATSONS OOB COTTON RELAUNCH")
- ALL of these lines belong to the SAME product row.
- Combine them into ONE table_row section, joining description sub-lines with \n.
- The key indicator that lines belong to the same row: they share the SAME barcode, quantity, unit price, and amount on the right side.
- If description text appears on lines below the barcode/quantity/price row, and those lines have NO barcode, NO quantity, NO unit price, and NO amount of their own, they are sub-descriptions of the row above — merge them into that row.
- NEVER output a sub-description line as a separate table_row.
"""


def ocr_image_with_chat_model(image_path: Path, user_prompt: str) -> str:
  data_url = _image_file_to_data_url(image_path)

  try:
    completion = client.chat.completions.create(
      model=deployment,
      messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
          "role": "user",
          "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
          ],
        },
      ],
      temperature=1.0,
    )
    return completion.choices[0].message.content or ""
  except Exception as e:
    # Common cause: Azure content filter flags the *prompt* (often when referencing system messages).
    return json.dumps(
      {
        "error": "ocr_failed",
        "message": str(e),
      },
      ensure_ascii=False,
    )


def ocr_images_with_chat_model(image_paths: list[Path], user_prompt: str) -> str:
  content: list[dict] = [
    {
      "type": "text",
      "text": (
        f"{user_prompt}\n\n"
        "You will receive multiple images. Treat them as pages of ONE single document in the order given. "
        "Include one entry per image in pages[], preserving the order. "
        "Set page_number starting from 1. Include the filename in a field named file_name. "
        "Do NOT skip any page, even if its content repeats a previous page."
      ),
    }
  ]

  for idx, image_path in enumerate(image_paths, start=1):
    content.append({"type": "text", "text": f"Image {idx} filename: {image_path.name}"})
    content.append({"type": "image_url", "image_url": {"url": _image_file_to_data_url(image_path)}})

  try:
    completion = client.chat.completions.create(
      model=deployment,
      messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
      ],
      temperature=1.0,
    )
    return completion.choices[0].message.content or ""
  except Exception as e:
    return json.dumps(
      {
        "error": "ocr_failed",
        "message": str(e),
      },
      ensure_ascii=False,
    )


def _maybe_parse_json(text: str) -> object:
  try:
    return json.loads(text)
  except Exception:
    return text


def main() -> None:
  parser = argparse.ArgumentParser(description="OCR images in memo folder")
  parser.add_argument(
    "--batch",
    action="store_true",
    help="Send all images in ONE request (may hit context limits for many/large images).",
  )
  args = parser.parse_args()

  memo_dir = Path(__file__).resolve().parents[1] / "src/docs/Utility_5_images"
  image_paths = sorted(
    [p for p in memo_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
  )
  if not image_paths:
    raise RuntimeError(f"No images found in: {memo_dir}")

  user_prompt = (
    "Transcribe ALL visible text from this document image exactly as it appears. "
    "Output the result as a single valid JSON object following the schema in your instructions. "
    "Do NOT interpret, summarize, or calculate anything. "
    "For every section, set a realistic confidence score — lower it if any character is uncertain. "
    "Preserve all numbers, punctuation, and formatting exactly."
  )

  if args.batch:
    content = ocr_images_with_chat_model(image_paths=image_paths, user_prompt=user_prompt)
    output_obj = {
      "mode": "batch",
      "files": [p.name for p in image_paths],
      "model_output": _maybe_parse_json(content),
    }
  else:
    outputs: list[dict] = []
    for idx, image_path in enumerate(image_paths, start=1):
      content = ocr_image_with_chat_model(image_path=image_path, user_prompt=user_prompt)
      outputs.append(
        {
          "page_number": idx,
          "file": image_path.name,
          "model_output": _maybe_parse_json(content),
        }
      )
    output_obj = {"mode": "per_image", "results": outputs}

  with open("./ocr_output/Utility5.json", "w", encoding="utf-8") as f:
    json.dump(output_obj, f, ensure_ascii=False, indent=2)

  print(json.dumps({"results": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()