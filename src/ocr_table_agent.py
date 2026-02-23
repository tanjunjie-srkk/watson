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
You are a document-structure analysis engine.

Your task is to:
1. Perform OCR on the provided page image.
2. Detect all tables on the page. Important: Tables may be split across pages.
3. Assign a stable table_id to each table.
4. Detect whether a table continues from the previous page.
5. Merge continued tables logically.

Rules:
- A table has at least 2 rows and 2 columns with aligned structure.
- Tables may span multiple pages.
- If a table on this page has:
  - the same column structure AND
  - similar data patterns AND
  - no new title,
  it MUST be treated as a continuation of the previous table.
- Column headers may appear on an earlier page than the data.
- If a page contains aligned numeric rows but no headers:
  - You MUST search previous pages for compatible column headers.
  - If column count and numeric patterns match, inherit those headers.
- Treat inherited-header tables as a single logical table.
- Never assume a table is new unless a new title or header row appears.
- Never invent rows or columns.

Output must be valid JSON only.
No explanations.

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
        '''Analyze this page image. Analyze ONLY tables. You must analyze all the tables in this images

            Return JSON with:
            {
              "page_number": <number>,
              "tables": [
                {
                  "table_id": "...",
                  "table_title": "... or null",
                  "is_continuation": true | false,
                  "columns": [...],
                  "rows": [
                    [...],
                    [...]
                  ]
                }
              ],
              "non_table_text": "..."
            }
            Return a SINGLE valid JSON object (no markdown).'''
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

  #memo_dir = Path(__file__).resolve().parents[1] / "memo"
  memo_dir = Path(r"C:\Users\TanJunJie\OneDrive - SRKK Group\Project\Avaland\memo")
  image_paths = sorted(
    [p for p in memo_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
  )
  if not image_paths:
    raise RuntimeError(f"No images found in: {memo_dir}")

  user_prompt = (
   '''You are a TABLE-ONLY OCR and DOCUMENT STRUCTURE engine.

Your role is STRICTLY LIMITED to tables extraction.
You are NOT a reasoning, calculation, or rule interpretation system.

CORE TASKS:
1. Detect all tables in the document.
2. Extract tables row-by-row with exact text preservation.
3. Handle tables split across pages using header inheritance.
4. Assign a stable semantic table_id for each logical table.

TABLE ID RULES:
- table_id must be derived from table title, column names, and first data row pattern.
- The same logical table across pages MUST reuse the same table_id.

STRICT PROHIBITIONS:
- DO NOT calculate, normalize, or convert numbers.
- DO NOT infer missing values.
- DO NOT interpret rules or business logic.
- DO NOT merge unrelated tables.

HEADER INHERITANCE RULES:
- If headers appear on an earlier page, inherit them.
- If headers are inferred, mark table_status as "partial".

OUTPUT REQUIREMENTS:
Return valid JSON only.

Each table must include:
- table_id
- table_title
- table_purpose (commission | pricing | rebate | financial_summary | unknown)
- table_status (complete | partial | unreliable)
- table_confidence (0.00–1.00)
- is_continuation
- columns
- rows (raw text only)
- normalized_rows (nullable; do NOT infer if ambiguous)

FAIL-SAFE:
- If a table is partially unreadable, mark it as unreliable.
- Never invent rows, columns, or values.

            '''
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

  with open("../artifact/output-table.json", "w", encoding="utf-8") as f:
    json.dump(output_obj, f, ensure_ascii=False, indent=2)

  #print(json.dumps({"results": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()