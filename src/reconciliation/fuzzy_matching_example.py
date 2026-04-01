"""
Fuzzy Matching — Simple Explained Example
==========================================
This script demonstrates how fuzzy matching works using rapidfuzz.

Fuzzy matching answers the question:
  "How SIMILAR are two strings?"  (0 = totally different, 100 = identical)

It is useful when data is messy — typos, abbreviations, extra words, etc.
"""

from rapidfuzz import fuzz

# ===================================================================
# 1. BASIC RATIO — overall similarity of two full strings
# ===================================================================
print("=" * 60)
print("1. BASIC RATIO  (compare two full strings)")
print("=" * 60)

pairs = [
    ("Shopee Seller Payout", "Shopee Seller Payout"),   # identical
    ("Shopee Seller Payout", "shopee seller payout"),   # case difference
    ("Shopee Seller Payout", "Shopee Seler Payuot"),    # typos
    ("Shopee Seller Payout", "General Transfer"),        # completely different
]

for a, b in pairs:
    score = fuzz.ratio(a, b)
    print(f'  "{a}"  vs  "{b}"')
    print(f'  → Score: {score}\n')


# ===================================================================
# 2. PARTIAL RATIO — find best substring match
# ===================================================================
print("=" * 60)
print("2. PARTIAL RATIO  (does one string CONTAIN the other?)")
print("=" * 60)

pairs_partial = [
    ("Shopee", "Shopee Seller Payout"),            # keyword inside longer text
    ("Shopee", "SPMY WD - Shopee Settlement"),     # keyword buried in text
    ("Shopee", "General Bank Transfer"),            # keyword absent
    ("Shopee", "Shoppee Payout"),                   # close but misspelled
]

for a, b in pairs_partial:
    score = fuzz.partial_ratio(a, b)
    print(f'  Search "{a}"  in  "{b}"')
    print(f'  → Score: {score}\n')


# ===================================================================
# 3. TOKEN SORT RATIO — ignore word ORDER
# ===================================================================
print("=" * 60)
print("3. TOKEN SORT RATIO  (same words, different order)")
print("=" * 60)

pairs_sort = [
    ("Seller Shopee Payout", "Shopee Seller Payout"),  # reordered
    ("Payout Shopee", "Shopee Payout"),                 # reordered
    ("Shopee Payout", "Amazon Payout"),                 # different word
]

for a, b in pairs_sort:
    score = fuzz.token_sort_ratio(a, b)
    print(f'  "{a}"  vs  "{b}"')
    print(f'  → Score: {score}\n')


# ===================================================================
# 4. REAL-WORLD USE CASE — match bank descriptions
# ===================================================================
print("=" * 60)
print("4. REAL-WORLD EXAMPLE — Which bank entry is from Shopee?")
print("=" * 60)

bank_descriptions = [
    "Shopee Seller Payout",
    "SPMY WD",
    "General Transfer",
    "Shoppee Settlement",
    "Merchant Payment",
    "Shopee MY",
]

keyword = "Shopee"
print(f'  Keyword: "{keyword}"\n')

for desc in bank_descriptions:
    score = fuzz.partial_ratio(keyword.lower(), desc.lower())
    label = "✅ MATCH" if score >= 70 else "❌ NO MATCH"
    print(f'  "{desc}"  →  score {score:5.1f}  {label}')

print()
print("=" * 60)
print("HOW IT WORKS UNDER THE HOOD (simplified)")
print("=" * 60)
print("""
  Fuzzy matching uses the Levenshtein distance algorithm.
  It counts the MINIMUM number of edits (insert, delete, replace)
  needed to turn one string into another.

  Example:
    "Shopee"  →  "Shoppee"
     Step 1: insert extra 'p'   →  1 edit

    Similarity = 1 - (edits / max_length)
               = 1 - (1 / 7) ≈ 0.86  →  86%

  partial_ratio goes further — it slides the shorter string
  across the longer one and picks the best-matching window:

    "Shopee"  in  "SPMY WD - Shopee Settlement"
     Window slides: "SPMY W" → "PMY WD" → … → "Shopee" → exact! → 100

  This is why partial_ratio is great for finding keywords
  buried inside longer descriptions.
""")
