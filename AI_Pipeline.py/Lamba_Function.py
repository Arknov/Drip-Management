import json
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Initialize Gemini client
client = genai.Client(api_key="AIzaSyDTJGqqbuixU0wCn0Qd_xVlTSjKyjNyLkQ")

MODEL_ID = "gemini-1.5-flash"


# -----------------------------
# 1. NORMALIZE INPUT
# -----------------------------
def normalize_wardrobe(dynamo_items: list) -> list:
    normalized = []

    for i, item in enumerate(dynamo_items):
        try:
            normalized.append({
                "item_id": f"item_{i}",
                "type":    (item[0] or "unknown").lower(),
                "color":   (item[1] or "unknown").lower(),
            })
        except Exception:
            logger.warning(f"Skipping bad item: {item}")

    return normalized


# -----------------------------
# 2. BUILD PROMPT
# -----------------------------
def build_prompt(items: list, preferences: dict) -> str:
    return f"""
You are an expert fashion stylist.

Each item includes:
- type (e.g. shirt, jeans, sneakers)
- color

WARDROBE:
{json.dumps(items, indent=2)}

USER PREFERENCES:
- styles: {preferences.get("styles", [])}
- colors: {preferences.get("colors", [])}
- fit: {preferences.get("fit", "any")}
- occasion: {preferences.get("occasion", "casual")}

TASK:
Generate 3 cohesive outfits using ONLY the given items.

RULES:
- Colors must coordinate (neutral base preferred)
- Outfits must be realistic (top + bottom minimum)
- Base your reasoning on type and color only

RETURN ONLY VALID JSON.
NO markdown, NO explanation, NO code fences.

FORMAT:
[
  {{
    "outfit_id": "o1",
    "items": ["item_0", "item_2"],
    "style": "casual",
    "reasoning": "Explain briefly based on type and color",
    "confidence_score": 0.9
  }}
]
"""


# -----------------------------
# 3. CALL LLM (Gemini)
# -----------------------------
def call_llm(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=500,
        )
    )
    return response.text


# -----------------------------
# 4. EXTRACT JSON SAFELY
# -----------------------------
def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        start = text.find("[")
        end   = text.rfind("]") + 1
        return json.loads(text[start:end])
    except Exception:
        logger.error("Failed to parse JSON")
        return None


# -----------------------------
# 5. VALIDATE OUTPUT
# -----------------------------
def validate_outfits(outfits: list, items: list) -> list:
    valid_ids   = {item["item_id"] for item in items}
    type_lookup = {item["item_id"]: item["type"] for item in items}

    TOPS    = {"shirt", "top", "blouse", "sweater", "hoodie", "jacket", "coat"}
    BOTTOMS = {"pants", "jeans", "trousers", "shorts", "skirt", "dress", "bottom"}

    cleaned = []

    for outfit in outfits:
        if not isinstance(outfit, dict):
            continue
        if "items" not in outfit:
            continue
        if not all(i in valid_ids for i in outfit["items"]):
            logger.warning(f"Outfit {outfit.get('outfit_id')} has invalid item_ids — dropping")
            continue

        outfit_types = {type_lookup.get(i, "") for i in outfit["items"]}
        has_top    = bool(outfit_types & TOPS)
        has_bottom = bool(outfit_types & BOTTOMS)

        if not has_top or not has_bottom:
            logger.warning(f"Outfit {outfit.get('outfit_id')} missing top or bottom — dropping")
            continue

        cleaned.append(outfit)

    return cleaned


# -----------------------------
# 6. MAIN PIPELINE
# -----------------------------
def generate_outfits(dynamo_items: list, preferences: dict) -> list:
    wardrobe = normalize_wardrobe(dynamo_items)

    if not wardrobe:
        return []

    try:
        prompt  = build_prompt(wardrobe, preferences)
        raw     = call_llm(prompt)

        outfits = extract_json(raw)
        if not outfits:
            return []

        return validate_outfits(outfits, wardrobe)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return []