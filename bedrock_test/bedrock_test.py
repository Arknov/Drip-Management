import json
import boto3
import os
import logging
from boto3.dynamodb.conditions import Key
import google.generativeai as genai

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── ENV ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

WARDROBE_TABLE = os.environ["WARDROBE_TABLE"]   # Same table your Rekognition Lambda writes to
OUTFITS_TABLE  = os.environ["OUTFITS_TABLE"]    # Separate table for outfit results

# ── AWS ───────────────────────────────────────────────────────────────────────
dynamodb = boto3.resource("dynamodb")
wardrobe_table = dynamodb.Table(WARDROBE_TABLE)
outfits_table  = dynamodb.Table(OUTFITS_TABLE)

# ── GEMINI ────────────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


# ── 1. FETCH WARDROBE ─────────────────────────────────────────────────────────

def get_wardrobe(user_id: str):
    """
    Query all clothing items for a user from DynamoDB.
    Table key: userId (PK) + itemId (SK) — matches Arnav's Rekognition pipeline output.
    Only returns items with status=success.
    """
    response = wardrobe_table.query(
        KeyConditionExpression=Key("userId").eq(user_id)
    )
    items = response.get("Items", [])

    # Filter to only successfully processed items
    valid = [i for i in items if i.get("status") == "success"]
    logger.info(f"Found {len(valid)} valid wardrobe items for user {user_id}")
    return valid


# ── 2. BUILD PROMPT ───────────────────────────────────────────────────────────

def build_prompt(items: list):
    """
    Format wardrobe items for Gemini using actual itemIds so
    outfit references can be resolved back to real DynamoDB records.
    """
    tagged = []
    for item in items:
        labels = item.get("cleanedLabels", {})
        detected = item.get("detectedItems", [])

        # Build per-item color list — prefer detailed detectedItems colors
        colors = []
        for di in detected:
            colors.extend(di.get("colors", []))
        if not colors:
            colors = labels.get("colors", [])

        tagged.append({
            "id": item.get("itemId"),
            "types": labels.get("itemTypes", []),
            "colors": list(set(colors)),
            "materials": labels.get("materials", []),
            "style": labels.get("style", []),
            "fit": labels.get("fit", []),
            "patterns": labels.get("patterns", []),
        })

    return f"""You are a world-class fashion stylist AI.

Given the wardrobe below, generate 3 complete outfit combinations.

RULES:
- Use ONLY the item ids exactly as provided — do not invent ids
- Each outfit must include at least 2 items
- Ensure color harmony across the outfit
- Prefer realistic, wearable combinations
- No two outfits should be identical
- If wardrobe is limited, it is OK to reuse items across outfits

WARDROBE:
{json.dumps(tagged, indent=2)}

RETURN STRICT JSON ONLY — no markdown, no backticks, no explanation outside the JSON:

[
  {{
    "outfit_id": "o1",
    "item_ids": ["exact-uuid-from-wardrobe", "exact-uuid-from-wardrobe"],
    "style": "casual",
    "reasoning": "brief explanation of why this outfit works",
    "confidence": 0.9
  }},
  {{
    "outfit_id": "o2",
    "item_ids": ["exact-uuid-from-wardrobe"],
    "style": "formal",
    "reasoning": "brief explanation",
    "confidence": 0.85
  }},
  {{
    "outfit_id": "o3",
    "item_ids": ["exact-uuid-from-wardrobe"],
    "style": "streetwear",
    "reasoning": "brief explanation",
    "confidence": 0.8
  }}
]"""


# ── 3. CALL GEMINI ────────────────────────────────────────────────────────────

def call_gemini(prompt: str):
    """Call Gemini and return raw text response."""
    logger.info("Calling Gemini...")
    response = model.generate_content(prompt)
    raw = response.text
    logger.info(f"Gemini raw response (first 500 chars): {raw[:500]}")
    return raw


# ── 4. PARSE RESPONSE ─────────────────────────────────────────────────────────

def parse_outfits(text: str, valid_ids: set):
    """
    Parse Gemini JSON response and validate item_ids against real wardrobe.
    Raises on parse failure so Lambda retries rather than silently saving garbage.
    """
    # Strip markdown code fences if Gemini adds them despite instructions
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    start = text.find("[")
    end = text.rfind("]") + 1

    if start == -1 or end == 0:
        raise ValueError(f"No JSON array found in Gemini response: {text[:300]}")

    outfits = json.loads(text[start:end])

    # Validate and clean item_ids — remove any hallucinated ids
    cleaned = []
    for outfit in outfits:
        valid_items = [i for i in outfit.get("item_ids", []) if i in valid_ids]
        if len(valid_items) < 1:
            logger.warning(f"Outfit {outfit.get('outfit_id')} has no valid item ids — skipping")
            continue
        outfit["item_ids"] = valid_items
        cleaned.append(outfit)

    if not cleaned:
        raise ValueError("Gemini returned no valid outfits after id validation")

    logger.info(f"Parsed {len(cleaned)} valid outfits")
    return cleaned


# ── 5. ENRICH OUTFITS ─────────────────────────────────────────────────────────

def enrich_outfits(outfits: list, wardrobe_map: dict):
    """
    Replace item_ids with full item detail so the frontend
    doesn't need to make separate DynamoDB lookups per item.
    """
    enriched = []
    for outfit in outfits:
        items_detail = []
        for item_id in outfit.get("item_ids", []):
            item = wardrobe_map.get(item_id)
            if item:
                labels = item.get("cleanedLabels", {})
                items_detail.append({
                    "itemId": item_id,
                    "imageUrl": item.get("imageUrl", ""),
                    "itemTypes": labels.get("itemTypes", []),
                    "colors": labels.get("colors", []),
                    "materials": labels.get("materials", []),
                })
        outfit["items_detail"] = items_detail
        enriched.append(outfit)
    return enriched


# ── 6. SAVE OUTFITS ───────────────────────────────────────────────────────────

def save_outfits(user_id: str, outfits: list):
    """Write outfits to the outfits table."""
    import uuid
    from datetime import datetime

    outfits_table.put_item(Item={
        "userId": user_id,
        "outfitSetId": str(uuid.uuid4()),
        "outfits": outfits,
        "generatedAt": datetime.utcnow().isoformat() + "Z",
    })
    logger.info(f"Saved {len(outfits)} outfits for user {user_id}")


# ── 7. LAMBDA HANDLER ─────────────────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        user_id = event.get("userId")
        if not user_id:
            return {"status": "error", "message": "missing userId"}

        # Step 1: fetch wardrobe
        wardrobe = get_wardrobe(user_id)
        if not wardrobe:
            return {"status": "error", "message": "no wardrobe items found for user"}

        # Build a map for fast lookup during enrichment and validation
        wardrobe_map = {item["itemId"]: item for item in wardrobe}
        valid_ids = set(wardrobe_map.keys())

        # Step 2: build prompt
        prompt = build_prompt(wardrobe)

        # Step 3: call Gemini
        raw_response = call_gemini(prompt)

        # Step 4: parse + validate ids
        outfits = parse_outfits(raw_response, valid_ids)

        # Step 5: enrich with item detail
        outfits = enrich_outfits(outfits, wardrobe_map)

        # Step 6: save
        save_outfits(user_id, outfits)

        return {
            "status": "success",
            "userId": user_id,
            "outfitCount": len(outfits),
            "outfits": outfits,
        }

    except Exception as e:
        logger.error(f"Lambda failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}