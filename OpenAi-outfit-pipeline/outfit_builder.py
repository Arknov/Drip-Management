import json
import boto3
import os
import logging
import uuid
import urllib.request
import urllib.error
from decimal import Decimal
from datetime import datetime
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── ENV ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

WARDROBE_TABLE = os.environ["WARDROBE_TABLE"]
OUTFITS_TABLE  = os.environ["OUTFITS_TABLE"]

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# ── AWS ───────────────────────────────────────────────────────────────────────
dynamodb       = boto3.resource("dynamodb")
wardrobe_table = dynamodb.Table(WARDROBE_TABLE)
outfits_table  = dynamodb.Table(OUTFITS_TABLE)


# ── 1. FETCH WARDROBE ─────────────────────────────────────────────────────────

def get_wardrobe(user_id: str):
    response = wardrobe_table.query(
        KeyConditionExpression=Key("userId").eq(user_id)
    )
    items = response.get("Items", [])
    valid = [i for i in items if i.get("status") == "success"]
    logger.info(f"Found {len(valid)} valid wardrobe items for user {user_id}")
    return valid


# ── 2. BUILD PROMPT ───────────────────────────────────────────────────────────

def build_prompt(items: list):
    tagged = []
    for item in items:
        labels   = item.get("cleanedLabels", {})
        detected = item.get("detectedItems", [])

        colors = []
        for di in detected:
            colors.extend(di.get("colors", []))
        if not colors:
            colors = labels.get("colors", [])

        tagged.append({
            "id":        item.get("itemId"),
            "types":     labels.get("itemTypes", []),
            "colors":    list(set(colors)),
            "materials": labels.get("materials", []),
            "style":     labels.get("style", []),
            "fit":       labels.get("fit", []),
            "patterns":  labels.get("patterns", []),
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
    "item_ids": ["exact-uuid-from-wardrobe", "exact-uuid-from-wardrobe"],
    "style": "formal",
    "reasoning": "brief explanation",
    "confidence": 0.85
  }},
  {{
    "outfit_id": "o3",
    "item_ids": ["exact-uuid-from-wardrobe", "exact-uuid-from-wardrobe"],
    "style": "streetwear",
    "reasoning": "brief explanation",
    "confidence": 0.8
  }}
]"""


# ── 3. CALL OPENAI ────────────────────────────────────────────────────────────

def call_openai(prompt: str):
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENAI_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            logger.info(f"OpenAI response (first 500 chars): {text[:500]}")
            return text
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise ValueError(f"OpenAI API error {e.code}: {error_body}")


# ── 4. PARSE RESPONSE ─────────────────────────────────────────────────────────

def parse_outfits(text: str, valid_ids: set):
    text = text.strip()

    # Strip markdown fences if model adds them
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    start = text.find("[")
    end   = text.rfind("]") + 1

    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in response: {text[:300]}")

    outfits = json.loads(text[start:end])

    cleaned = []
    for outfit in outfits:
        valid_items = [i for i in outfit.get("item_ids", []) if i in valid_ids]
        if not valid_items:
            logger.warning(f"Outfit {outfit.get('outfit_id')} has no valid item ids — skipping")
            continue
        outfit["item_ids"] = valid_items
        cleaned.append(outfit)

    if not cleaned:
        raise ValueError("No valid outfits after id validation")

    logger.info(f"Parsed {len(cleaned)} valid outfits")
    return cleaned


# ── 5. ENRICH OUTFITS ─────────────────────────────────────────────────────────

def enrich_outfits(outfits: list, wardrobe_map: dict):
    enriched = []
    for outfit in outfits:
        items_detail = []
        for item_id in outfit.get("item_ids", []):
            item = wardrobe_map.get(item_id)
            if item:
                labels = item.get("cleanedLabels", {})
                items_detail.append({
                    "itemId":    item_id,
                    "imageUrl":  item.get("imageUrl", ""),
                    "itemTypes": labels.get("itemTypes", []),
                    "colors":    labels.get("colors", []),
                    "materials": labels.get("materials", []),
                })
        outfit["items_detail"] = items_detail
        enriched.append(outfit)
    return enriched


# ── 6. SAVE OUTFITS ───────────────────────────────────────────────────────────

def floats_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, list):
        return [floats_to_decimal(i) for i in obj]
    if isinstance(obj, dict):
        return {k: floats_to_decimal(v) for k, v in obj.items()}
    return obj


def save_outfits(user_id: str, outfits: list):
    outfits_table.put_item(Item={
        "userId":      user_id,
        "outfitSetId": str(uuid.uuid4()),
        "outfits":     floats_to_decimal(outfits),
        "generatedAt": datetime.utcnow().isoformat() + "Z",
    })
    logger.info(f"Saved {len(outfits)} outfits for user {user_id}")


# ── 7. LAMBDA HANDLER ─────────────────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        user_id = event.get("userId")
        if not user_id:
            return {"status": "error", "message": "missing userId"}

        # Fetch wardrobe
        wardrobe = get_wardrobe(user_id)
        if not wardrobe:
            return {"status": "error", "message": "no wardrobe items found for user"}

        wardrobe_map = {item["itemId"]: item for item in wardrobe}
        valid_ids    = set(wardrobe_map.keys())

        # Build prompt
        prompt = build_prompt(wardrobe)

        # Call OpenAI
        raw_response = call_openai(prompt)

        # Parse + validate
        outfits = parse_outfits(raw_response, valid_ids)

        # Enrich
        outfits = enrich_outfits(outfits, wardrobe_map)

        # Save
        save_outfits(user_id, outfits)

        return {
            "status":      "success",
            "userId":      user_id,
            "outfitCount": len(outfits),
            "outfits":     outfits,
        }

    except Exception as e:
        logger.error(f"Lambda failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}