"""
outfit_mockup_generator.py
──────────────────────────
Standalone Lambda that:
  1. Reads a user's outfit set from the 'Outfits' DynamoDB table
     (written by outfit_builder.py)
  2. Builds a DALL-E 3 prompt from the exact fields outfit_builder stored:
     style, reasoning, items_detail → itemTypes, colors, materials
  3. Generates a flat-lay mockup image per outfit
  4. Uploads each image to S3
  5. Writes the S3 URLs back into the same DynamoDB record

Required env vars:
  OUTFITS_TABLE   – DynamoDB table name (should be 'Outfits')
  MOCKUP_BUCKET   – S3 bucket to store generated images
  OPENAI_API_KEY  – OpenAI secret key

Event payload:
  {
    "userId":      "user-abc-123",
    "outfitSetId": "uuid-of-the-set"   ← optional; omit to use the latest set
  }

Response:
  {
    "status": "success",
    "userId": "...",
    "outfitSetId": "...",
    "mockups": [
      {
        "outfit_id": "o1",
        "style": "casual",
        "mockup_image_url": "https://your-bucket.s3.amazonaws.com/..."
      },
      ...
    ]
  }
"""

import json
import boto3
import os
import logging
import uuid
import urllib.request
import urllib.error
from decimal import Decimal
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── ENV ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

OUTFITS_TABLE = os.environ["OUTFITS_TABLE"]   # 'Outfits'
MOCKUP_BUCKET = os.environ["MOCKUP_BUCKET"]

OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations"

# ── AWS ───────────────────────────────────────────────────────────────────────
dynamodb     = boto3.resource("dynamodb")
outfits_table = dynamodb.Table(OUTFITS_TABLE)
s3           = boto3.client("s3")


# ── 1. FETCH OUTFIT SET FROM DYNAMODB ─────────────────────────────────────────
#
# outfit_builder stores records with the shape:
#   {
#     "userId":      "...",
#     "outfitSetId": "...",
#     "generatedAt": "2024-...",
#     "outfits": [
#       {
#         "outfit_id":    "o1",
#         "item_ids":     ["uuid", ...],
#         "style":        "casual",
#         "reasoning":    "why this works",
#         "confidence":   0.9,
#         "items_detail": [
#           {
#             "itemId":    "uuid",
#             "imageUrl":  "https://...",
#             "itemTypes": ["t-shirt"],
#             "colors":    ["white", "navy"],
#             "materials": ["cotton"]
#           },
#           ...
#         ]
#       },
#       ...
#     ]
#   }

def fetch_outfit_set(user_id: str, outfit_set_id: str = None) -> dict:
    """
    If outfit_set_id is provided, fetch that specific record.
    Otherwise query by userId and return the most recently generated set.
    """
    if outfit_set_id:
        response = outfits_table.get_item(
            Key={"userId": user_id, "outfitSetId": outfit_set_id}
        )
        record = response.get("Item")
        if not record:
            raise ValueError(f"No outfit set found for userId={user_id} outfitSetId={outfit_set_id}")
        return record

    # No specific set — query all sets for this user and pick the latest
    response = outfits_table.query(
        KeyConditionExpression=Key("userId").eq(user_id)
    )
    items = response.get("Items", [])
    if not items:
        raise ValueError(f"No outfit sets found for userId={user_id}")

    # Sort by generatedAt descending, pick most recent
    items.sort(key=lambda x: x.get("generatedAt", ""), reverse=True)
    record = items[0]
    logger.info(f"Using latest outfit set {record['outfitSetId']} generated at {record.get('generatedAt')}")
    return record


# ── 2. BUILD DALL-E PROMPT ────────────────────────────────────────────────────
#
# We use every field outfit_builder stored to make the prompt as specific
# as possible:
#   - outfit.style        → overall vibe
#   - outfit.reasoning    → stylist's intent (tells DALL-E *why* items go together)
#   - items_detail[].itemTypes, .colors, .materials → concrete garment descriptions

def build_dalle_prompt(outfit: dict) -> str:
    style     = outfit.get("style", "casual")
    reasoning = outfit.get("reasoning", "")

    # Build one natural-language description per garment
    garment_lines = []
    for item in outfit.get("items_detail", []):
        item_types = ", ".join(item.get("itemTypes", [])) or "clothing item"
        colors     = ", ".join(item.get("colors", []))
        materials  = ", ".join(item.get("materials", []))

        parts = []
        if colors:
            parts.append(colors)
        if materials:
            parts.append(materials)
        parts.append(item_types)

        garment_lines.append(" ".join(parts))

    garments_text = "; ".join(garment_lines) if garment_lines else "stylish clothing"

    # Reasoning gives DALL-E the stylist's intent — keeps the image coherent
    reasoning_clause = f" The stylist's note: {reasoning}." if reasoning else ""

    prompt = (
        f"Professional fashion lookbook flat lay photograph on a pure white background. "
        f"Style: {style}.{reasoning_clause} "
        f"The outfit contains these pieces, neatly arranged together: {garments_text}. "
        f"Top-down view, soft studio lighting, subtle shadows, items slightly overlapping "
        f"to suggest a complete outfit. No mannequin, no people, no text, no watermarks, "
        f"no props. High-end editorial quality."
    )

    logger.info(f"DALL-E prompt for outfit {outfit.get('outfit_id')}: {prompt[:200]}…")
    return prompt


# ── 3. CALL DALL-E 3 ──────────────────────────────────────────────────────────

def call_dalle(prompt: str) -> str:
    """Returns a temporary DALL-E image URL (valid ~1 hour)."""
    payload = json.dumps({
        "model":           "dall-e-3",
        "prompt":          prompt,
        "n":               1,
        "size":            "1024x1024",
        "quality":         "standard",
        "response_format": "url",
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENAI_IMAGE_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["data"][0]["url"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise ValueError(f"DALL-E API error {e.code}: {error_body}")


# ── 4. DOWNLOAD + UPLOAD TO S3 ────────────────────────────────────────────────

def persist_image_to_s3(temp_url: str, user_id: str, outfit_set_id: str, outfit_id: str) -> str:
    """
    Downloads the temporary DALL-E URL and uploads to S3.
    Returns the permanent S3 URL.
    """
    try:
        with urllib.request.urlopen(temp_url, timeout=30) as img_resp:
            image_bytes = img_resp.read()
    except Exception as e:
        raise ValueError(f"Failed to download DALL-E image: {e}")

    s3_key = f"outfit-mockups/{user_id}/{outfit_set_id}/{outfit_id}.png"

    s3.put_object(
        Bucket=MOCKUP_BUCKET,
        Key=s3_key,
        Body=image_bytes,
        ContentType="image/png",
    )

    permanent_url = f"https://{MOCKUP_BUCKET}.s3.amazonaws.com/{s3_key}"
    logger.info(f"Mockup persisted → {permanent_url}")
    return permanent_url


# ── 5. WRITE MOCKUP URLS BACK TO DYNAMODB ────────────────────────────────────
#
# We update the existing record in-place, adding a mockup_image_url field
# to each outfit inside the outfits list.

def decimal_safe(obj):
    """DynamoDB returns Decimals; convert back to float for JSON serialisation."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [decimal_safe(i) for i in obj]
    if isinstance(obj, dict):
        return {k: decimal_safe(v) for k, v in obj.items()}
    return obj

def floats_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, list):
        return [floats_to_decimal(i) for i in obj]
    if isinstance(obj, dict):
        return {k: floats_to_decimal(v) for k, v in obj.items()}
    return obj

def write_mockup_urls_to_dynamo(user_id: str, outfit_set_id: str, outfits: list):
    """
    Replaces the 'outfits' attribute on the existing DynamoDB record with
    the updated list that now includes mockup_image_url on each outfit.
    """
    outfits_table.update_item(
        Key={"userId": user_id, "outfitSetId": outfit_set_id},
        UpdateExpression="SET outfits = :o",
        ExpressionAttributeValues={":o": floats_to_decimal(outfits)},
    )
    logger.info(f"Updated DynamoDB record {outfit_set_id} with mockup URLs")


# ── 6. LAMBDA HANDLER ─────────────────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        user_id      = event.get("userId")
        outfit_set_id = event.get("outfitSetId")   # optional

        if not user_id:
            return {"status": "error", "message": "missing userId"}

        # ── Fetch the outfit set from DynamoDB ────────────────────────────────
        record = fetch_outfit_set(user_id, outfit_set_id)
        outfit_set_id = record["outfitSetId"]
        outfits       = decimal_safe(record.get("outfits", []))

        if not outfits:
            return {"status": "error", "message": "outfit set exists but contains no outfits"}

        logger.info(f"Processing {len(outfits)} outfits in set {outfit_set_id}")

        # ── Generate a mockup for each outfit ────────────────────────────────
        mockup_results = []

        for outfit in outfits:
            outfit_id = outfit.get("outfit_id", str(uuid.uuid4()))

            try:
                prompt    = build_dalle_prompt(outfit)
                temp_url  = call_dalle(prompt)
                final_url = persist_image_to_s3(temp_url, user_id, outfit_set_id, outfit_id)

                outfit["mockup_image_url"] = final_url

                mockup_results.append({
                    "outfit_id":        outfit_id,
                    "style":            outfit.get("style"),
                    "mockup_image_url": final_url,
                })

                logger.info(f"✓ Mockup generated for outfit {outfit_id}")

            except Exception as e:
                # Non-fatal: log and continue with remaining outfits
                logger.error(f"✗ Mockup failed for outfit {outfit_id}: {e}", exc_info=True)
                outfit["mockup_image_url"] = None
                mockup_results.append({
                    "outfit_id":        outfit_id,
                    "style":            outfit.get("style"),
                    "mockup_image_url": None,
                    "error":            str(e),
                })

        # ── Write URLs back to DynamoDB ───────────────────────────────────────
        write_mockup_urls_to_dynamo(user_id, outfit_set_id, outfits)

        successful = [m for m in mockup_results if m["mockup_image_url"]]

        return {
            "status":       "success",
            "userId":       user_id,
            "outfitSetId":  outfit_set_id,
            "totalOutfits": len(outfits),
            "generated":    len(successful),
            "mockups":      mockup_results,
        }

    except Exception as e:
        logger.error(f"Lambda failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}