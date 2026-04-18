import json
import boto3
import uuid
import os
import logging
from decimal import Decimal
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── ENV ───────────────────────────────────────────────────────────────────────
AWS_REGION     = os.environ.get("AWS_REGION_NAME", "us-west-2")
WARDROBE_TABLE = os.environ["WARDROBE_TABLE_NAME"]

# ── CLIENTS ───────────────────────────────────────────────────────────────────
rekognition = boto3.client("rekognition", region_name=AWS_REGION)
s3          = boto3.client("s3",          region_name=AWS_REGION)
dynamodb    = boto3.resource("dynamodb",  region_name=AWS_REGION)
table       = dynamodb.Table(WARDROBE_TABLE)

# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────

NORMALIZATION_MAP = {
    "Jeans":    "Pants",
    "Trousers": "Pants",
    "Sneaker":  "Shoes",
    "Boot":     "Shoes",
    "Heel":     "Shoes",
    "Loafer":   "Shoes",
}

BOTTOMS     = {"Pants", "Shorts", "Skirt", "Leggings"}
SHOES       = {"Shoes"}
ACCESSORIES = {"Bag", "Handbag", "Backpack", "Purse", "Belt", "Hat", "Cap", "Scarf", "Tie"}
NOISE       = {
    "Person", "Human", "Body", "Room", "Background", "Furniture", "Clothing",
    "Sleeve", "Pedestrian", "Adult", "Man", "Woman", "Male", "Female",
}

COLOR_NAMES  = {
    "Black", "White", "Blue", "Grey", "Gray", "Brown", "Navy",
    "Red", "Green", "Yellow", "Orange", "Pink", "Purple", "Beige",
    "Cream", "Khaki", "Olive", "Teal", "Maroon", "Coral",
}
MATERIALS    = {
    "Denim", "Cotton", "Leather", "Silk", "Wool", "Linen",
    "Polyester", "Nylon", "Velvet", "Suede", "Fleece", "Knit",
}
FIT_LABELS   = {"Slim", "Loose", "Oversized", "Fitted", "Baggy", "Skinny", "Relaxed"}
STYLE_LABELS = {"Casual", "Formal", "Streetwear", "Business", "Elegant", "Athletic", "Vintage", "Preppy"}
PATTERNS     = {"Striped", "Plaid", "Floral", "Solid", "Printed", "Checkered", "Camouflage", "Graphic"}

# ─────────────────────────────────────────────────────────────────────────────
# REKOGNITION CALLS
# ─────────────────────────────────────────────────────────────────────────────

def detect(bucket, key):
    return rekognition.detect_labels(
        Image={"S3Object": {"Bucket": bucket, "Name": key}},
        MaxLabels=50,
        MinConfidence=50,
        Features=["GENERAL_LABELS", "IMAGE_PROPERTIES"],
    )

def detect_moderation(bucket, key):
    return rekognition.detect_moderation_labels(
        Image={"S3Object": {"Bucket": bucket, "Name": key}},
        MinConfidence=60,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TOP CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_top(labels):
    names = {l["Name"].lower() for l in labels}

    if "t-shirt" in names or "tee" in names:
        return "T-Shirt"
    if "hoodie" in names:
        return "Hoodie"
    if "sweater" in names or "knit" in names:
        return "Sweater"
    if "blouse" in names:
        return "Blouse"
    if "jacket" in names or "blazer" in names:
        return "Jacket"
    if "shirt" in names or "collar" in names:
        return "Shirt"
    return None

# ─────────────────────────────────────────────────────────────────────────────
# COLOR EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_colors(image_props):
    """Full-image dominant colors — used as fallback on the summary."""
    colors = []
    for c in image_props.get("DominantColors", []):
        if c["PixelPercent"] > 5:
            simplified = c["SimplifiedColor"].title()  # normalize to Title Case
            if simplified not in colors:
                colors.append(simplified)
    return colors

# ─────────────────────────────────────────────────────────────────────────────
# ITEM EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_items(labels):
    items = []

    for l in labels:
        name = l["Name"]
        conf = l["Confidence"]

        if name in NOISE or conf < 55:
            continue

        name = NORMALIZATION_MAP.get(name, name)

        if name in BOTTOMS or name in SHOES or name in ACCESSORIES:
            items.append({
                "itemType":  name,
                "confidence": round(conf, 1),
                "colors":    [],
                "materials": [],
                "fit":       [],
                "style":     [],
                "patterns":  [],
            })

    # Tops — inferred since Rekognition is weak on top classification
    top = classify_top(labels)
    if top:
        items.append({
            "itemType":  top,
            "confidence": 70.0,
            "colors":    [],
            "materials": [],
            "fit":       [],
            "style":     [],
            "patterns":  [],
        })

    return merge_duplicates(items)

# ─────────────────────────────────────────────────────────────────────────────
# MERGE DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────

def merge_duplicates(items):
    best = {}
    for item in items:
        key = item["itemType"]
        if key not in best or item["confidence"] > best[key]["confidence"]:
            best[key] = item
    return list(best.values())

# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def enrich(items, labels):

    for l in labels:
        name = l["Name"]

        for i in items:

            if name in {"Black","White","Blue","Grey","Gray","Brown","Navy","Red","Green"}:
                if name not in i["colors"]:
                    i["colors"].append(name)

            if name in {"Denim","Cotton","Leather","Silk","Wool","Linen"}:
                i["materials"].append(name)

            if name in {"Slim","Loose","Oversized","Fitted","Baggy"}:
                i["fit"].append(name)

            if name in {"Casual","Formal","Streetwear","Business","Elegant"}:
                i["style"].append(name)

            if name in {"Striped","Plaid","Floral","Solid","Printed"}:
                i["patterns"].append(name)

    return items

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(items):
    return {
        "itemTypes": list({i["itemType"] for i in items}),
        "colors":    list({c for i in items for c in i["colors"]}),
        "materials": list({m for i in items for m in i["materials"]}),
        "fit":       list({f for i in items for f in i["fit"]}),
        "style":     list({s for i in items for s in i["style"]}),
        "patterns":  list({p for i in items for p in i["patterns"]}),
    }

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMO WRITE
# ─────────────────────────────────────────────────────────────────────────────

def floats_to_decimal(obj):
    """Recursively convert all floats to Decimal for DynamoDB compatibility."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, list):
        return [floats_to_decimal(i) for i in obj]
    if isinstance(obj, dict):
        return {k: floats_to_decimal(v) for k, v in obj.items()}
    return obj


def write_to_dynamo(user_id, result):
    try:
        table.put_item(Item={
            "userId":        user_id,
            "itemId":        result["itemId"],
            "imageUrl":      result["imageUrl"],
            "cleanedLabels": floats_to_decimal(result["cleanedLabels"]),
            "detectedItems": floats_to_decimal(result["detectedItems"]),
            "confidence":    str(result["confidence"]),
            "status":        result["status"],
            "processedAt":   result["processedAt"],
        })
        logger.info(f"Wrote item {result['itemId']} for user {user_id} to DynamoDB")
    except Exception as e:
        logger.error(f"DynamoDB write failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# EVENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_s3_event(event):
    from urllib.parse import unquote_plus

    for record in event.get("Records", []):

        # Format 1: Direct S3 event (Lambda console test)
        if record.get("eventSource") == "aws:s3" or "s3" in record:
            bucket = record["s3"]["bucket"]["name"]
            key    = unquote_plus(record["s3"]["object"]["key"])
            return bucket, key

        # Format 2: SQS → (SNS →) S3
        if "body" in record:
            try:
                body = json.loads(record["body"])
            except (json.JSONDecodeError, TypeError):
                continue

            if "Message" in body:
                try:
                    body = json.loads(body["Message"])
                except (json.JSONDecodeError, TypeError):
                    continue

            for s3_record in body.get("Records", []):
                if s3_record.get("eventSource") == "aws:s3" or "s3" in s3_record:
                    bucket = s3_record["s3"]["bucket"]["name"]
                    key    = unquote_plus(s3_record["s3"]["object"]["key"])
                    return bucket, key

    raise ValueError(f"Could not parse S3 event: {json.dumps(event)[:500]}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    item_id = str(uuid.uuid4())

    try:
        bucket, key = parse_s3_event(event)

        # Extract userId from key: wardrobe/{userId}/{filename}
        key_parts = key.split("/")
        user_id   = key_parts[1] if len(key_parts) >= 2 else "unknown"
        image_url = f"s3://{bucket}/{key}"

        logger.info(f"Processing {image_url} for user {user_id}")

        # Moderation check
        mod = detect_moderation(bucket, key)
        if mod.get("ModerationLabels"):
            logger.warning(f"Moderation flag on {key}")
            return {"status": "error", "reason": "unsafe image"}

        # Rekognition
        response    = detect(bucket, key)
        labels      = response.get("Labels", [])
        image_props = response.get("ImageProperties", {})

        # pipeline
        items = extract_items(labels)
        items = enrich(items, labels)

        # Pipeline
        items   = extract_items(labels)
        items   = enrich(items, labels)
        summary = build_summary(items)
        colors = extract_colors(image_props)

        result = {
            "itemId":        item_id,
            "imageUrl":      image_url,
            "detectedItems": items,
            "cleanedLabels": summary,
            "confidence":    round(confidence, 1),
            "status":        status,
            "processedAt":   datetime.utcnow().isoformat() + "Z",
        }

        logger.info(f"Result: status={status}, items={len(items)}, "
                    f"types={summary['itemTypes']}, colors={summary['colors']}")

        write_to_dynamo(user_id, result)
        return result

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}