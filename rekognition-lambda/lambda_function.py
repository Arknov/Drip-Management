import json
import boto3
import uuid
import os
from datetime import datetime

rekognition = boto3.client("rekognition")
s3 = boto3.client("s3")

# ─────────────────────────────
# NORMALIZATION
# ─────────────────────────────

NORMALIZATION_MAP = {
    "Jeans": "Pants",
    "Trousers": "Pants",
    "Sneaker": "Shoes",
    "Boot": "Shoes",
    "Heel": "Shoes",
    "Loafer": "Shoes"
}

TOPS = {
    "Shirt","T-Shirt","Blouse","Top","Tank Top",
    "Sweater","Hoodie","Sweatshirt","Cardigan","Jacket","Blazer"
}

BOTTOMS = {"Pants","Shorts","Skirt","Leggings"}
SHOES = {"Shoes"}

ACCESSORIES = {"Bag","Handbag","Backpack","Purse","Belt","Hat","Cap","Scarf","Tie"}

NOISE = {"Person","Human","Body","Room","Background","Furniture","Clothing"}

# ─────────────────────────────
# REKOGNITION
# ─────────────────────────────

def detect(bucket, key):
    return rekognition.detect_labels(
        Image={"S3Object": {"Bucket": bucket, "Name": key}},
        MaxLabels=50,
        MinConfidence=50,
        Features=["GENERAL_LABELS", "IMAGE_PROPERTIES"]
    )

def detect_moderation(bucket, key):
    return rekognition.detect_moderation_labels(
        Image={"S3Object": {"Bucket": bucket, "Name": key}},
        MinConfidence=60
    )

# ─────────────────────────────
# TOP CLASSIFICATION (CORE FIX)
# ─────────────────────────────

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

    if "shirt" in names:
        return "Shirt"

    if "collar" in names:
        return "Shirt"

    return None

# ─────────────────────────────
# COLORS
# ─────────────────────────────

def extract_colors(image_props):
    colors = []
    for c in image_props.get("DominantColors", []):
        if c["PixelPercent"] > 5:
            colors.append(c["SimplifiedColor"])
    return list(set(colors))

# ─────────────────────────────
# ITEM EXTRACTION
# ─────────────────────────────

def extract_items(labels):

    items = []

    # ── BOTTOMS + SHOES + ACCESSORIES ──
    for l in labels:
        name = l["Name"]
        conf = l["Confidence"]

        if name in NOISE:
            continue

        name = NORMALIZATION_MAP.get(name, name)

        if conf < 55:
            continue

        if name in BOTTOMS or name in SHOES or name in ACCESSORIES:
            items.append({
                "itemType": name,
                "confidence": round(conf, 1),
                "colors": [],
                "materials": [],
                "fit": [],
                "style": [],
                "patterns": []
            })

    # ── TOP INFERENCE (IMPORTANT FIX) ──
    top = classify_top(labels)

    if top:
        items.append({
            "itemType": top,
            "confidence": 70.0,
            "colors": [],
            "materials": [],
            "fit": [],
            "style": [],
            "patterns": []
        })

    return merge(items)

# ─────────────────────────────
# MERGE DUPLICATES
# ─────────────────────────────

def merge(items):
    best = {}

    for i in items:
        key = i["itemType"]

        if key not in best or i["confidence"] > best[key]["confidence"]:
            best[key] = i

    return list(best.values())

# ─────────────────────────────
# ATTRIBUTE ENRICHMENT
# ─────────────────────────────

def enrich(items, labels, dominant_colors=None):
    COLOR_KEYWORDS = {"Black","White","Blue","Grey","Gray","Brown","Navy","Red","Green"}
    detected_colors = [l["Name"] for l in labels if l["Name"] in COLOR_KEYWORDS]

    for i in items:
        # Only assign first detected color, not all of them
        if detected_colors and not i["colors"]:
            i["colors"].append(detected_colors[0])

        for l in labels:
            name = l["Name"]
            if name in {"Denim","Cotton","Leather","Silk","Wool","Linen"}:
                i["materials"].append(name)
            if name in {"Slim","Loose","Oversized","Fitted","Baggy"}:
                i["fit"].append(name)
            if name in {"Casual","Formal","Streetwear","Business","Elegant"}:
                i["style"].append(name)
            if name in {"Striped","Plaid","Floral","Solid","Printed"}:
                i["patterns"].append(name)

    return items

# ─────────────────────────────
# SUMMARY OUTPUT
# ─────────────────────────────

def build_summary(items):

    return {
        "itemTypes": list({i["itemType"] for i in items}),
        "colors": list({c for i in items for c in i["colors"]}),
        "materials": list({m for i in items for m in i["materials"]}),
        "fit": list({f for i in items for f in i["fit"]}),
        "style": list({s for i in items for s in i["style"]}),
        "patterns": list({p for i in items for p in i["patterns"]}),
    }

# ─────────────────────────────
# MAIN HANDLER
# ─────────────────────────────

def lambda_handler(event, context):

    item_id = str(uuid.uuid4())

    try:
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        # moderation
        mod = detect_moderation(bucket, key)
        if mod.get("ModerationLabels"):
            return {"status": "error", "reason": "unsafe image"}

        # rekognition
        response = detect(bucket, key)

        labels = response.get("Labels", [])
        image_props = response.get("ImageProperties", {})

        # pipeline
        items = extract_items(labels)
        colors = extract_colors(image_props)
        items = enrich(items, labels, colors)

        summary = build_summary(items)
        

        result = {
            "itemId": item_id,
            "image": f"s3://{bucket}/{key}",
            "detectedItems": items,
            "cleanedLabels": summary,
            "colors": colors,
            "confidence": max([i["confidence"] for i in items], default=0),
            "status": "success",
            "processedAt": datetime.utcnow().isoformat()
        }

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}