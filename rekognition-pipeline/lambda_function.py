import json
import boto3
import uuid
import logging
import os
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

rekognition = boto3.client('rekognition', region_name=os.environ['AWS_REGION_NAME'])

# ── Taxonomy ─────────────────────────────────────────────────────────────────

ITEM_TYPE_LABELS = {
    "Shirt", "T-Shirt", "Dress Shirt", "Blouse", "Top",
    "Pants", "Jeans", "Shorts", "Skirt", "Dress",
    "Jacket", "Coat", "Blazer", "Hoodie", "Sweater",
    "Cardigan", "Vest", "Tank Top", "Leggings", "Suit",
    "Overalls", "Jumpsuit", "Romper", "Sweatshirt",
    "Sweatpants", "Trousers", "Tuxedo", "Underwear",
    "Bra", "Swimwear", "Bikini", "Robe", "Nightgown",
    "Socks", "Tights", "Scarf", "Hat", "Cap", "Beanie",
    "Belt", "Gloves", "Shoe", "Sneaker", "Boot", "Sandal",
    "Heel", "Loafer", "Bag", "Handbag", "Backpack", "Purse",
    "Tie", "Bow Tie", "Clothing",  # fallback
}

COLOR_LABELS = {
    "Red", "Blue", "Green", "Yellow", "Orange", "Purple",
    "Pink", "Black", "White", "Gray", "Grey", "Brown",
    "Beige", "Cream", "Navy", "Teal", "Maroon", "Gold",
    "Silver", "Ivory", "Khaki", "Olive", "Turquoise",
    "Coral", "Lavender", "Burgundy", "Cyan", "Magenta",
    "Tan", "Charcoal","Khaki",
}

MATERIAL_LABELS = {
    "Denim", "Cotton", "Leather", "Silk", "Wool", "Linen",
    "Polyester", "Nylon", "Velvet", "Suede", "Fur",
    "Fleece", "Knit", "Lace", "Mesh", "Satin", "Tweed",
    "Corduroy", "Canvas", "Jersey", "Spandex", "Chiffon",
    "Twill", "Flannel","Sleeve", "Pedestrian",
}

FIT_LABELS = {
    "Slim", "Tight", "Loose", "Oversized", "Fitted",
    "Baggy", "Cropped", "Flared", "Skinny", "Relaxed",
}

STYLE_LABELS = {
    "Casual", "Formal", "Business", "Athletic", "Sportswear",
    "Vintage", "Bohemian", "Streetwear", "Elegant", "Classic",
    "Preppy", "Military", "Western", "Minimalist",
}

PATTERN_LABELS = {
    "Striped", "Plaid", "Floral", "Checkered", "Polka Dot",
    "Animal Print", "Camouflage", "Abstract", "Geometric",
    "Paisley", "Solid", "Tie Dye", "Houndstooth", "Graphic",
    "Printed",
}

# Anything in this set → skip entirely (not fashion-relevant)
NOISE_LABELS = {
    "Person", "Human", "Adult", "Man", "Woman", "Boy", "Girl",
    "Face", "Head", "Hand", "Arm", "Leg", "Body", "Skin",
    "Hair", "Eye", "Finger", "Accessories", "Room", "Indoor",
    "Furniture", "Floor", "Wall", "Background", "Photography",
    "Portrait", "Selfie", "Mirror", "Hanger", "Mannequin",
}

# Confidence thresholds
LABEL_CONFIDENCE_THRESHOLD = 70.0       # Minimum to even consider a label
ITEM_TYPE_CONFIDENCE_THRESHOLD = 75.0   # Higher bar for item classification
OVERALL_CONFIDENCE_THRESHOLD = 60.0     # Below this → low_confidence status
MODERATION_CONFIDENCE_THRESHOLD = 60.0  # Flag if moderation hits above this

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_s3_event(event):
    """Extract bucket and key from SQS-wrapped S3 event."""
    # SQS wraps SNS which wraps S3 event — need to unwrap
    for record in event.get('Records', []):
        # SQS record
        body = json.loads(record.get('body', '{}'))
        
        # SNS wrapper (if SNS → SQS)
        if 'Message' in body:
            body = json.loads(body['Message'])
        
        # S3 event records
        for s3_record in body.get('Records', []):
            if s3_record.get('eventSource') == 'aws:s3' or 's3' in s3_record:
                bucket = s3_record['s3']['bucket']['name']
                key = s3_record['s3']['object']['key']
                # URL-decode the key (S3 encodes spaces as +)
                from urllib.parse import unquote_plus
                key = unquote_plus(key)
                return bucket, key
    
    raise ValueError(f"Could not parse S3 event from: {json.dumps(event)[:500]}")


def extract_user_id_from_key(key):
    """
    Assumes S3 key structure: wardrobe/{userId}/{filename}
    Adjust if Eric uses a different path convention.
    """
    parts = key.split('/')
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def call_detect_labels(bucket, key):
    """Call Rekognition DetectLabels. Returns raw response."""
    response = rekognition.detect_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': key}},
        MaxLabels=50,           # Cast wide net, we'll filter
        MinConfidence=55.0,     # Low floor here — we filter ourselves
        Features=['GENERAL_LABELS'],  # Don't need image properties
    )
    return response.get('Labels', [])


def call_detect_moderation(bucket, key):
    """Check for inappropriate content. Returns list of flagged labels."""
    response = rekognition.detect_moderation_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': key}},
        MinConfidence=MODERATION_CONFIDENCE_THRESHOLD,
    )
    return response.get('ModerationLabels', [])


def classify_label(label_name, confidence):
    """
    Map a Rekognition label to its taxonomy category.
    Returns (category, label_name) or (None, None) if noise/irrelevant.
    """
    # Skip noise first
    if label_name in NOISE_LABELS:
        return None, None

    # Split compound labels like "Blue Jeans" → check each token
    tokens = label_name.split()
    for token in tokens:
        if token in COLOR_LABELS and confidence >= LABEL_CONFIDENCE_THRESHOLD:
            return 'colors', token
        if token in MATERIAL_LABELS and confidence >= LABEL_CONFIDENCE_THRESHOLD:
            return 'materials', token

    # Check full label against each taxonomy
    if label_name in ITEM_TYPE_LABELS and confidence >= ITEM_TYPE_CONFIDENCE_THRESHOLD:
        return 'itemTypes', label_name
    if label_name in COLOR_LABELS and confidence >= LABEL_CONFIDENCE_THRESHOLD:
        return 'colors', label_name
    if label_name in MATERIAL_LABELS and confidence >= LABEL_CONFIDENCE_THRESHOLD:
        return 'materials', label_name
    if label_name in FIT_LABELS and confidence >= LABEL_CONFIDENCE_THRESHOLD:
        return 'fit', label_name
    if label_name in STYLE_LABELS and confidence >= LABEL_CONFIDENCE_THRESHOLD:
        return 'style', label_name
    if label_name in PATTERN_LABELS and confidence >= LABEL_CONFIDENCE_THRESHOLD:
        return 'patterns', label_name

    return None, None


def build_cleaned_labels(raw_labels):
    """
    Parse raw Rekognition labels into the taxonomy structure.
    Returns (cleaned_labels_dict, top_confidence_score)
    """
    cleaned = {
        'itemTypes': [],
        'colors': [],
        'materials': [],
        'fit': [],
        'style': [],
        'patterns': [],
    }
    
    top_confidence = 0.0
    raw_output = []

    for label in raw_labels:
        name = label.get('Name', '')
        confidence = label.get('Confidence', 0.0)
        
        # Track raw labels above minimum threshold
        if confidence >= LABEL_CONFIDENCE_THRESHOLD:
            raw_output.append({'name': name, 'confidence': round(confidence, 1)})

        category, clean_name = classify_label(name, confidence)
        if category and clean_name not in cleaned[category]:
            cleaned[category].append(clean_name)
            top_confidence = max(top_confidence, confidence)

        # Also check label aliases/parents (Rekognition nests these)
        for parent in label.get('Parents', []):
            parent_name = parent.get('Name', '')
            p_category, p_clean = classify_label(parent_name, confidence)
            if p_category and p_clean not in cleaned[p_category]:
                cleaned[p_category].append(p_clean)

    return cleaned, round(top_confidence, 1), raw_output


def determine_status(cleaned_labels, top_confidence, moderation_flags):
    """Determine pipeline status code."""
    if moderation_flags:
        return 'error'  # Treat moderation hits as hard errors for a demo
    
    has_item_types = len(cleaned_labels.get('itemTypes', [])) > 0
    
    if not has_item_types and top_confidence < OVERALL_CONFIDENCE_THRESHOLD:
        return 'no_clothing'
    
    if top_confidence < OVERALL_CONFIDENCE_THRESHOLD:
        return 'low_confidence'
    
    if not has_item_types:
        # Has some labels but couldn't identify what kind of clothing
        return 'low_confidence'
    
    return 'success'

# ── Main Handler ──────────────────────────────────────────────────────────────

def handler(event, context):
    logger.info(f"Event received: {json.dumps(event)[:1000]}")
    
    item_id = str(uuid.uuid4())
    bucket, key = None, None

    try:
        bucket, key = parse_s3_event(event)
        user_id = extract_user_id_from_key(key)
        image_url = f"s3://{bucket}/{key}"
        
        logger.info(f"Processing image: {image_url} for user: {user_id}")

        # ── Moderation check first ──
        moderation_flags = call_detect_moderation(bucket, key)
        if moderation_flags:
            logger.warning(f"Moderation flags on {key}: {moderation_flags}")
            return build_error_response(
                item_id, user_id, image_url,
                status='error',
                error_message=f"Image flagged by moderation: {[f['Name'] for f in moderation_flags]}"
            )

        # ── Label detection ──
        raw_labels = call_detect_labels(bucket, key)
        logger.info(f"Rekognition returned {len(raw_labels)} labels")

        cleaned_labels, top_confidence, raw_output = build_cleaned_labels(raw_labels)
        status = determine_status(cleaned_labels, top_confidence, moderation_flags)

        result = {
            'itemId': item_id,
            'userId': user_id,
            'imageUrl': image_url,
            'rawLabels': raw_output,
            'cleanedLabels': cleaned_labels,
            'confidence': top_confidence,
            'status': status,
            'processedAt': datetime.utcnow().isoformat() + 'Z',
        }

        logger.info(f"Pipeline result: status={status}, confidence={top_confidence}, "
                    f"itemTypes={cleaned_labels['itemTypes']}")

        # ── Write to DynamoDB (Eric's table) ──
        write_to_dynamo(result)

        return result

    except ValueError as e:
        logger.error(f"Event parsing error: {e}")
        return build_error_response(item_id, "unknown", "", status='error', error_message=str(e))

    except rekognition.exceptions.InvalidImageException as e:
        logger.error(f"Invalid image: {e}")
        return build_error_response(item_id, "unknown", f"s3://{bucket}/{key}",
                                    status='error', error_message="Invalid or corrupt image file")

    except rekognition.exceptions.ImageTooLargeException:
        logger.error("Image too large for Rekognition (max 5MB for API, 15MB for S3)")
        return build_error_response(item_id, "unknown", f"s3://{bucket}/{key}",
                                    status='error', error_message="Image exceeds Rekognition size limit")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return build_error_response(item_id, "unknown",
                                    f"s3://{bucket}/{key}" if bucket and key else "",
                                    status='error', error_message=str(e))


def build_error_response(item_id, user_id, image_url, status, error_message):
    return {
        'itemId': item_id,
        'userId': user_id,
        'imageUrl': image_url,
        'rawLabels': [],
        'cleanedLabels': {
            'itemTypes': [], 'colors': [], 'materials': [],
            'fit': [], 'style': [], 'patterns': [],
        },
        'confidence': 0.0,
        'status': status,
        'errorMessage': error_message,
        'processedAt': datetime.utcnow().isoformat() + 'Z',
    }


def write_to_dynamo(result):
    """
    Write result to DynamoDB wardrobe items table.
    Coordinate table name and key schema with Eric.
    """
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['WARDROBE_TABLE_NAME'])
        table.put_item(Item={
            'userId': result['userId'],
            'itemId': result['itemId'],
            'imageUrl': result['imageUrl'],
            'cleanedLabels': result['cleanedLabels'],
            'confidence': str(result['confidence']),  # DynamoDB doesn't store float directly
            'status': result['status'],
            'processedAt': result['processedAt'],
        })
        logger.info(f"Wrote item {result['itemId']} to DynamoDB")
    except Exception as e:
        # Don't fail the whole pipeline for a DynamoDB write error
        logger.error(f"DynamoDB write failed: {e}")