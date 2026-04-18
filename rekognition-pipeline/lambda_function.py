import json
import boto3
import uuid
import logging
import os
from datetime import datetime
from PIL import Image
import io

logger = logging.getLogger()
logger.setLevel(logging.INFO)

rekognition = boto3.client('rekognition', region_name=os.environ['AWS_REGION_NAME'])
s3 = boto3.client('s3', region_name=os.environ['AWS_REGION_NAME'])

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
    "Tan", "Charcoal",
}

MATERIAL_LABELS = {
    "Denim", "Cotton", "Leather", "Silk", "Wool", "Linen",
    "Polyester", "Nylon", "Velvet", "Suede", "Fur",
    "Fleece", "Knit", "Lace", "Mesh", "Satin", "Tweed",
    "Corduroy", "Canvas", "Jersey", "Spandex", "Chiffon",
    "Twill", "Flannel",
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
# ── Image helpers ─────────────────────────────────────────────────────────────

def download_image(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    image_bytes = response['Body'].read()
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')

def crop_bounding_box(pil_image, bounding_box):
    """Internal helper to crop for color detection logic."""
    img_w, img_h = pil_image.size
    left, top, width, height = bounding_box['Left'], bounding_box['Top'], bounding_box['Width'], bounding_box['Height']
    
    padding_x, padding_y = width * 0.05, height * 0.05

    x1 = max(0.0, left - padding_x) * img_w
    y1 = max(0.0, top - padding_y) * img_h
    x2 = min(1.0, left + width + padding_x) * img_w
    y2 = min(1.0, top + height + padding_y) * img_h

    return pil_image.crop((int(x1), int(y1), int(x2), int(y2)))

def pil_image_to_bytes(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    return buffer.getvalue()

# ── Rekognition calls ─────────────────────────────────────────────────────────

def call_detect_labels(bucket, key):
    response = rekognition.detect_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': key}},
        MaxLabels=50,
        MinConfidence=55.0,
        Features=['GENERAL_LABELS', 'IMAGE_PROPERTIES'],
    )
    return response.get('Labels', []), response.get('ImageProperties', {})

def call_detect_moderation(bucket, key):
    response = rekognition.detect_moderation_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': key}},
        MinConfidence=60.0,
    )
    return response.get('ModerationLabels', [])

def get_colors_for_crop(crop_bytes):
    try:
        response = rekognition.detect_labels(
            Image={'Bytes': crop_bytes},
            MaxLabels=1,
            MinConfidence=99.0,
            Features=['IMAGE_PROPERTIES'],
        )
        dominant_colors = response.get('ImageProperties', {}).get('DominantColors', [])
        colors = []
        for color in dominant_colors:
            simplified = color.get('SimplifiedColor', '')
            if color.get('PixelPercent', 0) < 8.0: continue
            if simplified.lower() in ('white', 'black') and color.get('PixelPercent', 0) > 80: continue
            if simplified and simplified not in colors: colors.append(simplified)
        return colors
    except Exception as e:
        logger.warning(f"Crop color detection failed: {e}")
        return []

# ── Per-item detection (Updated to exclude BB from output) ────────────────────

def detect_clothing_items(raw_labels, pil_image):
    clothing_items = []
    for label in raw_labels:
        name = label.get('Name', '')
        confidence = label.get('Confidence', 0.0)

        if name not in ITEM_TYPE_LABELS or confidence < 75.0:
            continue

        instances = label.get('Instances', [])

        if not instances:
            clothing_items.append({
                'itemType': name,
                'confidence': round(confidence, 1),
                'colors': [], 'materials': [], 'patterns': [], 'fit': [], 'style': [],
            })
            continue

        for i, instance in enumerate(instances):
            bounding_box = instance.get('BoundingBox')
            item_colors = []
            
            if bounding_box and pil_image:
                try:
                    crop = crop_bounding_box(pil_image, bounding_box)
                    item_colors = get_colors_for_crop(pil_image_to_bytes(crop))
                except Exception as e:
                    logger.warning(f"Crop failed for {name}: {e}")

            clothing_items.append({
                'itemType': name,
                'confidence': round(instance.get('Confidence', confidence), 1),
                'colors': item_colors,
                'materials': [], 'patterns': [], 'fit': [], 'style': [],
                '_internal_box': bounding_box # Kept temporarily for attribute matching
            })
    return clothing_items

def associate_attributes(clothing_items, raw_labels):
    """Uses internal boxes to associate, then strips them before returning."""
    for label in raw_labels:
        name, conf = label.get('Name', ''), label.get('Confidence', 0.0)
        if conf < 70.0 or name in NOISE_LABELS or name in ITEM_TYPE_LABELS: continue

        category, clean_name = None, name
        # ... [Category matching logic from your original code] ...
        if name in MATERIAL_LABELS: category = 'materials'
        elif name in PATTERN_LABELS: category = 'patterns'
        elif name in FIT_LABELS: category = 'fit'
        elif name in STYLE_LABELS: category = 'style'

        if category:
            instances = label.get('Instances', [])
            if not instances or not any(i.get('_internal_box') for i in clothing_items):
                for item in clothing_items:
                    if clean_name not in item[category]: item[category].append(clean_name)
            else:
                for inst in instances:
                    best_item = find_best_matching_item(clothing_items, inst.get('BoundingBox', {}))
                    if best_item and clean_name not in best_item[category]:
                        best_item[category].append(clean_name)

    # Final Step: Clean up internal boxes before returning to Ryan/Dynamo
    for item in clothing_items:
        item.pop('_internal_box', None)
        
    return clothing_items

def find_best_matching_item(clothing_items, attr_box):
    best_item, best_overlap = None, 0.0
    for item in clothing_items:
        box_a = item.get('_internal_box')
        if not box_a: continue
        
        # Simple IoU logic
        a_x1, a_y1 = box_a['Left'], box_a['Top']
        a_x2, a_y2 = a_x1 + box_a['Width'], a_y1 + box_a['Height']
        b_x1, b_y1 = attr_box.get('Left', 0), attr_box.get('Top', 0)
        b_x2, b_y2 = b_x1 + attr_box.get('Width', 0), b_y1 + attr_box.get('Height', 0)

        inter_area = max(0, min(a_x2, b_x2) - max(a_x1, b_x1)) * max(0, min(a_y2, b_y2) - max(a_y1, b_y1))
        union_area = (box_a['Width'] * box_a['Height']) + (attr_box.get('Width', 0) * attr_box.get('Height', 0)) - inter_area
        overlap = inter_area / union_area if union_area > 0 else 0.0

        if overlap > best_overlap:
            best_overlap, best_item = overlap, item
            
    return best_item or next((i for i in clothing_items if i.get('_internal_box')), None)

# ── Summary, Status, Handler, and Dynamo Logic ───────────────────────────────

# ... [The rest of your functions: build_summary_labels, determine_status, 
#      handler, parse_s3_event, write_to_dynamo remain the same] ...
# ── Summary builder (Ryan's contract) ────────────────────────────────────────

def build_summary_labels(clothing_items):
    """
    Flatten per-item attributes into the original cleanedLabels format.
    Ryan gets both detectedItems (detailed) and cleanedLabels (summary).
    """
    summary = {
        'itemTypes': [], 'colors': [], 'materials': [],
        'fit': [], 'style': [], 'patterns': [],
    }
    for item in clothing_items:
        for field in summary:
            key = 'itemType' if field == 'itemTypes' else field
            values = [item[key]] if field == 'itemTypes' else item.get(key, [])
            for v in values:
                if v not in summary[field]:
                    summary[field].append(v)
    return summary


def determine_status(clothing_items, moderation_flags):
    if moderation_flags:
        return 'error'
    if not clothing_items:
        return 'no_clothing'
    confidences = [item['confidence'] for item in clothing_items]
    if max(confidences) < 60.0:  # Using OVERALL_CONFIDENCE_THRESHOLD
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

        logger.info(f"Processing: {image_url} for user: {user_id}")

        # 1. Moderation check
        moderation_flags = call_detect_moderation(bucket, key)
        if moderation_flags:
            logger.warning(f"Moderation flags: {moderation_flags}")
            return build_error_response(
                item_id, user_id, image_url, status='error',
                error_message=f"Image flagged: {[f['Name'] for f in moderation_flags]}"
            )

        # 2. Preparation for Detection
        pil_image = download_image(bucket, key)
        raw_labels, _ = call_detect_labels(bucket, key)

        # 3. Detection & Association (Internal boxes used here)
        clothing_items = detect_clothing_items(raw_labels, pil_image)
        clothing_items = associate_attributes(clothing_items, raw_labels)

        # 4. Final Aggregation (Boxes are already stripped by associate_attributes)
        summary_labels = build_summary_labels(clothing_items)
        status = determine_status(clothing_items, moderation_flags)

        raw_output = [
            {'name': l['Name'], 'confidence': round(l['Confidence'], 1)}
            for l in raw_labels if l['Confidence'] >= 70.0
        ]

        result = {
            'itemId': item_id,
            'userId': user_id,
            'imageUrl': image_url,
            'rawLabels': raw_output,
            'cleanedLabels': summary_labels,
            'detectedItems': clothing_items, # Cleaned list (No Bounding Boxes)
            'confidence': round(max((i['confidence'] for i in clothing_items), default=0.0), 1),
            'status': status,
            'processedAt': datetime.utcnow().isoformat() + 'Z',
        }

        write_to_dynamo(result)
        return result

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return build_error_response(
            item_id, "unknown", 
            f"s3://{bucket}/{key}" if bucket and key else "", 
            status='error', 
            error_message=str(e)
        )


# ── Utilities ─────────────────────────────────────────────────────────────────

def parse_s3_event(event):
    for record in event.get('Records', []):
        body = json.loads(record.get('body', '{}'))
        if 'Message' in body:
            body = json.loads(body['Message'])
        for s3_record in body.get('Records', []):
            if s3_record.get('eventSource') == 'aws:s3' or 's3' in s3_record:
                bucket = s3_record['s3']['bucket']['name']
                from urllib.parse import unquote_plus
                key = unquote_plus(s3_record['s3']['object']['key'])
                return bucket, key
    raise ValueError("Could not parse S3 event")


def extract_user_id_from_key(key):
    parts = key.split('/')
    return parts[1] if len(parts) >= 2 else "unknown"


def build_error_response(item_id, user_id, image_url, status, error_message):
    return {
        'itemId': item_id,
        'userId': user_id,
        'imageUrl': image_url,
        'rawLabels': [],
        'cleanedLabels': {'itemTypes': [], 'colors': [], 'materials': [], 'fit': [], 'style': [], 'patterns': []},
        'detectedItems': [],
        'confidence': 0.0,
        'status': status,
        'errorMessage': error_message,
        'processedAt': datetime.utcnow().isoformat() + 'Z',
    }


def write_to_dynamo(result):
    try:
        dynamodb = boto3.resource('dynamodb', region_name=os.environ['AWS_REGION_NAME'])
        table = dynamodb.Table(os.environ['WARDROBE_TABLE_NAME'])
        
        # Note: We store the same result object. 
        # Since boundingBox was removed earlier, it won't be in Dynamo.
        table.put_item(Item={
            'userId': result['userId'],
            'itemId': result['itemId'],
            'imageUrl': result['imageUrl'],
            'cleanedLabels': result['cleanedLabels'],
            'detectedItems': result['detectedItems'],
            'confidence': str(result['confidence']),
            'status': result['status'],
            'processedAt': result['processedAt'],
        })
        logger.info(f"Wrote item {result['itemId']} to DynamoDB")
    except Exception as e:
        logger.error(f"DynamoDB write failed: {e}")