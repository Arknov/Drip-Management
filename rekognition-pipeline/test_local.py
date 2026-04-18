import boto3
import io
import os
from PIL import Image

rekognition = boto3.client('rekognition', region_name='us-west-2')
s3 = boto3.client('s3', region_name='us-west-2')

BUCKET = 'arnav-fashion-test'

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

TESTS = [
    ('jeans.jpg',       'Single item'),
    ('mens_casual.jpg', 'Full outfit'),
    ('corolla.jpg',     'Non-clothing'),
]

def download_image(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    return Image.open(io.BytesIO(response['Body'].read())).convert('RGB')

def crop_bounding_box(pil_image, bb):
    img_w, img_h = pil_image.size
    pad_x = bb['Width'] * 0.05
    pad_y = bb['Height'] * 0.05
    x1 = max(0.0, bb['Left'] - pad_x) * img_w
    y1 = max(0.0, bb['Top'] - pad_y) * img_h
    x2 = min(1.0, bb['Left'] + bb['Width'] + pad_x) * img_w
    y2 = min(1.0, bb['Top'] + bb['Height'] + pad_y) * img_h
    return pil_image.crop((int(x1), int(y1), int(x2), int(y2)))

def get_crop_colors(crop):
    buf = io.BytesIO()
    crop.save(buf, format='JPEG', quality=90)
    response = rekognition.detect_labels(
        Image={'Bytes': buf.getvalue()},
        MaxLabels=1,
        MinConfidence=99.0,
        Features=['IMAGE_PROPERTIES'],
    )
    colors = []
    for c in response.get('ImageProperties', {}).get('DominantColors', []):
        if c['PixelPercent'] >= 8.0:
            if c['SimplifiedColor'].lower() in ('white', 'black') and c['PixelPercent'] > 80:
                continue
            colors.append(f"{c['SimplifiedColor']} ({c['PixelPercent']:.1f}%)")
    return colors

for key, label in TESTS:
    print(f"\n{'='*60}")
    print(f"TEST: {label} ({key})")
    print('='*60)

    # Full image labels
    response = rekognition.detect_labels(
        Image={'S3Object': {'Bucket': BUCKET, 'Name': key}},
        MaxLabels=50,
        MinConfidence=55.0,
        Features=['GENERAL_LABELS', 'IMAGE_PROPERTIES'],
    )

    labels = response.get('Labels', [])
    image_props = response.get('ImageProperties', {})

    print("\n--- ALL LABELS ---")
    for l in labels:
        print(f"  {l['Confidence']:.1f}%  {l['Name']}")
        if l.get('Parents'):
            print(f"           parents: {[p['Name'] for p in l['Parents']]}")
        if l.get('Instances'):
            print(f"           instances: {len(l['Instances'])} bounding box(es)")

    print("\n--- FULL IMAGE DOMINANT COLORS ---")
    for c in image_props.get('DominantColors', []):
        print(f"  {c['PixelPercent']:.1f}%  {c['SimplifiedColor']}  ({c.get('CssColor','n/a')})")

    # Per-item bounding box + color
    print("\n--- PER-ITEM BOUNDING BOX + COLOR ---")
    pil_image = download_image(BUCKET, key)

    found_items = False
    for l in labels:
        if l['Name'] not in ITEM_TYPE_LABELS or l['Confidence'] < 75.0:
            continue
        instances = l.get('Instances', [])
        if not instances:
            print(f"  [{l['Name']}] — no bounding box returned by Rekognition")
            continue
        for i, inst in enumerate(instances):
            bb = inst.get('BoundingBox')
            if not bb:
                continue
            found_items = True
            print(f"\n  [{l['Name']}] instance {i+1}")
            print(f"    BoundingBox → left:{bb['Left']:.3f} top:{bb['Top']:.3f} "
                  f"w:{bb['Width']:.3f} h:{bb['Height']:.3f}")
            try:
                crop = crop_bounding_box(pil_image, bb)
                colors = get_crop_colors(crop)
                print(f"    Colors from crop → {colors if colors else 'none detected'}")
                # Save crop for visual inspection
                crop_path = f"crop_{key.split('.')[0]}_{l['Name'].replace(' ','_')}_{i+1}.jpg"
                crop.save(crop_path)
                print(f"    Saved crop → {crop_path}")
            except Exception as e:
                print(f"    Crop failed: {e}")

    if not found_items:
        print("  No clothing items with bounding boxes found.")