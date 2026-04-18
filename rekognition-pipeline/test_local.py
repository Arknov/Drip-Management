import boto3
import json

rekognition = boto3.client('rekognition', region_name='us-west-2')
BUCKET = 'arnav-fashion-test'

TESTS = [
    ('jeans.jpg', 'Single item'),
    ('mens_casual.jpg', 'Full outfit'),
    ('corolla.jpg', 'Non-clothing'),
]

for key, label in TESTS:
    print(f"\n{'='*50}")
    print(f"TEST: {label} ({key})")
    print('='*50)
    
    response = rekognition.detect_labels(
        Image={'S3Object': {'Bucket': BUCKET, 'Name': key}},
        MaxLabels=50,
        MinConfidence=55.0,
    )
    
    for l in response['Labels']:
        print(f"{l['Confidence']:.1f}%  {l['Name']}")
        if l.get('Parents'):
            print(f"         parents: {[p['Name'] for p in l['Parents']]}")