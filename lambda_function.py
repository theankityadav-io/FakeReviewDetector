import json
import re

def lambda_handler(event, context):
    """Fake Review Detector with CORS"""
    
    print("Event:", json.dumps(event, default=str))
    
    # Get HTTP method - check multiple locations
    http_method = None
    
    # API Gateway format
    if 'httpMethod' in event:
        http_method = event['httpMethod']
    # Lambda Function URL format
    elif 'requestContext' in event:
        if 'http' in event['requestContext']:
            http_method = event['requestContext']['http'].get('method')
        elif 'httpMethod' in event['requestContext']:
            http_method = event['requestContext']['httpMethod']
    
    print(f"HTTP Method: {http_method}")
    
    # Handle OPTIONS preflight - CRITICAL!
    if http_method == 'OPTIONS':
        print("Returning OPTIONS response with CORS headers")
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, X-Requested-With'
            },
            'body': ''
        }
    
    # Handle POST request
    try:
        # Parse body
        if 'body' in event and event['body']:
            body = json.loads(event['body'])
        else:
            body = event
        
        review_text = body.get('review_text', '')
        
        # Validate
        if not review_text:
            return create_response(400, {'error': 'review_text is required'})
        
        # Analyze
        result = analyze_review(review_text)
        return create_response(200, result)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return create_response(500, {'error': str(e)})

def analyze_review(text):
    """Pattern-based fake review detection"""
    text_lower = text.lower()
    
    patterns = [
        (r'!{3,}', 30, 'Excessive punctuation'),
        (r'(amazing|perfect|outstanding)\s*\1', 35, 'Repeated superlatives'),
        (r'best\s+(ever|buy|product)', 25, 'Generic praise'),
        (r'buy\s+(now|today)', 30, 'Urgency language'),
        (r'5\s*stars?\s*!+', 25, 'Exaggerated rating'),
        (r'highly\s+recommend', 20, 'Generic recommendation'),
        (r'everyone\s+should', 25, 'Universal claims'),
        (r'(great|fast)\s+(price|deal)', 20, 'Promotional phrases'),
        (r'no\s+complaints?', 25, 'Suspiciously perfect'),
        (r'(top|best)\s+quality', 20, 'Marketing buzzwords'),
    ]
    
    score = 0
    matches = []
    
    for pattern, weight, desc in patterns:
        if re.search(pattern, text_lower):
            score += weight
            matches.append(desc)
    
    confidence = min(score, 100)
    
    return {
        'prediction': 'FAKE' if confidence >= 50 else 'GENUINE',
        'confidence': round(confidence, 2),
        'review_text': text,
        'indicators_found': len(matches),
        'matched_patterns': matches[:5]
    }

def create_response(status_code, body):
    """Create response with CORS headers - ALWAYS!"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, X-Requested-With'
        },
        'body': json.dumps(body)
    }