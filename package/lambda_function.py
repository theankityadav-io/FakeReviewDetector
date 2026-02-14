import json
import joblib

model = None
vectorizer = None

def load_model():
    global model, vectorizer
    if model is None:
        model = joblib.load('fake_review_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

def lambda_handler(event, context):
    # Handle OPTIONS request for CORS
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': ''
        }
    
    try:
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        review_text = body.get('review_text', '')
        
        if not review_text:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': json.dumps({'error': 'Please provide review_text'})
            }
        
        model, vectorizer = load_model()
        review_tfidf = vectorizer.transform([review_text])
        prediction = model.predict(review_tfidf)[0]
        confidence = model.predict_proba(review_tfidf)[0]
        
        result = {
            'prediction': 'FAKE' if prediction == 1 else 'GENUINE',
            'confidence': float(max(confidence) * 100),
            'review_text': review_text
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({'error': str(e)})
        }
